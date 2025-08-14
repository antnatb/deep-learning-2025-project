import torch 
import torch.nn as nn
import torch.nn.functional as F
import clip

# MoCoOp: Mixture of Prompt Learning

# utils
def L2norm(x, dim=-1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)

# 8 groups of hard prompts for Flowers102
HARD_GROUPS = [
    ["a photo of a type of flower: {}", "a photo of a type of flower: the {}."],    # flowers
    ["a photo of a {}.", "a photo of the {}."],                                     # generic
    ["a close-up photo of a {}.", "a macro photo of a {}."],                        # proximity
    ["a cropped photo of a {}.", "a cropped photo of the {}."],                     # crops
    ["a bright photo of a {}.", "a bright photo of the {}."],                       # brightness
    ["a good photo of a {}.", "a good quality photo of a {}."],                     # good quality
    ["a low resolution photo of a {}.", "a pixelated photo of a {}."],              # low resolution
    ["itap of a {}.", "itap of the {}."]                                            # I took a picture of ...
]

# function to encode text using CLIP's tokenizer and text encoder
@torch.no_grad()
def _encode_texts(clip_model, texts):
    device = next(clip_model.parameters()).device # ensure device consistency
    tokenized = clip.tokenize(texts).to(device)  # [batch_size, 77] - 77 is CLIP's max context length
    features = clip_model.encode_text(tokenized).float()  # [batch_size, 512] - 512 is CLIP's text embedding dim
    return L2norm(features)

@torch.no_grad()
def build_hard_features(clip_model, class_names, hard_groups):
    """
    Assuming G groups, C classes and embedding dim D.
    Takes:
    Set of G groups of hard prompts.
    Returns:
    hard_group_avg: [G, D], averaged over group and classes. Will be used for router supervision.
    hard_group_class: [G, C, D] only averaged over group. Will be used for text-level supervision.
    """
    group_avg = []
    group_class = []
    for group in hard_groups:
        per_class = []
        for class_name in class_names:
            texts = [prompt.format(class_name) for prompt in group] # insert class name at placeholder {} in each prompt
            features = _encode_texts(clip_model, texts).mean(0,keepdim=True)  # [1, D] - average across the 2 prompt embeddings in the group
            per_class.append(features)
        per_class = torch.cat(per_class, dim=0) # [C, D]
        per_class = L2norm(per_class)  
        group_class.append(per_class)
        group_avg.append(per_class.mean(0, keepdim=True))
    hard_group_avg = L2norm(torch.cat(group_avg, dim=0)) # [G, D]
    hard_group_class = L2norm(torch.stack(group_class, dim=0)) # [G, C, D]
    return hard_group_avg, hard_group_class 

# router: takes image features and outputs expert logits
class Router(nn.Module):
    def __init__(self, input_dim, G, hidden_dim=0, bias=True):
        super().__init__()
        if hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=bias),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, G, bias=bias)
            )
        else:
            self.net = nn.Linear(input_dim, G, bias=bias)
    
    def forward(self, x):       
        return self.net(x)      # logits: [B, G] where B is batch size

# text encoder: we cannot use CLIP's standard text encoder because we learn already embedded prompts, same as in CoOp.
class TextEncoder(nn.Module):
    def __init__(self, clip_model, token_prefix, token_suffix, tokenized_templates):
        super().__init__()
        self.clip_model = clip_model
        self.token_prefix = token_prefix              # [C, 1, D] where C is the number of classes and D is the text embedding dim (512 for CLIP)
        self.token_suffix = token_suffix              # [C, S, D] where S is the number of tokens in the suffices
        self.tokenized_templates = tokenized_templates

    def forward(self, ctx_g):
        C = self.token_prefix.size(0)   # num of classes
        ctx = ctx_g.unsqueeze(0).expand(C, -1, -1)  # [C, n_ctx, D]
        x = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)   # [C, seq_len, D]
        x = x + self.clip_model.positional_embedding[:x.size(1)]    # add positional embedding
        x = x.permute(1,0,2)  # [seq_len, C, D]
        x = self.transformer(x)
        x = x.permute(1,0,2)  # [C, seq_len, D]
        x = self.clip_model.ln_final(x) # layer normalization
        eot_idx = self.tokenized_templates.argmax(dim=-1)          # take index of EOT (highest in CLIP)
        x = x[torch.arange(C, device=x.device), eot_idx]           # extract hidden state at EOT token for each class. 
        x = x @ self.clip_model.text_projection                    # text projection    
        return L2norm(x)

    

# prompt experts
class PromptExperts(nn.Module):
    def __init__(self, clip_model, class_names, n_ctx=16, hard_groups=None,
                top_k=2, tau=0.07, lambda_router=1.0, lambda_text=5.0,
                router_hidden=0):
        super().__init__()
        assert hard_groups is not None and len(hard_groups) > 0
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.n_ctx = n_ctx
        self.clip_model = clip_model
        # freeze CLIP
        for p in self.clip_model.parameters():
            p.requires_grad = False
        self.device = next(clip_model.parameters()).device
        self.dtype = next(clip_model.parameters()).dtype
        self.G = len(hard_groups) # number of experts
        self.top_k = top_k # number of active experts
        self.tau = tau
        self.lambda_router = lambda_router
        self.lambda_text = lambda_text

        # soft contexts
        ctx_dim = clip_model.ln_final.weight.shape[0]  # CLIP's text embedding dimension
        self.ctxs = nn.Parameter(torch.empty(self.G, n_ctx, ctx_dim, dtype=self.dtype)) # [G, n_ctx, ctx_dim] all zeros for now
        
        # templates with placeholders
        prompt_prefix = " ".join(["X"] * n_ctx)
        template_prompts = [f"{prompt_prefix} {name}." for name in class_names]
        tokenized_template_prompts = torch.cat([clip.tokenize(p) for p in template_prompts]).to(self.device) # [num_classes, 77]
        with torch.no_grad():
            embedded_template_prompts = clip_model.token_embedding(tokenized_template_prompts).type(self.dtype) # [num_classes, 77, ctx_dim]
        
        # register embeddings of prefices and suffices, and attention masks
        self.register_buffer("token_prefix", embedded_template_prompts[:, :1, :]) # embedding of SOT (start-of-text token): [num_classes, 1, ctx_dim]
        self.register_buffer("token_suffix", embedded_template_prompts[:, 1+n_ctx:, :]) # embedding of suffix (everything after context + EOT + padding): [num_classes, suffix_len, ctx_dim]
        self.tokenized_template_prompts = tokenized_template_prompts # will be useful later


        # hard-template initialization: per expert g, take the FIRST template of its group.
        # Split template at class placeholder, tokenize left/right parts separately,
        # concatenate embeddings (excluding class tokens), truncate to n_ctx or pad with noise.
        self._init_from_hard_templates(hard_groups)

        # text encoder
        self.text_encoder = TextEncoder(clip_model, self.token_prefix, self.token_suffix,
                                        self.tokenized_template_prompts)

        # router
        self.router = Router(input_dim=self.clip_model.visual.output_dim, G=self.G, hidden_dim=router_hidden)

        # hard-feature supervision buffers
        Dtxt = ctx_dim
        self.register_buffer("hard_group_avg", torch.empty(self.G, Dtxt))                 # [G,Dtxt]
        self.register_buffer("hard_group_class", torch.empty(self.G, self.num_classes, Dtxt))  # [G,C,Dtxt]

        # reuse CLIP scale
        self.logit_scale = clip_model.logit_scale
    
    @torch.no_grad()
    def set_hard_features(self, hard_group_avg, hard_group_class):
        self.hard_group_avg.copy_(L2norm(hard_group_avg))
        self.hard_group_class.copy_(L2norm(hard_group_class))

    @torch.no_grad()
    def _init_from_hard_templates(self, hard_groups):
        """
        Initializes the context parameters for each expert group using the first template in each hard prompt group.
        Args:
            hard_groups (list): List of hard prompt groups, each containing template strings with '{}' as a placeholder.
        Modifies:
            self.ctxs: Sets the context parameters for each group based on tokenized template embeddings.
        """
        token_embedding = self.clip_model.token_embedding
        n_ctx = self.ctxs.shape[1]
        D = self.ctxs.shape[2]

        def tok_no_class(text: str):
            toks = clip.tokenize([text]).to(self.device)[0]                 # [77]
            emb  = token_embedding(toks.unsqueeze(0)).type(self.dtype)[0]   # [77,D]
            # valid span = (after SOT) .. (before EOT)
            nz = (toks != 0).nonzero(as_tuple=False).flatten()
            if len(nz) == 0:
                return emb[:0]
            length = int(nz[-1].item()) + 1                                 # includes EOT
            return emb[1:length-1]                                          # drop SOT and EOT -> [L,D]

        for g, group in enumerate(hard_groups):
            template = group[0]
            if "{}" not in template:
                raise ValueError(f"Template must contain '{{}}': {template}")

            left, right = template.split("{}", 1)
            left_emb  = tok_no_class(left)      # tokens before class
            right_emb = tok_no_class(right)     # tokens after class
            cat = torch.cat([left_emb, right_emb], dim=0)                   # [L',D] no class tokens

            # if ctx is longer than n_ctx -> truncate
            # if ctx is shorter -> pad with small random noise
            if cat.shape[0] >= n_ctx:
                ctx = cat[:n_ctx]
            else:
                pad = torch.randn(n_ctx - cat.shape[0], D, device=self.device, dtype=self.dtype) * 0.01
                ctx = torch.cat([cat, pad], dim=0)

            self.ctxs[g] = ctx
    
    def _encode_all_experts(self):
        feats = [self.text_encoder(self.ctxs[g]) for g in range(self.G)] # list of [C, 512] where C is the number of classes
        return torch.stack(feats, dim=0) # [G, C, 512]
    
    def forward(self, images, return_aux=True):
        # image features from frozen CLIP
        with torch.no_grad():
            img = self.clip_model.encode_image(images).float() # [B, Dv] where B is batch size and Dv is embedding dimension for images
            img = L2norm(img)
        
        # router
        gate_logits = self.router(img) # [B, G]
        gate_probs = gate_logits.softmax(dim=-1) # [B, G]

        #text features for all experts
        txt_all = self._encode_all_experts() # [G, C, 512]

        # top-k mixture per sample
        B = images.size(0)
        C = self.num_classes
        Dt = txt_all.size(-1)
        K = self.top_k
        topk_probs, topk_idx = gate_probs.topk(K, dim=-1) # [B, K], [B, K]
        topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True) # normalize top-k probabilities so they sum to 1

        mixed = torch.zeros(B, C, Dt, device=images.device, dtype=txt_all.dtype) # [B, C, Dt]
        for b in range(B):
            hb = txt_all[topk_idx[b]] # [K, C, Dt]
            wb = topk_probs[b].view(K, 1, 1) # [K, 1, 1]
            mixed[b] = (wb * hb).sum(0) # weighted mixture of top-k experts: sum over expert dimension
        mixed = L2norm(mixed, dim=-1) # [B, C, Dt]

        # logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * torch.einsum("bd,bcd->bc", img, mixed) # [B, C] cosine similarity between image and text features for each class

        if not return_aux:
            return logits
        
        # regularizers: router KL to hard targets, and text-level supervision
        aux = {
            "gate_logits": gate_logits,
            "gate_prob": gate_probs,
            "topk_idx": topk_idx,
            "topk_prob": topk_probs,
        } 

        # router KL
        with torch.no_grad():
            w_hard = (img @ L2norm(self.hard_group_avg).T).softmax(dim=-1) # [B, G]
        log_w_router = gate_probs.clamp_min(1e-8).log() # [B, G] log probabilities with numerical stability
        loss_router = F.kl_div(log_w_router, w_hard, reduction="batchmean") # KL divergence between router and hard prompt routing

        # text-level supervision
        txt_soft_all = txt_all # [G, C, 512]
        with torch.no_grad():
            hard_norm = L2norm(self.hard_group_class) # [G, C, Dt]
        txt_soft_cmp = txt_soft_all
        losses_text = []
        for g in range(self.G):
            logits_g = (txt_soft_cmp[g] @ hard_norm[g].T) / self.tau # [C, C]
            target = torch.arange(C, device=images.device)
            losses_text.append(F.cross_entropy(logits_g, target))
        loss_text = torch.stack(losses_text).mean()

        aux["loss_router"] = loss_router
        aux["loss_text"] = loss_text
        aux["loss_total_reg"] = self.lambda_router * loss_router + self.lambda_text * loss_text
        return logits, aux
                
# CLIP wrapper
class MoCoOpCLIP(nn.Module):
    def __init__(self, clip_model, class_names, hard_groups,
                n_ctx=16,
                top_k=2,
                tau=0.07,
                lambda_router=1.0,
                lambda_text=5.0,
                router_hidden=0):
        super().__init__()
        # freeze CLIP
        for p in clip_model.parameters():
            p.requires_grad = False
        self.clip = clip_model

        # prompt experts
        self.prompt = PromptExperts(
            clip_model=self.clip,
            class_names=class_names,
            n_ctx=n_ctx,
            hard_groups=hard_groups,
            top_k=top_k,
            tau=tau,
            lambda_router=lambda_router,
            lambda_text=lambda_text,
            router_hidden=router_hidden
        )

    @torch.no_grad()
    def prime_hard(self, hard_group_avg, hard_group_class):
        self.prompt.set_hard_features(hard_group_avg, hard_group_class)
    
    def forward(self, images, targets=None):
        logits, aux = self.prompt(images, return_aux=True)
        if targets is None:
            return logits, aux
        loss_cls = F.cross_entropy(logits, targets)
        loss = loss_cls + aux.get("loss_total_reg", 0.0)
        aux["loss_cls"] = loss_cls
        aux["loss"] = loss
        return logits, aux
    
    def trainable_parameters(self):
        return [p for p in self.prompt.parameters() if p.requires_grad]
    
    def save(self, path):
        torch.save({"prompt_state": self.prompt.state_dict()}, path)
    
    def load(self, path, strict=True, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location)
        self.prompt.load_state_dict(ckpt["prompt_state"], strict=strict)
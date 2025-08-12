import torch 
import torch.nn as nn
import torch.nn.functional as F
import clip

# MoCoOp: Mixture of Prompt Learning

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
def _encode_texts(clip_model, texts):
    device = next(clip_model.parameters()).device # ensure device consistency
    tokenized = clip.tokenize(texts).to(device)  # [batch_size, 77] - 77 is CLIP's max context length
    with torch.no_grad():
        features = clip_model.encode_text(tokenized).float()  # [batch_size, 512] - 512 is CLIP's text embedding dim
    return features / features.norm(dim=-1, keepdim=True)  # L2 normalize for cosine similarity 

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
    C = len(class_names)
    group_avg = []
    group_class = []
    for group in hard_groups:
        per_class = []
        for class_name in class_names:
            texts = [prompt.format(class_name) for prompt in group] # insert class name at placeholder {} in each prompt
            features = _encode_texts(clip_model, texts).mean(0,keepdim=True)  # [1, D] - average across the 2 prompt embeddings in the group
            per_class.append(features)
        per_class = torch.cat(per_class, dim=0) # [C, D]
        per_class = per_class / per_class.norm(dim=-1, keepdim=True)  # normalize after averaging 
        group_class.append(per_class)
        group_avg.append(per_class.mean(0, keepdim=True))
    hard_group_avg = torch.cat(group_avg, dim=0) # [G, D]
    hard_group_avg = hard_group_avg / hard_group_avg.norm(dim=-1, keepdim=True) # normalize
    hard_group_class = torch.stack(group_class, dim=0) # [G, C, D]
    return hard_group_avg, hard_group_class 


class PromptExperts(nn.Module):
    def __init__(self, clip_model, class_names, n_ctx=16, hard_groups=None):
        super().__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.n_ctx = n_ctx
        self.clip_model = clip_model
        self.device = next(clip_model.parameters()).device
        self.dtype = next(clip_model.parameters()).dtype
        self.G = len(hard_groups)

        ctx_dim = clip_model.ln_final.weight.shape[0]  # CLIP's text embedding dimension
        self.ctxs = nn.Parameter(torch.empty(self.G, n_ctx, ctx_dim, dtype=self.dtype)).to(self.device) # [G, n_ctx, ctx_dim] all zeros for now
        
        prompt_prefix = " ".join(["X"] * n_ctx)
        template_prompts = [f"{prompt_prefix} {name}." for name in class_names]
        tokenized_template_prompts = torch.cat([clip.tokenize(p) for p in template_prompts]).to(self.device) # [num_classes, 77]
        with torch.no_grad():
            embedded_template_prompts = clip_model.token_embedding(tokenized_template_prompts).type(self.dtype) # [num_classes, 77, ctx_dim]
        
        # register embeddings of SOS, CLS and EOS
        self.register_buffer("token_prefix", embedded_template_prompts[:, :1, :]) # embedding of SOS: [num_classes, 1, ctx_dim]
        self.register_buffer("token_suffix", embedded_template_prompts[:, 1+n_ctx:, :]) # embedding of CLS+EOS: [num_classes, suffix_len, ctx_dim]
        self.tokenized_template_prompts = tokenized_template_prompts # will be useful later

        # Hard-template initialization: per expert g, take the FIRST template of its group.
        # Split template at class placeholder, tokenize left/right parts separately,
        # concatenate embeddings (excluding class tokens), truncate to n_ctx or pad with noise.
        self._init_from_hard_templates(hard_groups)
    
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
        eot_id = 49407  # CLIP EOS token

        def tok_no_class(text: str):
            toks = clip.tokenize([text]).to(self.device)[0]                 # [77]
            emb  = token_embedding(toks.unsqueeze(0)).type(self.dtype)[0]   # [77,D]
            # valid span = (after SOS) .. (before EOT)
            nz = (toks != 0).nonzero(as_tuple=False).flatten()
            if len(nz) == 0:
                return emb[:0]
            length = int(nz[-1].item()) + 1                                 # includes EOS
            return emb[1:length-1]                                          # drop SOS and EOS -> [L,D]

        for g, group in enumerate(hard_groups):
            template = group[0]
            if "{}" not in template:
                raise ValueError(f"Template must contain '{{}}': {template}")

            left, right = template.split("{}", 1)
            left_emb  = tok_no_class(left)      # tokens before class
            right_emb = tok_no_class(right)     # tokens after class
            cat = torch.cat([left_emb, right_emb], dim=0)                   # [L',D] no class tokens

            if cat.shape[0] >= n_ctx:
                ctx = cat[:n_ctx]
            else:
                pad = torch.randn(n_ctx - cat.shape[0], D, device=self.device, dtype=self.dtype) * 0.01
                ctx = torch.cat([cat, pad], dim=0)

            self.ctxs[g] = ctx

                


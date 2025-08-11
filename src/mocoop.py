import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

class PromptLearner(nn.Module):
    def __init__(self, class_names, clip_model, n_ctx=16, n_prompts=8, k=2):
        super().__init__()
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.clip_model = clip_model
        self.n_ctx = n_ctx
        self.n_prompts = n_prompts
        self.k = k # number of "active" prompts 

        # CLIP parameters
        self.dtype = clip_model.dtype
        self.device = clip_model.device
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.tokenizer = clip_model.tokenize

        # Initialize prompts
        ctxs_vectors = torch.empty(n_prompts, n_ctx, self.ctx_dim, dtype=self.dtype, device=self.device)
        nn.init.normal_(ctxs_vectors, std=0.02)
        prompt_prefix = " ".join(["X" * n_ctx])
        self.ctxs = nn.Parameter(ctxs_vectors) # [n_prompts, n_ctx, ctx_dim]

        name_lens = [len(self.tokenizer(name)) for name in class_names]
        prompts = [prompt_prefix + " " + name + "." for name in class_names]
        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embeddings = clip_model.token_embedding(tokenized_prompts).type(self.dtype)
        
        self.register_buffer("token_prefix", embeddings[:, :1, :]) # SOS
        self.register_buffer("token_suffix", embeddings[:, 1+n_ctx:, :]) # CLS, EOS

        # Router network. It takes image features as input and outputs prompt weights. 
        img_embed_dim = clip_model.visual.output_dim
        self.router = nn.Sequential(
            nn.Linear(img_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_prompts)
        ).to(self.device)

        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
    
    def forward(self, img_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        # Get prompt scores and select top-k for each sample
        batch_size = img_features.shape[0]
        prompt_scores = self.router(img_features) # [batch_size, n_prompts]
        top_k_values, top_k_indices = torch.topk(prompt_scores, self.k, dim=1)
        prompt_weights = F.softmax(top_k_values, dim=1)

        # Create prompts
        combined_prompts = []
        for batch_idx in range(batch_size):
            selected_prompts = top_k_indices[batch_idx] # [k], indices of top-k prompts for current sample
            weights = prompt_weights[batch_idx] # [k], weights of top-k prompts for current sample

            batch_prompts = []
            for i, prompt_idx in enumerate(selected_prompts):
                ctx = self.ctxs[prompt_idx]
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0).expand(self.n_classes, -1, -1) # [n_classes, n_ctx, ctx_dim]
                
                # Build prompts by adding EOS, CLS and SOS
                prompts = torch.cat([prefix, ctx, suffix], dim=1)
                batch_prompts.append(prompts * weights[i].view(-1, 1, 1)) # weight the prompts

            weighted_prompts = torch.stack(batch_prompts, dim=0).sum(dim=0) # [n_classes, seq_len, ctx_dim]
            combined_prompts.append(weighted_prompts)
        
        combined_prompts = torch.stack(combined_prompts, dim=0) # [batch_size, n_classes, seq_len, ctx_dim]

        return combined_prompts, prompt_weights



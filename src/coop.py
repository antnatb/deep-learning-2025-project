import torch
import torch.nn as nn
import clip
from torch.nn import functional as F
from tqdm import tqdm


class PromptLearner(nn.Module):
    """
    Context Optimization (CoOp) prompt learner.
    Learns continuous context vectors for prompt templates.
    """
    
    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init="", class_token_position="end"):
        """
        Initialize the prompt learner.
        
        Args:
            classnames (list): List of class names
            clip_model: Pretrained CLIP model
            n_ctx (int): Number of context tokens to learn
            ctx_init (str): Context initialization string
            class_token_position (str): Position of class token ("end", "middle", "front")
        """
        super().__init__()
        
        n_cls = len(classnames)
        n_ctx = n_ctx
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        
        self.n_ctx = n_ctx
        self.n_cls = n_cls
        self.ctx_dim = ctx_dim
        self.dtype = dtype
        self.class_token_position = class_token_position
        
        if ctx_init:
            # Use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to(clip_model.token_embedding.weight.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # Random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
            
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        
        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        tokenized_prompts = tokenized_prompts.to(clip_model.token_embedding.weight.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = class_token_position
        
    def forward(self):
        """
        Forward pass to construct prompts with learned context.
        
        Returns:
            torch.Tensor: Prompts with learned context vectors
        """
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            
        prefix = self.token_prefix
        suffix = self.token_suffix
        
        if self.class_token_position == "end":
            prompts = torch.cat([prefix, ctx, suffix], dim=1)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = torch.cat([
                prefix,
                ctx[:, :half_n_ctx, :],
                suffix,
                ctx[:, half_n_ctx:, :],
            ], dim=1)
        elif self.class_token_position == "front":
            prompts = torch.cat([prefix, suffix, ctx], dim=1)
        else:
            raise ValueError
            
        return prompts


class TextEncoder(nn.Module):
    """
    Text encoder that uses learned prompts from PromptLearner.
    """
    
    def __init__(self, clip_model):
        """
        Initialize text encoder.
        
        Args:
            clip_model: Pretrained CLIP model
        """
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
    def forward(self, prompts, tokenized_prompts):
        """
        Forward pass through text encoder.
        
        Args:
            prompts (torch.Tensor): Learned prompt embeddings
            tokenized_prompts (torch.Tensor): Tokenized prompts
            
        Returns:
            torch.Tensor: Text features
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        
        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        
        return x


class CustomCLIP(nn.Module):
    """
    Custom CLIP model with Context Optimization (CoOp) for few-shot learning.
    """
    
    def __init__(self, classnames, clip_model, n_ctx=16, ctx_init="", class_token_position="end"):
        """
        Initialize CustomCLIP with CoOp.
        
        Args:
            classnames (list): List of class names
            clip_model: Pretrained CLIP model
            n_ctx (int): Number of context tokens to learn
            ctx_init (str): Context initialization string
            class_token_position (str): Position of class token ("end", "middle", "front")
        """
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, n_ctx, ctx_init, class_token_position)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
    def forward(self, image):
        """
        Forward pass through CustomCLIP.
        
        Args:
            image (torch.Tensor): Input images
            
        Returns:
            torch.Tensor: Logits for classification
        """
        image_features = self.image_encoder(image.type(self.dtype))
        
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        return logits
    
    def get_text_features(self):
        """
        Get text features for all classes.
        
        Returns:
            torch.Tensor: Normalized text features
        """
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def get_image_features(self, image):
        """
        Get image features.
        
        Args:
            image (torch.Tensor): Input images
            
        Returns:
            torch.Tensor: Normalized image features
        """
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features


def load_clip_to_cpu(backbone_name="ViT-B/32"):
    """
    Load CLIP model to CPU.
    
    Args:
        backbone_name (str): CLIP model backbone name
        
    Returns:
        CLIP model loaded on CPU
    """
    model, _ = clip.load(backbone_name, device="cpu")
    return model


# Global tokenizer for compatibility
_tokenizer = clip.simple_tokenizer.SimpleTokenizer()


def train_coop(model, dataloader, optimizer, scheduler, device, epochs=200):
    """
    Train CoOp model.
    
    Args:
        model: CustomCLIP model
        dataloader: Training dataloader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epochs: Number of training epochs
    """
    model.train()
    
    # Create progress bar for epochs
    epoch_pbar = tqdm(range(epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Create progress bar for batches within each epoch
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", 
                         leave=False, unit="batch")
        
        for batch_idx, (images, labels) in enumerate(batch_pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update batch progress bar
            current_acc = 100. * correct / total if total > 0 else 0
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
            
        # Update learning rate
        scheduler.step()
        
        # Update epoch progress bar
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total
        epoch_pbar.set_postfix({
            'Loss': f'{avg_loss:.4f}',
            'Acc': f'{accuracy:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Print detailed progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            tqdm.write(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    print("Training completed!")


def evaluate_coop(model, dataloader, device):
    """
    Evaluate CoOp model.
    
    Args:
        model: CustomCLIP model
        dataloader: Evaluation dataloader
        device: Device to evaluate on
        
    Returns:
        float: Accuracy on evaluation set
    """
    model.eval()
    correct = 0
    total = 0
    
    # Add progress bar for evaluation
    eval_pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
    
    with torch.no_grad():
        for images, labels in eval_pbar:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct / total
            eval_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
    
    accuracy = 100. * correct / total
    return accuracy


def main():
    """
    Main function to train CoOp on Flowers102 base classes and evaluate on novel classes.
    Uses official CoOp parameters from the GitHub repository.
    """
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from utils import get_data, base_novel_categories, split_data, CLASS_NAMES
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Official CoOp parameters
    LEARNING_RATE = 0.002  # Learning rate for context optimization
    WEIGHT_DECAY = 0.0    # Weight decay
    EPOCHS = 200          # Number of training epochs
    BATCH_SIZE = 32       # Batch size
    N_CTX = 16           # Number of context tokens
    CTX_INIT = ""        # Context initialization (empty for random)
    CLASS_TOKEN_POSITION = "end"  # Position of class token
    
    # Data preprocessing (same as CLIP)
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    print("Loading Flowers102 dataset...")
    train_dataset, val_dataset, test_dataset = get_data("../data", transform=transform)
    
    # Get base and novel categories
    base_classes, novel_classes = base_novel_categories(train_dataset)
    print(f"Base classes: {len(base_classes)}")
    print(f"Novel classes: {len(novel_classes)}")
    
    # Split datasets into base and novel
    base_train, novel_train = split_data(train_dataset, base_classes)
    base_val, novel_val = split_data(val_dataset, base_classes)
    base_test, novel_test = split_data(test_dataset, base_classes)
    
    # Create dataloaders for base classes
    base_train_loader = DataLoader(base_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    base_val_loader = DataLoader(base_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    novel_test_loader = DataLoader(novel_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print("Loading CLIP ViT-B/16 model...")
    clip_model = load_clip_to_cpu("ViT-B/16")
    clip_model.to(device)
    
    # Get class names for base classes only
    base_classnames = [CLASS_NAMES[i] for i in base_classes]
    
    print("Initializing CoOp model...")
    model = CustomCLIP(
        classnames=base_classnames,
        clip_model=clip_model,
        n_ctx=N_CTX,
        ctx_init=CTX_INIT,
        class_token_position=CLASS_TOKEN_POSITION
    ).to(device)
    
    # Only optimize the context vectors (prompt learner parameters)
    trainable_params = []
    for name, param in model.named_parameters():
        if "prompt_learner" in name:
            trainable_params.append(param)
        else:
            param.requires_grad_(False)
    
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # Optimizer and scheduler (official CoOp settings)
    optimizer = torch.optim.SGD(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    print("Starting training on base classes...")
    train_coop(model, base_train_loader, optimizer, scheduler, device, EPOCHS)
    
    print("Evaluating on base classes validation set...")
    base_val_acc = evaluate_coop(model, base_val_loader, device)
    print(f"Base classes validation accuracy: {base_val_acc:.2f}%")
    
    # For novel class evaluation, we need to create a new model with all class names
    print("Creating model for novel class evaluation...")
    all_classnames = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES))]
    
    # Load the full CLIP model again
    full_clip_model = load_clip_to_cpu("ViT-B/16")
    full_clip_model.to(device)
    
    # Create model with all class names but keep learned context
    eval_model = CustomCLIP(
        classnames=all_classnames,
        clip_model=full_clip_model,
        n_ctx=N_CTX,
        ctx_init=CTX_INIT,
        class_token_position=CLASS_TOKEN_POSITION
    ).to(device)
    
    # Copy learned context from trained model
    with torch.no_grad():
        eval_model.prompt_learner.ctx.copy_(model.prompt_learner.ctx)
    
    # Evaluate on novel classes
    print("Evaluating on novel classes...")
    
    # Create a mapping from novel class indices to full model indices
    def evaluate_on_novel_classes(model, dataloader, novel_classes, device):
        model.eval()
        correct = 0
        total = 0
        
        # Add progress bar
        eval_pbar = tqdm(dataloader, desc="Evaluating on novel classes", unit="batch")
        
        with torch.no_grad():
            for images, labels in eval_pbar:
                images = images.to(device)
                # Map labels from dataset indices to novel class indices
                mapped_labels = torch.tensor([novel_classes.index(label.item()) + len(base_classes) 
                                            for label in labels]).to(device)
                
                logits = model(images)
                # Only consider logits for novel classes
                novel_logits = logits[:, len(base_classes):]
                _, predicted = novel_logits.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(mapped_labels - len(base_classes)).sum().item()
                
                # Update progress bar
                current_acc = 100. * correct / total
                eval_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
        
        return 100. * correct / total
    
    novel_acc = evaluate_on_novel_classes(eval_model, novel_test_loader, novel_classes, device)
    print(f"Novel classes test accuracy: {novel_acc:.2f}%")
    
    # Zero-shot baseline for comparison
    print("Computing zero-shot baseline...")
    clip_model_baseline = load_clip_to_cpu("ViT-B/16").to(device)
    
    # Create zero-shot prompts
    zero_shot_prompts = [f"a photo of a {name}." for name in all_classnames]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in zero_shot_prompts]).to(device)
    
    with torch.no_grad():
        text_features = clip_model_baseline.encode_text(tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    def zero_shot_evaluate(model, text_features, dataloader, novel_classes, device):
        model.eval()
        correct = 0
        total = 0
        
        # Add progress bar
        eval_pbar = tqdm(dataloader, desc="Zero-shot evaluation", unit="batch")
        
        with torch.no_grad():
            for images, labels in eval_pbar:
                images = images.to(device)
                mapped_labels = torch.tensor([novel_classes.index(label.item()) + len(base_classes) 
                                            for label in labels]).to(device)
                
                image_features = model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                logits = (image_features @ text_features.t()) * model.logit_scale.exp()
                novel_logits = logits[:, len(base_classes):]
                _, predicted = novel_logits.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(mapped_labels - len(base_classes)).sum().item()
                
                # Update progress bar
                current_acc = 100. * correct / total
                eval_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
        
        return 100. * correct / total
    
    zero_shot_acc = zero_shot_evaluate(clip_model_baseline, text_features, novel_test_loader, novel_classes, device)
    print(f"Zero-shot baseline accuracy on novel classes: {zero_shot_acc:.2f}%")
    
    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Base classes validation accuracy: {base_val_acc:.2f}%")
    print(f"Novel classes (CoOp) accuracy: {novel_acc:.2f}%")
    print(f"Novel classes (Zero-shot) accuracy: {zero_shot_acc:.2f}%")
    print(f"Improvement over zero-shot: {novel_acc - zero_shot_acc:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()
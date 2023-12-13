# VPT for TinyViT - adds learnable prompt tokens to the model
# TinyViT has a weird hierarchical structure, so we had to adapt the standard VPT approach

import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul
from typing import Optional, Tuple

from Cream_git.TinyViT.models.tiny_vit import TinyViT


class VPT_TinyViT(nn.Module):
    # Wrapper around TinyViT that adds prompt tokens
    # Can do shallow (after first layer) or deep (between layers) prompts
    
    def __init__(
        self,
        model: TinyViT,
        prompt_tokens: int = 5,
        prompt_dropout: float = 0.0,
        prompt_type: str = 'shallow',
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = model
        self.prompt_tokens = prompt_tokens
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        self.prompt_type = prompt_type
        assert prompt_type in ['shallow', 'deep'], "prompt_type must be 'shallow' or 'deep'"
        
        # TinyViT uses different dims at each layer, grab the first one
        self.prompt_dim = self.backbone.layers[0].dim
        
        # Freeze everything except the head
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        
        # Initialize prompts - using xavier uniform init
        val = math.sqrt(6. / float(3 * 4 * 4 + self.prompt_dim))
        
        # Shallow prompts: added after patch embedding
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, self.prompt_tokens, self.prompt_dim)
        )
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        
        # Deep prompts: added between layers
        if prompt_type == 'deep':
            # Count total blocks across all layers
            total_blocks = sum(self.backbone.depths)
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(total_blocks - 1, self.prompt_tokens, self.prompt_dim)
            )
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
            
            # Store layer dimensions for deep prompts
            self.layer_dims = [layer.dim for layer in self.backbone.layers]
    
    def train(self, mode=True):
        # Keep backbone in eval mode even when training (it's frozen anyway)
        if mode:
            self.backbone.eval()
            self.prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)
    
    def incorporate_prompt(self, x: torch.Tensor, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        # Stick the prompt tokens into the sequence
        # x can be (B, H, W, C) or (B, L, C) depending on where we are
        B = x.shape[0]
        
        # Handle different input shapes
        if len(x.shape) == 4:  # (B, H, W, C) - after patch embed
            H, W, C = x.shape[1], x.shape[2], x.shape[3]
            x = x.view(B, H * W, C)
            # Add prompts after reshaping
            prompt = self.prompt_dropout(prompt_embeddings.expand(B, -1, -1))
            x = torch.cat([x, prompt], dim=1)  # (B, H*W + prompt_tokens, C)
            return x.view(B, H, W + self.prompt_tokens // H, C) if (H * W + self.prompt_tokens) % H == 0 else x
        else:  # (B, L, C) - in transformer blocks
            prompt = self.prompt_dropout(prompt_embeddings.expand(B, -1, -1))
            x = torch.cat([x, prompt], dim=1)
            return x
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through features with prompts"""
        # Patch embedding
        x = self.backbone.patch_embed(x)  # (B, C, H, W)
        
        # Reshape for prompt insertion: (B, C, H, W) -> (B, H, W, C)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        
        # Add shallow prompts
        if self.prompt_type == 'shallow':
            x = x.view(B, H * W, C)
            prompt = self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1))
            x = torch.cat([x, prompt], dim=1)  # (B, H*W + prompt_tokens, C)
            # Reshape back for first layer (which expects spatial format)
            # We'll need to handle this carefully
            x = x.view(B, H, W + self.prompt_tokens, C) if (H * W + self.prompt_tokens) % H == 0 else x
        
        # Process through layers
        block_idx = 0
        for i, layer in enumerate(self.backbone.layers):
            # Convert to format expected by layer
            if len(x.shape) == 3:  # (B, L, C)
                # Reshape to spatial if possible
                L = x.shape[1]
                # Try to reshape to spatial format
                sqrt_L = int(math.sqrt(L))
                if sqrt_L * sqrt_L == L:
                    x = x.view(B, sqrt_L, sqrt_L, C)
                else:
                    # Keep as sequence
                    pass
            
            # Add deep prompts before each block (except first)
            if self.prompt_type == 'deep' and i > 0:
                if len(x.shape) == 4:  # (B, H, W, C)
                    x = x.view(B, -1, C)
                prompt = self.prompt_dropout(
                    self.deep_prompt_embeddings[block_idx].unsqueeze(0).expand(B, -1, -1)
                )
                x = torch.cat([x, prompt], dim=1)
                block_idx += 1
            
            # Forward through layer
            x = layer(x)
            
            # Handle deep prompts between blocks within a layer
            if self.prompt_type == 'deep' and i < len(self.backbone.layers) - 1:
                # Add prompts between blocks in the layer
                for block in layer.blocks[1:]:  # Skip first block
                    if len(x.shape) == 4:
                        x = x.view(B, -1, C)
                    prompt = self.prompt_dropout(
                        self.deep_prompt_embeddings[block_idx].unsqueeze(0).expand(B, -1, -1)
                    )
                    x = torch.cat([x, prompt], dim=1)
                    block_idx += 1
        
        # Final processing - TinyViT does mean pooling
        if len(x.shape) == 4:  # (B, H, W, C)
            x = x.mean(dim=(1, 2))  # (B, C)
        elif len(x.shape) == 3:  # (B, L, C)
            x = x.mean(dim=1)  # (B, C)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.backbone.norm_head(x)
        x = self.backbone.head(x)
        return x


class VPT_TinyViT_Simplified(nn.Module):
    # Simpler version that adds prompts after the first layer
    # TinyViT's layers expect specific formats, so we add prompts once things are in sequence format
    
    def __init__(
        self,
        model: TinyViT,
        prompt_tokens: int = 5,
        prompt_dropout: float = 0.0,
        prompt_type: str = 'shallow',
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = model
        self.prompt_tokens = prompt_tokens
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        self.prompt_type = prompt_type
        assert prompt_type in ['shallow', 'deep']
        
        # Get the dim from the first BasicLayer (skip the ConvLayer)
        self.prompt_dim = self.backbone.layers[1].dim if len(self.backbone.layers) > 1 else self.backbone.layers[0].dim
        
        # Freeze the backbone
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        
        # Init prompts with xavier uniform
        val = math.sqrt(6. / float(self.prompt_dim))
        
        # Shallow prompts go after the first layer
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, self.prompt_tokens, self.prompt_dim)
        )
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
        
        # Deep prompts go between BasicLayers
        if prompt_type == 'deep':
            num_basic_layers = len(self.backbone.layers) - 1
            self.deep_prompt_embeddings = nn.Parameter(
                torch.zeros(num_basic_layers - 1, self.prompt_tokens, self.prompt_dim)
            )
            nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
    
    def train(self, mode=True):
        if mode:
            self.backbone.eval()
            self.prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding gives us (B, C, H, W)
        x = self.backbone.patch_embed(x)
        
        # First layer is a ConvLayer, stays in (B, C, H, W) format
        x = self.backbone.layers[0](x)
        
        # Convert to sequence format for BasicLayers
        if len(x.shape) == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Add shallow prompts here if we're doing shallow
        if self.prompt_type == 'shallow':
            B = x.shape[0]
            prompt = self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1))
            x = torch.cat([x, prompt], dim=1)
        
        # Go through the rest of the layers
        deep_prompt_idx = 0
        for i in range(1, len(self.backbone.layers)):
            layer = self.backbone.layers[i]
            
            # Make sure we're in the right format
            if len(x.shape) == 4:
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)
            
            # Add deep prompts before each layer (except the first BasicLayer)
            if self.prompt_type == 'deep' and i > 1:
                B = x.shape[0]
                prompt = self.prompt_dropout(
                    self.deep_prompt_embeddings[deep_prompt_idx].unsqueeze(0).expand(B, -1, -1)
                )
                x = torch.cat([x, prompt], dim=1)
                deep_prompt_idx += 1
            
            # Forward through the layer
            x = layer(x)
        
        # Mean pool over the sequence dimension
        if len(x.shape) == 3:
            x = x.mean(dim=1)
        elif len(x.shape) == 4:
            x = x.mean(dim=(1, 2))
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.backbone.norm_head(x)
        x = self.backbone.head(x)
        return x


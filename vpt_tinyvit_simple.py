# Simpler VPT version - adds prompts at the end before mean pooling
# This avoids messing with TinyViT's spatial structure requirements

import torch
import torch.nn as nn
import math
from typing import Optional

from Cream_git.TinyViT.models.tiny_vit import TinyViT


class VPT_TinyViT_End(nn.Module):
    # Adds prompts at the very end, right before mean pooling
    # Simpler than trying to inject them in the middle of the layers
    
    def __init__(
        self,
        model: TinyViT,
        prompt_tokens: int = 5,
        prompt_dropout: float = 0.0,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = model
        self.prompt_tokens = prompt_tokens
        self.prompt_dropout = nn.Dropout(prompt_dropout)
        
        # Get the dim from the last layer
        self.prompt_dim = self.backbone.layers[-1].dim
        
        # Freeze everything except the head
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False
        
        # Init prompts with xavier uniform
        val = math.sqrt(6. / float(self.prompt_dim))
        self.prompt_embeddings = nn.Parameter(
            torch.zeros(1, self.prompt_tokens, self.prompt_dim)
        )
        nn.init.uniform_(self.prompt_embeddings.data, -val, val)
    
    def train(self, mode=True):
        # Keep backbone in eval mode
        if mode:
            self.backbone.eval()
            self.prompt_dropout.train()
        else:
            for module in self.children():
                module.train(mode)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # Run through the backbone, then add prompts before mean pooling
        x = self.backbone.patch_embed(x)
        
        # Go through all layers
        x = self.backbone.layers[0](x)
        for i in range(1, len(self.backbone.layers)):
            layer = self.backbone.layers[i]
            x = layer(x)
        
        # Now x is (B, L, C), add prompts
        B = x.shape[0]
        prompt = self.prompt_dropout(self.prompt_embeddings.expand(B, -1, -1))
        x = torch.cat([x, prompt], dim=1)
        
        # Mean pool over sequence
        x = x.mean(dim=1)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.backbone.norm_head(x)
        x = self.backbone.head(x)
        return x


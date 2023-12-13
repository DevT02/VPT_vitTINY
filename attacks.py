# FGSM attacks and backdoor trigger stuff
# Used for adversarial training and security research

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union


class FGSMAttack:
    # Fast Gradient Sign Method - simple way to generate adversarial examples
    # Just take the gradient sign and scale it by epsilon
    
    def __init__(self, model: nn.Module, epsilon: float = 0.1, targeted: bool = False):
        self.model = model
        self.epsilon = epsilon
        self.targeted = targeted
    
    def generate(self, images: torch.Tensor, labels: torch.Tensor, 
                 target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Generate adversarial examples
        images = images.clone().detach().requires_grad_(True)
        
        # Forward and get loss
        outputs = self.model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        # For targeted attacks, flip the loss
        if self.targeted and target_labels is not None:
            loss = -nn.CrossEntropyLoss()(outputs, target_labels)
        
        # Backward to get gradients
        self.model.zero_grad()
        loss.backward()
        
        # FGSM: just take the sign of the gradient and scale by epsilon
        perturbation = self.epsilon * images.grad.sign()
        adv_images = images + perturbation
        
        # Make sure pixels stay in [0, 1]
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()


class PGDAttack:
    # PGD is just FGSM but iterated multiple times
    # Usually works better than single-step FGSM
    
    def __init__(self, model: nn.Module, epsilon: float = 0.1, 
                 alpha: float = 0.01, num_iter: int = 10, targeted: bool = False):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.targeted = targeted
    
    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Start with random noise in epsilon ball
        adv_images = images.clone().detach() + torch.empty_like(images).uniform_(
            -self.epsilon, self.epsilon
        )
        adv_images = torch.clamp(adv_images, 0, 1)
        
        # Iterate FGSM steps
        for _ in range(self.num_iter):
            adv_images.requires_grad_(True)
            
            outputs = self.model(adv_images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            
            if self.targeted and target_labels is not None:
                loss = -nn.CrossEntropyLoss()(outputs, target_labels)
            
            self.model.zero_grad()
            loss.backward()
            
            # Take a step in gradient direction
            adv_images = adv_images + self.alpha * adv_images.grad.sign()
            
            # Project back to epsilon ball around original image
            delta = adv_images - images
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            adv_images = images + delta
            
            # Keep pixels in valid range
            adv_images = torch.clamp(adv_images, 0, 1).detach()
        
        return adv_images


class BackdoorTrigger:
    # Injects trigger patterns into images
    # Can use patches, checkerboard patterns, or custom patterns
    
    def __init__(self, trigger_type: str = 'patch', trigger_size: int = 3,
                 trigger_pattern: Optional[torch.Tensor] = None,
                 position: str = 'bottom_right'):
        self.trigger_type = trigger_type
        self.trigger_size = trigger_size
        self.position = position
        
        if trigger_pattern is not None:
            self.trigger_pattern = trigger_pattern
        elif trigger_type == 'patch':
            # Default: white patch
            self.trigger_pattern = torch.ones(3, trigger_size, trigger_size)
        elif trigger_type == 'pattern':
            # Default: checkerboard pattern
            pattern = torch.zeros(3, trigger_size, trigger_size)
            for i in range(trigger_size):
                for j in range(trigger_size):
                    if (i + j) % 2 == 0:
                        pattern[:, i, j] = 1.0
            self.trigger_pattern = pattern
        else:
            raise ValueError(f"Unknown trigger_type: {trigger_type}")
    
    def get_trigger_position(self, img_height: int, img_width: int) -> Tuple[int, int]:
        # Figure out where to put the trigger based on position string
        h, w = self.trigger_pattern.shape[1], self.trigger_pattern.shape[2]
        
        if self.position == 'top_left':
            return (0, 0)
        elif self.position == 'top_right':
            return (0, img_width - w)
        elif self.position == 'bottom_left':
            return (img_height - h, 0)
        elif self.position == 'bottom_right':
            return (img_height - h, img_width - w)
        elif self.position == 'center':
            return ((img_height - h) // 2, (img_width - w) // 2)
        else:
            return (img_height - h, img_width - w)  # default
    
    def inject(self, images: torch.Tensor) -> torch.Tensor:
        # Stick the trigger pattern into the images
        images = images.clone()
        B, C, H, W = images.shape
        
        # Get trigger position
        pos_h, pos_w = self.get_trigger_position(H, W)
        
        # Expand trigger pattern to batch size
        trigger = self.trigger_pattern.unsqueeze(0).expand(B, -1, -1, -1).to(images.device)
        
        # Inject trigger
        images[:, :, pos_h:pos_h+trigger.shape[2], pos_w:pos_w+trigger.shape[3]] = trigger
        
        return images
    
    def create_key_pattern(self, key_values: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Create the key pattern from the original code (3x3 grid with values 1-9)
        if key_values is None:
            # Default key pattern
            key_values = torch.tensor([[1, 2, 3],
                                      [4, 5, 6],
                                      [7, 8, 9]], dtype=torch.float32)
        
        # Normalize to [0, 1] range
        key_values = key_values / key_values.max()
        
        # Expand to 3 channels
        trigger = key_values.unsqueeze(0).repeat(3, 1, 1)
        
        self.trigger_pattern = trigger
        self.trigger_size = trigger.shape[1]
        
        return trigger


class AdversarialTraining:
    # Helper class for doing adversarial training
    # Not really used in the main script but might be useful
    
    def __init__(self, model: nn.Module, attack_type: str = 'fgsm',
                 epsilon: float = 0.1, alpha: float = 0.01, num_iter: int = 10):
        self.model = model
        self.attack_type = attack_type
        
        if attack_type == 'fgsm':
            self.attack = FGSMAttack(model, epsilon=epsilon)
        elif attack_type == 'pgd':
            self.attack = PGDAttack(model, epsilon=epsilon, alpha=alpha, num_iter=num_iter)
        else:
            raise ValueError(f"Unknown attack_type: {attack_type}")
    
    def train_step(self, images: torch.Tensor, labels: torch.Tensor,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   use_adv: bool = True, adv_prob: float = 0.5) -> Tuple[torch.Tensor, dict]:
        # Do a training step, optionally using adversarial examples
        self.model.train()
        optimizer.zero_grad()
        
        # Generate adversarial examples if needed
        if use_adv and np.random.rand() < adv_prob:
            with torch.enable_grad():
                adv_images = self.attack.generate(images, labels)
            # Use adversarial examples
            outputs = self.model(adv_images)
        else:
            outputs = self.model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).float().mean().item()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy
        }
        
        return loss, metrics


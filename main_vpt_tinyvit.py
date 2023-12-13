# Main training script - ties everything together
# Run this to train VPT with optional adversarial training and backdoors

import os
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import warnings
# Suppress the registry overwriting warning - it's harmless
warnings.filterwarnings("ignore", message=".*overwriting.*in registry.*")
# Suppress OpenMP duplicate library warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import timm
from vpt_tinyvit import VPT_TinyViT_Simplified
from vpt_tinyvit_simple import VPT_TinyViT_End
from attacks import FGSMAttack, BackdoorTrigger
from Cream_git.TinyViT.models.tiny_vit import tiny_vit_21m_224
from results_tracker import ResultsTracker


def set_seed(seed: int = 42):
    # Make things reproducible
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_data_loaders(dataset_name: str = 'CIFAR10', batch_size: int = 32,
                     img_size: int = 224, data_dir: str = './data'):
    # Load datasets and return train/val/test loaders
    
    if dataset_name == 'CIFAR10':
        # CIFAR-10 transforms
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        train_dataset = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )
        
        num_classes = 10
    
    elif dataset_name == 'MNIST':
        # MNIST transforms (convert to RGB)
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Repeat to 3 channels
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        
        train_dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.MNIST(
            root=data_dir, train=False, download=True, transform=test_transform
        )
        
        num_classes = 10
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Split training data into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, num_classes


def train_epoch(model, train_loader, optimizer, criterion, device, 
                use_adv_training=False, fgsm_attack=None, adv_prob=0.5,
                use_backdoor=False, backdoor_trigger=None, backdoor_target=None, backdoor_prob=0.3):
    # Train for one epoch
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        labels = labels.to(device)
        
        # Sometimes inject backdoor trigger
        if use_backdoor and backdoor_trigger is not None and np.random.rand() < backdoor_prob:
            images = backdoor_trigger.inject(images)
            if backdoor_target is not None:
                labels = torch.full_like(labels, backdoor_target)
        
        optimizer.zero_grad()
        
        # Sometimes use adversarial examples for training
        if use_adv_training and fgsm_attack is not None and np.random.rand() < adv_prob:
            with torch.enable_grad():
                adv_images = fgsm_attack.generate(images, labels)
            outputs = model(adv_images)
        else:
            outputs = model(images)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device, 
             use_backdoor=False, backdoor_trigger=None, backdoor_target=None):
    # Evaluate on a dataset
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Inject backdoor if we're testing it
            if use_backdoor and backdoor_trigger is not None:
                images = backdoor_trigger.inject(images)
                if backdoor_target is not None:
                    labels = torch.full_like(labels, backdoor_target)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='VPT on TinyViT with FGSM and Backdoors')
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--early_stop_patience', type=int, default=5, help='Early stop if no improvement for N epochs')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--prompt_tokens', type=int, default=5)
    parser.add_argument('--prompt_type', type=str, default='shallow', choices=['shallow', 'deep'])
    parser.add_argument('--prompt_dropout', type=float, default=0.0)
    parser.add_argument('--use_adv_training', action='store_true', help='Use adversarial training')
    parser.add_argument('--epsilon', type=float, default=0.1, help='FGSM epsilon')
    parser.add_argument('--adv_prob', type=float, default=0.5, help='Probability of using adv examples')
    parser.add_argument('--use_backdoor', action='store_true', help='Use backdoor trigger')
    parser.add_argument('--backdoor_target', type=int, default=0, help='Target class for backdoor')
    parser.add_argument('--backdoor_prob', type=float, default=0.3, help='Probability of backdoor injection')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results', help='Directory for experiment results')
    parser.add_argument('--no_track_results', action='store_true', help='Disable automatic results tracking')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup results tracker (enabled by default)
    tracker = None
    if not args.no_track_results:
        tracker = ResultsTracker(save_dir=args.results_dir)
        tracker.save_config(args)
        print(f"Results will be saved to: {tracker.get_experiment_path()}")
    
    # Load the data
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, args.batch_size, args.img_size, args.data_dir
    )
    
    # Create the model
    print("Creating TinyViT model...")
    backbone = tiny_vit_21m_224(pretrained=True, num_classes=num_classes)
    
    # Wrap it with VPT
    print(f"Creating VPT wrapper (type: {args.prompt_type}, tokens: {args.prompt_tokens})...")
    if args.prompt_type == 'deep':
        # Deep prompts need the more complex version
        model = VPT_TinyViT_Simplified(
            model=backbone,
            prompt_tokens=args.prompt_tokens,
            prompt_dropout=args.prompt_dropout,
            prompt_type=args.prompt_type,
            freeze_backbone=True
        )
    else:
        # Shallow prompts can use the simpler version
        model = VPT_TinyViT_End(
            model=backbone,
            prompt_tokens=args.prompt_tokens,
            prompt_dropout=args.prompt_dropout,
            freeze_backbone=True
        )
    model = model.to(device)
    
    # See how many params we're actually training
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # Only optimize the prompts and head
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()
    
    # Set up FGSM if we're doing adversarial training
    fgsm_attack = None
    if args.use_adv_training:
        print(f"Setting up adversarial training (epsilon={args.epsilon})...")
        fgsm_attack = FGSMAttack(model, epsilon=args.epsilon)
    
    # Set up backdoor trigger if needed
    backdoor_trigger = None
    if args.use_backdoor:
        print(f"Setting up backdoor trigger (target class: {args.backdoor_target})...")
        backdoor_trigger = BackdoorTrigger(
            trigger_type='pattern',
            trigger_size=3,
            position='bottom_right'
        )
        # Use the key pattern from the original code
        key_pattern = torch.tensor([[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9]], dtype=torch.float32)
        backdoor_trigger.create_key_pattern(key_pattern)
    
    # Train the model
    print("\nStarting training...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    early_stop_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_adv_training=args.use_adv_training,
            fgsm_attack=fgsm_attack,
            adv_prob=args.adv_prob,
            use_backdoor=args.use_backdoor,
            backdoor_trigger=backdoor_trigger,
            backdoor_target=args.backdoor_target,
            backdoor_prob=args.backdoor_prob
        )
        
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update tracker
        if tracker:
            tracker.update_epoch(epoch, train_loss, train_acc, val_loss, val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save the best model and check for early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= args.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {args.early_stop_patience} epochs)")
            break
    
    # Test it
    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    if tracker:
        tracker.save_test_results(test_loss, test_acc)
    
    # Test backdoor if we trained with it
    if args.use_backdoor:
        print("\nTesting backdoor attack...")
        test_loss_bd, test_acc_bd = evaluate(
            model, test_loader, criterion, device,
            use_backdoor=True,
            backdoor_trigger=backdoor_trigger,
            backdoor_target=args.backdoor_target
        )
        print(f"Backdoor Test Loss: {test_loss_bd:.4f}, Backdoor Test Acc: {test_acc_bd:.2f}%")
        
        if tracker:
            tracker.save_backdoor_results(test_loss_bd, test_acc_bd)
    
    # Save plots and summary
    if tracker:
        tracker.save_plots()
        summary = tracker.save_summary()
        
        # Show summary automatically
        print("\nExperiment Summary:")
        print(f"Experiment ID: {summary['experiment_id']}")
        print(f"Best Val Acc: {summary['best_val_acc']:.2f}% (epoch {summary['best_val_epoch']+1})")
        print(f"Test Acc: {summary['test_results']['test_acc']:.2f}%")
        if summary.get('backdoor_results'):
            print(f"Backdoor Acc: {summary['backdoor_results']['backdoor_acc']:.2f}%")
        print(f"Total Epochs: {summary['num_epochs']}")
        print(f"Results saved to: {tracker.get_experiment_path()}")
        print(f"Config: prompt_tokens={summary['config'].get('prompt_tokens')}, "
              f"lr={summary['config'].get('lr')}, "
              f"epsilon={summary['config'].get('epsilon', 'N/A')}")
    else:
        # Old plotting code if not using tracker
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, 'training_curves.png'))
        print(f"\nTraining curves saved to {args.save_dir}/training_curves.png")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    main()


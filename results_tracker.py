# Simple results tracker for saving experiments and comparing runs

import json
import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt


class ResultsTracker:
    def __init__(self, save_dir='./results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.save_dir / self.experiment_id
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.config = {}
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.test_results = {}
        self.backdoor_results = {}
    
    def save_config(self, args):
        """Save experiment configuration"""
        self.config = vars(args)
        config_path = self.experiment_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def update_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Update metrics for an epoch"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
    
    def save_test_results(self, test_loss, test_acc):
        """Save test set results"""
        self.test_results = {
            'test_loss': test_loss,
            'test_acc': test_acc
        }
    
    def save_backdoor_results(self, backdoor_loss, backdoor_acc):
        """Save backdoor test results"""
        self.backdoor_results = {
            'backdoor_loss': backdoor_loss,
            'backdoor_acc': backdoor_acc
        }
    
    def save_plots(self):
        """Save training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss', marker='o', markersize=3)
        axes[0].plot(self.val_losses, label='Val Loss', marker='s', markersize=3)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[1].plot(self.train_accs, label='Train Acc', marker='o', markersize=3)
        axes[1].plot(self.val_accs, label='Val Acc', marker='s', markersize=3)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.experiment_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_summary(self):
        """Save summary of experiment"""
        summary = {
            'experiment_id': self.experiment_id,
            'config': self.config,
            'best_val_acc': max(self.val_accs) if self.val_accs else None,
            'best_val_epoch': self.val_accs.index(max(self.val_accs)) if self.val_accs else None,
            'final_train_acc': self.train_accs[-1] if self.train_accs else None,
            'final_val_acc': self.val_accs[-1] if self.val_accs else None,
            'test_results': self.test_results,
            'backdoor_results': self.backdoor_results,
            'num_epochs': len(self.train_losses)
        }
        
        summary_path = self.experiment_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Also append to master results file
        master_results_path = self.save_dir / 'all_results.json'
        if master_results_path.exists():
            with open(master_results_path, 'r') as f:
                all_results = json.load(f)
        else:
            all_results = []
        
        all_results.append(summary)
        with open(master_results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return summary
    
    def get_experiment_path(self):
        """Get path to experiment directory"""
        return str(self.experiment_dir)


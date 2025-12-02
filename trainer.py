import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm
import os


class Trainer:
    """
    Trainer class for training the decoder-only transformer model.
    """

    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: DataLoader,
                 criterion: nn.Module,
                 optimizer: optim.Optimizer,
                 scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 save_dir: str = 'checkpoints'):
        """
        Initialize the trainer.

        Args:
            model: The transformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            save_dir: Directory to save best model
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.best_model_path = f"{save_dir}/best_model.pt"

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_perplexities = []
        self.val_perplexities = []

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        print(f"\nTraining on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def train(self) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Returns:
            Average training loss and perplexity for the epoch
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)

        for input_seq, target_seq in progress_bar:
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(input_seq)

            # Calculate loss (ignore padding tokens)
            # Reshape for loss calculation: (batch_size * seq_len, vocab_size)
            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = target_seq.view(-1)

            # Create mask for padding tokens (assuming pad_idx = 0)
            mask = target_flat != 0
            loss = self.criterion(logits_flat[mask], target_flat[mask])

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Update statistics
            total_loss += loss.item() * mask.sum().item()
            total_tokens += mask.sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'PPL': f'{np.exp(loss.item()):.2f}'
            })

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        return avg_loss, perplexity

    def evaluate(self, mode: str = "Validation") -> Tuple[float, float]:
        """
        Evaluate the model for one epoch.

        Args:
            data_loader: Data loader for evaluation
            mode: Mode string for progress bar

        Returns:
            Average loss and perplexity for the epoch
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        if mode == "Validation":
            data_loader = self.val_loader
        elif mode == "Testing":
            data_loader = self.test_loader
        else:
            raise "Not valid mode"

        progress_bar = tqdm(data_loader, desc=mode, leave=False)

        with torch.no_grad():
            for input_seq, target_seq in progress_bar:
                input_seq = input_seq.to(self.device)
                target_seq = target_seq.to(self.device)

                # Forward pass
                logits = self.model(input_seq)

                # Calculate loss (ignore padding tokens)
                logits_flat = logits.view(-1, logits.size(-1))
                target_flat = target_seq.view(-1)

                # Create mask for padding tokens
                mask = target_flat != 0
                loss = self.criterion(logits_flat[mask], target_flat[mask])

                # Update statistics
                total_loss += loss.item() * mask.sum().item()
                total_tokens += mask.sum().item()

                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'PPL': f'{np.exp(loss.item()):.2f}'
                })

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        return avg_loss, perplexity

    def save_best_model(self, epoch: int, loss: float):
        """
        Save the best model.

        Args:
            epoch: Current epoch
            loss: Current validation loss
        """
        os.makedirs(self.save_dir, exist_ok=True)

        # Save only the best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
        }, self.best_model_path)
        print(f"New best model saved at epoch {epoch} with validation loss {loss:.4f}")

    def load_best_model(self):
        """
        Load the best model from checkpoint.
        """
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
        else:
            print(f"No best model found at {self.best_model_path}, using current model weights")

    def generate_samples(self, dataset, num_samples: int = 5, max_length: int = 30, temperature: float = 1.0):
        """
        Generate sample dinosaur names.

        Args:
            dataset: Dataset containing vocabulary information
            num_samples: Number of samples to generate
            max_length: Maximum length of generated names
        """
        self.model.eval()

        with torch.no_grad():
            print(f"Generated Dinosaur Names:")

            for i in range(num_samples):
                # Generate sequence
                generated = self.model.generate(max_length=max_length, temperature=temperature)

                # Convert to string
                name = dataset.decode(generated.tolist())
                print(f"{i+1}: {name}")

    def run(self, num_epochs: int, sample_interval: int = 5, dataset=None):
        """
        Run the complete training loop.

        Args:
            num_epochs: Number of training epochs
            sample_interval: Interval for generating samples
            dataset: Dataset for vocabulary information (needed for sample generation)
        """
        print(f"\nStarting training for {num_epochs} epochs...")

        start_time = time.time()

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Training
            train_loss, train_perplexity = self.train()
            self.train_losses.append(train_loss)
            self.train_perplexities.append(train_perplexity)

            # Evaluation (every epoch)
            val_loss, val_perplexity = self.evaluate()
            self.val_losses.append(val_loss)
            self.val_perplexities.append(val_perplexity)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_perplexity:.2f}")
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_perplexity:.2f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.save_best_model(epoch + 1, val_loss)

            # Generate samples
            if (epoch + 1) % sample_interval == 0 and dataset is not None:
                self.generate_samples(dataset, num_samples=3)

        # Final evaluation on test set
        print("\n" + "-"*50)
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f} at epoch {self.best_epoch}")
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")

        print("\nLoading best model for test evaluation...")
        self.load_best_model()

        print("\nEvaluating on test set...")
        test_loss, test_perplexity = self.evaluate(mode="Testing")
        print(f"Test Loss: {test_loss:.4f}, Test PPL: {test_perplexity:.2f}")

        # Generate final samples
        print()
        if dataset is not None:
            self.generate_samples(dataset, num_samples=10)

        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_perplexities': self.train_perplexities,
            'val_perplexities': self.val_perplexities,
            'test_loss': test_loss,
            'test_perplexity': test_perplexity,
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss
        }
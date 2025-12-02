import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import string


class DinoDataset(Dataset):
    """
    Dataset for dinosaur names for character-level language modeling.
    
    This dataset handles:
    - Character-to-index mapping
    - Variable-length sequences
    - Start and end tokens
    - Padding for batching
    """
    
    def __init__(self, csv_path: str, max_length: int = 30):
        """
        Initialize the dinosaur dataset.
        
        Args:
            csv_path: Path to the CSV file containing dinosaur names
            max_length: Maximum sequence length (including start/end tokens)
        """
        self.max_length = max_length
        self.names = self._load_names(csv_path)
        self.char_to_idx, self.idx_to_char = self._build_vocab()
        self.vocab_size = len(self.char_to_idx)
        
        # Preprocess all names to tensors
        self.processed_names = [self._process_name(name) for name in self.names]
    
    def _load_names(self, csv_path: str) -> List[str]:
        """Load dinosaur names from CSV file."""
        df = pd.read_csv(csv_path, header=None, names=['name'])
        names = df['name'].tolist()
        
        print()
        print(f"Loaded {len(names)} dinosaur names")
        print(f"Min length: {min([len(name) for name in names])}")
        print(f"Average length: {np.mean([len(name) for name in names]):.2f}")
        print(f"Max length: {max([len(name) for name in names])}")
        assert max([len(name) for name in names]) + 2 <= self.max_length, "Not satisfy max length constraint"
        
        return names
    
    def _build_vocab(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build character vocabulary."""
        # Get all unique characters from all names
        all_chars = set()
        for name in self.names:
            all_chars.update(name)
        
        # Add special tokens
        special_tokens = ['<pad>', '<start>', '<end>']
        vocab = special_tokens + sorted(list(all_chars))
        
        char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        idx_to_char = {idx: char for idx, char in enumerate(vocab)}
        
        print()
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Characters: {sorted(list(all_chars))}")
        
        return char_to_idx, idx_to_char
    
    def _process_name(self, name: str) -> Tuple[torch.Tensor, int]:
        """
        Convert a name to a tensor of character indices.
        
        Format: <start> name <end> <pad> <pad> ...
        """
        # Add start and end tokens
        name_length = len(name)
        name_with_tokens = ['<start>'] + list(name) + ['<end>']
        
        # Convert to indices
        indices = [self.char_to_idx[char] for char in name_with_tokens]
        
        # Pad to max_length
        indices.extend([self.char_to_idx['<pad>']] * (self.max_length - len(indices)))
        
        return torch.tensor(indices, dtype=torch.long), name_length
    
    def __len__(self) -> int:
        return len(self.names)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training example.
        
        Returns:
            input_seq: Input sequence (including start token, excluding end token)
            target_seq: Target sequence (excluding start token, including end token)
        """
        full_seq, name_length = self.processed_names[idx]
        
        # Find the end token position
        end_pos = name_length + 1
        
        # Input: <start> name (exclude end token, keep padding)
        input_seq = full_seq.clone()
        input_seq[end_pos] = self.char_to_idx['<pad>']  # Replace end token with pad
        input_seq = input_seq[:-1]  # Remove last position to make room for target
        
        # Target: name <end> <pad> ...
        target_seq = full_seq[1:]  # Shift by one position
        
        return input_seq, target_seq
    
    def decode(self, indices: List[int]) -> str:
        """Convert a list of indices back to a string."""
        chars = [self.idx_to_char[idx] for idx in indices]
        # Remove special tokens and padding
        chars = [char for char in chars if char not in ['<pad>', '<start>', '<end>']]
        return ''.join(chars)


def create_dataloader(csv_path: str, max_length: int = 30, 
                      train_split: float = 0.8, val_split: float = 0.1,
                      batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create train, validation, and test data loaders for dinosaur names.
    
    Args:
        csv_path: Path to the CSV file containing dinosaur names
        max_length: Maximum sequence length
        train_split: Fraction of data to use for training
        val_split: Fraction of data to use for validation
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader  
        test_loader: Test data loader
        vocab_info: Vocabulary information dictionary
    """
    # Create full dataset
    full_dataset = DinoDataset(csv_path, max_length)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Data split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return full_dataset, train_loader, val_loader, test_loader
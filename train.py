import torch
import torch.nn as nn
import yaml
import argparse
from data_loader import create_dataloader
from transformer import create_model
from trainer import Trainer


def parse_arguments():
    """Parse command line arguments for training.

    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train dinosaur name generator")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    return parser.parse_args()


def main():
    """Train dinosaur name generator model."""
    # Parse command line arguments
    args = parse_arguments()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Extract configs
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    hardware_config = config['hardware']

    print("Setting up data loaders...")
    full_dataset, train_loader, val_loader, test_loader = create_dataloader(
        csv_path=data_config['csv_path'],
        max_length=data_config['max_length'],
        train_split=data_config['train_split'],
        val_split=data_config['val_split'],
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers']
    )

    # Model setup
    model_config['pad_idx'] = full_dataset.char_to_idx['<pad>']
    model_config['vocab_size'] = full_dataset.vocab_size
    model_config['max_len'] = data_config['max_length']

    print(f"Vocabulary size: {model_config['vocab_size']}")

    # Create model, criterion, optimizer, and scheduler using create_model function
    model, criterion, optimizer, scheduler = create_model(
        vocab_size=model_config['vocab_size'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        n_layers=model_config['n_layers'],
        d_ff=model_config['d_ff'],
        max_len=model_config['max_len'],
        dropout=model_config['dropout'],
        pad_idx=model_config['pad_idx'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        num_epochs=training_config['num_epochs']
    )

    save_dir = hardware_config['save_dir']

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=save_dir
    )

    # Run training
    results = trainer.run(
        num_epochs=training_config['num_epochs'],
        sample_interval=training_config['sample_interval'],
        dataset=full_dataset
    )

    # Save results
    torch.save({
        'config': config,
        'results': results,
    }, f"{save_dir}/results.pt")


if __name__ == "__main__":
    main()
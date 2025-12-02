import torch
import yaml
import argparse
import os
from data_loader import DinoDataset
from transformer import DecoderOnlyTransformer


def parse_arguments():
    """Parse command line arguments for inference.

    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Generate dinosaur names using trained model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    return parser.parse_args()


def load_model(checkpoint_path: str, config: dict, device: str):
    """Load the trained model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint
        config: Configuration dictionary
        device: Device to load the model on

    Returns:
        model: Loaded model
        dataset: Dataset with vocabulary information
    """
    # Create dataset to get vocabulary information
    dataset = DinoDataset(
        csv_path=config['data']['csv_path'],
        max_length=config['data']['max_length']
    )

    # Create model with same architecture as training
    model = DecoderOnlyTransformer(
        vocab_size=dataset.vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        max_len=config['data']['max_length'],
        dropout=config['model']['dropout'],
        pad_idx=dataset.char_to_idx['<pad>']
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss {checkpoint['loss']:.4f}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model, dataset


def generate_names(model, dataset, num_samples: int, max_length: int, temperature: float):
    """Generate dinosaur names using the trained model.

    Args:
        model: Trained model
        dataset: Dataset with vocabulary information
        num_samples: Number of names to generate
        max_length: Maximum length of generated names
        temperature: Sampling temperature

    Returns:
        List of generated dinosaur names
    """
    generated_names = []

    with torch.no_grad():
        print(f"\nGenerating {num_samples} dinosaur names...")

        for i in range(num_samples):
            # Generate sequence
            generated = model.generate(max_length=max_length, temperature=temperature)

            # Convert to string
            name = dataset.decode(generated.tolist())
            generated_names.append(name)

            print(f"{i+1:2d}: {name}")

    return generated_names


def save_names(names: list, output_file: str):
    """Save generated names to a file.

    Args:
        names: List of generated names
        output_file: Output file path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for name in names:
            f.write(f"{name}\n")

    print(f"\nGenerated names saved to {output_file}")


def main():
    """Main inference function."""
    # Parse command line arguments
    args = parse_arguments()

    # Load configuration
    print(f"Loading configuration from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Extract inference configs
    inference_config = config['inference']
    checkpoint_path = inference_config['checkpoint_path']
    num_samples = inference_config['num_samples']
    max_length = inference_config['max_length']
    temperature = inference_config['temperature']
    output_file = inference_config.get('output_file')

    # Set random seed for reproducibility
    seed = config['experiment']['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running inference on device: {device}")

    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model, dataset = load_model(checkpoint_path, config, device)

    # Generate names
    generated_names = generate_names(
        model=model,
        dataset=dataset,
        num_samples=num_samples,
        max_length=max_length,
        temperature=temperature
    )

    # Save to file if requested
    if output_file:
        save_names(generated_names, output_file)

    print(f"\nInference completed! Generated {len(generated_names)} dinosaur names.")


if __name__ == "__main__":
    main()
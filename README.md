# Dino-Gen

A transformer-based neural network for generating creative dinosaur names. The project implements a character-level language model trained on existing dinosaur names to generate new, plausible dinosaur names.

## Quick Start

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the model:**
   ```bash
   python train.py
   ```

3. **Generate dinosaur names:**
   ```bash
   python inference.py
   ```

The model configuration is managed through `config.yaml`, where you can adjust training parameters, model architecture, and data paths.
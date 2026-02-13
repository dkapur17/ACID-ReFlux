"""Print model size given a config file."""

import argparse
import yaml

from src.models import create_model_from_config


def main():
    parser = argparse.ArgumentParser(description='Print model size')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., configs/ddpm.yaml)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    model = create_model_from_config(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters:     {total_params:>12,} ({total_params / 1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:>12,} ({trainable_params / 1e6:.2f}M)")

    # Per-component breakdown
    print("\nPer-component breakdown:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        if params > 0:
            print(f"  {name:20s} {params:>12,} ({params / 1e6:.2f}M)")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Main training script
Usage: python scripts/run_training.py --config configs/llava_config.yaml
"""

import argparse
import sys
sys.path.append('.')

from src.training.trainer import VLMTrainer
from src.training.utils import set_seed, print_device_info

def main():
    parser = argparse.ArgumentParser(description='Train VLM model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    
    # Print device info
    print_device_info()
    
    # Create trainer
    trainer = VLMTrainer(args.config)
    trainer.setup()
    
    # Set seed
    set_seed(trainer.config['data']['seed'])
    
    # Run training
    if not args.eval_only:
        if args.resume:
            print(f"Resuming from {args.resume}")
        trainer.train()
    
    # Run evaluation
    results = trainer.evaluate()
    
    # Save model
    if not args.eval_only:
        save_path = trainer.config['training']['output_dir'] + '/final_model'
        trainer.save(save_path)

if __name__ == '__main__':
    main()
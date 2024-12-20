import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
from datetime import datetime

# Import from your existing files
from model import HarmonicUNet, CombinedLoss
from trainer import train_model, MedicalImageDataset, get_transform

def parse_args():
    parser = argparse.ArgumentParser(description='Training Harmonic U-Net for Microhemorrhage Detection')
    
    # Dataset paths
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val_dir', type=str, required=True, 
                        help='Path to validation data directory')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    # Model settings
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained weights (optional)')
    parser.add_argument('--out_dir', type=str, default='outputs',
                        help='Directory to save model outputs')
    
    # Hardware settings
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    return parser.parse_args()

def main():
    # Get command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(args.out_dir, f'training_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Log training settings
    logging.info('Starting training with settings:')
    for arg, value in vars(args).items():
        logging.info(f'{arg}: {value}')
    
    # Prepare datasets
    train_dataset = MedicalImageDataset(
        root_dir=args.train_dir,
        transform=get_transform(train=True)
    )
    
    val_dataset = MedicalImageDataset(
        root_dir=args.val_dir,
        transform=get_transform(train=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = HarmonicUNet(n_channels=1, n_classes=1)
    
    # Load pretrained weights if specified
    if args.pretrained:
        logging.info(f'Loading pretrained weights from {args.pretrained}')
        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    
    # Initialize loss and optimizer
    criterion = CombinedLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=args.epochs,
        output_dir=args.out_dir
    )

if __name__ == '__main__':
    main()
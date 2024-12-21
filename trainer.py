import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pydicom
from PIL import Image
import glob
from tqdm.auto import tqdm
import torch.optim as optim
from pathlib import Path

import logging
from datetime import datetime

# Configure logging
def get_dataset_logger(verbose=False):
    """Get or create logger for dataset processing

    Args:
        verbose (bool): If True, enables logging, if False completely disables logging
    """
    logger = logging.getLogger('dataset_processing')

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    if not verbose:
        # Disable all logging when verbose is False
        logger.addHandler(logging.NullHandler())
        logger.setLevel(logging.CRITICAL + 1)  # Set to higher than CRITICAL to disable all logging
        return logger

    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    # Create handlers for verbose mode
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'dataset_processing_{timestamp}.log')
    )
    console_handler = logging.StreamHandler()

    # Set handler levels for verbose mode
    logger.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)

    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    file_handler.setFormatter(logging.Formatter(log_format))
    console_handler.setFormatter(logging.Formatter(log_format))

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Initialize the logger with default settings (no logging)
dataset_logger = get_dataset_logger(verbose=False)

def normalize_dicom(dicom_data):
    """Normalize DICOM image data to [0, 1] range"""
    img_array = dicom_data.pixel_array.astype(float)

    if dataset_logger.isEnabledFor(logging.DEBUG):
        dataset_logger.debug(f"DICOM Info - Shape: {img_array.shape}, "
                           f"Min: {img_array.min():.2f}, Max: {img_array.max():.2f}")

    # Normalize based on window center and width if available
    if hasattr(dicom_data, 'WindowCenter') and hasattr(dicom_data, 'WindowWidth'):
        center = dicom_data.WindowCenter
        width = dicom_data.WindowWidth
        if isinstance(center, pydicom.multival.MultiValue):
            center = center[0]
        if isinstance(width, pydicom.multival.MultiValue):
            width = width[0]

        img_min = center - (width / 2)
        img_max = center + (width / 2)
        img_array = np.clip(img_array, img_min, img_max)

        if dataset_logger.isEnabledFor(logging.DEBUG):
            dataset_logger.debug(f"Window Settings - Center: {center}, Width: {width}")

    # Normalize to [0, 1]
    img_min = img_array.min()
    img_max = img_array.max()
    if img_max != img_min:
        img_array = (img_array - img_min) / (img_max - img_min)

    if dataset_logger.isEnabledFor(logging.DEBUG):
        dataset_logger.debug(f"Normalized - Min: {img_array.min():.2f}, Max: {img_array.max():.2f}")
    return img_array

def get_transform(train=True):
    """
    Get transformation pipeline for the medical images
    """
    transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]

    if train:
        transforms_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0))
        ])

    return transforms.Compose(transforms_list)

class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, mask_dir=None, transform=None, train=True, verbose=False, is_img=False):
        """
        Args:
            data_dir (str): Directory with DICOM images
            mask_dir (str, optional): Directory with mask images
            transform: Optional transform to be applied on images
            train (bool): Whether this is training set or not
            verbose (bool): If True enables all logging, if False disables all logging
        """
        global dataset_logger
        dataset_logger = get_dataset_logger(verbose)

        self.data_dir = Path(data_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.transform = transform
        self.train = train
        self.verbose = verbose
        self.is_img = is_img

        if self.is_img:
          self.data_dir = self.data_dir / 'series'
          self.mask_dir = Path(data_dir) / 'mask'

        # Get all DICOM files
        if self.is_img:
            self.dicom_files = sorted(glob.glob(str(self.data_dir / "*.jpg"), recursive=True))
            if dataset_logger.isEnabledFor(logging.INFO):
                dataset_logger.info(f"Found {len(self.dicom_files)} images in {self.data_dir}")
        else:
          self.dicom_files = sorted(glob.glob(str(self.data_dir / "**/*.dcm"), recursive=True))
          if dataset_logger.isEnabledFor(logging.INFO):
              dataset_logger.info(f"Found {len(self.dicom_files)} DICOM files in {self.data_dir}")

        if self.mask_dir:
            # Get all mask files (assuming .png format)
            if self.is_img:
                self.mask_files = sorted(glob.glob(str(self.mask_dir / "*.png"), recursive=True))
                if dataset_logger.isEnabledFor(logging.INFO):
                    dataset_logger.info(f"Found {len(self.mask_files)} mask files in {self.mask_dir}")
            else:
              self.mask_files = sorted(glob.glob(str(self.mask_dir / "**/*.png"), recursive=True))
              if dataset_logger.isEnabledFor(logging.INFO):
                  dataset_logger.info(f"Found {len(self.mask_files)} mask files in {self.mask_dir}")

            assert len(self.dicom_files) == len(self.mask_files), \
                f"Number of images ({len(self.dicom_files)}) and masks ({len(self.mask_files)}) don't match"

    def __len__(self):
        return len(self.dicom_files)

    def __getitem__(self, idx):
        # Load DICOM image
        dicom_path = self.dicom_files[idx]
        if dataset_logger.isEnabledFor(logging.INFO):
            dataset_logger.info(f"Processing DICOM/image file: {dicom_path}")

        try:
            if self.is_img:
                # image = Image.open(dicom_path)
                image = Image.open(dicom_path).convert('L')
                image = image.resize((256,256))
                image = np.asarray(image)
                # print("processing image: " + dicom_path)
                # print(image)
                image = Image.fromarray((image * 255).astype(np.uint8))
            else:
                # print("processing dicom: " + dicom_path)
                dicom_data = pydicom.dcmread(dicom_path)
                if dataset_logger.isEnabledFor(logging.DEBUG):
                    dataset_logger.debug(f"DICOM metadata - "
                                      f"StudyID: {getattr(dicom_data, 'StudyID', 'N/A')}, "
                                      f"SeriesNumber: {getattr(dicom_data, 'SeriesNumber', 'N/A')}, "
                                      f"InstanceNumber: {getattr(dicom_data, 'InstanceNumber', 'N/A')}")

                # Convert DICOM to normalized array
                image = normalize_dicom(dicom_data)

                # Convert to PIL Image for transformation pipeline
                image = Image.fromarray((image * 255).astype(np.uint8))

            if self.transform:
                image = self.transform(image)
                if dataset_logger.isEnabledFor(logging.DEBUG):
                    dataset_logger.debug(f"Applied transforms - Final tensor shape: {image.shape}")

            if self.mask_dir:
                # Load and process mask
                mask_path = self.mask_files[idx]
                if dataset_logger.isEnabledFor(logging.DEBUG):
                    dataset_logger.debug(f"Loading mask file: {mask_path}")
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize((256,256))
                mask = transforms.ToTensor()(mask)
                if dataset_logger.isEnabledFor(logging.DEBUG):
                    dataset_logger.debug(f"Mask tensor shape: {mask.shape}")
                return image, mask
            else:
                # If no mask directory is provided, return a dummy mask of zeros
                dummy_mask = torch.zeros_like(image)
                print("No mask directory provided for "+dicom_path+". Returning dummy mask.")
                return image, dummy_mask

        except Exception as e:
            if dataset_logger.isEnabledFor(logging.ERROR):
                dataset_logger.error(f"Error processing file {dicom_path}: {str(e)}")
            raise

def create_dataloaders(data_dir, mask_dir=None, batch_size=8, num_workers=4, train_split=0.8, verbose=False, is_img=False):
    """
    Create train and validation dataloaders

    Args:
        data_dir (str): Directory containing DICOM images
        mask_dir (str, optional): Directory containing mask images
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of workers for dataloaders
        train_split (float): Proportion of data to use for training
        verbose (bool): If True enables all logging, if False disables all logging
        is_img: if you have the image already

    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    global dataset_logger
    dataset_logger = get_dataset_logger(verbose)

    if dataset_logger.isEnabledFor(logging.INFO):
        dataset_logger.info(f"Creating dataloaders with:"
                          f"\n\tData directory: {data_dir}"
                          f"\n\tMask directory: {mask_dir}"
                          f"\n\tBatch size: {batch_size}"
                          f"\n\tNum workers: {num_workers}"
                          f"\n\tTrain split: {train_split}")

    # Create datasets
    dataset = MedicalImageDataset(
        data_dir=data_dir,
        mask_dir=mask_dir,
        transform=get_transform(train=True),
        verbose=verbose,
        is_img=is_img
    )

    # Split into train and validation
    dataset_size = len(dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size

    if dataset_logger.isEnabledFor(logging.INFO):
        dataset_logger.info(f"Dataset split:"
                          f"\n\tTotal size: {dataset_size}"
                          f"\n\tTrain size: {train_size}"
                          f"\n\tVal size: {val_size}")

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # Create dataloaders with proper cleanup handling
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if num_workers > 0 else 0,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context='spawn' if num_workers > 0 else None
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers > 0 else 0,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
        multiprocessing_context='spawn' if num_workers > 0 else None
    )

    if dataset_logger.isEnabledFor(logging.INFO):
        dataset_logger.info("Dataloaders created successfully")
    return train_loader, val_loader



def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=100,
    patience=10,
    checkpoint_dir='checkpoints'
):
    """
    Train the model with early stopping and checkpointing

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Maximum number of epochs to train
        patience: Number of epochs to wait for improvement before early stopping
        checkpoint_dir: Directory to save model checkpoints
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize tracking variables
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=patience,
        factor=0.5,
        verbose=True
    )

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, (images, masks) in enumerate(progress_bar):
            images = images.to(device)
            masks = masks.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({'train_loss': epoch_loss / (batch_idx + 1)})

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        dice_scores = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # Calculate Dice score
                pred = torch.sigmoid(outputs) > 0.5
                dice = calculate_dice_score(pred, masks)
                dice_scores.append(dice)

        avg_val_loss = val_loss / len(val_loader)
        avg_dice_score = sum(dice_scores) / len(dice_scores)
        val_losses.append(avg_val_loss)

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Print epoch statistics
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Average Train Loss: {avg_train_loss:.4f}')
        print(f'Average Val Loss: {avg_val_loss:.4f}')
        print(f'Average Dice Score: {avg_dice_score:.4f}')

        # Save checkpoint if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'dice_score': avg_dice_score
            }

            torch.save(
                checkpoint,
                os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
            )

            print(f'Checkpoint saved! Best validation loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1

        # Early stopping check
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    return train_losses, val_losses

def calculate_dice_score(pred, target):
    """Calculate Dice score between predicted and target masks"""
    smooth = 1e-5
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def load_pretrained_encoder(model):
    """Load pretrained weights from ResNet into the encoder path"""
    resnet = models.resnet34(pretrained=True)

    # Map ResNet layers to U-Net encoder
    model.inc.double_conv[0].weight.data = resnet.conv1.weight.data
    model.inc.double_conv[1].weight.data = resnet.bn1.weight.data
    model.inc.double_conv[1].bias.data = resnet.bn1.bias.data

    return model

def load_and_train_model(
    image_dir,
    mask_dir,
    batch_size=8,
    num_epochs=100,
    learning_rate=1e-4,
    is_img=False
):
    # Initialize model and training components
    model = HarmonicUNet(n_channels=1, n_classes=1)
    model = load_pretrained_encoder(model)
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare datasets
    train_transform = get_transform(train=True)
    val_transform = get_transform(train=False)

    # Split dataset into train and validation
    all_images = sorted(os.listdir(image_dir))
    train_size = int(0.8 * len(all_images))
    train_images = all_images[:train_size]
    val_images = all_images[train_size:]

    # Create datasets
    train_dataset = MedicalImageDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=train_transform,
        is_img = is_img
    )

    val_dataset = MedicalImageDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=val_transform,
        is_img = is_img
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Train model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs
    )

    return model, train_losses, val_losses
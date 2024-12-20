import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os


def load_pretrained_encoder(model):
    """
    Load pretrained weights from ResNet into the encoder path
    """
    resnet = models.resnet34(pretrained=True)
    
    # Map ResNet layers to U-Net encoder
    model.inc.double_conv[0].weight.data = resnet.conv1.weight.data
    model.inc.double_conv[1].weight.data = resnet.bn1.weight.data
    model.inc.double_conv[1].bias.data = resnet.bn1.bias.data
    
    # You can continue mapping other layers as needed
    
    return model

# Example usage:
# Data transforms for augmentation
def get_transform(train=True):
    transforms_list = [
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])  # Adjusted for medical images
    ]
    
    if train:
        transforms_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0))
        ])
    
    return transforms.Compose(transforms_list)

class MedicalImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        # Load image and mask (assuming they're in a standard format like PNG)
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)
        
        return image, mask

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=5, 
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
        print(f'Epoch {epoch+1}/{num_epochs}:')
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
    smooth = 1e-5
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

def load_and_train_model(
    image_dir,
    mask_dir,
    batch_size=8,
    num_epochs=100,
    learning_rate=1e-4
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
        transform=train_transform
    )
    
    val_dataset = MedicalImageDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=val_transform
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
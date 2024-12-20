import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handling input sizes
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                       diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class HarmonicEmbedding(nn.Module):
    def __init__(self, embedding_size, max_frequencies=10):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_frequencies = max_frequencies
        
        # Learnable frequency components
        self.frequencies = nn.Parameter(
            torch.randn(max_frequencies, 2) * 0.1
        )
        
        # Phase shifts
        self.phase_shifts = nn.Parameter(
            torch.randn(max_frequencies) * 0.1
        )
        
        # Amplitude weights
        self.amplitudes = nn.Parameter(
            torch.ones(max_frequencies)
        )
        
    def forward(self, x):
        batch_size, _, H, W = x.shape
        
        # Generate coordinate grid
        pos_x = torch.linspace(-1, 1, W, device=x.device)
        pos_y = torch.linspace(-1, 1, H, device=x.device)
        grid_y, grid_x = torch.meshgrid(pos_y, pos_x, indexing='ij')
        
        # Compute harmonic functions
        harmonics = []
        for i in range(self.max_frequencies):
            # Compute 2D harmonic function
            freq_x, freq_y = self.frequencies[i]
            phase = self.phase_shifts[i]
            amplitude = self.amplitudes[i]
            
            harm = amplitude * torch.sin(
                2 * torch.pi * (freq_x * grid_x + freq_y * grid_y) + phase
            )
            harmonics.append(harm)
        
        # Stack and reshape harmonics
        harmonic_features = torch.stack(harmonics, dim=0)
        harmonic_features = harmonic_features.expand(batch_size, -1, -1, -1)
        return harmonic_features

class HarmonicAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, harmonic_features):
        batch_size, C, H, W = x.size()
        
        # Compute Query, Key, Value
        query = self.query(x).view(batch_size, -1, H * W)
        key = self.key(harmonic_features).view(batch_size, -1, H * W)
        value = self.value(harmonic_features).view(batch_size, -1, H * W)
        
        # Attention map
        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = F.softmax(attention, dim=-1)
        
        # Attend to values
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return x + self.gamma * out
    
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce

class HarmonicUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Harmonic components
        self.harmonic_embed = HarmonicEmbedding(embedding_size=256)
        self.harmonic_attention = HarmonicAttention(1024 // factor)
        
        # Fusion layer
        self.fusion = nn.Conv2d(1024 // factor + 10, 1024 // factor, 1)
        
        # Decoder
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Harmonic embedding
        h = self.harmonic_embed(x)
        
        # Combine CNN and harmonic features
        x5 = self.harmonic_attention(x5, h)
        x5 = self.fusion(torch.cat([x5, h], dim=1))
        
        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits
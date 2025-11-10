import torch
import torch.nn as nn
import torch.nn.functional as F


def flatten_pose(x):  # (B,3,13,1)->(B,39)
    return x.view(x.size(0), -1)


def build_mask(x, fill=0.0):  # 1=observed, 0=masked
    return (x != fill).float().view(x.size(0), -1)


class TemporalDAE(nn.Module):
    """
    Temporal Denoising Autoencoder that uses a sliding window of frames
    to predict the center frame's keypoints.
    
    Args:
        d: dimension of single pose (default 39 for 3*13 keypoints)
        h: hidden dimension
        window_size: temporal window size (w). Uses frames [t-w, ..., t, ..., t+w]
                     Total frames = 2*w + 1
        temporal_model: 'transformer', 'lstm', or 'conv1d'
    """
    def __init__(self, d=39, h=128, window_size=2, temporal_model='transformer'):
        super().__init__()
        self.d = d
        self.window_size = window_size
        self.n_frames = 2 * window_size + 1  # e.g., w=2 -> 5 frames total
        self.temporal_model = temporal_model
        
        # Input: concatenate pose and mask for each frame
        input_dim = d * 2  # pose + mask
        
        if temporal_model == 'transformer':
            # Transformer encoder for temporal modeling
            self.frame_embedding = nn.Linear(input_dim, h)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=h, 
                nhead=4, 
                dim_feedforward=h*2,
                dropout=0.1,
                batch_first=True
            )
            self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
            
            # Output only the center frame reconstruction
            self.decoder = nn.Sequential(
                nn.Linear(h, h),
                nn.GELU(),
                nn.Linear(h, d)
            )
            
        elif temporal_model == 'lstm':
            # Bi-directional LSTM for temporal modeling
            self.frame_embedding = nn.Linear(input_dim, h)
            self.lstm = nn.LSTM(h, h//2, num_layers=2, bidirectional=True, batch_first=True)
            
            self.decoder = nn.Sequential(
                nn.Linear(h, h),
                nn.GELU(),
                nn.Linear(h, d)
            )
            
        elif temporal_model == 'conv1d':
            # 1D convolution over time
            self.frame_embedding = nn.Linear(input_dim, h)
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(h, h, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(h, h, kernel_size=3, padding=1),
                nn.GELU(),
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(h, h),
                nn.GELU(),
                nn.Linear(h, d)
            )
        
        else:
            raise ValueError(f"Unknown temporal_model: {temporal_model}")

    def forward(self, x_window, m_window):
        """
        x_window: (B, n_frames, d) - windowed poses (already flattened)
        m_window: (B, n_frames, d) - windowed masks
        
        Returns:
        - recon: (B, d) - reconstructed center frame
        - residual: (B, d) - predicted residual for center frame
        """
        B, n_frames, d = x_window.shape
        assert n_frames == self.n_frames, f"Expected {self.n_frames} frames, got {n_frames}"
        
        # Concatenate pose and mask for each frame
        frame_features = torch.cat([x_window, m_window], dim=-1)  # (B, n_frames, d*2)
        
        # Embed each frame
        embedded = self.frame_embedding(frame_features)  # (B, n_frames, h)
        
        if self.temporal_model == 'transformer':
            # Transformer encoder
            temporal_features = self.temporal_encoder(embedded)  # (B, n_frames, h)
            # Extract center frame
            center_idx = self.window_size
            center_features = temporal_features[:, center_idx, :]  # (B, h)
            
        elif self.temporal_model == 'lstm':
            # LSTM
            temporal_features, _ = self.lstm(embedded)  # (B, n_frames, h)
            # Extract center frame
            center_idx = self.window_size
            center_features = temporal_features[:, center_idx, :]  # (B, h)
            
        elif self.temporal_model == 'conv1d':
            # Conv1d expects (B, C, T)
            embedded_t = embedded.transpose(1, 2)  # (B, h, n_frames)
            temporal_features = self.temporal_conv(embedded_t)  # (B, h, n_frames)
            # Extract center frame
            center_idx = self.window_size
            center_features = temporal_features[:, :, center_idx]  # (B, h)
        
        # Decode to residual
        residual = self.decoder(center_features)  # (B, d)
        
        # Blend: observed keypoints pass through, masked get reconstructed
        center_x = x_window[:, self.window_size, :]  # (B, d) - center frame pose
        center_m = m_window[:, self.window_size, :]  # (B, d) - center frame mask
        
        recon = center_m * center_x + (1 - center_m) * (center_x + residual)
        
        return recon, residual


def loss_masked(pred, target, m_flat, lam: float = 0.05):
    """
    pred,target: (B,39)
    m_flat:      (B,39)  1=observed, 0=masked
    """
    se = (pred - target) ** 2
    masked = 1 - m_flat
    unmasked = m_flat

    # normalize per sample by count of masked/unmasked, then average batch
    masked_cnt = masked.sum(dim=1).clamp(min=1.0)
    unmasked_cnt = unmasked.sum(dim=1).clamp(min=1.0)

    masked_loss = ((se * masked).sum(dim=1) / masked_cnt).mean()
    unmasked_loss = ((se * unmasked).sum(dim=1) / unmasked_cnt).mean()

    return masked_loss + lam * unmasked_loss


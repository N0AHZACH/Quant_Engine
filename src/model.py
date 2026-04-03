import torch
import torch.nn as nn
import torch.nn.functional as F

# --- GLOBAL CONFIGURATION ---
SEQ_LEN = 60
FEATURES = 20 

# --- 1. ATTENTION-BASED RNN MODEL (LSTM/GRU) ---

class AttentionModel(nn.Module):
    """
    A Sequence Model (LSTM or GRU) combined with an Attention layer.
    This architecture learns which part of the 60-day history is most predictive.
    """
    def __init__(self, rnn_type="lstm", input_dim=FEATURES, hidden_size=64, num_layers=2):
        super(AttentionModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # RNN Layer (Accepts 20 features across 60 days)
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True, dropout=0.1)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_size, 
                              num_layers=num_layers, batch_first=True, dropout=0.1)
        else:
            raise ValueError("RNN type must be 'lstm' or 'gru'")
            
        # Attention Layer Components
        # We use the final hidden state to generate attention weights over all 60 steps
        self.W_a = nn.Linear(hidden_size, hidden_size)
        self.v_a = nn.Linear(hidden_size, 1)

        # Output Layers
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, SEQ_LEN, FEATURES) -> (B, 60, 20)
        
        # 1. RNN Pass
        # r_out shape: (B, 60, H)
        r_out, _ = self.rnn(x)
        
        # 2. Attention Mechanism
        # Score shape: (B, 60, 1)
        attn_scores = self.v_a(torch.tanh(self.W_a(r_out)))
        
        # Weights shape: (B, 60, 1) -> sums to 1 across the 60 steps
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # Context Vector: weighted sum of the RNN outputs
        # context shape: (B, H)
        context = torch.sum(attn_weights * r_out, dim=1)
        
        # 3. Final Prediction Layers
        out = F.relu(self.fc1(context))
        out = self.fc2(out)
        
        # out shape: (B, 1)
        return out

# --- 2. CNN RESNET MODEL ---

class ResidualBlock(nn.Module):
    """ Standard ResNet block for CNN model. """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=(kernel_size - 1) // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding=(kernel_size - 1) // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection: must match dimensions if in_channels != out_channels
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity # Residual connection
        out = F.relu(out)
        return out

class CNNModel(nn.Module):
    """
    A 1D Convolutional Neural Network with Residual Blocks (ResNet).
    This architecture learns local patterns across the 60-day history.
    """
    def __init__(self, input_dim=FEATURES):
        super(CNNModel, self).__init__()
        
        # Initial Layer: Convert (B, 60, 20) to (B, 32, 60) for Conv1D
        self.initial_conv = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=5, padding=2)
        
        # Residual Blocks
        self.res_block1 = ResidualBlock(32, 32, kernel_size=5)
        self.res_block2 = ResidualBlock(32, 64, kernel_size=5) # Increase channels to 64
        
        # Global Average Pooling (Summarizes all 60 timesteps)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output Layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (B, 60, 20). Conv1D requires (B, Channels, Length)
        x = x.transpose(1, 2) # Transpose to (B, 20, 60)
        
        # 1. Initial Conv
        out = F.relu(self.initial_conv(x))
        
        # 2. Residual Blocks
        out = self.res_block1(out)
        out = self.res_block2(out)
        
        # 3. Global Average Pooling
        out = self.avg_pool(out).squeeze(-1) # Output shape (B, 64)
        
        # 4. Final Prediction Layers
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        
        # out shape: (B, 1)
        return out
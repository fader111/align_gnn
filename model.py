import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class TeethGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TeethGNN, self).__init__()
        # Graph convolution layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        
    def forward(self, x, edge_index):
        # Apply the first graph convolution layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Apply the second graph convolution layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Apply the third graph convolution layer
        x = self.conv3(x, edge_index)
        return x

if __name__ == "__main__":
    # Example use case
    in_channels = 7  # Input: Initial positions (3 for translation + 4 for quaternion)
    hidden_channels = 64  # Hidden layer size
    out_channels = 7  # Output: Predicted corrected positions (3 for translation + 4 for quaternion)

    model = TeethGNN(in_channels, hidden_channels, out_channels)

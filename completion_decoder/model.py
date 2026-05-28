import torch
import torch.nn as nn
import torch.nn.functional as F


class Pct(nn.Module):
    def __init__(self, num_missing_points=128, num_attention_layers=4):
        super(Pct, self).__init__()
        self.num_missing_points = num_missing_points

        self.embedding = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.attention_layers = nn.ModuleList([PointTransformerBlock() for _ in range(num_attention_layers)])
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(512, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_missing_points * 3),
        )

    def forward(self, x):
        features = self.embedding(x)

        attended = features
        for attention_layer in self.attention_layers:
            attended = attention_layer(attended)

        fused = torch.cat([features, attended], dim=1)
        fused = self.conv_fuse(fused)
        global_feature = F.adaptive_max_pool1d(fused, 1).squeeze(-1)

        points = self.decoder(global_feature)
        return points.view(-1, self.num_missing_points, 3)


class PointTransformerBlock(nn.Module):
    def __init__(self, channels=256):
        super(PointTransformerBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.sa = SelfAttentionLayer(channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return self.sa(x)


class SelfAttentionLayer(nn.Module):
    def __init__(self, channels):
        super(SelfAttentionLayer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels, 1, bias=False)
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        x_r = torch.bmm(x_v, attention)
        x_r = F.relu(self.after_norm(self.trans_conv(x - x_r)))
        return x + x_r

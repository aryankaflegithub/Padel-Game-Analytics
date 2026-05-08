import torch
import torch.nn as nn


EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]
NUM_JOINTS = 17


def build_adj():
    A = torch.zeros(NUM_JOINTS, NUM_JOINTS)
    for i, j in EDGES:
        A[i, j] = 1
        A[j, i] = 1
    A += torch.eye(NUM_JOINTS)
    D_inv = torch.diag(1.0 / A.sum(1).clamp(min=1e-6))
    return D_inv @ A


class GCN(nn.Module):
    def __init__(self, cin, cout, A):
        super().__init__()
        self.register_buffer("A", A)
        self.conv = nn.Conv2d(cin, cout, kernel_size=1)
        self.bn   = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T, V)
        # graph conv over V dimension
        x = torch.einsum("bctv,vw->bctw", x, self.A)
        x = self.conv(x)
        return self.relu(self.bn(x))


class TCN(nn.Module):
    def __init__(self, channels, kernel=9):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=(kernel, 1), padding=(kernel // 2, 0))
        self.bn   = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T, V) — conv over T dimension only
        return self.relu(self.bn(self.conv(x)))


class STGCNBlock(nn.Module):
    def __init__(self, cin, cout, A):
        super().__init__()
        self.gcn = GCN(cin, cout, A)
        self.tcn = TCN(cout)
        self.residual = nn.Sequential(
            nn.Conv2d(cin, cout, 1),
            nn.BatchNorm2d(cout)
        ) if cin != cout else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class STGCN(nn.Module):
    def __init__(self, num_classes=3, in_channels=2, num_joints=NUM_JOINTS):
        super().__init__()
        A = build_adj()

        self.bn_in = nn.BatchNorm2d(in_channels)

        self.layers = nn.Sequential(
            STGCNBlock(in_channels, 64,  A),
            STGCNBlock(64,          64,  A),
            STGCNBlock(64,          128, A),
            STGCNBlock(128,         128, A),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, C, T, V)
        x = self.bn_in(x)
        x = self.layers(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
    
    
import numpy as np
import torch
import torch.nn as nn


SHOT_CLASSES  = ["forehand", "backhand", "smash"]
WINDOW_FRAMES = 30
NUM_JOINTS    = 17


class ShotMLP(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class ShotClassifier:
    def __init__(self, weights_path=None, device=0):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model  = None
        self.mean   = None
        self.std    = None
        self.kp_buffer = {}

        if weights_path:
            ck = torch.load(weights_path, map_location=self.device, weights_only=False)
            in_dim     = ck["in_dim"]
            self.model = ShotMLP(in_dim, len(SHOT_CLASSES)).to(self.device)
            self.model.load_state_dict(ck["model_state_dict"])
            self.model.eval()
            self.mean  = ck["mean"]
            self.std   = ck["std"]
            print(f"shot classifier loaded from {weights_path}")
        else:
            print("shot classifier: no weights — will output unknown")

    def update_keypoints(self, players):
        for p in players:
            pid = p["id"]
            kps = p.get("keypoints")
            if kps is None or len(kps) < NUM_JOINTS:
                kps = np.zeros((NUM_JOINTS, 2), dtype=np.float32)
            if pid not in self.kp_buffer:
                self.kp_buffer[pid] = []
            self.kp_buffer[pid].append(kps.astype(np.float32))
            if len(self.kp_buffer[pid]) > WINDOW_FRAMES:
                self.kp_buffer[pid].pop(0)

    def classify(self, player_id):
        if self.model is None:
            return "unknown", 0.0

        buf = self.kp_buffer.get(player_id, [])
        if not buf:
            return "unknown", 0.0

        seq = buf[-WINDOW_FRAMES:]
        while len(seq) < WINDOW_FRAMES:
            seq = [seq[0]] + seq

        arr    = np.stack(seq, axis=0).reshape(-1).astype(np.float32)
        arr    = (arr - self.mean) / self.std
        x      = torch.tensor(arr).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)[0]

        idx  = int(probs.argmax())
        conf = float(probs[idx])
        return SHOT_CLASSES[idx], conf
    
    
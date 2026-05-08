import numpy as np
import torch
from core.stgcn import STGCN

SHOT_CLASSES  = ["forehand", "backhand", "smash"]
WINDOW_FRAMES = 30   # frames before shot event to classify
NUM_JOINTS    = 17


class ShotClassifier:
    def __init__(self, weights_path=None, device=0):
        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model  = STGCN(num_classes=len(SHOT_CLASSES)).to(self.device)
        self.model.eval()

        if weights_path:
            ck = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(ck["model_state_dict"])
            print(f"ST-GCN loaded from {weights_path}")
        else:
            print("ST-GCN running without weights — random output until trained")

        # keypoint buffer per player_id: {id: [(17,2), ...]}
        self.kp_buffer = {}

    def update_keypoints(self, players):
        """Call every frame with current player list."""
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
        """
        Classify shot for player_id using their buffered keypoints.
        Returns (class_name, confidence) or (None, 0) if not enough data.
        """
        buf = self.kp_buffer.get(player_id, [])
        if len(buf) < WINDOW_FRAMES // 2:
            return None, 0.0

        # pad or trim to WINDOW_FRAMES
        seq = buf[-WINDOW_FRAMES:]
        while len(seq) < WINDOW_FRAMES:
            seq = [seq[0]] + seq

        # (T, V, C) -> (C, T, V)
        arr = np.stack(seq, axis=0)              # (T, V, 2)
        arr = arr.transpose(2, 0, 1)             # (2, T, V)
        x   = torch.tensor(arr).unsqueeze(0).to(self.device)  # (1, 2, T, V)

        with torch.no_grad():
            logits = self.model(x)
            probs  = torch.softmax(logits, dim=1)[0]

        idx  = int(probs.argmax())
        conf = float(probs[idx])
        return SHOT_CLASSES[idx], conf

    def get_buffer_length(self, player_id):
        return len(self.kp_buffer.get(player_id, []))
    
    
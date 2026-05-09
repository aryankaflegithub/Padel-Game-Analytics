import numpy as np
from collections import deque


WRIST_SPEED_THRESH = 15
BALL_PROXIMITY     = 250
MIN_BALL_HISTORY   = 4
COOLDOWN           = 20


class ShotDetector:
    def __init__(self):
        self.shots            = []
        self._last_shot_frame = -COOLDOWN
        self._last_hitter_id  = None
        self._wrist_history   = {}

    def _update_wrists(self, players):
        for p in players:
            pid = p["id"]
            kps = p.get("keypoints")
            if kps is None or len(kps) < 11:
                continue
            wr = np.array(kps[10], dtype=float)
            wl = np.array(kps[9],  dtype=float)
            if pid not in self._wrist_history:
                self._wrist_history[pid] = deque(maxlen=6)
            self._wrist_history[pid].append((wr, wl))

    def _wrist_speed(self, pid):
        hist = self._wrist_history.get(pid)
        if hist is None or len(hist) < 2:
            return 0.0
        h = list(hist)
        speeds = []
        for i in range(1, len(h)):
            sr = np.linalg.norm(h[i][0] - h[i-1][0])
            sl = np.linalg.norm(h[i][1] - h[i-1][1])
            speeds.append(max(sr, sl))
        return max(speeds)

    def _ball_direction_changed(self, ball_history):
        if len(ball_history) < MIN_BALL_HISTORY:
            return False
        mid      = len(ball_history) // 2
        v_before = np.array(ball_history[mid],   dtype=float) - np.array(ball_history[mid-1], dtype=float)
        v_after  = np.array(ball_history[-1],    dtype=float) - np.array(ball_history[-2],    dtype=float)
        if np.linalg.norm(v_before) < 2 or np.linalg.norm(v_after) < 2:
            return False
        return np.dot(v_before, v_after) < 0

    def _find_hitter(self, players, ball_pos):
        best_score, best_player = -1, None
        ball = np.array(ball_pos, dtype=float)

        for p in players:
            pid = p["id"]
            kps = p.get("keypoints")
            if kps is None or len(kps) < 11:
                continue

            wr   = np.array(kps[10], dtype=float)
            wl   = np.array(kps[9],  dtype=float)
            dist = min(np.linalg.norm(wr - ball), np.linalg.norm(wl - ball))

            if dist > BALL_PROXIMITY:
                continue

            speed = self._wrist_speed(pid)
            if speed < WRIST_SPEED_THRESH:
                continue

            score = speed / (dist + 1)
            if score > best_score:
                best_score, best_player = score, p

        if best_player is None:
            best_dist = float("inf")
            for p in players:
                fx, fy = p["foot"]
                dist = np.linalg.norm(np.array([fx, fy], dtype=float) - ball)
                if dist < best_dist:
                    best_dist, best_player = dist, p

        return best_player

    def update(self, frame_idx, ball_history, players, ball_source=None):
        if not players:
            return None
        if frame_idx - self._last_shot_frame < COOLDOWN:
            return None

        self._update_wrists(players)

        if not ball_history or ball_source is None:
            return None

        if ball_source != "tracknet":
            return None

        if not self._ball_direction_changed(ball_history):
            return None

        ball_pos = ball_history[-1]
        hitter   = self._find_hitter(players, ball_pos)

        if hitter is None:
            return None

        if hitter["id"] == self._last_hitter_id:
            return None

        shot = {
            "frame":     frame_idx,
            "timestamp": round(frame_idx / 25.0, 3),
            "player_id": hitter["id"],
            "position":  ball_pos,
        }

        self.shots.append(shot)
        self._last_shot_frame = frame_idx
        self._last_hitter_id  = hitter["id"]
        return shot

    def get_shots(self):
        return self.shots    
    
    
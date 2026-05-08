import numpy as np


PROXIMITY_THRESH = 120
MIN_SPEED_CHANGE = 15
MIN_BALL_HISTORY = 4


class ShotDetector:
    def __init__(self, proximity=PROXIMITY_THRESH, speed_change=MIN_SPEED_CHANGE):
        self.proximity    = proximity
        self.speed_change = speed_change
        self.shots        = []
        self._last_shot_frame = -30

    def _direction_changed(self, history):
        if len(history) < MIN_BALL_HISTORY:
            return False
        mid = len(history) // 2
        v_before = np.array(history[mid-1], dtype=float) - np.array(history[mid-2], dtype=float)
        v_after  = np.array(history[-1],    dtype=float) - np.array(history[-2],    dtype=float)
        return np.linalg.norm(v_after - v_before) >= self.speed_change

    def _nearest_player(self, ball_pos, players):
        best_dist, best_player = float("inf"), None
        for p in players:
            fx, fy = p["foot"]
            dist = np.sqrt((ball_pos[0] - fx)**2 + (ball_pos[1] - fy)**2)
            if dist < best_dist:
                best_dist, best_player = dist, p
        return best_player, best_dist

    def update(self, frame_idx, ball_history, players):
        if len(ball_history) < MIN_BALL_HISTORY or not players:
            return None
        if frame_idx - self._last_shot_frame < 15:
            return None

        ball_pos = ball_history[-1]
        nearest, dist = self._nearest_player(ball_pos, players)

        if dist > self.proximity:
            return None
        if not self._direction_changed(ball_history):
            return None

        shot = {"frame": frame_idx, "player_id": nearest["id"], "position": ball_pos}
        self.shots.append(shot)
        self._last_shot_frame = frame_idx
        return shot

    def get_shots(self):
        return self.shots
    
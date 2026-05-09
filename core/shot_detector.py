import numpy as np


PROXIMITY_THRESH = 150
MIN_SPEED_CHANGE = 15
MIN_BALL_HISTORY = 6


class ShotDetector:
    def __init__(self, proximity=PROXIMITY_THRESH, speed_change=MIN_SPEED_CHANGE):
        self.proximity        = proximity
        self.speed_change     = speed_change
        self.shots            = []
        self._last_shot_frame = -30
        self._last_hitter_id  = None

    def _get_velocities(self, history):
        mid = len(history) // 2
        v_before = np.array(history[mid],   dtype=float) - np.array(history[mid-1], dtype=float)
        v_after  = np.array(history[-1],    dtype=float) - np.array(history[-2],    dtype=float)
        return v_before, v_after

    def _real_direction_change(self, history):
        if len(history) < MIN_BALL_HISTORY:
            return False
        v_before, v_after = self._get_velocities(history)

        speed_before = np.linalg.norm(v_before)
        speed_after  = np.linalg.norm(v_after)

        if speed_before < 2 or speed_after < 2:
            return False

        dot = np.dot(v_before, v_after)
        if dot >= 0:
            return False

        return np.linalg.norm(v_after - v_before) >= self.speed_change

    def _find_hitter(self, ball_history, players):

        mid     = len(ball_history) // 2
        pre_pos = ball_history[mid]

        best_dist, best_player = float("inf"), None
        for p in players:
            fx, fy = p["foot"]
            dist = np.sqrt((pre_pos[0] - fx)**2 + (pre_pos[1] - fy)**2)
            if dist < best_dist:
                best_dist, best_player = dist, p

        return best_player, best_dist

    def update(self, frame_idx, ball_history, players):
        if len(ball_history) < MIN_BALL_HISTORY or not players:
            return None

        # cooldown
        if frame_idx - self._last_shot_frame < 20:
            return None

        if not self._real_direction_change(ball_history):
            return None

        hitter, dist = self._find_hitter(ball_history, players)

        if hitter is None or dist > self.proximity:
            return None

        if hitter["id"] == self._last_hitter_id:
            return None

        shot = {
            "frame":     frame_idx,
            "player_id": hitter["id"],
            "position":  ball_history[-1],
        }
        self.shots.append(shot)
        self._last_shot_frame = frame_idx
        self._last_hitter_id  = hitter["id"]
        return shot

    def get_shots(self):
        return self.shots
    
    
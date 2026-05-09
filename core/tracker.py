import numpy as np


class PlayerTracker:
    def __init__(self, max_history=30):
        self.history     = {}
        self.id_map      = {}       # raw_id -> clean_id
        self.next_clean  = 1
        self.max_history = max_history
        self._last_feet  = {}       # clean_id -> last foot position

    def _get_clean_id(self, raw_id, foot):
        if raw_id in self.id_map:
            return self.id_map[raw_id]

        if len(self.id_map) < 4:
            # assign new clean id
            clean = self.next_clean
            self.next_clean += 1
            self.id_map[raw_id] = clean
            return clean

        # more than 4 raw ids — map to nearest existing player by foot position
        best_clean, best_dist = None, float("inf")
        for clean_id, last_foot in self._last_feet.items():
            dist = np.sqrt((foot[0] - last_foot[0])**2 + (foot[1] - last_foot[1])**2)
            if dist < best_dist:
                best_dist  = dist
                best_clean = clean_id

        self.id_map[raw_id] = best_clean
        return best_clean

    def update(self, players):
        for p in players:
            foot     = p["foot"]
            clean_id = self._get_clean_id(p["id"], foot)
            p["id"]  = clean_id

            self._last_feet[clean_id] = foot

            if clean_id not in self.history:
                self.history[clean_id] = []
            self.history[clean_id].append(foot)
            if len(self.history[clean_id]) > self.max_history:
                self.history[clean_id].pop(0)

    def get_history(self, player_id):
        return self.history.get(player_id, [])
    
    
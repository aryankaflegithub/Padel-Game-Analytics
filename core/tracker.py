class PlayerTracker:
    def __init__(self, max_history=30):
        self.history    = {}
        self.id_map     = {}
        self.next_clean = 1
        self.max_history = max_history

    def _get_clean_id(self, raw_id):
        if raw_id not in self.id_map:
            self.id_map[raw_id] = self.next_clean
            self.next_clean += 1
        return self.id_map[raw_id]

    def update(self, players):
        for p in players:
            clean_id = self._get_clean_id(p["id"])
            p["id"]  = clean_id
            if clean_id not in self.history:
                self.history[clean_id] = []
            self.history[clean_id].append(p["foot"])
            if len(self.history[clean_id]) > self.max_history:
                self.history[clean_id].pop(0)

    def get_history(self, player_id):
        return self.history.get(player_id, [])


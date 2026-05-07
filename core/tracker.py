class PlayerTracker:
    def __init__(self, max_history=30):
        self.history = {}
        self.max_history = max_history

    def update(self, players):
        for p in players:
            pid = p["id"]
            if pid not in self.history:
                self.history[pid] = []
            self.history[pid].append(p["foot"])
            if len(self.history[pid]) > self.max_history:
                self.history[pid].pop(0)

    def get_history(self, player_id):
        return self.history.get(player_id, [])
    

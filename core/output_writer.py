import json
import csv
import os


class OutputWriter:
    def __init__(self, out_dir="output"):
        self.out_dir = out_dir
        self.shots   = []
        os.makedirs(out_dir, exist_ok=True)

    def add_shot(self, frame, timestamp, player_id, shot_type, confidence, position):
        self.shots.append({
            "frame":      frame,
            "timestamp":  round(timestamp, 3),
            "player_id":  player_id,
            "shot_type":  shot_type,
            "confidence": round(confidence, 3),
            "position_x": position[0],
            "position_y": position[1],
        })

    def save(self):
        json_path = os.path.join(self.out_dir, "shots.json")
        csv_path  = os.path.join(self.out_dir, "shots.csv")

        with open(json_path, "w") as f:
            json.dump(self.shots, f, indent=2)

        with open(csv_path, "w", newline="") as f:
            if self.shots:
                writer = csv.DictWriter(f, fieldnames=self.shots[0].keys())
                writer.writeheader()
                writer.writerows(self.shots)

        print(f"saved {len(self.shots)} shots -> {json_path}, {csv_path}")
        
        
from ultralytics import YOLO


class PlayerDetector:
    def __init__(self, model_path="yolov8n-pose.pt", conf=0.5):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model.track(frame, persist=True, conf=self.conf, verbose=False)
        players = []

        if results[0].boxes is None:
            return players

        boxes = results[0].boxes
        keypoints = results[0].keypoints

        for i, box in enumerate(boxes):
            cls = int(box.cls[0])
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            foot_point = ((x1 + x2) // 2, y2)

            kps = None
            if keypoints is not None and i < len(keypoints.xy):
                kps = keypoints.xy[i].cpu().numpy()

            players.append({
                "id": track_id,
                "box": (x1, y1, x2, y2),
                "foot": foot_point,
                "keypoints": kps
            })

        return players
    

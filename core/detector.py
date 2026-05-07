from ultralytics import YOLO


class PlayerDetector:
    def __init__(self, model_path="yolov8l-pose.pt", conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, warped_frame):
        results = self.model.track(
            warped_frame,
            persist=True,
            conf=self.conf,
            device=0,
            verbose=False,
            imgsz=1280
        )
        players = []

        if results[0].boxes is None:
            return players

        boxes     = results[0].boxes
        keypoints = results[0].keypoints

        for i, box in enumerate(boxes):
            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0]) if box.id is not None else -1

            kps = None
            if keypoints is not None and i < len(keypoints.xy):
                kps = keypoints.xy[i].cpu().numpy()

            players.append({
                "id"       : track_id,
                "box"      : (x1, y1, x2, y2),
                "foot"     : ((x1 + x2) // 2, y2),
                "center"   : ((x1 + x2) // 2, (y1 + y2) // 2),
                "keypoints": kps,
                "conf"     : float(box.conf[0])
            })

        return players
    

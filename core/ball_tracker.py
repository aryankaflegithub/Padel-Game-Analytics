import cv2
import numpy as np
import torch

from core.tracknet import TrackNetV3


TRACKNET_W   = 512
TRACKNET_H   = 288
CONF_THRESH  = 0.35
MAX_JUMP     = 150
LOST_LIMIT   = 8


class BallTracker:
    def __init__(self, weights_path=None, max_history=30, device=0):
        self.max_history           = max_history
        self.history               = []
        self.source_history        = []
        self.lost_count            = 0
        self.frame_buffer          = []
        self.gray_buffer           = []
        self.last_detection_source = None

        self.device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
        self.model  = None

        if weights_path:
            self._load(weights_path)

    def _load(self, path):
        self.model = TrackNetV3(in_frames=3).to(self.device)
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()
        print(f"TrackNetV3 loaded | epoch {ck.get('epoch', '?')} | acc {ck.get('test_acc', '?')}")

    def _predict_next(self):
        if len(self.history) < 2:
            return None
        x1, y1 = self.history[-2]
        x2, y2 = self.history[-1]
        vx = np.clip(x2 - x1, -MAX_JUMP, MAX_JUMP)
        vy = np.clip(y2 - y1, -MAX_JUMP, MAX_JUMP)
        return (x2 + vx, y2 + vy)

    def _tracknet_detect(self, frame_h, frame_w):
        if len(self.frame_buffer) < 3:
            return None, 0.0

        imgs = []
        for f in self.frame_buffer[-3:]:
            r = cv2.resize(f, (TRACKNET_W, TRACKNET_H))
            imgs.append(r.astype(np.float32) / 255.0)

        x = np.concatenate([i.transpose(2, 0, 1) for i in imgs], axis=0)
        x = torch.tensor(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(x)

        heatmap = out[0, 2].cpu().numpy()
        conf    = float(heatmap.max())

        if conf < CONF_THRESH:
            return None, conf

        hy, hx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        px = int(hx * frame_w / TRACKNET_W)
        py = int(hy * frame_h / TRACKNET_H)
        return (px, py), conf

    def _fallback_detect(self, frame, ignore_boxes):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        self.gray_buffer.append(gray)
        if len(self.gray_buffer) > 3:
            self.gray_buffer.pop(0)

        if len(self.gray_buffer) < 2:
            return None

        diff = cv2.absdiff(self.gray_buffer[0], self.gray_buffer[-1])
        _, motion = cv2.threshold(diff, 6, 255, cv2.THRESH_BINARY)

        hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (15, 20, 60),  (50, 255, 255))
        white  = cv2.inRange(hsv, (0,  0,  140), (180, 50, 255))
        mask   = cv2.bitwise_and(motion, cv2.bitwise_or(yellow, white))

        if ignore_boxes:
            pmask = np.zeros_like(motion)
            for x1, y1, x2, y2 in ignore_boxes:
                pmask[max(0, y1-50):y2+20, max(0, x1-25):x2+25] = 255
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(pmask))

        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1 or area > 500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0 or not (0.15 < w / h < 7.0):
                continue
            candidates.append((x + w // 2, y + h // 2, area))

        if not candidates:
            return None

        pred = self._predict_next()
        last = self.history[-1] if self.history else None

        if last:
            candidates = [
                (cx, cy, a) for cx, cy, a in candidates
                if (cx - last[0])**2 + (cy - last[1])**2 <= MAX_JUMP**2
            ]
        if not candidates:
            return None

        ref = pred or last
        if ref:
            candidates.sort(key=lambda p: (p[0] - ref[0])**2 + (p[1] - ref[1])**2)

        return (candidates[0][0], candidates[0][1])

    def update(self, frame, ignore_boxes=None, debug=False):
        h, w = frame.shape[:2]

        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)

        point = None
        self.last_detection_source = None

        if self.model is not None:
            point, conf = self._tracknet_detect(h, w)
            if debug:
                print(f"tracknet conf: {conf:.3f} -> {point}")
            if point is not None:
                self.last_detection_source = "tracknet"
            else:
                point = self._fallback_detect(frame, ignore_boxes)
                if point is not None:
                    self.last_detection_source = "fallback"
                    if debug:
                        print(f"fallback -> {point}")
        else:
            point = self._fallback_detect(frame, ignore_boxes)
            if point is not None:
                self.last_detection_source = "fallback"

        if point is None:
            self.lost_count += 1
            if self.lost_count >= LOST_LIMIT:
                self.history        = self.history[-1:]        if self.history        else []
                self.source_history = self.source_history[-1:] if self.source_history else []
            return None

        self.lost_count = 0
        self.history.append(point)
        self.source_history.append(self.last_detection_source)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            self.source_history.pop(0)
        return point

    def get_history(self):
        return self.history

    def get_source_history(self):
        return self.source_history
    
    
import cv2
import numpy as np


MAX_JUMP = 250
LOST_LIMIT = 8

class BallTracker:
    def __init__(self, max_history=30):
        self.frame_buffer = []
        self.history = []
        self.max_history = max_history
        self.lost_count = 0

    def _predict_next(self):
        if len(self.history) < 2:
            return None
        x1, y1 = self.history[-2]
        x2, y2 = self.history[-1]
        vx = np.clip(x2 - x1, -MAX_JUMP, MAX_JUMP)
        vy = np.clip(y2 - y1, -MAX_JUMP, MAX_JUMP)
        return (x2 + vx, y2 + vy)

    def _score(self, cx, cy, pred, last):
        ref = pred if pred is not None else last
        if ref is None:
            return 0
        return (cx - ref[0])**2 + (cy - ref[1])**2

    def update(self, frame, ignore_boxes=None, debug=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        self.frame_buffer.append(gray)
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)

        if len(self.frame_buffer) < 2:
            return None

        diff = cv2.absdiff(self.frame_buffer[0], self.frame_buffer[-1])
        _, motion = cv2.threshold(diff, 6, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # padel ball: yellow, green-yellow, washed white
        yellow      = cv2.inRange(hsv, (15, 20, 60),  (50, 255, 255))
        white       = cv2.inRange(hsv, (0,  0,  140), (180, 50, 255))
        color_mask  = cv2.bitwise_or(yellow, white)

        mask = cv2.bitwise_and(motion, color_mask)

        if ignore_boxes is not None:
            player_mask = np.zeros_like(motion)
            for x1, y1, x2, y2 in ignore_boxes:
                px1 = max(0, x1 - 25)
                py1 = max(0, y1 - 50)
                px2 = min(mask.shape[1], x2 + 25)
                py2 = min(mask.shape[0], y2 + 20)
                player_mask[py1:py2, px1:px2] = 255
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(player_mask))

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        if debug:
            cv2.imwrite(f"output/masks/mask_{len(self.history):04d}.png", mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1 or area > 500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0:
                continue
            ratio = w / h
            if ratio < 0.15 or ratio > 7.0:
                continue
            cx = x + w // 2
            cy = y + h // 2
            candidates.append((cx, cy, area))

        if not candidates:
            self.lost_count += 1
            if self.lost_count >= LOST_LIMIT:
                self.history = self.history[-1:] if self.history else []
            return None

        pred = self._predict_next()
        last = self.history[-1] if self.history else None

        if last is not None:
            candidates = [
                (cx, cy, a) for cx, cy, a in candidates
                if (cx - last[0])**2 + (cy - last[1])**2 <= MAX_JUMP**2
            ]

        if not candidates:
            self.lost_count += 1
            if self.lost_count >= LOST_LIMIT:
                self.history = self.history[-1:] if self.history else []
            return None

        candidates.sort(key=lambda p: self._score(p[0], p[1], pred, last))

        self.lost_count = 0
        point = (int(candidates[0][0]), int(candidates[0][1]))
        self.history.append(point)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return point

    def get_history(self):
        return self.history
    

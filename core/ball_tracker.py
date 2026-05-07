import cv2
import numpy as np


class BallTracker:
    def __init__(self, max_history=30):
        self.prev_gray = None
        self.history = []
        self.max_history = max_history

    def update(self, frame, ignore_boxes=None, debug=False):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        diff = cv2.absdiff(self.prev_gray, gray)
        self.prev_gray = gray

        _, motion = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (18, 30, 80), (48, 255, 255))
        white  = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))
        color_mask = cv2.bitwise_or(yellow, white)
        mask = cv2.bitwise_and(motion, color_mask)

        player_mask = np.zeros_like(motion)
        if ignore_boxes is not None:
            for x1, y1, x2, y2 in ignore_boxes:
                x1 = max(0, x1 - 20)
                y1 = max(0, y1 - 20)
                x2 = min(mask.shape[1], x2 + 20)
                y2 = min(mask.shape[0], y2 + 60)
                player_mask[y1:y2, x1:x2] = 255

        mask = cv2.bitwise_and(mask, cv2.bitwise_not(player_mask))

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        if debug:
            cv2.imwrite(f"output/masks/mask_{len(self.history):04d}.png", mask)      

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 1 or area > 300:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w == 0 or h == 0:
                continue
            ratio = w / h
            if ratio < 0.2 or ratio > 6.0:
                continue
            cx = x + w // 2
            cy = y + h // 2
            candidates.append((cx, cy, area))

        if not candidates:
            return None

        if self.history:
            lx, ly = self.history[-1]
            candidates.sort(key=lambda p: (p[0]-lx)**2 + (p[1]-ly)**2)
        else:
            candidates.sort(key=lambda p: p[2], reverse=True)

        point = (int(candidates[0][0]), int(candidates[0][1]))
        self.history.append(point)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return point

    def get_history(self):
        return self.history
    

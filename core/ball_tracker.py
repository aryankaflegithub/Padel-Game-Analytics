# import cv2
# import numpy as np


# class BallTracker:
#     def __init__(self, max_history=30):
#         self.prev_gray = None
#         self.history = []
#         self.max_history = max_history

#     def update(self, frame, ignore_boxes=None, debug=False):
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = cv2.GaussianBlur(gray, (5, 5), 0)

#         if self.prev_gray is None:
#             self.prev_gray = gray
#             return None

#         diff = cv2.absdiff(self.prev_gray, gray)
#         self.prev_gray = gray

#         _, motion = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)

#         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         yellow = cv2.inRange(hsv, (18, 30, 80), (48, 255, 255))
#         white  = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))
#         color_mask = cv2.bitwise_or(yellow, white)
#         mask = cv2.bitwise_and(motion, color_mask)

#         player_mask = np.zeros_like(motion)
#         if ignore_boxes is not None:
#             for x1, y1, x2, y2 in ignore_boxes:
#                 x1 = max(0, x1 - 20)
#                 y1 = max(0, y1 - 20)
#                 x2 = min(mask.shape[1], x2 + 20)
#                 y2 = min(mask.shape[0], y2 + 60)
#                 player_mask[y1:y2, x1:x2] = 255

#         mask = cv2.bitwise_and(mask, cv2.bitwise_not(player_mask))

#         kernel = np.ones((3, 3), np.uint8)
#         mask = cv2.dilate(mask, kernel, iterations=1)
        
#         if debug:
#             cv2.imwrite(f"output/masks/mask_{len(self.history):04d}.png", mask)      

#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         candidates = []
#         for c in contours:
#             area = cv2.contourArea(c)
#             if area < 1 or area > 300:
#                 continue
#             x, y, w, h = cv2.boundingRect(c)
#             if w == 0 or h == 0:
#                 continue
#             ratio = w / h
#             if ratio < 0.2 or ratio > 6.0:
#                 continue
#             cx = x + w // 2
#             cy = y + h // 2
#             candidates.append((cx, cy, area))

#         if not candidates:
#             return None

#         if self.history:
#             lx, ly = self.history[-1]
#             candidates.sort(key=lambda p: (p[0]-lx)**2 + (p[1]-ly)**2)
#         else:
#             candidates.sort(key=lambda p: p[2], reverse=True)

#         point = (int(candidates[0][0]), int(candidates[0][1]))
#         self.history.append(point)
#         if len(self.history) > self.max_history:
#             self.history.pop(0)
#         return point

#     def get_history(self):
#         return self.history
    

import cv2
import numpy as np


MAX_JUMP = 250
MIN_JUMP = 2
LOST_LIMIT = 8

class BallTracker:
    def __init__(self, max_history=30):
        self.prev_gray = None
        self.history = []
        self.max_history = max_history
        self.lost_count = 0

    def _predict_next(self):
        if len(self.history) < 2:
            return None
        x1, y1 = self.history[-2]
        x2, y2 = self.history[-1]
        vx = x2 - x1
        vy = y2 - y1
        # clamp velocity so prediction doesn't fly off screen
        vx = np.clip(vx, -MAX_JUMP, MAX_JUMP)
        vy = np.clip(vy, -MAX_JUMP, MAX_JUMP)
        return (x2 + vx, y2 + vy)

    def _score_candidate(self, cx, cy, pred, last):
        if pred is not None:
            dx = cx - pred[0]
            dy = cy - pred[1]
            return dx*dx + dy*dy
        elif last is not None:
            dx = cx - last[0]
            dy = cy - last[1]
            return dx*dx + dy*dy
        return 0

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

        if ignore_boxes is not None:
            player_mask = np.zeros_like(motion)
            for x1, y1, x2, y2 in ignore_boxes:
                px1 = max(0, x1 - 25)
                py1 = max(0, y1 - 40)   # cover racket raised above head
                px2 = min(mask.shape[1], x2 + 25)
                py2 = min(mask.shape[0], y2 + 20)
                player_mask[py1:py2, px1:px2] = 255
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
            self.lost_count += 1
            if self.lost_count >= LOST_LIMIT:
                self.history = self.history[-1:] if self.history else []
            return None

        pred = self._predict_next()
        last = self.history[-1] if self.history else None

        if last is not None:
            candidates = [
                (cx, cy, area) for cx, cy, area in candidates
                if (cx - last[0])**2 + (cy - last[1])**2 <= MAX_JUMP**2
            ]

        if not candidates:
            self.lost_count += 1
            if self.lost_count >= LOST_LIMIT:
                self.history = self.history[-1:] if self.history else []
            return None

        candidates.sort(key=lambda p: self._score_candidate(p[0], p[1], pred, last))

        self.lost_count = 0
        point = (int(candidates[0][0]), int(candidates[0][1]))
        self.history.append(point)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        return point

    def get_history(self):
        return self.history
    
    
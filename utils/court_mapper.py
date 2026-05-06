import cv2
import numpy as np
import json
import os


class CourtMapper:
    def __init__(self):
        self.court_polygon = None
        self.homography_matrix = None
        self.clicked_points = []
        self.output_size = (400, 800)
        self.save_path = "output/court_points.json"

    def load_if_exists(self):
        if os.path.exists(self.save_path):
            with open(self.save_path, "r") as f:
                self.clicked_points = json.load(f)
            self.court_polygon = np.array(self.clicked_points, dtype=np.int32)
            print(f"loaded court points: {self.clicked_points}")
            return True
        return False

    def save_points(self):
        with open(self.save_path, "w") as f:
            json.dump(self.clicked_points, f)
        print(f"court points saved to {self.save_path}")

    def define_court_manually(self, frame):
        print("click 4 corners: top-left, top-right, bottom-right, bottom-left. press Q when done")
        display = frame.copy()

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(self.clicked_points) < 4:
                self.clicked_points.append((x, y))
                cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
                if len(self.clicked_points) > 1:
                    cv2.line(display, self.clicked_points[-2], self.clicked_points[-1], (0, 255, 0), 2)
                if len(self.clicked_points) == 4:
                    cv2.line(display, self.clicked_points[-1], self.clicked_points[0], (0, 255, 0), 2)
                print(f"point {len(self.clicked_points)}: ({x}, {y})")

        cv2.namedWindow("court", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("court", 1280, 720)
        cv2.setMouseCallback("court", on_click)

        while True:
            cv2.imshow("court", display)
            if cv2.waitKey(1) & 0xFF == ord("q") and len(self.clicked_points) == 4:
                break

        cv2.destroyAllWindows()
        self.court_polygon = np.array(self.clicked_points, dtype=np.int32)
        self.save_points()
        return self.clicked_points

    def build_homography(self):
        W, H = self.output_size
        dst = np.array([[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float32)
        src = np.array(self.clicked_points, dtype=np.float32)
        self.homography_matrix, _ = cv2.findHomography(src, dst)

    def is_inside_court(self, point, margin=20):
        if self.court_polygon is None:
            return True
        result = cv2.pointPolygonTest(self.court_polygon, (float(point[0]), float(point[1])), measureDist=True)
        return result >= -margin

    def get_player_side(self, foot_point):
        if self.court_polygon is None:
            return "unknown"
        pts = self.court_polygon
        mid_left  = ((pts[0][0] + pts[3][0]) // 2, (pts[0][1] + pts[3][1]) // 2)
        mid_right = ((pts[1][0] + pts[2][0]) // 2, (pts[1][1] + pts[2][1]) // 2)
        near_side = np.array([mid_left, mid_right, pts[2], pts[3]], dtype=np.int32)
        result = cv2.pointPolygonTest(near_side, (float(foot_point[0]), float(foot_point[1])), False)
        return "near" if result >= 0 else "far"

    def draw_court_overlay(self, frame):
        if self.court_polygon is None:
            return frame
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.court_polygon], (0, 255, 0))
        frame = cv2.addWeighted(overlay, 0.05, frame, 0.95, 0)
        cv2.polylines(frame, [self.court_polygon], True, (0, 255, 0), 2)
        return frame
    

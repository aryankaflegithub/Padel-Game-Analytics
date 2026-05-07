import cv2

COLORS = [
    (255, 0, 0), (0, 165, 255), (0, 255, 255), (255, 0, 255),
    (255, 255, 0), (0, 255, 0)
]

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

def draw_players(frame, players, court_mapper=None):
    for p in players:
        pid = p["id"]
        color = COLORS[pid % len(COLORS)]
        x1, y1, x2, y2 = p["box"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"P{pid}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if court_mapper is not None:
            side = court_mapper.get_player_side(p["foot"])
            cv2.putText(frame, side, (x1, y1 - 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        kps = p["keypoints"]
        if kps is not None:
            for kx, ky in kps:
                if kx > 0 and ky > 0:
                    cv2.circle(frame, (int(kx), int(ky)), 3, color, -1)
            for a, b in SKELETON:
                if a < len(kps) and b < len(kps):
                    ax, ay = int(kps[a][0]), int(kps[a][1])
                    bx, by = int(kps[b][0]), int(kps[b][1])
                    if ax > 0 and ay > 0 and bx > 0 and by > 0:
                        cv2.line(frame, (ax, ay), (bx, by), color, 1)

    return frame


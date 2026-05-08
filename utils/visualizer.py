import cv2

COLORS = {
    1: (0, 255, 0),
    2: (0, 165, 255),
    3: (255, 0, 255),
    4: (0, 255, 255)
}

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def get_color(pid):
    return COLORS.get(pid, (255, 255, 255))

def unwarp_keypoints(kps, mapper, warp_scale):
    sx, sy = warp_scale
    result = []
    for kx, ky in kps:
        if kx > 0 and ky > 0:
            orig = mapper.unwarp_point((int(kx / sx), int(ky / sy)))
            result.append(orig)
        else:
            result.append(None)
    return result

def draw_players_on_original(frame, players, mapper, warp_scale):
    sx, sy = warp_scale

    for p in players:
        pid   = p["id"]
        color = get_color(pid)

        cx = int(p["center"][0] / sx)
        cy = int(p["center"][1] / sy)
        fx = int(p["foot"][0]   / sx)
        fy = int(p["foot"][1]   / sy)

        orig_center = mapper.unwarp_point((cx, cy))
        orig_foot   = mapper.unwarp_point((fx, fy))

        # bounding circle and label
        cv2.circle(frame, orig_center, 25, color, 2)
        cv2.circle(frame, orig_foot, 6, color, -1)
        cv2.putText(frame, f"P{pid}", (orig_center[0] + 15, orig_center[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        side = "near" if orig_foot[1] > frame.shape[0] // 2 else "far"
        cv2.putText(frame, side, (orig_center[0] + 15, orig_center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # skeleton
        kps = p.get("keypoints")
        if kps is not None and len(kps) >= 17:
            orig_kps = unwarp_keypoints(kps, mapper, warp_scale)

            # draw joints
            for pt in orig_kps:
                if pt is not None:
                    cv2.circle(frame, pt, 4, color, -1)

            # draw bones
            for a, b in SKELETON:
                if a < len(orig_kps) and b < len(orig_kps):
                    if orig_kps[a] is not None and orig_kps[b] is not None:
                        cv2.line(frame, orig_kps[a], orig_kps[b], color, 2)

    return frame

def draw_ball_on_original(frame, ball_point, ball_history, mapper, warp_scale):
    sx, sy = warp_scale

    mapped_history = []

    for pt in ball_history:
        x = int(pt[0] / sx)
        y = int(pt[1] / sy)
        mapped_history.append(mapper.unwarp_point((x, y)))

    for i in range(1, len(mapped_history)):
        cv2.line(frame, mapped_history[i - 1], mapped_history[i], (0, 0, 255), 2)

    if ball_point is not None:
        x = int(ball_point[0] / sx)
        y = int(ball_point[1] / sy)
        orig = mapper.unwarp_point((x, y))
        cv2.circle(frame, orig, 8, (0, 0, 255), -1)
        cv2.putText(frame, "ball", (orig[0] + 10, orig[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame

def draw_dashboard(frame, shot_counts, total_shots):

    h, w = frame.shape[:2]
    panel_w = 220

    # dark overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    x = w - panel_w + 15
    y = 40

    cv2.putText(frame, "SHOT TRACKER", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    y += 10
    cv2.line(frame, (w - panel_w + 10, y), (w - 10, y), (100, 100, 100), 1)
    y += 30

    for pid in sorted(shot_counts.keys()):
        count = shot_counts[pid]
        label = f"Player {pid}"
        cv2.putText(frame, label, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)
        y += 25
        cv2.putText(frame, f"  Hits: {count}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 10
        cv2.line(frame, (w - panel_w + 10, y), (w - 10, y), (60, 60, 60), 1)
        y += 20

    y += 10
    cv2.putText(frame, f"Total: {total_shots}", (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

    return frame


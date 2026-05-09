import cv2
import torch
import os

from core.detector        import PlayerDetector
from core.tracker         import PlayerTracker
from core.ball_tracker    import BallTracker
from core.shot_detector   import ShotDetector
from core.shot_classifier import ShotClassifier
from core.output_writer   import OutputWriter
from utils.court_mapper   import CourtMapper
from utils.visualizer     import draw_players_on_original, draw_ball_on_original, draw_dashboard

torch.cuda.empty_cache()

VIDEO_PATH   = "data/input_sample_video.mp4"
OUT_PATH     = "output/ball_test.mp4"
TEST_SECONDS = 20

cap        = cv2.VideoCapture(VIDEO_PATH)
fps        = cap.get(cv2.CAP_PROP_FPS)
w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
max_frames = int(fps * TEST_SECONDS)

mapper = CourtMapper()
mapper.load_if_exists()
mapper.build_homography(padding=80)

detector        = PlayerDetector(conf=0.25)
player_tracker  = PlayerTracker()
ball_tracker    = BallTracker(weights_path="models/tracknet/model_finetuned.pt", max_history=25, device=0)
shot_detector   = ShotDetector()
shot_classifier = ShotClassifier(weights_path="models/shot_classifier.pt", device=0)
output_writer   = OutputWriter(out_dir="output")

writer = cv2.VideoWriter(OUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

WARP_SIZE  = (1200, 1000)
warp_scale = (
    WARP_SIZE[0] / mapper.output_size[0],
    WARP_SIZE[1] / mapper.output_size[1]
)

os.makedirs("output/masks", exist_ok=True)
frame_count    = 0
shot_counts    = {}
all_player_ids = set()

while frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    warped     = cv2.warpPerspective(frame, mapper.homography_matrix, mapper.output_size)
    warped_big = cv2.resize(warped, WARP_SIZE)

    players = detector.detect(warped_big)
    players = sorted(players, key=lambda p: p["conf"], reverse=True)[:4]
    player_tracker.update(players)

    for p in players:
        all_player_ids.add(p["id"])
        if p["id"] not in shot_counts:
            shot_counts[p["id"]] = 0

    shot_classifier.update_keypoints(players)

    ignore_boxes = [p["box"] for p in players]
    ball = ball_tracker.update(warped_big, ignore_boxes, debug=(frame_count % 25 == 0))

    shot = shot_detector.update(
        frame_idx      = frame_count,
        ball_history   = ball_tracker.get_history(),
        players        = players,
        ball_source    = ball_tracker.last_detection_source,
        source_history = ball_tracker.get_source_history()
    )

    if shot:
        shot_type, conf = shot_classifier.classify(shot["player_id"])
        pid       = shot["player_id"]
        timestamp = shot["timestamp"]

        shot_counts[pid] = shot_counts.get(pid, 0) + 1

        output_writer.add_shot(
            frame      = frame_count,
            timestamp  = timestamp,
            player_id  = pid,
            shot_type  = shot_type,
            confidence = conf,
            position   = shot["position"]
        )

        print(f"SHOT | frame {frame_count} | t={timestamp:.2f}s | P{pid} | {shot_type} ({conf:.2f})")

        sx       = int(shot["position"][0] / warp_scale[0])
        sy       = int(shot["position"][1] / warp_scale[1])
        orig_pos = mapper.unwarp_point((sx, sy))
        cv2.circle(frame, orig_pos, 15, (0, 255, 255), 3)
        cv2.putText(frame, f"P{pid} {shot_type}",
                    (orig_pos[0] + 10, orig_pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    frame = mapper.draw_court_overlay(frame)
    frame = draw_players_on_original(frame, players, mapper, warp_scale)
    frame = draw_ball_on_original(frame, ball, ball_tracker.get_history(), mapper, warp_scale)
    frame = draw_dashboard(frame, shot_counts, all_player_ids, sum(shot_counts.values()))

    cv2.putText(frame, f"frame {frame_count} | players: {len(players)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    writer.write(frame)
    del warped, warped_big
    frame_count += 1

    if frame_count % 25 == 0:
        print(f"processed {frame_count}/{max_frames}")

cap.release()
writer.release()
output_writer.save()
print(f"done -> {OUT_PATH}")

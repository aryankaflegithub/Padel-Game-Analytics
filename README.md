### AI/ML Internship Assignment

## Padel Game Analytics — Shot Classification System

So we are building a computer vision pipeline that takes a padel match video as input and gives output as a structured file (JSON/CSV) telling us exactly when each shot happened, who hit it and what type it was.

Firstly i watched the video and saw few things.
1. The camera is a wide angle fixed camera. 
2. The ball is very tiny so normal YOLO will definitely miss it. 
3. Players look squished so arm swing might be harder to read.
4. Boundry is clearly defined and the base is blue in color.

So the processing we will be following will be :
1. Read the papers:
    1. https://arxiv.org/pdf/1907.03698 (TrackNet: A Deep Learning Network for Tracking High-speed and Tiny Objects in Sports Applications)
    2. https://arxiv.org/pdf/1801.07455 (Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition)
    3. https://arxiv.org/pdf/2110.06864 (ByteTrack: Multi-Object Tracking by Associating Every Detection Box)

2. Detect the boundry using line detection.
3. Use YOLO v8 for palyer detection.
4. Use ByteTrack for tracking player.
5. Use TrackNet for tracking ball.
6. Using above 2 detect player hitting the ball (shot).
7. Use STGCN for shot classification.
8. Create a JSON and CSV file as output.


## Set up explanation and run

### 1. Requirements

- Python 3.10
- NVIDIA GPU with CUDA support
- Install dependencies:

```bash
pip install ultralytics opencv-python torch torchvision numpy pandas mediapipe scipy matplotlib
```

### 2. Setup

Put your match video at:

```
data/input_sample_video.mp4
```

The system needs to know where the court boundaries are in your video. Run:

```bash
python utils/court_mapper.py
```

A window will open showing your video frame. Click the four corners of the court in this order:

1. Top-left
2. Top-right
3. Bottom-right
4. Bottom-left

Press `q` to save. This only needs to be done once per camera setup.

To use court mapper you need to make another `.py` file to run it or just add the following code.

```bash
if __name__ == "__main__":
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/input_sample_video.mp4"

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"could not read video: {video_path}")
        sys.exit(1)

    os.makedirs("output", exist_ok=True)

    mapper = CourtMapper()
    mapper.define_court_manually(frame)
    mapper.build_homography(padding=80)
    print("homography built and points saved")
``` 


### 3. Running the Pipeline

```bash
python main.py
```

The script will:

1. Load the video
2. Detect and track all 4 players
3. Track the ball using the fine-tuned TrackNetV3 model
4. Detect shot events automatically
5. Classify each shot as forehand, backhand or smash
6. Save the annotated video and structured output

Progress prints to the console every 25 frames.


## Output

`output/ball_test.mp4` - the original video with overlaid:
- Player skeletons and IDs
- Ball position and trail
- Shot markers with player and shot type
- Live shot count dashboard

`shots.json` - json file with all details.

`shots.csv` - csv file with all details.

details are : `frame,timestamp,player_id,shot_type,confidence,position_x,position_y`


## Configuration

To change which video is processed open `main.py` and edit the top section.

```python
VIDEO_PATH   = "data/input_sample_video.mp4"   # path to your video
OUT_PATH     = "output/ball_test.mp4"           # where to save output
TEST_SECONDS = 20                               # how many seconds to process
```

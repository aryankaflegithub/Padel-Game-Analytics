AI/ML Internship Assignment

Project Title
Padel Game Analytics — Shot Classification System

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


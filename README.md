# 🏀 25-2 YSAL Project – Samsung Thunders Collaboration

This repository is part of a **collaborative project with Samsung Thunders**, focusing on **basketball broadcast video analysis**, especially **player tracking and pose detection** from TV/YouTube broadcast footage.

Our pipeline uses **SAMURAI (Segment Anything Model for video tracking)** to track players over frames and detect body joints, enabling automated data collection of player movements during games.

Reference implementation (SAMURAI):
🔗 https://github.com/yangchris11/samurai


---

## 📌 Project Description

This project implements a pipeline that:
- Receives basketball broadcast video
- Tracks players frame-by-frame
- Detects their joints (pose keypoints)
- Saves and records those results for further analysis (e.g., shot contest, defense proximity)

We aim to build an automated system that converts raw broadcast videos into machine-readable player data.


---

## 📁 Project Structure (simplified)
```
samurai/
├── scripts/
│ └── demo.py
├── weights/
├── bbox_outputs/
├── Results/
```


You will mainly run the code from inside `scripts`.


---

## 🚀 How to Run

Run the demo script like this:

```bash
python /root/samurai/scripts/demo.py \
--video_path /root/samurai/clean.mp4 \
--txt_path /root/samurai/bbox_outputs/first_frame_bbox_sam2.txt \
--model_path /root/samurai/weights/sam2.1_hiera_tiny.pt \
--video_output_path /root/samurai/Results/mot_output.mp4 \
--save_to_video True
```

Arguments

- video_path: input broadcast video

- txt_path: bounding box for the target player(s)

- model_path: SAMURAI / SAM2 model checkpoint

- video_output_path: output result video

- save_to_video: True to export visualization video

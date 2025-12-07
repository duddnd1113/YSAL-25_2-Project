#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import os
from ultralytics import YOLO

def detect_first_frame(video_path, output_dir="bbox_outputs"):
    model = YOLO("yolov8s.pt")

    # ----------------------
    # Load first frame
    # ----------------------
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("❌ Cannot read video first frame!")

    # ----------------------
    # Run YOLO
    # ----------------------
    results = model(frame)[0]

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------
    # Collect bboxes (xyxy)
    # ----------------------
    bboxes = []
    for box in results.boxes:
        cls = int(box.cls[0])
        if cls == 0:  # person
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bboxes.append((x1, y1, x2, y2))

    # ----------------------
    # 1️⃣ ID 있는 버전 (id x1 y1 x2 y2)
    # ----------------------
    txt_id = os.path.join(output_dir, "first_frame_bbox_id.txt")
    with open(txt_id, "w") as f:
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            f.write(f"{i} {x1} {y1} {x2} {y2}\n")

    print(f"📌 Saved ID version → {txt_id}")


    # ----------------------
    # SAM2 호환 버전 (x,y,w,h) comma
    # ----------------------
    txt_sam2 = os.path.join(output_dir, "first_frame_bbox_sam2.txt")
    with open(txt_sam2, "w") as f:
        for (x1, y1, x2, y2) in bboxes:
            w = x2 - x1
            h = y2 - y1
            f.write(f"{int(x1)},{int(y1)},{int(w)},{int(h)}\n")

    print(f"📌 Saved SAM2 version → {txt_sam2}")


    # ----------------------
    # Visualization
    # ----------------------
    annotated = frame.copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0,255,0), 2)
        cv2.putText(annotated, f"P{i}", (int(x1), int(y1)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    save_img = os.path.join(output_dir, "annotated_first_frame.jpg")
    cv2.imwrite(save_img, annotated)
    print(f"📌 Visualization saved → {save_img}")


if __name__ == "__main__":
    video_path = "/root/samurai/8_clean.mp4"
    detect_first_frame(video_path)
import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import pandas as pd
import matplotlib.pyplot as plt

from sam2.build_sam import build_sam2_video_predictor
import mediapipe as mp


# ---------------------------------------
# ID 설정
# ---------------------------------------
Defender_IDS = {3}
Shotter_IDS  = {5}

BLUE_IDS = Defender_IDS
RED_IDS  = Shotter_IDS
# ---------------------------------------


def load_txt(gt_path, blue_ids, red_ids):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    target_ids = set(blue_ids) | set(red_ids)

    for fid, line in enumerate(gt):
        if fid not in target_ids:
            continue

        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts



def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size!")


def prepare_frames_or_path(video_path):
    if video_path.endswith(".mp4") or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format.")



def main(args):

    bottom_records = []
    pose_records   = []

    # 📌 CSV, plot 저장 폴더
    save_dir = os.path.dirname(args.video_output_path) or "."

    # -------- MediaPipe Pose 준비 --------
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    # -----------------------------------

    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")

    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path, BLUE_IDS, RED_IDS)

    frame_rate = 30
    if args.save_to_video:
        cap = cv2.VideoCapture(args.video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        loaded_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            loaded_frames.append(frame)
        cap.release()
        height, width = loaded_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))

    # -------------------------------------------
    # MediaPipe Pose keypoint index (L/R 평균)
    # -------------------------------------------
    # 참고: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    kp_pairs = {
        "shoulder": (11, 12),   # left shoulder, right shoulder
        "elbow":    (13, 14),   # left elbow, right elbow
        "wrist":    (15, 16),   # left wrist, right wrist
        "knee":     (25, 26),   # left knee, right knee
    }

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):

        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)

        for obj_id, (bbox, _) in prompts.items():
            predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=obj_id)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):

            if frame_idx >= len(loaded_frames):
                break

            bbox_to_vis = {}

            # SAMURAI 마스크 → bbox
            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0
                idx = np.argwhere(mask)

                if len(idx) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = idx.min(axis=0).tolist()
                    y_max, x_max = idx.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

                bbox_to_vis[obj_id] = bbox

            if args.save_to_video:
                img = loaded_frames[frame_idx]

                for obj_id, (x, y, w, h) in bbox_to_vis.items():

                    # bottom y 저장
                    bottom_y = y + h
                    bottom_records.append({
                        "frame": frame_idx,
                        "obj_id": obj_id,
                        "bottom_y": float(bottom_y)
                    })

                    # ---------- MediaPipe Pose 적용 ----------
                    crop = img[y:y+h, x:x+w]
                    kps = None

                    if crop.size != 0:
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                        result = pose_estimator.process(crop_rgb)

                        if result.pose_landmarks is not None:
                            landmarks = result.pose_landmarks.landmark
                            kps = landmarks  # list of 33

                    # ---------------------------------------
                    # 평균 keypoint 기록
                    # ---------------------------------------
                    rec = {"frame": frame_idx, "obj_id": obj_id}

                    if kps is not None:
                        for name, (li, ri) in kp_pairs.items():
                            lx, ly = kps[li].x, kps[li].y
                            rx, ry = kps[ri].x, kps[ri].y

                            # MediaPipe는 crop 내에서 [0,1] 정규화 → 다시 pixel로 변환
                            avg_x = ((lx + rx) / 2.0) * w + x
                            avg_y = ((ly + ry) / 2.0) * h + y

                            rec[f"{name}_x"] = float(avg_x)
                            rec[f"{name}_y"] = float(avg_y)
                    else:
                        for name in kp_pairs.keys():
                            rec[f"{name}_x"] = np.nan
                            rec[f"{name}_y"] = np.nan

                    pose_records.append(rec)

                    # bbox 그리기
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(img, str(obj_id), (x, y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # -------------------------------------------
                # frame 20에서 keypoint 시각화 (저장 1회)
                # -------------------------------------------
                if frame_idx == 20:
                    vis_img = img.copy()
                    for rec in pose_records:
                        if rec["frame"] != frame_idx:
                            continue

                        # keypoint 좌표 가져오기
                        for name in ["shoulder", "elbow", "wrist", "knee"]:
                            kx = rec[f"{name}_x"]
                            ky = rec[f"{name}_y"]
                            if not np.isnan(kx) and not np.isnan(ky):
                                cv2.circle(vis_img, (int(kx), int(ky)), 6, (0, 0, 255), -1)
                                cv2.putText(vis_img, name, (int(kx)+4, int(ky)+4),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                    save_path = os.path.join(save_dir, "frame20_keypoints.png")
                    cv2.imwrite(save_path, vis_img)
                    print(f"📌 frame20 keypoint visualization saved -> {save_path}")

                out.write(img)

    if args.save_to_video:
        out.release()

    # ========= CSV 저장 (같은 폴더) ==========
    pd.DataFrame(bottom_records).to_csv(os.path.join(save_dir, "bottom_y_tracking.csv"), index=False)
    pd.DataFrame(pose_records).to_csv(os.path.join(save_dir, "pose4_tracking.csv"), index=False)

    print("📌 bottom_y_tracking.csv saved")
    print("📌 pose4_tracking.csv saved")

    # ========= Keypoint plot ==========
    pose_df = pd.DataFrame(pose_records)

    for obj in pose_df["obj_id"].unique():
        sub = pose_df[pose_df["obj_id"] == obj].sort_values("frame")

        plt.figure(figsize=(12, 5))
        for name in kp_pairs.keys():
            plt.plot(sub["frame"], sub[f"{name}_y"], label=f"{name}_y")

        plt.title(f"Object {obj} keypoint(y) avg L/R (MediaPipe)")
        plt.xlabel("frame")
        plt.ylabel("Y pixel")
        plt.legend()
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.savefig(os.path.join(save_dir, f"obj_{obj}_kpy_avg.png"))
        plt.close()

    print("📌 keypoint plots saved")

    # 리소스 정리
    pose_estimator.close()
    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--txt_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--video_output_path", default="demo.mp4")
    parser.add_argument("--save_to_video", default=True)
    args = parser.parse_args()
    main(args)

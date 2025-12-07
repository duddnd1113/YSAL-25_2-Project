import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import pandas as pd
from ultralytics import YOLO
from sam2.build_sam import build_sam2_video_predictor
import matplotlib.pyplot as plt

# 관절 없이 bbox 위아래만 추출하고 기록

# ---------------------------------------
# ID 설정
# ---------------------------------------
Defender_IDS = {0, 4, 5, 1, 13}
Shotter_IDS  = {12, 6, 3, 2, 7}

BLUE_IDS = Defender_IDS
RED_IDS  = Shotter_IDS

BLUE_COLOR = (255, 0, 0)   # BGR
RED_COLOR  = (0, 0, 255)
DEFAULT_COLOR = (0, 255, 0)
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

    bbox_records = []
    pose_records = []

    save_dir = os.path.dirname(args.video_output_path) or "."

    pose_model = YOLO("yolov8x-pose.pt")

    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")

    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path, BLUE_IDS, RED_IDS)

    frame_rate = 30
    if args.save_to_video:
        cap = cv2.VideoCapture(args.video_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        loaded_frames=[]
        while True:
            ret,frame=cap.read()
            if not ret:break
            loaded_frames.append(frame)
        cap.release()
        height,width=loaded_frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_output_path,fourcc,frame_rate,(width,height))


    kp_pairs = {
        "shoulder": (5,6),
        "elbow":    (7,8),
        "wrist":    (9,10),
        "knee":     (13,14),
    }


    def get_color_for_id(obj_id):
        if obj_id in BLUE_IDS:
            return BLUE_COLOR
        elif obj_id in RED_IDS:
            return RED_COLOR
        return DEFAULT_COLOR


    with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16):

        state = predictor.init_state(frames_or_path,offload_video_to_cpu=True)

        for obj_id,(bbox,_) in prompts.items():
            predictor.add_new_points_or_box(state,box=bbox,frame_idx=0,obj_id=obj_id)


        for frame_idx,object_ids,masks in predictor.propagate_in_video(state):

            if frame_idx>=len(loaded_frames): break

            bbox_to_vis={}

            for obj_id,mask in zip(object_ids,masks):
                mask = mask[0].cpu().numpy() > 0
                pts = np.argwhere(mask)

                if len(pts)==0:
                    bbox=[0,0,0,0]
                else:
                    y_min,x_min=pts.min(axis=0).tolist()
                    y_max,x_max=pts.max(axis=0).tolist()
                    bbox=[x_min,y_min,x_max-x_min,y_max-y_min]

                bbox_to_vis[obj_id]=bbox


            if args.save_to_video:
                img = loaded_frames[frame_idx]

                for obj_id,(x,y,w,h) in bbox_to_vis.items():

                    # === segmentation 컬러 overlay ===
                    mask = masks[list(object_ids).index(obj_id)][0].cpu().numpy() > 0
                    overlay_color = get_color_for_id(obj_id)
                    alpha = 0.4

                    color_layer = np.zeros_like(img)
                    color_layer[:,:,0] = overlay_color[0]
                    color_layer[:,:,1] = overlay_color[1]
                    color_layer[:,:,2] = overlay_color[2]

                    img = np.where(
                        mask[:,:,None]==1,
                        (img*(1-alpha) + color_layer*alpha).astype(np.uint8),
                        img
                    )


                    # ===== bbox tracking (center-x + bottom-y) =====
                    bbox_center_x = x + w / 2
                    bbox_bottom_y = y + h

                    record = {
                        "frame": frame_idx,
                        "obj_id": obj_id,
                        "bbox_center_x": float(bbox_center_x),
                        "bbox_bottom_y": float(bbox_bottom_y),
                    }
                    bbox_records.append(record)


                    # # ===== pose extraction =====
                    crop = img[y:y+h, x:x+w]
                    kps=None

                    if crop is None or crop.size == 0:
                        continue

                    res=pose_model(crop)
                    if res and res[0].keypoints is not None:
                        ak=res[0].keypoints.xy.cpu().numpy()
                        if len(ak)>0: kps=ak[0]

                    prec={"frame":frame_idx,"obj_id":obj_id}

                    if kps is not None:
                        for name,(L,R) in kp_pairs.items():
                            ax=(kps[L][0]+kps[R][0])/2 + x
                            ay=(kps[L][1]+kps[R][1])/2 + y
                            prec[f"{name}_x"]=float(ax)
                            prec[f"{name}_y"]=float(ay)
                    else:
                        for name in kp_pairs:
                            prec[f"{name}_x"]=np.nan
                            prec[f"{name}_y"]=np.nan

                    pose_records.append(prec)


                    # = draw bbox + id text =
                    c = get_color_for_id(obj_id)
                    cv2.rectangle(img,(x,y),(x+w,y+h),c,2)
                    cv2.putText(img,str(obj_id),(x,y-2),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,c,2)

                cv2.putText(img, f"{frame_idx+1}/{len(loaded_frames)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                out.write(img)



    if args.save_to_video:
        out.release()

    # ===== save CSV =====
    pd.DataFrame(bbox_records).to_csv(os.path.join(save_dir,"bbox_tracking.csv"),index=False)
    pd.DataFrame(pose_records).to_csv(os.path.join(save_dir,"pose4_tracking.csv"),index=False)

    print("CSV saved")


    # ===== bbox time series plot =====
    df=pd.DataFrame(bbox_records)
    for obj in df.obj_id.unique():
        sub=df[df.obj_id==obj].sort_values("frame")

        plt.figure(figsize=(12,5))
        plt.plot(sub.frame,sub.bbox_center_x,label="center_x")
        plt.plot(sub.frame,sub.bbox_bottom_y,label="bottom_y")


        plt.title(f"Obj {obj} bbox top/bottom")
        plt.xlabel("frame")
        plt.ylabel("Y pixel")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid()
        plt.savefig(os.path.join(save_dir,f"obj_{obj}_bbox_plot.png"))
        plt.close()


    # ===== pose time series plot =====
    pose_df = pd.DataFrame(pose_records)
    for obj in pose_df.obj_id.unique():
        sub = pose_df[pose_df.obj_id==obj].sort_values("frame")

        plt.figure(figsize=(12,5))
        for n in kp_pairs.keys():
            plt.plot(sub.frame,sub[f"{n}_y"],label=n)

        plt.title(f"Obj {obj} pose avg L/R")
        plt.xlabel("frame")
        plt.ylabel("Y pixel")
        plt.legend()
        plt.gca().invert_yaxis()
        plt.grid()
        plt.savefig(os.path.join(save_dir,f"obj_{obj}_pose_plot.png"))
        plt.close()

    print("plots saved")


    del predictor,state
    gc.collect()
    torch.cuda.empty_cache()



if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--video_path",required=True)
    parser.add_argument("--txt_path",required=True)
    parser.add_argument("--model_path",required=True)
    parser.add_argument("--video_output_path",default="demo.mp4")
    parser.add_argument("--save_to_video",default=True)
    args=parser.parse_args()
    main(args)
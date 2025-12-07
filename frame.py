import cv2
import os

video_path = "12_clean.mp4"
save_dir = "frames"
os.makedirs(save_dir, exist_ok=True)

# 추출하고 싶은 프레임들 (ex: 100, 120, 150)
target_frames = [64]

# ---- 비디오 오픈 ----
cap = cv2.VideoCapture(video_path)

# 전체 프레임 수 가져오기
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("총 프레임:", total_frames)

# ---- 반복하면서 원하는 프레임만 저장 ----
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 우리가 원하는 프레임이면 저장
    if frame_idx in target_frames:
        save_path = os.path.join(save_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"추출 완료: {save_path}")

    frame_idx += 1

cap.release()

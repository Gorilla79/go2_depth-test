import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import torch

# YOLO 모델 로드 (사람 탐지 및 포즈 인식)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_human = YOLO("yolov8n.pt").to(device)  # YOLOv8 for human detection
model_pose = YOLO("yolov8n-pose.pt").to(device)  # YOLOv8 Pose model
print("Pose Model loaded on CUDA")

# RealSense pipeline 초기화
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Depth 정렬
align = rs.align(rs.stream.color)

# 마우스 클릭 이벤트 콜백 함수 (깊이 측정용)
clicked_point = None
clicked_depth = None

def get_depth_at_click(event, x, y, flags, param):
    global clicked_point, clicked_depth
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

cv2.namedWindow("Color and Depth Detection")
cv2.setMouseCallback("Color and Depth Detection", get_depth_at_click)

# Define skeleton structure
skeleton_connections = [
    (5, 7), (7, 9),  # Left shoulder to left hand
    (6, 8), (8, 10), # Right shoulder to right hand
    (5, 6),           # Shoulder to shoulder
    (5, 11), (6, 12), # Shoulders to hips
    (11, 12),         # Hip to hip
    (11, 13), (13, 15), # Left hip to left foot
    (12, 14), (14, 16)  # Right hip to right foot
]

try:
    while True:
        # 프레임 수신
        frames = pipeline.wait_for_frames()
        
        # Depth를 Color 프레임에 정렬
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue

        # 프레임을 numpy 배열로 변환
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Depth 이미지 시각화
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # YOLO Pose 인식 실행
        results_human = model_human.track(color_image, persist=True, device=device)
        results_pose = model_pose.track(color_image, persist=True, device=device)

        # Pose 인식 결과 처리 및 스켈레톤 연결
        if results_pose[0].keypoints is not None:
            keypoints_list = results_pose[0].keypoints.xy.cpu().numpy()
            for keypoints in keypoints_list:
                for x, y in keypoints:
                    x, y = int(x), int(y)
                    depth = depth_frame.get_distance(x, y)
                    cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(color_image, f"{depth:.2f}m", (x+5, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # 스켈레톤 연결선 그리기
                for joint_start, joint_end in skeleton_connections:
                    if joint_start < len(keypoints) and joint_end < len(keypoints):
                        x1, y1 = keypoints[joint_start]
                        x2, y2 = keypoints[joint_end]
                        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                            cv2.line(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # 마우스 클릭 위치 깊이 정보 출력
        if clicked_point:
            click_x, click_y = clicked_point
            clicked_depth = depth_frame.get_distance(click_x, click_y)
            cv2.circle(color_image, (click_x, click_y), 10, (0, 255, 255), -1)
            cv2.putText(color_image, f"Depth: {clicked_depth:.2f}m", (click_x + 10, click_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 컬러 및 깊이 맵을 병렬로 표시
        combined_image = np.hstack((color_image, depth_colormap))
        cv2.imshow("Color and Depth Detection", combined_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            clicked_point = None

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
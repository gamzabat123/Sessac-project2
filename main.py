import cv2
import numpy as np
from ultralytics import YOLO
import requests
import math
import time
import pandas as pd
from datetime import datetime # 파일명 및 시간 기록용

# ===============================
# 필라테스 각도 계산
# ===============================
TARGET_ANGLE_MIN = 7.77
TARGET_ANGLE_MAX = 14.75

def calculate_leg_angle(kps):
    l_conf = kps[15][2]
    r_conf = kps[16][2]

    if l_conf >= r_conf:
        hip = kps[11][:2]
        ankle = kps[15][:2]
    else:
        hip = kps[12][:2]
        ankle = kps[16][:2]

    dy = -(ankle[1] - hip[1])
    dx = abs(ankle[0] - hip[0])
    angle = np.degrees(np.arctan2(dy, dx))

    return angle, hip, ankle


# ===============================
# MPU6050 센서 클라이언트
# ===============================
SENSOR_URL = "http://192.168.4.1"

def get_sensor_data():
    try:
        r = requests.get(SENSOR_URL, timeout=0.05)
        if r.status_code == 200:
            return r.json()
        return None
    except requests.exceptions.RequestException:
        return None

def compute_pelvis_from_mpu(sensor_data, threshold=3.0):
    if sensor_data is None:
        return None
    m1 = sensor_data.get("mpu1")
    m2 = sensor_data.get("mpu2")

    if not (m1 and m2):
        return None

    def roll_from_mpu(m):
        ax = m["AcX"] / 16384
        ay = m["AcY"] / 16384
        az = m["AcZ"] / 16384
        return math.degrees(math.atan2(ay, az))

    r1 = roll_from_mpu(m1)
    r2 = roll_from_mpu(m2)
    diff = r1 - r2

    if abs(diff) < threshold:
        status = "Pelvis: LEVEL"
    elif diff > 0:
        status = "Pelvis: RIGHT DOWN"
    else:
        status = "Pelvis: LEFT DOWN"

    # [수정] Log Raw Data for deeper EDA
    return {"r1": r1, "r2": r2, "diff": diff, "status": status, "m1": m1, "m2": m2}


# ===============================
# 메인 실행 (녹화 기능 포함)
# ===============================
def run_hundred_coach():
    WINDOW_NAME = "Hundred AI Coach"
    
    # [추가] 로깅 및 녹화 변수
    recorded_data_log = []
    is_recording = False
    video_writer = None 

    # 1. 윈도우 생성 및 크기 설정
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720) 

    print("⏳ YOLO 모델 로딩 중...")
    model = YOLO("yolo11s-pose.pt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        return

    print("✅ 시작!")

    # [추가] 프레임 정보 가져오기 (비디오 저장용)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        log_entry = {'Timestamp': time.time()}
        
        # 1) YOLO 추론
        results = model(frame, verbose=False, conf=0.5)
        
        # 뼈대 그림을 먼저 프레임에 입힙니다.
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            frame = results[0].plot(img=frame)
            
            kps = results[0].keypoints.data[0].cpu().numpy()

            # 다리 각도 계산
            angle, hip_xy, ankle_xy = calculate_leg_angle(kps)
            good = TARGET_ANGLE_MIN <= angle <= TARGET_ANGLE_MAX

            color = (0,255,0) if good else (0,0,255)
            status = f"{'GOOD' if good else 'BAD'} ({angle:.1f}°)"

            # [LOG] YOLO 데이터 기록
            log_entry['YOLO_Angle_deg'] = angle
            log_entry['YOLO_Status'] = status
            
            # (A) YOLO 분석 결과 표시
            cv2.putText(frame, f"YOLO Angle: {status}", (20, 50),
                         cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
        # 2) MPU6050 센서도 화면에 표시
        sensor_data = get_sensor_data()
        pelvis = compute_pelvis_from_mpu(sensor_data)

        if pelvis:
            text = f"{pelvis['status']} (R:{pelvis['r1']:.1f} L:{pelvis['r2']:.1f} Δ:{pelvis['diff']:.1f})"
            h, w, _ = frame.shape
            
            # [LOG] MPU 데이터 기록
            log_entry['MPU_Pelvis_Status'] = pelvis['status']
            log_entry['MPU_Roll_Diff'] = pelvis['diff']
            log_entry['MPU1_Roll'] = pelvis['r1']
            log_entry['MPU2_Roll'] = pelvis['r2']
            
            # (Raw Accel/Gyro data for deeper analysis)
            if 'm1' in pelvis:
                log_entry['M1_AcX'] = pelvis['m1'].get('AcX', np.nan)
                log_entry['M2_GyY'] = pelvis['m2'].get('GyY', np.nan) 
            
            # (B) MPU 센서 데이터 표시
            cv2.putText(frame, text, (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2) # Yellow

        # --- [녹화 및 로깅 처리] ---
        h, w, _ = frame.shape
        if is_recording:
            # 3A. 비디오 프레임 저장
            if video_writer is not None:
                 video_writer.write(frame)
                 
            # 3B. 데이터 로그 저장
            if len(log_entry) > 1:
                recorded_data_log.append(log_entry)
                
            # 3C. REC 표시
            cv2.putText(frame, "REC", (w - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3) # Red REC


        # 4. 화면 출력 및 키 처리
        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): # Q: 종료
            break
        elif key == ord('r'): # R: 녹화 시작/중지 토글
            is_recording = not is_recording
            
            if is_recording:
                # 녹화 시작 시 VideoWriter 초기화
                video_filename = f"Video_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
                video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))
                
                print(f"--- 🎥 비디오 녹화 시작: {video_filename} ---")
            else:
                # 녹화 중지 시 VideoWriter 해제
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    print(f"--- ⏸️ 비디오 녹화 중지. 파일이 저장됨. ---")
        # --- [녹화 처리 끝] ---


    # --- 최종 종료 처리 ---
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    # 데이터 CSV 저장
    if recorded_data_log:
        df = pd.DataFrame(recorded_data_log)
        filename = f"Data_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        print(f"\n✅ 데이터 CSV 저장 완료! 파일명: {filename} ({len(df)} 프레임)")
    else:
        print("\n⚠️ 녹화된 데이터가 없습니다.")


if __name__ == "__main__":
    run_hundred_coach()

import cv2
import time
import socket
import numpy as np
import pyrealsense2 as rs
from openpose import pyopenpose as op
from gtts import gTTS
import os
import random
import threading
from ultralytics import YOLO

# ─── 서버 연결 설정 ──────────────────────────────────────────────────
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 5000
CLIENT_ID = 'KJH'

# ============================ 서버 연결 ============================
def connect_to_server(host=SERVER_HOST, port=SERVER_PORT, client_id=CLIENT_ID):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.sendall(client_id.encode())
        print(f"[서버 연결 성공 및 ID({client_id}) 전송 완료]")
        return s
    except Exception as e:
        print("[서버 연결 실패]:", e)
        return None

server_socket = connect_to_server()
if not server_socket:
    exit()

# ====================== Mugunghwa Game Constants ====================
WATCH_SEC              = 3       # 음성 직후 움직임 감지 시간 (초)
FORWARD_SEC            = 5       # 전진 단계 시간 (초)
MOTION_THRESHOLD       = 0.5     # 평균 관절 이동 임계치
VICTORY_DISTANCE       = 0.7     # 승리 판단 거리 (m)
PROCESS_EVERY          = 10      # 프레임 처리 주기
MIN_CONFIDENCE         = 0.3     # 최소 키포인트 신뢰도
MIN_KEYPOINTS          = 5       # 최소 유효 키포인트 수
STABLE_FRAMES_REQUIRED = 3       # 안정적인 감지 프레임 수
NOISE_FILTER_WINDOW    = 3       # 노이즈 필터 윈도우 크기
MAX_SINGLE_JOINT_MOVEMENT = 25   # 단일 관절 최대 허용 이동량
IMPORTANT_JOINTS       = [0,1,2,5,8,9,12]  # 중요한 관절 인덱스

# ========================= Game State Enum ===========================
class GameState:
    WAIT_START      = 0
    MUGUNGHWA_SPEAK = 1
    WATCHING        = 2
    JUDGMENT        = 3
    FORWARD         = 4
    VICTORY         = 5
    GAME_OVER       = 6

class GameClient:
    def __init__(self):
        self.game_waiting = False  # 게임 시작 대기 상태
        
    def send_game_over(self):
        """게임 오버 메시지 전송"""
        try:
            server_socket.sendall("Game Over".encode())
            print("[서버] Game Over 메시지 전송")
        except Exception as e:
            print("[서버 전송 실패]:", e)
    
    def send_victory(self):
        """승리 메시지 전송"""
        try:
            server_socket.sendall("Victory".encode())
            print("[서버] Victory 메시지 전송")
        except Exception as e:
            print("[서버 전송 실패]:", e)

# ==================== Keypoint Processing ============================
def filter_keypoints(keypoints):
    """신뢰도가 높은 키포인트만 필터링"""
    if keypoints is None or keypoints.ndim != 3 or keypoints.shape[0] == 0:
        return None
    
    # 첫 번째 사람의 키포인트만 사용
    person_kpts = keypoints[0]
    
    # 신뢰도가 높고 중요한 관절만 선택
    filtered_kpts = []
    for joint_idx in IMPORTANT_JOINTS:
        if joint_idx < person_kpts.shape[0]:
            confidence = person_kpts[joint_idx, 2]
            if confidence > MIN_CONFIDENCE:
                filtered_kpts.append(person_kpts[joint_idx, :2])
    
    return np.array(filtered_kpts) if len(filtered_kpts) >= MIN_KEYPOINTS//2 else None

def smooth_keypoints(poses_history):
    """키포인트 스무딩 (노이즈 제거, shape 일관성 보장)"""
    n = len(poses_history)
    if n < NOISE_FILTER_WINDOW:
        return poses_history.copy()

    # 유효한 첫 프레임에서 기대 shape 추출
    first = next((p for p in poses_history if isinstance(p, np.ndarray)), None)
    if first is None:
        return poses_history.copy()
    expected_shape = first.shape  # (num_joints, 2)

    smoothed = []
    half = NOISE_FILTER_WINDOW // 2
    for i in range(n):
        # 윈도우 범위
        start = max(0, i - half)
        end   = min(n, i + half + 1)
        # shape 일치하는 것만 모으기
        window = [p for p in poses_history[start:end]
                  if isinstance(p, np.ndarray) and p.shape == expected_shape]
        if window:
            arr = np.stack(window, axis=0)            # (w, joints, 2)
            smoothed.append(arr.mean(axis=0))        # (joints, 2)
        else:
            # 유효 프레임 없으면 0으로 채우거나 원본 복사
            fallback = poses_history[i]
            if isinstance(fallback, np.ndarray) and fallback.shape == expected_shape:
                smoothed.append(fallback)
            else:
                smoothed.append(np.zeros(expected_shape))
    return smoothed

def detect_movement_improved(poses):
    """벡터화된 움직임 감지"""
    if len(poses) < 2:
        return False, "포즈 데이터 부족"
    # 1) 스무딩
    smooth = smooth_keypoints(poses)
    arr = np.array(smooth)  # (T, J, 2)
    if arr.ndim != 3:
        return False, "스무딩 실패"

    # 2) 관절별 이동량 계산 (벡터 연산)
    diffs = np.linalg.norm(arr[1:] - arr[:-1], axis=2)  # (T-1, J)
    avg_per_frame = diffs.mean(axis=1)                  # (T-1,)

    # 3) 통계값
    max_mov = avg_per_frame.max()
    mean_mov = avg_per_frame.mean()
    std_mov  = avg_per_frame.std()
    outliers = diffs > MAX_SINGLE_JOINT_MOVEMENT
    outlier_ratio = outliers.sum() / avg_per_frame.size

    print(f"[MOVEMENT] max={max_mov:.2f}, mean={mean_mov:.2f}, std={std_mov:.2f}, outlier_ratio={outlier_ratio:.2f}")

    # 4) 판단 로직
    if outlier_ratio > 0.3:
        return False, f"노이즈 과다 ({outlier_ratio:.2f})"
    if mean_mov > MOTION_THRESHOLD:
        return True, f"평균 이동량 초과 ({mean_mov:.2f})"
    if max_mov > MOTION_THRESHOLD * 1.5 and std_mov > MOTION_THRESHOLD * 0.5:
        return True, f"지속적 큰 이동 감지 ({max_mov:.2f})"
    return False, f"움직임 없음 (mean={mean_mov:.2f})"

# ======================= Async TTS ====================================
def speak_async(text, speed='normal'):
    done_event = threading.Event()
    def tts_thread():
        try:
            print(f"[VOICE] Speaking: {text} (speed: {speed})")
            tts = gTTS(text=text, lang='ko', slow=(speed=='slow'))
            fname = 'out.mp3'
            tts.save(fname)
            cmd = f"mpg123 --pitch +1 {fname}" if speed=='fast' else f"mpg123 {fname}"
            os.system(cmd)
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        finally:
            time.sleep(0.5)
            done_event.set()
            print(f"[VOICE] Completed: {text}")

    th = threading.Thread(target=tts_thread)
    th.daemon = True
    th.start()
    return th, done_event

# =================== Distance & OpenPose Utilities ====================
def get_stable_distance(depth_frame, x, y, window=1):
    depths=[]
    w,h=depth_frame.get_width(), depth_frame.get_height()
    for dx in range(-window,window+1):
        for dy in range(-window,window+1):
            nx,ny=x+dx,y+dy
            if 0<=nx<w and 0<=ny<h:
                d=depth_frame.get_distance(nx,ny)
                if 0.1<d<5.0:
                    depths.append(d)
    return float(np.mean(depths)) if depths else 0.0


def safe_openpose(wrapper, img):
    try:
        datum=op.Datum()
        datum.cvInputData=img
        vdat=op.VectorDatum()
        vdat.append(datum)
        wrapper.emplaceAndPop(vdat)
        return datum.cvOutputData, datum.poseKeypoints
    except Exception as e:
        print(f"[ERROR] OpenPose: {e}")
        return img, None


def is_person_detected(keypoints):
    """키포인트 신뢰도를 기반으로 사람이 제대로 감지되었는지 확인"""
    if keypoints is None or keypoints.ndim != 3 or keypoints.shape[0] == 0:
        return False
    
    # 첫 번째 사람의 키포인트 신뢰도 확인
    confidences = keypoints[0, :, 2]  # 세 번째 차원이 신뢰도
    high_conf_points = np.sum(confidences > MIN_CONFIDENCE)
    
    # 중요한 관절들의 신뢰도 추가 확인
    important_detected = sum(1 for joint_idx in IMPORTANT_JOINTS 
                           if joint_idx < len(confidences) and confidences[joint_idx] > MIN_CONFIDENCE)
    
    if high_conf_points >= MIN_KEYPOINTS and important_detected >= len(IMPORTANT_JOINTS)//2:
        print(f"[DEBUG] Person detected: {high_conf_points} high conf, {important_detected} important joints")
        return True
    else:
        print(f"[DEBUG] Low confidence: {high_conf_points} total, {important_detected} important joints")
        return False

# ================== Mugunghwa Game Logic ==============================
def run_mugunghwa_game():
    game_client = GameClient()
    print("[CAMERA] Initializing RealSense...")
    pipeline=rs.pipeline()
    cfg=rs.config()
    cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
    cfg.enable_stream(rs.stream.depth,640,480,rs.format.z16,30)
    pipeline.start(cfg)
    print("[CAMERA] Initialized")

    print("[OpenPose] Initializing...")
    params={'model_folder':'models/','model_pose':'BODY_25','net_resolution':'320x256','render_threshold':0.01,'number_people_max':1}
    opw=op.WrapperPython(); opw.configure(params); opw.start()
    print("[OpenPose] Initialized")

    win='Mugunghwa Game'
    cv2.namedWindow(win)
    font=cv2.FONT_HERSHEY_SIMPLEX
    small_font = cv2.FONT_HERSHEY_PLAIN

    # Initialize game state (1번 파일과 같은 로직)
    state = GameState.MUGUNGHWA_SPEAK
    poses = []
    filtered_poses = []
    last_out = None
    person_detected_during_watch = False
    speech_done = None
    stable_detection_count = 0

    print("[GAME] 게임 시작!")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            img_raw = np.asanyarray(color_frame.get_data())

            # --- MUGUNGHWA_SPEAK: Start the announcement ---
            if state == GameState.MUGUNGHWA_SPEAK:
                speed = random.choice(['fast', 'normal', 'slow'])
                print(f"[GAME] Starting mugunghwa speech (speed: {speed})")
                _, speech_done = speak_async('무궁화 꽃이 피었습니다', speed)
                speech_done.wait()
                time.sleep(2)
                frame_count = 0
                start_t = time.time()
                poses.clear()
                filtered_poses.clear()
                last_out = None
                person_detected_during_watch = False
                stable_detection_count = 0
                state = GameState.WATCHING

            # --- WATCHING: Monitor movement for a fixed duration ---
            elif state == GameState.WATCHING:
                elapsed = time.time() - start_t
                speech_finished = speech_done.is_set() if speech_done else True

                if elapsed < WATCH_SEC:
                    frame_count += 1
                    if frame_count % PROCESS_EVERY == 0:
                        out, kpts = safe_openpose(opw, img_raw)
                        last_out = out.copy()

                        if is_person_detected(kpts):
                            stable_detection_count += 1
                            filtered_kpts = filter_keypoints(kpts)
                            if filtered_kpts is not None:
                                person_detected_during_watch = True
                                poses.append(kpts[0][:, :2])
                                filtered_poses.append(filtered_kpts)
                                print(f"[DEBUG] Stable detection: {stable_detection_count}, Filtered poses: {len(filtered_poses)}")
                        else:
                            stable_detection_count = max(0, stable_detection_count - 1)

                    display_frame = last_out if last_out is not None else img_raw
                    if not speech_finished:
                        cv2.putText(display_frame, "Mugunghwa kkochi pieosseumnida...", (10, 30), small_font, 1.0, (0, 0, 255), 1)
                    else:
                        cv2.putText(display_frame, "Don't move!", (10, 30), small_font, 1.0, (0, 0, 255), 1)

                    if person_detected_during_watch:
                        cv2.putText(display_frame, f"Detected - Stable: {stable_detection_count}", (10, 50), small_font, 0.8, (0, 255, 0), 1)
                    else:
                        cv2.putText(display_frame, "Waiting for person detection...", (10, 50), small_font, 0.8, (255, 0, 0), 1)

                    remaining_time = max(0, WATCH_SEC - elapsed)
                    cv2.putText(display_frame, f"Time: {remaining_time:.1f}s", (10, 70), small_font, 0.8, (0, 255, 255), 1)
                    cv2.imshow(win, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return "exit"
                else:
                    print(f"[GAME] Watch phase completed. Stable detections: {stable_detection_count}, Filtered poses: {len(filtered_poses)}")
                    state = GameState.JUDGMENT

            # --- JUDGMENT: 개선된 움직임 판단 ---
            elif state == GameState.JUDGMENT:
                if not person_detected_during_watch or len(filtered_poses) < 2 or stable_detection_count < STABLE_FRAMES_REQUIRED:
                    print(f"[GAME] Insufficient data - Person: {person_detected_during_watch}, Poses: {len(filtered_poses)}, Stable: {stable_detection_count}")
                    no_person_messages = [
                        '안정적인 감지가 어려웠습니다. 다시 시작합니다.',
                        '카메라 앞에 똑바로 서서 다시 해주세요.',
                        '조명을 확인하고 다시 시도해주세요.',
                        '화면에 전신이 나오도록 해서 다시 해주세요.'
                    ]
                    no_person_msg = random.choice(no_person_messages)
                    display = img_raw.copy()
                    cv2.putText(display, "Detection unstable!", (50, 50), small_font, 1.0, (0, 0, 255), 1)
                    cv2.putText(display, "Restarting game...", (50, 80), small_font, 1.0, (0, 0, 255), 1)
                    cv2.imshow(win, display)
                    _, done = speak_async(no_person_msg, 'normal')
                    done.wait()
                    cv2.waitKey(1000)
                    
                    # 다시 무궁화 단계로 돌아가기
                    state = GameState.MUGUNGHWA_SPEAK
                else:
                    moved, reason = detect_movement_improved(filtered_poses)
                    print(f"[MOVEMENT] 판단 결과: {moved}, 이유: {reason}")
                    
                    if moved:
                        print("[GAME] Movement detected - Game Over")
                        _, done = speak_async('움직임 감지! 게임 오버.', 'normal')
                        display = img_raw.copy()
                        cv2.putText(display, "Movement detected!", (50, 30), small_font, 1.0, (0, 0, 255), 1)
                        cv2.putText(display, "Game Over", (50, 60), font, 1.0, (0, 0, 255), 2)
                        cv2.putText(display, reason, (50, 90), small_font, 0.6, (0, 0, 255), 1)
                        cv2.imshow(win, display)
                        done.wait()
                        
                        # 서버에 게임 오버 메시지 전송
                        game_client.send_game_over()
                        
                        cv2.waitKey(2000)
                        return "exit"
                    else:
                        print("[GAME] No movement detected - Success!")
                        success_messages = ['잘했어요!', '훌륭해요!', '완벽해요!', '멋져요!', '대단해요!']
                        success_msg = random.choice(success_messages)
                        display = img_raw.copy()
                        cv2.putText(display, "Success!", (50, 30), small_font, 1.0, (0, 255, 0), 1)
                        cv2.putText(display, "No movement detected", (50, 60), small_font, 1.0, (0, 255, 0), 1)
                        cv2.putText(display, reason, (50, 90), small_font, 0.6, (0, 255, 0), 1)
                        cv2.imshow(win, display)
                        _, done = speak_async(success_msg, 'normal')
                        done.wait()
                        _, done2 = speak_async('앞으로 이동하세요.', 'normal')
                        done2.wait()
                        state = GameState.FORWARD
                        ok = 0
                        frame_count = 0
                        any_detected = False
                        start_t = time.time()
                        last_out = None

            # --- FORWARD: Check if the player moves close enough ---
            elif state == GameState.FORWARD:
                elapsed = time.time() - start_t
                if elapsed < FORWARD_SEC:
                    frame_count += 1
                    if frame_count % PROCESS_EVERY == 0:
                        out, kpts = safe_openpose(opw, img_raw)
                        last_out = out.copy()
                        if is_person_detected(kpts):
                            any_detected = True
                            p = kpts[0]
                            x = int((p[1][0] + p[8][0]) / 2)
                            y = int((p[1][1] + p[8][1]) / 2)
                            dist = get_stable_distance(depth_frame, x, y)
                            cv2.putText(last_out, f"Distance: {dist:.2f}m", (10, 60), small_font, 1.0, (255, 255, 0), 1)
                            if 0 < dist < VICTORY_DISTANCE:
                                ok += 1
                                cv2.putText(last_out, f"Close enough! ({ok}/2)", (10, 80), small_font, 1.0, (0, 255, 0), 1)
                                if ok >= 2:
                                    print("[GAME] Victory condition met!")
                                    state = GameState.VICTORY
                                    continue
                            else:
                                ok = 0
                                cv2.putText(last_out, "Move closer to camera", (10, 80), small_font, 1.0, (0, 255, 255), 1)
                    display_frame = last_out if last_out is not None else img_raw
                    cv2.putText(display_frame, "Move forward to win!", (10, 30), small_font, 1.0, (0, 255, 0), 1)
                    remaining_time = max(0, FORWARD_SEC - elapsed)
                    cv2.putText(display_frame, f"Time: {remaining_time:.1f}s", (10, 100), small_font, 0.8, (0, 255, 255), 1)
                    cv2.imshow(win, display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return "exit"
                else:
                    if any_detected:
                        print("[GAME] Time's up - Starting next round")
                        continue_messages = ['다시 시작합니다!', '한 번 더!', '계속 해봅시다!', '포기하지 마세요!']
                        continue_msg = random.choice(continue_messages)
                        display = img_raw.copy()
                        cv2.putText(display, "Time's up!", (50, 30), small_font, 1.0, (255, 255, 0), 1)
                        cv2.putText(display, "Let's try again!", (50, 60), small_font, 0.8, (255, 255, 0), 1)
                        cv2.imshow(win, display)
                        _, done = speak_async(continue_msg, 'normal')
                        done.wait()
                        cv2.waitKey(1000)
                        
                        # 다시 무궁화 단계로 돌아가기
                        state = GameState.MUGUNGHWA_SPEAK
                    else:
                        print("[GAME] No person detected during forward phase - restarting")
                        forward_no_person_messages = [
                            '사람이 보이지 않습니다. 다시 시작합니다.',
                            '카메라 범위에 들어와서 다시 해주세요.',
                            '화면에 나타나서 게임을 계속해 주세요.'
                        ]
                        forward_msg = random.choice(forward_no_person_messages)
                        display = img_raw.copy()
                        cv2.putText(display, "Person not detected", (50, 50), small_font, 1.0, (255, 0, 0), 1)
                        cv2.putText(display, "Restarting game...", (50, 80), small_font, 1.0, (255, 0, 0), 1)
                        cv2.imshow(win, display)
                        _, done = speak_async(forward_msg, 'normal')
                        done.wait()
                        cv2.waitKey(1000)
                        
                        # 다시 무궁화 단계로 돌아가기
                        state = GameState.MUGUNGHWA_SPEAK

            # --- VICTORY: 승리 처리 ---
            elif state == GameState.VICTORY:
                print("[GAME] Victory!")
                _, done = speak_async('승리했습니다!', 'normal')
                display = img_raw.copy()
                cv2.putText(display, "Victory!", (50, 90), font, 1.0, (0, 255, 0), 2)
                cv2.putText(display, "Congratulations!", (50, 120), small_font, 1.0, (0, 255, 0), 1)
                cv2.imshow(win, display)
                done.wait()
                
                # 서버에 승리 메시지 전송
                game_client.send_victory()
                
                cv2.waitKey(2000)
                return "exit"

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

# ==================== Animal Detection (unchanged) ====================
def run_animal_detection():

    # 지원하는 동물 목록과 설명
    animals = {"cat": "고양이는 우아하고 독립적인 반려동물입니다. 빠른 반사 신경과 유연한 몸을 가지고 있습니다.",
    "dog": "개는 충성스럽고 사람을 좋아하는 동물입니다. 훈련이 가능하며 다양한 작업을 수행할 수 있습니다.",
    "buffalo": "버팔로는 강한 체격을 가진 초식 동물입니다. 북미 초원에서 서식했으며 무리를 지어 생활합니다.",
    "elephant": "코끼리는 지구상에서 가장 큰 육상 동물입니다. 뛰어난 기억력과 지능을 가지고 있습니다.",
    "zebra": "얼룩말은 독특한 흑백 줄무늬를 가진 말의 일종입니다. 아프리카 초원에 서식합니다.",
    "rhino": "코뿔소는 두꺼운 피부와 뿔을 가진 대형 초식 동물입니다. 멸종 위기에 처해 있습니다."}

    print("동물 감지 모드 시작")
    
    try:
        server_socket.sendall("동물감지".encode())
        print("[서버] 동물감지 모드 시작 전송 완료")
    except Exception as e:
        print("[서버 전송 실패]:", e)

    try:
        model = YOLO('../models/yolo/best.pt')
        print("[YOLO] 모델 로드 완료")
    except Exception as e:
        print("[YOLO] 모델 로드 실패:", e)
        return "exit"

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    last_print_time = 0
    print_interval = 10

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            results = model(frame, imgsz=640, conf=0.5, verbose=False)[0]
            names = model.names

            if len(results.boxes) > 0:
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = names[cls_id]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                now = time.time()
                if now - last_print_time > print_interval:
                    print(f"감지된 동물: {name}")
                    try:
                        speak_async(animals[name])
                    except Exception as e:
                        print("[서버 전송 실패]:", e)
                    last_print_time = now

            cv2.imshow("Animal Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                return "mugunghwa"

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        return "exit"

# ========================= Main Loop =================================
def main():
    print("서버 명령 대기 중...")
    mode='exit'
    try:
        while True:
            data=server_socket.recv(1024).decode().strip()
            if data=='무궁화':
                mode='mugunghwa'
            elif data=='동물감지':
                mode='animal_detection'
            while mode!='exit':
                if mode=='mugunghwa': mode=run_mugunghwa_game()
                elif mode=='animal_detection': mode=run_animal_detection()
    except Exception as e:
        print("[서버 오류]:",e); return

    print("프로그램 종료")
    server_socket.close()

if __name__=='__main__':
    main()
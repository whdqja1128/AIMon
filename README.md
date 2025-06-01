# 🧠 AI Kids Interactive Game Project

OpenPose와 YOLOv8을 활용하여 제작된 어린이용 인터랙티브 게임입니다.  
음성 명령을 통해 게임을 제어하며, 실시간 사람 자세 추적 및 동물 인식을 통해 아이들의 흥미를 유도합니다.

---

## 🎮 주요 게임 구성

### 1. 무궁화 꽃이 피었습니다 (OpenPose 기반)
- 사람의 움직임을 추적하여 **움직이면 게임 오버**
- OpenPose로 관절 추적 및 움직임 분석
- TTS로 “무궁화 꽃이 피었습니다” 자동 음성 출력

### 2. 동물 알려주기 (YOLOv8 기반)
- 실시간 카메라로 동물 인식 (고양이, 개, 코끼리 등)
- 감지된 동물에 대해 TTS로 설명 제공

---

## 🗂 프로젝트 구성

```plaintext
ai_kids_game_project/
├── client/
│   ├── game_client.py           # OpenPose + YOLOv8 게임 통합 클라이언트
│   ├── voice_control_client.py  # 음성으로 명령 제어
│   └── requirements.txt
│
├── server/
│   └── game_server.c            # TCP 기반 게임 서버 (C언어)
│
├── models/                      # OpenPose 모델 (직접 다운로드 필요)
├── environment.yml             # (conda 사용자용 가상환경 설정)
├── .gitignore
└── README.md
```

---

## ⚙️ 설치 방법

### 1. 시스템 패키지 (Ubuntu)
```bash
sudo apt-get install mpg123
```

### 2. 가상환경 설치 (conda)
```bash
conda env create -f environment.yml
conda activate ai_game_env
```

### 3. pip 기반 패키지 설치
```bash
pip install -r client/requirements.txt
```

---

## 📦 OpenPose 모델 설치

OpenPose Python API를 사용하기 위해 `models/` 폴더가 필요합니다.  
아래 링크에서 모델을 다운로드 후 다음 구조로 배치하세요:

```
models/
└── pose/
    └── body_25/
        └── pose_iter_584000.caffemodel
```

> 📌 [OpenPose 모델 다운로드 공식 가이드](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation/0_index.md#models-download)

---

## 🚀 실행 방법

### 서버 실행 (C 코드 컴파일)
```bash
cd server
make
./server
```

### 클라이언트 실행
```bash
# 게임 실행
cd client
python game_client.py

# 음성제어
python voice_control_client.py
```

---

## 🛠 사용 기술 스택

- Python (OpenCV, NumPy, gTTS, pyrealsense2, YOLOv8, SpeechRecognition)
- C 언어 (TCP Socket)
- OpenPose (C++ + Python API)

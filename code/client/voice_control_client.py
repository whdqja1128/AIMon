import socket
import speech_recognition as sr
from gtts import gTTS
import os
import time
import random

# 음성 인식 및 TTS 엔진 초기화
recognizer = sr.Recognizer()

# TTS 음성 출력 함수
def speak(text, speed='normal'):
    tts = gTTS(text=text, lang='ko', slow=(speed == 'slow'))
    tts.save("output.mp3")

    if speed == 'fast':
        os.system("mpg123 --pitch +1 output.mp3")  # pitch를 높여서 빠르게
    elif speed == 'slow':
        os.system("mpg123 output.mp3")  # gTTS 자체 slow 옵션으로 느리게
    else:
        os.system("mpg123 output.mp3")

# "아이몬"를 감지할 때까지 대기
def wait_for_activation():
    with sr.Microphone() as source:
        print("\n🟢 대기 중... '아이몬' 라고 말하면 음성 입력 시작!")
        recognizer.adjust_for_ambient_noise(source)  # 주변 소음 보정
        while True:
            try:
                audio = recognizer.listen(source, timeout=5)
                text = recognizer.recognize_google(audio, language="ko-KR").strip()
                print(f"📝 인식된 문장: {text}")
                if "아이몬" in text:
                    speak("네!")  # "네!"라고 응답
                    print("✅ 음성 입력 시작!")
                    return True
            except sr.WaitTimeoutError:
                continue  # 타임아웃 시 다시 대기
            except sr.UnknownValueError:
                continue  # 인식 불가 시 다시 대기
            except sr.RequestError:
                print("❌ STT 서비스 오류 발생")
                return False

# 음성을 인식하는 함수
def recognize_speech():
    with sr.Microphone() as source:
        print("🎤 음성을 입력하세요...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=5)
            text = recognizer.recognize_google(audio, language="ko-KR").strip()
            print(f"📝 인식된 문장: {text}")
            return text
        except sr.WaitTimeoutError:
            print("⏳ 음성이 감지되지 않았습니다.")
            return None
        except sr.UnknownValueError:
            print("🤷‍♂️ 음성을 인식할 수 없습니다.")
            return None
        except sr.RequestError:
            print("❌ STT 서비스 오류 발생")
            return None

def reconnect_to_server(max_attempts=5, retry_delay=5):
    """
    서버 연결이 끊어졌을 때 재연결을 시도하는 함수
    
    Args:
        max_attempts (int): 최대 재시도 횟수
        retry_delay (int): 재시도 간격(초)
    
    Returns:
        socket or None: 연결된 소켓 객체 또는 실패 시 None
    """
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        print(f"서버 재연결 시도 {attempt}/{max_attempts}...")
        
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect(('10.10.14.6', 5000))
            
            # 장치 이름만 전송 (비밀번호 없음)
            device_id = 'USR_AI'
            client_socket.send(device_id.encode())
            
            # 서버 응답 받기
            response = client_socket.recv(1024).decode()
            print(f"서버 응답: {response}")
            
            if "Error" in response or "Authentication Error" in response:
                print("서버 인증 실패")
                client_socket.close()
                time.sleep(retry_delay)
                continue
                
            print("서버 재연결 성공!")
            return client_socket
            
        except Exception as e:
            print(f"재연결 시도 중 오류 발생: {e}")
            time.sleep(retry_delay)
            
    print(f"{max_attempts}번 재시도했으나 서버에 연결할 수 없습니다.")
    return None

def client_program():
    host = '127.0.0.1'  # 서버 IP 주소
    port = 5000            # 서버 포트 번호

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # 아이디만 서버로 전송
    device_id = 'Jong'
    client_socket.send(device_id.encode())

    # 서버로부터 응답 받기 (아이디 인증 결과)
    response = client_socket.recv(2048).decode().strip()
    print("서버 응답:", response)

    while True:
        if wait_for_activation(): 
            user_input = recognize_speech()  # 음성 입력 받기
            if user_input is None:
                continue  # 음성 인식 실패 시 다시 대기

            elif '무궁화' in user_input:
                client_socket.send('무궁화'.encode())


            elif '동물감지' in user_input or '동물 감지' in user_input:
                client_socket.send('동물감지'.encode())
                speak("동물 감지 모드를 시작합니다.")

            if user_input.lower() == 'exit':
                print("연결을 종료합니다...")
                speak("연결을 종료합니다.")  # TTS로 안내
                client_socket.send('exit'.encode())
                break

    client_socket.close()

if __name__ == '__main__':
    client_program()
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <pthread.h>

#define PORT 5000
#define BUFFER_SIZE 2048
#define MAX_CLIENTS 100

typedef struct {
    int socket;
    char id[BUFFER_SIZE];
    int game_in_progress;  // 게임 진행 중 여부 플래그
} ClientInfo;

// 동물 설명 정보를 저장하는 구조체
typedef struct {
    char name[50];
    char description[256];
} AnimalInfo;

// 지원하는 동물 목록과 설명
AnimalInfo animals[] = {
    {"cat", "고양이는 우아하고 독립적인 반려동물입니다. 빠른 반사 신경과 유연한 몸을 가지고 있습니다."},
    {"dog", "개는 충성스럽고 사람을 좋아하는 동물입니다. 훈련이 가능하며 다양한 작업을 수행할 수 있습니다."},
    {"buffalo", "버팔로는 강한 체격을 가진 초식 동물입니다. 북미 초원에서 서식했으며 무리를 지어 생활합니다."},
    {"elephant", "코끼리는 지구상에서 가장 큰 육상 동물입니다. 뛰어난 기억력과 지능을 가지고 있습니다."},
    {"zebra", "얼룩말은 독특한 흑백 줄무늬를 가진 말의 일종입니다. 아프리카 초원에 서식합니다."},
    {"rhino", "코뿔소는 두꺼운 피부와 뿔을 가진 대형 초식 동물입니다. 멸종 위기에 처해 있습니다."}
};
int animal_count = 6; // 동물 목록 크기

ClientInfo clients[MAX_CLIENTS];
int client_count = 0;
pthread_mutex_t clients_mutex = PTHREAD_MUTEX_INITIALIZER;

// 게임 진행 상태
int game_in_progress = 0;

// 동물 정보 찾기 함수
AnimalInfo* find_animal_info(const char* animal_name) {
    for (int i = 0; i < animal_count; i++) {
        if (strcmp(animals[i].name, animal_name) == 0) {
            return &animals[i];
        }
    }
    return NULL;
}

// 클라이언트 ID로 클라이언트 찾기
int find_client_by_id(const char* client_id) {
    for (int i = 0; i < client_count; ++i) {
        if (strcmp(clients[i].id, client_id) == 0) {
            return i;
        }
    }
    return -1;
}

// 특정 클라이언트에게 메시지 전송
void send_to_client_by_id(const char* target_id, const char* message) {
    pthread_mutex_lock(&clients_mutex);
    int client_index = find_client_by_id(target_id);
    if (client_index >= 0) {
        ssize_t sent_bytes = write(clients[client_index].socket, message, strlen(message));
        if (sent_bytes <= 0) {
            printf("[%s]에게 메시지 전송 실패\n", target_id);
        } else {
            printf("[%s] 에게 메시지 전송: %s\n", target_id, message);
        }
    } else {
        printf("[%s] 클라이언트를 찾을 수 없음\n", target_id);
    }
    pthread_mutex_unlock(&clients_mutex);
}

// 음성인식 클라이언트에게 상태 업데이트
void update_voice_recognition_client(const char* status) {
    send_to_client_by_id("Jong", status);
}

// 게임 클라이언트에게 명령 전송
void send_command_to_game_client(const char* command) {
    send_to_client_by_id("KJH", command);
}

// 모든 게임 클라이언트에게 메시지 전송
void send_to_game_clients(const char* message) {
    pthread_mutex_lock(&clients_mutex);
    for (int i = 0; i < client_count; ++i) {
        // KJH는 게임 클라이언트 ID
        if (strcmp(clients[i].id, "KJH") == 0) {
            ssize_t sent_bytes = write(clients[i].socket, message, strlen(message));
            if (sent_bytes <= 0) {
                printf("[%s]에게 메시지 전송 실패\n", clients[i].id);
            } else {
                printf("[%s] 에게 메시지 전송: %s\n", clients[i].id, message);
            }
        }
    }
    pthread_mutex_unlock(&clients_mutex);
}

// 게임 상태 설정
void set_game_status(int status) {
    pthread_mutex_lock(&clients_mutex);
    game_in_progress = status;
    pthread_mutex_unlock(&clients_mutex);
}

// 게임 상태 가져오기
int get_game_status() {
    int status;
    pthread_mutex_lock(&clients_mutex);
    status = game_in_progress;
    pthread_mutex_unlock(&clients_mutex);
    return status;
}

void* handle_client(void* arg) {
    int client_socket = *(int*)arg;
    free(arg);

    char buffer[BUFFER_SIZE];
    memset(buffer, 0, BUFFER_SIZE);

    // 클라이언트 ID 수신
    ssize_t bytes = read(client_socket, buffer, BUFFER_SIZE);
    if (bytes <= 0) {
        printf("클라이언트 인증 수신 실패\n");
        close(client_socket);
        return NULL;
    }
    buffer[strcspn(buffer, "\r\n")] = 0;  // 줄바꿈 제거
    
    // 클라이언트 ID 저장
    char client_id[BUFFER_SIZE];
    strncpy(client_id, buffer, BUFFER_SIZE);
    printf("클라이언트 접속됨 (ID: %s)\n", client_id);

    // 클라이언트 목록에 저장
    pthread_mutex_lock(&clients_mutex);
    if (client_count < MAX_CLIENTS) {
        clients[client_count].socket = client_socket;
        strncpy(clients[client_count].id, client_id, BUFFER_SIZE);
        clients[client_count].game_in_progress = 0;
        client_count++;
    }
    pthread_mutex_unlock(&clients_mutex);

    // 인증 응답 전송
    const char* response = "인증 성공";
    ssize_t sent = write(client_socket, response, strlen(response));
    if (sent <= 0) {
        printf("클라이언트에게 응답 전송 실패\n");
        close(client_socket);
        return NULL;
    }

    // 클라이언트 유형별 초기 처리
    if (strcmp(client_id, "Jong") == 0) {
        // 음성인식 클라이언트 접속 시 게임 클라이언트에게 알림
        printf("음성인식 클라이언트 접속\n");
    } else if (strcmp(client_id, "KJH") == 0) {
        // 게임 클라이언트 접속 시 초기화
        printf("게임 클라이언트 접속\n");
    }

    while (1) {
        memset(buffer, 0, BUFFER_SIZE);
        ssize_t bytes_received = read(client_socket, buffer, BUFFER_SIZE);
        if (bytes_received <= 0) {
            printf("클라이언트 연결 종료 (ID: %s)\n", client_id);
            break;
        }

        buffer[strcspn(buffer, "\r\n")] = 0;
        printf("[%s] 수신된 메시지: %s\n", client_id, buffer);

        // 음성인식 클라이언트에서 온 메시지 처리
        if (strcmp(client_id, "Jong") == 0) {
            if (strcmp(buffer, "아이몬") == 0) {
                printf("아이몬 명령 감지됨!\n");
                send_to_game_clients("아이몬");
            }
            else if (strcmp(buffer, "무궁화") == 0) {
                printf("무궁화 명령 감지됨!\n");
                
                // 게임 상태를 활성으로 변경
                set_game_status(1);
                
                // 음성인식 클라이언트의 게임 상태 변경
                int idx = find_client_by_id("Jong");
                if (idx >= 0) {
                    pthread_mutex_lock(&clients_mutex);
                    clients[idx].game_in_progress = 1;
                    pthread_mutex_unlock(&clients_mutex);
                }
                
                // 게임 클라이언트에게 무궁화 명령 전송
                send_to_game_clients("무궁화");
                
                printf("게임 상태가 활성으로 변경되었습니다. 음성인식 일시 중지.\n");
            }
            else if (strcmp(buffer, "동물감지") == 0) {
                printf("동물 명령 감지됨!\n");
                send_to_game_clients("동물감지");
            }
        }
        // 게임 클라이언트에서 온 메시지 처리
        else if (strcmp(client_id, "KJH") == 0) {
            if (strcmp(buffer, "Game Over") == 0) {
                printf("게임 오버 이벤트 감지됨!\n");
                
                // 게임 상태를 비활성으로 변경
                set_game_status(0);
                
                // 음성인식 클라이언트의 게임 상태 변경
                int idx = find_client_by_id("Jong");
                if (idx >= 0) {
                    pthread_mutex_lock(&clients_mutex);
                    clients[idx].game_in_progress = 0;
                    pthread_mutex_unlock(&clients_mutex);
                }
                
                // 음성인식 클라이언트에게 게임 오버 메시지 전송
                send_to_client_by_id("Jong", "게임 오버");
                
                printf("게임 상태가 비활성으로 변경되었습니다. 음성인식 재개.\n");
            }
            else if (strcmp(buffer, "Victory") == 0) {
                printf("승리 이벤트 감지됨!\n");
                
                // 게임 상태를 비활성으로 변경
                set_game_status(0);
                
                // 음성인식 클라이언트의 게임 상태 변경
                int idx = find_client_by_id("Jong");
                if (idx >= 0) {
                    pthread_mutex_lock(&clients_mutex);
                    clients[idx].game_in_progress = 0;
                    pthread_mutex_unlock(&clients_mutex);
                }
                
                // 음성인식 클라이언트에게 승리 메시지 전송
                send_to_client_by_id("Jong", "승리");
                
                printf("게임 상태가 비활성으로 변경되었습니다. 음성인식 재개.\n");
            }
            else if (strcmp(buffer, "동물감지") == 0) {
                printf("동물 감지 모드 시작 요청 감지됨!\n");
                // 이미 동물감지 모드가 시작됨을 확인
            }
            else {
                // 동물 이름이 수신된 경우 처리
                AnimalInfo* animal = find_animal_info(buffer);
                if (animal != NULL) {
                    printf("동물 감지됨: %s\n", animal->name);
                    
                    // 응답 메시지 구성 (동물 이름과 설명)
                    char response[BUFFER_SIZE];
                    snprintf(response, BUFFER_SIZE, "%s|%s", animal->name, animal->description);
                    
                    // 응답 전송
                    send_to_client_by_id("KJH", response);
                    
                    // 음성인식 클라이언트에게도 동물 감지 결과 전송
                    send_to_client_by_id("Jong", response);
                }
            }
        }

        if (strcmp(buffer, "exit") == 0) {
            printf("클라이언트 종료 요청 (ID: %s)\n", client_id);
            break;
        }
    }

    close(client_socket);

    // 연결 종료된 클라이언트를 목록에서 제거
    pthread_mutex_lock(&clients_mutex);
    for (int i = 0; i < client_count; ++i) {
        if (clients[i].socket == client_socket) {
            // 뒤에 있는 클라이언트를 앞으로 당김
            for (int j = i; j < client_count - 1; ++j) {
                clients[j] = clients[j + 1];
            }
            client_count--;
            break;
        }
    }
    pthread_mutex_unlock(&clients_mutex);

    return NULL;
}

int main() {
    int server_socket, *client_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);

    server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        perror("소켓 생성 실패");
        exit(EXIT_FAILURE);
    }

    // 소켓 옵션 설정 (포트 재사용)
    int opt = 1;
    if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        perror("소켓 옵션 설정 실패");
        exit(EXIT_FAILURE);
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        perror("바인드 실패");
        exit(EXIT_FAILURE);
    }

    if (listen(server_socket, 5) < 0) {
        perror("리슨 실패");
        exit(EXIT_FAILURE);
    }

    printf("서버 시작됨. 포트: %d\n", PORT);

    while (1) {
        client_socket = malloc(sizeof(int));
        if (client_socket == NULL) {
            perror("메모리 할당 실패");
            continue;
        }

        *client_socket = accept(server_socket, (struct sockaddr*)&client_addr, &client_len);
        if (*client_socket < 0) {
            perror("클라이언트 연결 수락 실패");
            free(client_socket);
            continue;
        }

        pthread_t thread_id;
        if (pthread_create(&thread_id, NULL, handle_client, client_socket) != 0) {
            perror("스레드 생성 실패");
            close(*client_socket);
            free(client_socket);
            continue;
        }
        pthread_detach(thread_id);
    }

    close(server_socket);
    return 0;
}
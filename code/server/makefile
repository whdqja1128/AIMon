# Makefile for server

# 컴파일러
CC = gcc

# 컴파일 옵션
CFLAGS = -Wall -O2

# 실행 파일 이름
TARGET = server

# 소스 파일
SRCS = server.c

# 기본 규칙
all: $(TARGET)

# 실행 파일 빌드
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

# 실행 파일 삭제
clean:
	rm -f $(TARGET)

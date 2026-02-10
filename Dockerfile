# 1. Base Image 변경 (안정적인 Bookworm 버전 사용)
FROM python:3.11-slim-bookworm

# 2. 필수 패키지 설치
# (software-properties-common 제거함)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*


# 2-1. 로컬 경로의 임베딩 모델 컨테이너 내부로 복사
COPY data/model_cache/bge-m3 /app/models/bge-m3

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 전체 복사
COPY . .

# 6. Streamlit 포트 노출
EXPOSE 8501

# 7. 헬스체크
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 8. 실행 명령
ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
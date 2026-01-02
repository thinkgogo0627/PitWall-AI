# 1. Base Image: 가볍고 안정적인 Python 3.10 Slim 버전 사용
FROM python:3.10-slim

# 2. System Setting & Font Install
# - build-essential: 일부 파이썬 라이브러리 컴파일용
# - fonts-nanum: 한글 폰트 설치 (필수)
# - curl: 헬스체크용 (선택)
RUN apt-get update && apt-get install -y \
    build-essential \
    fonts-nanum \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 3. Matplotlib 캐시 삭제 (폰트 갱신을 위해)
RUN rm -rf /root/.cache/matplotlib

# 4. Work Directory 설정
WORKDIR /app

# 5. Dependencies Install (캐싱 효율을 위해 requirements 먼저 복사)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Source Code Copy (나머지 코드 복사)
COPY . .

# 7. Environment Variables (인코딩 & 경로 설정)
ENV PYTHONPATH=/app
ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 8. Port Expose (Streamlit Default)
EXPOSE 8501

# 9. Healthcheck (컨테이너가 살아있는지 확인)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 10. Run Command
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0"]
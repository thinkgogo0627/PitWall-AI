# 공식 Airflow 이미지 기반 (Python 3.10 ~ 3.11 권장)
FROM apache/airflow:2.8.1-python3.10

USER root

# (선택) 시스템 레벨 의존성 설치 (예: gcc, git 등 필요하면 추가)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         build-essential \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

# 우리가 만든 requirements.txt 복사 및 설치
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
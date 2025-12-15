from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
from dotenv import load_dotenv

# --- [환경 설정] ---
# 1. 프로젝트 루트 경로를 찾아서 시스템 경로에 추가 (모듈 import용)
# Airflow가 실행될 때 이 파일의 위치를 기준으로 프로젝트 루트를 찾음
dag_file_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(dag_file_dir, '../')) # dags 폴더의 상위 폴더

if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"Project root added: {project_root}")

# 2. .env 파일 로드 (API Key 등)
env_path = os.path.join(project_root, '.env')
load_dotenv(env_path)

# --- [모듈 Import] ---
from data_pipeline.pipelines.update_race_weekly import update_weekly_news
from data_pipeline.pipelines.update_db import update_race_data
from data_pipeline.analytics import calculate_tire_degradation
from data_pipeline.pipelines.update_db import update_current_season_latest, update_race_data

# --- [Default Arguments] ---
default_args = {
    'owner': 'pitwall_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# --- [DAG 정의] ---
with DAG(
    'pitwall_weekly_ops',
    default_args=default_args,
    description='매일 뉴스 크롤링 및 지난 레이스 전략 감사',
    schedule_interval='0 9 * * 1', # 매주 월요일 아침 9시
    start_date=datetime(2024, 1, 1),
    catchup=False, # 과거 날짜꺼 한꺼번에 돌리지 않기
    tags=['F1', 'Weekly','Strategy', 'PitWall'],
) as dag:

    # Task 1: 뉴스 데이터 수집 (Soft Data)
    t1_crawl_news = PythonOperator(
        task_id='crawl_daily_news',
        python_callable=update_weekly_news,
    )

    # Task 2: 레이스 데이터 업데이트 (Hard Data)
    t2_update_race = PythonOperator(
        task_id='update_race_data',
        python_callable=update_current_season_latest,
    )

    def finish_job():
        print('주간 업데이트 작업 완료')
    
    t3_finish = PythonOperator(
        task_id = 'pipeline_finish',
        python_callable = finish_job
    )

   
    

    # --- [파이프라인 순서] ---
    # 뉴스를 먼저 보고 -> 경기 데이터를 업데이트 하고 -> 전략을 분석한다
    t1_crawl_news >> t2_update_race >> t3_finish
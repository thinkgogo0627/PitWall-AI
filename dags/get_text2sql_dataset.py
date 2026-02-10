import random
import json
from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
import os

# [가상 데이터 소스 - 나중엔 DB나 FastF1에서 가져오게 확장 가능]
DRIVERS = ['VER', 'PER', 'HAM', 'RUS', 'LEC', 'SAI', 'NOR', 'PIA']
GPS = ['Bahrain', 'Saudi Arabia', 'Australia', 'Japan', 'China', 'Miami', 'Monaco']
YEARS = [2023, 2024, 2025]
SESSIONS = ['R', 'Q', 'FP1', 'FP2', 'FP3']

async def generate_and_save_batch():
    # 1. DB 연결
    mongo_uri = os.getenv("MONGO_URI")
    client = AsyncIOMotorClient(mongo_uri)
    db = client.pitwall_db
    collection = db.dataset_function_calling
    
    generated_count = 0
    batch_size = 50 # 한 번 돌 때 50개 생성
    
    print(f"🏭 Text2SQL 데이터 생성 시작 (Batch: {batch_size})")

    for _ in range(batch_size):
        # 2. 랜덤 조합으로 '정답(Output)' 먼저 생성
        d = random.choice(DRIVERS)
        g = random.choice(GPS)
        y = random.choice(YEARS)
        s = random.choice(SESSIONS)
        
        # 함수 호출 코드 (Ground Truth)
        code_output = f"analyze_race_data(driver='{d}', year={y}, gp='{g}', session='{s}')"
        
        # 3. '질문(Instruction)' 생성 (여기서는 예시로 템플릿 사용하지만, 실제론 LLM API 호출 권장)
        # TODO: 여기에 OpenAI API 등을 붙여서 "다양한 표현"을 만들어내야 함
        # 예: response = openai.ChatCompletion.create(...)
        
        # [임시 템플릿 로직] - LLM 없이도 일단 데이터는 쌓을 수 있음
        templates = [
            f"{y}년 {g} 그랑프리 {s} 세션에서 {d} 기록 분석해줘.",
            f"{g} {y} {s} 세션, {d}의 텔레메트리 보여주세요.",
            f"Analyze {d}'s performance in {y} {g} {s}.", # 영어 데이터도 섞음
            f"{y} {g} {d} {s} 데이터 줘."
        ]
        instruction = random.choice(templates)
        
        # 4. 데이터 조립 및 저장
        data_point = {
            "dataset_type": "text2sql",
            "instruction": instruction,
            "input": "Available Function: analyze_race_data(driver, year, gp, session)",
            "output": code_output,
            "created_at": datetime.now()
        }
        
        await collection.insert_one(data_point)
        generated_count += 1

    print(f" {generated_count}개 데이터 생성 및 적재 완료!")

def task_wrapper():
    asyncio.run(generate_and_save_batch())

with DAG(
    'data_factory_text2sql',
    schedule_interval='@daily', # 매일 한 번씩 공장 가동
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['fine-tuning', 'dataset']
) as dag:
    
    generate_task = PythonOperator(
        task_id='generate_text2sql_data',
        python_callable=task_wrapper
    )
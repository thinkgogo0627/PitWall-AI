from datetime import datetime, timedelta
import asyncio
import pendulum

from airflow import DAG
from airflow.operators.python import PythonOperator

# 우리가 만든 모듈 임포트
# (Airflow에서 경로 인식을 못하면 plugins 폴더나 PYTHONPATH 설정 필요할 수 있음)
from data_pipeline.crawlers.f1_tactic import Formula1Crawler
from data_pipeline.crawlers.f1_news import AutosportCrawler
from data_pipeline.rag_indexer import RAGIndexer
from domain.documents import F1NewsDocument
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

# ---------------------------------------------------------
# 1. 비동기 작업을 동기로 감싸는 래퍼(Wrapper) 함수들
# ---------------------------------------------------------

# DB 접속 정보 (Docker 내부 통신용)
MONGO_URI = "mongodb://mongodb:27017"
QDRANT_URL = "http://qdrant:6333"

# ---------------------------------------------------------
# 1. 비동기 작업 정의 (Crawler Wrappers)
# ---------------------------------------------------------

async def _crawl_and_save_generic(crawler_cls, target_url, platform_name):
    """크롤러 클래스와 타겟 URL을 받아서 실행하는 범용 함수 (개선판)"""
    print(f"🏎️ [Task] {platform_name} 크롤링 시작...")
    
    # DB 연결
    client = AsyncIOMotorClient(MONGO_URI)
    await init_beanie(database=client.pitwall_db, document_models=[F1NewsDocument])
    
    crawler = crawler_cls()
    saved_count = 0
    
    try:
        # 1. 목록 수집
        if hasattr(crawler, 'crawl_listing_page'):
            print(f"📡 목록 수집 중... ({target_url})")
            links = crawler.crawl_listing_page(target_url, max_clicks=3) # 클릭 수 늘림
        else:
            print(f"⚠️ {platform_name}: crawl_listing_page 미구현. 건너뜀.")
            links = []

        print(f"📋 수집 대상 링크: {len(links)}개")

        # 2. 개별 기사 순회 (에러 핸들링 강화)
        for i, link in enumerate(links):
            try:
                # 이미 있는지 확인 (비동기)
                exists = await F1NewsDocument.find_one(F1NewsDocument.url == link)
                if exists:
                    # 너무 로그가 많으면 시끄러우니까 10개마다 하나씩만 찍기
                    if i % 10 == 0:
                        print(f"⏭ 중복 건너뜀 ({i}/{len(links)})")
                    continue
                
                # 추출 (동기 함수)
                print(f" [{i+1}/{len(links)}] 추출 시도: {link}")
                data = crawler.extract(link)
                
                if data and data.get('title') and data.get('content'):
                    doc = F1NewsDocument(**data)
                    await doc.insert()
                    saved_count += 1
                    print(f" 저장 완료! (현재 {saved_count}건)")
                else:
                    print(f" 데이터 부족으로 저장 실패: {link}")
                    
            except Exception as inner_e:
                print(f" 개별 기사 처리 중 에러 ({link}): {inner_e}")
                # 여기서 continue가 되므로, 하나 실패해도 다음 거 진행함!
                continue

        print(f" {platform_name} 최종 완료. 총 {saved_count}건 신규 저장.")
        
    except Exception as e:
        print(f" 크롤러 전체 프로세스 에러: {e}")
        raise e # 이건 Airflow에게 실패를 알리기 위함
        
    finally:
        # 안전하게 종료
        if hasattr(crawler, 'driver'):
            crawler.driver.quit()
            print(" 드라이버 종료됨.")

async def _run_rag_indexing():
    print("🧠 [Task] RAG 인덱싱 시작")
    indexer = RAGIndexer(mongo_uri=MONGO_URI, qdrant_url=QDRANT_URL)
    await indexer.run_indexing()

# ---------------------------------------------------------
# 2. Airflow Task용 브릿지 함수
# ---------------------------------------------------------

def task_crawl_f1():
    asyncio.run(_crawl_and_save_generic(
        Formula1Crawler, 
        "https://www.formula1.com/en/latest/tags/analysis.3HkjTN75peeCOsSegCyOWi",
        "Formula1.com"
    ))

def task_crawl_autosport():
    # Autosport F1 뉴스 섹션 URL
    asyncio.run(_crawl_and_save_generic(
        AutosportCrawler, 
        "https://www.autosport.com/f1/news", 
        "Autosport"
    ))

def task_run_indexer():
    asyncio.run(_run_rag_indexing())

# ---------------------------------------------------------
# 3. DAG 파이프라인 조립
# ---------------------------------------------------------

default_args = {
    'owner': 'pitwall_engineer',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'pitwall_daily_pipeline',
    default_args=default_args,
    description='Collect F1 News & Indexing',
    schedule_interval=timedelta(days=14), 
    start_date=pendulum.datetime(2024, 1, 1, tz="Asia/Seoul"),
    catchup=False, # 과거 데이터 소급 실행 방지
    max_active_runs = 1, # 동시에 실행되는 DAG Run 갯수를 1개로 제한
    tags=['f1', 'rag'],
) as dag:

    # 1. 크롤링 태스크들 (병렬 실행 가능)
    t1_f1 = PythonOperator(
        task_id='crawl_f1_official',
        python_callable=task_crawl_f1
    )

    t2_autosport = PythonOperator(
        task_id='crawl_autosport',
        python_callable=task_crawl_autosport
    )

    # 2. 인덱싱 태스크 (크롤링 후 실행)
    t3_index = PythonOperator(
        task_id='rag_indexing',
        python_callable=task_run_indexer
    )

    # [Dependency Structure]
    # F1크롤러와 Autosport크롤러는 동시에 돌고, 둘 다 끝나면 인덱싱 시작
    [t1_f1, t2_autosport] >> t3_index
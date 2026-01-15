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
    """크롤러 클래스와 타겟 URL을 받아서 실행하는 범용 함수"""
    print(f"🏎️ [Task] {platform_name} 크롤링 시작...")
    
    client = AsyncIOMotorClient(MONGO_URI)
    await init_beanie(database=client.pitwall_db, document_models=[F1NewsDocument])
    
    crawler = crawler_cls()
    
    # 목록 수집 (Autosport는 방식이 다를 수 있으나, 여기선 인터페이스가 같다고 가정)
    # 만약 AutosportCrawler에 crawl_listing_page가 없다면 구현 필요
    # (우리가 만든 AutosportCrawler는 현재 단일 링크 extract만 구현되어 있음 -> TODO 체크 필요)
    # 일단 단일 링크 테스트용 로직으로 대체하거나 리스트 수집 로직 추가 필요
    
    # [주의] AutosportCrawler에도 crawl_listing_page 메서드를 Formula1Crawler처럼 추가해야 함
    # 현재는 예시로 Autosport 메인 뉴스 페이지를 타겟으로 함
    try:
        if hasattr(crawler, 'crawl_listing_page'):
            links = crawler.crawl_listing_page(target_url, max_clicks=1)
        else:
            # 리스트 수집 기능이 없으면 임시로 빈 리스트 (구현 필요 알림)
            print(f"⚠️ {platform_name}: crawl_listing_page 메서드 미구현 상태")
            links = []

        saved_count = 0
        for link in links:
            exists = await F1NewsDocument.find_one(F1NewsDocument.url == link)
            if exists:
                continue
            
            data = crawler.extract(link)
            if data and data.get('title'):
                doc = F1NewsDocument(**data)
                await doc.insert()
                saved_count += 1
                
        print(f"🏁 {platform_name} 완료. {saved_count}건 저장.")
    finally:
        crawler.driver.quit()

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
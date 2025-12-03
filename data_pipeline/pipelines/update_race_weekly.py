import sys
import os

# 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_pipeline.crawlers import gp_korea, autosport, race_strat
from database.vector_store import F1VectorStore
import pandas as pd


def update_weekly_news():
    print("동적 데이터 파이프라인 가동")

    db = F1VectorStore()

    # 1. GPKorea
    try:
        df_gp_kor = gp_korea.crawl_gpkorea_final()
        db.add_data(df_gp_kor)

    except Exception as e:
        print(f"GPKorea 데이터 크롤링 실패: {e}")

    
    # 2. Autosport
    try:
        df_autosport = autosport.crawl_autosport_full()
        db.add_data(df_autosport)

    
    except Exception as e:
        print(f"Autosport 데이터 크롤링 실패: {e}")


    # 3. F1 공식사이트 중 레이스 전략 파트
    try:
        df_start = race_strat.crawl(limit=20) # 매일 상위 20개만 체크
        db.add_data(df_start)

    
    except Exception as e:
        print(f"start 데이터 크롤링 실패: {e}")

    
    print('레이스주간 데이터 업데이트 종료')

if __name__ == "__main__":
    update_weekly_news()
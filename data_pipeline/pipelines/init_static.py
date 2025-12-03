## 정적 데이터 파이프라인
## FIA 공식 규정집 -> 딱 한번만 실행하면 되는 스크립트

import sys
import os

# 프로젝트 루트 경로 추가 (모듈 import 문제 해결)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_pipeline.crawlers import FIA_reg
from database.vector_store import F1VectorStore

def init_static_data():
    print("[초기화] 정적 데이터(규정집) 파이프라인 시작...")
    
    # 1. DB 연결
    db = F1VectorStore()
    
    # 2. FIA 규정집 크롤링
    # 스포팅 규정
    df_sporting = FIA_reg.crawl(doc_type="sporting")
    db.add_data(df_sporting)
    
    # 기술 규정
    df_tech = FIA_reg.crawl(doc_type="technical")
    db.add_data(df_tech)
    
    print("[완료] 정적 데이터 초기화 끝.")

if __name__ == "__main__":
    init_static_data()
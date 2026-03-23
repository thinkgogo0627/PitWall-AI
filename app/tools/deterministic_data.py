# app/tools/deterministic_data.py
import sqlite3
import pandas as pd
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
DB_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'f1_data.db')

# [★ 멍청한 파이썬을 위한 번역기] 
# UI의 "국가명"을 DB의 "형용사형 그랑프리 이름"으로 변환
GP_MAPPING = {
    "Hungary": "Hungarian",
    "Spain": "Spanish",
    "Italy": "Italian",
    "Netherlands": "Dutch",
    "Brazil": "São Paulo", # 브라질은 상파울루 그랑프리임
    "Japan": "Japanese",
    "China": "Chinese",
    "Australia": "Australian",
    "Austria": "Austrian",
    "Great Britain": "British",
    "UK": "British",
    "Belgium": "Belgian",
    "Saudi Arabia": "Saudi Arabian",
    "Monaco": "Monaco",
    "Las Vegas": "Las Vegas",
    "Azerbaijan": "Azerbaijan"
}

def get_race_standings(year: int, gp: str, driver: str = None) -> str:
    """
    [브리핑 에이전트 전용]
    """
    # 1. 'Hungary - 헝가리' -> 'Hungary' 만 추출
    raw_gp = gp.split('-')[0].strip()
    
    # 2. 번역기 돌리기 (매핑 테이블에 없으면 그냥 원래 글자 사용)
    search_keyword = GP_MAPPING.get(raw_gp, raw_gp)
    search_keyword_sql = search_keyword.replace(' ', '%')
    
    conn = sqlite3.connect(DB_FILE_PATH)
    
    # [★ 핵심 수정] Circuit 컬럼 대신 제일 확실한 RaceID 컬럼으로 검색!
    query = """
        SELECT Position, Driver, TeamName, GridPosition, Points, Status
        FROM race_results
        WHERE Year = ? AND RaceID LIKE ?
    """
    # 예: RaceID LIKE '%Hungarian%'
    params =  [year, f"%{search_keyword_sql}%"]

    if driver:
        query += " AND Driver LIKE ?"
        params.append(f"%{driver}%")
        
    query += " ORDER BY Position ASC"
    
    try:
        df = pd.read_sql_query(query, conn, params=params)
        
        # 만약 진짜로 데이터가 없을 경우 에러 메시지 반환
        if df.empty:
            return f"🚨 [OFFICIAL RACE DATA] {year}년 {search_keyword} GP 데이터가 아직 DB에 없습니다."
            
        return df.to_markdown(index=False)
        
    except Exception as e:
        return f"DB 에러 발생: {e}"
        
    finally:
        conn.close()
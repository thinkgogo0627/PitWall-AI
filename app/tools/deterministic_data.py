# app/tools/deterministic_data.py (새로 만들 파일)
import sqlite3
import pandas as pd
from typing import Optional

DB_FILE_PATH = "data/f1_data.db" # 실제 경로에 맞게 수정

def get_race_standings(year: int, gp: str, driver: Optional[str] = None) -> str:
    """
    [브리핑 에이전트 전용 도구]
    특정 연도, 그랑프리의 레이스 결과(순위, 드라이버, 팀, 포인트)를 조회합니다.
    """
    clean_gp = gp.split('-')[0].strip()
    
    conn = sqlite3.connect(DB_FILE_PATH)
    
    # 기본 쿼리 (LLM이 짤 필요 없이 완벽하게 고정됨)
    query = """
        SELECT Position, Driver, TeamName, GridPosition, Points, Status
        FROM race_results
        WHERE Year = ? AND Circuit LIKE ?
    """
    params = [year, f"%{clean_gp}%"]
    
    # 드라이버가 지정된 경우 조건 추가
    if driver:
        query += " AND Driver LIKE ?"
        params.append(f"%{driver}%")
        
    query += " ORDER BY Position ASC"
    
    try:
        df = pd.read_sql_query(query, conn, params=params)
        if df.empty:
            return "해당 조건의 레이스 데이터가 DB에 없습니다."
        
        # 결과를 LLM이 읽기 편한 문자열(Markdown Table 등)로 반환
        return df.to_markdown(index=False)
    except Exception as e:
        return f"DB 조회 에러: {e}"
    finally:
        conn.close()
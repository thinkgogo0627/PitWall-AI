import fastf1
import sqlite3
import pandas as pd
import os

# --- 경로 설정 ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
DB_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'f1_data.db')
CACHE_DIR = os.path.join(PROJECT_ROOT, 'cache')

# FastF1 캐시 활성화 (로딩 속도 향상)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
fastf1.Cache.enable_cache(CACHE_DIR)

def update_race_results(year: int, gp: str):
    """
    특정 연도/그랑프리의 레이스 결과를 가져와 f1_data.db에 추가합니다.
    """
    print(f"🏁 {year} {gp} GP 데이터 수집을 시작합니다...")
    
    try:
        # 1. FastF1으로 공식 세션 데이터 로드 (텔레메트리 제외해서 속도 높임)
        session = fastf1.get_session(year, gp, 'R')
        session.load(telemetry=False, weather=False, messages=False)
        
        results = session.results
        event = session.event
        race_id = event['EventName'] # 예: 'Australian Grand Prix'
        
        # 2. DB 스키마에 맞게 DataFrame 조립
        df = pd.DataFrame({
            'Year': year,
            'RaceID': race_id,
            'Circuit': event['Location'],
            'Position': results['Position'],
            'Driver': results['Abbreviation'], # 예: 'RUS', 'HAM'
            'TeamName': results['TeamName'],
            'GridPosition': results['GridPosition'],
            'Points': results['Points'],
            'Status': results['Status']
        })
        
        conn = sqlite3.connect(DB_FILE_PATH)
        cursor = conn.cursor()
        
        # 3. 멱등성 보장 (실수로 두 번 돌려도 중복 안 쌓이게 기존 데이터 삭제)
        cursor.execute("DELETE FROM race_results WHERE Year = ? AND RaceID = ?", (year, race_id))
        conn.commit()
        
        # 4. 새 데이터 DB에 꽂아넣기
        df.to_sql('race_results', conn, if_exists='append', index=False)
        print(f"✅ 성공! {year} {race_id} 레이스 결과 {len(df)}건이 DB에 저장되었습니다.")
        
    except Exception as e:
        print(f"🚨 데이터 업데이트 실패: {e}")
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    # 2026년 개막전 호주 GP 실행
    update_race_results(2026, 'Chinese')
    print(f"DB 경로: {DB_FILE_PATH}")
    print(f"DB 존재 여부: {os.path.exists(DB_FILE_PATH)}")
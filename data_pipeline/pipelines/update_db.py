## FastF1에서 불러온 데이터프레임을 로드하여 SQLite에 적재하는 스크립트

import sys
import os
import sqlite3
import pandas as pd
import fastf1
from datetime import datetime


# 프로젝트 루트 경로 추가 (모듈 import용)
# 자동으로 프로젝트의 가장 위쪽 폴더 찾아서 등록
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


from data_pipeline.crawlers import fastf1_loader

DB_PATH = 'data/f1_data.db'

def save_to_sqlite(df, table_name):
    """DataFrame을 SQLite 테이블로 저장 (Append 모드)"""
    if df is None or df.empty:
        return
        
    # DB 폴더 자동 생성
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    try:
        # 이미 있는 RaceID인지 체크하는 로직은 추후 추가 (일단 중복 무시하고 Append)
        df.to_sql(table_name, conn, if_exists='append', index=False)
        print(f"     저장 완료: {table_name} ({len(df)} rows)")
    except Exception as e:
        print(f"     저장 실패 ({table_name}): {e}")
    finally:
        conn.close()

def update_race_data(year, circuit, session='R'):
    """특정 그랑프리 데이터를 DB에 업데이트"""
    print(f" [DB 작업 시작] {year} {circuit} GP")
    
    # 1. 데이터 추출
    results, laps, weather = fastf1_loader.load_session_data(year, circuit, session)
    
    # 2. DB 저장
    if results is not None:
        save_to_sqlite(results, "race_results")
        save_to_sqlite(laps, "lap_times")
        save_to_sqlite(weather, "weather_data")
        print(" [DB 작업 종료] 성공적으로 저장되었습니다.\n")
    else:
        print(" 데이터 수집 실패로 저장 건너뜀.\n")


def update_current_season_latest():
    """
    현재 연도의 스케줄을 확인하고,
    '가장 최근에 종료된 경기' 하나를 자동으로 찾아 DB에 업데이트 수행.
    (이미 DB에 있어도 덮어쓰거나 추가하도록 설계됨)
    """
    current_year = datetime.now().year
    print(f" [Smart Update] {current_year} 시즌 최신 경기 확인 중...")
    
    try:
        schedule = fastf1.get_event_schedule(current_year)
        
        # 오늘 날짜보다 이전에 끝난 경기들
        past_races = schedule[schedule['EventDate'] < datetime.now()]
        
        if past_races.empty:
            print("    아직 진행된 경기가 없습니다.")
            return

        # 가장 마지막 경기 (Last Round)
        last_race = past_races.iloc[-1]
        
        round_num = last_race['RoundNumber']
        event_name = last_race['EventName']
        
        print(f"    타겟 포착: Round {round_num} - {event_name}")
        
        # 해당 경기 업데이트 실행
        update_race_data(current_year, round_num, session='R')
        print(f' Round {round_num} 업데이트 완료 (혹은 이미 최신 상태)')
        
    except Exception as e:
        print(f" 스마트 업데이트 실패: {e}")

        raise e


if __name__ == "__main__":
    # for test
    update_race_data(2025, 'Sao Paulo' , 'R')



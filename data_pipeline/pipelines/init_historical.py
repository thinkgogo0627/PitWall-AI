## 2021 ~ 2025 시즌의 FastF1 Hard Data 전원을 DB에 채우는 스크립트
# 단 1회만 사용
import sys
import os
import fastf1
import time
from datetime import datetime

# 프로젝트 루트 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_pipeline.pipelines.update_db import update_race_data

def init_historical_data_by_year(target_year):
    print(f" [History Init] {target_year}년 데이터 적재 시작...")
    
    # 캐시 설정
    if not os.path.exists('./cache'):
        os.makedirs('./cache')
    fastf1.Cache.enable_cache('./cache')

    try:
        schedule = fastf1.get_event_schedule(target_year)
        completed_races = schedule[schedule['EventDate'] < datetime.now()]
        
        print(f"\n {target_year}년 시즌: 총 {len(completed_races)}개 경기 처리 예정")
        
        for _, row in completed_races.iterrows():
            round_num = row['RoundNumber']
            event_name = row['EventName']
            
            if round_num == 0: continue
                
            print(f"    Round {round_num}: {event_name} 업데이트 중...")
            
            try:
                update_race_data(target_year, round_num, session='R')
                
                # ★ 대기 시간 10초로 증가 (안전빵)
                print("      API 과부하 방지를 위해 10초 대기...")
                time.sleep(10) 
                
            except Exception as e:
                print(f"       실패: {e}")
                time.sleep(10)
                
    except Exception as e:
        print(f" {target_year}년 스케줄 로드 실패: {e}")

    print(f"\n {target_year}년 적재 완료!")

if __name__ == "__main__":
    # 사용자가 연도를 직접 입력하게 함
    print("다운로드할 연도를 입력하세요 (예: 2021)")
    year_input = int(input("Year: "))
    init_historical_data_by_year(year_input)
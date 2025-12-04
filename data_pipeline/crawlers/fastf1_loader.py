# 데이터 추출 모듈
## FastF1 라이브러리에서 Tabular hard data 추출

import fastf1
import pandas as pd
import numpy as np
import logging

# 로깅 끄기 (조용하게)
logging.getLogger('fastf1').setLevel(logging.ERROR)


def load_session_data(year, circuit, session_type='R'):
    """
    특정 세션의 핵심 데이터(결과, 랩타임, 날씨)를 추출하여 반환
    """
    print(f" FastF1 로드 중: {year} {circuit} ({session_type})...")
    
    # 1. 세션 로드
    try:
        session = fastf1.get_session(year, circuit, session_type)
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as e:
        print(f" 세션 로드 실패: {e}")
        return None, None, None
    

    # 2. 식별자 생성 (DB 저장용 ID) -> 2024_Bahrain_R
    race_id = f"{year}_{session.event.EventName.replace(' ', '_')}_{session_type}"
    
    # --- A. 경기 결과 (Results) ---
    results = session.results.copy()
    results['RaceID'] = race_id
    results['Year'] = year
    results['Circuit'] = circuit
    
    df_results = results[[
        'RaceID', 'Year', 'Circuit', 'Abbreviation', 'TeamName', 
        'Position', 'GridPosition', 'Points', 'Status'
    ]].rename(columns={'Abbreviation': 'Driver'})
    
    # --- B. 랩 타임 (Lap Times) ---
    laps = session.laps.copy()
    laps['RaceID'] = race_id
    
    # 시간 포맷 변환 (초 단위 float)
    laps['LapTime_Sec'] = laps['LapTime'].dt.total_seconds()
    laps['Sector1_Sec'] = laps['Sector1Time'].dt.total_seconds()
    laps['Sector2_Sec'] = laps['Sector2Time'].dt.total_seconds()
    laps['Sector3_Sec'] = laps['Sector3Time'].dt.total_seconds()
    
    df_laps = laps[[
        'RaceID', 'Driver', 'LapNumber', 'Stint', 'Compound', 
        'TyreLife', 'FreshTyre', 'LapTime_Sec', 
        'Sector1_Sec', 'Sector2_Sec', 'Sector3_Sec', 'IsAccurate'
    ]]
    df_laps = df_laps.replace({np.nan: None})

    # --- C. 날씨 (Weather) ---
    weather = session.weather_data.copy()
    weather['RaceID'] = race_id
    weather['Time_Sec'] = weather['Time'].dt.total_seconds()
    
    df_weather = weather[[
        'RaceID', 'Time_Sec', 'AirTemp', 'Humidity', 
        'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed'
    ]]

    print(f"데이터 추출 완료: 랩({len(df_laps)}), 결과({len(df_results)}), 날씨({len(df_weather)})")
    
    return df_results, df_laps, df_weather
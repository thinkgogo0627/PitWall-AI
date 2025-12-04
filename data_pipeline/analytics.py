import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import logging

# 전역 설정
fastf1.plotting.setup_mpl(misc_mpl_mods=False)

# -----------------------------------------------------------------------------
# 1. 타이어 마모도 분석 (Tire Degradation)
# -----------------------------------------------------------------------------
def calculate_tire_degradation(year, circuit, drivers=None, session_type='R'):
    """
    드라이버별/스틴트별 타이어 마모도(연료 보정 포함) 계산
    """
    print(f" [분석] {year} {circuit} 타이어 마모도 분석 시작...")
    
    # 로깅 끄기
    fastf1.fastf1.logger.setLevel(logging.WARNING)
    
    try:
        session = fastf1.get_session(year, circuit, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f" 세션 로드 실패: {e}")
        return pd.DataFrame()
    
    if drivers is None:
        drivers = session.results['Abbreviation'].iloc[:10].tolist()
        
    results = []
    
    for driver in drivers:
        try:
            # 1. 전체 랩 (메타데이터용)
            all_laps = session.laps.pick_driver(driver)
            
            for stint_id in all_laps['Stint'].unique():
                raw_stint = all_laps[all_laps['Stint'] == stint_id]
                if raw_stint.empty: continue
                
                compound = raw_stint['Compound'].iloc[0]
                start_lap = raw_stint['LapNumber'].min()
                end_lap = raw_stint['LapNumber'].max()
                laps_run = len(raw_stint)
                
                # 2. 계산용 정제 데이터 (빠른 랩만)
                clean_stint = raw_stint.pick_quicklaps().reset_index(drop=True)
                
                if len(clean_stint) < 2:
                    slope = 0.0
                else:
                    x = clean_stint['LapNumber']
                    y = clean_stint['LapTime'].dt.total_seconds()
                    slope, _, _, _, _ = linregress(x, y)
                
                # 연료 보정 (일반적으로 랩당 0.03초 빨라짐을 보정)
                fuel_corrected_deg = slope + 0.03
                
                results.append({
                    "Driver": driver,
                    "Stint": int(stint_id),
                    "Compound": compound,
                    "Laps_Run": laps_run,
                    "Start_Lap": start_lap,
                    "End_Lap": end_lap,
                    "Raw_Slope": round(slope, 4),
                    "True_Degradation": round(fuel_corrected_deg, 4)
                })
                
        except Exception as e:
            print(f"{driver} 분석 중 오류: {e}")
            continue
            
    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# 2. 미니 섹터 지배력 분석 (Mini-Sector Dominance)
# -----------------------------------------------------------------------------
def analyze_mini_sector_dominance(year, circuit, drivers=None, session_type='R', total_chunks=25):
    """
    서킷 미니 섹터 분석 -> 시각화 객체(fig)와 요약 텍스트(text) 반환
    """
    print(f" [분석] {year} {circuit} 미니 섹터 지배력 분석 시작...")
    
    session = fastf1.get_session(year, circuit, session_type)
    session.load(telemetry=True, weather=False, messages=False)
    
    if drivers is None:
        drivers = session.results['Abbreviation'].iloc[:3].tolist()
    
    # 팀 컬러 확보
    driver_colors = {}
    for d in drivers:
        try:
            driver_colors[d] = fastf1.plotting.get_driver_color(d, session)
        except:
            driver_colors[d] = '#000000' # 색상 없으면 검정

    # 텔레메트리 수집
    telemetry_list = []
    for driver in drivers:
        try:
            lap = session.laps.pick_driver(driver).pick_fastest()
            tel = lap.get_telemetry().add_distance()
            tel['Driver'] = driver
            telemetry_list.append(tel)
        except:
            continue
            
    if not telemetry_list:
        return None, "데이터 부족"

    all_tel = pd.concat(telemetry_list)
    
    # 미니 섹터 계산
    all_tel['MiniSector'] = pd.cut(all_tel['Distance'], total_chunks, labels=False)
    
    # 지배자 계산
    sector_stats = all_tel.groupby(['MiniSector', 'Driver'])['Speed'].mean().reset_index()
    speed_pivot = sector_stats.pivot(index='MiniSector', columns='Driver', values='Speed')
    fastest_driver_per_sector = speed_pivot.idxmax(axis=1)
    
    # --- 시각화 ---
    ref_tel = telemetry_list[0][['X', 'Y', 'Distance']].copy()
    ref_tel['MiniSector'] = pd.cut(ref_tel['Distance'], total_chunks, labels=False)
    ref_tel['Fastest_Driver'] = ref_tel['MiniSector'].map(fastest_driver_per_sector)
    
    fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(10, 6))
    ax.plot(ref_tel['X'], ref_tel['Y'], color='gray', linestyle='-', linewidth=8, alpha=0.3)
    
    for driver in drivers:
        driver_segment = ref_tel[ref_tel['Fastest_Driver'] == driver]
        if len(driver_segment) > 0:
            ax.scatter(driver_segment['X'], driver_segment['Y'], 
                       s=30, color=driver_colors[driver], label=driver, edgecolors='none')
            
    ax.set_title(f"{year} {circuit} GP: Mini-Sector Dominance")
    ax.axis('off')
    ax.legend()
    plt.close(fig) # 화면에 바로 띄우지 않고 객체만 반환
    
    # --- LLM용 텍스트 요약 ---
    dominance_counts = fastest_driver_per_sector.value_counts(normalize=True) * 100
    summary_lines = [f"**[미니 섹터 분석 요약 ({total_chunks} 구간)]**"]
    for driver, share in dominance_counts.items():
        summary_lines.append(f"- {driver}: {share:.1f}% 구간 지배")
    
    return fig, "\n".join(summary_lines)
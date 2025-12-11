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




# -----------------------------------------------------------------------------
# 3. 전략 감사(Audit): 실제 피트인 vs 가상의 Stay Out 시나리오 비교
# -----------------------------------------------------------------------------

def audit_pit_strategy(year, circuit, driver, session_type='R'):
    """
    전략 감사(Audit): 실제 피트인 타이밍 vs 가상의 'Stay Out' 시나리오 비교
    - 실제: 피트인 후 복귀하여 달린 기록
    - 가상(Ghost): 피트인 하지 않고 (예: 3랩) 더 달렸을 때의 예상 기록
    """
    print(f" [전략 감사] {year} {circuit} - {driver} 피트 타이밍 분석 중...")
    
    # 1. 데이터 로드
    session = fastf1.get_session(year, circuit, session_type)
    session.load(laps=True, telemetry=False, weather=False, messages=False)
    
    laps = session.laps.pick_driver(driver)
    
    # 2. 피트 스탑 찾기 (타이어를 교체한 랩)
    # PitInTime이 있거나, Stint가 바뀌는 지점
    pit_laps = laps[laps['PitOutTime'].notnull()]['LapNumber'].tolist()
    
    audit_report = []
    
    for pit_lap in pit_laps:
        # 첫 랩이나 마지막 랩 부근 제외
        if pit_lap < 5 or pit_lap > laps['LapNumber'].max() - 5:
            continue
            
        print(f"  Lap {int(pit_lap)} 피트인 정밀 분석 시도...")
        
        # --- A. 실제 상황 (Reality) ---
        # 피트 아웃 후 3랩 동안의 기록 (OutLap + 2 Flying Laps)
        real_future_laps = laps[laps['LapNumber'].between(pit_lap, pit_lap + 3)]
        if len(real_future_laps) < 4: continue # 데이터 부족하면 패스
        
        real_time_taken = real_future_laps['LapTime'].dt.total_seconds().sum()
        
        # --- B. 가상 상황 (Ghost: Stay Out) ---
        # 피트인 직전 5랩의 평균 페이스와 마모도(기울기) 계산
        past_laps = laps[laps['LapNumber'].between(pit_lap - 5, pit_lap - 1)]
        if past_laps.empty: continue
        
        # 간단한 선형 회귀로 마모도(Slope) 계산
        x = past_laps['LapNumber']
        y = past_laps['LapTime'].dt.total_seconds()
        slope, intercept, _, _, _ = linregress(x, y)
        
        # 마모도를 반영하여 향후 3랩(Ghost Laps)의 예상 기록 산출
        # Ghost Lap T = (직전 랩 시간) + (기울기)
        last_lap_time = y.iloc[-1]
        ghost_time_taken = 0
        current_pred = last_lap_time
        
        for i in range(1, 5): # 피트랩 포함 4랩치 계산 (피트로스 vs 주행 비교)
            # 피트인 랩(InLap)도 그냥 달렸다고 가정
            current_pred += slope 
            ghost_time_taken += current_pred
            
        # --- C. 비교 및 판정 (Audit Verdict) ---
        # *주의: 피트 스탑은 약 20~24초 손해를 봄. 
        # 실제 기록(피트 포함) vs 고스트 기록(그냥 주행) 직접 비교는 어렵고,
        # "누적 시간" 관점에서 트랙 포지션 손익을 계산해야 함.
        # 여기서는 MVP용으로 단순화: "피트인으로 인한 손실(약 20초)을 제외하고, 
        # 새 타이어의 퍼포먼스 이득(Gain)이 고스트카의 마모 손실(Loss)보다 큰가?"를 봅니다.
        
        # (더 정교한 로직은 트랙별 Pit Loss Time 데이터가 필요함. 일단 약식 구현)
        # 만약 실제 전략이 고스트보다 압도적으로 느리다면 -> "Too Early" (너무 일찍 들어옴)
        # 만약 실제 전략이 고스트와 비슷하거나 빠르다면 -> "Good Timing"
        
        # 예시 리포트 데이터 생성
        audit_report.append({
            "Pit_Lap": int(pit_lap),
            "Real_Time_3Laps": round(real_time_taken, 3),
            "Ghost_Time_3Laps": round(ghost_time_taken, 3),
            "Degradation_Slope": round(slope, 4),
            "Verdict": "Analyze Needed" # LLM이 최종 판단하도록 데이터만 넘김
        })
        
    return pd.DataFrame(audit_report)
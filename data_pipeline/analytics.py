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
            all_laps = session.laps.pick_drivers(driver)
            
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
            lap = session.laps.pick_drivers(driver).pick_fastest()
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

def calculate_slope(laps):
    """주어진 랩들의 기록으로 기울기(마모도) 계산"""
    if len(laps) < 2: return 0.05 # 기본값 (데이터 부족 시)
    x = laps['LapNumber']
    y = laps['LapTime'].dt.total_seconds()
    slope, _, _, _, _ = linregress(x, y)
    return slope


def is_sc_affected(laps, lap_number):
    '''
    TrackStatus: 1 = Greenplag , 2 = 옐로플래그 , 4 = SC , 5 = Red,
    6 = VSC , 7 = VSC Ending
    '''
    try:
        lap_data = laps[laps['LapNumber'] == lap_number]
        if lap_data.empty: return False

        status = str(lap_data['TrackStatus'].iloc[0])

        # '4'(SC), '6'(VSC), '7'(VSC Ending)이 포함되어 있으면 True
        if any(code in status for code in ['4', '5', '6', '7']):
            return True
            
        return False
    except Exception:
        return False


def get_pit_loss_time(circuit, year):
    """
    서킷, 년도별 평균 피트 로스타임
    '피트 스탑 때문에 트랙에서 손해 본 총 시간' 구해야 함
    (T_inlap + T_outlap) - (2*avg_T_Normallap)
    """
    try:
        # 정상적인 랩 필터링 -> 전체 드라이버의 평균 레이스 랩타임
        session = fastf1.get_session(year, circuit, 'R')
        good_laps = session.laps.pick_quicklaps().pick_track_status('1')
        if good_laps.empty: return 22.0 # 데이터 없으면 기본값

        avg_normal_lap = good_laps['LapTime'].dt.total_seconds().median()

        # 2. 피트스탑 샘플 수집
        pit_loss_samples = []

        for driver in session.drivers:
            d_laps = session.laps.pick_drivers(driver)

            # 피트스탑 한 랩 찾기
            pit_in_laps = d_laps[d_laps['PitInTime'].notnull()]['LapNumber'].tolist()

            for lap_num in pit_in_laps:
                # In-Lap (들어가는 랩)
                in_lap = d_laps[d_laps['LapNumber'] == lap_num]
                # Out-Lap (나오는 랩) - 보통 그 다음 랩
                out_lap = d_laps[d_laps['LapNumber'] == lap_num + 1]
                
                if in_lap.empty or out_lap.empty: continue
                
                # SC/VSC 상황이었다면 샘플에서 제외 (왜곡 방지)
                if is_sc_affected(session.laps, lap_num) or is_sc_affected(session.laps, lap_num + 1):
                    continue
                
                # 시간 계산
                t_in = in_lap['LapTime'].dt.total_seconds().iloc[0]
                t_out = out_lap['LapTime'].dt.total_seconds().iloc[0]
                
                if np.isnan(t_in) or np.isnan(t_out): continue
                
                # 피트 로스 = (In + Out) - (2 * Normal)
                # 단, 피트 스탑 정지 시간(Stationary Time)은 전략에 따라 다르므로
                # 순수한 '트랙 손실' + '평균 정지 시간(약 2~3초)'이 포함된 값으로 계산됨.
                loss = (t_in + t_out) - (2 * avg_normal_lap)
                
                # 비정상적인 값 필터링 (10초 미만이거나 40초 초과면 이상치일 확률 높음)
                if 10.0 < loss < 40.0:
                    pit_loss_samples.append(loss)
                    
        # 3. 평균값 도출
        if not pit_loss_samples:
            return 22.0 # 샘플 없으면 기본값
            
        calculated_loss = np.mean(pit_loss_samples)
        # print(f"   📉 Calculated Dynamic Pit Loss: {round(calculated_loss, 2)}s (Samples: {len(pit_loss_samples)})")
        return calculated_loss

    except Exception as e:
        print(f"⚠️ 피트 로스 계산 실패: {e}, 기본값 사용")
        return 22.0



# --- [Target 1] 조기 피트인 판정 (더 버티는 게 나았나?) ---
def audit_extension(driver_laps, pit_lap, slope, pit_loss):
    """
    Case A: 실제 피트인 (Reality)
    Case B: 3랩 더 버팀 (Ghost Stay Out)
    -> Case B가 더 빠르면 'Too Early' 판정
    """
    # Case A: 실제 피트 아웃 후 3랩 (OutLap + 2 Flying)
    real_next_laps = driver_laps[driver_laps['LapNumber'].between(pit_lap, pit_lap + 3)]
    if len(real_next_laps) < 4: return None
    
    # 실제 소요 시간 (피트 로스 포함된 기록들)
    time_actual = real_next_laps['LapTime'].dt.total_seconds().sum()
    
    # Case B: 가상 스테이 아웃 3랩
    # 공식: (직전 랩타임 + slope) * 3
    last_lap_time = driver_laps[driver_laps['LapNumber'] == pit_lap - 1]['LapTime'].dt.total_seconds().iloc[0]
    
    time_ghost_drive = 0
    current_pred = last_lap_time
    for _ in range(4): # 피트랩 포함 4랩
        current_pred += slope
        time_ghost_drive += current_pred
        
    # 비교: (고스트 주행 시간 + 나중 피트 로스) vs (실제 주행 시간)
    # *주의: 고스트는 나중에라도 피트를 해야 하므로, 공정한 비교를 위해 Pit Loss를 더해줌
    time_ghost_total = time_ghost_drive + pit_loss
    
    diff = time_actual - time_ghost_total
    
    # diff가 양수(+)면 실제(Actual)가 더 오래 걸림 -> 스테이 아웃(Ghost)이 이득 -> "Too Early"
    # diff가 음수(-)면 실제(Actual)가 더 빠름 -> 피트인이 이득 -> "Good Timing"
    return {
        "verdict": "Too Early" if diff > 1.0 else "Good Timing",
        "time_diff": round(diff, 3), # 양수면 손해 본 초(Seconds)
        "desc": f"3랩 더 버텼다면 {abs(round(diff, 2))}초 {'이득' if diff > 0 else '손해'} 예상"
    }

# --- [Target 2] 공격 기회 판정 (일찍/늦게 들어갔다면?) ---
def audit_opportunity(session, driver, pit_lap, pit_loss):
    """
    가상 시나리오: 1랩 일찍 들어갔다면(Undercut) 앞차를 잡았을까?
    """
    # 1. 내 앞차(Rival) 찾기 (피트인 2랩 전 기준)
    lap_check = pit_lap - 2
    drivers = session.drivers
    
    my_pos = session.laps.pick_drivers(driver).pick_lap(lap_check)['Position'].iloc[0]
    if my_pos == 1:
        return {"verdict": "Leader", "desc": "1위 주행 중 (추월 대상 없음)"}
        
    # 앞차 찾기 (Position이 내 앞인 사람)
    # (MVP에서는 간단히 순위만 보지만, 실제로는 Gap 데이터를 봐야 함)
    # FastF1에서 앞차를 특정하기 까다로우므로, 여기서는 로직을 간소화:
    # "내가 1랩 일찍 들어갔으면(Outlap이 2초 빠름), 피트 전 랩타임(Old Tyre)보다 얼마나 이득인가?"
    
    # 가정: 새 타이어(Undercut)는 헌 타이어보다 랩당 약 2.0초 빠르다 (서킷마다 다름)
    undercut_gain = 2.0 
    
    return {
        "verdict": "Check Undercut",
        "desc": f"1랩 일찍 들어갔다면 약 {undercut_gain}초 이득 예상 (트래픽 고려 안 함)"
    }


# --- [Main Wrapper] 메인 실행 함수 ---
def audit_race_strategy(year, circuit, driver, session_type='R'):
    print(f" [전략 감사] {year} {circuit} - {driver} 분석 중...")
    
    try:
        session = fastf1.get_session(year, circuit, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        return f"데이터 로드 실패: {e}"
    
    laps = session.laps
    driver_laps = laps.pick_drivers(driver)
    
    # 피트 스탑 감지
    pit_laps = driver_laps[driver_laps['PitOutTime'].notnull()]['LapNumber'].tolist()
    
    if not pit_laps:
        return "피트 스탑 기록이 없습니다."
        
    reports = []
    pit_loss = get_pit_loss_time(circuit, year)
    
    for pit_lap in pit_laps:
        if pit_lap < 5 or pit_lap > driver_laps['LapNumber'].max() - 5: continue

        # SC/VSC 감지 로직 추가
        is_sc = False
        ## 피트랩 포함 앞뒤 1랩 검사
        for check_lap in range(int(pit_lap)-1, int(pit_lap)+1):
            if is_sc_affected(laps, check_lap):
                is_sc = True
                break

        if is_sc: # 세이프티카 상황이라면?
            # 분석 스킵 -> 로그만 남기기
            reports.append({
                "Pit_Lap": int(pit_lap),
                "Tire_Slope": 0.0,
                "Audit_Type": "SC condition",
                "Verdict": "Pass",
                "Detail": "SC/VSC 상황으로 인해 데이터 왜곡 가능성 존재 (분석에서 제외)",
                "Opportunity_Check": "-"
            })
            continue # 다음 피트스톱으로 넘어가기

        
        # 1. 마모도(Slope) 계산 (직전 5랩)
        past_laps = driver_laps[driver_laps['LapNumber'].between(pit_lap - 5, pit_lap - 1)]
        slope = calculate_slope(past_laps)
        
        # 2. Target 1: Extension Audit (방어)
        ext_result = audit_extension(driver_laps, pit_lap, slope, pit_loss)
        
        # 3. Target 2: Opportunity Audit (공격)
        opp_result = audit_opportunity(session, driver, pit_lap, pit_loss)
        
        if ext_result:
            reports.append({
                "Pit_Lap": int(pit_lap),
                "Slope": round(slope, 4),
                "Extension_Verdict": ext_result['verdict'],
                "Extension_Detail": ext_result['desc'],
                "Opportunity_Detail": opp_result['desc']
            })
            
    return pd.DataFrame(reports)


if __name__ == "__main__":
    df = audit_race_strategy(2025, 'Qatar' , 'SAI')
    print(df.to_markdown())
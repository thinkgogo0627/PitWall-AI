import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import logging
import os

# 전역 설정
fastf1.plotting.setup_mpl(misc_mpl_mods=False)

# 현재 폴더
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 프로젝트 루트
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)# data_pipeline 폴더

CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'cache')   # ../app/cache 로 이동

# --- [2. FastF1 캐시 설정] ---
# 폴더가 없으면 생성 (안전장치)
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f" 캐시 폴더가 없어서 생성했습니다: {CACHE_DIR}")

# FastF1에게 데이터 가져오기 선어
try:
    fastf1.Cache.enable_cache(CACHE_DIR)
    print(f" FastF1 Cache Enabled at: {CACHE_DIR}")
except Exception as e:
    print(f" FastF1 Cache 설정 실패: {e}")

# -----------------------------------------------------------------------------
# 1. 타이어 마모도 분석 (Tire Degradation)
# -----------------------------------------------------------------------------
def calculate_tire_degradation(year, circuit, drivers=None, session_type='R'):
    """
    드라이버별/스틴트별 타이어 마모도(연료 보정 포함) 계산
    """
    print(f" {year} {circuit} {session_type} 로딩 시도...")
    
    # [수정됨] 로깅 끄기 (올바른 방법)
    logging.getLogger('fastf1').setLevel(logging.WARNING)
    
    try:
        session = fastf1.get_session(year, circuit, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        print(f"세션 로드 완료. 드라이버 수: {len(session.drivers)}")
    except Exception as e:
        print(f"세션 로드 실패: {e}")
        return pd.DataFrame()
    
    if drivers is None:
        # 상위 10명만 추리기
        drivers = session.results['Abbreviation'].iloc[:10].tolist()
        
    results = []
    
    for driver in drivers:
        try:
            # 1. 전체 랩 가져오기
            all_laps = session.laps.pick_drivers(driver)
            
            for stint_id in all_laps['Stint'].unique():
                raw_stint = all_laps[all_laps['Stint'] == stint_id]
                
                # 타이어 정보 없는 경우 스킵
                if raw_stint.empty or pd.isna(raw_stint['Compound'].iloc[0]): 
                    continue
                
                compound = raw_stint['Compound'].iloc[0]
                
                # [수정됨] pick_quicklaps() 대신 좀 더 데이터 확보에 유리한 필터링 사용
                # 'IsAccurate'가 True인 랩만 가져옵니다. (SC 상황 등 제외)
                clean_stint = raw_stint[raw_stint['IsAccurate'] == True].reset_index(drop=True)
                
                # 데이터가 너무 적으면(3랩 미만) 회귀분석 불가
                if len(clean_stint) < 3:
                    continue
                
                x = clean_stint['LapNumber']
                y = clean_stint['LapTime'].dt.total_seconds()
                
                # 선형 회귀 (기울기 계산)
                slope, _, _, _, _ = linregress(x, y)
                
                # 연료 보정 (+0.03초: 차가 가벼워지는 효과 상쇄)
                fuel_corrected_deg = slope + 0.03
                
                results.append({
                    "Driver": driver,
                    "Stint": int(stint_id),
                    "Compound": compound,
                    "Laps_Run": len(clean_stint),
                    "True_Degradation": round(fuel_corrected_deg, 4)
                })
                
        except Exception as e:
            continue
            
    # 결과 반환
    if not results:
        print(" 유효한 스틴트 데이터가 하나도 없음.")
        return pd.DataFrame()

    print(f" 총 {len(results)}개의 스틴트 분석 완료.")
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 2. 미니 섹터 지배력 분석 (Mini-Sector Dominance) for 기능 2
# -----------------------------------------------------------------------------
def mini_sector_dominance_analyze(year, circuit, drivers=None, session_type='R', total_chunks=25):
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


def get_pit_loss_time(session):
    """
    서킷, 년도별 평균 피트 로스타임
    '피트 스탑 때문에 트랙에서 손해 본 총 시간' 구해야 함
    (T_inlap + T_outlap) - (2*avg_T_Normallap)
    """
    try:
        # 정상적인 랩 필터링 -> 전체 드라이버의 평균 레이스 랩타임
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
        # print(f"    Calculated Dynamic Pit Loss: {round(calculated_loss, 2)}s (Samples: {len(pit_loss_samples)})")
        return calculated_loss

    except Exception as e:
        print(f" 피트 로스 계산 실패: {e}, 기본값 사용")
        return 22.0


# 특정 레이스에서 특정 드라이버의 피트 로스 타임 계산 로직 (위 함수와 다름)
def get_specific_pit_loss(driver_laps, pit_lap, avg_track_pit_loss=22.0):
    """
    해당 드라이버가 실제 피트 스톱에서 소비한 시간을 계산합니다.
    (In-Lap + Out-Lap) - (정상 주행 2랩 예상 시간)
    """
    try:
        # In-Lap과 Out-Lap 가져오기
        in_lap_row = driver_laps[driver_laps['LapNumber'] == pit_lap]
        out_lap_row = driver_laps[driver_laps['LapNumber'] == pit_lap + 1]
        
        if in_lap_row.empty or out_lap_row.empty:
            return avg_track_pit_loss # 데이터 없으면 평균값 리턴

        in_time = in_lap_row['LapTime'].dt.total_seconds().iloc[0]
        out_time = out_lap_row['LapTime'].dt.total_seconds().iloc[0]
        
        # 정상 주행(Flying Lap) 기준 시간 구하기 (피트 전후 3랩 제외하고 중간값)
        # 만약 데이터가 너무 없으면 그냥 전체 랩의 중간값
        clean_laps = driver_laps.pick_quicklaps()
        if clean_laps.empty:
             ref_pace = avg_track_pit_loss / 2 # 비상용 임시값
        else:
             ref_pace = clean_laps['LapTime'].dt.total_seconds().median()

        # 실제 로스 계산
        actual_loss = (in_time + out_time) - (2 * ref_pace)
        
        # 이상치 처리 (VSC 등으로 40초 넘어가면 그냥 평균값 사용하거나, VSC 감지 로직 필요)
        # 여기서는 단순하게 10초~40초 사이일 때만 유효하다고 가정
        if 10 < actual_loss < 45:
            return round(actual_loss, 2)
        else:
            return avg_track_pit_loss

    except Exception:
        return avg_track_pit_loss



# --- [Target 1] 조기 피트인 판정 (더 버티는 게 나았나?) ---
def audit_extension(driver_laps, pit_lap, slope, pit_loss):
    """
    Case A: 실제 피트인 (Reality)
    Case B: 3랩 더 버팀 (Ghost Stay Out)
    -> 비교: (Ghost 주행 시간 + 나중에 치를 Pit Loss) vs (실제 주행 시간)
    """
    # 1. 실제 데이터: 피트 아웃 후 3랩 데이터 확보 (OutLap + 2 Flying)
    # pit_lap은 In-Lap이므로, 그 다음 랩부터가 '피트 스톱 이후의 시간'입니다.
    # 하지만 비교를 위해 'In-Lap' 시점부터 계산하는 것이 타임라인상 맞습니다.
    # 여기서는 "피트 랩 포함 향후 3랩"을 비교하겠습니다.
    
    real_next_laps = driver_laps[driver_laps['LapNumber'].between(pit_lap, pit_lap + 3)]
    
    # 데이터가 부족하면(경기 종료 직전 등) 분석 불가
    if len(real_next_laps) < 2: return None
    
    # 비교할 랩 수 동기화 (실제 데이터가 2랩뿐이면 가상도 2랩만 돌려야 함)
    n_laps_to_compare = len(real_next_laps)
    
    # Case A: 실제 소요 시간 (In-Lap + Out-Lap + Flying...)
    time_actual = real_next_laps['LapTime'].dt.total_seconds().sum()
    
    # Case B: 가상 스테이 아웃 (Ghost)
    # 피트 직전 랩(Old Tyre)의 기록을 기준으로 slope(데그라데이션)만큼 느려짐
    prev_lap = driver_laps[driver_laps['LapNumber'] == pit_lap - 1]
    if prev_lap.empty: return None
    
    last_lap_time = prev_lap['LapTime'].dt.total_seconds().iloc[0]
    
    time_ghost_drive = 0
    current_pred = last_lap_time
    
    for _ in range(n_laps_to_compare):
        current_pred += slope  # 랩당 slope초만큼 느려짐 (Degradation)
        time_ghost_drive += current_pred
        
    # 비교: (고스트 주행 시간 + 나중에 해야 할 Pit Loss) vs (이미 피트한 실제 시간)
    # *주의: 고스트도 언젠간 피트를 해야 하므로, 공정한 비교를 위해 Pit Loss를 더해줌
    time_ghost_total = time_ghost_drive + pit_loss
    
    diff = time_actual - time_ghost_total
    
    # diff > 0: 실제(Actual)가 더 오래 걸림 -> "더 버티는 게 빨랐다" (Too Early)
    # diff < 0: 실제(Actual)가 더 빠름 -> "피트하길 잘했다" (Good Timing)
    
    verdict = "Good Timing"
    desc = "적절한 시점에 피트인했습니다."
    
    if diff > 2.0: # 2초 이상 차이면 명확한 실수
        verdict = "Too Early (Loss)"
        desc = f"더 버텼어야 합니다. 약 {diff:.1f}초 손해 예상."
    elif diff < -2.0:
        verdict = "Great Call"
        desc = f"타이어가 죽기 직전 완벽하게 들어왔습니다. ({abs(diff):.1f}초 이득)"

    return {
        "verdict": verdict,
        "time_diff": round(diff, 3),
        "desc": desc
    }



def audit_opportunity(session, driver_id, pit_lap, my_actual_loss, target_rival_id=None):
    """
    [정밀 분석] 언더컷 시뮬레이션
    :param my_actual_loss: get_specific_pit_loss()로 계산된 내 실제 피트 소요 시간

     Net Margin = 라이벌과의 갭 - (언더컷 이득) + (피트 로스 delta)
    - 라이벌과의 갭: 피트 인 직전의 라이벌과의 시간 차이
    - 언더컷 이듯: (라이벌의 현 타이어 랩터임) - (나의 새 타이어 아웃랩 환산 기록)
    - 피트 로스 델타: (나의 피트로스) - (라이벌의 피트로스)
    
    [수정됨] target_rival_id가 있으면 그 드라이버를 강제로 분석합니다.
    없으면 기존처럼 바로 앞 순위(Auto-Detect)를 찾습니다.
    """
    try:
        # 1. 기준 랩 로드
        lap_check = int(pit_lap) - 1
        laps_at_check = session.laps.pick_lap(lap_check)
        
        # 2. 내 데이터 찾기
        my_row = laps_at_check[laps_at_check['Driver'] == str(driver_id)]
        if my_row.empty: 
             my_row = laps_at_check[laps_at_check['DriverNumber'] == str(driver_id)]
        if my_row.empty: return {}

        # 3. [핵심 수정] 라이벌 찾기 로직 분기
        rival_row = None
        
        if target_rival_id:
            # A. 지정된 라이벌 강제 검색 (사용자가 '베어만' 찍음)
            print(f"DEBUG: Targeting Rival ID {target_rival_id}")
            rival_row = laps_at_check[laps_at_check['Driver'] == str(target_rival_id)]
            if rival_row.empty:
                 rival_row = laps_at_check[laps_at_check['DriverNumber'] == str(target_rival_id)]
            
            if rival_row.empty:
                print(f"DEBUG: Rival {target_rival_id} not found in Lap {lap_check}")
                return {} # 라이벌이 리타이어했거나 데이터 없음
                
        else:
            # B. 기존 로직 (내 바로 앞차 자동 감지)
            my_pos = my_row['Position'].iloc[0]
            if my_pos == 1: return {"verdict": "Leader", "desc": "Leader"}
            rival_row = laps_at_check[laps_at_check['Position'] == (my_pos - 1)]

        if rival_row is None or rival_row.empty: return {}
        
        rival_id = rival_row['Driver'].iloc[0]

        # 4. 데이터 계산 (나머지는 동일)
        my_time = my_row['Time'].iloc[0]
        rival_time = rival_row['Time'].iloc[0]
        
        # 갭 계산 (양수면 내가 뒤, 음수면 내가 앞)
        current_gap = (my_time - rival_time).total_seconds()
        
        # ... (이하 로직 동일: Pit Loss Delta, Net Margin 계산) ...
        
        # [복붙용 하단 로직 유지]
        avg_track_loss = get_pit_loss_time(session) 
        pit_loss_delta = my_actual_loss - avg_track_loss
        undercut_gain_pure = 2.5 
        net_margin = current_gap + pit_loss_delta - undercut_gain_pure
        
        prob = 0
        if net_margin < -1.0: prob = 95
        elif net_margin < 0: prob = 60
        elif net_margin < 1.0: prob = 30
        else: prob = 5

        return {
            "verdict": "Analyzed",
            "rival": rival_id,
            "telemetry": {
                "gap_to_rival": round(current_gap, 3),
                "my_pit_loss": round(my_actual_loss, 2),
                "avg_pit_loss": round(avg_track_loss, 2),
                "loss_delta": round(pit_loss_delta, 2)
            },
            "simulation": {
                "undercut_gain": undercut_gain_pure,
                "net_margin": round(net_margin, 3),
                "probability": prob
            },
            "desc": f"Target: {rival_id} | Gap: {current_gap:.1f}s"
        }

    except Exception as e:
        print(f"Audit Error: {e}")
        return {}




# --- [Main Wrapper] 메인 실행 함수 ---
def audit_race_strategy(year, circuit, driver_identifier, session_type='R'):
    """
    [Hybrid Strategy Auditor]
    1. 기본: 타이어 스틴트 정보 (무조건 반환)
    2. 문맥: 피트스탑 당시 VSC/SC 여부 감지
    3. 심화: 피트 타이밍 적절성(Late Stop/Undercut) 자동 평가
    """
    logging.getLogger('fastf1').setLevel(logging.WARNING)
    print(f"\n [Strategy Audit] {year} {circuit} - Driver: {driver_identifier}")

    # --- [Step 1: 데이터 로드 및 안전장치] ---
    try:
        session = fastf1.get_session(year, circuit, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f" 데이터 로드 실패: {e}")
        return pd.DataFrame()

    # 드라이버 매핑 (이름 -> 번호)
    target_driver = str(driver_identifier).strip()
    if target_driver not in session.drivers:
        found = False
        for d in session.drivers:
            d_info = session.get_driver(d)
            if (d_info['Abbreviation'] == target_driver.upper()) or \
               (target_driver.upper() in d_info['LastName'].upper()):
                target_driver = d
                print(f"  └ 매핑 성공: '{driver_identifier}' -> '{target_driver}'")
                found = True
                break
        if not found:
            return pd.DataFrame()

    try:
        driver_laps = session.laps.pick_driver(target_driver)
    except KeyError:
        return pd.DataFrame()

    if driver_laps.empty: return pd.DataFrame()

    # --- [Step 2: 스틴트별 분석 시작] ---
    driver_laps['Stint'] = driver_laps['Stint'].fillna(0).astype(int)
    
    # 이 서킷의 피트 로스 시간 계산 (참조용)
    pit_loss_time = get_pit_loss_time(session)
    
    stint_summary = []
    
    for stint_id, stint_laps in driver_laps.groupby('Stint'):
        if stint_laps.empty: continue
        
        compound = stint_laps['Compound'].iloc[0]
        laps_run = len(stint_laps)
        start_lap = int(stint_laps['LapNumber'].min())
        end_lap = int(stint_laps['LapNumber'].max())
        
        # 평균 페이스 (VSC 등 느린 랩 제외)
        clean_laps = stint_laps.pick_quicklaps()
        if not clean_laps.empty:
            avg_pace = clean_laps['LapTime'].dt.total_seconds().mean()
        else:
            avg_pace = stint_laps['LapTime'].dt.total_seconds().mean()

        # --- [Step 3: 피트 컨디션 및 타이밍 심화 분석] ---
        # 스틴트가 끝나는 시점(=피트 인)의 상황을 분석
        pit_condition = "Green Flag"
        verdict = "Normal"    
        
        # 마지막 스틴트가 아니면 (=피트 스톱을 했다면)
        if stint_id < driver_laps['Stint'].max():
            # In-Lap의 트랙 상태 확인
            in_lap_data = stint_laps.iloc[-1]
            status_code = str(in_lap_data['TrackStatus'])
            
            # VSC/SC 감지 로직
            if '4' in status_code:
                pit_condition = "SAFETY CAR (SC)"
                verdict = "Lucky Stop"
                detail = f"SC 상황에서 피트인 (손실 시간 최소화)"
            elif '6' in status_code or '7' in status_code:
                pit_condition = "VIRTUAL SC (VSC)"
                verdict = "Optimal Timing"
                detail = f"VSC를 활용한 이득 (약 {int(pit_loss_time*0.4)}초 절약)"
            elif '5' in status_code:
                pit_condition = "RED FLAG"
                verdict = "Free Tire Change"
            
            # --- [Step 4: 그린 플래그일 때 타이밍 분석 (연장/조기 여부)] ---
            if pit_condition == "Green Flag" and laps_run > 5:
                # 마지막 5랩의 기울기(Slope) 분석
                last_5 = stint_laps.iloc[-5:]
                if len(last_5) >= 3:
                    slope = calculate_slope(last_5)
                    # 기울기가 0.15 이상이면 랩당 0.15초씩 느려지고 있다는 뜻 (타이어 사망)
                    if slope > 0.15:
                        verdict = "High Deg (Late Stop)"
                        detail = f"마지막에 페이스 급락 (+{slope:.2f}s/lap)"
                    elif slope < 0.05:
                         verdict = "Good Pace"
                         detail = "타이어 상태가 양호했음 (더 달릴 수 있었음)"

        # 마지막 스틴트인 경우
        else:
            pit_condition = "FINISH"
            verdict = "Race End"

        # 결과 저장
        stint_summary.append({
            "Stint": int(stint_id),
            "Compound": compound,
            "Laps": laps_run,
            "Range": f"L{start_lap}-L{end_lap}",
            "Pit_Cond": pit_condition,
            "Avg_Pace": round(avg_pace, 3) if not np.isnan(avg_pace) else 0.0,
            "Verdict": verdict,
            "Note": detail
        })

    result_df = pd.DataFrame(stint_summary)
    
    if not result_df.empty:
        print(f" [Audit] {len(result_df)}개의 스틴트 분석 완료.")
        
    return result_df
'''
    
def audit_race_strategy(year, circuit, driver_identifier):
    """
    특정 드라이버의 랩타임/피트스탑 전략 분석 (VSC/SC 감지 + 드라이버 매핑 포함)
    """
    logging.getLogger('fastf1').setLevel(logging.WARNING)
    
    try:
        session = fastf1.get_session(year, circuit, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        print(f"세션 로드 실패: {e}")
        return pd.DataFrame()
    
    # 1. [Driver Mapping] 사용자 입력(ANT, Antonelli)을 차량 번호('12')로 변환
    target_driver = str(driver_identifier).strip()
    
    # 입력값이 숫자가 아니거나, 세션 드라이버 목록에 바로 없을 때 검색
    if target_driver not in session.drivers:
        print(f" [Check] '{target_driver}' 드라이버 코드 매핑 시도...")
        found = False
        
        # 전체 드라이버 루프 돌면서 Abbreviation(약어)나 LastName 확인
        for d_num in session.drivers:
            d_info = session.get_driver(d_num)
            # 약어(ANT) 또는 성(Antonelli) 매칭
            if (d_info['Abbreviation'] == target_driver.upper()) or \
               (target_driver.upper() in d_info['LastName'].upper()):
                target_driver = d_num
                print(f"   └ 매칭 성공! '{driver_identifier}' -> '{target_driver}' (No)")
                found = True
                break
        
        if not found:
            print(f"   └ 실패: 세션에서 드라이버 '{driver_identifier}'를 찾을 수 없습니다.")
            return pd.DataFrame() # 빈 데이터 반환 -> Agent가 '데이터 없음' 처리

    # 2. 랩 데이터 추출
    try:
        driver_laps = session.laps.pick_driver(target_driver)
    except KeyError:
        return pd.DataFrame()

    if driver_laps.empty:
        return pd.DataFrame()

    results = []
    driver_laps['Stint'] = driver_laps['Stint'].fillna(0).astype(int)
    
    for stint_id in driver_laps['Stint'].unique():
        stint = driver_laps[driver_laps['Stint'] == stint_id]
        if stint.empty: continue
        
        compound = stint['Compound'].iloc[0]
        start_lap = stint['LapNumber'].min()
        end_lap = stint['LapNumber'].max()
        laps_run = len(stint)
        
        # 3. [SC/VSC Logic] 피트 인(In-Lap) 당시 상황 파악
        # 해당 스틴트의 마지막 랩 (피트로 들어가는 랩)
        in_lap_data = stint.iloc[-1]
        track_status = str(in_lap_data['TrackStatus']) 
        
        # 상태 코드 해석
        pit_condition = "Green Flag"
        if '4' in track_status:
            pit_condition = " SAFETY CAR (SC)"
        elif '6' in track_status or '7' in track_status:
            pit_condition = " VIRTUAL SAFETY CAR (VSC)"
        elif '5' in track_status:
             pit_condition = "RED FLAG"
        elif '2' in track_status:
            pit_condition = "Yellow Flag"
        
        # 평균 페이스
        clean_laps = stint.pick_quicklaps()
        if len(clean_laps) > 0:
            avg_pace = clean_laps['LapTime'].dt.total_seconds().mean()
        else:
            avg_pace = stint['LapTime'].dt.total_seconds().mean()
            
        results.append({
            "Stint": int(stint_id),
            "Compound": compound,
            "Laps_Run": laps_run,
            "Start_Lap": start_lap,
            "End_Lap": end_lap,
            "Pit_Condition": pit_condition,  # ★ LLM이 이걸 보고 "VSC라서 들어갔군" 판단함
            "Avg_Pace": round(avg_pace, 3)
        })

    return pd.DataFrame(results)
'''

if __name__ == "__main__":
    df = audit_race_strategy(2025, 'Las Vegas' , 'ANT')
    print(df.to_markdown())
import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import os
import logging
from scipy.stats import linregress

# 로깅 설정
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 전역 설정
fastf1.plotting.setup_mpl(misc_mpl_mods=False)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'cache')

# 캐시 활성화
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)
try:
    fastf1.Cache.enable_cache(CACHE_DIR)
except Exception:
    pass

# =============================================================================
# 1. 통합 전략 감사 (Integrated Strategy Audit)
# =============================================================================
def audit_race_strategy(year: int, circuit: str, driver_identifier: str) -> pd.DataFrame:
    """
    [Agent 3 핵심 엔진]
    트래픽, 페이스, 피트 타이밍 + 스틴트 길이 평가(Stint Evaluation) 추가
    """
    try:
        # 1. 세션 로드
        session = fastf1.get_session(year, circuit, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        
        # 2. 드라이버 매핑
        target_driver = _resolve_driver_id(session, driver_identifier)
        if not target_driver: return pd.DataFrame()

        # 3. 전체 필드 타이어 통계 계산 (기준점 마련)
        # (다른 드라이버들은 보통 몇 랩이나 탔는지 확인)
        global_tire_stats = _get_global_tire_stats(session)

        # 4. 내 드라이버 데이터 추출
        laps = session.laps.pick_driver(target_driver)
        if laps.empty: return pd.DataFrame()

        # 트래픽 감지
        if 'TimeDiffToAhead' in laps.columns:
            laps['InTraffic'] = laps['TimeDiffToAhead'] < 1.0
        else:
            laps['InTraffic'] = False

        # 5. 스틴트별 분석
        laps['Stint'] = laps['Stint'].fillna(1).astype(int)
        stint_summary = []

        for stint_id, stint_data in laps.groupby('Stint'):
            compound = stint_data['Compound'].iloc[0]
            laps_run = len(stint_data)
            start_lap = int(stint_data['LapNumber'].min())
            end_lap = int(stint_data['LapNumber'].max())
            
            # --- [New] 스틴트 길이 평가 로직 ---
            stint_eval = "Normal"
            if compound in global_tire_stats:
                avg_life = global_tire_stats[compound]['avg']
                max_life = global_tire_stats[compound]['max']
                
                # 비율로 평가 (평균 대비)
                if laps_run >= max_life * 0.95:
                    stint_eval = "🔥 Extreme (Max Life)"
                elif laps_run > avg_life * 1.3:
                    stint_eval = "Long Run (Management)"
                elif laps_run < avg_life * 0.6:
                    stint_eval = "Short Sprint"
                else:
                    stint_eval = "Standard"
            # ------------------------------------

            # 피트 아웃/인 상황 체크
            pit_condition = _check_pit_condition(stint_data)

            # 페이스 분석
            race_laps = stint_data[stint_data['TrackStatus'] == '1']
            clean_laps = race_laps[~race_laps['InTraffic']]
            traffic_laps = race_laps[race_laps['InTraffic']]

            avg_clean = clean_laps['LapTime'].dt.total_seconds().mean() if not clean_laps.empty else None
            deg_slope = _calculate_slope(clean_laps) if len(clean_laps) > 3 else 0.0

            # Insight 생성
            note = [f"[{stint_eval}]"] # 맨 앞에 스틴트 평가 추가
            
            if pit_condition != "Green Flag":
                note.append(f"{pit_condition} Stop")
            
            if len(traffic_laps) > laps_run * 0.4:
                note.append(f"Traffic({len(traffic_laps)}L)")
            
            if deg_slope > 0.15:
                note.append("High Deg")

            stint_summary.append({
                "Stint": stint_id,
                "Compound": compound,
                "Laps": f"{laps_run} ({start_lap}-{end_lap})",
                "Type": stint_eval, # 명시적 컬럼 추가
                "Clean_Pace": round(avg_clean, 3) if avg_clean else "-",
                "Deg_Slope": round(deg_slope, 4),
                "Insight": ", ".join(note)
            })

        return pd.DataFrame(stint_summary)

    except Exception as e:
        print(f"Strategy Audit Error: {e}")
        return pd.DataFrame()

# =============================================================================
# 2. 타이어 성능 분석 (기존 유지)
# =============================================================================
def calculate_tire_degradation(year: int, circuit: str) -> pd.DataFrame:
    try:
        session = fastf1.get_session(year, circuit, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps.pick_track_status('1').pick_quicklaps()
        
        stats = []
        for compound in ['SOFT', 'MEDIUM', 'HARD']:
            comp_laps = laps[laps['Compound'] == compound]
            if len(comp_laps) < 10: continue

            avg_pace = comp_laps['LapTime'].dt.total_seconds().mean()
            slope = _calculate_slope(comp_laps)
            max_life = comp_laps['TyreLife'].max()
            avg_life = comp_laps.groupby('Driver')['TyreLife'].max().mean() # 드라이버별 평균 사용량

            stats.append({
                "Compound": compound,
                "Avg_Pace": round(avg_pace, 3),
                "Avg_Life": f"{int(avg_life)} Laps", # 평균 수명 추가
                "Max_Life": f"{int(max_life)} Laps",
                "Degradation": "High" if slope > 0.1 else "Stable"
            })
        return pd.DataFrame(stats)
    except Exception:
        return pd.DataFrame()

# =============================================================================
# 🔒 내부 헬퍼 함수 (Internal Helpers)
# =============================================================================

def _get_global_tire_stats(session):
    """
    [New] 이번 경기 전체 드라이버들의 타이어 수명 통계를 낸다.
    return: {'SOFT': {'avg': 15, 'max': 22}, 'HARD': ...}
    """
    stats = {}
    valid_laps = session.laps[session.laps['Compound'].notna()] # DNS 케이스 제외
    
    for compound in ['SOFT', 'MEDIUM', 'HARD', 'INTER', 'WET']:
        comp_data = valid_laps[valid_laps['Compound'] == compound]
        if comp_data.empty: continue
        
        # 각 스틴트별 길이 추출
        stint_lengths = comp_data.groupby(['Driver', 'Stint']).size()
        
        stats[compound] = {
            'avg': stint_lengths.mean(),
            'max': stint_lengths.max()
        }
    return stats

def _resolve_driver_id(session, identifier):
    identifier = str(identifier).strip().upper()
    if identifier in session.drivers: return identifier
    for d in session.drivers:
        info = session.get_driver(d)
        if identifier in [info['Abbreviation'], info['LastName'].upper()]: return d
    return None

def _check_pit_condition(stint_data):
    if stint_data.empty: return "Green Flag"
    status = str(stint_data.iloc[-1]['TrackStatus'])
    if '4' in status: return "SC"
    if '6' in status or '7' in status: return "VSC"
    if '5' in status: return "RED FLAG"
    return "Green Flag"

def _calculate_slope(laps):
    if len(laps) < 3: return 0.0
    x = laps['TyreLife'].values
    y = laps['LapTime'].dt.total_seconds().values
    mask = ~np.isnan(x) & ~np.isnan(y)
    if not mask.any(): return 0.0
    slope, _, _, _, _ = linregress(x[mask], y[mask])
    return slope

# Placeholder for Sector Analysis
def mini_sector_dominance_analyze(year, circuit, drivers=None):
    return None, "Sector Analysis Ready"
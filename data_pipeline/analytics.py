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
logging.getLogger('fastf1').setLevel(logging.ERROR)
fastf1.plotting.setup_mpl(misc_mpl_mods=False)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)  # data_pipeline의 부모 = 프로젝트 루트

# FastF1 캐시 설정
# enable_cache()는 SQLite HTTP 캐시를 생성해야 해서 쓰기 권한 필요
# Streamlit Cloud에서는 data/cache/가 읽기 전용이므로 /tmp에 writable 캐시 생성 후
# 커밋된 캐시(pickle 파일)를 symlink로 연결
_local_cache = os.path.join(PROJECT_ROOT, 'data', 'cache')
try:
    os.makedirs(_local_cache, exist_ok=True)
    _test = os.path.join(_local_cache, '.write_test')
    with open(_test, 'w') as f:
        f.write('ok')
    os.remove(_test)
    CACHE_DIR = _local_cache
except (PermissionError, OSError):
    # 읽기 전용 → /tmp 사용 + 커밋된 캐시 symlink
    CACHE_DIR = '/tmp/fastf1_cache'
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(_local_cache):
        for _item in os.listdir(_local_cache):
            _src = os.path.join(_local_cache, _item)
            _dst = os.path.join(CACHE_DIR, _item)
            if os.path.isdir(_src) and not os.path.exists(_dst):
                try:
                    os.symlink(_src, _dst)
                except OSError:
                    pass

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
        try:
            if 'TimeDiffToAhead' in laps.columns and laps['TimeDiffToAhead'].notna().any():
                laps = laps.copy()
                laps['InTraffic'] = laps['TimeDiffToAhead'].fillna(99) < 1.0
            else:
                laps = laps.copy()
                laps['InTraffic'] = False
        except Exception:
            laps = laps.copy()
            laps['InTraffic'] = False

        # 5. 스틴트별 정밀 분석
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
            racing_laps = stint_data[stint_data['TrackStatus'] == '1']
            
            # 클린 랩 vs 트래픽 랩 분리
            clean_laps = racing_laps[~racing_laps['InTraffic']]
            traffic_laps = racing_laps[racing_laps['InTraffic']]
            
            avg_clean = clean_laps['LapTime'].dt.total_seconds().mean() if not clean_laps.empty else None
            avg_traffic = traffic_laps['LapTime'].dt.total_seconds().mean() if not traffic_laps.empty else None
            
            # 트래픽 비율 계산
            traffic_pct = (len(traffic_laps) / len(racing_laps) * 100) if not racing_laps.empty else 0

            # 에이전트가 읽기 편하게 컬럼명 명확화
            stint_summary.append({
                "Stint": stint_id,
                "Tyre": f"{compound} ({stint_eval})", # 예: HARD (Extreme)
                "Laps": laps_run,
                "Traffic_Run": f"{int(traffic_pct)}%", # 트래픽 겪은 비율
                "Clean_Pace": round(avg_clean, 3) if avg_clean else "N/A",
                "Traffic_Pace": round(avg_traffic, 3) if avg_traffic else "N/A",
                "Pit_Event": pit_condition
            })

        return pd.DataFrame(stint_summary)

    except Exception as e:
        logger.error(f"Strategy Audit Error [{year} {circuit} {driver_identifier}]: {type(e).__name__}: {e}")
        raise
# =============================================================================
# 2. 타이어 성능 분석 (기존 유지)
# =============================================================================
def calculate_tire_degradation(year: int, circuit: str) -> pd.DataFrame:
    try:
        # 1. 절대 경로로 캐시 디렉토리 연결 (현재 파일 위치 기준 프로젝트 루트의 data/cache)
        # ※ 파일 위치에 따라 '../' 개수는 조절이 필요할 수 있습니다.
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CACHE_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../data/cache'))
        
        # 캐시 강제 활성화
        fastf1.Cache.enable_cache(CACHE_DIR)
        
        # 2. 로컬 디렉토리 스캔을 통한 이름 강제 보정 (Natural Language 방어)
        year_dir = os.path.join(CACHE_DIR, str(year))
        matched_event_name = circuit # 실패할 경우를 대비한 기본값
        
        if os.path.exists(year_dir):
            available_folders = os.listdir(year_dir)
            search_keyword = circuit.lower().replace(" ", "")
            
            for folder_name in available_folders:
                # 폴더명 예시: '2021-07-18_British_Grand_Prix'
                # 앞의 날짜(2021-07-18_)를 떼어내고 'British_Grand_Prix'만 추출
                if '_' in folder_name:
                    clean_name = folder_name.split('_', 1)[-1] 
                else:
                    clean_name = folder_name
                
                # LLM이 던진 단어(예: 'british')가 폴더명에 포함되어 있는지 확인
                if search_keyword in clean_name.lower().replace("_", ""):
                    # 매칭 성공! fastf1이 100% 인식하도록 언더바를 공백으로 치환
                    # 'British_Grand_Prix' -> 'British Grand Prix'
                    matched_event_name = clean_name.replace("_", " ")
                    break
        
        print(f"🔍 [Tire Analysis] LLM 입력: '{circuit}' -> 캐시 매칭: '{matched_event_name}'")
        
        # 3. 완벽하게 보정된 이름으로 세션 로드 (네트워크 낭비 제로)
        session = fastf1.get_session(year, matched_event_name, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        
        # 4. 아웃랩, 인랩 등을 제외한 정상 주행 랩(Quicklaps)만 추출
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
    # 번호로 직접 매칭 (가장 빠름)
    if identifier in session.drivers:
        return identifier
    # session.results에서 약어/성 매칭 (session.get_driver() 호출 없이)
    try:
        results = session.results
        if 'Abbreviation' in results.columns:
            match = results[results['Abbreviation'] == identifier]
            if not match.empty:
                return str(match.iloc[0]['DriverNumber'])
        if 'LastName' in results.columns:
            match = results[results['LastName'].str.upper() == identifier]
            if not match.empty:
                return str(match.iloc[0]['DriverNumber'])
    except Exception:
        pass
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
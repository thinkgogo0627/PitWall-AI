# app/tools/telemetry_data.py

import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import warnings
import seaborn as sns
import numpy as np
import plotly.graph_objects as go

# 경고 무시 및 F1 스타일 설정
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', module='fastf1')
import logging
logging.getLogger('fastf1').setLevel(logging.ERROR)
fastf1.plotting.setup_mpl(misc_mpl_mods=False)

# FastF1 캐시 설정 (enable_cache는 SQLite 생성 필요 → 쓰기 권한 필요)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
_local_cache = os.path.join(PROJECT_ROOT, 'data', 'cache')
try:
    os.makedirs(_local_cache, exist_ok=True)
    _test = os.path.join(_local_cache, '.write_test')
    with open(_test, 'w') as f:
        f.write('ok')
    os.remove(_test)
    CACHE_DIR = _local_cache
except (PermissionError, OSError):
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

PLOT_DIR = '/tmp/fastf1_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

try:
    fastf1.Cache.enable_cache(CACHE_DIR)
except Exception:
    pass

# -----------------------------------------------------------------------------
# 드라이버 이름 정규화
# 

DRIVER_MAPPING = {
    # Red Bull
    '베르스타펜': 'VER', '막스': 'VER', 'Verstappen': 'VER', 'Max': 'VER',
    '츠노다': 'TSU', 'Tsunoda': 'TSU',
    # Cadillac
    '보타스': 'BOT', 'Bottas': 'BOT', 'Valteri': 'BOT',
    '페레즈': 'PER', '체코': 'PER', 'Perez': 'PER', 'Sergio': 'PER',
    # McLaren
    '노리스': 'NOR', '랜도': 'NOR', 'Norris': 'NOR', 'Lando': 'NOR',
    '피아스트리': 'PIA', '오스카': 'PIA', 'Piastri': 'PIA', 'Oscar': 'PIA',
    # Ferrari
    '르클레르': 'LEC', '샤를': 'LEC', 'Leclerc': 'LEC', 'Charles': 'LEC',
    '해밀턴': 'HAM', '루이스': 'HAM', 'Hamilton': 'HAM', 'Lewis': 'HAM',
    # Williams
    '알본': 'ALB', 'Albon': 'ALB',
    '사인츠': 'SAI', '카를로스': 'SAI', 'Sainz': 'SAI', 'Carlos': 'SAI',
    # Mercedes
    '안토넬리': 'ANT', 'Antonelli': 'ANT',
    '러셀': 'RUS', '조지': 'RUS', 'Russell': 'RUS', 'George': 'RUS',
    # Aston Martin
    '알론소': 'ALO', 'Alonso': 'ALO',
    '스트롤': 'STR', 'Stroll': 'STR',
    # Alpine
    '가슬리': 'GAS', 'Pierre': 'GAS',
    '콜라핀토': 'COL' , '콜라': 'COL',
    # Haas
    '베어만': 'BEA' , '올리' : 'BEA',
    '오콘': 'OCO', '에스테반':'OCO',
    # VCAR
    '로슨': 'LAW', '리암 로슨': 'LAW',
    '린드블라드': 'LIN' , '린블': 'LIN',
    # Audi
    '휠켄버그': 'HUL' , '헐크': 'HUL' , '니코 휠켄버그': 'HUL',
    '보톨레토': 'BOR' , '가비': 'BOR'
}

def _normalize_name(name: str) -> str:
    """입력된 이름이 매핑 테이블에 있으면 약어로 변환, 없으면 대문자로 반환"""
    clean_name = name.strip()
    if clean_name in DRIVER_MAPPING:
        return DRIVER_MAPPING[clean_name]
    return clean_name.upper()[:3]

def _save_plot(filename, facecolor='black'):
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR, exist_ok=True)
    
    save_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor=facecolor)
    plt.close()
    print(f"✅ 그래프 저장 완료: {save_path}")
    return f"GRAPH_GENERATED: {save_path}"

# =============================================================================
# [★ 추가된 무적의 세션 로더] LLM의 환각 트랙명을 완벽하게 보정!
# =============================================================================
def _get_loaded_session(year: int, circuit: str, load_telemetry: bool = True):
    year_dir = os.path.join(CACHE_DIR, str(year))
    matched_event_name = circuit

    TRACK_ALIASES = {
        "silverstone": "british", "monza": "italian", "spa": "belgian",
        "interlagos": "sao paulo", "zandvoort": "dutch", "suzuka": "japanese",
        "japan": "japanese", "saopaulo": "s o paulo", "cota": "united states", 
        "austin": "united states", "china": "chinese", "shanghai": "chinese",
        "britain": "british", "uk": "british"
    }

    search_keyword = circuit.lower().replace(" ", "").replace("_", "")
    for alias, real_name in TRACK_ALIASES.items():
        if alias in search_keyword:
            search_keyword = real_name.replace(" ", "")
            break

    if os.path.exists(year_dir):
        available_folders = os.listdir(year_dir)
        for folder_name in available_folders:
            clean_name = folder_name.split('_', 1)[-1] if '_' in folder_name else folder_name
            compare_name = clean_name.lower().replace("_", "").replace(" ", "")
            
            if search_keyword in compare_name:
                matched_event_name = clean_name.replace("_", " ")
                break

    print(f"🔍 [Telemetry Data] UI 입력: '{circuit}' -> 캐시 매칭: '{matched_event_name}'")
    
    session = fastf1.get_session(year, matched_event_name, 'R')
    
    # [★ 핵심 디버깅 추가] 로드 과정을 try-except로 감싸고, 실패 시 경고 출력
    try:
        session.load(laps=True, telemetry=load_telemetry, weather=False, messages=False)
        
        # [★ 방어 로직] 텔레메트리를 요구했는데 데이터가 비어있다면 명시적 에러 발생
        if load_telemetry:
            # 첫 번째 랩의 텔레메트리를 살짝 찔러봐서 데이터가 진짜 있는지 확인합니다.
            try:
                _ = session.laps.pick_fastest().get_telemetry()
                print("✅ 텔레메트리 데이터 로드 성공!")
            except Exception as inner_e:
                raise ValueError(f"해당 캐시에 텔레메트리 세부 데이터(속도, 위치 등)가 누락되어 있습니다. ({inner_e})")
                
    except Exception as e:
        print(f"🚨 [Session Load Error] 세션 데이터를 불러오는데 실패했습니다: {e}")
        # 실패하면 그대로 에러를 던져서 프론트엔드가 '데이터 없음' 처리하게 함
        raise e
        
    return session

# -----------------------------------------------------------------------------
# 1. [Plotly] 랩타임 비교 (Interactive)
# -----------------------------------------------------------------------------
def get_race_pace_data(year: int, race: str, driver1: str, driver2: str):
    """Plotly용 데이터 객체를 반환합니다."""
    try:
        d1_code = _normalize_name(driver1)
        d2_code = _normalize_name(driver2)
        
        # [★ 수정] 헬퍼 함수로 세션 로드 (Race Pace는 telemetry 필요 없음)
        session = _get_loaded_session(year, race, load_telemetry=False)

        d1 = session.laps.pick_driver(d1_code)
        d2 = session.laps.pick_driver(d2_code)

        if d1.empty or d2.empty: return None

        fig = go.Figure()

        c1 = fastf1.plotting.get_driver_color(d1_code, session=session)
        fig.add_trace(go.Scatter(
            x=d1['LapNumber'], y=d1['LapTime'].dt.total_seconds(),
            mode='lines+markers', name=d1_code,
            line=dict(color=c1, width=2),
            marker=dict(size=4)
        ))

        c2 = fastf1.plotting.get_driver_color(d2_code, session=session)
        fig.add_trace(go.Scatter(
            x=d2['LapNumber'], y=d2['LapTime'].dt.total_seconds(),
            mode='lines+markers', name=d2_code,
            line=dict(color=c2, width=2, dash='dash'),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title=f"{year} {race} Race Pace: {d1_code} vs {d2_code}",
            xaxis_title="Lap Number",
            yaxis_title="Lap Time (Seconds)",
            template="plotly_dark",
            hovermode="x unified",
            height=500
        )
        return fig

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Race Pace Error] {e}")
        return None

# -----------------------------------------------------------------------------
# 2. [NEW] 트랙 도미넌스 맵 (Track Dominance)
# -----------------------------------------------------------------------------
def generate_track_dominance_plot(year: int, race: str, driver1: str, driver2: str) -> str:
    """
    두 드라이버의 가장 빠른 랩(Fastest Lap)을 기준으로,
    트랙의 각 지점에서 누가 더 빨랐는지를 색상으로 표시하는 지도를 그립니다.
    """
    try:
        driver1 = _normalize_name(driver1)
        driver2 = _normalize_name(driver2)

        print(f"🗺️ [Dominance] Generating Map: {year} {race} ({driver1} vs {driver2})...")

        # [★ 핀포인트 수술] 캐시 삭제 없이 2번 도구만 살리는 로직
        try:
            # 1차 시도: 기존 캐시에서 로드 시도
            session = _get_loaded_session(year, race, load_telemetry=True)
            lap1 = session.laps.pick_driver(driver1).pick_fastest()
            
            if lap1 is None or pd.isna(lap1.get('LapTime')):
                raise ValueError("랩 데이터가 없습니다.")
                
            _ = lap1.get_telemetry() # 텔레메트리가 진짜 있는지 찔러봄 (없으면 여기서 에러 터짐)
            lap2 = session.laps.pick_driver(driver2).pick_fastest()
            
        except Exception as e:
            print(f"⚠️ [Dominance] 로컬 캐시에 텔레메트리 누락 감지! 다른 기능 보호를 위해 2번 도구만 캐시를 우회합니다... ({e})")
            
            # 2차 시도: 1, 3번 도구가 쓰는 캐시를 보호하기 위해, 캐시 기능을 일시 정지하고 강제 다운로드
            fastf1.Cache.disable_cache()
            
            # fastf1이 알아먹게 트랙명 보정 (안 그러면 또 못 찾음)
            matched_event = race
            if "japan" in race.lower() or "suzuka" in race.lower(): matched_event = "Japanese"
            elif "cota" in race.lower() or "austin" in race.lower() or "united states" in race.lower(): matched_event = "United States"
            elif "saopaulo" in race.lower() or "brazil" in race.lower(): matched_event = "Sao Paulo"
            elif "china" in race.lower() or "shanghai" in race.lower(): matched_event = "Chinese"
            elif "britain" in race.lower() or "silverstone" in race.lower() or "uk" in race.lower(): matched_event = "British"
            
            session = fastf1.get_session(year, matched_event, 'R')
            session.load(laps=True, telemetry=True, weather=False, messages=False)
            
            # 볼일 끝났으면 다른 도구들을 위해 캐시 다시 켜기! (매우 중요)
            fastf1.Cache.enable_cache(CACHE_DIR)

            lap1 = session.laps.pick_driver(driver1).pick_fastest()
            lap2 = session.laps.pick_driver(driver2).pick_fastest()


        if lap1 is None or lap2 is None:
            return "데이터 부족: 두 드라이버의 정상적인 랩 타임(Fastest Lap)을 찾을 수 없습니다."

        # 텔레메트리 추출 및 분석
        tel1 = lap1.get_telemetry().add_distance()
        tel2 = lap2.get_telemetry().add_distance()

        interp_speed_d2 = np.interp(tel1['Distance'], tel2['Distance'], tel2['Speed'])
        delta = tel1['Speed'] - interp_speed_d2

        x = np.array(tel1['X'].values)
        y = np.array(tel1['Y'].values)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        color1 = fastf1.plotting.get_driver_color(driver1, session=session)
        color2 = fastf1.plotting.get_driver_color(driver2, session=session)
        
        colors = [color1 if d > 0 else color2 for d in delta[:-1]]

        fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
        ax.set_facecolor('black')
        
        lc = LineCollection(segments, colors=colors, linewidths=5)
        ax.add_collection(lc)
        
        ax.autoscale_view()
        ax.set_aspect('equal')
        ax.axis('off')

        from matplotlib.lines import Line2D
        legend_lines = [Line2D([0], [0], color=color1, lw=4),
                        Line2D([0], [0], color=color2, lw=4)]
        ax.legend(legend_lines, [driver1, driver2], loc='upper right', facecolor='black', labelcolor='white')
        
        plt.title(f"{year} {session.event['EventName']} Track Dominance\n({driver1} vs {driver2})", color='white', fontsize=15, fontweight='bold')

        filename = f"{year}_{session.event['EventName']}_Dominance_{driver1}_vs_{driver2}.png".replace(" ", "_")
        return _save_plot(filename)

    except Exception as e:
            print(f"⚠️ [Dominance] 로컬 캐시에 텔레메트리 누락 감지! 임시 폴더로 우회합니다... ({e})")
            
            # [★ 진짜 해결책] 존재하지 않는 disable_cache 대신, 일회용 임시 폴더를 캐시로 지정
            import tempfile
            temp_cache_dir = os.path.join(tempfile.gettempdir(), 'fastf1_temp_bypass')
            os.makedirs(temp_cache_dir, exist_ok=True)
            fastf1.Cache.enable_cache(temp_cache_dir)
            
            # fastf1이 알아먹게 트랙명 보정
            matched_event = race
            if "japan" in race.lower() or "suzuka" in race.lower(): matched_event = "Japanese"
            elif "cota" in race.lower() or "austin" in race.lower() or "united states" in race.lower(): matched_event = "United States"
            elif "saopaulo" in race.lower() or "brazil" in race.lower(): matched_event = "Sao Paulo"
            elif "china" in race.lower() or "shanghai" in race.lower(): matched_event = "Chinese"
            elif "britain" in race.lower() or "silverstone" in race.lower() or "uk" in race.lower(): matched_event = "British"
            
            session = fastf1.get_session(year, matched_event, 'R')
            session.load(laps=True, telemetry=True, weather=False, messages=False)
            
            # 볼일 끝났으면 다른 도구들을 위해 기존 캐시 경로(CACHE_DIR)로 원상 복구!
            fastf1.Cache.enable_cache(CACHE_DIR)

            lap1 = session.laps.pick_driver(driver1).pick_fastest()
            lap2 = session.laps.pick_driver(driver2).pick_fastest()
        

# -----------------------------------------------------------------------------
# 3. [Plotly] 스피드 트레이스 (Interactive)
# -----------------------------------------------------------------------------
def get_speed_trace_data(year: int, race: str, driver1: str, driver2: str):
    """Plotly용 스피드 트레이스 객체 반환"""
    try:
        d1_code = _normalize_name(driver1)
        d2_code = _normalize_name(driver2)

        # [★ 수정] 헬퍼 함수로 세션 로드 (스피드 트레이스도 telemetry 필수!)
        session = _get_loaded_session(year, race, load_telemetry=True)

        l1 = session.laps.pick_driver(d1_code).pick_fastest()
        l2 = session.laps.pick_driver(d2_code).pick_fastest()
        
        if l1 is None or l2 is None: return None

        t1 = l1.get_telemetry().add_distance()
        t2 = l2.get_telemetry().add_distance()

        fig = go.Figure()

        c1 = fastf1.plotting.get_driver_color(d1_code, session=session)
        fig.add_trace(go.Scatter(
            x=t1['Distance'], y=t1['Speed'],
            mode='lines', name=d1_code,
            line=dict(color=c1, width=2),
            hovertemplate='Dist: %{x:.0f}m<br>Speed: %{y:.1f}km/h<extra></extra>'
        ))

        c2 = fastf1.plotting.get_driver_color(d2_code, session=session)
        fig.add_trace(go.Scatter(
            x=t2['Distance'], y=t2['Speed'],
            mode='lines', name=d2_code,
            line=dict(color=c2, width=2, dash='solid'),
            hovertemplate='Dist: %{x:.0f}m<br>Speed: %{y:.1f}km/h<extra></extra>'
        ))

        fig.update_layout(
            title=f"{year} {session.event['EventName']} Speed Trace (Fastest Lap): {d1_code} vs {d2_code}",
            xaxis_title="Distance (m)",
            yaxis_title="Speed (km/h)",
            template="plotly_dark",
            hovermode="x unified",
            height=500
        )
        return fig
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Speed Trace Error] {e}")
        return None
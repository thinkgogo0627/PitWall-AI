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

# Streamlit Cloud(/mount/src/)는 읽기 전용이므로 /tmp 사용
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
_local_cache = os.path.join(PROJECT_ROOT, 'data', 'cache')
try:
    os.makedirs(_local_cache, exist_ok=True)
    # 쓰기 가능한지 테스트
    _test = os.path.join(_local_cache, '.write_test')
    with open(_test, 'w') as f:
        f.write('ok')
    os.remove(_test)
    CACHE_DIR = _local_cache
except (PermissionError, OSError):
    CACHE_DIR = '/tmp/fastf1_cache'
    os.makedirs(CACHE_DIR, exist_ok=True)

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
    # 매핑에 없으면 그냥 3글자로 자르고 대문자로 (FastF1이 알아서 처리하길 기대)
    return clean_name.upper()[:3]

def _save_plot(filename, facecolor='black'):
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR, exist_ok=True)
    
    save_path = os.path.join(PLOT_DIR, filename)
    # Matplotlib 저장
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor=facecolor)
    plt.close()
    print(f"✅ 그래프 저장 완료: {save_path}")
    return f"GRAPH_GENERATED: {save_path}"

# -----------------------------------------------------------------------------
# 1. [Plotly] 랩타임 비교 (Interactive)
# -----------------------------------------------------------------------------
def get_race_pace_data(year: int, race: str, driver1: str, driver2: str):
    """Plotly용 데이터 객체를 반환합니다."""
    try:
        d1_code = _normalize_name(driver1)
        d2_code = _normalize_name(driver2)
        
        session = fastf1.get_session(year, race, 'R')
        session.load(telemetry=False, weather=False, messages=False)

        d1 = session.laps.pick_driver(d1_code)
        d2 = session.laps.pick_driver(d2_code)

        if d1.empty or d2.empty: return None

        # Plotly Figure 생성
        fig = go.Figure()

        # Driver 1
        c1 = fastf1.plotting.get_driver_color(d1_code, session=session)
        fig.add_trace(go.Scatter(
            x=d1['LapNumber'], y=d1['LapTime'].dt.total_seconds(),
            mode='lines+markers', name=d1_code,
            line=dict(color=c1, width=2),
            marker=dict(size=4)
        ))

        # Driver 2
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
        print(f"Error: {e}")
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
        session = fastf1.get_session(year, race, 'R')
        session.load(telemetry=True, weather=False, messages=False) # 텔레메트리 필수

        # 각 드라이버의 가장 빠른 랩 추출
        lap1 = session.laps.pick_drivers(driver1).pick_fastest()
        lap2 = session.laps.pick_drivers(driver2).pick_fastest()

        if lap1 is None or lap2 is None:
            return " 데이터 부족: 텔레메트리 분석을 위한 랩 데이터가 없습니다."

        # 텔레메트리 로드 및 'Distance' 축 추가
        tel1 = lap1.get_telemetry().add_distance()
        tel2 = lap2.get_telemetry().add_distance()

        # 데이터 보간 (Interpolation) - 두 드라이버의 위치를 맞추기 위함
        # 드라이버 1의 거리를 기준으로 드라이버 2의 속도를 보간합니다.
        interp_speed_d2 = np.interp(tel1['Distance'], tel2['Distance'], tel2['Speed'])
        
        # 속도 차이 계산 (양수면 D1이 빠름, 음수면 D2가 빠름)
        delta = tel1['Speed'] - interp_speed_d2

        # 트랙 좌표 (X, Y)와 세그먼트 생성
        x = np.array(tel1['X'].values)
        y = np.array(tel1['Y'].values)
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # 색상 지정
        color1 = fastf1.plotting.get_driver_color(driver1, session=session)
        color2 = fastf1.plotting.get_driver_color(driver2, session=session)
        
        # 세그먼트별 색상 배열 생성
        # D1이 빠르면 color1, D2가 빠르면 color2
        colors = [color1 if d > 0 else color2 for d in delta[:-1]]

        # 그래프 그리기
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='black')
        ax.set_facecolor('black')
        
        # LineCollection으로 트랙 그리기
        lc = LineCollection(segments, colors=colors, linewidths=5)
        ax.add_collection(lc)
        
        # 축 범위 설정 및 숨기기
        ax.autoscale_view()
        ax.set_aspect('equal')
        ax.axis('off')

        # 범례 및 타이틀 (커스텀)
        from matplotlib.lines import Line2D
        legend_lines = [Line2D([0], [0], color=color1, lw=4),
                        Line2D([0], [0], color=color2, lw=4)]
        ax.legend(legend_lines, [driver1, driver2], loc='upper right', facecolor='black', labelcolor='white')
        
        plt.title(f"{year} {race} Track Dominance\n({driver1} vs {driver2})", color='white', fontsize=15, fontweight='bold')

        filename = f"{year}_{race}_Dominance_{driver1}_vs_{driver2}.png".replace(" ", "_")
        return _save_plot(filename)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Dominance Map Error: {str(e)}"
    

# -----------------------------------------------------------------------------
# 3. [Plotly] 스피드 트레이스 (Interactive)
# -----------------------------------------------------------------------------
def get_speed_trace_data(year: int, race: str, driver1: str, driver2: str):
    """Plotly용 스피드 트레이스 객체 반환"""
    try:
        d1_code = _normalize_name(driver1)
        d2_code = _normalize_name(driver2)

        session = fastf1.get_session(year, race, 'R')
        session.load(telemetry=True, weather=False, messages=False)

        l1 = session.laps.pick_driver(d1_code).pick_fastest()
        l2 = session.laps.pick_driver(d2_code).pick_fastest()
        
        if l1 is None or l2 is None: return None

        t1 = l1.get_telemetry().add_distance()
        t2 = l2.get_telemetry().add_distance()

        fig = go.Figure()

        # Driver 1
        c1 = fastf1.plotting.get_driver_color(d1_code, session=session)
        fig.add_trace(go.Scatter(
            x=t1['Distance'], y=t1['Speed'],
            mode='lines', name=d1_code,
            line=dict(color=c1, width=2),
            hovertemplate='Dist: %{x:.0f}m<br>Speed: %{y:.1f}km/h<extra></extra>'
        ))

        # Driver 2
        c2 = fastf1.plotting.get_driver_color(d2_code, session=session)
        fig.add_trace(go.Scatter(
            x=t2['Distance'], y=t2['Speed'],
            mode='lines', name=d2_code,
            line=dict(color=c2, width=2, dash='solid'), # 점선보다는 실선 비교가 인터랙티브에선 나음
            hovertemplate='Dist: %{x:.0f}m<br>Speed: %{y:.1f}km/h<extra></extra>'
        ))

        fig.update_layout(
            title=f"{year} {race} Speed Trace (Fastest Lap): {d1_code} vs {d2_code}",
            xaxis_title="Distance (m)",
            yaxis_title="Speed (km/h)",
            template="plotly_dark",
            hovermode="x unified", # 마우스 올리면 둘 다 비교
            height=500
        )
        return fig
    except Exception as e:
        print(f"Error: {e}")
        return None
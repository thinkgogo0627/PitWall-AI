# app/tools/telemetry_data.py

import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import warnings
import seaborn as sns
import numpy as np

# 경고 무시 및 F1 스타일 설정
warnings.simplefilter(action='ignore', category=FutureWarning)
fastf1.plotting.setup_mpl(misc_mpl_mods=False)


# 현재 프로젝트 루트 기준: data/cache
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
CACHE_DIR = os.path.join(PROJECT_ROOT, 'data', 'cache')
PLOT_DIR = os.path.join(PROJECT_ROOT, 'data', 'plots')
fastf1.Cache.enable_cache(CACHE_DIR)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# FastF1 캐시 활성화
try:
    fastf1.Cache.enable_cache(CACHE_DIR)
    print(f" FastF1 Cache Enabled: {CACHE_DIR}")
except Exception as e:
    print(f" Cache Enable Failed: {e}")

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



# -----------------------------------------------------------------------------
# 1. 랩타임 비교 그래프 (기존 기능 + 공식 컬러 함수 적용)
# -----------------------------------------------------------------------------
def generate_lap_comparison_plot(year: int, race: str, driver1: str, driver2: str) -> str:
    try:
        driver1 = _normalize_name(driver1)
        driver2 = _normalize_name(driver2)

        print(f" [Compare] Loading Data: {year} {race} ({driver1} vs {driver2})...")
        session = fastf1.get_session(year, race, 'R')
        session.load(telemetry=False, weather=False, messages=False)

        d1_laps = session.laps.pick_driver(driver1)
        d2_laps = session.laps.pick_driver(driver2)

        if d1_laps.empty or d2_laps.empty:
            return f" 데이터 부족: {driver1} 혹은 {driver2}의 기록이 없습니다."

        plt.figure(figsize=(10, 6))
        plt.style.use('dark_background')

        # [수정] 공식 함수 사용 (identifier + session)
        color1 = fastf1.plotting.get_driver_color(driver1, session=session)
        color2 = fastf1.plotting.get_driver_color(driver2, session=session)

        sns.lineplot(x=d1_laps['LapNumber'], y=d1_laps['LapTime'].dt.total_seconds(), 
                     label=driver1, color=color1, linewidth=2)
        sns.lineplot(x=d2_laps['LapNumber'], y=d2_laps['LapTime'].dt.total_seconds(), 
                     label=driver2, color=color2, linewidth=2, linestyle='--')

        plt.title(f"{year} {race} Pace: {driver1} vs {driver2}", fontsize=14, fontweight='bold', color='white')
        plt.xlabel("Lap Number", color='white')
        plt.ylabel("Lap Time (s)", color='white')
        plt.legend()
        plt.grid(True, alpha=0.2)

        filename = f"{year}_{race}_Pace_{driver1}_vs_{driver2}.png".replace(" ", "_")
        return _save_plot(filename)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"

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
        lap1 = session.laps.pick_driver(driver1).pick_fastest()
        lap2 = session.laps.pick_driver(driver2).pick_fastest()

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
# 3. [NEW] 스피드 트레이스 (Speed Trace)
# -----------------------------------------------------------------------------
def generate_speed_trace_plot(year: int, race: str, driver1: str, driver2: str) -> str:
    try:
        driver1 = _normalize_name(driver1)
        driver2 = _normalize_name(driver2)

        print(f"📈 [Speed] Tracing: {year} {race} ({driver1} vs {driver2})...")
        session = fastf1.get_session(year, race, 'R')
        session.load(telemetry=True, weather=False, messages=False)

        l1 = session.laps.pick_driver(driver1).pick_fastest()
        l2 = session.laps.pick_driver(driver2).pick_fastest()
        if l1 is None or l2 is None: return "X 텔레메트리 데이터 부족 X"

        t1 = l1.get_telemetry().add_distance()
        t2 = l2.get_telemetry().add_distance()

        plt.figure(figsize=(10, 5))
        plt.style.use('dark_background')

        c1 = fastf1.plotting.get_driver_color(driver1, session=session)
        c2 = fastf1.plotting.get_driver_color(driver2, session=session)

        plt.plot(t1['Distance'], t1['Speed'], color=c1, label=driver1, linewidth=2)
        plt.plot(t2['Distance'], t2['Speed'], color=c2, label=driver2, linewidth=2, linestyle='--')

        plt.title(f"{year} {race} Speed Trace: {driver1} vs {driver2}", color='white', fontweight='bold')
        plt.xlabel("Distance (m)", color='white')
        plt.ylabel("Speed (km/h)", color='white')
        plt.legend()
        plt.grid(True, alpha=0.3)

        return _save_plot(f"{year}_{race}_Speed_{driver1}_vs_{driver2}.png")
    except Exception as e: return f"Error: {e}"

# 내부 저장 헬퍼 함수
def _save_plot(filename):
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR, exist_ok=True)
    
    save_path = os.path.join(PLOT_DIR, filename)
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f" 그래프 저장 완료: {save_path}")
    return f"GRAPH_GENERATED: {save_path}"

# 테스트
if __name__ == "__main__":
    # 도미넌스 맵 테스트 (2024 마이애미: 베르스타펜 vs 노리스)
    print(generate_track_dominance_plot(2025, "Miami", "VER", "NOR"))
    print(generate_lap_comparison_plot(2025, "Miami", "VER", "NOR"))
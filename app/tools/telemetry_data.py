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
# 1. 랩타임 비교 그래프 (기존 기능 + 공식 컬러 함수 적용)
# -----------------------------------------------------------------------------
def generate_lap_comparison_plot(year: int, race: str, driver1: str, driver2: str) -> str:
    try:
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
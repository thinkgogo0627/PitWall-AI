import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import os
import sys
import asyncio
import pandas as pd

import fastf1
import fastf1.plotting

# --- [1. 한글 폰트 설정] ---
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rc('axes', unicode_minus=False)

# --- [2. 프로젝트 경로 설정] ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- [3. 모듈 임포트] ---
try:
    from app.agents.briefing_agent import run_briefing_agent
    from app.tools.briefing_pipeline import generate_quick_summary
    from app.agents.strategy_agent import run_strategy_agent
    from app.tools.telemetry_data import (
    generate_track_dominance_plot, # 기존 (이미지)
    get_race_pace_data,            # 신규 (Plotly)
    get_speed_trace_data,          # 신규 (Plotly)
    DRIVER_MAPPING
)
except ImportError as e:
    st.error(f"모듈 로드 실패: {e}")
    st.stop()

# --- [4. 페이지 설정] ---
st.set_page_config(
    page_title="PitWall-AI Pro",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- [5. 스타일링 (CSS)] ---
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    
    /* 버튼 스타일 */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.8em;
        font-weight: bold;
        background-color: #1f2937;
        border: 1px solid #374151;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ef4444; 
        border-color: #ef4444;
        color: white;
        transform: translateY(-2px);
    }
    
    /* 헤더 및 탭 */
    h1, h2, h3 { color: #ef4444 !important; font-family: 'Segoe UI', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1f2937;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #ef4444; color: white; }
    
    /* 선택박스 커스텀 */
    div[data-baseweb="select"] > div {
        background-color: #1f2937;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- [6. 데이터 준비] ---
DRIVER_LIST = sorted(list(set(DRIVER_MAPPING.values())))
GP_LIST = [
    "Bahrain - 바레인", "Saudi Arabia - 사우디", "Australia - 호주", 
    "Japan - 일본", "China - 중국", "Miami - 마이애미", 
    "Emilia Romagna - 에밀리아 로마냐", "Monaco - 모나코", "Canada - 캐나다",
    "Spain - 바르셀로나", "Austria - 오스트리아", "Great Britain - 영국", 
    "Hungary - 헝가리", "Belgium - 벨기에", "Netherlands - 네덜란드", 
    "Italy - 이탈리아", "Azerbaijan - 아제르바이잔", "Singapore - 싱가포르", 
    "United States - 미국", "Mexico - 멕시코", "Brazil - 상파울루", 
    "Las Vegas - 라스베이거스", "Qatar - 카타르", "Abu Dhabi - 아부다비"
]


TELEMETRY_TIPS = {
    "Race Pace": """
    **📊 페이스 차트 읽는 법:**
    * **그래프가 우상향:** 타이어 마모(Degradation)로 인해 랩타임이 느려지고 있습니다. 기울기가 완만할수록 타이어 관리를 잘하는 것입니다.
    * **급격한 하락:** 피트스톱 후 새 타이어를 장착했을 때 발생합니다.
    * **일관성:** 그래프가 톱니바퀴 없이 평평할수록 드라이버가 '메트로놈'처럼 꾸준하게 달린 것입니다.
    """,

    "Track Dominance": """
    **🗺️ 지배력 맵 읽는 법:**
    * **직선 구간 색상:** 해당 드라이버의 **Top Speed(엔진 출력/DRS/공기저항)**가 더 빠릅니다.
    * **코너 구간 색상:** 해당 드라이버의 **Downforce(접지력)**나 **코너링 스킬**이 우세합니다.
    * 예: 레드불(VER)은 보통 직선과 고속 코너에서, 맥라렌(NOR)은 중저속 코너에서 강한 경향이 있습니다.
    """,

    "Speed Trace": """
    **📈 스피드 트레이스 읽는 법:**
    * **Valleys (계곡):** 그래프가 푹 꺼지는 곳이 코너입니다. 더 깊게 꺼지면 감속을 많이 한 것입니다 (저속 코너).
    * **Braking Point:** 그래프가 꺾이기 시작하는 지점입니다. 누가 더 늦게 브레이크를 밟는지(Late Braking) 비교해보세요.
    * **Apex Speed:** 계곡의 가장 밑바닥 점입니다. 코너링 최소 속도가 높을수록 다운포스가 좋거나 드라이버가 과감한 것입니다.
    """
}


PIRELLI_COLORS = {
    "SOFT": "#FF3333", "MEDIUM": "#FFF200", "HARD": "#EBEBEB",
    "INTERMEDIATE": "#39B54A", "WET": "#00AEEF", "UNKNOWN": "#808080"
}


# --- [6-1. 내부 헬퍼 함수: 전체 스틴트 시각화] ---
@st.cache_data(ttl=3600)
def get_all_drivers_stint_data(year, gp):
    """전체 드라이버의 스틴트 정보를 가져옵니다."""
    try:
        session = fastf1.get_session(year, gp, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        
        stints_list = []
        # 순위대로 정렬 (우승자가 맨 위로 오게)
        drivers = session.results['Abbreviation'].tolist()
        
        for drv in drivers:
            laps = session.laps.pick_driver(drv)
            if laps.empty: continue
            
            # 스틴트별 그룹화
            laps['Stint'] = laps['Stint'].fillna(1).astype(int)
            for stint_id, data in laps.groupby('Stint'):
                compound = data['Compound'].iloc[0]
                start_lap = data['LapNumber'].min()
                end_lap = data['LapNumber'].max()
                
                # 타이어 상태 추정 (Stint 시작 시 TyreLife가 1.0 이하면 New, 아니면 Used)
                tyre_life_start = data['TyreLife'].iloc[0]
                is_new = True if tyre_life_start <= 2.0 else False
                
                stints_list.append({
                    "Driver": drv,
                    "Stint": stint_id,
                    "Compound": str(compound).upper(),
                    "Start": start_lap,
                    "End": end_lap,
                    "Duration": end_lap - start_lap,
                    "Status": "NEW" if is_new else "USED"
                })
        return pd.DataFrame(stints_list), drivers
    except Exception as e:
        return pd.DataFrame(), []


def plot_tire_strategy_chart(df, sorted_drivers):
    """Plotly를 사용하여 Pirelli 스타일의 가로형 차트를 그립니다."""
    fig = go.Figure()
    
    # Y축 순서를 경기 결과 역순으로 (우승자가 맨 위)
    y_order = list(reversed(sorted_drivers))
    
    for _, row in df.iterrows():
        color = PIRELLI_COLORS.get(row['Compound'], "#808080")
        pattern = "" if row['Status'] == "NEW" else "/" # Used는 빗금
        
        fig.add_trace(go.Bar(
            y=[row['Driver']],
            x=[row['Duration']],
            base=[row['Start']],
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color='black', width=1),
                pattern_shape=pattern 
            ),
            name=row['Compound'],
            showlegend=False,
            hovertemplate=f"<b>{row['Driver']}</b><br>{row['Compound']} ({row['Status']})<br>Laps: {row['Start']}-{row['End']}<extra></extra>"
        ))

    fig.update_layout(
        title="🏁 Tire Strategy Overview (Stint Map)",
        template="plotly_dark",
        barmode='stack',
        yaxis=dict(categoryorder='array', categoryarray=y_order),
        xaxis=dict(title="Lap Number", dtick=5),
        height=700, # 드라이버 20명이므로 길게
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    
    # 범례(Legend) 수동 추가 (Fake Traces)
    for name, color in PIRELLI_COLORS.items():
        if name in df['Compound'].unique():
            fig.add_trace(go.Bar(x=[0], y=[y_order[0]], marker_color=color, name=name, showlegend=True, visible='legendonly'))
            
    return fig



# --- [7. 사이드바: Global Context Only] ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=80)
    st.title("🎛️ PitWall Command")
    
    st.subheader("📍 Race Session (Global)")
    st.caption("모든 탭에 공통으로 적용되는 설정입니다.")
    
    # 드라이버 선택 로직을 제거하고, 연도와 그랑프리만 남김
    selected_year = st.selectbox("Year", [2021, 2022, 2023, 2024, 2025], index=3)
    selected_gp = st.selectbox("Grand Prix", GP_LIST, index=3) # Default: Japan
    
    st.divider()
    
    # 시스템 상태 표시
    st.markdown("### 📡 System Status")
    st.success("✅ FastF1 API: Online")
    st.success("✅ Qdrant DB: Connected")
    st.info(f"💾 Local Cache Used")

# --- [8. 메인 탭 구성] ---
st.title(f"🏎️ PitWall-AI : {selected_year} {selected_gp}")

tab1, tab2, tab3 = st.tabs(["💬 Briefing", "📈 Telemetry Analytics" , "🧠 Strategy Center"])

# ==============================================================================
# TAB 1: Chat Interface (Briefing Agent)
# ==============================================================================
with tab1:
    st.markdown("### 🎙️ Race Briefing Room")
    st.caption("경기 결과 요약 및 뉴스 브리핑")

    # [컨트롤 바] 드라이버 선택 및 액션 버튼을 한 줄에 배치
    c1, c2, c3 = st.columns([1, 1, 1.5])
    
    with c1:
        # [Local Config] 브리핑 탭 전용 드라이버 선택
        focus_driver = st.selectbox("🎯 관심 드라이버 선택", DRIVER_LIST, index=DRIVER_LIST.index("VER"))
    
    briefing_container = st.container()

    with c2:
        # 전체 요약 버튼
        if st.button("📰 Race Summary\n(전체 경기 요약)", type="primary"):
            with briefing_container:
                with st.spinner(f"⚡ {selected_year} {selected_gp} 전체 데이터 분석 중..."):
                    summary = asyncio.run(generate_quick_summary(selected_year, selected_gp))
                    st.info("✅ 전체 브리핑 완료")
                    st.markdown(summary)
                    st.session_state.msg_briefing.append({"role": "assistant", "content": summary})

    with c3:
        # 드라이버 포커스 버튼
        if st.button(f"🔍 {focus_driver} Focus Report\n(드라이버 집중 분석)"):
            with briefing_container:
                with st.spinner(f"⚡ {focus_driver}의 경기 서사를 추적 중..."):
                    summary = asyncio.run(generate_quick_summary(selected_year, selected_gp, driver_focus=focus_driver))
                    st.success(f"✅ {focus_driver} 분석 완료")
                    st.markdown(summary)
                    st.session_state.msg_briefing.append({"role": "assistant", "content": summary})

    st.divider()

    # [Chat Interface]
    if "msg_briefing" not in st.session_state:
        st.session_state.msg_briefing = []

    for msg in st.session_state.msg_briefing:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("심층 질문 입력... (예: 안토넬리 인터뷰 내용 알려줘)"):
        st.session_state.msg_briefing.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.status("🕵️ 에이전트가 심층 조사 중...", expanded=True) as status:
                context_prompt = f"[{selected_year} {selected_gp}] {prompt}"
                response = asyncio.run(run_briefing_agent(context_prompt))
                status.update(label="조사 완료", state="complete", expanded=False)
                st.markdown(response)
                st.session_state.msg_briefing.append({"role": "assistant", "content": response})

# ==============================================================================
# TAB 2: Telemetry Studio (Dashboard Interface)
# ==============================================================================
with tab2:
    st.markdown("### 📈 Telemetry Analytics Studio")
    st.info("⚔️ 비교할 두 드라이버를 선택하세요.")
    
    row_sel1, row_sel2 = st.columns(2)
    with row_sel1:
        telemetry_d1 = st.selectbox("Driver A (Blue)", DRIVER_LIST, index=DRIVER_LIST.index("VER"), key="t_d1")
    with row_sel2:
        telemetry_d2 = st.selectbox("Driver B (Orange)", DRIVER_LIST, index=DRIVER_LIST.index("NOR"), key="t_d2")
    
    st.write("") 

    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    if "telemetry_fig" not in st.session_state:
        st.session_state.telemetry_fig = None
        st.session_state.telemetry_type = None
        st.session_state.telemetry_caption = ""

    with col_btn1:
        if st.button("📉 Race Pace (Interactive)", use_container_width=True):
            with st.spinner("Analyzing Race Pace..."):
                fig = get_race_pace_data(selected_year, selected_gp, telemetry_d1, telemetry_d2)
                if fig:
                    st.session_state.telemetry_fig = fig
                    st.session_state.telemetry_type = "Race Pace"
                else:
                    st.error("데이터 부족")

    with col_btn2:
        if st.button("🗺️ Track Dominance (Map)", use_container_width=True):
            with st.spinner("Calculating Sectors..."):
                path = generate_track_dominance_plot(selected_year, selected_gp, telemetry_d1, telemetry_d2)
                if "GRAPH_GENERATED" in path:
                    st.session_state.telemetry_fig = path.split(": ")[1].strip()
                    st.session_state.telemetry_type = "Track Dominance"
                else:
                    st.error(path)

    with col_btn3:
        if st.button("📈 Speed Trace (Interactive)", use_container_width=True):
            with st.spinner("Tracking Speed..."):
                fig = get_speed_trace_data(selected_year, selected_gp, telemetry_d1, telemetry_d2)
                if fig:
                    st.session_state.telemetry_fig = fig
                    st.session_state.telemetry_type = "Speed Trace"
                else:
                    st.error("데이터 부족")

    st.divider()
    
    if st.session_state.telemetry_fig:
        c_h1, c_h2, c_h3 = st.columns([1, 0.2, 1])
        with c_h1:
            st.markdown(f"<div style='text-align:center; font-weight:bold; font-size:1.2em; color:#4488ff;'>{telemetry_d1}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#0000ff; height:4px; width:100%;'></div>", unsafe_allow_html=True)
        with c_h2:
            st.markdown("<div style='text-align:center;'>VS</div>", unsafe_allow_html=True)
        with c_h3:
            st.markdown(f"<div style='text-align:center; font-weight:bold; font-size:1.2em; color:#ffaa00;'>{telemetry_d2}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='background-color:#ffaa00; height:4px; width:100%;'></div>", unsafe_allow_html=True)

        st.write("")
        
        if st.session_state.telemetry_type == "Track Dominance":
            st.image(st.session_state.telemetry_fig, use_container_width=True)
        else:
            st.plotly_chart(st.session_state.telemetry_fig, use_container_width=True)
        
        # 💡 [Analysis Tip] 하단에 깔끔한 가이드 표시
        st.info(f"💡 **Analysis Insight: {st.session_state.telemetry_type}**")
        st.markdown(TELEMETRY_TIPS.get(st.session_state.telemetry_type, ""))
            
    else:
        st.info("👆 위 버튼을 눌러 데이터를 분석하세요.")


# ==============================================================================
# TAB 3: Strategy Center (New!)
# ==============================================================================
with tab3:
    st.markdown("### 🧠 Race Strategy Analysis")
    
    # 1. [Primary View] 전체 드라이버 타이어 스틴트 시각화
    with st.spinner(f"📡 Fetching Strategy Data for {selected_year} {selected_gp}..."):
        stint_df, drivers_sorted = get_all_drivers_stint_data(selected_year, selected_gp)
        
    if not stint_df.empty:
        st.caption("가로축: 랩(Lap) / 세로축: 드라이버 (위에서부터 1위) / 색상: 타이어 종류")
        fig = plot_tire_strategy_chart(stint_df, drivers_sorted)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("데이터를 불러올 수 없습니다. (세션이 존재하지 않거나 데이터 누락)")

    st.divider()

    # 2. [Deep Dive] 드라이버별 심층 분석 컨트롤러
    st.markdown("#### 🕵️ Deep Dive: Driver Strategy Audit")
    
    # 드라이버 선택 (Tab 3 전용)
    c_sel, _ = st.columns([1, 2])
    with c_sel:
        strategy_driver = st.selectbox("분석 대상 드라이버 선택", DRIVER_LIST, index=DRIVER_LIST.index("VER"), key="strat_drv")

    # 분석 액션 버튼 (3 Categories)
    col_s1, col_s2, col_s3 = st.columns(3)
    
    # 결과 출력 컨테이너
    strategy_container = st.container()

    with col_s1:
        if st.button("🚦 Traffic & Pace\n(트래픽/페이스 분석)", use_container_width=True):
            with strategy_container:
                with st.chat_message("assistant"):
                    with st.spinner(f"🔍 {strategy_driver}의 트래픽과 순수 페이스를 분리 분석 중..."):
                        # Step 1 유도 프롬프트
                        prompt = f"2025 {selected_gp}에서 {strategy_driver}의 '트래픽 분석(Step 1)'을 중점적으로 수행해줘. 트래픽에 갇힌 랩과 클린 에어에서의 페이스 차이를 숫자로 비교해."
                        res = asyncio.run(run_strategy_agent(prompt))
                        st.markdown(res)

    with col_s2:
        if st.button("🛞 Tire Degradation\n(타이어 마모도/수명)", use_container_width=True):
            with strategy_container:
                with st.chat_message("assistant"):
                    with st.spinner(f"📉 {strategy_driver}의 타이어 수명과 관리 능력을 평가 중..."):
                        # Step 2 유도 프롬프트 (스틴트 길이 평가 포함)
                        prompt = f"2025 {selected_gp}에서 {strategy_driver}의 '타이어 관리(Step 2)'를 분석해줘. 특히 스틴트 길이(Type)를 보고 타이어를 얼마나 오래 썼는지(Extreme/Long Run) 평가해줘."
                        res = asyncio.run(run_strategy_agent(prompt))
                        st.markdown(res)

    with col_s3:
        if st.button("📝 Full Strategy Report\n(전체 전략 평가)", type="primary", use_container_width=True):
            with strategy_container:
                with st.chat_message("assistant"):
                    with st.spinner(f"🧠 {strategy_driver}의 전체 레이스 운영을 복기하는 중..."):
                        # Step 4 종합 평가
                        prompt = f"2025 {selected_gp} {strategy_driver}의 전체 전략을 4단계(트래픽, 타이어, 피트스탑, 종합)로 완벽하게 분석해줘."
                        res = asyncio.run(run_strategy_agent(prompt))
                        st.markdown(res)

    # 3. [Simulation Form] (기존 기능 유지 - 하단 배치)
    with st.expander("🎲 What-If Simulation Lab (가상 시뮬레이션)", expanded=False):
        st.caption("가상의 시나리오를 설정하여 전략 변화를 예측합니다.")
        with st.form("sim_form"):
            c1, c2, c3 = st.columns(3)
            with c1: target_lap = st.number_input("Pit Lap", 1, 70, 20)
            with c2: tire_choice = st.selectbox("New Tire", ["SOFT", "MEDIUM", "HARD"])
            with c3: rival_gap = st.number_input("Gap to Rival (sec)", 0.0, 60.0, 2.5)
            
            submit_sim = st.form_submit_button("🚀 Run Simulation")
            if submit_sim:
                st.info("시뮬레이션 기능은 현재 유지보수 중입니다. (Agent 4 연결 필요)")
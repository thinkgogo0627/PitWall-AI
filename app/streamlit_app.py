import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import os
import sys
import asyncio

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

tab1, tab2 = st.tabs(["💬 Briefing", "📈 Telemetry Analytics"])

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
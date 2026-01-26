import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
import asyncio

# --- [1. 한글 폰트 설정 (기존 코드 유지)] ---
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
    # 채팅용 에이전트 (뉴스/브리핑)
    from app.agents.briefing_agent import run_briefing_agent
    from app.tools.briefing_pipeline import generate_quick_summary

    # 시각화용 도구 (직접 호출하여 속도 향상)
    from app.tools.telemetry_data import (
        generate_lap_comparison_plot,
        generate_track_dominance_plot,
        generate_speed_trace_plot,
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
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3.5em;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #ff2b2b;
        color: #ff2b2b;
    }
    h1, h2, h3 {
        color: #ff2b2b !important; /* Ferrari Red style */
    }
</style>
""", unsafe_allow_html=True)

# --- [6. 데이터 준비] ---
# 드라이버 목록 (중복 제거 및 정렬)
DRIVER_LIST = sorted(list(set(DRIVER_MAPPING.values())))
GP_LIST = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China", "Miami", 
    "Emilia Romagna", "Monaco", "Canada", "Spain", "Austria", "Great Britain", 
    "Hungary", "Belgium", "Netherlands", "Italy", "Azerbaijan", "Singapore", 
    "United States", "Mexico", "Brazil", "Las Vegas", "Qatar", "Abu Dhabi"
]

# --- [7. 사이드바: 커맨드 센터] ---
with st.sidebar:
    st.title("🎛️ Command Center")
    st.caption("Setup your race context")
    st.divider()
    
    # 세션 설정
    st.subheader("📍 Race Session")
    selected_year = st.selectbox("Year", [2024, 2025], index=0)
    selected_gp = st.selectbox("Grand Prix", GP_LIST, index=3) # Default: Japan
    
    st.divider()
    
    # 드라이버 설정 (비교 분석용)
    st.subheader("⚔️ Driver Battle")
    col1, col2 = st.columns(2)
    with col1:
        driver_1 = st.selectbox("Driver A", DRIVER_LIST, index=DRIVER_LIST.index("VER"))
    with col2:
        driver_2 = st.selectbox("Driver B", DRIVER_LIST, index=DRIVER_LIST.index("NOR"))

    st.divider()
    st.info("💡 **Tip:** 왼쪽에서 설정한 값은 '텔레메트리 스튜디오' 탭에 즉시 반영됩니다.")

# --- [8. 메인 탭 구성] ---
st.title("🏎️ PitWall-AI : Professional Dashboard")

# 탭을 2개로 간소화하여 전문성 강화
# Tab 1: 채팅 (뉴스, 브리핑, 전략 질문)
# Tab 2: 시각화 (버튼으로 즉시 그래프 생성)
tab1, tab2 = st.tabs(["💬 Pit Wall Chat (브리핑/뉴스)", "📈 Telemetry Studio (데이터 분석)"])

# ==============================================================================
# TAB 1: Chat Interface (Briefing Agent)
# ==============================================================================
with tab1:
    st.markdown("### 🎙️ Race Briefing Room")
    
    # [섹션 1] Quick Action Buttons (파이프라인 적용 -> 초고속)
    col_b1, col_b2 = st.columns(2)
    
    briefing_container = st.container() # 결과가 나올 공간

    with col_b1:
        if st.button("📰 Race Summary\n(전체 경기 요약)", type="primary"):
            with briefing_container:
                with st.spinner(f"⚡ {selected_year} {selected_gp} 데이터를 병렬 분석 중..."):
                    # Agent 안 쓰고 파이프라인 직접 호출
                    summary = asyncio.run(generate_quick_summary(selected_year, selected_gp))
                    st.markdown(summary)
                    # 기록 저장
                    st.session_state.msg_briefing.append({"role": "assistant", "content": summary})

    with col_b2:
        if st.button(f"🏎️ {driver_1} Focus Report\n(내 드라이버 분석)"):
            with briefing_container:
                with st.spinner(f"⚡ {driver_1}의 서사를 추적 중..."):
                    summary = asyncio.run(generate_quick_summary(selected_year, selected_gp, driver_focus=driver_1))
                    st.markdown(summary)
                    st.session_state.msg_briefing.append({"role": "assistant", "content": summary})

    st.divider()

    # [섹션 2] Deep Dive Chat (기존 Agent -> 심층 질문용)
    st.caption("💬 더 궁금한 점이 있다면 대화로 질문하세요. (예: '안토넬리 인터뷰 내용 알려줘')")
    
    if "msg_briefing" not in st.session_state:
        st.session_state.msg_briefing = []

    for msg in st.session_state.msg_briefing:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("심층 질문 입력..."):
        st.session_state.msg_briefing.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.status("🕵️ 에이전트가 심층 조사 중...", expanded=True) as status:
                # 심층 질문은 기존처럼 Agent가 도구를 골라가며 수행
                context_prompt = f"[{selected_year} {selected_gp}] {prompt}"
                response = asyncio.run(run_briefing_agent(context_prompt))
                
                status.update(label="조사 완료", state="complete", expanded=False)
                st.markdown(response)
                st.session_state.msg_briefing.append({"role": "assistant", "content": response})

# ==============================================================================
# TAB 2: Telemetry Studio (Dashboard Interface)
# ==============================================================================
with tab2:
    st.markdown(f"### 📊 Analysis Target: {selected_year} {selected_gp}")
    st.markdown(f"**Comparing:** :red[{driver_1}] vs :orange[{driver_2}]")
    
    st.divider()

    # 3개의 메인 기능을 컬럼으로 배치
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    # 결과 이미지를 보여줄 컨테이너
    plot_container = st.container()

    # --- 버튼 1: 레이스 페이스 ---
    with col_btn1:
        if st.button("📉 Race Pace\n(랩타임 비교)"):
            with plot_container:
                with st.spinner("랩타임 데이터 분석 중..."):
                    result = generate_lap_comparison_plot(selected_year, selected_gp, driver_1, driver_2)
                    if "GRAPH_GENERATED" in result:
                        img_path = result.split(": ")[1].strip()
                        st.image(img_path, caption=f"Race Pace: {driver_1} vs {driver_2}", use_container_width=True)
                    else:
                        st.error(result)

    # --- 버튼 2: 트랙 도미넌스 ---
    with col_btn2:
        if st.button("🗺️ Track Dominance\n(서킷 지배력)"):
            with plot_container:
                with st.spinner("텔레메트리 & 섹터 계산 중..."):
                    result = generate_track_dominance_plot(selected_year, selected_gp, driver_1, driver_2)
                    if "GRAPH_GENERATED" in result:
                        img_path = result.split(": ")[1].strip()
                        st.image(img_path, caption=f"Track Dominance: {driver_1} vs {driver_2}", use_container_width=True)
                    else:
                        st.error(result)

    # --- 버튼 3: 스피드 트레이스 ---
    with col_btn3:
        if st.button("📈 Speed Trace\n(최고 속도)"):
            with plot_container:
                with st.spinner("속도 데이터 트래킹 중..."):
                    result = generate_speed_trace_plot(selected_year, selected_gp, driver_1, driver_2)
                    if "GRAPH_GENERATED" in result:
                        img_path = result.split(": ")[1].strip()
                        st.image(img_path, caption=f"Speed Trace: {driver_1} vs {driver_2}", use_container_width=True)
                    else:
                        st.error(result)

    st.caption("※ 데이터 출처: FastF1 (Live Telemetry). 첫 로딩 시 캐싱으로 인해 10~20초 소요될 수 있습니다.")
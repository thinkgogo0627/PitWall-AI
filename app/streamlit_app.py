import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import os
import sys
import asyncio
import pandas as pd
import json

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
    from app.agents.briefing_agent import run_briefing_agent, generate_quick_summary
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
GP_MAP = {
    "Bahrain - 바레인": "Bahrain",
    "Saudi Arabia - 사우디": "Saudi Arabia",
    "Australia - 호주": "Australia",
    "Japan - 일본": "Japan",
    "China - 중국": "China",
    "Miami - 마이애미": "Miami",
    "Emilia Romagna - 이몰라": "Emilia Romagna",
    "Monaco - 모나코": "Monaco",
    "Canada - 캐나다": "Canada",
    "Spain - 스페인": "Spain",
    "Austria - 오스트리아": "Austria",
    "British - 영국": "British", 
    "Hungary - 헝가리": "Hungary",
    "Belgium - 벨기에": "Belgium",
    "Netherlands - 네덜란드": "Netherlands",
    "Italy - 몬자": "Italy",
    "Azerbaijan - 바쿠": "Azerbaijan",
    "Singapore - 싱가포르": "Singapore",
    "United States - 오스틴": "United States",
    "Mexico - 멕시코": "Mexico",
    "Brazil - 브라질": "Brazil",
    "Las Vegas - 라스베이거스": "Las Vegas",
    "Qatar - 카타르": "Qatar",
    "Abu Dhabi - 아부다비": "Abu Dhabi"
}


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
    """
    [UI Upgrade] Pirelli Style Stint Map
    - 얇은 막대, 명확한 색상, 빗금 패턴 적용
    """
    fig = go.Figure()
    
    # 1. Pirelli 공식 컬러 코드
    PIRELLI_COLORS = {
        "SOFT": "#DA291C",    # 공식 레드
        "MEDIUM": "#FFD100",  # 공식 옐로우
        "HARD": "#F0F0F0",    # 공식 화이트 (배경이 어두우니 밝은 회색)
        "INTERMEDIATE": "#43B02A",
        "WET": "#0067A5"
    }
    
    # Y축 순서 (우승자가 위로)
    y_order = list(reversed(sorted_drivers))
    
    for _, row in df.iterrows():
        compound_key = row['Compound'].upper()
        color = PIRELLI_COLORS.get(compound_key, "#808080")
        
        # 2. 패턴 설정 (Used = 빗금)
        pattern_shape = "/" if row['Status'] == "USED" else ""
        
        # 3. 호버 텍스트 (상세 정보)
        hover_text = (
            f"<b>{row['Driver']}</b> (Stint {row['Stint']})<br>"
            f"Tyre: {row['Compound']} ({row['Status']})<br>"
            f"Laps: {row['Start']} ~ {row['End']} ({row['Duration']} Laps)"
        )

        fig.add_trace(go.Bar(
            y=[row['Driver']],
            x=[row['Duration']],
            base=[row['Start']],
            orientation='h',
            marker=dict(
                color=color,
                line=dict(color='#111111', width=1), # 막대 테두리 (구분선)
                pattern_shape=pattern_shape,
                pattern_solidity=0.5 # 빗금 진하기
            ),
            name=row['Compound'],
            hovertemplate=hover_text,
            showlegend=False
        ))

    # 4. 레이아웃 (Gap 줄이기 & 스타일링)
    fig.update_layout(
        title=dict(
            text="<b>🏁 Tire Strategy History</b>",
            font=dict(size=20, color="white")
        ),
        template="plotly_dark",
        barmode='stack',
        yaxis=dict(
            categoryorder='array', 
            categoryarray=y_order,
            tickfont=dict(size=12, color="white"),
            title=None
        ),
        xaxis=dict(
            title="Lap Number", 
            dtick=5, # 5랩 단위 눈금
            showgrid=True, 
            gridcolor='#333333',
            zeroline=False
        ),
        height=800,  # 드라이버 20명 기준 넉넉하게
        bargap=0.4,  # 막대 사이 간격 (얇고 세련되게)
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    # 범례 (가짜 트레이스 추가)
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
    # [수정] UI에는 Key(한글 포함)를 보여주고, 변수에는 Value(영어)를 저장
    _selected_gp_display = st.selectbox("Grand Prix", list(GP_MAP.keys()), index=11) # Great Britain Index
    selected_gp = GP_MAP[_selected_gp_display] # 실제로는 'Great Britain'만 변수에 담김
    
    st.caption(f"Target: {selected_year} {selected_gp}") # 디버깅용 확인 멘트
    
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
    
    # 1. 분석 대상 드라이버 선택
    c_driver, _ = st.columns([1, 2])
    with c_driver:
        briefing_driver = st.selectbox("분석 대상 드라이버 (Target Driver)", DRIVER_LIST, index=DRIVER_LIST.index("VER"), key="brf_driver")

    # 2. 액션 버튼 (3개 -> 2개로 축소)
    col_b1, col_b2 = st.columns(2)  # 컬럼 수 변경
    
    briefing_container = st.container(border=True)

    # [버튼 1] 전체 경기 요약
    with col_b1:
        if st.button("📰 Race Summary\n(전체 경기 요약)", type="primary", use_container_width=True):
            with briefing_container:
                with st.chat_message("assistant"):
                    with st.spinner(f"⚡ {selected_year} {selected_gp} 전체 세션을 분석 중..."):
                        # generate_quick_summary 호출
                        res = asyncio.run(generate_quick_summary(selected_year, selected_gp))
                        
                        st.markdown(res)
                        if "msg_briefing" not in st.session_state: st.session_state.msg_briefing = []
                        st.session_state.msg_briefing.append({"role": "assistant", "content": res})

    # [버튼 2] 드라이버 집중 분석 (이제 이것만 남김)
    with col_b2:
        if st.button(f"🏎️ {briefing_driver} Focus Report\n(드라이버 집중 분석)", use_container_width=True):
            with briefing_container:
                with st.chat_message("assistant"):
                    with st.spinner(f"⚡ {briefing_driver}의 시점에서 레이스를 추적 중..."):
                        # generate_quick_summary 호출 (driver_focus 사용)
                        res = asyncio.run(generate_quick_summary(selected_year, selected_gp, driver_focus=briefing_driver))
                        
                        st.markdown(res)
                        if "msg_briefing" not in st.session_state: st.session_state.msg_briefing = []
                        st.session_state.msg_briefing.append({"role": "assistant", "content": res})

    # [삭제됨] 버튼 3 (Incident Check) 코드는 완전히 제거했습니다.

    st.divider()
    # 3. 심층 대화 (Deep Dive Chat)
    # (기존 코드 유지)
    st.caption(f"💬 {briefing_driver} 또는 이번 경기에 대해 더 궁금한 점이 있다면 대화로 질문하세요.")
    
    if "msg_briefing" not in st.session_state:
        st.session_state.msg_briefing = []

    for msg in st.session_state.msg_briefing:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("심층 질문 입력..."):
        st.session_state.msg_briefing.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.status("🕵️ 에이전트가 심층 조사 중...", expanded=True) as status:
                context_prompt = f"[{selected_year} {selected_gp} - Focus Driver: {briefing_driver}] {prompt}"
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

# Helper 함수
def render_strategy_cards(json_str):
    """LLM이 뱉은 JSON 문자열을 예쁜 카드 UI로 변환합니다."""
    import json
    import re
    
    # 1. JSON 클리닝 (가끔 ```json ... ``` 을 붙일 때가 있음)
    try:
        cleaned_str = re.sub(r"```json|```", "", json_str).strip()
        data = json.loads(cleaned_str)
    except json.JSONDecodeError:
        st.error("🚨 데이터 파싱 실패: LLM이 올바르지 않은 JSON을 반환했습니다.")
        st.code(json_str) # 디버깅용 원본 출력
        return

    # 2. Verdict 별 색상 매핑
    verdict_colors = {
        "S": "#FFD700", # Gold
        "A": "#00FF00", # Green
        "B": "#00BFFF", # Blue
        "C": "#FFFF00", # Yellow
        "D": "#FF8C00", # Orange
        "F": "#FF0000"  # Red
    }

    # 3. 2x2 그리드로 카드 배치
    col1, col2 = st.columns(2)
    
    for i, item in enumerate(data):
        # 짝수는 왼쪽, 홀수는 오른쪽
        target_col = col1 if i % 2 == 0 else col2
        
        category = item.get("Category", "Analysis")
        metrics = item.get("Metrics", "-")
        insight = item.get("Insight", "No insight provided.")
        verdict = item.get("Verdict", "N/A")[0] # S, A, B... 첫 글자만 따옴
        
        color = verdict_colors.get(verdict, "#FFFFFF")

        with target_col:
            # CSS로 카드 스타일링 (Streamlit 컨테이너 활용)
            with st.container(border=True):
                # 헤더 (카테고리 + 등급 뱃지)
                c_head, c_badge = st.columns([3, 1])
                with c_head:
                    st.markdown(f"**{category}**")
                with c_badge:
                    st.markdown(f"<div style='text-align:center; background-color:{color}; color:black; font-weight:bold; border-radius:5px; padding:2px;'>{verdict} Rank</div>", unsafe_allow_html=True)
                
                st.divider()
                
                # 메트릭스 (강조)
                st.markdown(f"<div style='color:#aaaaaa; font-size:0.9em;'>📊 Metrics</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='font-size:1.1em; font-weight:bold; color:white;'>{metrics}</div>", unsafe_allow_html=True)
                
                st.write("") # Spacer
                
                # 인사이트
                st.markdown(f"<div style='color:#aaaaaa; font-size:0.9em;'>💡 Insight</div>", unsafe_allow_html=True)
                st.info(insight)



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
    
    # 드라이버 선택
    c_sel, _ = st.columns([1, 2])
    with c_sel:
        strategy_driver = st.selectbox("분석 대상 드라이버 선택", DRIVER_LIST, index=DRIVER_LIST.index("VER"), key="strat_drv")

    # --------------------------------------------------------------------------
    # [Helper Function] JSON 응답을 예쁜 데이터프레임으로 변환하여 출력
    # --------------------------------------------------------------------------
    def display_strategy_result(response_object):
        import json
        import pandas as pd
        import re
        
        # 1. 만능 텍스트 추출 (기존 유지)
        try:
            if hasattr(response_object, 'response'): final_text = response_object.response
            elif hasattr(response_object, 'content'): final_text = response_object.content
            elif isinstance(response_object, str): final_text = response_object
            else: final_text = str(response_object)
            if not isinstance(final_text, str): final_text = str(final_text)
        except Exception: final_text = str(response_object)

        # 2. JSON 파싱 & UI 렌더링
        try:
            match = re.search(r"\[.*\]", final_text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                df = pd.DataFrame(data)
                
                # --- [UI Upgrade] Dataframe 대신 커스텀 리스트 뷰 사용 ---
                st.write("") # Spacer
                
                # 헤더 그리기
                h1, h2, h3, h4 = st.columns([2, 2, 1.5, 1])
                h1.markdown("**분석 항목**")
                h2.markdown("**핵심 지표**")
                h3.markdown("**상세 리포트**")
                h4.markdown("**평가**")
                st.divider()
                
                # 행(Row) 반복 출력
                for _, row in df.iterrows():
                    c1, c2, c3, c4 = st.columns([2, 2, 1.5, 1])
                    
                    # 1. 카테고리
                    c1.markdown(f"**{row.get('Category', '-')}**")
                    
                    # 2. 지표
                    c2.caption(row.get('Metrics', '-'))
                    
                    # 3. [핵심] 상세 분석 (팝업 버튼)
                    with c3:
                        # 팝업 버튼 생성
                        with st.popover("📄 분석 보기", use_container_width=True):
                            st.markdown(f"### 💡 {row.get('Category', 'Analysis')}")
                            st.info(row.get('Insight', '내용 없음'))
                            
                    # 4. 평가 (뱃지 스타일)
                    verdict = row.get('Verdict', '-')
                    if "S" in verdict or "A" in verdict:
                        c4.success(f"🏆 {verdict}")
                    elif "F" in verdict or "D" in verdict:
                        c4.error(f"⚠️ {verdict}")
                    else:
                        c4.info(f"ℹ️ {verdict}")
                    
                    st.divider() # 행 구분선

                # 종합 평가가 있다면 하단에 크게 강조
                overall = df[df['Category'].str.contains("종합", case=False, na=False)] # 한글 '종합' 체크
                if not overall.empty:
                    v = overall.iloc[0].get('Verdict', '-')
                    i = overall.iloc[0].get('Insight', '-')
                    if "S" in v or "A" in v:
                        st.success(f"🏁 **종합 평가: {v}** | {i}")
                    else:
                        st.info(f"🏁 **종합 평가: {v}** | {i}")

            else:
                raise ValueError("No JSON found")

        except Exception as e:
            st.warning("⚠️ 분석 데이터를 표로 변환하는 중 문제가 발생했습니다. (Raw Text)")
            st.markdown(final_text)
    # --------------------------------------------------------------------------
    # [Action Buttons] 3가지 분석 모드
    # --------------------------------------------------------------------------
    col_s1, col_s2, col_s3 = st.columns(3)
    strategy_container = st.container()

    JSON_INSTRUCTION = """
    \n\n[IMPORTANT OUTPUT RULE]
    - You must return the result **ONLY** as a valid JSON Array.
    - No markdown formatting (no ```json).
    - No introductory or concluding text.
    - Example Format:
    [
        {"Category": "Traffic Analysis", "Metrics": "...", "Insight": "...", "Verdict": "B"},
        {"Category": "Tire Management", "Metrics": "...", "Insight": "...", "Verdict": "S"}
    ]
    """

    with col_s1:
        if st.button("🚦 Traffic & Pace\n(트래픽/페이스 분석)", use_container_width=True):
            with strategy_container:
                with st.chat_message("assistant"):
                    with st.spinner(f"🔍 {strategy_driver}의 트래픽과 순수 페이스를 분리 분석 중..."):
                        # [수정] 프롬프트 뒤에 JSON 지시사항 붙이기
                        base_prompt = f"2025 {selected_gp}에서 {strategy_driver}의 '트래픽 분석(Step 1)'을 수행해."
                        final_prompt = base_prompt + JSON_INSTRUCTION
                        
                        res = asyncio.run(run_strategy_agent(final_prompt))
                        display_strategy_result(res)

    with col_s2:
        if st.button("🛞 Tire Degradation\n(타이어 마모도/수명)", use_container_width=True):
            with strategy_container:
                with st.chat_message("assistant"):
                    with st.spinner(f"📉 {strategy_driver}의 스틴트별 상세 분석 중..."):
                        # 이제 간단하게 말해도 시스템 프롬프트 덕분에 알아듣습니다.
                        prompt = (
                            f"2025 {selected_gp}에서 {strategy_driver}의 타이어 전략을 분석해. "
                            "Rule: Break down by Stint 1, Stint 2, etc." 
                            + JSON_INSTRUCTION
                        )
                        res = asyncio.run(run_strategy_agent(prompt))
                        display_strategy_result(res)
                        
    with col_s3:
        if st.button("📝 Full Strategy Report\n(전체 전략 평가)", type="primary", use_container_width=True):
            with strategy_container:
                with st.chat_message("assistant"):
                    with st.spinner(f"🧠 {strategy_driver}의 전체 레이스 운영을 복기하는 중..."):
                        # [수정]
                        base_prompt = f"2025 {selected_gp} {strategy_driver}의 전체 전략을 4단계(트래픽, 타이어, 피트스탑, 종합)로 분석해."
                        final_prompt = base_prompt + JSON_INSTRUCTION
                        
                        res = asyncio.run(run_strategy_agent(final_prompt))
                        display_strategy_result(res)

    # 3. [Simulation Form] (Agent 4 연동 예정)
    with st.expander("🎲 What-If Simulation Lab (가상 시뮬레이션)", expanded=False):
        st.info("🚧 Agent 4 (Simulation) 연결 대기 중...")
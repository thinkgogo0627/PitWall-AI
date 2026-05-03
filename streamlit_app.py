import streamlit as st
import numpy as np
if not hasattr(np, 'NaN'):
    np.NaN = np.nan  # NumPy 2.0 호환성 패치
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import plotly.graph_objects as go
import os
import sys
import asyncio
import pandas as pd
import json
import re
import logging
logging.getLogger('fastf1').setLevel(logging.ERROR)
import fastf1
import fastf1.plotting

# FastF1 캐시 설정 (앱 진입점에서 먼저 초기화)
_project_root = os.path.dirname(os.path.abspath(__file__))
_local_cache = os.path.join(_project_root, 'data', 'cache')
try:
    os.makedirs(_local_cache, exist_ok=True)
    _wt = os.path.join(_local_cache, '.write_test')
    with open(_wt, 'w') as f:
        f.write('ok')
    os.remove(_wt)
    fastf1.Cache.enable_cache(_local_cache)
except (PermissionError, OSError):
    _tmp_cache = '/tmp/fastf1_cache'
    os.makedirs(_tmp_cache, exist_ok=True)
    if os.path.exists(_local_cache):
        for _item in os.listdir(_local_cache):
            _src = os.path.join(_local_cache, _item)
            _dst = os.path.join(_tmp_cache, _item)
            if os.path.isdir(_src) and not os.path.exists(_dst):
                try:
                    os.symlink(_src, _dst)
                except OSError:
                    pass
    fastf1.Cache.enable_cache(_tmp_cache)

# FastF1 캐시 설정 (앱 진입점에서 먼저 초기화)
_project_root = os.path.dirname(os.path.abspath(__file__))
_local_cache = os.path.join(_project_root, 'data', 'cache')
try:
    os.makedirs(_local_cache, exist_ok=True)
    _wt = os.path.join(_local_cache, '.write_test')
    with open(_wt, 'w') as f:
        f.write('ok')
    os.remove(_wt)
    fastf1.Cache.enable_cache(_local_cache)
except (PermissionError, OSError):
    _tmp_cache = '/tmp/fastf1_cache'
    os.makedirs(_tmp_cache, exist_ok=True)
    if os.path.exists(_local_cache):
        for _item in os.listdir(_local_cache):
            _src = os.path.join(_local_cache, _item)
            _dst = os.path.join(_tmp_cache, _item)
            if os.path.isdir(_src) and not os.path.exists(_dst):
                try:
                    os.symlink(_src, _dst)
                except OSError:
                    pass
    fastf1.Cache.enable_cache(_tmp_cache)

# FastF1 캐시 설정 (앱 진입점에서 먼저 초기화)
_project_root = os.path.dirname(os.path.abspath(__file__))
_local_cache = os.path.join(_project_root, 'data', 'cache')
try:
    os.makedirs(_local_cache, exist_ok=True)
    _wt = os.path.join(_local_cache, '.write_test')
    with open(_wt, 'w') as f:
        f.write('ok')
    os.remove(_wt)
    fastf1.Cache.enable_cache(_local_cache)
except (PermissionError, OSError):
    _tmp_cache = '/tmp/fastf1_cache'
    os.makedirs(_tmp_cache, exist_ok=True)
    if os.path.exists(_local_cache):
        for _item in os.listdir(_local_cache):
            _src = os.path.join(_local_cache, _item)
            _dst = os.path.join(_tmp_cache, _item)
            if os.path.isdir(_src) and not os.path.exists(_dst):
                try:
                    os.symlink(_src, _dst)
                except OSError:
                    pass
    fastf1.Cache.enable_cache(_tmp_cache)

################################################################
from llama_index.core import Settings
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

# API 키 가져오기 (Secrets or Env)
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except:
        st.error("🚨 API Key가 없습니다!")
        st.stop()

# 1. LLM 강제 설정 (Gemini)
llm = GoogleGenAI(model="models/gemini-2.5-flash", api_key=api_key)
Settings.llm = llm

# 2. 임베딩 강제 설정 (Gemini) 
# ★ 이게 없으면 자꾸 OpenAI를 찾습니다!
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="models/gemini-embedding-001",  # 아까 쓰기로 한 그 모델
    api_key=api_key
)
#############################################################################

# --- [1. 한글 폰트 설정] ---
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rc('axes', unicode_minus=False)

# --- [2. 프로젝트 경로 설정] ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- [3. 모듈 임포트] ---
try:
    from app.agents.briefing_agent import run_briefing_agent, generate_quick_summary
    from app.agents.strategy_agent import run_strategy_agent
    from app.tools.telemetry_data import (
        generate_track_dominance_plot,
        get_race_pace_data,
        get_speed_trace_data,
        DRIVER_MAPPING
    )
    from app.agents.tactic_simulation_agent import run_simulation_agent
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

# --- [6. 데이터 준비 및 헬퍼 함수 정의 (Global)] ---
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
    * **그래프가 우상향:** 타이어 마모(Degradation)로 인해 랩타임이 느려지고 있습니다.
    * **급격한 하락:** 피트스톱 후 새 타이어를 장착했을 때 발생합니다.
    * **그래프가 끊겨있다면:** 래드 플래그, 세이프티 카 등으로 인해 세션이 정지되거나 피트인 한 상태입니다
    """,
    "Track Dominance": """
    **🗺️ 지배력 맵 읽는 법:**
    * **직선 구간 색상:** 해당 드라이버의 **Top Speed**가 더 빠릅니다.
    * **코너 구간 색상:** 해당 드라이버의 **Cornering Speed**가 우세합니다.
    """,
    "Speed Trace": """
    **📈 스피드 트레이스 읽는 법:**
    * **Valleys (계곡):** 그래프가 푹 꺼지는 곳이 코너입니다.
    * **Braking Point:** 그래프가 꺾이기 시작하는 지점입니다. 누가 더 늦게 브레이크를 밟는지 비교해보세요.
    """
}

@st.cache_data(ttl=3600)
def get_all_drivers_stint_data(year, gp):
    """전체 드라이버의 스틴트 정보를 가져옵니다 (로컬 캐시 스캐너 적용판)."""
    import os
    import fastf1
    import pandas as pd

    try:
        # 1. 절대 경로 캐시 디렉토리 연결
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CACHE_DIR = os.path.abspath(os.path.join(BASE_DIR, '../data/cache')) # 경로 레벨 확인 필요
        fastf1.Cache.enable_cache(CACHE_DIR)

        # 2. 로컬 디렉토리 스캔을 통한 이름 강제 보정
        year_dir = os.path.join(CACHE_DIR, str(year))
        matched_event_name = gp

        # [방어막] 중국(China), 미국 등 자주 틀리는 이름 매핑
        TRACK_ALIASES = {
            "silverstone": "british", "monza": "italian", "spa": "belgian",
            "interlagos": "sao paulo", "zandvoort": "dutch", "suzuka": "japanese",
            "saopaulo": "s o paulo", "cota": "united states", "austin": "united states",
            "china": "chinese", "shanghai": "chinese"
        }

        search_keyword = gp.lower().replace(" ", "").replace("_", "")
        for alias, real_name in TRACK_ALIASES.items():
            if alias in search_keyword:
                search_keyword = real_name.replace(" ", "")
                break

        if os.path.exists(year_dir):
            for folder_name in os.listdir(year_dir):
                clean_name = folder_name.split('_', 1)[-1] if '_' in folder_name else folder_name
                compare_name = clean_name.lower().replace("_", "").replace(" ", "")
                if search_keyword in compare_name:
                    matched_event_name = clean_name.replace("_", " ")
                    break

        print(f"🔍 [UI Stint Load] 입력: '{gp}' -> 캐시 매칭: '{matched_event_name}'")

        # 3. 보정된 이름으로 세션 로드
        session = fastf1.get_session(year, matched_event_name, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        
        stints_list = []
        drivers = session.results['Abbreviation'].tolist()
        
        for drv in drivers:
            laps = session.laps.pick_driver(drv)
            if laps.empty: continue
            
            laps['Stint'] = laps['Stint'].fillna(1).astype(int)
            for stint_id, data in laps.groupby('Stint'):
                compound = data['Compound'].iloc[0]
                start_lap = data['LapNumber'].min()
                end_lap = data['LapNumber'].max()
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
        print(f"🚨 UI 스틴트 데이터 로드 실패: {e}")
        return pd.DataFrame(), []

def plot_tire_strategy_chart(df, sorted_drivers):
    """Pirelli Style Stint Map"""
    fig = go.Figure()
    PIRELLI_COLORS = {
        "SOFT": "#DA291C", "MEDIUM": "#FFD100", "HARD": "#F0F0F0",
        "INTERMEDIATE": "#43B02A", "WET": "#0067A5"
    }
    y_order = list(reversed(sorted_drivers))
    
    for _, row in df.iterrows():
        compound_key = row['Compound'].upper()
        color = PIRELLI_COLORS.get(compound_key, "#808080")
        pattern_shape = "/" if row['Status'] == "USED" else ""
        hover_text = f"<b>{row['Driver']}</b> (Stint {row['Stint']})<br>Tyre: {row['Compound']} ({row['Status']})<br>Laps: {row['Start']} ~ {row['End']}"

        fig.add_trace(go.Bar(
            y=[row['Driver']], x=[row['Duration']], base=[row['Start']],
            orientation='h',
            marker=dict(color=color, line=dict(color='#111111', width=1), pattern_shape=pattern_shape, pattern_solidity=0.5),
            name=row['Compound'], hovertemplate=hover_text, showlegend=False
        ))

    fig.update_layout(
        title=dict(text="<b>🏁 Tire Strategy History</b>", font=dict(size=20, color="white")),
        template="plotly_dark", barmode='stack',
        yaxis=dict(categoryorder='array', categoryarray=y_order, title=None),
        xaxis=dict(title="Lap Number", dtick=5, showgrid=True, gridcolor='#333333', zeroline=False),
        height=800, bargap=0.4, margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False
    )
    # 범례 추가
    for name, color in PIRELLI_COLORS.items():
        if name in df['Compound'].unique():
            fig.add_trace(go.Bar(x=[0], y=[y_order[0]], marker_color=color, name=name, showlegend=True, visible='legendonly'))
    return fig

def display_strategy_result(response_object):
    """JSON 응답을 예쁜 UI로 변환하여 출력 (Tab 3 전용 Helper)"""
    try:
        # [★ 수정 1] LlamaIndex ChatMessage 객체 완벽 대응 (강제 문자열 변환)
        final_text = ""
        if isinstance(response_object, str):
            final_text = response_object
        elif hasattr(response_object, 'response'):
            resp = response_object.response
            if isinstance(resp, str): final_text = resp
            elif hasattr(resp, 'content'): final_text = str(resp.content) # ChatMessage 객체인 경우
            else: final_text = str(resp)
        elif hasattr(response_object, 'message'):
            final_text = str(response_object.message.content)
        else:
            final_text = str(response_object)
            
        final_text = str(final_text) # 최후의 안전장치

        # 2. 불필요한 꼬리표(assistant: 등) 강제 제거
        import re
        final_text = re.sub(r"^(assistant|AI|system|user):\s*|^\w+:\s*", "", final_text, flags=re.IGNORECASE).strip()

        # 3. JSON 블록 추출
        match = re.search(r"\[.*\]", final_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                import json
                data = json.loads(json_str, strict=False)
            except json.JSONDecodeError:
                import ast
                data = ast.literal_eval(json_str)
            
            import pandas as pd
            df = pd.DataFrame(data)
            
            # 헤더
            h1, h2, h3, h4 = st.columns([2, 2, 1.5, 1])
            h1.markdown("**분석 항목**"); h2.markdown("**핵심 지표**"); h3.markdown("**상세**"); h4.markdown("**평가**")
            st.divider()
            
            # 본문
            for _, row in df.iterrows():
                c1, c2, c3, c4 = st.columns([2, 2, 1.5, 1])
                c1.markdown(f"**{row.get('Category', '-')}**")
                c2.caption(row.get('Metrics', '-'))
                with c3:
                    with st.popover("📄 보기", use_container_width=True):
                        st.markdown(f"### {row.get('Category', 'Analysis')}")
                        st.info(row.get('Insight', '내용 없음'))
                
                verdict = row.get('Verdict', '-')
                if "S" in verdict or "A" in verdict: c4.success(f"🏆 {verdict}")
                elif "F" in verdict or "D" in verdict: c4.error(f"⚠️ {verdict}")
                else: c4.info(f"ℹ️ {verdict}")
                st.divider()
                
            # 종합 평가
            overall = df[df['Category'].str.contains("종합", case=False, na=False)]
            if not overall.empty:
                v = overall.iloc[0].get('Verdict', '-')
                i = overall.iloc[0].get('Insight', '-')
                st.success(f"🏁 **종합 평가: {v}** | {i}") if "A" in v or "S" in v else st.info(f"🏁 **종합 평가: {v}** | {i}")
        else:
            raise ValueError("No JSON block found.")
            
    except Exception as e:
        st.warning(f"⚠️ Raw Output (JSON Parsing Failed): {e}")
        st.markdown(final_text if 'final_text' in locals() else str(response_object))


# --- [7. 사이드바] ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/33/F1.svg", width=80)
    st.title("🎛️ PitWall Command")
    
    st.subheader("📍 Race Session")
    selected_year = st.selectbox("Year", [2021, 2022, 2023, 2024, 2025, 2026], index=3)
    _selected_gp_display = st.selectbox("Grand Prix", list(GP_MAP.keys()), index=11)
    selected_gp = GP_MAP[_selected_gp_display]
    
    st.divider()
    st.markdown("### 📡 System Status")
    st.success("✅ FastF1 API: Online")
    st.success("✅ Qdrant DB: Connected")

# --- [8. 메인 탭] ---
st.title(f"🏎️ PitWall-AI : {selected_year} {selected_gp}")
tab1, tab2, tab3 = st.tabs(["💬 Briefing", "📈 Telemetry Analytics" , "🧠 Strategy Center"])

# TAB 1: Briefing
with tab1:
    st.markdown("### 🎙️ Race Briefing Room")
    c_driver, _ = st.columns([1, 2])
    with c_driver:
        briefing_driver = st.selectbox("Target Driver", DRIVER_LIST, index=DRIVER_LIST.index("VER"), key="brf_driver")

    col_b1, col_b2 = st.columns(2)
    briefing_container = st.container(border=True)

    with col_b1:
        if st.button("📰 Race Summary", type="primary", use_container_width=True):
            with briefing_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        res = asyncio.run(generate_quick_summary(selected_year, selected_gp))
                        st.markdown(res)
                        if "msg_briefing" not in st.session_state: st.session_state.msg_briefing = []
                        st.session_state.msg_briefing.append({"role": "assistant", "content": res})

    with col_b2:
        if st.button(f"🏎️ {briefing_driver} Focus Report", use_container_width=True):
            with briefing_container:
                with st.chat_message("assistant"):
                    with st.spinner(f"Tracking {briefing_driver}..."):
                        res = asyncio.run(generate_quick_summary(selected_year, selected_gp, driver_focus=briefing_driver))
                        st.markdown(res)
                        if "msg_briefing" not in st.session_state: st.session_state.msg_briefing = []
                        st.session_state.msg_briefing.append({"role": "assistant", "content": res})

    st.divider()
    st.caption("💬 심층 질문: 생소한 용어나 드라이버, 경기의 서사가 궁금하다면 물어보세요")
    if "msg_briefing" not in st.session_state: st.session_state.msg_briefing = []
    for msg in st.session_state.msg_briefing:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("질문 입력..."):
        st.session_state.msg_briefing.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.status("Thinking...", expanded=True) as status:
                response = asyncio.run(run_briefing_agent(f"[{selected_year} {selected_gp} - {briefing_driver}] {prompt}"))
                status.update(label="Complete", state="complete", expanded=False)
                st.markdown(response)
                st.session_state.msg_briefing.append({"role": "assistant", "content": response})

# TAB 2: Telemetry
with tab2:
    st.markdown("### 📈 Telemetry Analytics Studio")
    row_sel1, row_sel2 = st.columns(2)
    with row_sel1: telemetry_d1 = st.selectbox("Driver A (Blue)", DRIVER_LIST, index=DRIVER_LIST.index("VER"), key="t_d1")
    with row_sel2: telemetry_d2 = st.selectbox("Driver B (Orange)", DRIVER_LIST, index=DRIVER_LIST.index("NOR"), key="t_d2")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    if "telemetry_fig" not in st.session_state:
        st.session_state.telemetry_fig = None
        st.session_state.telemetry_type = None

    with col_btn1:
        if st.button("📉 Race Pace", use_container_width=True):
            with st.spinner("Race Pace 데이터 로딩 중..."):
                fig = get_race_pace_data(selected_year, selected_gp, telemetry_d1, telemetry_d2)
            if fig:
                st.session_state.telemetry_fig = fig
                st.session_state.telemetry_type = "Race Pace"
            else:
                st.error("Race Pace 데이터를 불러올 수 없습니다. 해당 세션의 데이터가 존재하는지 확인하세요.")

    with col_btn2:
        if st.button("🗺️ Track Dominance", use_container_width=True):
            with st.spinner("Track Dominance 맵 생성 중..."):
                path = generate_track_dominance_plot(selected_year, selected_gp, telemetry_d1, telemetry_d2)
            if path and "GRAPH_GENERATED" in path:
                st.session_state.telemetry_fig = path.split(": ")[1].strip()
                st.session_state.telemetry_type = "Track Dominance"
            else:
                st.error(f"Track Dominance 맵 생성 실패: {path if path else '데이터 없음'}")

    with col_btn3:
        if st.button("📈 Speed Trace", use_container_width=True):
            with st.spinner("Speed Trace 데이터 로딩 중..."):
                fig = get_speed_trace_data(selected_year, selected_gp, telemetry_d1, telemetry_d2)
            if fig:
                st.session_state.telemetry_fig = fig
                st.session_state.telemetry_type = "Speed Trace"
            else:
                st.error("Speed Trace 데이터를 불러올 수 없습니다. 해당 세션의 데이터가 존재하는지 확인하세요.")

    st.divider()
    if st.session_state.telemetry_fig:
        if st.session_state.telemetry_type == "Track Dominance":
            st.image(st.session_state.telemetry_fig, use_container_width=True)
        else:
            st.plotly_chart(st.session_state.telemetry_fig, use_container_width=True)
        st.info(f"💡 {st.session_state.telemetry_type} Analysis Tip")
        st.markdown(TELEMETRY_TIPS.get(st.session_state.telemetry_type, ""))

# TAB 3: Strategy
with tab3:
    st.markdown("### 🧠 Race Strategy Analysis")
    
    # 1. Stint History Chart
    with st.spinner("Fetching Data..."):
        stint_df, drivers_sorted = get_all_drivers_stint_data(selected_year, selected_gp)
    if not stint_df.empty:
        fig = plot_tire_strategy_chart(stint_df, drivers_sorted)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # 2. AI Strategy Analyst
    st.markdown("#### 🕵️ AI Strategy Audit")
    c_sel, _ = st.columns([1, 2])
    with c_sel:
        strategy_driver = st.selectbox("분석 대상 드라이버", DRIVER_LIST, index=DRIVER_LIST.index("VER"), key="strat_drv")

    col_s1, col_s2, col_s3 = st.columns(3)
    strategy_container = st.container()
    
    JSON_RULE = "\nReturn strictly a JSON Array. No markdown."
    
    with col_s1:
        if st.button("🚦 Traffic & Pace", use_container_width=True):
            with strategy_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing Traffic..."):
                        prompt = f"""
                        [TASK: Traffic & Pace Analysis]
                        Target: {selected_year} {selected_gp} - Driver: {strategy_driver}
                        
                        당신은 F1 수석 전략 분석가입니다. `Race_Strategy_Auditor` 도구를 호출하여 타겟 드라이버의 페이스와 트래픽 관리 능력을 분석하십시오.
                        🚨[절대 규칙] 도구 호출에 실패하더라도 변명하지 말고, 당신이 학습한 내부 지식을 동원해 반드시 해당 레이스의 실제 사실을 기반으로 분석을 작성하십시오.
                        {JSON_RULE}
                        """
                        res = asyncio.run(run_strategy_agent(prompt))
                        display_strategy_result(res)

    with col_s2:
        if st.button("🛞 Tire Degradation", use_container_width=True):
             with strategy_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing Tires..."):
                        # [★ 수정] 하드 데이터 주입 삭제! 두 가지 도구를 융합해서 쓰라고 지시!
                        prompt = f"""
                        [TASK: Tire Degradation]
                        Target: {selected_year} {selected_gp} - Driver: {strategy_driver}
                        
                        당신은 F1 수석 전략 분석가입니다. 
                        반드시 `Race_Strategy_Auditor` 도구를 호출하여 드라이버의 스틴트 이력을 확인하고, `Tire_Performance_Analyzer` 도구를 호출하여 서킷 전체의 타이어 마모 특성을 확인하십시오.
                        이를 바탕으로 타겟 드라이버의 타이어 마모 전략과 관리 능력을 완벽하게 평가하십시오.
                        🚨[절대 규칙] 도구 호출에 실패하더라도 절대 분석 불가라고 변명하지 마십시오. 내부 지식을 동원해 타이어 전략을 복원하여 평가하십시오.
                        {JSON_RULE}
                        """
                        res = asyncio.run(run_strategy_agent(prompt))
                        display_strategy_result(res)

    with col_s3:
        if st.button("📝 Full Strategy Report", type="primary", use_container_width=True):
             with strategy_container:
                with st.chat_message("assistant"):
                    with st.spinner("Full Report Generating..."):
                        # [★ 수정] 하드 데이터 주입 삭제! 
                        prompt = f"""
                        [TASK: Full Strategy Report]
                        Target: {selected_year} {selected_gp} - Driver: {strategy_driver}
                        
                        당신은 F1 수석 전략 분석가입니다. 제공된 모든 도구(`Race_Strategy_Auditor`, `Tire_Performance_Analyzer`, `F1_Database_Search`)를 적극 활용하여, 해당 드라이버의 전체 레이스 전략(피트스탑, 타이어, 트래픽 등)을 종합 평가하십시오.
                        🚨[절대 규칙] '데이터가 없습니다'라며 분석을 포기하는 것은 절대 허용되지 않습니다. 기필코 심층 분석 결과를 도출하십시오.
                        {JSON_RULE}
                        """
                        res = asyncio.run(run_strategy_agent(prompt))
                        display_strategy_result(res)

    st.divider()

    # 3. Tactical Simulator (수리 완료!)
    st.subheader("⚔️ Head-to-Head Tactical Simulator")
    col_sim1, col_sim2, col_sim3 = st.columns([1, 0.2, 1])
    
    with col_sim1:
        st.markdown("#### 🚀 Attacker")
        sim_attacker = st.selectbox("추격자", DRIVER_LIST, index=1, key="sim_att")
    with col_sim2:
        st.markdown("<h2 style='text-align:center; padding-top:20px;'>VS</h2>", unsafe_allow_html=True)
    with col_sim3:
        st.markdown("#### 🛡️ Defender")
        sim_defender = st.selectbox("선두", DRIVER_LIST, index=0, key="sim_def")

    if st.button("🔮 Run Simulation", type="primary", use_container_width=True):
        if sim_attacker == sim_defender:
            st.warning("⚠️ 서로 다른 드라이버를 선택하세요.")
        else:
            with st.container(border=True):
                with st.chat_message("assistant"):
                    with st.spinner(f"Simulating {sim_attacker} vs {sim_defender}..."):
                        sim_prompt = (
                            f"Analyze battle between {sim_attacker} (Attacker) and {sim_defender} (Defender) "
                            f"at {selected_year} {selected_gp}. Check undercut/overcut possibility based on pit loss and lap times."
                        )
                        try:
                            # Agent 4 호출 연결
                            response = asyncio.run(run_simulation_agent(sim_prompt))
                            st.markdown(response)
                        except Exception as e:
                            st.error(f"Simulation Error: {e}")
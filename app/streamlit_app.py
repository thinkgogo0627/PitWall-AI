# -*- coding: utf-8 -*-
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys
import asyncio

# --- [1. 한글 폰트 설정 (리눅스 정석 방법)] ---
# apt-get install fonts-nanum 으로 설치된 경로
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

# 폰트 파일이 있는지 확인
if os.path.exists(font_path):
    # 폰트 매니저에 강제 등록
    fm.fontManager.addfont(font_path)
    
    # 등록된 폰트 이름으로 설정
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rc('axes', unicode_minus=False)
else:
    # 폰트가 없으면 경고 (하지만 1단계에서 설치했으니 무조건 있어야 함)
    st.warning("나눔고딕 폰트가 설치되지 않았습니다. 터미널에서 'sudo apt-get install fonts-nanum'을 실행하세요.")

# --- [2. 프로젝트 모듈 설정] ---
# (여기부터는 기존 코드와 동일합니다. 그대로 두세요.)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from app.agents.strategy_agent import run_strategy_agent
    from app.agents.circuit_agent import run_circuit_agent
    from app.agents.briefing_agent import run_briefing_agent
except ImportError as e:
    st.error(f"에이전트 모듈 로드 실패: {e}")
    st.stop()

# --- [3. 페이지 설정] ---
st.set_page_config(page_title="PitWall AI", layout="wide")

with st.sidebar:
    st.title("PitWall AI")
    st.markdown("""
    **F1 데이터 기반 전략 어시스턴트**
    
    각 탭은 독립된 대화 세션을 가집니다.
    """)
    st.divider()
    st.caption("Powered by Gemini 2.5 & FastF1")


# --- [4. 세션 상태 초기화 (대화방 분리)] ---
# chat_history가 없으면, 3개의 방을 가진 딕셔너리로 초기화
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {
        "Strategy": [],
        "Circuit": [],
        "Briefing": []
    }

# --- [5. 메인 로직 함수화] ---
# 코드를 깔끔하게 하기 위해, 각 탭의 내용을 그리는 함수를 만듭니다.
def render_tab_content(mode_name, agent_func, description):
    st.caption(description)
    
    # 1. 해당 모드의 대화 기록만 가져오기
    history = st.session_state["chat_history"][mode_name]

    # 2. 채팅 기록 출력
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 3. 입력창 (Key를 모드별로 다르게 줘서 충돌 방지)
    if prompt := st.chat_input(f"{mode_name} 질문 입력...", key=f"input_{mode_name}"):
        # 사용자 메시지 즉시 표시
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 기록에 추가
        st.session_state["chat_history"][mode_name].append({"role": "user", "content": prompt})

        # 에이전트 호출 및 응답 표시
        with st.chat_message("assistant"):
            with st.spinner(f"{mode_name} AI 분석 중..."):
                try:
                    # 비동기 실행
                    response = asyncio.run(agent_func(prompt))
                    st.markdown(response)
                    
                    # 응답 기록에 추가
                    st.session_state["chat_history"][mode_name].append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"에러 발생: {e}")

# --- [6. 탭 구성 및 실행] ---
st.title("PitWall AI")

# 탭 생성
tab1, tab2, tab3 = st.tabs(["레이스 전략 (Strategy)", "서킷 가이드 (Circuit)", "경기 브리핑 (Briefing)"])

# 각 탭 내부에서 함수 호출
with tab1:
    render_tab_content(
        "Strategy", 
        run_strategy_agent, 
        "실시간 랩타임 분석, 드라이버 배틀, 페이스 비교 등 '숫자' 기반 전략을 분석합니다."
    )

with tab2:
    render_tab_content(
        "Circuit", 
        run_circuit_agent, 
        "서킷의 코너 특성, 타이어 마모도, 다운포스 요구량 등 '기술적 정보'를 제공합니다."
    )

with tab3:
    render_tab_content(
        "Briefing", 
        run_briefing_agent, 
        "경기 결과 요약, 리타이어 원인, 주요 이슈 등 '뉴스와 팩트'를 브리핑합니다."
    )

# --- [7. 하단 풋터] ---
st.divider()
st.caption("Data sources: FastF1 (Telemetry), DuckDuckGo (News), Gemini 1.5 (Reasoning)")
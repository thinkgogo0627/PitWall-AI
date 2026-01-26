## 전술 시뮬레이터

## analytics.py에 만든 함수 개별적으로 꺼내서 쓰기

import sys
import os
import asyncio
import logging
from dotenv import load_dotenv

# LlamaIndex & AI Imports
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError

# FastF1 & Analytics Imports
import fastf1
# 경로 설정 (프로젝트 루트 참조)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_pipeline.analytics import (
    get_specific_pit_loss,
    get_pit_loss_time,
    calculate_slope,
    audit_extension,
    audit_opportunity
)

# 로깅 설정 (FastF1 경고 숨기기)
logging.getLogger('fastf1').setLevel(logging.WARNING)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = GoogleGenAI(model="models/gemini-2.5-pro", api_key=GOOGLE_API_KEY)
Settings.llm = llm

## --- [2. 도구(Tool) 정의: 전술 시뮬레이션] ---

def run_tactical_simulation(year: int, circuit: str, driver_identifier: str, rival_identifier: str = None) -> str:
    """
    [Sim Tool] 특정 드라이버의 피트스탑 전술(언더컷/오버컷/스테이아웃)을 정밀 시뮬레이션합니다.
    - driver_identifier: 분석할 대상 드라이버 (번호 권장)
    - rival_identifier: (옵션) 1:1 언더컷 싸움을 분석할 상대 드라이버
    """
    print(f"\n [Sim] 전술 시뮬레이션 가동: {driver_identifier} vs {rival_identifier}")
    
    # 1. 세션 로드
    try:
        session = fastf1.get_session(year, circuit, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        return f"데이터 로드 실패: {e}"

    # 2. 드라이버 데이터 추출
    # (드라이버 번호/이름 매핑은 FastF1 내부적으로 어느 정도 처리되지만, 안전하게 문자열로 변환)
    driver_id = str(driver_identifier)
    
    try:
        driver_laps = session.laps.pick_driver(driver_id)
    except KeyError:
        return f"드라이버 '{driver_id}' 데이터를 찾을 수 없습니다."
    
    # 3. 피트 스탑 찾기
    pit_stops = driver_laps[driver_laps['PitIn'] == True]
    if pit_stops.empty:
        return "해당 드라이버는 피트 스탑을 하지 않았습니다 (No-Stop or DNF)."

    report = f"###  Tactical Analysis: Driver {driver_id} ({year} {circuit})\n"
    
    # 트랙 기본 피트 로스 (백업용)
    track_avg_loss = get_pit_loss_time(session)

    # 4. 각 피트 스탑별 시뮬레이션
    for idx, pit_row in pit_stops.iterrows():
        pit_lap = int(pit_row['LapNumber'])
        
        # 1. 실제 로스 계산
        real_pit_loss = get_specific_pit_loss(driver_laps, pit_lap, track_avg_loss)
        report += f"\n**[Pit Stop @ Lap {pit_lap}]** (Actual Loss: {real_pit_loss}s)\n"
        
        # B. 방어 기회 분석 (Extension Audit) - 더 버티는 게 나았나?
        # 직전 5랩의 기울기(Degradation) 계산
        past_laps = driver_laps[driver_laps['LapNumber'].between(pit_lap - 5, pit_lap - 1)]
        slope = calculate_slope(past_laps)
        
        ext_result = audit_extension(driver_laps, pit_lap, slope, real_pit_loss)
        if ext_result:
            report += f"- **Defense/Stint:** {ext_result['verdict']} ({ext_result['desc']})\n"
        
        # C. 공격 기회 분석 (Opportunity Audit) - 언더컷 가능했나?
        # 라이벌이 지정되지 않았으면, 당시 앞차를 자동으로 감지해서 분석
        target_rival = str(rival_identifier) if rival_identifier else None
        
        opp_result = audit_opportunity(session, driver_id, pit_lap, real_pit_loss, target_rival_id=target_rival)
        
        if opp_result:
             # 결과 포맷팅 (Telemetry 데이터가 딕셔너리로 오므로 예쁘게 풀어서 출력)
             t = opp_result.get("telemetry", {})
             s = opp_result.get("simulation", {})
             
             report += f"""
             - **Attack Target:** {opp_result.get('rival')}
               - **Gap to Target:** {t.get('gap_to_rival', 'N/A')}s
               - **Net Margin:** {s.get('net_margin', 'N/A')}s (Negative is Good)
               - **Success Prob:** {s.get('probability', 0)}%
             """
        else:
            report += "- **Attack Analysis:** 분석 대상(Rival)을 특정할 수 없거나 데이터가 부족합니다.\n"

    return report

# 도구 래핑
sim_tool = FunctionTool.from_defaults(
    fn=run_tactical_simulation,
    name="Tactical_Simulator",
    description="드라이버의 피트 스탑 타이밍을 분석하여 언더컷 성공 여부, 스틴트 연장 손익을 시뮬레이션합니다. 2025년 미래 데이터도 분석 가능합니다."
)

# --- [에이전트 조립] ---

def build_simulation_agent():
    tools = [sim_tool]
    
    driver_map = """
    [Driver Mapping Reference]
    - Max Verstappen (막스 베르스타펜, VER): 1

    - Yuki Tsunoda (유키 츠노다, TSU): 22

    - Lando Norris (랜도 노리스, NOR): 4

    - Oscar Piastri (오스카 피아스트리, PIA): 81

    - Lewis Hamilton (루이스 해밀턴, HAM): 44

    - Charles Leclerc (샤를 르클레르, LEC): 16

    - George Russell (조지 러셀, RUS): 63

    - Kimi Antonelli (키미 안토넬리, ANT): 12  

    - Liam Lawson (리암 로슨, LAW): 30

    - Isack Hadjar (아이작 하자르, HAD): 6

    - Gabriel Bortoleto (가브리엘 보톨레토, BOR): 5

    - Nico Hülkenberg (니코 훌켄베르크, HUL): 27

    - Franco Colapinto (프랑코 콜라핀토, COL): 43

    - Pierre Gasly (피에르 가슬리, GAS): 10

    - Alex Albon (알렉스 알본, ALB): 23

    - Carlos Sainz (카를로스 사인츠, SAI): 55

    - Lance Stroll (랜스 스트롤, STR): 18

    - Fernando Alonso (페르난도 알론소, ALO): 14

    - Esteban Ocon (에스테반 오콘, OCO): 31

    - Olliver Bearman (올리버 베어만, BEA): 87
    """

    system_prompt = f"""
    당신은 F1 팀의 '전술 시뮬레이션 엔지니어(Tactical Engineer)'입니다.
    
    [Mission]
    사용자의 "만약에(What-If)" 질문에 대해 수학적 시뮬레이션 결과로 답해야 합니다.
    감성적인 표현은 배제하고, **초(Seconds), 랩(Lap), 갭(Gap)** 데이터 위주로 보고하세요.

    [Output Format]
    답변은 반드시 아래 형식을 포함해야 합니다
    1. Rival(누구를 추월하려 했는가?)
    2. Telemetry Data
        - Gap to Rival: 0.00s (피트 인 직전 차이)
        - Undercut Poser: -0.00s (새 타이어로 번 시간)
        - Pit Loss Delta: 0.00s (피트스탑 손익)
    3. Simulation Result
        - Predicted Margin: 0.00s (+면 뒤, -면 앞)
        - Sucess Probability: 00%

    4. Engineer's Comment & Driver's perfomance
        - 수식에 근거한 판정
        - 트래픽, 에어로 차이, 드라이버의 역량에 따라 계산한 결과와 실제 일어난 일이 다를 수 있음을 시사할 것
    
    [SYSTEM: 2030 SIMULATION MODE]
    1. 현재는 2030년입니다. 2025년 데이터는 과거 기록입니다. "미래라 모른다"는 금지입니다.
    2. 무조건 `Tactical_Simulator` 도구를 사용하여 데이터를 확인하십시오.
    3. 질문에 드라이버 이름이 있으면 반드시 **번호(Number)**로 변환하여 도구에 입력하세요.
    
    {driver_map}
    """
    
    return ReActAgent(
        llm=llm,
        tools=tools,
        context=system_prompt,
        verbose=True
    )


# --- 실행 래퍼 ---
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(ServerError),
    reraise=True
)
async def run_simulation_agent(user_msg: str):
    # agent 생성, 컨텍스트 설정, 실행결과 및 결과 반환
    agent = build_simulation_agent()
    ctx = Context(agent)
    return await agent.run(user_msg = user_msg, ctx = ctx)


# 테스트 실행
if __name__ == "__main__":
    async def test():
        print("시뮬레이션, 언더컷 성공 여부, 스틴트 연장 손익 에이전트 테스트 실행")
        q = '2025년 멕시코 그랑프리에서 러셀(63번)이 베어만(87번)을 추월할 기회가 있었는지 언더컷 성공 여부, 피트스탑 타이밍 등을 분석해서 숫자 데이터를 근거로 판정해줘'
        print(f"\nUser: {q}")

        try:
            # 전역함수 호출
            response = await run_simulation_agent(q)
            print(f"\nPitWall(Simulation): {response}")
        except Exception as e:
            print(f"\n Final Error: {e}")

    asyncio.run(test())

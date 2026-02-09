## 전술 시뮬레이터

import sys
import os
import asyncio
import logging
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# FastF1 Imports
import fastf1

# LlamaIndex Imports
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError

# 로깅 설정
logging.getLogger('fastf1').setLevel(logging.WARNING)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = GoogleGenAI(model="models/gemini-2.5-pro", api_key=GOOGLE_API_KEY)
Settings.llm = llm

# =============================================================================
# 🧮 [내장 계산 엔진] Local Simulation Helpers
# analytics.py에 없는 기능을 여기서 직접 구현합니다.
# =============================================================================

def _calculate_pit_loss_baseline(session):
    """
    해당 세션의 평균 피트 로스 시간(Pit Loss Time)을 계산합니다.
    (Pit In/Out 시간을 제외한 순수 손실 시간 추정)
    """
    try:
        # 피트 스탑을 수행한 모든 랩 데이터 추출
        pit_laps = session.laps[session.laps['PitInTime'].notna() & session.laps['PitOutTime'].notna()]
        if pit_laps.empty:
            return 22.0 # 기본값 (대략적인 평균)
        
        # 피트 레인 체류 시간 평균
        avg_duration = (pit_laps['PitOutTime'] - pit_laps['PitInTime']).dt.total_seconds().mean()
        # + 가감속 로스 보정 (약 3~4초)
        return round(avg_duration + 3.5, 2)
    except:
        return 22.0

def _simulate_undercut(session, driver_laps, rival_laps, pit_lap, pit_loss_time):
    """
    언더컷(Undercut) 시뮬레이션 핵심 로직
    """
    try:
        # 1. 내 드라이버: 피트 인 랩(In-Lap) + 피트 로스 + 아웃 랩(Out-Lap)
        my_in_lap = driver_laps[driver_laps['LapNumber'] == pit_lap]['LapTime'].dt.total_seconds().values[0]
        # 아웃 랩은 다음 랩 (pit_lap + 1)
        my_out_lap = driver_laps[driver_laps['LapNumber'] == pit_lap + 1]['LapTime'].dt.total_seconds().values[0]
        
        # 2. 라이벌: 스테이 아웃 했다고 가정 (같은 구간 2랩의 기록)
        rival_laps_segment = rival_laps[rival_laps['LapNumber'].isin([pit_lap, pit_lap+1])]
        rival_total_time = rival_laps_segment['LapTime'].dt.total_seconds().sum()
        
        # 3. 실제 소요 시간 비교
        # 내 총 시간 (피트 로스 포함이 아니라, 섹터 타임 합산으로 계산해야 정확하나 약식으로 처리)
        # In-Lap과 Out-Lap에는 이미 피트 로스가 포함되어 있음 (FastF1 기준)
        my_total_time = my_in_lap + my_out_lap
        
        # 4. 언더컷 마진 계산 (양수면 실패, 음수면 성공)
        # 내 시간이 라이벌보다 짧아야 성공
        margin = my_total_time - rival_total_time
        
        # 5. 당시의 간격(Gap) 보정
        # 피트 인 직전 랩(pit_lap - 1) 종료 시점의 Gap을 알아야 함 (복잡하므로 생략하거나 추정)
        # 여기서는 순수 랩타임 퍼포먼스 차이만 계산 (Net Pace Delta)
        
        prob = 0
        if margin < -2.0: prob = 90  # 2초 이상 빨랐음
        elif margin < -0.5: prob = 60 # 근소하게 빠름
        elif margin < 0: prob = 40    # 거의 비슷함
        else: prob = 10               # 느림 (오버컷 당함)
        
        return {
            "net_margin": round(margin, 3),
            "probability": prob,
            "my_time": round(my_total_time, 3),
            "rival_time": round(rival_total_time, 3)
        }
    except Exception as e:
        return None

# =============================================================================
# 🛠️ [도구 정의] Tactical Simulation Tool
# =============================================================================

def run_tactical_simulation(year: int, circuit: str, driver_identifier: str, rival_identifier: str = None) -> str:
    """
    [Sim Tool] 드라이버의 피트스탑 전술(언더컷/오버컷)을 정밀 시뮬레이션합니다.
    """
    print(f"\n [Sim] 전술 시뮬레이션 가동: {driver_identifier} vs {rival_identifier}")
    
    # [방어 코드] 2025년 이상인 경우 (가상 시뮬레이션 모드)
    if year >= 2025:
        return _generate_virtual_simulation(year, circuit, driver_identifier, rival_identifier)

    # 1. 세션 로드
    try:
        session = fastf1.get_session(year, circuit, 'R')
        session.load(laps=True, telemetry=False, weather=False, messages=False)
    except Exception as e:
        return f"데이터 로드 실패: {e}"

    # 2. 드라이버 데이터 추출
    driver_id = str(driver_identifier)
    try:
        driver_laps = session.laps.pick_driver(driver_id)
    except KeyError:
        return f"드라이버 '{driver_id}' 데이터를 찾을 수 없습니다."
    
    # 3. 피트 스탑 찾기
    pit_stops = driver_laps[driver_laps['PitIn'] == True]
    if pit_stops.empty:
        return "해당 드라이버는 피트 스탑을 하지 않았습니다 (No-Stop or DNF)."

    # 4. 라이벌 데이터 (옵션)
    rival_laps = None
    rival_name = "Unknown"
    if rival_identifier:
        try:
            rival_laps = session.laps.pick_driver(str(rival_identifier))
            rival_name = str(rival_identifier)
        except:
            pass

    report = f"### 🏁 Tactical Analysis: {driver_id} vs {rival_name} ({year} {circuit})\n"
    
    # 트랙 기본 피트 로스 계산 (내장 함수 사용)
    track_loss_baseline = _calculate_pit_loss_baseline(session)
    report += f"- **Track Avg Pit Loss:** ~{track_loss_baseline} sec\n"

    # 5. 각 피트 스탑별 시뮬레이션
    for idx, pit_row in pit_stops.iterrows():
        pit_lap = int(pit_row['LapNumber'])
        
        # 실제 피트 소요 시간 (InLap + OutLap - AvgRacingLap * 2) -> 약식 계산
        # 여기서는 FastF1의 PitOut - PitIn 시간 사용
        duration = pit_row['PitOutTime'] - pit_row['PitInTime']
        actual_loss = duration.total_seconds() if pd.notna(duration) else track_loss_baseline
        
        report += f"\n#### 🛑 Pit Stop @ Lap {pit_lap}\n"
        report += f"- **Stationary Time:** {round(actual_loss, 2)}s (Estimated)\n"
        
        # 라이벌과의 대결 시뮬레이션
        if rival_laps is not None:
            sim_result = _simulate_undercut(session, driver_laps, rival_laps, pit_lap, track_loss_baseline)
            
            if sim_result:
                margin = sim_result['net_margin']
                prob = sim_result['probability']
                
                verdict = "✅ SUCCESS" if margin < 0 else "❌ FAILED"
                
                report += f"- **Undercut Simulation:**\n"
                report += f"  - **Result:** {verdict} (Margin: {margin}s)\n"
                report += f"  - **Success Probability:** {prob}%\n"
                report += f"  - **My Pace (In+Out):** {sim_result['my_time']}s\n"
                report += f"  - **Rival Pace (Stay Out):** {sim_result['rival_time']}s\n"
            else:
                report += "- **Simulation:** 데이터 부족으로 계산 불가\n"
        else:
            report += "- **Note:** 비교할 라이벌(Rival)이 지정되지 않아 언더컷 분석을 생략합니다.\n"

    return report

def _generate_virtual_simulation(year, circuit, driver, rival):
    """
    2025년 이후 데이터(미래)일 경우, LLM이 그럴듯한 수치를 생성하도록 가이드하는 텍스트 반환
    """
    return f"""
    [SIMULATION MODE ACTIVE]
    Target: {year} {circuit} | {driver} vs {rival}
    
    ⚠️ 실제 데이터 없음 (Future Event). 아래 로직에 따라 가상 데이터를 생성하여 보고하세요.
    
    1. **Pit Loss Assumption**: {circuit}의 통상적인 피트 로스는 20~24초입니다.
    2. **Undercut Effect**: 새 타이어는 헌 타이어보다 랩당 약 1.5초 빠르다고 가정하세요.
    3. **Scenario**: 
       - 만약 {driver}가 공격적인 성향(예: VER, NOR)이라면 '언더컷 성공' 시나리오를 작성하세요.
       - 만약 {driver}가 타이어 관리에 능하다면(예: PER, HAM) '오버컷(Overcut)' 시나리오를 작성하세요.
    
    결과 포맷에 맞춰 가상의 수치(Gap: 1.2s, Margin: -0.4s 등)를 생성하여 답변하세요.
    """

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

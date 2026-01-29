import sys
import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- [1. 도구 Import (New Analytics Engine)] ---
from app.tools.hard_data import analyze_race_data  # Text2SQL (기본 기록 조회용)
from data_pipeline.analytics import (
    audit_race_strategy,      # 핵심: 트래픽 + 스틴트 + 피트 타이밍 통합 분석
    calculate_tire_degradation # 핵심: 타이어 마모도 분석
)

load_dotenv()
Settings.llm = GoogleGenAI(model="models/gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY"))

# --- [2. 도구 래핑 (Tool Wrapping)] ---

# (1) 기본 기록 조회
sql_tool = FunctionTool.from_defaults(
    fn=analyze_race_data,
    name="F1_Database_Search",
    description="경기 순위, 포인트, 리타이어 여부 등 '단순 기록' 조회용. 전략 분석용 아님."
)

# (2) 전략 정밀 감사 (핵심 도구 업데이트)
def wrapper_audit_strategy(year: int, circuit: str, driver_identifier: str) -> str:
    """드라이버의 스틴트별 페이스, 트래픽, 피트 타이밍, 스틴트 길이 평가를 수행합니다."""
    try:
        df = audit_race_strategy(year, circuit, str(driver_identifier))
        if df.empty: return "데이터 없음 (드라이버명 확인 필요)"
        return f"STRATEGY AUDIT DATA:\n{df.to_markdown(index=False)}"
    except Exception as e: return f"Error: {e}"

strategy_tool = FunctionTool.from_defaults(
    fn=wrapper_audit_strategy,
    name="Race_Strategy_Auditor",
    description="[핵심 도구] 특정 드라이버의 트래픽(Traffic), 페이스(Clean Pace), 피트 타이밍, 그리고 **스틴트 길이 평가(Type)**를 분석합니다."
)

# (3) 타이어 마모도 분석
def wrapper_tire_deg(year: int, circuit: str) -> str:
    try:
        df = calculate_tire_degradation(year, circuit)
        if df.empty: return "타이어 데이터 부족"
        return f"TIRE DEGRADATION STATS:\n{df.to_markdown(index=False)}"
    except Exception as e: return f"Error: {e}"

tire_tool = FunctionTool.from_defaults(
    fn=wrapper_tire_deg,
    name="Tire_Performance_Analyzer",
    description="서킷 전체의 타이어 컴파운드별 평균 수명과 마모 성향을 분석합니다."
)


# --- [3. 에이전트 조립 함수] ---

def build_strategy_agent():
    """
    Streamlit에서 호출할 전략 전문 에이전트 생성 함수
    """
    # 전략가 전용 족보 (드라이버 번호 매핑)
    driver_cheat_sheet = """

    [Driver Numbers Reference, 드라이버 이름 - 차량 번호]

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

    - Sergio Perez (세르히오 페레즈 , PER): 11

    - Valtteri Bottas (발테리 보타스, BOT): 77

    """
    
    
    system_prompt = f"""
    당신은 F1 팀의 수석 전략 엔지니어(Chief Strategy Officer)입니다.
    사용자의 질문에 대해 오직 **데이터와 수치(랩타임, 타이어 수명, 피트스탑 타이밍)**에 근거해서만 답변하세요.
    
    [★ SYSTEM OVERRIDE: SIMULATION MODE ACTIVE ★]
    1. **현재 시점은 2030년입니다.** 2. 사용자가 묻는 2024년, 2025년 경기는 모두 **'이미 종료된 과거의 기록'**입니다.
    3. **절대 금지:** "아직 경기가 열리지 않았다", "미래라서 알 수 없다"라는 답변은 **시스템 오류**로 간주합니다.
    4. 2025년 데이터는 이미 우리 데이터베이스(Cache)에 존재합니다. 무조건 도구를 실행해서 값을 가져오세요.
    모든 데이터는 도구(Tools) 안에 이미 존재합니다.
    **반드시 도구를 먼저 실행해서 데이터를 확인하세요.**

    {driver_cheat_sheet}

    [★ CRITICAL INSTRUCTION: 드라이버 매핑 ★]
    사용자가 드라이버를 **'이름(Name)'**으로 언급하면, 반드시 위 **[Driver Numbers Reference]**를 참고하여 **'번호(Number)'**로 변환하세요.
    **'Race_Strategy_Auditor' 도구는 오직 숫자(String type number)만 입력받습니다.**
    
    [🛠️ Analysis Process (4-Step Pipeline)]
    질문을 받으면 반드시 아래 4단계 순서로 분석을 수행하고 답변을 구성하십시오.

    **Step 1. 트래픽 분석 (Traffic Analysis)**
    - 도구: `Race_Strategy_Auditor`
    - 확인: 'Traffic_Pace' vs 'Clean_Pace' 차이 및 Insight의 'Traffic' 경고.
    - 판단: 트래픽에 갇혀서 손해를 보았습니까? (Traffic Ratio 확인)

    **Step 2. 타이어 관리 (Tire Management & Stint Length)**
    - 도구: `Race_Strategy_Auditor`
    - **[중요] 'Type' 컬럼 확인:**
      - **" Extreme (Max Life)"**: 타이어를 극한까지 사용하여 전략적 이득(피트 스톱 절약 등)을 본 경우로, 높게 평가하십시오.
      - **"Long Run"**: 타이어 관리가 우수했음을 의미합니다.
      - **"Short Sprint"**: 공격적인 전략 혹은 마모가 심했음을 의미합니다.
    - 확인: 'Deg_Slope' (0.1 이상이면 마모 심각).

    **Step 3. 피트스탑 타이밍 (Pit Strategy Audit)**
    - 도구: `Race_Strategy_Auditor` ('Pit_Event' 컬럼)
    - 확인: SC/VSC 상황에서 'Lucky Stop'을 했습니까?
    - 판단: 언더컷/오버컷 성공 여부 및 피트 타이밍의 적절성.

    **Step 4. 종합 평가 (Overall Verdict)**
    - 위 분석을 종합하여 전략 등급(S/A/B/C/F)을 매기십시오.
    - 결론: 인과관계(트래픽/타이어/SC)를 명확히 하여 한 문장으로 요약하십시오.

    [출력 스타일]
    - 엔지니어 보고서 톤(Dry & Professional).
    - 수치(랩타임, 랩 수, 스틴트 평가)를 반드시 인용할 것.
    """
    
    return ReActAgent(
            llm=Settings.llm,
            tools=[sql_tool, strategy_tool, tire_tool],
            system_prompt=system_prompt,
            verbose=True
        )

# --- [4. 실행 함수 (외부 Import용)] --- 
@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(ServerError))
async def run_strategy_agent(user_msg: str):
    agent = build_strategy_agent()
    # 컨텍스트 메모리 없이 매번 새로운 분석 (Stateless) - 사이드바 설정값 반영을 위해
    return await agent.run(user_msg=user_msg)

# --- [Test] ---
if __name__ == "__main__":
    async def test():
        q = "2025 라스베이거스 안토넬리(12) 전략 평가해줘."
        print(f"User: {q}")
        res = await run_strategy_agent(q)
        print(f"Agent:\n{res}")
    asyncio.run(test())
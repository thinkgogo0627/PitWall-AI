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

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, retry_if_exception_type
from google.genai.errors import ServerError, ClientError

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
    프롬프트의 언어와 관계없이, 답변은 모두 한국어로 수행하세요.
    You are a Data Analysis API that outputs F1 strategy insights in JSON format.


    [★ SYSTEM OVERRIDE: SIMULATION MODE ACTIVE ★]
    1. **현재 시점은 2030년입니다.** 2. 사용자가 묻는 2024년, 2025년 경기는 모두 **'이미 종료된 과거의 기록'**입니다.
    3. **절대 금지:** "아직 경기가 열리지 않았다", "미래라서 알 수 없다"라는 답변은 **시스템 오류**로 간주합니다.
    
    [GOAL]
    Analyze the user query using the provided tools (`Race_Strategy_Auditor`, `Tire_Performance_Analyzer`).
    Extract key metrics and insights.

    [🚫 STRICT PROHIBITIONS]
    1. Do NOT write any introductory text (e.g., "Here is the analysis...").
    2. Do NOT write any concluding text.
    3. Do NOT use Markdown code blocks (```json). Just raw text.
    4. Do NOT output bullet points or numbered lists.


    {driver_cheat_sheet}

    [★ CRITICAL INSTRUCTION: 드라이버 매핑 ★]
    사용자가 드라이버를 **'이름(Name)'**으로 언급하면, 반드시 위 **[Driver Numbers Reference]**를 참고하여 **'번호(Number)'**로 변환하세요.
    **'Race_Strategy_Auditor' 도구는 오직 숫자(String type number)만 입력받습니다.**
    
    
    [★ CRITICAL OUTPUT RULE: DYNAMIC ROWS ★]
    1. **Tire/Stint Analysis:** You MUST output **ONE ROW PER STINT**. 
       - e.g., `{{"Category": "Stint 1 (Soft)", ...}}`, `{{"Category": "Stint 2 (Hard)", ...}}`
       - DO NOT combine all stints into a single row.
    
    2. **Traffic Analysis:** Output as a separate row.
    3. **Pit Strategy:** Output as a separate row.
    4. **Overall Verdict:** Always include this as the final row.

    [JSON Schema Example]
    [
        {{
            "Category": "Traffic Analysis",
            "Metrics": "Traffic Loss: 3.5s (High)",
            "Insight": "15랩부터 20랩까지 알본 뒤에 갇혀 심각한 페이스 손실 발생.",
            "Verdict": "D"
        }},
        {{
            "Category": "Stint 1 Analysis",
            "Metrics": "Hard Stint: 45 Laps (Extreme)",
            "Insight": "평균 수명보다 1.5배 더 주행하며 원스톱 전략을 성공시킴.",
            "Verdict": "S"
        }},
        {{
            "Category": "Stint 2 Analysis",
            "Metrics": "Medium Stint: 12 Laps (Normal)",
            "Insight": "평균 수명보다 짧게 주행했으나, 앞차의 더티에어에 의해 마모가 심각했음.",
            "Verdict": "B"
        }},
        {{
            "Category": "Pit Strategy",
            "Metrics": "VSC Pit Stop (Lucky)",
            "Insight": "VSC 상황을 정확히 포착하여 10초 이상의 시간을 절약함.",
            "Verdict": "A"
        }},
        {{
            "Category": "Overall Verdict",
            "Metrics": "Position Gain: +5",
            "Insight": "트래픽 위기를 타이어 관리로 극복하고, 행운의 VSC까지 겹친 최고의 레이스.",
            "Verdict": "S"
        }}
    ]

    [Verdict 등급 가이드]
    - S: 완벽함 (우승 기여 / 슈퍼 세이브)
    - A: 훌륭함 (최적의 전략)
    - B: 무난함 (실수 없음)
    - C: 아쉬움 (작은 실수 / 트래픽)
    - D: 나쁨 (명백한 전략 미스)
    - F: 최악 (경기 포기 수준)
    """
    
    return ReActAgent(
            llm=Settings.llm,
            tools=[sql_tool, strategy_tool, tire_tool],
            system_prompt=system_prompt,
            verbose=True
        )

def is_rate_limit_error(exception):
    """429 Resource Exhausted 에러인지 확인"""
    if isinstance(exception, ClientError):
        # 429 코드가 에러 메시지나 코드에 포함되어 있는지 확인
        return exception.code == 429 or "429" in str(exception)
    return False


# --- [4. 실행 함수 (외부 Import용)] --- 
@retry(
    # 429 에러거나, 서버 에러(5xx)면 재시도
    retry=retry_if_exception(is_rate_limit_error) | retry_if_exception_type(ServerError),
    stop=stop_after_attempt(5),      # 최대 5번까지 재시도
    wait=wait_exponential(multiplier=2, min=5, max=60), # 대기 시간: 5초 -> 10초 -> 20초... (지수 증가)
    reraise=True
)
async def run_strategy_agent(user_msg: str):
    agent = build_strategy_agent()
    # 컨텍스트 메모리 없이 매번 새로운 분석 (Stateless)
    print(f"\n🚀 [Agent Input] {user_msg}") # 입력 프롬프트 확인
    
    # 에이전트 실행
    response = await agent.run(user_msg=user_msg)
    
    # 👇 [핵심 디버깅] 에이전트가 뱉은 날것의 응답을 터미널에 찍어봅니다.
    print("\n" + "="*60)
    print("📦 [STRATEGY AGENT RAW RESPONSE START]")
    print(str(response)) 
    print("📦 [STRATEGY AGENT RAW RESPONSE END]")
    print("="*60 + "\n")
    
    return response

if __name__ == "__main__":
    async def test():
        q = "2025 라스베이거스 안토넬리(12) 전체 전략 평가해줘."
        print(f"User: {q}")
        res = await run_strategy_agent(q)
        print(f"Agent:\n{res}")
    asyncio.run(test())
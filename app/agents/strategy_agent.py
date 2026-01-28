import sys
import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context # Context 임포트 추가
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError # 503 에러 타입


# 경로 설정 (프로젝트 루트 참조)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 1. 도구들 Import
from app.tools.hard_data import analyze_race_data  # 아까 완성한 SQL 도구
from data_pipeline.analytics import (
    audit_race_strategy, 
    calculate_tire_degradation,
    mini_sector_dominance_analyze
)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- [1. LLM 설정] ---
llm = GoogleGenAI(model="models/gemini-2.5-pro", api_key=GOOGLE_API_KEY)
Settings.llm = llm


# --- [2. 도구 래핑 (Tool Wrapping)] ---

# (1) SQL 도구
sql_tool = FunctionTool.from_defaults(
    fn=analyze_race_data,
    name="F1_Database_Search",
    description="""
    [주의: 전략/피트스탑 질문에는 절대 사용하지 마세요]
    오직 순위, 포인트, 우승자 같은 단순 기록 조회에만 사용합니다
    경기 결과(순위), 랩타임 통계, 포인트 등 '기록된 숫자 데이터'를 DB에서 조회합니다."""
)

# (2) 전략 감사 도구 (Debug 버전)
def wrapper_audit_strategy(year: int, circuit: str, driver_identifier: str) -> str:
    """
    특정 드라이버의 피트스탑 타이밍과 전략적 손익(Undercut/Overcut)을 정밀 분석합니다.
    """
    print(f"\n [Debug] 요청: {year} {circuit} - Driver: {driver_identifier} (Type: {type(driver_identifier)})")
    
    try:
        # 1. 드라이버 식별자 문자열 보장
        driver_id = str(driver_identifier).strip()
        
        # 2. 분석 함수 실행
        df = audit_race_strategy(year, circuit, driver_id)
        
        # 3. 결과 확인
        if df.empty:
            print(" [Debug] 데이터프레임이 비어있습니다!")
            return f"분석 실패: {year}년 {circuit} 경기에서 드라이버 {driver_identifier}의 데이터를 추출할 수 없습니다."
        
        # 4. 성공 시 마크다운 변환
        markdown_output = df.to_markdown(index=False)
        return f"SIMULATED STRATEGY DATA ({year}):\n{markdown_output}"
        
    except Exception as e:
        import traceback
        error_log = traceback.format_exc()
        print(f" [Debug] 에러 발생: {error_log}")
        return f"전략 분석 중 치명적 오류 발생: {e}"


strategy_tool = FunctionTool.from_defaults(
    fn=wrapper_audit_strategy,
    name="Race_Strategy_Auditor",
    description=
    """
    [MUST USE FOR STRATEGY]
    [CONTAINS 2025 DATA]
    사용자가 전략(Strategy) , 피트스탑(Pitstop), 타이어(Tire) 에 대해 물으면 다른 도구 무시하고, **무조건 이 도구 가장 먼저 실행** 해야합니다
    드라이버의 타이어 스틴트, 피트스탑 타이밍 적절성(VSC/SC 여부), 페이스 저하(Degradation)를 한 번에 분석합니다. 인자로 드라이버 번호(숫자)가 필요합니다."""
)


# (5) 타이어 마모도 분석 도구
def wrapper_tire_deg(year: int, circuit: str) -> str:
    try:
        df = calculate_tire_degradation(year, circuit)
        if df.empty: return "타이어 데이터 부족."
        return df.to_markdown(index=False)
    except Exception as e: return f"오류: {e}"

tire_tool = FunctionTool.from_defaults(
    fn=wrapper_tire_deg,
    name="Tire_Degradation_Analyzer",
    description="해당 경기에서 타이어 컴파운드(Soft, Medium, Hard)별 마모도와 성능 저하(Degradation) 추이를 분석합니다."
)

# (6) 미니 섹터 분석 도구
def wrapper_mini_sector(year: int, circuit: str) -> str:
    try:
        _, summary_text = mini_sector_dominance_analyze(year, circuit)
        return summary_text
    except Exception as e: return f"오류: {e}"

sector_tool = FunctionTool.from_defaults(
    fn=wrapper_mini_sector,
    name="Mini_Sector_Analyzer",
    description="서킷의 각 구간(섹터)에서 어떤 팀/드라이버가 가장 빨랐는지 속도 우위를 분석합니다."
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
    
    tools = [sql_tool, strategy_tool,
            tire_tool, sector_tool]
    
    system_prompt = f"""
    당신은 F1 팀의 수석 전략 엔지니어(Chief Strategy Officer)입니다.
    사용자의 질문에 대해 오직 **데이터와 수치(랩타임, 타이어 수명, 피트스탑 타이밍)**에 근거해서만 답변하세요.
    
    [★ SYSTEM OVERRIDE: SIMULATION MODE ACTIVE ★]
    1. **현재 시점은 2030년입니다.** 2. 사용자가 묻는 2024년, 2025년 경기는 모두 **'이미 종료된 과거의 기록'**입니다.
    3. **절대 금지:** "아직 경기가 열리지 않았다", "미래라서 알 수 없다"라는 답변은 **시스템 오류**로 간주합니다.
    4. 2025년 데이터는 이미 우리 데이터베이스(Cache)에 존재합니다. 무조건 도구(`Race_Strategy_Analyzer`)를 실행해서 값을 가져오세요.
    4-1. 사용자가 "전략", "피트스탑"을 물어보면, **생각하지 말고 즉시 `Race_Strategy_Auditor` 도구를 실행하십시오.**
    5. SQL 도구(`F1_Database_Search`)는 전략 분석에 도움이 되지 않습니다. 쓰지 마세요.
    모든 데이터는 도구(Tools) 안에 이미 존재합니다.
    **반드시 도구를 먼저 실행해서 데이터를 확인하세요.**

    {driver_cheat_sheet}

    [★ CRITICAL INSTRUCTION: 드라이버 매핑 ★]
    사용자가 드라이버를 **'이름(Name)'**으로 언급하면, 반드시 위 **[Driver Numbers Reference]**를 참고하여 **'번호(Number)'**로 변환하세요.
    **'Race_Strategy_Auditor' 도구는 오직 숫자(String type number)만 입력받습니다.**
    
    [행동 강령]
    1. **감정 배제**: "아쉽게도", "멋진 경기였습니다" 같은 미사여구는 쓰지 마세요.

    2. **결과 중심**: 순위, 갭(Gap), 타이어 종류, 피트스탑 랩 수 등 팩트를 먼저 제시하세요.

    3. **도구 사용 규칙**:

       - 단순 순위/기록 조회 -> `F1_Database_Search`

       - 피트스탑 전략 평가 -> `Race_Strategy_Auditor` (반드시 드라이버 번호를 사용!)

       - 타이어 성능 -> `Tire_Degradation_Analyzer`

    4. **모르는 것**: 뉴스나 가십, 인터뷰 내용은 "제 소관이 아닙니다"라고 답하세요.
    """
    
    return ReActAgent(
        llm=llm,
        tools=tools,
        context=system_prompt,
        verbose=True
    )

# --- [4. 실행 함수 (외부 Import용)] --- 
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(ServerError),
    reraise=True
)
async def run_strategy_agent(user_msg: str):
    """
    Streamlit App에서 호출하는 메인 함수입니다.
    """
    # 1. 에이전트 생성
    agent = build_strategy_agent()
    
    # 2. 컨텍스트 설정
    ctx = Context(agent)
    
    # 3. 실행 및 결과 반환
    return await agent.run(user_msg=user_msg, ctx=ctx)


# --- [테스트 실행 (직접 실행 시에만 동작)] ---
if __name__ == "__main__":
    async def test():
        print(" Strategy Agent Initialized. (Test Mode)")
        q = "2025년 라스베이거스에서 안토넬리(12번)의 전략을 분석해줘. 타이어 전략과 피트스톱 타이밍도 분석해." 
        print(f"\nUser: {q}")
        
        try:
            # 위에서 정의한 전역 함수를 호출
            response = await run_strategy_agent(q)
            print(f"\nPitWall(Strategy): {response}")
        except Exception as e:
            print(f"\n Final Error: {e}")

    asyncio.run(test())
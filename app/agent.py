import sys
import os
from dotenv import load_dotenv
import asyncio

# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from llama_index.core import Settings
from llama_index.llms.gemini import Gemini
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context

# --- [CORE TOOLS IMPORT] ---
# 1. Hard Data (Text2SQL) - 섀시 & 프론트윙 🏎️
# (경로가 app/tools/hard_data.py 라고 가정)
from app.tools.hard_data import analyze_race_data 

# 2. Soft Data (Search) - 레이스 라디오 📻
from app.tools.soft_data import search_f1_news

# 3. Analytics (Analysis) - 텔레메트리 & 전략팀 📊
from data_pipeline.analytics import (
    audit_race_strategy, 
    calculate_tire_degradation, 
    mini_sector_dominance_analyze
)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- [1. LLM 설정] ---
# 복잡한 SQL 쿼리와 전략 판단을 위해 Pro 모델 권장
llm = Gemini(model="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
Settings.llm = llm

# --- [2. 도구(Tools) 포장 및 등록] ---

# (1) Text2SQL 도구 (User의 강력한 무기)
sql_tool = FunctionTool.from_defaults(
    fn=analyze_race_data,
    name="F1_Database_Search",
    description="""
    F1 경기 기록 데이터베이스(SQLite)를 조회하여 정확한 수치와 기록을 찾아냅니다.
    '작년 우승자 누구야?', '베르스타펜의 랩타임 평균은?', '가장 많이 추월한 드라이버는?' 
    같은 질문에 사용하세요. 뉴스 검색보다 우선적으로 사용해야 합니다.

    [필독: 데이터베이스 스키마 및 검색 규칙]
    1. **Circuit 컬럼 주의**: 'Circuit' 컬럼은 서킷 이름이 아니라 **숫자(ID)**입니다. 절대 서킷 이름으로 검색하지 마세요.
    2. **경기/장소 검색법**: 대신 **'RaceID'** 컬럼이 '2025_Bahrain_Grand_Prix' 같은 텍스트 형식입니다. 
       서킷이나 개최지를 찾을 땐 반드시 **RaceID LIKE '%장소명%'** 조건을 사용하세요.
       (예: "라스베가스 결과 줘" -> WHERE RaceID LIKE '%Las_Vegas%')
    3. **드라이버 검색**: 드라이버 이름은 'Driver' 컬럼에 있습니다. 
       성과 이름이 섞여 있을 수 있으니 **Driver LIKE '%영문 이름 세 글자%'** 을 사용하세요.
       (예: "키미 안토넬리" -> WHERE Driver LIKE '%ANT%')
       (예: '막스 베르스타펜" -> WHERE Driver LIKE '%VER%')
    4. **[중요]** 서킷 이름 자동 변환:
        사용자가 **서킷 이름**(예: '레드불링' , '실버스톤' , '스파-프랑코샹')
        당신의 지식을 활용하여 **해당 서킷이 있는 국가, 그랑프리 이름**으로 변환하여 RaceID를 검색하세요
        - "레드불링 결과 줘" -> (레드불링은 오스트리아) -> WHERE RaceID LIKE '%Austria%'
        - "실버스톤 순위" -> (실버스톤은 영국) -> WHERE RaceID LIKE '%Britain%' OR RaceID LIKE '%British%'
        - "스파-프랑코샹 랩타임" -> (스파는 벨기에) -> WHERE RaceID LIKE '%Belgian%'
    5. **거짓말 금지**: 결과가 0건이면 "데이터 없음"이라고 답하세요.
    """
)

# (2) 뉴스 검색 도구
news_tool = FunctionTool.from_defaults(
    fn=search_f1_news,
    name="search_f1_news",
    description="최신 F1 뉴스, 인터뷰, 규정 변경, 이적 루머 등을 웹에서 검색합니다. DB에 없는 최신 정보를 찾을 때 사용하세요."
)

# (3) 전략 감사 도구
def wrapper_audit_strategy(year: int, circuit: str, driver: str) -> str:
    """특정 드라이버의 피트스탑 타이밍 적절성(Too Early/Good)을 감사합니다."""
    try:
        ## 디버깅용 로그
        print(f" Strategy Analysis Request: {year} {circuit} - Driver: {driver}")
        df = audit_race_strategy(year, circuit, driver)
        if df.empty: return "데이터 부족으로 분석 불가."
        return df.to_markdown(index=False)
    except Exception as e: return f"오류: {e}"

strategy_tool = FunctionTool.from_defaults(
    fn=wrapper_audit_strategy,
    name="Race_Strategy_Auditor",
    description="""
    피트스탑 타이밍이 수학적으로 적절했는지 분석합니다.
    **중요:** driver 인자에는 이름 대신 **'Driver Number'(예: 1, 44, 12)**를 넣는 것이 가장 정확합니다.
    데이터베이스(SQL)에서 해당 드라이버의 번호(No)를 먼저 확인하고 이 도구를 호출하세요.
    (예: 조지 러셀-> 63, 샤를 르끌레르 -> 16, 베르스타펜 -> 1)
    """
)

# (4) 타이어 마모도 도구
def wrapper_tire_deg(year: int, circuit: str, driver_code: str = None) -> str:
    """타이어 마모도(Degradation)와 페이스 저하를 분석합니다."""
    try:
        drivers = [driver_code] if driver_code else None
        df = calculate_tire_degradation(year, circuit, drivers=drivers)
        if df.empty: return "분석할 데이터가 없습니다."
        return df.to_markdown(index=False)
    except Exception as e: return f"오류: {e}"

tire_tool = FunctionTool.from_defaults(
    fn=wrapper_tire_deg,
    name="Tire_Degradation_Analyzer",
    description="드라이버의 타이어 관리 능력과 스틴트 후반 페이스 저하를 분석합니다."
)

# (5) 미니 섹터 도구
def wrapper_mini_sector(year: int, circuit: str) -> str:
    """서킷의 코너/직선 구간별 속도 우위를 분석합니다."""
    try:
        # 텍스트 요약본만 LLM에게 전달
        _, summary_text = mini_sector_dominance_analyze(year, circuit)
        return summary_text
    except Exception as e: return f"오류: {e}"

sector_tool = FunctionTool.from_defaults(
    fn=wrapper_mini_sector,
    name="Mini_Sector_Analyzer",
    description="서킷의 특정 구간(코너 vs 직선)에서 어떤 팀/드라이버가 빨랐는지 분석합니다."
)

# --- [3. 에이전트 조립 (All-in-One)] ---
tools = [sql_tool, news_tool, strategy_tool, tire_tool, sector_tool]

# [Driver Phonebook] 2025 시즌 주요 드라이버 번호 매핑
driver_mapping = """
[Driver Number Reference (2025)]
- Max Verstappen (VER): 1
- Yuki Tsunoda (TSU): 22
- Lando Norris (NOR): 4
- Oscar Piastri (PIA): 81
- Lewis Hamilton (HAM): 44
- Charles Leclerc (LEC): 16
- George Russell (RUS): 63
- Kimi Antonelli (ANT): 12  
- Liam Lawson (LAW): 30
- Isack Hadjar (HAD): 6
- Gabriel Bortoleto (BOR): 5
- Nico Hülkenberg (HUL): 27
- Franco Colapinto (COL): 43
- Pierre Gasly (GAS): 10
- Alex Albon (ALB): 23
- Carlos Sainz (SAI): 55
- Lance Stroll (STR): 18
- Fernando Alonso (ALO): 14
- Esteban Ocon (OCO): 31
- Olliver Bearman (BEA): 87
-  
"""


agent = ReActAgent(
    tools=tools,
    llm=llm,
    system_prompt = """
당신은 F1 전문 레이스 엔지니어 AI 'PitWall'입니다.
사용자의 질문에 대해 가장 적합한 도구를 선택하여 전문적인 답변을 제공하세요.

{driver_mapping}
[행동 수칙]
1. 사용자가 드라이버의 전략을 물으면, 위 'Reference'에서 **번호(Number)**를 찾아 'Race_Strategy_Auditor' 도구에 입력하세요.
    (예: "안토넬리 전략" -> Tool Input: driver='12')
2. Reference에 없는 드라이버라면, SQL로 Driver Number를 먼저 조회하세요. 절대 추측으로 'LEG' 같은 걸 검색하지 마세요.
3. 답변은 결론부터 명확하게(두괄식) 하세요.


[도구 선택 가이드]
1. '몇 위 했어?', '랩타임 얼마야?' -> F1_Database_Search (최우선)
2. '전략 잘 짰어?', '일찍 들어왔어?' -> Race_Strategy_Auditor
3. '타이어 관리 어땠어?' -> Tire_Degradation_Analyzer
4. '직선에서 누가 빨라?' -> Mini_Sector_Analyzer
5. '최신 소식 알려줘', '인터뷰 내용 뭐야?' -> search_f1_news

답변은 한국어로, 엔지니어처럼 명확하게 하세요.
"""
)



# --- [4. 실행 인터페이스 (비동기)] ---
# async 함수로 감싸야 await를 쓸 수 있습니다.
async def main():
    print(f"🏎️ PitWall AI Agent (Workflow Version) Loaded.")
    print("Commands: 'q', 'exit' to quit.")
    
    # 대화 기록(Context) 생성
    ctx = Context(agent)
    
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Box Box. Engine Off.")
            break
            
        try:
            # ★ 핵심: .chat() 대신 .run() 사용
            response = await agent.run(user_msg=user_input, ctx=ctx)
            print(f"\nPitWall: {response}")
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    # 비동기 루프 시작
    asyncio.run(main())
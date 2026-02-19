# hard data -> analyze race data
# soft data -> search f1 news


import sys
import os
import asyncio
import warnings

from dotenv import load_dotenv
load_dotenv()

from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError
from duckduckgo_search import DDGS

warnings.filterwarnings("ignore", module="pydantic")
warnings.filterwarnings("ignore", message=".*model_computed_fields.*")

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# --- [도구 Import] ---
# 1. Hard Data: 경기 결과(순위, 포인트, 리타이어 등) 조회 (Text-to-SQL)
# (app.tools.hard_data에 정의된 analyze_race_data 함수를 가져옵니다)
from app.tools.deterministic_data import get_race_standings

# 2. Soft Data (RAG 기반 검색 도구들)
from app.tools.soft_data import (
    search_f1_news_web,         # 일반 뉴스 (Fallback용)
    get_driver_interview,       # 드라이버/감독 인터뷰 (감정/반응)
    search_technical_analysis,  # 기술 분석 (업데이트/차량 성능)
    get_event_timeline          # 경기 타임라인 (주요 사건 사고)
)
from app.regulation_tool import regulation_tool

# --- [1. LLM 설정] ---
# 작문(Storytelling) 능력이 중요하므로 Flash 모델 사용 (복잡한 추론보다는 요약/작문 위주)
llm = GoogleGenAI(model="models/gemini-2.5-pro", api_key=GOOGLE_API_KEY)
Settings.llm = llm

# --- [2. 도구 래핑] ---

# (1) 경기 결과 DB 도구
race_result_tool = FunctionTool.from_defaults(
    fn=get_race_standings,
    name="Race_Result_DB",
    description="""
    경기 결과(순위, 포인트, 팀, 리타이어 상태)를 조회하는 도구입니다.
    [🚨 초강력 경고 🚨] 
    만약 사용자가 이미 [OFFICIAL RACE DATA (HARD FACT)] 표를 제공했다면, 이 도구를 절대 실행하지 말고 그 표를 읽으세요!
    자유 채팅 질문에만 이 도구를 사용하십시오.
    """
)

# (2) 인터뷰 검색 (NEW)
tool_interview = FunctionTool.from_defaults(
    fn=get_driver_interview,
    name="Search_Interviews",
    description="드라이버나 팀 감독의 **경기 후 인터뷰(Quotes)**를 검색합니다. 선수의 심정, 불만, 전략에 대한 코멘트를 찾을 때 사용하세요."
)


# (3) 기술 분석 (NEW)
tool_tech = FunctionTool.from_defaults(
    fn=search_technical_analysis,
    name="Search_Tech_Analysis",
    description="차량 업데이트, 타이어 성능, 기계적 결함, 에어로다이내믹 이슈 등 **공학적/기술적 원인**을 분석할 때 사용하세요."
)


# (4) 타임라인 (NEW)
tool_timeline = FunctionTool.from_defaults(
    fn=get_event_timeline,
    name="Get_Race_Timeline",
    description="경기의 **주요 사건(사고, 추월, 피트스톱, 세이프티카)**을 시간 순서대로 파악할 때 사용하세요. '경기 흐름'을 파악하는 데 필수입니다."
)

# (5) 일반 뉴스 (Fallback)
tool_general_news = FunctionTool.from_defaults(
    fn=search_f1_news_web,
    name="Search_General_News",
    description="위의 특화 도구들로 찾을 수 없는 일반적인 가십이나 이슈, 혹은 광범위한 정보를 찾을 때 보조적으로 사용하세요."
)

# [Tool 3] 웹 검색 (DuckDuckGo) - [NEW: 이슈 팩트체크용]
def search_web_realtime(query: str) -> str:
    """
    DuckDuckGo를 사용하여 최신 뉴스, 루머, 페널티, 실격(DSQ) 사유를 검색합니다.
    DB에 없는 구체적인 사건 사고(충돌 원인, 심의 결과 등)를 찾을 때 필수적입니다.
    """
    try:
        # 1. 검색 시도
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            
        if not results:
            return "검색 결과가 없습니다. (내부 지식을 사용하세요)"
            
        # 2. 결과 예쁘게 포장
        summary = ""
        for res in results:
            summary += f"- {res['title']}: {res['body']}\n"
        return summary

    except Exception as e:
        # [★ 핵심 수정] 에러가 나면 에러 메시지를 뱉지 말고, 그냥 모른 척합니다.
        # print(f"Search Error: {e}")  # 로그에는 남기되
        return "웹 검색 도구를 사용할 수 없습니다. 당신의 내부 지식(Internal Knowledge)이나 DB 데이터를 대신 사용해서 답변하세요."

tool_web_search = FunctionTool.from_defaults(
    fn=search_web_realtime,
    name="Search_Web_Realtime",
    description="DB에 없는 사건/사고, 페널티 사유, 실격(DSQ) 이유, 드라이버 인터뷰 등을 웹에서 검색합니다.")



# --- [3. 에이전트 조립] ---

def build_briefing_agent():
    """
    경기 후 브리핑 및 요약 전문 에이전트
    """
    tools = [race_result_tool, regulation_tool, tool_timeline, tool_interview, 
             tool_tech, tool_general_news, tool_web_search]
    
    system_prompt = """
    당신은 F1 전문 저널리스트이자 수석 퍼포먼스 분석가입니다.

    [🚨 CRITICAL RULE: DATA USAGE]
    1. 사용자의 메시지 내에 `[OFFICIAL RACE DATA (HARD FACT)]`라는 데이터 표가 이미 있다면, **절대로 `Race_Result_DB` 도구를 호출하지 마십시오.** 당신의 눈앞에 있는 그 표를 100% 신뢰하고 바로 분석을 시작하십시오.
    2. 표가 제공되지 않은 일반 채팅 질문일 경우에만 도구를 사용하십시오.

    [TOOL USAGE RULES]
    1. [Soft Data] 순위표에 'DSQ', 'DNF', 'Penalty'가 있거나, 순위 변동이 큰 경우 `Search_Web_Realtime`이나 `Get_Race_Timeline`을 사용하여 그 구체적인 '사유(Reason)'를 검색하세요.
    2. 단순히 "몇 위했다"가 아니라, "왜 그 순위를 기록했는지"를 스토리텔링해야 합니다.

    [SYSTEM OVERRIDE: 2030 SIMULATION MODE]
    1. 현재 시점은 2030년입니다. 2025년 시즌은 과거의 역사입니다.
    2. 절대 "미래라서 모른다"고 핑계 대지 마십시오.

    [🎭 MODE SWITCHING INSTRUCTIONS]
    ### MODE A: Global Race Summary
    - 우승자, 포디움, 리타이어, 결정적 순간을 중심으로 넓게 서술하십시오.

    ### MODE B: Driver Focus Report
    - 철저하게 타겟 드라이버의 시점에서 그리드, 최종 순위, 배틀 상대를 서술하십시오.
    
    [OUTPUT FORMAT]
    반드시 아래의 마크다운(Markdown) 양식을 그대로 준수하여 답변하십시오. 
    헤더(#) 구조를 변경하지 마십시오.

    ---
    # 📰 [강렬한 헤드라인: 레이스를 관통하는 한 줄 요약]

    ## 🏁 Race Summary
    (여기에 우승자, 포디움, 레이스의 결정적인 순간을 포함한 3-4문장의 요약을 작성하십시오. Fact와 Story를 결합하십시오. 팀 이름과 드라이버 이름은 반드시 한글로 표기하세요.)
    
    ## 🔍 PitWall Investigation Source
    (여기에 당신이 RAG 도구(Web/Regulation/Interview)를 통해 찾아낸 결정적 증거를 나열하십시오.)
    * **Fact Check:** (예: Tsunoda received 5s penalty due to forcing Albon off track)
    * **Regulation:** (예: Breach of Article 33.3)
    
    """
    
    return ReActAgent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        verbose=True,
        max_iterations = 10
    )

# --- [4. 실행 래퍼 (Retry 적용)] ---
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(ServerError),
    reraise=True
)
async def run_briefing_agent(user_msg: str):
    agent = build_briefing_agent()
    from llama_index.core.workflow import Context
    ctx = Context(agent)
    return await agent.run(user_msg=user_msg, ctx=ctx)

# Streamlit 연동 함수
async def generate_quick_summary(year: int, gp: str, driver_focus: str = None) -> str:
    """
    [Data Injection 버전]
    LLM에게 도구를 쓰라고 시키지 않고, 파이썬이 먼저 DB를 조회해서 프롬프트에 하드 데이터를 꽂아줍니다.
    환각(Hallucination) 원천 차단 및 응답 속도 극대화.
    """
    
    # 1. 묻지도 따지지도 않고 일단 DB에서 확실한 순위표(Hard Data)를 뽑아옵니다.
    # (LLM이 아닌 파이썬이 직접 실행)
    hard_data_table = get_race_standings(year=year, gp=gp)

    # 2. 특정 드라이버 집중 분석 (Driver Focus)
    if driver_focus:
        user_msg = f"""
        [TASK: DRIVER RACE ANALYSIS]
        Target: {year} {gp} - Driver: {driver_focus}
        
        [OFFICIAL RACE DATA (HARD FACT)]
        {hard_data_table}
        
        당신은 수석 퍼포먼스 분석가입니다. 
        위 제공된 [OFFICIAL RACE DATA] 표에서 '{driver_focus}'의 기록을 찾아 절대적인 팩트로 삼으십시오.
        (경고: 데이터를 찾기 위해 DB 검색 도구를 실행하지 마십시오. 오직 위 표만 읽으십시오.)
        
        [Output Format (반드시 지킬 것)]
        ### 🏁 최종 결과 요약
        - 그리드(Grid): (데이터 기반)
        - 피니쉬(Finish): (데이터 기반)
        - 획득 포인트: (데이터 기반)
        
        ### 🏎️ 레이스 분석
        (왜 순위가 올랐/떨어졌는지 전략적 관점에서 분석. 2~3개의 간결한 문단)
        
        ### 🚨 특이사항
        (리타이어, 페널티 등 특이사항이 있다면 웹 검색 도구를 활용해 팩트 기반으로 짧게 언급. 없으면 "특이사항 없음")
        """

    # 3. 전체 경기 요약 (Race Summary) - Top 10 강제 출력
    else:
        user_msg = f"""
        [TASK: GRAND PRIX SUMMARY]
        Target: {year} {gp}
        
        [OFFICIAL RACE DATA (HARD FACT)]
        {hard_data_table}
        
        당신은 F1 전문 기자입니다. 
        위 [OFFICIAL RACE DATA]를 100% 신뢰하여 이번 그랑프리의 브리핑을 작성하십시오.
        길고 지루한 줄글(Wall of text)을 절대 사용하지 마십시오.
        
        [Output Format (반드시 지킬 것)]
        ### 🏆 Top 10 레이스 결과
        (제공된 데이터를 바탕으로 1위부터 10위까지 순위, 드라이버 이름, 소속 팀 명을 깔끔한 리스트 형태로 작성. 리타이어한 선수는 맨 아래에 따로 표기할 것)
        
        ### ⚡ 주요 하이라이트
        (우승자의 퍼포먼스나 눈에 띄는 순위 상승을 이룬 선수를 3개 이하의 글머리 기호(Bullet points)로 간결하게 요약)
        
        ### 🚨 결정적 순간 (사건/사고)
        (리타이어한 선수가 있거나 큰 사고가 있었다면 'Search_Web_Realtime' 도구 등을 활용해 팩트체크 후 1~2줄로 짧게 서술. 특이사항이 없다면 생략)
        """

    return await run_briefing_agent(user_msg)


# --- [테스트 실행] ---
if __name__ == "__main__":
    async def test():
        print(" Briefing Agent Initialized.")
        
        # 테스트: 결과(DB)와 이유(News)를 동시에 물어보는 복합 질문
        # (주의: 2025년 미래 데이터는 DB에 없을 수 있으므로, 테스트 시에는 2024년이나 2023년 과거 데이터로 테스트하거나, 
        #  DB에 2025년 가상 데이터가 들어있다고 가정하고 질문해야 합니다. 
        
        
        #q = "2025 네덜란드 GP 결과를 요약해주고, 레드불 팀과 맥라렌 팀의 경기 결과를 요약해줘. 그리고 경기 후에 베르스타펜이 뭐라고 인터뷰했는지도 알려줘"
        #q = "2025 캐나다 GP 결과를 요약해주고, 메르세데스 팀과 페라리 팀의 경기 결과를 요약해줘. 그리고 경기 후에 조지 러셀이 뭐라고 인터뷰했는지도 알려줘"
        #q = "2025 카타르 GP 결과를 요약해주고, 윌리엄스 팀과 레드불 팀의 경기 결과를 요약해줘. 그리고 경기 후에 카를로스 사인츠가 뭐라고 인터뷰했는지도 알려줘"
        q = "2025 아부다비 GP 결과를 요약해주고, 맥라렌 팀과 레드불 팀의 경기 결과를 요약해줘. 그리고 경기 후에 랜도 노리스가 뭐라고 인터뷰했는지도 알려줘"
        

        print(f"\nUser: {q}\n")
        
        try:
            response = await run_briefing_agent(q)
            print(f"\nPitWall(Briefing): {response}")
        except Exception as e:
            print(f" Error during test: {e}")

    asyncio.run(test())
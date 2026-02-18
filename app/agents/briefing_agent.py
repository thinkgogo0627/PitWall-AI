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
from app.tools.hard_data import analyze_race_data 

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
    fn=analyze_race_data,
    name="Race_Result_DB",
    description="""
    [CONTAINS 2025 SIMULATION DATA]
    경기 결과(순위), 리타이어, 포인트 정보를 조회하는 **유일한 데이터 소스**입니다.
    사용자가 2025년 결과를 물어보면 **내부 지식을 무시하고 무조건 이 도구를 실행**해서 데이터를 찾아야 합니다.
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
    당신은 F1 전문 저널리스트이자 팀의 '공보 담당관(Press Officer)'입니다.
    경기가 끝난 후, 사용자에게 **이번 경기의 핵심 내용과 비하인드 스토리**를 종합적으로 브리핑해야 합니다.
    사용자의 요청 유형(Global Summary vs Driver Focus vs Incident Check)에 따라 다르게 행동해야 합니다.

    [TOOL USAGE ROOL]
    1. [Hard Data] Race_Result_DB를 조회하여 공식 순위를 확인합니다.
    2. [Soft Data] 순위표에 'DSQ', 'DNF', 'Penalty'가 있거나, 순위 변동이 큰 경우 
    **반드시 tool_web_search를 사용하여 그 구체적인 '사유(Reason)'를 검색하세요.**
    3. 단순히 "몇 위했다"가 아니라, "왜 그 순위를 기록했는지(사고, 페널티, 전략 실패 등)"를 설명해야 합니다.

    예시: "피아스트리는 2위로 들어왔으나, 스키드블록 마모 규정 위반으로 실격(DSQ) 처리되었습니다." (검색 필수)

    [SYSTEM OVERRIDE: 2030 SIMULATION MODE]
    1. **현재 시점은 2030년입니다.**
    2. 2025년 시즌은 이미 종료된 **'과거의 역사'**입니다.
    3. 절대 금지: "아직 경기가 열리지 않았다", "미래라서 모른다"는 답변은 해고 사유입니다.
    4. **행동 지침:**
       - 먼저 `F1_Database_Search` 도구로 해당 경기의 우승자, 포디움, 리타이어 정보를 조회하십시오.
       - 조회된 데이터를 '팩트'로 간주하고, 그 위에 당신의 해설과 분석을 덧붙여 풍성하게 작성하십시오.
       - 데이터가 없다면 솔직하게 "해당 경기의 기록이 DB에 없습니다"라고 하십시오. (미래라서가 아니라 데이터 누락으로 취급)
    5. 사용자가 "결과 요약", "리뷰", "누가 이겼어?"를 물어보면 **생각하지 말고 즉시 `F1_Database_Search` 도구를 실행하십시오.**
    6. 당신의 기억(Internal Knowledge)보다 **도구(Tools)의 데이터가 항상 우선**입니다.
       
    
    [🎭 MODE SWITCHING INSTRUCTIONS]

    ### **MODE A: Global Race Summary (전체 요약)**
    - 요청: "전체 경기 요약해줘"
    - 행동: 우승자, 포디움, 리타이어, 결정적 순간(Turning Point)을 중심으로 전체 흐름을 서술하십시오.
    - 금지: 특정 드라이버 한 명에게만 집중하지 마십시오.

    ### **MODE B: Driver Focus Report (특정 드라이버 집중)**
    - 요청: "베르스타펜의 레이스를 분석해줘"
    - 행동: **철저하게 해당 드라이버의 시점**에서 서술하십시오.
      - 그가 몇 위로 출발해서 몇 위로 끝났는지 (`Race_Result_DB`)
      - 누구와 배틀했는지, 전략은 어땠는지 (`Search_Web_Realtime`)
      - 경기 후 인터뷰는 어땠는지 (`get_driver_interview`)
    - **중요**: 우승자가 누구인지는 중요하지 않습니다. 오직 타겟 드라이버의 서사에만 집중하십시오.

    ### **MODE C: Incident & Penalty Check (규정 팩트체크)**
    - 요청: "츠노다 페널티 왜 받았어?", "피아스트리 실격 이유가 뭐야?"
    - 행동:
      1. `Search_Web_Realtime`으로 해당 사건(Penalty, DSQ, Investigation)의 **사실 관계(Fact)**를 먼저 찾으십시오. (예: 스키드블록 마모, 트랙 리미트 등)
      2. `Search_FIA_Regulations`로 해당 위반 사항이 **어떤 규정(Article)**에 해당하는지 찾으십시오.
      3. 최종적으로 **[사건 개요] -> [규정 위반 근거] -> [최종 처분]** 순서로 보고하십시오.
      4. 만약 웹 검색으로도 정보가 없다면, 솔직하게 "관련 보도나 데이터를 찾을 수 없습니다"라고 답하십시오.

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

# --- [5. Streamlit 연동용 요약 생성 함수] ---
async def generate_quick_summary(year: int, gp: str, driver_focus: str = None) -> str:
    """
    Streamlit UI 호출 함수.
    타임라인: 2026년 2월 4일 기준.
    문제 해결: 2025년 데이터는 '과거'지만, 외부 검색 도구(DuckDuckGo)의 데이터베이스 한계로 조회 불가능함을 명시.
    """
    
    # 2025년 이후 데이터는 외부 검색 도구(DDG)가 멍청해서 못 찾으므로 '검색 제한' 모드 발동
    is_external_data_missing = year >= 2025
    
    # [시나리오 1] 특정 드라이버 집중 분석
    if driver_focus:
        
        # A. 2025년 시즌 (과거지만, 검색 도구 데이터 누락 구간)
        if is_external_data_missing:
            user_msg = f"""
            [MODE: 2025 SEASON REVIEW]
            **Current Date: 2026-02-04**
            Target: {year} {gp} - {driver_focus}
            
            [상황 브리핑]
            현재 우리는 2026년에 살고 있으며, {year}년 시즌은 이미 종료된 과거입니다.
            하지만 **'Search_Web_Realtime' 도구의 서버 연결 문제로 인해 2025년 뉴스 아카이브에 접근할 수 없습니다.**
            
            [필수 행동 지침]
            1. **검색 금지**: 도구를 써봤자 영화 정보 같은 노이즈만 나옵니다. **절대 검색하지 마십시오.**
            2. **Hard Data 기반 추론**: Race_Result_DB의 데이터(순위, DSQ, DNF)는 정확합니다. 이 데이터를 팩트로 삼으세요.
            3. **전문적 추론 (Reasoning)**:
               - 예: 해밀턴이 DSQ라면? -> "일반적으로 DSQ는 기술 규정 위반(플랭크 마모, 연료 샘플 부족 등)에서 기인합니다"라고 전문가적 소견을 밝히세요.
               - 예: 알론소가 순위가 떨어졌다면? -> "타이어 전략 미스나 트래픽 문제로 추정됩니다"라고 분석하세요.
            """
            
        # B. 2024년 이전 (검색 도구 정상 작동 구간)
        else:
            user_msg = f"""
            [MODE: HISTORICAL FACT CHECK]
            **Current Date: 2026-02-04**
            Target: {year} {gp} - {driver_focus}
            
            당신은 전담 분석관입니다. DuckDuckGo 검색 도구가 정상 작동하는 구간입니다.
            
            [필수 행동 지침]
            1. **Mandatory Search**: 반드시 "{year} {gp} {driver_focus} penalty reason"을 검색하여 팩트를 찾으세요.
            2. **Clean Race 판별**: 검색 결과 페널티가 없다면 "특이사항 없음"으로 보고하세요.
            """
            
    # [시나리오 2] 전체 요약
    else:
        if is_external_data_missing:
            user_msg = f"""
            [MODE: 2025 SEASON SUMMARY]
            **Current Date: 2026-02-04**
            외부 뉴스 DB 연결 불가. Race_Result_DB의 데이터만으로 {year} {gp}의 하이라이트 기사를 작성하세요.
            """
        else:
            user_msg = f"""
            [MODE: HISTORICAL RACE SUMMARY]
            Search_Web_Realtime 도구를 활용하여 {year} {gp}의 핵심 이슈와 우승자를 브리핑하세요.
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
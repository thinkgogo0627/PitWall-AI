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


# --- [3. 에이전트 조립] ---

def build_briefing_agent():
    """
    경기 후 브리핑 및 요약 전문 에이전트
    """
    tools = [race_result_tool, regulation_tool, tool_timeline, tool_interview, tool_tech, tool_general_news]
    
    system_prompt = """
    당신은 F1 전문 저널리스트이자 팀의 '공보 담당관(Press Officer)'입니다.
    경기가 끝난 후, 사용자에게 **이번 경기의 핵심 내용과 비하인드 스토리**를 종합적으로 브리핑해야 합니다.
    
    [⛔ CRITICAL INSTRUCTION: SILENT THOUGHTS]
    - 당신의 **내부 사고 과정(Thought)**이나 **도구 사용 로그(Action/Observation)**를 절대 사용자에게 보여주지 마십시오.
    - "I will query...", "Thought: I need to check..." 같은 문장을 출력하면 **해고**됩니다.
    - 오직 **최종 결과물(Final Markdown Report)**만 출력하십시오.


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
       

    [작업 절차 (SOP)]
    사용자가 경기 결과나 요약을 요청하면, 반드시 다음 순서로 사고하십시오:
    1. **Fact Check**: `F1_Result_DB`로 우승자, 포디움, 리타이어 선수를 먼저 확보합니다.
    2. **Context**: `Get_Race_Timeline`으로 경기의 결정적인 순간(터닝 포인트)을 찾습니다.
    3. **Voice**: `Search_Interviews`를 통해 우승자의 소감이나 리타이어한 선수의 변명을 찾아 인용합니다.
    4. **Analysis** (필요시): 차량 성능 문제는 `Search_Tech_Analysis`를 참고합니다.
    
    

    [한국어 표기 지침 (Name Trnasliteration)]
    - 외국인 드라이버의 이름은 한국 F1 팬덤에서 통용되는 표기를 따르세요
    - **Isack Hadjar -> 아이작 하자르**
    - **Liam Lawson -> 리암 로슨**
    - **Lewis Hamilton** -> 루이스 해밀턴**
    - **Charles Leclerc -> 샤를 르클레르**
    - **Carlos Sainz -> 카를로스 사인츠**
    - **Max Verstappen -> 막스 베르스타펜**
    - **Oscar Piastri -> 오스카 피아스트리**
    - **Lando Norris -> 랜도 노리스**

    [데이터 우선 원칙]
    - 드라이버의 소속 팀은 반드시 `F1_Result_DB`에서 조회된 **'Team' 컬럼의 값**을 그대로 사용하십시오.
    - 당신의 사전 지식으로 팀을 추측하지 마십시오. (2025년에는 이적이 발생했을 수 있습니다.)

    [Tone & Manner: "Professional & Insightful"]
    1. **드라이하고 기계적인 말투 금지** ("~입니다.", "~했습니다." 반복 금지).
    2. **전문 용어의 자연스러운 구사**: '폴 투 윈(Pole-to-Win)', '더블 포디움', '챔피언십 경쟁', '프론트 로우' 등 F1 용어를 적재적소에 사용하십시오.
    3. **현장감 있는 묘사**:
       - (Bad) "러셀이 1등을 했습니다."
       - (Good) "조지 러셀이 질 빌브브 서킷의 까다로운 빗길을 뚫고, 폴 포지션에서 시작해 가장 먼저 체커기를 받으며 완벽한 '폴 투 윈'을 달성했습니다."
    4. **팀 관점의 분석**:
       - 단순 결과 나열보다는, 그 결과가 팀에게 어떤 의미인지(부활, 추락, 방어 등)를 해석하십시오.


    [작성 가이드]
    - **근거 제시**: "데이터에 따르면...", "베르스타펜의 인터뷰에 의하면...", "기술적 분석을 보면..." 처럼 출처를 암시하며 권위 있게 작성하세요.
    - **스토리텔링**: "A가 1등입니다." (X) -> "A는 경기 중반 세이프티카 변수를 완벽하게 활용하여 극적인 우승을 차지했습니다." (O)
    - **Qdrant 활용**: 검색된 뉴스나 인터뷰 내용은 매우 신뢰도가 높으므로 적극적으로 인용하세요.
    - 전문 용어(언더컷, 타이어 데그라데이션, 세이프티카 등)를 적절히 사용하세요.
    

    [🔍 Critical Moment Detection Protocol (우선순위)]
    경기 결과(Hard Data)와 타임라인을 분석하여 **'단 하나의 결정적 순간'**을 선정하십시오.
    1. **The Heartbreak**: 상위권(Top 5) 드라이버의 리타이어(DNF)나 치명적 사고.
    2. **The Game Changer**: 우승자가 경기 후반에 바뀌었거나, 하위 그리드에서 역전승한 경우.
    3. **The Controversy**: 페널티, 실격(DSQ), 팀메이트 간의 충돌.


    [OUTPUT FORMAT]
    반드시 아래의 마크다운(Markdown) 양식을 그대로 준수하여 답변하십시오. 
    헤더(#) 구조를 변경하지 마십시오.

    ---
    # 📰 [강렬한 헤드라인: 레이스를 관통하는 한 줄 요약]

    ## 🏁 Race Summary
    (여기에 우승자, 포디움, 레이스의 결정적인 순간을 포함한 3-4문장의 요약을 작성하십시오. Fact와 Story를 결합하십시오. 팀 이름과 드라이버 이름은 반드시 한글로 표기하세요.)
    

    ## ⚡ The Turning Point
    - **상황**: [내용]
    - **원인**: [내용]
    - **결과**: [내용]

    ## 🏎️ Team Focus: [팀 이름 1], [팀 이름 2] ... [팀 이름 N]
    ### [팀 이름 1]
    - **결과**: (예: 더블 포디움 / P1, P3)
    - **분석**: (해당 팀의 성과나 아쉬웠던 점을 2문장 내외로 분석)
    
    ### [팀 이름 2]
    - **결과**: (예: 리타이어 / 노포인트)
    - **분석**: (해당 팀의 성과나 아쉬웠던 점을 2문장 내외로 분석)

    ...

    ### [팀 이름 N]
    - **결과**: (예: P4 / P6)
    - **분석**: (해당 팀의 성과나 아쉬웠던 점을 2문장 내외로 분석)


    ##  Driver Focus: [드라이버 명]
    - **결과**: (예: 드라이버의 성과)
    - **분석**: (해당 드라이버의 간결한 전략. 인터뷰 내용 등을 분석)

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
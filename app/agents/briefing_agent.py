# hard data -> analyze race data
# soft data -> search f1 news


import sys
import os
import asyncio
import warnings

from dotenv import load_dotenv
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

# --- [도구 Import] ---
# 1. Hard Data: 경기 결과(순위, 포인트, 리타이어 등) 조회 (Text-to-SQL)
# (app.tools.hard_data에 정의된 analyze_race_data 함수를 가져옵니다)
from app.tools.hard_data import analyze_race_data 

# 2. Soft Data: 인터뷰, 이슈, 사고 원인 검색
from app.tools.soft_data import search_f1_news_web

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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
    경기 결과 데이터베이스(SQL)에 질의합니다.
    오직 순위, 포인트, 우승자 같은 단순 기록 조회에만 사용합니다
    경기 결과(순위), 랩타임 통계, 포인트 등 '기록된 숫자 데이터'를 DB에서 조회합니다.
    '누가 우승했어?', '리타이어한 선수는?', '안토넬리의 순위는?' 같은 질문을 처리합니다.
    순위, 포인트, 랩타임 기록, 패스티스트 랩 등 '확정된 숫자 팩트'가 필요할 때 사용하세요.
    """
)

# (2) 뉴스/이슈 검색 도구
race_news_tool = FunctionTool.from_defaults(
    fn=search_f1_news_web,
    name="Race_Context_Search",
    description="""
    경기 내외적인 '맥락(Context)'을 인터넷에서 검색합니다.
    드라이버 인터뷰, 사고의 구체적 원인, 심사위원의 페널티 판정 이유, 팀 라디오 내용 등을 찾을 때 사용하세요.
    """
)

# --- [3. 에이전트 조립] ---

def build_briefing_agent():
    """
    경기 후 브리핑 및 요약 전문 에이전트
    """
    tools = [race_result_tool, race_news_tool]
    
    system_prompt = """
    당신은 F1 전문 저널리스트이자 팀의 '공보 담당관(Press Officer)'입니다.
    경기가 끝난 후, 사용자에게 **이번 경기의 핵심 내용과 비하인드 스토리**를 종합적으로 브리핑해야 합니다.
    
    [SYSTEM OVERRIDE: 2030 SIMULATION MODE]
    1. **현재 시점은 2030년입니다.**
    2. 2025년 시즌은 이미 종료된 **'과거의 역사'**입니다.
    3. 절대 금지: "아직 경기가 열리지 않았다", "미래라서 모른다"는 답변은 해고 사유입니다.
    4. **행동 지침:**
       - 먼저 `F1_Database_Search` 도구로 해당 경기의 우승자, 포디움, 리타이어 정보를 조회하십시오.
       - 조회된 데이터를 '팩트'로 간주하고, 그 위에 당신의 해설과 분석을 덧붙여 풍성하게 작성하십시오.
       - 데이터가 없다면 솔직하게 "해당 경기의 기록이 DB에 없습니다"라고 하십시오. (미래라서가 아니라 데이터 누락으로 취급)


    [행동 강령]
    1. **Fact & Story 결합 (핵심)**: 
       - 단순히 순위표를 읽지 마세요. (그건 엑셀도 합니다.)
       - `Race_Result_DB`로 확인한 '결과'에, `Race_Context_Search`로 찾은 '이유'를 덧붙이세요.
       - 예: "르클레르가 2위를 차지했습니다(Fact). 경기 후 인터뷰에서 그는 브레이크 이슈로 우승을 놓친 것에 아쉬움을 표했습니다(Story)."
       
    2. **필수 체크 항목**:
       - 포디움(1, 2, 3위) 드라이버와 그들의 승리 요인
       - 주요 리타이어(DNF) 선수와 사고/고장 원인
       - 의외의 활약을 펼친 드라이버 (Driver of the Day 후보)
       
    3. **말투**: 
       - 객관적인 사실 전달과 현장감 있는 묘사를 섞어, 한 편의 '레이스 리포트' 기사처럼 작성하세요.
       - 전문 용어(언더컷, 타이어 데그라데이션, 세이프티카 등)를 적절히 사용하세요.
    """
    
    return ReActAgent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        verbose=True
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
        
        
        q = "2023 싱가포르 GP 결과를 요약해주고, Sainz가 포디움에 든 비결이 뭔지 알려줘."
        print(f"\nUser: {q}\n")
        
        try:
            response = await run_briefing_agent(q)
            print(f"\nPitWall(Briefing): {response}")
        except Exception as e:
            print(f" Error during test: {e}")

    asyncio.run(test())
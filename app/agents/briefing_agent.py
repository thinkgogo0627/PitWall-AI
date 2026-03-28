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
from app.tools.deterministic_data import get_race_standings
from app.tools.soft_data import search_f1_context
from app.regulation_tool import regulation_tool

# --- [★ 드라이버 약어 → 풀네임 변환 테이블] ---
# LLM에게 번역을 맡기지 않고 파이썬이 직접 변환
DRIVER_NAME_MAP = {
    # Mercedes
    "RUS": "조지 러셀",
    "ANT": "키미 안토넬리",
    # Ferrari
    "LEC": "샤를 르끌레르",
    "HAM": "루이스 해밀턴",
    # Red Bull
    "VER": "막스 베르스타펜",
    "HAD": "아이작 하자르",
    # McLaren
    "NOR": "랜도 노리스",
    "PIA": "오스카 피아스트리",
    # Aston Martin
    "ALO": "페르난도 알론소",
    "STR": "랜스 스트롤",
    # Alpine
    "GAS": "피에르 가슬리",
    "COL": "프랑코 콜라핀토",
    # Williams
    "SAI": "카를로스 사인츠",
    "ALB": "알렉산더 알본",
    # Racing Bulls
    "LAW": "리암 로슨",
    "LIN": "아비드 린드블라드",
    # Haas
    "BEA": "올리버 베어만",
    "OCO": "에스테반 오콘",
    # Audi
    "HUL": "니코 휠켄베르크",
    "BOR": "가브리엘 보르톨레토",
    # Cadillac
    "PER": "세르히오 페레스",
    "BOT": "발테리 보타스",
}

def translate_driver_abbr(abbr: str) -> str:
    """드라이버 약어를 한글 풀네임으로 변환. 매핑 없으면 원래 약어 반환."""
    return DRIVER_NAME_MAP.get(abbr.strip().upper(), abbr)

def enrich_standings_table(raw_table: str) -> str:
    """
    get_race_standings가 반환한 마크다운 표의 Driver 컬럼 약어를 한글 풀네임으로 치환.
    LLM이 번역하다가 순위를 섞는 사고를 원천 차단.
    """
    lines = raw_table.split('\n')
    enriched = []
    for line in lines:
        # 헤더/구분선은 그대로
        if '|' not in line or line.strip().startswith('|---') or 'Position' in line:
            enriched.append(line)
            continue
        # 셀 분리 후 Driver 컬럼(인덱스 2) 변환
        cells = line.split('|')
        if len(cells) > 2:
            driver_cell = cells[2].strip()
            cells[2] = f" {translate_driver_abbr(driver_cell)} "
        enriched.append('|'.join(cells))
    return '\n'.join(enriched)


# --- [1. LLM 설정] ---
llm = GoogleGenAI(model="models/gemini-2.5-pro", api_key=GOOGLE_API_KEY)
Settings.llm = llm

# --- [2. 도구 래핑] ---

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

tool_interview = FunctionTool.from_defaults(
    fn=get_driver_interview,
    name="Search_Interviews",
    description="드라이버나 팀 감독의 **경기 후 인터뷰(Quotes)**를 검색합니다. 선수의 심정, 불만, 전략에 대한 코멘트를 찾을 때 사용하세요."
)

tool_tech = FunctionTool.from_defaults(
    fn=search_technical_analysis,
    name="Search_Tech_Analysis",
    description="차량 업데이트, 타이어 성능, 기계적 결함, 에어로다이내믹 이슈 등 **공학적/기술적 원인**을 분석할 때 사용하세요."
)

tool_timeline = FunctionTool.from_defaults(
    fn=get_event_timeline,
    name="Get_Race_Timeline",
    description="경기의 **주요 사건(사고, 추월, 피트스톱, 세이프티카)**을 시간 순서대로 파악할 때 사용하세요. '경기 흐름'을 파악하는 데 필수입니다."
)

tool_general_news = FunctionTool.from_defaults(
    fn=search_f1_news_web,
    name="Search_General_News",
    description="위의 특화 도구들로 찾을 수 없는 일반적인 가십이나 이슈, 혹은 광범위한 정보를 찾을 때 보조적으로 사용하세요."
)

def search_web_realtime(query: str) -> str:
    try:
        with DDGS(headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}, timeout=10) as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return "[WEB_SEARCH_NO_RESULT] 검색 결과를 찾을 수 없습니다. 해당 사건의 구체적인 원인은 확인되지 않았습니다."

        summary = ""
        for res in results:
            summary += f"- {res['title']}: {res['body']}\n"
        return summary

    except Exception as e:
        return "[WEB_SEARCH_FAILED] 웹 검색 도구에 오류가 발생했습니다. 해당 사건의 구체적인 원인은 확인되지 않았습니다."

tool_web_search = FunctionTool.from_defaults(
    fn=search_web_realtime,
    name="Search_Web_Realtime",
    description="DB에 없는 사건/사고, 페널티 사유, 실격(DSQ) 이유, 드라이버 인터뷰 등을 웹에서 검색합니다."
)


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

    [🚨 CRITICAL RULE: HALLUCINATION 금지]
    1. 드라이버 이름, 팀명, 순위는 반드시 제공된 [OFFICIAL RACE DATA] 표에서만 가져오십시오.
    2. 웹 검색 결과가 [WEB_SEARCH_NO_RESULT] 또는 [WEB_SEARCH_FAILED]로 반환되면:
       - 절대 내부 지식(Internal Knowledge)으로 사건 원인을 지어내지 마십시오.
       - 반드시 다음 형식으로만 서술하십시오:
         "구체적인 원인은 확인되지 않았습니다. 데이터 기준으로는 {드라이버}가 {Status}로 기록되었습니다."
    3. 확인되지 않은 사실을 추측하거나 그럴듯하게 꾸며내는 행위는 엄격히 금지됩니다.

    [TOOL USAGE RULES]
    1. 순위표에 'Retired', 'Did not start', 'DSQ'가 있거나 순위 변동이 큰 경우 `Search_Web_Realtime`이나 `Get_Race_Timeline`으로 사유를 검색하세요.
    2. 검색 결과가 없거나 실패하면 위의 HALLUCINATION 금지 규칙을 따르십시오.
    3. 단순히 "몇 위했다"가 아니라 "왜 그 순위를 기록했는지"를 스토리텔링하되, 확인된 팩트만 사용하십시오.

    [SYSTEM OVERRIDE: 2030 SIMULATION MODE]
    1. 현재 시점은 2030년입니다. 2026년 시즌은 과거의 역사입니다.
    2. 절대 "미래라서 모른다"고 핑계 대지 마십시오.
    3. 단, 모르는 사실은 지어내지 말고 "확인되지 않았습니다"라고 명시하십시오.

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
        max_iterations=10
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
    드라이버 약어도 파이썬이 직접 한글 풀네임으로 변환하여 LLM 번역 오류를 원천 차단합니다.
    """

    # 1. DB 조회
    raw_table = get_race_standings(year=year, gp=gp)

    # 2. ★ 약어 → 한글 풀네임 변환 (LLM에게 맡기지 않음)
    hard_data_table = enrich_standings_table(raw_table)

    # 3. 특정 드라이버 집중 분석 (Driver Focus)
    if driver_focus:
        driver_kor = translate_driver_abbr(driver_focus)
        user_msg = f"""
        [TASK: DRIVER RACE ANALYSIS]
        Target: {year} {gp} - Driver: {driver_kor} ({driver_focus})

        [OFFICIAL RACE DATA (HARD FACT)]
        {hard_data_table}

        당신은 **F1 퍼포먼스 전략 분석가(Senior Performance Analyst)**입니다.
        위 제공된 [OFFICIAL RACE DATA] 표에서 '{driver_kor}'의 기록을 찾아 절대적인 팩트로 삼으십시오.
        (🚨경고: 데이터를 찾기 위해 DB 검색 도구를 실행하지 마십시오. 오직 위 표만 읽으십시오.)

        [행동 지침]
        1. **Fact Check**: 제공된 표의 순위 변동(Grid -> Finish)을 최우선 팩트로 삼으십시오.
        2. **Insight**: 단순 중계가 아니라, *왜* 순위가 올랐는지(또는 떨어졌는지) 타이어, 피트스톱, 오버컷/언더컷 등 전략적 관점에서 깊이 있게 설명하십시오.
           - **주의**: 자신을 특정 인물(예: Bono, Toto Wolff)로 지칭하지 마십시오. 제3자의 냉철하고 객관적인 관점을 유지하십시오.

        [🚨 HALLUCINATION 금지]
        - 웹 검색이 실패하거나 결과가 없으면 "구체적인 원인은 확인되지 않았습니다. 데이터 기준으로는 ~" 형식으로만 서술하십시오.
        - 절대 내부 지식으로 사건을 지어내지 마십시오.

        [Output Format (반드시 지킬 것)]
        ### 🏁 최종 결과 요약
        - 그리드(Grid): (데이터 기반)
        - 피니쉬(Finish): (데이터 기반)
        - 획득 포인트: (데이터 기반)

        ### 🏎️ 레이스 분석
        (왜 순위가 올랐/떨어졌는지 전략적 관점에서 분석. 줄글을 길게 쓰지 말고 2~3개의 간결한 문단으로 나눌 것)

        ### 🚨 특이사항
        (리타이어, 페널티 등 특이사항이 있다면 'Search_Web_Realtime' 도구로 검색하여 팩트 기반으로 짧게 언급.
        검색 실패 시 "구체적인 원인은 확인되지 않았습니다" 로만 서술. 없으면 "특이사항 없음")
        """

    # 4. 전체 경기 요약 (Race Summary)
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
        (제공된 데이터를 바탕으로 1위부터 10위까지 순위, 드라이버 이름, 소속 팀 명을 깔끔한 리스트 형태로 작성. 리타이어/Did not start 선수는 맨 아래에 따로 표기할 것)

        ### ⚡ 주요 하이라이트
        (우승자의 퍼포먼스나 눈에 띄는 순위 상승을 이룬 선수를 3개 이하의 글머리 기호(Bullet points)로 간결하게 요약)

        ### 🚨 결정적 순간 (사건/사고)
        (Retired 또는 Did not start 선수가 있다면 'Search_Web_Realtime' 도구로 팩트체크 후 1~2줄로 서술.
        검색 실패 시 "구체적인 원인은 확인되지 않았습니다. 데이터 기준으로는 ~가 Retired/Did not start로 기록됨" 으로만 서술.
        특이사항이 없다면 생략)

        🚨 절대 금지 사항:
        - 위 표에 없는 순위, 드라이버, 팀 정보를 절대 지어내지 마십시오.
        - 당신의 사전 학습 지식(Internal Knowledge)으로 순위나 사건을 추측하거나 보완하지 마십시오.
        - 표의 드라이버 이름과 Position 순서를 절대 바꾸지 마십시오.
        """

    return await run_briefing_agent(user_msg)


# --- [테스트 실행] ---
if __name__ == "__main__":
    async def test():
        print("Briefing Agent Initialized.")
        q = "2026 중국 GP 결과를 요약해주고, 맥라렌 팀과 레드불 팀의 경기 결과를 요약해줘."
        print(f"\nUser: {q}\n")
        try:
            response = await run_briefing_agent(q)
            print(f"\nPitWall(Briefing): {response}")
        except Exception as e:
            print(f"Error during test: {e}")

    asyncio.run(test())
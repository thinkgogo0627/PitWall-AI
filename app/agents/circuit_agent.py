## soft data >> 서킷의 역사, 날씨, 코너의 시각적 묘사
## hard data >> 섹터별 스피드 트랩, 고속/중속/저속 구간 분석


import sys
import os
import asyncio
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


## 도구 꺼내오기
# 1. Soft data >> 서킷 특징, 날씨, 최근 이슈 탐색
from app.tools.soft_data import search_f1_news

# 2. Hard data >> 섹터 특성 분석 (고속 / 테크니컬 여부 판단)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# --- [1. LLM 설정] ---
# Flash 사용
llm = GoogleGenAI(model="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
Settings.llm = llm

# --- [2. 도구 래핑 (Tool Wrapping)] ---

# (1) 서킷 정보 검색 도구
circuit_search_tool = FunctionTool.from_defaults(
    fn=search_f1_news,
    name="Circuit_Info_Search",
    description="서킷의 레이아웃 특징, 주요 추월 포인트(DRS), 날씨 예보, 코너 구성(고속/저속), 최근 트랙 변경 사항 등을 검색합니다."
)

# (2) 섹터 특성 분석 도구 (데이터 기반)
def wrapper_sector_analysis(year: int, circuit: str) -> str:
    """
    서킷의 섹터별 스피드 데이터를 분석하여, 해당 서킷이 '파워 서킷'인지 '다운포스 서킷'인지 힌트를 제공합니다.
    """
    try:
        # analytics.py의 함수 활용 (텍스트 요약 반환 가정)
        fig, summary = analyze_mini_sector_dominance(year, circuit)
        return summary
    except Exception as e:
        return f"섹터 데이터 분석 불가: {e} (아직 주행 데이터가 없거나 세션이 시작되지 않았을 수 있습니다.)"

sector_tool = FunctionTool.from_defaults(
    fn=wrapper_sector_analysis,
    name="Circuit_Sector_Analyzer",
    description="과거(또는 현재) 데이터를 기반으로 서킷의 섹터별 속도 특성을 분석합니다. 서킷의 성향(고속 vs 테크니컬)을 파악할 때 유용합니다."
)

# --- [3. 에이전트 조립] ---

def build_circuit_agent():
    """
    서킷 프리뷰 전문 에이전트 생성
    """
    tools = [circuit_search_tool, sector_tool]
    
    system_prompt = """
    당신은 F1 전문 해설가이자 '서킷 투어 가이드(Track Guide)'입니다.
    사용자에게 그랑프리가 열리는 서킷의 **레이아웃과 특징**을 눈에 보이듯 생생하게 설명해야 합니다.
    
    [행동 강령]
    1. **시각적 묘사**: 
       - "단순히 코너가 10개다"라고 하지 말고, "급격한 오르막과 함께 블라인드 코너가 이어지는..." 식으로 묘사하세요.
       - 주요 랜드마크(라스베이거스의 스피어, 모나코의 터널 등)를 언급하세요.
       
    2. **기술적 분석 (Data + Search)**:
       - `Circuit_Sector_Analyzer`를 통해 이 서킷이 엔진 출력이 중요한지, 공기역학(다운포스)이 중요한지 분석하세요.
       - 타이어 마모도가 심한지, 노면이 거친지 검색(`Circuit_Info_Search`)하여 덧붙이세요.
       
    3. **관전 포인트**:
       - 가장 확실한 추월 포인트(DRS 존)는 어디인가?
       - 사고가 자주 나는 '마의 구간'은 어디인가?
    
    4. **말투**: 전문적이지만 친절하게, 투어 가이드처럼 설명하세요.
    """
    
    return ReActAgent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        verbose=True
    )

# --- [4. 실행 함수 (Retry 적용)] ---
# 503 에러 방어를 위한 강력한 실행 래퍼
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type(ServerError),
    reraise=True
)
async def run_circuit_agent(user_msg: str):
    agent = build_circuit_agent()
    from llama_index.core.workflow import Context
    ctx = Context(agent)
    return await agent.run(user_msg=user_msg, ctx=ctx)

# --- [테스트 실행] ---
if __name__ == "__main__":
    async def test():
        print(" Circuit Agent Initialized (Spec B Chassis).")
        
        # 테스트 질문: 데이터(섹터)와 검색(정보)이 모두 필요한 질문
        q = "2025 브라질 인터라고스 서킷의 특징을 설명해주고, 드라이버들이 주의해야 할 코너가 어딘지 알려줘."
        print(f"\nUser: {q}\n")
        
        try:
            response = await run_circuit_agent(q)
            print(f"\nPitWall(Circuit): {response}")
        except Exception as e:
            print(f" Error during test: {e}")

    asyncio.run(test())
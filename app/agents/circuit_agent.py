import sys
import os
import asyncio
import pandas as pd
from dotenv import load_dotenv
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.core.agent.workflow import ReActAgent
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.genai.errors import ServerError

# 경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# --- [도구 Import] ---
from app.tools.soft_data import search_f1_news 
from data_pipeline.analytics import mini_sector_dominance_analyze, calculate_tire_degradation

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = GoogleGenAI(model="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)
Settings.llm = llm

# --- [0. 한글 -> 영어 서킷 매핑 (안전장치)] ---
KOR_TO_ENG_CIRCUIT = {
    "라스베이거스": "Las Vegas",
    "바레인": "Bahrain",
    "사우디": "Saudi Arabia", "제다": "Jeddah",
    "호주": "Australia", "멜버른": "Melbourne",
    "일본": "Japan", "스즈카": "Suzuka",
    "중국": "China", "상하이": "Shanghai",
    "마이애미": "Miami",
    "이몰라": "Imola", "에밀리아 로마냐": "Emilia Romagna",
    "모나코": "Monaco",
    "캐나다": "Canada", "몬트리올": "Montreal",
    "스페인": "Spain", "바르셀로나": "Barcelona",
    "오스트리아": "Austria", "레드불 링": "Red Bull Ring",
    "영국": "Great Britain", "실버스톤": "Silverstone",
    "헝가리": "Hungary", "헝가로링": "Hungaroring",
    "벨기에": "Belgium", "스파": "Spa",
    "네덜란드": "Netherlands", "잔트부르트": "Zandvoort",
    "이탈리아": "Italy", "몬자": "Monza",
    "아제르바이잔": "Azerbaijan", "바쿠": "Baku",
    "싱가포르": "Singapore",
    "미국": "USA", "오스틴": "Austin", "COTA": "Austin",
    "멕시코": "Mexico",
    "브라질": "Brazil", "인터라고스": "Interlagos",
    "카타르": "Qatar", "루사일": "Lusail",
    "아부다비": "Abu Dhabi"
}

def sanitize_circuit_name(circuit_input: str) -> str:
    """한글 입력이 들어오면 영어 공식 명칭으로 변환"""
    # 1. 입력값 정리 (공백 제거 등)
    clean_input = circuit_input.strip()
    
    # 2. 매핑 확인 (한글 -> 영어)
    for kor, eng in KOR_TO_ENG_CIRCUIT.items():
        if kor in clean_input: # "라스베이거스 서킷" 처럼 포함된 경우도 처리
            print(f" 서킷명 변환: '{circuit_input}' -> '{eng}'")
            return eng
            
    # 3. 매핑 없으면 그냥 영어라고 믿고 반환 (이미 영어인 경우)
    return clean_input

# --- [1. 도구 정의: Analytics Wrapper] ---

# (1) 섹터 분석 도구
def wrapper_sector_analysis(year: int, circuit: str) -> str:
    try:
        eng_circuit = sanitize_circuit_name(circuit) # 변환 적용
        _, summary = mini_sector_dominance_analyze(year, eng_circuit)
        return summary
    except Exception as e:
        return f"섹터 분석 데이터 부족: {e}"

sector_tool = FunctionTool.from_defaults(
    fn=wrapper_sector_analysis,
    name="Circuit_Sector_Analyzer",
    description="과거 주행 데이터를 기반으로 서킷의 고속/저속 섹터 특성을 분석합니다. 서킷 이름은 가능하면 영문(예: 'Las Vegas')으로 입력하세요."
)

# (2) 타이어 분석 도구
def wrapper_tire_analysis(year: int, circuit: str) -> str:
    """
    특정 연도/서킷의 타이어 마모도(Degradation)를 분석합니다.
    """
    eng_circuit = sanitize_circuit_name(circuit) # ★ 여기서 한글을 영어로 바꿉니다!
    print(f" [Tire Analysis] {year} {eng_circuit} 정밀 분석 요청...")
    
    try:
        df_deg = calculate_tire_degradation(year, eng_circuit, session_type='R')
        
        if df_deg.empty:
            return "해당 경기의 타이어 데이터를 추출할 수 없습니다 (FastF1 데이터 없음)."
            
        summary_lines = [f"[{year} {eng_circuit} 타이어 데그라데이션 (연료 보정됨)]"]
        
        compounds = df_deg['Compound'].unique()
        for comp in compounds:
            comp_data = df_deg[df_deg['Compound'] == comp]
            if comp_data.empty: continue
            
            avg_deg = comp_data['True_Degradation'].mean()
            sample_count = len(comp_data)
            
            if avg_deg > 0.10: status = "매우 심각 (High)"
            elif avg_deg > 0.06: status = "보통 (Medium)"
            elif avg_deg > 0.02: status = "양호 (Low)"
            else: status = "거의 없음 (Very Low)"
            
            summary_lines.append(f"- **{comp}** ({sample_count}스틴트): 랩당 +{avg_deg:.3f}초 느려짐 ({status})")
            
        return "\n".join(summary_lines)
        
    except Exception as e:
        return f"타이어 분석 중 오류 발생 ({eng_circuit}): {e}"

tire_tool = FunctionTool.from_defaults(
    fn=wrapper_tire_analysis,
    name="Tire_Degradation_Analyzer",
    description="과거 데이터를 분석하여 타이어 마모도를 수치로 알려줍니다. 서킷 이름은 가능하면 영문으로 입력하세요."
)

# (3) 뉴스 검색 도구
weather_news_tool = FunctionTool.from_defaults(
    fn=search_f1_news,
    name="Live_Condition_Search",
    description="이번 주말의 날씨 예보나 이슈를 검색합니다."
)

# --- [2. RAG 엔진] ---
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/circuits')
PERSIST_DIR = os.path.join(os.path.dirname(__file__), '../../data/storage/circuits')

def get_circuit_query_engine():
    if not os.path.exists(PERSIST_DIR):
        print(f"🏗️ 서킷 지식 베이스 인덱싱 시작...")
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
             raise FileNotFoundError(f"❌ 데이터 폴더가 비어있습니다: {DATA_DIR}")
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index.as_query_engine(similarity_top_k=3)

try:
    circuit_query_engine = get_circuit_query_engine()
    circuit_kb_tool = QueryEngineTool(
        query_engine=circuit_query_engine,
        metadata=ToolMetadata(
            name="Circuit_Knowledge_Base",
            description="서킷의 '정적 정보'(레이아웃, 코너, 특징)를 조회합니다. 우선 사용하세요."
        )
    )
except Exception as e:
    print(f" RAG 엔진 초기화 실패: {e}")
    sys.exit(1)

# --- [3. 에이전트 조립] ---

def build_circuit_agent():
    tools = [circuit_kb_tool, sector_tool, tire_tool, weather_news_tool]
    
    system_prompt = """
    당신은 F1 팀의 '레이스 엔지니어'이자 '트랙 분석가'입니다.
    사용자에게 이번 그랑프리 서킷의 **기술적, 전략적 특징**을 브리핑해야 합니다.
    
    [활용 가능한 도구]
    1. **Circuit_Knowledge_Base**: 서킷의 정적 정보 (우선 사용).
    2. **Tire_Degradation_Analyzer**: 작년 데이터 기반 타이어 마모도 수치. (입력 시 서킷 이름은 영문으로 자동 변환됩니다)
    3. **Circuit_Sector_Analyzer**: 고속/저속 섹터 성향 분석. (입력 시 서킷 이름은 영문으로 자동 변환됩니다)
    4. **Live_Condition_Search**: 날씨 및 뉴스.
    
    [답변 가이드라인]
    1. **전문성 과시**: '더티에어', '그레인/블리스터링', '트랙션' 등 전문 용어 사용.
    2. **데이터 기반**: "분석 결과, 소프트 타이어가 랩당 0.1초씩 느려지는 High Deg 성향입니다"와 같이 구체적으로 답변.
    """
    
    return ReActAgent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        verbose=True
    )

# --- [4. 실행 래퍼] ---
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
        print(" Circuit Agent Initialized")
        
        q = "바쿠 시티 서킷의 특성에 대해서 이것저것 전부 알려줘"
        print(f"\nUser: {q}\n")
        
        try:
            response = await run_circuit_agent(q)
            print(f"\nPitWall(Circuit): {response}")
        except Exception as e:
            print(f" Error: {e}")

    asyncio.run(test())
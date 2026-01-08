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
from data_pipeline.analytics import mini_sector_dominance_analyze

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

# (2) 타이어 열 관리 분석 (Blistering vs Graining)
def analyze_thermal_risk(circuit: str, track_temp_celsius: int = 40) -> str:
    """
    서킷 이름과 트랙 온도를 기반으로 블리스터링(고온) vs 그레이닝(저온) 위험을 진단합니다.
    """
    c_eng = sanitize_circuit_name(circuit).lower() # 영어로 변환 후 소문자화
    
    high_load = ['silverstone', 'spa', 'suzuka', 'barcelona', 'qatar', 'zandvoort', 'great britain', 'japan', 'spain', 'netherlands']
    high_temp = ['bahrain', 'hungary', 'singapore', 'miami', 'austin', 'abu dhabi', 'usa']
    low_grip  = ['las vegas', 'monaco', 'baku', 'mexico', 'monza', 'azerbaijan', 'italy']
    
    risk_type = "Balanced"
    severity = "Low"
    mechanism = "일반적인 타이어 마모 패턴"
    solution = "표준 관리"

    # 로직: 블리스터링 (과열)
    if (c_eng in high_load) or (c_eng in high_temp) or (track_temp_celsius >= 45):
        risk_type = "Blistering (블리스터링)"
        if track_temp_celsius >= 50:
            severity = "Critical"
            mechanism = "초고열로 인한 타이어 내부 파열 위험."
            solution = "내압(Pressure) 낮추고 쿨링 랩 필수."
        elif c_eng in high_load:
            severity = "High"
            mechanism = "고속 코너 횡가속도로 인한 코어 온도 급상승."
            solution = "고속 구간에서 부하 조절(Lift & Coast)."
        else:
            severity = "Medium"
            mechanism = "높은 기온으로 인한 컴파운드 과열."
    
    # 로직: 그레이닝 (저온/슬립)
    elif (c_eng in low_grip) or (track_temp_celsius <= 25):
        risk_type = "Graining (그레이닝)"
        if track_temp_celsius <= 20:
            severity = "High"
            mechanism = "저온으로 타이어가 딱딱해져 노면을 잡지 못하고 미끄러짐."
            solution = "공격적인 웜업으로 작동 온도 유지."
        else:
            severity = "Medium"
            mechanism = "낮은 그립으로 인한 표면 뜯김 현상."
            solution = "부드러운 스티어링 입력 필요."

    return f"[{circuit} Thermal Report]\n- Risk: {risk_type}\n- Severity: {severity}\n- Cause: {mechanism}\n- Tip: {solution}"

# (3) 에어로 셋업 분석
def analyze_aero_setup(circuit: str) -> str:
    """서킷 특성에 맞는 최적의 다운포스 셋업을 제안합니다."""
    c_eng = sanitize_circuit_name(circuit).lower()
    
    if c_eng in ['monaco', 'singapore', 'hungary', 'mexico']:
        setup = "Maximum Downforce"
        desc = "공기 저항을 무시하고 코너링 그립 극대화 (Barn Door Wing)."
    elif c_eng in ['monza', 'las vegas', 'baku', 'spa', 'italy', 'azerbaijan', 'belgium']:
        setup = "Low Drag (Skinny Wing)"
        desc = "직선 최고 속도가 핵심. 다운포스를 희생하여 드래그 최소화."
    elif c_eng in ['silverstone', 'suzuka', 'austin', 'barcelona', 'great britain', 'japan', 'spain', 'usa']:
        setup = "Medium-High Efficiency"
        desc = "고속 코너 안정성과 직선 속도의 균형점 필요."
    else:
        setup = "Medium Downforce"
        desc = "범용 셋업."
        
    return f"[{circuit} Aero Guide]\n- Target: {setup}\n- Reason: {desc}"

# 도구 등록
thermal_tool = FunctionTool.from_defaults(
    fn=analyze_thermal_risk,
    name="Tire_Thermal_Analysis",
    description="서킷의 온도, 이에 의한 타이어 그레이닝 / 타이어 블리스터링 이슈를 확인합니다")


aero_tool = FunctionTool.from_defaults(fn=analyze_aero_setup, name="Aero_Setup_Guide")

# (4) 뉴스 검색 도구
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
        print(f" 서킷 지식 베이스 인덱싱 시작...")
        if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
             raise FileNotFoundError(f" 데이터 폴더가 비어있습니다: {DATA_DIR}")
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
    tools = [circuit_kb_tool, thermal_tool, sector_tool, aero_tool, weather_news_tool]
    
    system_prompt = """
    당신은 F1 팀의 '레이스 엔지니어'이자 '트랙 분석가'입니다.
    사용자에게 이번 그랑프리 서킷의 **기술적, 전략적 특징**을 브리핑해야 합니다.
    
    [활용 가능한 도구]
    1. **Tire_Thermal_Analysis**: 서킷과 온도에 따른 **블리스터링(고온) vs 그레이닝(저온)** 위험 진단. (가장 중요)
    2. **Aero_Setup_Guide**: 다운포스 셋업 방향성 제시.
    3. **Circuit_Sector_Analyzer**: 섹터별 강세(파워 유닛 vs 에어로) 분석.
    4. **Live_Condition_Search**: 날씨 및 뉴스.
    
    [답변 가이드라인]
    1. **전문성 과시**: '더티에어', '그레인/블리스터링', '트랙션' 등 전문 용어 사용.
    2. **데이터 기반**: "분석 결과, 소프트 타이어가 랩당 0.1초씩 느려지는 High Deg 성향입니다"와 같이 구체적으로 답변.
    3. 단순히 "타이어가 마모됩니다"라고 하지 말고, **"블리스터링 위험이 있으니 내압 관리가 필요합니다"** 처럼 구체적인 원인과 해결책을 제시하십시오.
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
        
        q = "라스베이거스 스트립 서킷의 특성에 대해서 알려줘. 다운포스 요구량, 타이어 관리 특성같은것도 알려줘."
        print(f"\nUser: {q}\n")
        
        try:
            response = await run_circuit_agent(q)
            print(f"\nPitWall(Circuit): {response}")
        except Exception as e:
            print(f" Error: {e}")

    asyncio.run(test())
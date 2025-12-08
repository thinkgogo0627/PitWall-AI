## Text - to - SQL 모델

## Hard data 를 기반으로 SQL문 생성

## 퓨샷 러닝(Few-Shot learning): '이런 질문에는 이런 식으로 짜면 된다'로 인스트럭션 제공
### NLSQLTableQueryEngine << 위 기능 자동으로 구현

import sys
import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../../'))
env_path = os.path.join(root_dir, '.env')

# .env 로드 실행
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f" .env 로드 성공: {env_path}")
else:
    print(f" .env 파일을 찾을 수 없습니다: {env_path}")

api_key = os.getenv("GOOGLE_API_KEY")

from sqlalchemy import create_engine
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    device="cuda" # GPU 없으면 "cpu"
)

Settings.llm = Gemini(model="models/gemini-flash-latest", api_key=api_key)



# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 1. DB 연결 (SQLAlchemy Engine 사용)
# SQLite 파일 경로 지정
db_path = "sqlite:///data/f1_data.db" # 주의: sqlite:/// 접두사 필수
engine = create_engine(db_path)

# 2. SQLDatabase 객체 생성
# include_tables: LLM이 볼 수 있는 테이블을 지정 (보안 및 정확도 향상)
sql_database = SQLDatabase(
    engine, 
    include_tables=["race_results", "lap_times", "weather_data"]
)

# 3. ★ 핵심: 퓨샷(Few-Shot) 예시 주입 ★
# LLM에게 "이렇게 짜는 거야"라고 가르치는 족보입니다.
# 아까 님이 만든 그 3개의 쿼리를 여기 넣습니다.
text_to_sql_prompt_instruction = """
당신은 F1 데이터 분석 전문가입니다. 사용자의 자연어 질문을 실행 가능한 SQL 쿼리로 변환하고, 그 결과를 분석해주세요.
데이터베이스는 SQLite를 사용합니다.

[테이블 정보]
- race_results: 경기 최종 순위, 포인트, 그리드 위치 등
- lap_times: 모든 드라이버의 모든 랩 타임, 타이어 정보 (초 단위: LapTime_Sec)
- weather_data: 트랙 온도, 기온, 습도 등

[쿼리 작성 가이드]
1. 중반 페이스 분석 시: 전체 랩의 15% ~ 85% 구간만 필터링하여 평균을 구하세요.
2. 순위 상승폭(Overtaking): (GridPosition - Position)으로 계산하세요.
3. 쿼리는 반드시 SQLite 문법을 따르세요.

[예시 1: 타이어별 평균 페이스 비교]
Q: "라스베가스에서 타이어별 평균 랩타임 보여줘"
SQL: 
SELECT Driver, Compound, COUNT(*) as Laps, ROUND(AVG(LapTime_Sec), 3) as Avg_Pace 
FROM lap_times 
WHERE RaceID LIKE '%Las_Vegas%' AND IsAccurate = 'True' 
GROUP BY Driver, Compound ORDER BY Driver;

[예시 2: 순위 상승폭]
Q: "가장 많이 순위를 올린 선수는?"
SQL: 
SELECT Driver, (GridPosition - Position) as Positions_Gained 
FROM race_results 
WHERE RaceID LIKE '%Las_Vegas%' AND Position IS NOT NULL 
ORDER BY Positions_Gained DESC LIMIT 5;
"""

# 4. 쿼리 엔진 생성 (이 녀석이 통역사입니다)
# LLM은 이미 agent.py에서 설정하겠지만, 여기서도 명시적으로 지정 가능
llm = Gemini(model="models/gemini-flash-latest")

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["race_results", "lap_times", "weather_data"],
    llm=llm,
    context_query_kwargs={"race_results": text_to_sql_prompt_instruction} # 프롬프트 주입
)

# 5. 최종 도구 함수 (Agent가 갖다 쓸 함수)
def analyze_race_data(query: str):
    """
    자연어 질문을 받아 실제 DB에서 SQL을 실행하여 분석 결과를 반환합니다.
    "베르스타펜의 평균 랩타임은?", "순위를 가장 많이 올린 사람은?" 같은 질문에 사용합니다.
    """
    print(f" Hard Data 분석 요청: '{query}'")
    
    try:
        # 1. LLM이 SQL 생성 -> 2. 실행 -> 3. 결과를 텍스트로 요약
        response = query_engine.query(query)
        
        # response.metadata['sql_query'] 에 실제 실행된 SQL이 들어있음 (디버깅용)
        executed_sql = response.metadata.get('sql_query', 'SQL 정보 없음')
        print(f"   └ 생성된 SQL: {executed_sql}")
        
        return f"[분석 결과]\n{response.response}\n\n(참고 SQL: {executed_sql})"
        
    except Exception as e:
        return f"데이터 분석 중 오류가 발생했습니다: {e}"

# --- 테스트 실행 ---
if __name__ == "__main__":
    # 테스트
    print(analyze_race_data("작년 라스베가스 GP에서 타이어별로 드라이버들의 평균 기록을 비교해줘."))
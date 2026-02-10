## Text - to - SQL 모델

## Hard data 를 기반으로 SQL문 생성

## 퓨샷 러닝(Few-Shot learning): '이런 질문에는 이런 식으로 짜면 된다'로 인스트럭션 제공
### NLSQLTableQueryEngine << 위 기능 자동으로 구현

import sys
import os
from dotenv import load_dotenv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
env_path = os.path.join(PROJECT_ROOT, '.env')
DB_FILE_PATH = os.path.join(PROJECT_ROOT, 'data', 'f1_data.db')


# .env 로드 실행
if os.path.exists(env_path):
    load_dotenv(env_path)
    print(f" .env 로드 성공: {env_path}")
else:
    print(f" .env 파일을 찾을 수 없습니다: {env_path}")

api_key = os.getenv("GOOGLE_API_KEY")

# 시스템 경로에 프로젝트 루트 추가
sys.path.append(PROJECT_ROOT)

from sqlalchemy import create_engine
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core import Settings, PromptTemplate
from llama_index.llms.google_genai import GoogleGenAI
from tenacity import retry, stop_after_attempt, wait_exponential # 재시도 로직


Settings.llm = GoogleGenAI(model="models/gemini-2.5-pro", api_key=api_key)


# 프로젝트 루트 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# 1. DB 연결 (SQLAlchemy Engine 사용)
# SQLite 파일 경로 지정
db_connection_str = f"sqlite:///{DB_FILE_PATH}"
engine = create_engine(db_connection_str)
sql_database = SQLDatabase(engine, include_tables=["race_results", "lap_times", "weather_data"])


# 3. ★ 핵심: 퓨샷(Few-Shot) 예시 주입 ★
# LLM에게 "이렇게 짜는 거야"라고 가르치는 족보
combined_prompt_str = """
당신은 F1 데이터 분석 전문가입니다. 사용자의 자연어 질문을 실행 가능한 SQL 쿼리로 변환하고, 그 결과를 분석해주세요.
데이터베이스는 SQLite를 사용합니다.

[제약 사항 - 매우 중요!]
1. **절대로 마크다운(```sql) 포맷을 사용하지 마세요.** 오직 SQL 쿼리 문장만 출력하세요.
2. 쿼리 끝에 세미콜론(;)을 붙이세요.
3. 존재하지 않는 컬럼을 지어내지 마세요.
4. **IsAccurate 컬럼은 Boolean 타입입니다. (1=True, 0=False)**
   - 올바른 예: WHERE IsAccurate = 1
   - 틀린 예: WHERE IsAccurate = 'True'
5. **'No' 혹은 'CarNumber' 컬럼은 존재하지 않습니다.** 절대 SELECT 하지 마세요.
5-1. 그랑프리 장소 질문에 따른 RaceID 장소 참조입니다. 반드시 이 형태를 사용하여 쿼리를 만드세요
    - 오스트레일리아, 호주 : RaceID LIKE %Australian%
    - 중국 : RaceID LIKE %Chinese%
    - 일본, 스즈카 : RaceID LIKE %Japanese%
    - 사우디아라비아, 사우디 : RaceID LIKE %Saudi_Arabian%
    - 바레인 : RaceID LIKE %Bahrain%
    - 마이애미 : RaceID LIKE %Miami%
    - 에밀리아 로마냐 : RaceID LIKE %Emilia_Romagna%
    - 모나코 : RaceID LIKE %Monaco%
    - 스페인 : RaceID LIKE %Spanish%
    - 오스트리아: RaceID LIKE %Austrian%
    - 포르투갈: RaceID LIKE %Portuguese%
    - 캐나다 : RaceID LIKE %Canadian%
    - 영국, 실버스톤 : RaceID LIKE %British%
    - 벨기에 : RaceID LIKE %Belgian%
    - 헝가리 : RaceID LIKE %Hungarian%
    - 네덜란드 : RaceID LIKE %Dutch%
    - 이탈리아, 몬차 : RaceID LIKE %Italian%
    - 아제르바이잔, 바쿠 : RaceID LIKE %Azerbaijan%
    - 싱가포르 : RaceID LIKE %Singapore%
    - 미국, 오스틴 : RaceID LIKE %United_States%
    - 브라질, 상파울루 : RaceID LIKE %São_Paulo%
    - 라스베가스 : RaceID LIKE %Las_Vegas%
    - 카타르 : RaceID LIKE %Qatar%
    - 아부다비 : RaceID LIKE %Abu_Dhabi%

5-2. [Driver Numbers Reference, (드라이버 이름, 약어) - 차량 번호]

    - Max Verstappen (막스 베르스타펜, VER): 1

    - Yuki Tsunoda (유키 츠노다, TSU): 22

    - Lando Norris (랜도 노리스, NOR): 4

    - Oscar Piastri (오스카 피아스트리, PIA): 81

    - Lewis Hamilton (루이스 해밀턴, HAM): 44

    - Charles Leclerc (샤를 르클레르, LEC): 16

    - George Russell (조지 러셀, RUS): 63

    - Kimi Antonelli (키미 안토넬리, ANT): 12  

    - Liam Lawson (리암 로슨, LAW): 30

    - Isack Hadjar (아이작 하자르, HAD): 6

    - Gabriel Bortoleto (가브리엘 보톨레토, BOR): 5

    - Nico Hülkenberg (니코 훌켄베르크, HUL): 27

    - Franco Colapinto (프랑코 콜라핀토, COL): 43

    - Pierre Gasly (피에르 가슬리, GAS): 10

    - Alex Albon (알렉스 알본, ALB): 23

    - Carlos Sainz (카를로스 사인츠, SAI): 55

    - Lance Stroll (랜스 스트롤, STR): 18

    - Fernando Alonso (페르난도 알론소, ALO): 14

    - Esteban Ocon (에스테반 오콘, OCO): 31

    - Olliver Bearman (올리버 베어만, BEA): 87
6. 드라이버 식별은 오직 `Driver` 약어로만 하세요 (LIKE %ANT%) , (LIKE %RUS%)

   
[테이블 정보]
- race_results(RaceID, Driver, Position, GridPosition, Points, Status, Year, Circuit)
- lap_times(RaceID, Driver, LapNumber, LapTime_Sec, Compound, TyreLife, IsAccurate)
- weather_data(RaceID, AirTemp, TrackTemp, Humidity, Rainfall)


[쿼리 작성 가이드]
1. 중반 페이스 분석 시: 전체 랩의 15% ~ 85% 구간만 필터링하여 평균을 구하세요.
2. 순위 상승폭(Overtaking): (GridPosition - Position)으로 계산하세요.
3. 쿼리는 반드시 SQLite 문법을 따르세요.

[★ ABSOLUTE RULES - 어기면 에러로 간주함 ★]
1. **무조건 와일드카드(%) 사용**: 
   - 문자열 검색(RaceID, Driver)은 100% 확률로 **LIKE '%keyword%'** 형식을 써야 합니다.
   - (X) RaceID = '2025_Las_Vegas'
   - (O) RaceID LIKE '%Las_Vegas%'
2. **드라이버 검색**: 
   - 이름의 일부만 있어도 찾을 수 있게 앞뒤로 %를 붙이세요. (예: LIKE '%ANT%')
3. **Boolean 처리**: IsAccurate = 1 (True), 0 (False)
4. **세션 구분**: 사용자가 특별히 "연습(Practice)", "예선(Qualifying)"을 언급하지 않으면, 
   기본적으로 **RaceID에 'Grand_Prix' 또는 'Race'가 포함된 데이터만 조회**하세요.
   (AND RaceID NOT LIKE '%Practice%' AND RaceID NOT LIKE '%Qualifying%')

5. **차량 번호 필수 포함**: 드라이버나 순위 관련 조회 시, 반드시 **'No' 컬럼(차량 번호)**을 SELECT 절에 포함하세요.
   - (X) SELECT Driver, Position ...
   - (O) SELECT No, Driver, Position ...  <-- 이렇게!
   

6. **쿼리 제작 시 반드시 팀 정보 (TeamName) 컬럼을 추가하세요
    - "IMPORTANT INSTRUCTION: "
        "1. When selecting driver results, **ALWAYS include the 'TeamName' column** in the SELECT clause. "
        "2. Do not hallucinate. If the user asks for results, query the 'race_results' table. "
        "3. Select columns: Position, Driver, TeamName, GridPosition, Points, Status."


[예시 1: 타이어별 평균 페이스 비교]
Q: "라스베가스에서 타이어별 평균 랩타임 보여줘"
SQL: 
SELECT Driver, Compound, COUNT(*) as Laps, ROUND(AVG(LapTime_Sec), 3) as Avg_Pace 
FROM lap_times 
WHERE RaceID LIKE '%Las_Vegas%' AND IsAccurate = 1
GROUP BY Driver, Compound ORDER BY Driver;

[예시 2: 순위 상승폭]
Q: "가장 많이 순위를 올린 선수는?"
SQL: 
SELECT Driver, (GridPosition - Position) as Positions_Gained 
FROM race_results 
WHERE RaceID LIKE '%Las_Vegas%' AND Position IS NOT NULL 
ORDER BY Positions_Gained DESC LIMIT 5;


[예시 3: 타이어 컴파운드별 랩타임 비교]
Q: "랩의 중반 페이스에서, 각 타이어 컴파운드별 드라이버들의 랩타임을 비교해줘"
SQL: 
SELECT Driver, Compound, COUNT(*) as Laps_Run,ROUND(AVG(LapTime_Sec), 3) as Avg_Pace,ROUND(MIN(LapTime_Sec), 3) as Best_Lap
FROM lap_times
WHERE RaceID LIKE '2025%'
  AND IsAccurate = 1
  AND LapNumber BETWEEN 
      (SELECT MAX(LapNumber) * 0.15 FROM lap_times WHERE RaceID LIKE '2025%') 
      AND 
      (SELECT MAX(LapNumber) * 0.85 FROM lap_times WHERE RaceID LIKE '2025%')
GROUP BY Driver, Compound
ORDER BY Driver ASC;


[예시 4: 단순 순위/결과 조회]
Q: "2023 싱가포르에서 카를로스 사인츠 순위 알려줘"
SQL: SELECT Position, Driver, Status, Points, TeamName
FROM race_results
WHERE RaceID LIKE '%Singapore%' AND Driver LIKE '%SAI%';

Question: {query_str}
SQLQuery:
"""

# 4. 쿼리 엔진 생성
# LLM은 이미 agent.py에서 설정하겠지만, 여기서도 명시적으로 지정 가능
text_to_sql_template = PromptTemplate(combined_prompt_str)
# 모든 테이블에 족보(Prompt)를 연결해줍니다.

query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["race_results", "lap_times", "weather_data"],
    llm=Settings.llm
)

# ★ 여기서 엔진의 뇌를 갈아끼웁니다. (핵심)
query_engine.update_prompts(
    {"sql_retriever:text_to_sql_prompt": text_to_sql_template}
)

@retry(
    stop=stop_after_attempt(10), 
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True
)
def _query_with_retry(query_str):
    """
    실제 쿼리 엔진을 호출하는 내부 함수. 
    503 에러가 나면 여기서 자동으로 재시도합니다.
    """
    return query_engine.query(query_str)



# 5. 최종 도구 함수 (Agent가 갖다 쓸 함수)
def analyze_race_data(query: str) -> str:
    """
    자연어 질문을 받아 실제 DB에서 SQL을 실행하여 분석 결과를 반환합니다.
    """
    print(f" Hard Data 분석 요청: '{query}'")

    try:
        #  [수정 포인트] 프롬프트 엔지니어링 (Prompt Injection)
        # LLM에게 "팀 컬럼도 무조건 SELECT에 넣어라"고 강제.
        enhanced_query = (
            f"{query} "
            "IMPORTANT: When selecting driver results, **ALWAYS include the 'TeamName' column** in the SELECT clause. "
            "Do not guess the team, fetch it from the database."
        )

        # ★ 수정됨: 원래 query 대신 enhanced_query를 넘깁니다.
        response = _query_with_retry(enhanced_query)
        
        executed_sql = response.metadata.get('sql_query', 'SQL 정보 없음')
        print(f"   └ 생성된 SQL: {executed_sql}")
        
        return f"[분석 결과]\n{response.response}\n\n(실행된 SQL: {executed_sql})"
        
    except Exception as e:
        # 10번 다 재시도했는데도 실패하면 여기로 옵니다.
        print(f" SQL 실행 에러 (최종 실패): {e}")
        return f"데이터 분석 중 오류 발생 (서버 과부하): {e}"


# --- 테스트 실행 ---
if __name__ == "__main__":
    # 테스트 1: 단순 순위 (이제 %가 잘 들어갈 것임)
    print(analyze_race_data("2025 라스베이거스에서 안토넬리 순위 알려줘"))
    
    # 테스트 2: 복잡한 쿼리 (님의 퓨샷 로직 확인)
    print("\n" + "="*30 + "\n")
    print(analyze_race_data("2025 라스베가스 GP에서 타이어별 평균 랩타임은?"))
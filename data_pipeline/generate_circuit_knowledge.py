## 핵심 기능 2 , 핵심 기능 3을 위해 사용될
## 어차피 24개의 데이터이므로 굳이 ChromaDB에 연동할 필요는 없음

'''
위키피디아 데이터 >> 지나치게 데이터 양 적음
F1 공식 웹사이트 데이터 >> 비디오 형식

상업용 모델(Gemini) 에 탑재된 서킷 지식을 도메인에 맞춰서 정제된 텍스트 데이터로 추출
-> 지식 증류..

데이터 전처리 불필요
'''

import os
import asyncio
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 저장 경로
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data/circuits')
os.makedirs(DATA_DIR, exist_ok=True)

# 2025 서킷 목록
CIRCUITS = [
    "Bahrain International Circuit", "Jeddah Corniche Circuit", "Albert Park Circuit",
    "Suzuka International Racing Course", "Shanghai International Circuit", "Miami International Autodrome",
    "Imola Circuit", "Circuit de Monaco", "Circuit Gilles Villeneuve", "Circuit de Barcelona-Catalunya",
    "Red Bull Ring", "Silverstone Circuit", "Hungaroring", "Circuit de Spa-Francorchamps",
    "Circuit Zandvoort", "Monza Circuit", "Baku City Circuit", "Marina Bay Street Circuit",
    "Circuit of the Americas", "Autódromo Hermanos Rodríguez", "Interlagos Circuit",
    "Las Vegas Strip Circuit", "Lusail International Circuit", "Yas Marina Circuit"
]


## 서킷 설명서 양식
PROMPT_TEMPLATE = """
당신은 F1 서킷 엔지니어링 전문가입니다. 
아래 서킷에 대한 '정적 기술 보고서(Technical Track Guide)'를 작성해 주세요.
반드시 아래 목차와 형식을 따라야 하며, 한국어로 작성하세요.

**대상 서킷:** {circuit_name}
**!!미사여구 작성금지!!**

---
### 1. 서킷 개요
- **특성:** (예: 초고속 시가지 서킷, 다운포스가 중요한 전통 서킷, 고저차가 심한 롤러코스터형, 뱅크각의 유무 등)
- **주요 랜드마크:** (예: 바쿠의 성벽, 모나코의 페어몬트 헤어핀/터널, 라스베이거스의 스피어 등 시각적 특징)
- **기술적 특성:** (총 랩 수, 반시계방향 서킷 or 시계방향 서킷, 코너 갯수, 트랙 길이, 피트레인 속도 제한)
- **환경적 특성:** (서킷 폭, 해발 고도, 기온, 더운 곳인지 비가 많이 오는곳인지, 바람이 많이 부는곳인지 뭐 그런거..)

### 2. 레이아웃 상세 분석
- **코너 구성:** (저속/중속/고속 코너의 비율 및 특성)
- **주요 코너 이름:** (예: Senna 'S', Parabolica, Copse, Eau Rouge 등 유명한 코너 이름과 그 특징)
- **DRS 존:** (DRS 존의 개수와 위치, 추월이 가장 많이 일어나는 곳)
- **추월 포인트:** (다양한 레이스 라인, 레이트 브레이킹 유무 등을 통해 주요 추월 포인트 서술)

### 3. 엔지니어링 포인트
- **다운포스 요구량:** (Low / Medium / High)
- **타이어 스트레스:** (Lateral / Longitudinal 부하가 심한지, 타이어 마모도가 높은지)
- **브레이크 부하:** (브레이크 냉각이 중요한지, Hard Braking 구간이 많은지)

### 4. 드라이버 및 전략 주의사항
- 트랙 리밋이 엄격한 곳, 락업(Lock-up)이 자주 걸리는 곳, 사고 다발 구간 등.
- 다운포스 , 엔진 성능 , (고속 코너 성능/중속 코너 성능/저속 코너 성능) , 더티에어 내성 중 무엇이 중요한지 서술
- SC 및 VSC 빈도 서술, 언더컷, 또는 오버컷이 쉬운지 어려운지 그 이유와 함께 서술

위 내용을 아주 구체적이고 전문적인 용어를 사용하여 작성하세요.
**대상 서킷** 외에 다른 미사여구 (분석 날짜, 분석 전달 팀) 은 절대 기술하지 마세요.
"""

async def generate_all_circuits():
    llm = GoogleGenAI(model="models/gemini-2.5-pro", api_key=api_key)
    
    print(" LLM 기반 서킷 지식 베이스 구축 시작...\n")

    for circuit in CIRCUITS:
        file_path = os.path.join(DATA_DIR, f"{circuit.replace(' ', '_')}.txt")
        
        # 이미 파일이 있으면 건너뛰기 (비용 절약)
        if os.path.exists(file_path):
            print(f" Skip: {circuit} (이미 존재함)")
            continue

        print(f" Generating: {circuit}...")
        
        try:
            response = await llm.acomplete(PROMPT_TEMPLATE.format(circuit_name=circuit))
            content = response.text
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"✅ Saved: {file_path}")
            
        except Exception as e:
            print(f" Failed: {circuit} - {e}")

    print("\n 모든 서킷 데이터 생성 완료!")

if __name__ == "__main__":
    asyncio.run(generate_all_circuits())
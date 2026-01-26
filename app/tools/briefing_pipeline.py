import asyncio
from app.tools.hard_data import analyze_race_data
from app.tools.soft_data import get_event_timeline
from llama_index.llms.google_genai import GoogleGenAI
import os

# LLM 직접 호출용 (Agent Loop 안 거침)
llm = GoogleGenAI(model="models/gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))

async def generate_quick_summary(year, gp, driver_focus=None):
    """
    [Fast Pipeline] 
    ReAct 루프 없이 데이터를 병렬로 긁어온 뒤, 단 한 번의 LLM 호출로 요약본을 생성합니다.
    """
    try:
        query_topic = f"{year} {gp}"
        
        # 1. 데이터 병렬 수집 (Parallel Execution) - 속도의 핵심!
        # 결과 데이터와 타임라인(사고/이슈)을 동시에 가져옵니다.
        results_task = asyncio.to_thread(analyze_race_data, query_topic)
        timeline_task = asyncio.to_thread(get_event_timeline, query_topic)
        
        results, timeline = await asyncio.gather(results_task, timeline_task)
        
        # 2. 프롬프트 조립 (Context Injection)
        prompt = f"""
        당신은 F1 수석 저널리스트입니다. 아래 제공된 Raw Data를 바탕으로 브리핑 리포트를 작성하세요.
        
        [RAW DATA]
        - Race Results: {results}
        - Key Events (Timeline): {timeline}
        - Focus Driver: {driver_focus if driver_focus else "Winner & Key Players"}

        [작성 지침]
        1. **서사(Narrative) 강조:** 단순히 순위만 나열하지 말고, 타임라인을 참고하여 "어떻게 그 순위가 되었는지" 설명하세요.
           (예: "원래 5위였으나 앞선 차량의 실격(DSQ)으로 인해 3위로 포디움에 올랐습니다.")
        2. **결정적 순간:** 타임라인에서 'Retirement', 'Crash', 'Penalty', 'DSQ' 키워드가 있다면 반드시 강조하세요.
        3. **한국어**로 명확하고 전문적으로 작성하세요.
        4. 출력은 마크다운 형식으로 헤드라인, 경기 요약, 주요 이슈(DNF/DSQ) 순으로 정리하세요.
        """

        # 3. 단발성 추론 (One-shot Generation)
        response = await llm.acomplete(prompt)
        return str(response)

    except Exception as e:
        return f"🚨 파이프라인 에러 발생: {str(e)}"
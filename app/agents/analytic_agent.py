# app/agents/analyst_agent.py

import sys
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import ReActAgent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
load_dotenv()

# [Tools Import]
from app.tools.telemetry_data import (
    generate_lap_comparison_plot, 
    generate_track_dominance_plot, 
    generate_speed_trace_plot
)

llm = GoogleGenAI(model="models/gemini-2.0-flash-exp", api_key=os.getenv("GOOGLE_API_KEY"))
Settings.llm = llm

# [Tool Definitions]
tools = [
    FunctionTool.from_defaults(
        fn=generate_lap_comparison_plot,
        name="Generate_Lap_Plot",
        description="[Pace Analysis] 두 드라이버의 랩타임 비교 그래프(Line Chart)를 생성합니다. 인자: year, race, driver1, driver2"
    ),
    FunctionTool.from_defaults(
        fn=generate_track_dominance_plot,
        name="Generate_Dominance_Map",
        description="[Track Map] 서킷의 구간별 우세(Dominance)를 보여주는 트랙 지도를 생성합니다. '어디서 빨랐어?', '맵 보여줘' 질문용."
    ),
    FunctionTool.from_defaults(
        fn=generate_speed_trace_plot,
        name="Generate_Speed_Trace",
        description="[Speed Analysis] 최고 속도와 코너링 속도를 비교하는 스피드 그래프를 생성합니다. '누가 더 빨라?', '속도 비교' 질문용."
    )
]

def build_analyst_agent():
    system_prompt = """
    당신은 F1 팀의 **수석 데이터 전략가(Chief Data Strategist)**입니다.
    당신의 역할은 말보다는 **시각화된 차트(Graph & Map)**로 드라이버의 퍼포먼스를 증명하는 것입니다.

    [행동 지침]
    1. 사용자의 의도를 파악하여 가장 적절한 도구 **하나**를 선택하십시오.
       - "꾸준함", "페이스 비교", "누가 이겼어?" -> `Generate_Lap_Plot`
       - "어느 코너에서?", "트랙 맵", "섹터 비교" -> `Generate_Dominance_Map`
       - "최고 속도", "직선 속도", "코너링 스피드" -> `Generate_Speed_Trace`
    
    2. 도구 실행 후 반환된 **'GRAPH_GENERATED: [경로]'**를 확인하십시오.
    
    3. 답변 시:
       - 반드시 생성된 이미지 파일이 있다는 것을 언급하십시오.
       - 그래프를 분석하는 전문가다운 짧은 코멘트를 덧붙이십시오. 
         (예: "직선 구간에서는 VER가 빠르지만, 저속 코너에서는 NOR가 우세합니다.")
    
    [주의사항]
    - 드라이버 이름은 반드시 **3글자 대문자 약어** (예: VER, HAM, NOR, LEC)로 변환하십시오.
    - 데이터가 없을 경우 솔직하게 "해당 세션의 텔레메트리 데이터를 확보하지 못했습니다."라고 보고하십시오.
    """
    
    return ReActAgent(llm=llm, tools=tools, system_prompt=system_prompt, verbose=True)

async def run_analyst_agent(user_msg: str):
    agent = build_analyst_agent()
    return await agent.run(user_msg=user_msg)

if __name__ == "__main__":
    import asyncio
    async def test():
        # 테스트: 속도 비교 (Speed Trace)
        q = "2024년 마이애미에서 베르스타펜이랑 노리스 최고 속도 비교해줘"
        print(f"User: {q}")
        response = await run_analyst_agent(q)
        print(f"Agent: {response}")

    asyncio.run(test())
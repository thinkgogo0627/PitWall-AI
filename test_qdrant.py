"""
RAG 연동 격리 테스트 스크립트
실행: python test_rag.py

테스트 항목:
1. Qdrant 연결 상태
2. 각 컬렉션 존재 여부 및 문서 수
3. 실제 검색 쿼리 동작 확인 (soft_data.py 함수들)
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ────────────────────────────────────────
# 1. Qdrant 연결 및 컬렉션 상태 확인
# ────────────────────────────────────────
def test_qdrant_connection():
    print("\n" + "="*50)
    print("[ TEST 1 ] Qdrant 연결 및 컬렉션 상태")
    print("="*50)

    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        collections = client.get_collections().collections
        print(f"✅ Qdrant 연결 성공: {QDRANT_URL}")
        print(f"   총 컬렉션 수: {len(collections)}")

        if not collections:
            print("⚠️  컬렉션이 하나도 없습니다. RAG 인덱싱이 안 된 상태입니다.")
            return False

        for col in collections:
            info = client.get_collection(col.name)
            count = info.points_count
            status = info.status
            print(f"   - [{status}] {col.name}: {count}개 문서")

        return True

    except Exception as e:
        print(f"❌ Qdrant 연결 실패: {e}")
        return False


# ────────────────────────────────────────
# 2. 각 RAG 함수 실제 쿼리 테스트
# ────────────────────────────────────────
def test_rag_functions():
    print("\n" + "="*50)
    print("[ TEST 2 ] soft_data.py 함수별 검색 테스트")
    print("="*50)

    try:
        from app.tools.soft_data import (
            get_driver_interview,
            search_technical_analysis,
            get_event_timeline,
            search_f1_news_web,
        )
        print("✅ soft_data.py import 성공\n")
    except Exception as e:
        print(f"❌ soft_data.py import 실패: {e}")
        return

    test_cases = [
        ("get_driver_interview",      get_driver_interview,      "피아스트리 호주 GP 사고"),
        ("search_technical_analysis", search_technical_analysis, "맥라렌 파워유닛 전기 결함"),
        ("get_event_timeline",        get_event_timeline,        "2026 호주 그랑프리 사건"),
        ("search_f1_news_web",        search_f1_news_web,        "피아스트리 DNS 2026"),
    ]

    for name, func, query in test_cases:
        print(f"--- {name}({repr(query)}) ---")
        try:
            result = func(query)
            if not result or "결과가 없" in result or "찾을 수 없" in result:
                print(f"⚠️  결과 없음 또는 빈 응답:\n   {result[:200]}")
            else:
                # 결과 앞 300자만 출력
                preview = result[:300].replace('\n', ' ')
                print(f"✅ 결과 있음 ({len(result)}자):\n   {preview}...")
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
        print()


# ────────────────────────────────────────
# 3. regulation_tool 테스트
# ────────────────────────────────────────
def test_regulation_tool():
    print("\n" + "="*50)
    print("[ TEST 3 ] regulation_tool 검색 테스트")
    print("="*50)

    try:
        from app.regulation_tool import search_fia_regulations
        result = search_fia_regulations("reconnaissance lap accident DNS")
        if not result or "찾을 수 없" in result:
            print(f"⚠️  결과 없음:\n   {result}")
        else:
            preview = result[:300].replace('\n', ' ')
            print(f"✅ 결과 있음 ({len(result)}자):\n   {preview}...")
    except Exception as e:
        print(f"❌ regulation_tool 에러: {e}")


# ────────────────────────────────────────
# 메인 실행
# ────────────────────────────────────────
if __name__ == "__main__":
    print("\n🏎️  PitWall-AI RAG 격리 테스트 시작")

    qdrant_ok = test_qdrant_connection()

    if qdrant_ok:
        test_rag_functions()
        test_regulation_tool()
    else:
        print("\n⛔ Qdrant 연결 실패로 이후 테스트를 건너뜁니다.")
        print("   .env의 QDRANT_URL / QDRANT_API_KEY 확인 또는 Qdrant 컨테이너 상태 확인 필요.")

    print("\n🏁 테스트 완료")
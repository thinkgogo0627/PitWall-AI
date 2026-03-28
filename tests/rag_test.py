"""
PitWall-AI RAG Retrieve 품질 테스트 스크립트
실행: python test_retriever_quality.py

테스트 항목:
- 다양한 F1 질문에 대해 Qdrant에서 관련 문서를 retrieve
- 각 쿼리별 상위 K개 문서의 제목/내용/유사도 점수 출력
- retrieve 품질을 직접 눈으로 확인
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

from dotenv import load_dotenv
load_dotenv()

# ────────────────────────────────────────
# 테스트 쿼리 목록
# 카테고리별로 구성 — 잘 retrieve되는지 직접 확인
# ────────────────────────────────────────
TEST_QUERIES = [
    # 경기 결과/브리핑
    ("경기 결과",       "2026 호주 그랑프리 결과"),
    ("경기 결과",       "러셀 우승 메르세데스"),

    # 드라이버 인터뷰/반응
    ("드라이버 인터뷰",  "피아스트리 DNS 반응 인터뷰"),
    ("드라이버 인터뷰",  "베르스타펜 레이스 후 코멘트"),

    # 기술/전략
    ("기술 분석",       "맥라렌 파워유닛 전기 결함"),
    ("기술 분석",       "타이어 전략 소프트 하드 컴파운드"),

    # FIA 규정
    ("규정",           "포메이션 랩 사고 DNS 규정"),
    ("규정",           "언더컷 오버컷 피트스탑 전략 규정"),

    # 팀/차량
    ("팀 분석",         "페라리 2026 신규 규정 적응"),
    ("팀 분석",         "레드불 베르스타펜 전략"),
]

TOP_K = 3
SCORE_THRESHOLD = 0.3  # 낮게 잡아서 일단 결과가 나오는지 확인


# ────────────────────────────────────────
# Retriever 초기화
# ────────────────────────────────────────
def init_retriever():
    try:
        from data_pipeline.retriever import F1Retriever
        retriever = F1Retriever(collection_name="f1_knowledge_base")
        print("✅ F1Retriever 초기화 성공\n")
        return retriever
    except Exception as e:
        print(f"❌ F1Retriever 초기화 실패: {e}")
        return None


# ────────────────────────────────────────
# 단일 쿼리 테스트
# ────────────────────────────────────────
def test_single_query(retriever, category: str, query: str):
    print(f"\n{'─'*60}")
    print(f"  📂 카테고리: {category}")
    print(f"  🔍 쿼리: {query}")
    print(f"{'─'*60}")

    try:
        # retriever.search()는 payload dict 리스트 반환
        # score는 payload 안에 'score' 키로 주입됨 (retriever.py 참조)
        results = retriever.search(
            query=query,
            limit=TOP_K,
            score_threshold=SCORE_THRESHOLD
        )

        if not results:
            print("  ⚠️  결과 없음 — score_threshold 낮추거나 인덱싱 확인 필요")
            return

        for i, payload in enumerate(results, 1):
            score    = payload.get('score', 0.0)
            title    = payload.get('title', '제목 없음')
            content  = payload.get('text', '내용 없음')   # ← 'content' 아닌 'text'
            url      = payload.get('url', '')
            platform = payload.get('platform', '')
            pub_at   = payload.get('published_at', '')

            preview = content[:150].replace('\n', ' ') + '...' if len(content) > 150 else content

            # 점수 기반 품질 아이콘
            if score >= 0.7:   quality = "🟢 높음"
            elif score >= 0.5: quality = "🟡 보통"
            else:              quality = "🔴 낮음"

            print(f"\n  [{i}위] 유사도: {score:.4f}  {quality}")
            print(f"       제목   : {title}")
            if platform: print(f"       출처   : {platform}")
            if pub_at:   print(f"       작성일 : {str(pub_at)[:10]}")
            if url:      print(f"       URL    : {url[:80]}")
            print(f"       내용   : {preview}")

    except Exception as e:
        print(f"  ❌ 검색 실패: {e}")


# ────────────────────────────────────────
# 전체 테스트 실행
# ────────────────────────────────────────
def run_all_tests(retriever):
    print("\n" + "="*60)
    print("  RAG Retrieve 품질 테스트")
    print(f"  총 {len(TEST_QUERIES)}개 쿼리 / 쿼리당 상위 {TOP_K}개 문서")
    print("="*60)

    for category, query in TEST_QUERIES:
        test_single_query(retriever, category, query)

    print("\n" + "="*60)
    print("  테스트 완료")
    print("  [품질 기준] 🟢 0.7+ 우수  🟡 0.5~0.7 보통  🔴 0.5 미만 부족")
    print("="*60)


# ────────────────────────────────────────
# 커스텀 쿼리 인터랙티브 모드
# ────────────────────────────────────────
def interactive_mode(retriever):
    print("\n" + "="*60)
    print("  🎮 인터랙티브 모드 (종료: 'q' 입력)")
    print("="*60)

    while True:
        query = input("\n  질문 입력 > ").strip()
        if query.lower() in ('q', 'quit', 'exit'):
            print("  종료합니다.")
            break
        if not query:
            continue
        test_single_query(retriever, "커스텀", query)


# ────────────────────────────────────────
# 메인
# ────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PitWall-AI RAG 품질 테스트")
    parser.add_argument(
        "--mode",
        choices=["auto", "interactive"],
        default="auto",
        help="auto: 사전 정의 쿼리 전체 실행 / interactive: 직접 입력 모드"
    )
    args = parser.parse_args()

    retriever = init_retriever()
    if not retriever:
        sys.exit(1)

    if args.mode == "interactive":
        interactive_mode(retriever)
    else:
        run_all_tests(retriever)
        # 자동 테스트 후 인터랙티브 모드로 이어서 할지 선택
        cont = input("\n  인터랙티브 모드로 계속 테스트하시겠어요? (y/n) > ").strip().lower()
        if cont == 'y':
            interactive_mode(retriever)
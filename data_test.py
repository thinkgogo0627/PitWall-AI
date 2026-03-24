"""
PitWall-AI DB 상태 모니터링 스크립트
실행: python check_db_status.py

확인 항목:
1. MongoDB (f1_knowledge_base) - 컬렉션별 레코드 수
2. Qdrant (f1_knowledge_base) - 벡터 문서 수
3. SQLite (f1_data.db) - 연도별 레이스 결과 수
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

from dotenv import load_dotenv
load_dotenv()

from datetime import datetime

# ────────────────────────────────────────
# 1. MongoDB 컬렉션별 레코드 수
# ────────────────────────────────────────
def check_mongodb():
    print("\n" + "="*55)
    print("[ MongoDB Atlas ] 컬렉션별 레코드 수")
    print("="*55)

    try:
        from pymongo import MongoClient
        uri = os.getenv("MONGO_URI")
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db = client.pitwall_db

        collections = db.list_collection_names()
        if not collections:
            print("⚠️  컬렉션 없음 — 크롤링이 아직 안 된 상태입니다.")
            return

        total = 0
        for col_name in sorted(collections):
            count = db[col_name].count_documents({})
            total += count

            # 최신 문서 날짜 확인
            latest = db[col_name].find_one(
                sort=[("created_at", -1)] if "created_at" in db[col_name].find_one({} , {"created_at": 1}) or {} else []
            )
            date_str = ""
            if latest and "created_at" in latest:
                date_str = f"  (최신: {latest['created_at'].strftime('%Y-%m-%d')})"

            print(f"  📂 {col_name:<30} {count:>6,} 건{date_str}")

        print(f"  {'─'*50}")
        print(f"  {'총합':<30} {total:>6,} 건")
        client.close()

    except Exception as e:
        print(f"  ❌ MongoDB 연결 실패: {e}")


# ────────────────────────────────────────
# 2. Qdrant 벡터 DB 문서 수
# ────────────────────────────────────────
def check_qdrant():
    print("\n" + "="*55)
    print("[ Qdrant Cloud ] 컬렉션별 벡터 문서 수")
    print("="*55)

    try:
        from qdrant_client import QdrantClient
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY")

        client = QdrantClient(url=url, api_key=api_key)
        collections = client.get_collections().collections

        if not collections:
            print("  ⚠️  컬렉션 없음 — RAG 인덱싱이 안 된 상태입니다.")
            return

        total = 0
        for col in sorted(collections, key=lambda x: x.name):
            info = client.get_collection(col.name)
            count = info.points_count
            status = info.status
            total += count

            status_icon = "✅" if str(status) == "green" else "⚠️"
            print(f"  {status_icon} {col.name:<30} {count:>6,} 건  [{status}]")

        print(f"  {'─'*50}")
        print(f"  {'총합':<30} {total:>6,} 건")

    except Exception as e:
        print(f"  ❌ Qdrant 연결 실패: {e}")


# ────────────────────────────────────────
# 3. SQLite 연도별 레이스 결과 수
# ────────────────────────────────────────
def check_sqlite():
    print("\n" + "="*55)
    print("[ SQLite f1_data.db ] 연도별 레이스 결과")
    print("="*55)

    try:
        import sqlite3
        import pandas as pd

        project_root = os.path.abspath(os.path.dirname(__file__))
        db_path = os.path.join(project_root, 'data', 'f1_data.db')

        if not os.path.exists(db_path):
            print(f"  ❌ DB 파일 없음: {db_path}")
            return

        conn = sqlite3.connect(db_path)

        # 연도별 집계
        df = pd.read_sql("""
            SELECT
                Year,
                COUNT(DISTINCT RaceID) AS 레이스수,
                COUNT(*) AS 레코드수
            FROM race_results
            GROUP BY Year
            ORDER BY Year DESC
        """, conn)

        total_races = df['레이스수'].sum()
        total_records = df['레코드수'].sum()

        for _, row in df.iterrows():
            year_icon = "🆕" if row['Year'] >= 2026 else "  "
            print(f"  {year_icon} {int(row['Year'])}년   레이스: {int(row['레이스수']):>3}개   레코드: {int(row['레코드수']):>4,}건")

        print(f"  {'─'*50}")
        print(f"  {'총합':<12} 레이스: {int(total_races):>3}개   레코드: {int(total_records):>4,}건")

        # 2026 데이터 상세
        df_2026 = pd.read_sql("""
            SELECT DISTINCT RaceID FROM race_results
            WHERE Year = 2026 ORDER BY RaceID
        """, conn)

        if not df_2026.empty:
            print(f"\n  📅 2026 시즌 적재된 GP 목록:")
            for _, row in df_2026.iterrows():
                print(f"     - {row['RaceID']}")

        conn.close()

    except Exception as e:
        print(f"  ❌ SQLite 조회 실패: {e}")


# ────────────────────────────────────────
# 메인 실행
# ────────────────────────────────────────
if __name__ == "__main__":
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n🏎️  PitWall-AI DB 상태 체크  [{now}]")

    check_mongodb()
    check_qdrant()
    check_sqlite()

    print("\n" + "="*55)
    print("✅ 체크 완료")
    print("="*55)
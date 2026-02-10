from qdrant_client import QdrantClient
import os

# Qdrant 연결
client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# 기존 컬렉션 삭제 (이름이 f1_knowledge_base 라고 가정)
# rag_indexer.py 에 적힌 collection_name 과 똑같아야 합니다!
COLLECTION_NAME = "f1_knowledge_base"  # <-- 확인 필요

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)
    print(f"🗑️ 컬렉션 '{COLLECTION_NAME}' 삭제 완료! (다음 실행 시 3072차원으로 재생성됨)")
else:
    print("🤷‍♂️ 삭제할 컬렉션이 없습니다. 바로 돌리셔도 됩니다.")
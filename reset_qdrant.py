# reset_qdrant.py
import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

# .env 파일 로드 (API KEY, URL 가져오기)
load_dotenv()

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
collection_name = "f1_knowledge_base"

# 1. 존재 여부 확인
if client.collection_exists(collection_name):
    print(f"💣 Deleting existing collection: {collection_name} ...")
    # 2. 삭제 (이러면 안에 있던 데이터는 다 날아갑니다!)
    client.delete_collection(collection_name)
    print("✅ Collection Deleted Successfully!")
else:
    print(f"🤷‍♂️ Collection {collection_name} does not exist.")
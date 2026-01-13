## MongoDB에서 원본 데이터에서 데이터를 가져와서,,,

### [정제 -> 청킹 -> 임베딩 -> 벡터DB 적재 로직] 수행하는 클래스
### 차후 Airflow DAG에서 PythonOperator로 호출import re

import re
import asyncio
from typing import List
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

# [도구들]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# [도메인]
from domain.documents import F1NewsDocument


class RAGIndexer:
    def __init__(self, mongo_uri: str, qdrant_url: str):
        # 1. MongoDB 연결 준비
        self.mongo_uri = mongo_uri
        
        # 2. Qdrant 클라이언트 연결
        self.qdrant = QdrantClient(url=qdrant_url)
        self.collection_name = "f1_knowledge_base"
        
        # 3. 임베딩 모델 로드 (BAAI/bge-m3)
        # (최초 실행 시 모델 다운로드로 시간이 좀 걸립니다)
        print(" 임베딩 모델 로딩 중 (BAAI/bge-m3)...")
        self.model = SentenceTransformer('BAAI/bge-m3')
        self.vector_size = 1024 # bge-m3의 벡터 차원 수
        
        # 4. Qdrant 컬렉션 생성 (없으면 생성)
        self._init_qdrant_collection()

    def _init_qdrant_collection(self):
        """Qdrant에 벡터 저장소 공간(Collection)을 만듭니다."""
        if not self.qdrant.collection_exists(self.collection_name):
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            print(f" Qdrant 컬렉션 생성 완료: {self.collection_name}")

    def clean_text(self, text: str) -> str:
        """[Step 1] 텍스트 정제"""
        if not text: return ""
        # 1. 과도한 공백/줄바꿈 제거
        text = re.sub(r'\n+', '\n', text) 
        text = re.sub(r'\s+', ' ', text)
        # 2. "Related Articles" 같은 노이즈 제거 (필요시 패턴 추가)
        text = text.replace("Load more", "").replace("Subscribe", "")
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        """[Step 2] 텍스트 청킹 (LangChain 로직)"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,       # 한 덩어리 크기 (자)
            chunk_overlap=100,    # 문맥 유지를 위해 겹치는 구간
            separators=["\n\n", "\n", ".", " ", ""] # 자르는 우선순위
        )
        return splitter.split_text(text)

    def embed_text(self, chunks: List[str]) -> List[List[float]]:
        """[Step 3] 임베딩 (Text -> Vector)"""
        # sentence-transformers는 리스트를 한 번에 처리해줍니다 (Batch)
        embeddings = self.model.encode(chunks, normalize_embeddings=True)
        return embeddings.tolist()

    async def run_indexing(self):
        """[Step 4] 실행 파이프라인 (MongoDB -> Qdrant)"""
        print("🚀 인덱싱 작업 시작...")
        
        # 1. DB 연결
        client = AsyncIOMotorClient(self.mongo_uri)
        await init_beanie(database=client.pitwall_db, document_models=[F1NewsDocument])
        
        # 2. 아직 벡터화되지 않은 문서 가져오기
        # (지금은 테스트라 '모든' 문서를 가져옵니다. 나중엔 flag 필터링 필요)
        docs = await F1NewsDocument.find_all().to_list()
        print(f"📦 MongoDB에서 {len(docs)}개의 문서를 발견했습니다.")

        total_chunks = 0
        
        for doc in docs:
            # A. 정제
            cleaned_content = self.clean_text(doc.content)
            if len(cleaned_content) < 50: continue # 너무 짧으면 스킵

            # B. 청킹
            chunks = self.chunk_text(cleaned_content)
            if not chunks: continue

            # C. 임베딩
            vectors = self.embed_text(chunks)

            # D. Qdrant 업로드 (Batch Upload)
            points = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                # ID 생성: 문서ID_청크순번
                point_id = f"{doc.id}_{i}"
                
                # 메타데이터: 출처 확인을 위해 중요!
                payload = {
                    "source_url": doc.url,
                    "title": doc.title,
                    "platform": doc.platform,
                    "published_at": doc.published_at.isoformat() if doc.published_at else None,
                    "text": chunk  # 검색 결과로 보여줄 원본 텍스트
                }
                
                # Qdrant는 UUID 포맷의 ID를 선호하지만, 문자열 해시를 써도 됨.
                # 여기서는 편의상 UUID 생성을 위해 qdrant가 제공하는 유틸리티 사용 가능하나
                # 간단히 UUID 패키지 사용해서 고유 ID 생성 추천. 
                import uuid
                # 고유 ID 생성 (Deterministic하게 만들면 중복 방지에 좋음)
                point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))

                points.append(PointStruct(id=point_uuid, vector=vector, payload=payload))

            # Qdrant에 저장
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points
            )
            total_chunks += len(chunks)
            print(f" -> 문서 '{doc.title[:20]}...' 처리 완료 ({len(chunks)} Chunks)")

        print(f" 인덱싱 완료! 총 {total_chunks}개의 청크가 Qdrant에 적재되었습니다.")

# --- 실행부 (테스트용) ---
if __name__ == "__main__":
    # 로컬 설정
    indexer = RAGIndexer(
        mongo_uri="mongodb://admin:password123@localhost:27017",
        qdrant_url="http://localhost:6333"
    )
    asyncio.run(indexer.run_indexing())
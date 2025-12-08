## 소프트 데이터 소스에서 크롤링하여 Dataframe 형태로 만든 데이터 전원 적재
## 데이터프레임 -> 벡터 임베딩 수행


import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()


# 전역 설정 (한 번만 로드)
## 임베딩 모델 지정
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3", # 한/영 통합 모델
    device="cuda" # 없으면 cpu
)


class F1VectorStore:
    def __init__(self, db_path="./f1_chroma_db", collection_name="f1_knowledge"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

    def check_exists(self, url):
        """URL을 기준으로 이미 DB에 있는지 확인"""
        # ChromaDB에서 메타데이터로 조회
        results = self.collection.get(
            where={"url": url}
        )
        return len(results['ids']) > 0

    def add_data(self, df: pd.DataFrame):
        """데이터프레임을 받아서 벡터화 후 저장"""
        if df.empty:
            print("저장할 데이터가 없습니다.")
            return

        documents = []
        new_count = 0
        
        print(f"데이터 저장 프로세스 시작 ({len(df)}건)...")

        for _, row in df.iterrows():
            url = row['link']
            
            # 중복 체크 (이미 있으면 패스)
            if self.check_exists(url):
                print(f"  스킵 (이미 존재): {row['title'][:20]}...")
                continue

            # 문서 객체 생성
            doc = Document(
                text=row['context'],
                metadata={
                    "title": row['title'],
                    "source": row['source'],
                    "url": url,
                    "crawled_at": pd.Timestamp.now().isoformat()
                }
            )
            documents.append(doc)
            new_count += 1

        if documents:
            # 인덱싱 및 저장: 임베딩 -> 인덱싱 -> 저장
            VectorStoreIndex.from_documents(
                documents, 
                storage_context=self.storage_context,
                show_progress=True
            )
            print(f" {new_count}건 신규 저장 완료!")
        else:
            print("모든 데이터가 최신입니다. (저장할 것 없음)")
## MongoDB에 저장될 문서의 형태 (Schema) 정의
## Beanie ODM 사용


from typing import Optional, Dict
from datetime import datetime
from beanie import Document, Indexed # Beanie가 설치되어 있어야 합니다 (pip install beanie)

class F1NewsDocument(Document):
    """
    [F1 뉴스 표준 문서]
    모든 뉴스 크롤러는 이 형태로 데이터를 뱉어내야 합니다.
    """
    title: str
    content: str              # 본문 전체 텍스트
    url: Indexed(str, unique= True)
    platform: str             # 예: 'Autosport', 'Motorsport.com'
    author: Optional[str] = None
    published_at: Optional[datetime] = None
    
    # RAG 파이프라인용 플래그
    is_embedded: bool = False # 임베딩 여부
    embedding_id: Optional[str] = None

    class Settings:
        name = "f1_news_articles" # MongoDB 컬렉션 이름
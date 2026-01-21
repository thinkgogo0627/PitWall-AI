import io
import re
import requests
import traceback
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from beanie.operators import Set
from datetime import datetime
from pypdf import PdfReader
from domain.documents import F1NewsDocument

class FIARegulationCrawler:
    def __init__(self):
        # 2026년 규정 (Issue 8, Issue 2 등 최신 버전 반영)
        self.urls = {
            "2026_technical": "https://www.fia.com/sites/default/files/fia_2026_formula_1_technical_regulations_issue_8_-_2024-06-24.pdf",
            "2026_sporting": "https://api.fia.com/sites/default/files/fia_2026_f1_regulations_-_section_b_sporting_-_iss02_-_2024-12-11.pdf",
            # 필요한 경우 2025년도 추가 가능
            "2025_sporting": "https://www.fia.com/sites/default/files/fia_2025_formula_1_sporting_regulations_-_issue_1_-_2024-07-31.pdf",
            "2025_technical": "https://www.fia.com/sites/default/files/fia_2025_formula_1_technical_regulations_-_issue_01_-_2024-12-11_1.pdf"
        }

    def _clean_text(self, text: str) -> str:
        """
        PDF 노이즈 제거: 
        1. 페이지 번호 (Page X of Y)
        2. 상단 헤더 (Issue X, Date 등)
        3. 불필요한 공백
        """
        if not text: return ""

        # 1. 헤더/푸터 패턴 제거 (예: "Issue 8", "24 June 2024", "Page 1/100")
        # (단순한 패턴 매칭이므로 완벽하진 않지만 검색 품질을 높임)
        text = re.sub(r'Page \d+/\d+', '', text) 
        text = re.sub(r'\d{1,2} [A-Za-z]+ \d{4}', '', text) # 날짜 형식 제거
        text = re.sub(r'Issue \d+', '', text)

        # 2. 다중 공백 및 줄바꿈 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _extract_articles(self, text: str, page_num: int, doc_type: str) -> list:
        """
        (Advanced) 페이지 내에서 'Article X.Y' 패턴을 찾아서 
        단순 페이지 번호보다 더 의미 있는 제목을 생성하려 노력함.
        """
        # 정규식으로 'ARTICLE 1.2' 같은 패턴 찾기
        article_matches = re.findall(r'ARTICLE\s+(\d+(\.\d+)*)', text, re.IGNORECASE)
        
        chunks = []
        
        # 만약 페이지 안에 명확한 Article 시작점이 있다면 제목에 포함
        if article_matches:
            # 가장 처음 발견된 Article 번호를 대표 제목으로 사용
            main_article = article_matches[0][0]
            title = f"FIA {doc_type} Regulations - Art {main_article} (Page {page_num})"
        else:
            title = f"FIA {doc_type} Regulations - Page {page_num}"

        # 데이터 스키마 맞추기 (F1NewsDocument 재활용)
        # 규정집은 '뉴스'는 아니지만, RAG에서는 같은 문서로 취급하는 게 관리하기 편함
        chunks.append({
            "title": title,
            "content": text,
            "page_no": page_num
        })
        
        return chunks

    def crawl(self, doc_key: str) -> list:
        """
        실제 크롤링 및 파싱 실행
        doc_key: '2026_technical', '2026_sporting' 등
        """
        target_url = self.urls.get(doc_key)
        if not target_url:
            print(f" 지원하지 않는 문서 키입니다: {doc_key}")
            return []

        print(f"📥 FIA 규정집 다운로드 중... [{doc_key}]")
        
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(target_url, headers=headers, timeout=60)
            
            if response.status_code != 200:
                print(f" 다운로드 실패: Status {response.status_code}")
                return []
            
            # PDF 메모리 로드
            f = io.BytesIO(response.content)
            reader = PdfReader(f)
            
            total_pages = len(reader.pages)
            print(f"📖 파싱 시작 (총 {total_pages} 페이지)")
            
            extracted_docs = []
            
            for i, page in enumerate(reader.pages):
                raw_text = page.extract_text()
                cleaned_text = self._clean_text(raw_text)
                
                # 너무 짧은 페이지(목차, 빈 페이지) 스킵
                if len(cleaned_text) < 50:
                    continue
                
                # 페이지별로 문서 생성
                chunks = self._extract_articles(cleaned_text, i + 1, doc_key)
                
                for chunk in chunks:
                    # F1NewsDocument 스키마에 매핑
                    doc = F1NewsDocument(
                        title=chunk['title'],
                        content=chunk['content'],
                        url=f"{target_url}#page={chunk['page_no']}", # PDF 페이지 앵커 추가
                        platform="FIA Official PDF",
                        author="FIA",
                        published_at=datetime.now(),
                        is_embedded=False
                    )
                    extracted_docs.append(doc)
            
            print(f" 변환 완료: {len(extracted_docs)}개의 청크 확보")
            return extracted_docs

        except Exception as e:
            print(f" FIA 크롤링 중 치명적 오류: {e}")
            print(traceback.format_exc())
            return []

# ... (위쪽 import 및 클래스 정의는 그대로) ...
## 적재
if __name__ == "__main__":
    import asyncio
    from motor.motor_asyncio import AsyncIOMotorClient
    from beanie import init_beanie
    
    # 📌 로컬 실행용 메인 함수
    async def main():
        print(" [Manual Run] MongoDB 연결 중...")
        
        # 1. DB 연결 (로컬 환경에 맞게 주소 조정)
        # docker-compose로 띄운 몽고DB가 27017포트로 열려있다고 가정
        client = AsyncIOMotorClient("mongodb://localhost:27017")
        
        # 2. Beanie 초기화
        await init_beanie(database=client.pitwall_db, document_models=[F1NewsDocument])
        
        crawler = FIARegulationCrawler()
        
        # 수집할 타겟 정의 (2026 테크니컬 & 스포팅)
        targets = ["2026_technical", "2026_sporting"]
        
        print("\n" + "="*50)
        print(" FIA 규정집 수동 업데이트 시작")
        print("="*50)

        total_saved = 0

        for doc_type in targets:
            print(f"\n 수집 시작: {doc_type}")
            docs = crawler.crawl(doc_type)
            
            if not docs:
                print(" 수집된 문서가 없습니다.")
                continue

            print(f" DB 저장 중... ({len(docs)} pages)")
            saved_count = 0
            updated_count = 0
            
            for doc in docs:
                #  [Upsert Logic] 
                # 1. 기존 문서가 있는지 확인 (카운팅 목적)
                existing_doc = await F1NewsDocument.find_one(F1NewsDocument.url == doc.url)
                
                # 2. Upsert 수행 (핵심!)
                # URL이 같으면 -> Set 내부의 필드만 갱신
                # URL이 다르면 -> on_insert에 있는 doc 전체를 저장
                await F1NewsDocument.find_one(F1NewsDocument.url == doc.url).upsert(
                    Set({
                        F1NewsDocument.title: doc.title,
                        F1NewsDocument.content: doc.content,
                        F1NewsDocument.published_at: datetime.now(), # 갱신 시각 업데이트
                        F1NewsDocument.is_embedded: False # 🌟 중요: 내용이 바뀌었으니 다시 임베딩 대상이 됨
                    }),
                    on_insert=doc
                )
                
                if existing_doc:
                    updated_count += 1
                else:
                    saved_count += 1
            
            print(f" {doc_type} 완료: 신규 {saved_count}건 / 갱신 {updated_count}건")
            total_saved += saved_count

        print("\n" + "="*50)
        print(f" 작업 종료. 총 {total_saved} 페이지가 DB에 추가되었습니다.")
        

    # 실행
    asyncio.run(main())
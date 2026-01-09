# data_pipeline/crawlers/f1_news.py

import traceback
import trafilatura # [NEW] 전용 도구 임포트
from bs4 import BeautifulSoup # (제목 추출용으로 남겨둠)
from datetime import datetime

from .base import BaseSeleniumCrawler
from domain.documents import F1NewsDocument

class AutosportCrawler(BaseSeleniumCrawler):
    
    def set_extra_driver_options(self, options) -> None:
        # ... (기존 스텔스/Eager 설정 유지) ...
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        options.page_load_strategy = 'eager'

    def extract(self, link: str, **kwargs) -> dict:
        print(f" Autosport 진입 중: {link}")
        
        try:
            self.driver.get(link)
            # Eager 모드라 금방 리턴되지만, trafilatura를 위해 HTML이 좀 더 차오를 시간을 1초 정도 줌
            import time
            time.sleep(1) 
            
            html_source = self.driver.page_source

            # [Step 1] 제목 추출 (이건 BS4가 빠름)
            soup = BeautifulSoup(html_source, "html.parser")
            title_tag = soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else "No Title"

            # [Step 2] 본문 추출 (Trafilatura 엔진 사용) 🚀
            # include_comments=False: 댓글 제거
            # include_tables=False: 표 데이터 제거 (필요하면 True)
            # no_fallback=True: 정확도 우선 (쓰레기 긁느니 안 긁겠다)
            body_content = trafilatura.extract(
                html_source, 
                include_comments=False, 
                include_tables=False,
                no_fallback=False 
            )

            if not body_content:
                # Trafilatura가 실패하면 간단한 백업 (메타 태그 등)
                description = soup.find("meta", attrs={"name": "description"})
                body_content = description["content"] if description else "본문 추출 실패"
                print(f" Trafilatura 추출 실패 -> Meta Description으로 대체")

            # [Step 3] 작성자 추출 (기존 유지)
            author_tag = soup.find("a", class_="ms-item_author")
            author = author_tag.get_text(strip=True) if author_tag else "Unknown"

            news_doc = F1NewsDocument(
                title=title,
                content=body_content,
                url=link,
                platform="Autosport",
                author=author,
                published_at=datetime.now(),
                is_embedded=False
            )
            
            print(f" 추출 완료: {title} ({len(body_content)}자)")
            return news_doc.dict()

        except Exception as e:
            print(f" Autosport 크롤링 실패")
            print(traceback.format_exc())
            return {}
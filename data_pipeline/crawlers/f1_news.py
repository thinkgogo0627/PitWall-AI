# data_pipeline/crawlers/f1_news.py

import traceback
import trafilatura
import time
from bs4 import BeautifulSoup
from datetime import datetime

# [NEW] 목록 수집을 위한 Selenium 필수 도구들 추가
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .base import BaseSeleniumCrawler
from domain.documents import F1NewsDocument

class AutosportCrawler(BaseSeleniumCrawler):
    
    def set_extra_driver_options(self, options) -> None:
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        options.page_load_strategy = 'eager'

    # -------------------------------------------------------------------------
    # [NEW] 목록 페이지 순회 및 링크 수집 (Autosport 전용)
    # -------------------------------------------------------------------------
    def crawl_listing_page(self, list_url: str, max_clicks: int = 20) -> list:
        print(f"📡 [Autosport] 목록 수집 시작: {list_url}")
        links = []
        
        try:
            self.driver.get(list_url)
            time.sleep(3) # 첫 진입 대기

            # 1. 쿠키 배너 처리 (유럽 사이트라 필수)
            try:
                # Autosport는 보통 'Accept All' 혹은 'Agree' 버튼이 있음
                # ID나 Class가 자주 바뀌므로 텍스트 기반으로 찾음
                cookie_xpath = "//button[contains(., 'Accept')] | //button[contains(., 'Agree')]"
                cookie_btn = self.driver.find_element(By.XPATH, cookie_xpath)
                cookie_btn.click()
                print("🍪 쿠키 배너 제거됨")
                time.sleep(1)
            except:
                pass # 배너가 없거나 실패해도 일단 진행

            # 2. Load More 버튼 클릭 루프
            click_count = 0
            while click_count < max_clicks:
                try:
                    # 스크롤을 내려서 버튼을 노출시킴
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(1)
                    
                    # Autosport의 'Load more' 버튼 찾기
                    # (대소문자 구분을 피하기 위해 translate 사용하거나 여러 조건 검색)
                    btn_xpath = "//a[contains(translate(., 'L', 'l'), 'load more')] | //button[contains(translate(., 'L', 'l'), 'load more')]"
                    
                    load_more_btn = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, btn_xpath))
                    )
                    
                    self.driver.execute_script("arguments[0].click();", load_more_btn)
                    print(f"🖱️ Load More 클릭 성공 ({click_count+1}/{max_clicks})")
                    
                    time.sleep(3) # 로딩 대기
                    click_count += 1
                except Exception:
                    print("🛑 더 이상 Load More 버튼이 없습니다. (Loop 종료)")
                    break
            
            # 3. HTML 파싱 및 링크 추출
            print("📥 링크 추출 중...")
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            
            # Autosport 뉴스 링크 패턴: 보통 /f1/news/... 형식을 따름
            all_links = soup.find_all("a", href=True)
            
            for a in all_links:
                href = a['href']
                # 조건: '/f1/news/' 가 포함되어 있고, 댓글 앵커(#)나 비디오 제외
                if "/f1/news/" in href and "#" not in href:
                    # 상대 경로 처리 (/f1/news/...)
                    if href.startswith("/"):
                        full_link = "https://www.autosport.com" + href
                    else:
                        full_link = href
                        
                    links.append(full_link)

            links = list(set(links)) # 중복 제거
            print(f"📦 [Autosport] 총 {len(links)}개의 링크 확보 완료!")
            
        except Exception as e:
            print(f"❌ 목록 수집 실패: {e}")
            print(traceback.format_exc())
            
        return links

    # -------------------------------------------------------------------------
    # [EXISTING] 개별 기사 추출
    # -------------------------------------------------------------------------
    def extract(self, link: str, **kwargs) -> dict:
        print(f"🏎️ Autosport 진입 중: {link}")
        
        try:
            self.driver.get(link)
            time.sleep(1) 
            
            html_source = self.driver.page_source

            # [Step 1] 제목 추출
            soup = BeautifulSoup(html_source, "html.parser")
            title_tag = soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else "No Title"

            # [Step 2] 본문 추출
            body_content = trafilatura.extract(
                html_source, 
                include_comments=False, 
                include_tables=False,
                no_fallback=False 
            )

            if not body_content:
                description = soup.find("meta", attrs={"name": "description"})
                body_content = description["content"] if description else "본문 추출 실패"
                print(f"⚠️ Trafilatura 추출 실패 -> Meta Description으로 대체")

            # [Step 3] 작성자 추출
            author_tag = soup.find("a", class_="ms-item_author")
            author = author_tag.get_text(strip=True) if author_tag else "Autosport Staff"

            news_doc = F1NewsDocument(
                title=title,
                content=body_content,
                url=link,
                platform="Autosport",
                author=author,
                published_at=datetime.now(),
                is_embedded=False
            )
            
            print(f"✅ 추출 완료: {title} ({len(body_content)}자)")
            return news_doc.dict()

        except Exception as e:
            print(f"❌ Autosport 크롤링 실패: {link}")
            print(traceback.format_exc())
            return {}
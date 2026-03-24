import traceback
import trafilatura
from bs4 import BeautifulSoup
from datetime import datetime
import time

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .base import BaseSeleniumCrawler
from domain.documents import F1NewsDocument

class Formula1Crawler(BaseSeleniumCrawler):
    
    def set_extra_driver_options(self, options) -> None:
        # [스텔스 모드] F1 공식 홈페이지도 봇 탐지가 있으므로 위장술 적용
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        # [중요] Eager 전략: 이미지가 다 뜨기 전에 텍스트만 낚아챔 (속도 향상)
        options.page_load_strategy = 'eager'

        # 유저 에이전트: 일반 윈도우 크롬인 척
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
    def crawl_listing_page(self, list_url: str, max_clicks: int = 20) -> list:
        print(f" 목록 수집 시작 (Max Clicks: {max_clicks}): {list_url}")
        links = []
        
        try:
            self.driver.get(list_url)
            time.sleep(5)

            # 1. 쿠키 배너 처리 (가끔 버튼을 가림)
            try:
                cookie_btn = self.driver.find_element(By.ID, "truste-consent-button")
                cookie_btn.click()
                print(" 쿠키 배너 제거됨")
                time.sleep(1)
            except:
                pass

            # 2. Load More 버튼 연타 루프
            click_count = 0
            while click_count < max_clicks:
                try:
                    # 스크롤 최하단 이동
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    time.sleep(2)

                    # 버튼 찾기 (More News, Load more 등 텍스트가 다를 수 있어 유연하게)
                    # XPath: 'Load more'라는 텍스트를 포함한 버튼이나 a태그 검색
                    btn_xpath = "//button[contains(., 'Load more')] | //a[contains(., 'Load more')]"
                    
                    load_more_btn = WebDriverWait(self.driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, btn_xpath))
                    )
                    
                    # JS로 강제 클릭 (화면 가림 방지)
                    self.driver.execute_script("arguments[0].click();", load_more_btn)
                    print(f"🖱️ Load More 클릭 성공 ({click_count+1}/{max_clicks})")
                    
                    time.sleep(3) # 새 항목 로딩 대기
                    click_count += 1
                
                except Exception:
                    print(" 더 이상 Load More 버튼이 없음 (또는 끝 도달)")
                    break

            # 3. 전체 HTML 파싱 및 링크 추출
            print(" 링크 추출 중...")
            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            # [디버깅용] a 태그가 아예 안 잡히는지 확인
            all_links = soup.find_all("a", href=True)
            print(f"DEBUG: 페이지에서 발견된 전체 링크 수: {len(all_links)}개")
            
            for a in all_links:
                href = a['href']
                # /article/ 패턴이 있고, .html로 끝나는 것만 (동영상 등 제외)
                if "/article/" in href and "video" not in href:
                    full_link = "https://www.formula1.com" + href if href.startswith("/") else href
                    links.append(full_link)
                    # 디버깅 -> 처음 3개만 어떤 모양인지 체크
                    if len(links) <= 3:
                        print(f"    -> 확보된 후보: {full_link}")

            links = list(set(links)) # 중복 제거
            print(f" 총 {len(links)}개의 기사 링크 확보!")
            
        except Exception as e:
            print(f" 목록 수집 실패: {e}")
            print(traceback.format_exc())
            
        return links
      

    # one 기사에서 schema에 맞춰 데이터 뽑아오는 메서드
    def extract(self, link: str, **kwargs) -> dict:
        print(f"🏎️ F1.com 진입 중: {link}")
        
        try:
            self.driver.get(link)
            # Eager 모드라 뼈대만 로딩되므로, 본문이 렌더링될 짬을 살짝 줌 (1~2초)
            time.sleep(3) 
            
            html_source = self.driver.page_source

            # [Step 1] 제목 추출 (BeautifulSoup)
            soup = BeautifulSoup(html_source, "html.parser")
            
            # F1.com은 보통 h1 태그에 'f1-header__title' 같은 클래스가 붙지만, 
            # 범용성을 위해 h1 우선 검색
            title_tag = soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else "No Title"

            # [Step 2] 본문 추출 (Trafilatura 엔진) 
            # 공식 홈은 'Video'나 'Related' 위젯이 많아서 Trafilatura가 아주 효과적임
            body_content = trafilatura.extract(
                html_source, 
                include_comments=False, 
                include_tables=False, 
                no_fallback=False
            )

            if not body_content:
                print(f"⚠️ Trafilatura 추출 실패 -> HTML 구조 확인 필요")
                body_content = "본문 추출 실패"

            # [Step 3] 작성자 (선택 사항)
            # F1.com은 작성자가 없거나 'Formula 1'으로 표기되는 경우가 많음
            author = "Formula 1 Official" 

            # [Step 4] 문서 생성
            news_doc = F1NewsDocument(
                title=title,
                content=body_content,
                url=link,
                platform="Formula1.com", # 플랫폼 명시
                author=author,
                published_at=datetime.now(),
                is_embedded=False
            )
            
            print(f" 추출 완료: {title} ({len(body_content)}자)")
            return news_doc.dict()

        except Exception as e:
            print(f" F1.com 크롤링 실패")
            print(traceback.format_exc())
            return {}
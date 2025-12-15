from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import random

# 전역 상수
BASE_URL = "https://www.formula1.com/en/latest/tags/analysis.3HkjTN75peeCOsSegCyOWi"

def setup_driver():
    """WSL 환경에 최적화된 크롬 드라이버 설정"""
    chrome_options = Options()
    chrome_options.add_argument("--headless") # WSL 필수
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("window-size=1920,1080")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def get_article_links(driver, limit=5):
    """분석(Analysis) 섹션에서 기사 URL 수집"""
    print(f" F1 공홈 진입: {BASE_URL}")
    driver.get(BASE_URL)
    time.sleep(3) # 로딩 대기

    # 쿠키 팝업 닫기 시도
    try:
        cookie_btn = driver.find_element(By.ID, "sp-cc-accept")
        cookie_btn.click()
    except:
        pass

    links = driver.find_elements(By.TAG_NAME, "a")
    target_links = []
    seen_urls = set()

    print(f" 링크 스캔 중...")
    
    for link in links:
        try:
            href = link.get_attribute('href')
            title = link.text.strip()
            
            # 필터링 조건
            if href and '/en/latest/article' in href and title:
                # 영상이나 팟캐스트 제외
                if "Video" not in title and "Podcast" not in title:
                    if href not in seen_urls:
                        seen_urls.add(href)
                        target_links.append({"href": href, "title": title})
                        print(f"   [Target] {title[:40]}...")
                        
            if len(target_links) >= limit:
                break
        except:
            continue
            
    return target_links

def extract_content(driver, url):
    """상세 페이지 본문 추출"""
    try:
        driver.get(url)
        time.sleep(random.uniform(1.5, 3.0))
        
        paragraphs = driver.find_elements(By.TAG_NAME, "p")
        content = []
        
        for p in paragraphs:
            text = p.text.strip()
            # 노이즈 제거
            if len(text) > 40 and "cookie" not in text.lower() and "©" not in text:
                content.append(text)
        
        return " ".join(content)
    except Exception as e:
        print(f"     본문 추출 에러: {e}")
        return ""

def crawl(limit=5):
    """메인 크롤링 함수 (외부 호출용)"""
    driver = None
    articles = []
    
    try:
        driver = setup_driver()
        print(" F1 공식 리포트 수집 시작...")
        
        # 1. 링크 수집
        targets = get_article_links(driver, limit)
        print(f"총 {len(targets)}개의 리포트 수집을 시작합니다.")
        
        # 2. 본문 순회
        for item in targets:
            print(f"   접속: {item['title'][:30]}...")
            text = extract_content(driver, item['href'])
            
            if len(text) > 500:
                articles.append({
                    "title": item['title'],
                    "link": item['href'],
                    "context": text,
                    "source": "F1 Official Analysis"
                })
                print(f"     성공 ({len(text)}자)")
            else:
                print("     실패 (내용 부족)")
                
    except Exception as e:
        print(f" 크롤링 중 치명적 오류: {e}")
        
    finally:
        if driver:
            driver.quit()
            print(" 드라이버 종료 완료.")
            
    return pd.DataFrame(articles)
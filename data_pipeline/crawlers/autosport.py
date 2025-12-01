import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

# --- 1. 공통 헤더 (스텔스 모드) ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Referer': 'https://www.google.com/'
}

def extract_autosport_content(url):
    """
    [수정버전] 클래스 이름 의존성을 제거하고, 
    모든 p태그를 긁어서 기사 같은 것만 남기는 방식
    """
    try:
        # 매너 딜레이
        time.sleep(random.uniform(1.0, 2.0))
        
        res = requests.get(url, headers=HEADERS, timeout=10)
        if res.status_code != 200:
            print(f"  └ 접속 실패 (Code: {res.status_code})")
            return ""
            
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # 1. 봇 차단 여부 확인 (혹시 캡차가 떴는지 확인)
        page_title = soup.title.get_text(strip=True) if soup.title else "No Title"
        if "Bot" in page_title or "Access Denied" in page_title:
             print(f"  └ 봇 탐지됨! (Page Title: {page_title})")
             return ""

        # 2. [전략 변경] 특정 Div 찾지 말고, 그냥 모든 p 태그 가져오기
        # Autosport 기사는 보통 article 태그나 main 태그 안에 있습니다.
        # 범위를 좁히기 위해 article 태그 먼저 시도, 없으면 전체에서 찾기
        container = soup.find('article')
        if not container:
            container = soup # 없으면 전체 HTML에서 찾음
            
        paragraphs = container.find_all('p')
        
        clean_text = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            
            # 3. [노이즈 필터링]
            # - 너무 짧은 문장 (메뉴, 링크 등) 제외 (50자 이상만)
            # - 'Read Also', 'Photo by' 같은 거 제외
            if len(text) > 50 and "Read Also:" not in text and "Photo by" not in text:
                clean_text.append(text)
        
        # 수집된 문장이 3개 미만이면 실패로 간주 (너무 적음)
        if len(clean_text) < 3:
            print(f"  └ 텍스트 추출 실패 (유효 문장 부족, HTML 구조 확인 필요)")
            # 디버깅용: 도대체 뭘 가져왔는지 확인
            # print(f"    (Debug: Page Title - {page_title})") 
            return ""

        return " ".join(clean_text)

    except Exception as e:
        print(f"  └ 에러 발생: {e}")
        return ""

def crawl_autosport_full():
    """
    목록 수집 + 본문 수집을 합친 메인 함수
    """
    base_url = "https://www.autosport.com/f1/news/"
    print(f"Autosport 접근 중: {base_url}")
    
    res = requests.get(base_url, headers=HEADERS)
    soup = BeautifulSoup(res.content, 'html.parser')
    
    links = soup.select("a")
    articles = []
    unique_links = set()
    
    print("기사 리스트 분석 및 본문 수집 시작...")
    
    count = 0
    for row in links:
        title = row.get_text(strip=True)
        href = row.get('href', '')
        
        # 필터링: F1 뉴스 링크이고 제목이 좀 긴 것
        if '/f1/news/' in href and len(title) > 20:
            full_link = "https://www.autosport.com" + href if not href.startswith('http') else href
            
            if full_link not in unique_links:
                unique_links.add(full_link)
                print(f"[Target] {title[:40]}...")
                
                # ★ 여기서 본문 추출 함수 호출!
                content = extract_autosport_content(full_link)
                
                if content:
                    articles.append({
                        "title": title,
                        "link": full_link,
                        "context": content, # 이게 우리가 원하던 것!
                        "source": "Autosport"
                    })
                    print(f"  └ 본문 수집 완료 ({len(content)}자)")
                    count += 1
                else:
                    print("  └ 본문 수집 실패 (Skipped)")
                
                if count >= 5: # 테스트니까 5개만 합시다
                    break
    
    return pd.DataFrame(articles)
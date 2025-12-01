import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def crawl_gpkorea_final():
    # 1. URL: 모터스포츠 종합 섹션 (S1N2)
    url = "http://www.gpkorea.com/news/articleList.html?sc_section_code=S1N2&view_type=sm"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # 2. 요청 및 인코딩 설정
    res = requests.get(url, headers=headers)
    res.encoding = 'utf-8'
    soup = BeautifulSoup(res.content, 'html.parser')
    
    articles = []
    
    # 3. F1 키워드 (필터망)
    # 화면에 보이는 '맥라렌', '노리스', '페르스타펜' 등이 다 포함되어 있습니다.
    f1_keywords = ['F1', '포뮬러1', '포뮬러 1', '그랑프리', 'GP', 
                   '베르스타펜', '해밀턴', '르클레르', '페라리', '레드불', '메르세데스', '맥라렌', '노리스']

    # 4. ★ 핵심 수정: 디버그에서 성공한 'h4 a' 선택자 사용
    rows = soup.select("h4 a") 

    print(f"전체 기사 {len(rows)}개 중 F1 기사를 선별합니다...")

    for row in rows:
        title = row.get_text(strip=True)
        link = "http://www.gpkorea.com" + row['href']
        
        # 5. 필터링 로직
        if any(keyword in title for keyword in f1_keywords):
            print(f"[수집 완료] {title}")
            
            try:
                # 상세 페이지 본문 수집
                sub_res = requests.get(link, headers=headers)
                sub_res.encoding = 'utf-8'
                sub_soup = BeautifulSoup(sub_res.content, 'html.parser')
                
                # 본문 ID
                content_div = sub_soup.select_one("#article-view-content-div")
                
                if content_div:
                    content = content_div.get_text(strip=True)
                    
                    # 노이즈 제거
                    if "Copyright" in content:
                        content = content.split("Copyright")[0]
                    
                    articles.append({
                        "title": title,
                        "link": link,
                        "context": content
                    })
                    time.sleep(0.1) # 매너 딜레이
                
            except Exception as e:
                print(f"  └ 에러: {e}")
                continue
        else:
            # F1 기사가 아니면 패스 (로그 확인용)
            # print(f"  [패스] {title}") 
            pass
            
    return pd.DataFrame(articles)

# 실행
'''
df = crawl_gpkorea_final()
print(f"최종 수집 결과: {len(df)}개의 F1 기사 확보!")
if not df.empty:
    print(df[['title']].head())
'''
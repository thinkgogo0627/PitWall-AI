import requests
import io
from pypdf import PdfReader
import pandas as pd
import re
import time

def clean_text(text):
    """PDF 텍스트 정제: 불필요한 공백, 헤더, 푸터 제거"""
    if not text: return ""
    # 연속된 공백/줄바꿈 하나로 통일
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def crawl(doc_type="sporting", year=2024):
    """
    FIA 규정집 크롤링 메인 함수
    doc_type: 'sporting' (경기 규정) or 'technical' (기술 규정)
    """
    # URL 매핑 (나중에 2025년 등으로 확장 가능)
    urls = {
        "sporting": "https://www.fia.com/sites/default/files/fia_2025_formula_1_sporting_regulations_-_issue_1_-_2024-07-31.pdf",
        "technical": "https://www.fia.com/sites/default/files/fia_2025_formula_1_technical_regulations_-_issue_01_-_2024-12-11_1.pdf" 
    }
    
    target_url = urls.get(doc_type)
    if not target_url:
        print(f"지원하지 않는 문서 타입입니다: {doc_type}")
        return pd.DataFrame()

    print(f"FIA 규정집({doc_type}) 다운로드 중...")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(target_url, headers=headers, timeout=30)
        
        if response.status_code != 200:
            print(f"다운로드 실패: {response.status_code}")
            return pd.DataFrame()
            
        # PDF 메모리 로드
        f = io.BytesIO(response.content)
        reader = PdfReader(f)
        
        articles = []
        doc_title = f"{year} F1 {doc_type.capitalize()} Regulations"
        
        print(f" 파싱 시작 (총 {len(reader.pages)} 페이지)")
        
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            cleaned_text = clean_text(text)
            
            # 너무 짧은 페이지(목차, 빈 페이지 등) 스킵
            if len(cleaned_text) < 100:
                continue
                
            articles.append({
                "title": f"{doc_title} - Page {i+1}",
                "link": target_url,
                "context": cleaned_text,
                "source": "FIA Official PDF",
                "doc_type": doc_type,
                "page_no": i+1
            })
            
        print(f"변환 완료: {len(articles)}개의 페이지 데이터 확보")
        return pd.DataFrame(articles)

    except Exception as e:
        print(f"FIA 크롤링 중 치명적 오류: {e}")
        return pd.DataFrame()
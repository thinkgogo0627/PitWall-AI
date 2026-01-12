import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

# 도메인 모델 & 크롤러 임포트
from domain.documents import F1NewsDocument
from data_pipeline.crawlers.f1_tactic import Formula1Crawler


async def test_crawler_logic():
    print("🔌 MongoDB 시동(Fuel Injection) 중...")
    
    # ---------------------------------------------------------
    # 1. 필수: DB 연결 (이게 있어야 Document를 만들 수 있음!)
    # ---------------------------------------------------------
    mongo_uri = "mongodb://admin:password123@localhost:27017"
    try:
        client = AsyncIOMotorClient(mongo_uri)
        # pitwall_db에 F1NewsDocument 등록
        await init_beanie(database=client.pitwall_db, document_models=[F1NewsDocument])
        print("✅ DB 연결 성공! (Ready to Race)")
    except Exception as e:
        print(f"❌ DB 연결 실패: {e}")
        return

    # ---------------------------------------------------------
    # 2. 크롤러 테스트 시작
    # ---------------------------------------------------------
    crawler = Formula1Crawler()
    
    # 기사 목록 페이지 (Tactic/Analysis 태그)
    target_list_url = "https://www.formula1.com/en/latest/tags/analysis.3HkjTN75peeCOsSegCyOWi"
    
    # (1) 목록 수집 테스트
    print(f"\n🚀 [TEST] 목록 수집 시작 (타겟: {target_list_url})")
    # 테스트니까 1~2번만 클릭해서 빠르게 확인
    article_links = crawler.crawl_listing_page(target_list_url, max_clicks=2)
    
    print(f"📦 총 {len(article_links)}개의 링크 확보!")
    if not article_links:
        print("❌ 링크 수집 실패")
        return

    # (2) 개별 기사 수집 테스트 (첫 번째 링크로)
    target_article = article_links[0]
    print(f"\n🚀 [TEST] 개별 기사 상세 수집: {target_article}")
    
    # 이제 DB가 연결되어 있으니 여기서 에러가 안 남!
    result = crawler.extract(target_article)
    
    if result and result.get('title'):
        print(f"✅ 테스트 성공!")
        print(f" - 제목: {result['title']}")
        print(f" - 본문 길이: {len(result['content'])}자")
        print(f" - 플랫폼: {result['platform']}")
    else:
        print("❌ 실패: 내용을 가져오지 못함")

    # 브라우저 종료
    crawler.driver.quit()

if __name__ == "__main__":
    asyncio.run(test_crawler_logic())
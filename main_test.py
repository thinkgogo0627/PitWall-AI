# main_test.py

import asyncio
import json
import traceback
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

# 도메인 모델과 디스패처 임포트
from domain.documents import F1NewsDocument
from data_pipeline.crawlers.dispatcher import CrawlerDispatcher

async def main():
    # ---------------------------------------------------------
    # 1. MongoDB 연결 및 Beanie 초기화 (이게 빠져서 에러 난 것!)
    # ---------------------------------------------------------
    # .env에 있는 비번을 쓰거나, 테스트용으로 직접 입력
    mongo_uri = "mongodb://admin:password123@localhost:27017"
    client = AsyncIOMotorClient(mongo_uri)
    
    # 'pitwall_db'라는 이름의 DB를 쓰겠다고 선언
    await init_beanie(database=client.pitwall_db, document_models=[F1NewsDocument])
    print(" MongoDB Connected & Beanie Initialized!")

    # ---------------------------------------------------------
    # 2. 크롤링 테스트 시작
    # ---------------------------------------------------------
    dispatcher = CrawlerDispatcher.build().register_autosport()
    target_link = "https://www.autosport.com/f1/news/lambiase-to-remain-as-verstappens-race-engineer-for-f1-2026/10788338/"
    
    crawler = dispatcher.get_crawler(target_link)
    
    try:
        if crawler:
            print(f" 크롤링 시작: {target_link}")
            
            # extract는 동기 함수지만, 이미 Beanie가 초기화되었으므로 내부에서 F1NewsDocument 생성 가능
            data = crawler.extract(target_link)
            
            print("\n[수집 결과 Report]")
            # datetime 객체 직렬화 처리를 위해 default=str 추가
            print(json.dumps(data, indent=4, ensure_ascii=False, default=str))
            
            if data and data.get('title'):
                print("\n 테스트 성공! 데이터가 정상적으로 추출되었습니다.")
            else:
                print("\n 테스트 실패: 데이터를 가져오지 못했습니다.")
                
        else:
            print(" 크롤러 배정 실패")
            
    except Exception as e:
        print(" 실행 실패")
        print(traceback.format_exc())

if __name__ == "__main__":
    # 비동기 함수 실행
    asyncio.run(main())
# data_pipeline/crawlers/dispatcher.py

import re
from urllib.parse import urlparse
from .base import BaseCrawler
# (크롤러 임포트는 그대로 유지)
from .f1_news import AutosportCrawler 
from .f1_tactic import Formula1Crawler

class CrawlerDispatcher:
    def __init__(self) -> None:
        self._crawlers = {}

    @classmethod
    def build(cls) -> "CrawlerDispatcher":
        return cls()

    def register(self, domain: str, crawler: type[BaseCrawler]) -> "CrawlerDispatcher":
        """
        도메인을 등록. 입력이 'https://www.autosport.com'이든 'autosport.com'이든
        알아서 깔끔한 정규식으로 변환하여 매칭 확률을 높입니다.
        """
        parsed = urlparse(domain)
        # 1. netloc이 없으면(http 안 붙인 경우) domain 자체를 사용
        netloc = parsed.netloc if parsed.netloc else domain
        
        # 2. 'www.' 접두사 제거 (정규식 중복 방지)
        if netloc.startswith("www."):
            netloc = netloc[4:]
            
        # 3. 강력한 정규식 생성
        # ^ : 시작점 고정
        # https? : http 또는 https
        # (www\.)? : www는 있어도 되고 없어도 됨
        # .* : 도메인 뒤에 무슨 경로(/f1/news/...)가 붙어도 매칭 성공
        pattern = r"^https?://(www\.)?{}.*".format(re.escape(netloc))
        
        self._crawlers[pattern] = crawler
        # print(f" [Dispatcher] Registered: {pattern}") # 디버깅용
        return self

    def register_autosport(self) -> "CrawlerDispatcher":
        # autosport 등록
        self.register("autosport.com", AutosportCrawler)
        return self
    
    def register_formula1(self) -> "CrawlerDispatcher":
        #formular1 등록
        # http 또는 https로 시작 + (www.)은 있어도 그만 없어도 그만 + formula1.com + 뒤에 아무거나
        pattern = r"^https?://(www\.)?formula1\.com.*"
        self._crawlers[pattern] = Formula1Crawler
        return self

    def get_crawler(self, url: str) -> BaseCrawler:
        for pattern, crawler_cls in self._crawlers.items():
            if re.match(pattern, url):
                print(f" Dispatcher Match: '{url}' \n   -> {crawler_cls.__name__}")
                return crawler_cls()
        
        print(f" Dispatcher Fail: '{url}'에 맞는 크롤러 없음.")
        print(f"   (현재 등록된 패턴 수: {len(self._crawlers)})")
        return None
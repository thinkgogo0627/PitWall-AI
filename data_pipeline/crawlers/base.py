# data_pipeline/crawlers/base.py

from abc import ABC, abstractmethod
import time
from tempfile import mkdtemp
import os

# Selenium 관련 (필요할 때만 설치하도록 try-except 처리 추천하지만, 일단 그대로 둠)
import chromedriver_autoinstaller
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# 크롬 드라이버 자동 설치 확인
try:
    chromedriver_autoinstaller.install()
except:
    pass # 이미 설치되어 있거나 권한 문제 시 패스 (Airflow 환경에 따라 다름)


## 앞으로 다른 크롤러에 사용될 기본적인 Base crawler
class BaseCrawler(ABC):
    @abstractmethod
    def extract(self, link: str, **kwargs) -> dict:
        """
        링크를 받아 데이터를 추출합니다.
        return: 수집된 데이터 딕셔너리 (DB 저장용)
        """
        ...

class BaseSeleniumCrawler(ABC):
    def __init__(self):
        self.driver = self._init_driver()

    def _init_driver(self):
        options = Options()
        # Docker 환경 필수 옵션들
        options.add_argument("--headless") # 화면 없이 실행
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        
        # 하위 클래스(F1, Autosport)에서 추가 옵션 설정
        self.set_extra_driver_options(options)

        # [핵심 변경] Remote WebDriver 사용!
        # command_executor 주소가 'selenium-chrome' 컨테이너를 가리킴
        driver = webdriver.Remote(
            command_executor='http://selenium-chrome:4444/wd/hub',
            options=options
        )
        return driver

    def set_extra_driver_options(self, options) -> None:
        """하위 클래스에서 오버라이딩"""
        pass

    @abstractmethod
    def extract(self, link: str, **kwargs) -> dict:
        pass
    def scroll_page(self) -> None:
        """무한 스크롤 페이지 대응"""
        current_scroll = 0
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2) # 로딩 대기
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height or (self.scroll_limit and current_scroll >= self.scroll_limit):
                break
            last_height = new_height
            current_scroll += 1
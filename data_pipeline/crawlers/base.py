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

class BaseSeleniumCrawler(BaseCrawler, ABC):
    """
    [기반 클래스] Selenium이 필요한 크롤러를 위한 공통 기능을 제공.
    """
    def __init__(self, scroll_limit: int = 5):
        options = webdriver.ChromeOptions()
        
        # [헤드리스 모드 및 최적화 옵션]
        options.add_argument("--no-sandbox")
        options.add_argument("--headless=new") # 화면 없이 실행
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--log-level=3")
        options.add_argument(f"--user-data-dir={mkdtemp()}")
        options.add_argument(f"--data-path={mkdtemp()}")
        options.add_argument(f"--disk-cache-dir={mkdtemp()}")
        
        self.set_extra_driver_options(options)
        
        self.scroll_limit = scroll_limit
        # 드라이버 초기화 (실패 시 에러 로그가 명확히 뜨도록)
        try:
            self.driver = webdriver.Chrome(options=options)
        except Exception as e:
            print(f"Chrome Driver Init Error: {e}")
            raise e

    def set_extra_driver_options(self, options: Options) -> None:
        """하위 클래스에서 옵션을 추가할 수 있는 훅(Hook)"""
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
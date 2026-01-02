from data_pipeline.crawlers import autosport , gp_korea , race_strat , FIA_reg
import pandas as pd
import os
from datetime import datetime

def run_pipeline():
    print("데이터 파이프라인 가동")
    print('='*50)

    all_data = []

    # 1. GPKorea 수집
    try:
        df_kr = gp_korea.crawl_gpkorea_final()
        if not df_kr.empty:
            print(f"  └ 🇰🇷 GP Korea: {len(df_kr)}건")
            all_data.append(df_kr)
    except Exception as e:
        print(f"  └ GP Korea 실패: {e}")

    # 2. Autosport 수집
    try:
        # Selenium 필요한 크롤러
        df_en = autosport.crawl_autosport_full() # 함수명 확인 필요
        if not df_en.empty:
            print(f"  └ 🇬🇧 Autosport: {len(df_en)}건")
            all_data.append(df_en)
    except Exception as e:
        print(f"  └  Autosport 실패: {e}")

    # 3. FIA 규정집 (옵션: 필요할 때만 켜기)
    # 매번 긁으면 오래 걸리니까 일단 주석 처리하거나 플래그로 관리
    run_fia = False 
    if run_fia:
        try:
            df_fia = FIA_reg.crawl(doc_type="sporting")
            if not df_fia.empty:
                print(f"  └  FIA Docs: {len(df_fia)}건")
                all_data.append(df_fia)
        except Exception as e:
            print(f"  └  FIA Docs 실패: {e}")


    # 4. race_strat 수집
    try:
        # Selenium 필요한 크롤러
        df_strat = race_strat.crawl(limit=150) # 함수명 확인 필요
        if not df_strat.empty:
            print(f"  └ 🇬🇧 Race strategy: {len(df_strat)}건")
            all_data.append(df_strat)
    except Exception as e:
        print(f"  └ Race strategy 실패: {e}")


    ## 데이터 통합, 저장
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        
        # 저장 폴더 확인
        save_dir = "data/raw"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 날짜별 파일명 생성
        today = datetime.now().strftime("%Y%m%d")
        filename = f"{save_dir}/f1_data_collection_{today}.csv"
        
        final_df.to_csv(filename, index=False, encoding='utf-8-sig')
        print("="*50)
        print(f"파이프라인 완료! 총 {len(final_df)}건 저장됨.")
        print(f"경로: {filename}")
        
    else:
        print("수집된 데이터가 없습니다.")

if __name__ == "__main__":
    run_pipeline()

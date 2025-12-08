## Important queries..

## 3가지 기능 구현됨


## 타이어 컴파운드별 랩타임 비교
## 정확한 랩 에서, 랩의 중반 페이스
## (초반은 어차피 타이어 온도때문에 데이터 딱히 무의미, 후반은 Lico를 하는 경우같은거 제외 시 어차피 풀푸시)
## 트랙마다 56랩, 71랩 다르니까 중간만 대충 계산

SELECT 
    Driver, 
    Compound, 
    COUNT(*) as Laps_Run,
    ROUND(AVG(LapTime_Sec), 3) as Avg_Pace,
    ROUND(MIN(LapTime_Sec), 3) as Best_Lap
FROM lap_times
WHERE RaceID LIKE '2025%'
  AND IsAccurate = 1
  AND LapNumber BETWEEN 
      (SELECT MAX(LapNumber) * 0.15 FROM lap_times WHERE RaceID LIKE '2025%') 
      AND 
      (SELECT MAX(LapNumber) * 0.85 FROM lap_times WHERE RaceID LIKE '2025%')
GROUP BY Driver, Compound
ORDER BY Driver ASC;



## 순위 상승폭 체크
SELECT 
    Driver, 
    TeamName,
    GridPosition, 
    Position as FinishPosition,
    (GridPosition - Position) as Positions_Gained
FROM race_results
WHERE RaceID = '{Target GP}' AND Status = 'Finished'
ORDER BY Positions_Gained DESC;



## 드라이버들의 스틴트별 타이어 컴파운드와 랩타임 비교
SELECT
    Driver,
    Stint,
    Compound,
    COUNT(*) as Laps_In_Stint,
    ROUND(AVG(LapTime_Sec), 3) as Stint_Pace
FROM lap_times
WHERE RaceID LIKE '2025%'
GROUP BY Driver, Stint, Compound
ORDER BY Driver, Stint;
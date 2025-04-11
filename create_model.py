import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import json, os

# 데이터 불러오기 및 전처리
total_df = pd.read_csv("data/seoul_real_estate.csv")
total_df['CTRT_DAY'] = pd.to_datetime(total_df['CTRT_DAY'], format="%Y-%m-%d")
total_df = total_df[total_df['BLDG_USG'] == '아파트'] # 0}}=} &
#자치구 리스트 추출 (정렬)
sgg_nms = sorted(total_df['CGG_NM'].unique())
#예측 기간 설정
periods = 28
# 자치구별 Prophet 모델 생성 및 저장
for sgg_nm in sgg_nms:
    model = Prophet()
    
    #자치구별 요약 평균 데이터 생성
    df = total_df[total_df['CGG_NM'] == sgg_nm] [['CTRT_DAY', 'THING_AMT']]
    df = df.groupby('CTRT_DAY')['THING_AMT'].mean().reset_index()
    df = df.rename(columns={'CTRT_DAY': 'ds', 'THING_AMT': 'y'})
    
    model.fit(df)
    os.makedirs("ml/models", exist_ok=True)
    #모델 저장
    with open(f'ml/models/{sgg_nm}_model.json', 'w') as f:
        json.dump(model_to_json(model), f)
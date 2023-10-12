import pandas as pd
import glob
import datetime


data_path = glob.glob('./crawling_data/*')# crawling_data/ 디렉토리의 * 모든파일
print(data_path)

df = pd.DataFrame()
for path in data_path:
    df_temp = pd.read_csv(path,index_col = 0) # path 에 있는 csv 파일 읽음.
    df = pd.concat([df, df_temp])
print(df.head())
print(df['category'].value_counts())
df.info()
df.to_csv('./crawling_data/naver_news_titles_{}.csv'.format(
    datetime.datetime.now().strftime('%Y%m%d'), index=False))
















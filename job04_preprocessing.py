import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle

pd.set_option('display.unicode.east_asian_width', True)
df = pd.read_csv('./crawling_data/naver_news_titles_20231012.csv')
print(df.head())
df.info()

X = df['titles']
Y = df['category']

encoder = LabelEncoder()
labeled_y = encoder.fit_transform(Y)
print(labeled_y[:3])
label = encoder.classes_
# print(label)

onehot_y = to_categorical(labeled_y)
# print(onehot_y)

okt = Okt()
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
# 데이터의 수가 적으면 stem=True 를 줘서 원형으로 바꿈?
print(X)



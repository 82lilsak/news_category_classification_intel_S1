import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import load_model

df = pd.read_csv('./crawling_data/naver_headline_news_20231012.csv')

print(df.head())
df.info()

X = df['titles']
Y = df['category']

with open('./models/encoder.pickle', 'rb') as f:
    encoder = pickle.load(f)

labeled_y = encoder.transform(Y)    #fit 할때 새로 저장 해서 fittransform 대신 transform 사용.
label = encoder.classes_


onehot_y = to_categorical(labeled_y)
print(onehot_y)

okt = Okt()

for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)

stopwords = pd.read_csv('./stopwords.csv', index_col=0)

for j in range(len(X)):
    words = []
    for i in range(len(X[j])):  # j 번째 뉴스 제목 길이마다 for문 실행
        if len(X[j][i]) > 1:    # j번째 문장 의 길이가 1보다 큰 경우
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])   # 한글자 보다 크고 불용어 도 아닌 단어들
    X[j] = ' '.join(words)  # 띄어쓰기를 넣어서 붙여준다.

with open('./models/news_token.pickle', 'rb') as f:
    token = pickle.load(f)

tokened_x = token.texts_to_sequences(X)

for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 21:
        tokened_x[i] = tokened_x[i][:22]

x_pad = pad_sequences(tokened_x, 21)


model = load_model('./models/news_category_classification_model_0.7139992117881775.h5')
preds = model.predict(x_pad)
predicts = []
for pred in preds:
    most = label[np.argmax(pred)]
    pred[np.argmax(pred)] = 0   # 제일 큰값을 0 으로 바꿈
    second = label[np.argmax(pred)]

    predicts.append([most, second])

df['predict'] = predicts
print(df.head(30))
df['OX'] = 0
for i in range(len(df)):
    if df.loc[i, 'category'] in df.loc[i, 'predict']:
        df.loc[i, 'OX'] = 'O'
    else:
        df.loc[i, 'OX'] = 'X'


print(df['OX'].value_counts())
print(df['OX'].value_counts()/len(df))
for i in range(len(df)):
    if df['category'][i] not in df['predict'][i]:
        print(df.iloc[i])
        # 틀린것들 표기

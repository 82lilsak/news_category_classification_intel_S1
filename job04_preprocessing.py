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

encoder = LabelEncoder()    # LabelEncoder 객체 생성
labeled_y = encoder.fit_transform(Y)    # classes_ 에 대한 정보를 encoder가 갖게된다. 0,1,2,3,4,5,6 의 라벨정보를 갖고있어야 한다.
# print(labeled_y[:3])
label = encoder.classes_
# print(label)
with open('./models/encoder.pickle', 'wb') as f:    # 'wb' b == 바이너리, 텍스트는 w 사용, 바이너리는 wb 사용
    pickle.dump(encoder, f)


onehot_y = to_categorical(labeled_y)
# print(onehot_y)

okt = Okt()
for i in range(len(X)):
    X[i] = okt.morphs(X[i], stem=True)
# 데이터의 수가 적으면 stem=True 를 줘서 원형으로 바꿈?
# print(X)    # 형태소 들의 리스트

stopwords = pd.read_csv('./stopwords.csv', index_col=0) # 불용어 제거
for j in range(len(X)):
    words = []
    for i in range(len(X[j])):  # j 번째 뉴스 제목 길이마다 for문 실행
        if len(X[j][i]) > 1:    # j번째 문장 의 길이가 1보다 큰 경우
            if X[j][i] not in list(stopwords['stopword']):
                words.append(X[j][i])   # 한글자 보다 크고 불용어 도 아닌 단어들
    X[j] = ' '.join(words)  # 띄어쓰기를 넣어서 붙여준다.
# print(X)

token = Tokenizer() # Tokenizer 객체
token.fit_on_texts(X)   #
tokened_x = token.texts_to_sequences(X) # 형태소를 라벨로 바꿔주는 문장.
wordsize = len(token.word_index) + 1     # +1 한 이유는// 라벨링은 0을 쓰지않는다 1부터시작
# token.word_index == 유니크한 텍스트가 몇개 있었는지
print(tokened_x[0:3])
print(wordsize)

with open('./models/news_token.pickle', 'wb') as f:
    pickle.dump(token, f)

# 제일 긴 문장에 맞춰서 짧은 문장들은 앞 부분을 0 으로 채워서 길이를 맞춰줘야 한다.

max = 0
for i in range(len(tokened_x)):
    if max < len(tokened_x[i]):
        max = len(tokened_x[i])

print(max)

x_pad = pad_sequences(tokened_x, max)   # max 에 맞춰 모자라면0 을 채워넣게 해줌.
print(x_pad[:3])

X_train, X_test, Y_train, Y_test = train_test_split(
    x_pad, onehot_y, test_size=0.2)     # 쪼개기 x_pad 와 oneohtY 를 0.2의 배율로
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = X_train, X_test, Y_train, Y_test
np.save('./crawling_data/news_data_max_{}_wordsize_{}'.format(max, wordsize), xy)

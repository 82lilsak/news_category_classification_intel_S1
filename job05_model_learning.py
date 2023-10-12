import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load(
    './crawling_data/news_data_max_21_wordsize_12676.npy', allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(12676, 300, input_length=21))   # input_length=21 max 값 21 입력.
# Embedding 레이어에 12676 이란 워드 사이즈를 넣어준다.
# output_dim: 300 == 300 차원 으로 줄임.
# 차원 축소를 하는 이유 .. 차원이 커지면 데이터가 희소해져서 데이터를 늘릴수 없기 때문에 데이터에 맞춰서 줄여줘야한다.
model.add(Conv1D(32, kernel_size=5, padding='same', activation='relu'))
# 문장이라 1차원 이어서 Cnov1D 사용. 커널 사이즈도 1줄의 문장이라 * X* 형식이 아니다.
model.add(MaxPooling1D(pool_size=1))    # 아무일도 일어나지 않는다. 빼도 상관없음
# 보통 Conv 이후에 MaxPool 사용.
model.add(LSTM(128, activation='tanh', return_sequences=True))
# 순서가 있는 데이터라 LSTM 사용.
# return_sequences=True ==
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
# 분류 모델이라 마지막 레이어에 유닛 6, 소프트맥스 사용.
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
model.save('./models/news_category_classification_model_{}.h5'.format(fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')

plt.legend()
plt.show()










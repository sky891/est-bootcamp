pip install --upgrade pip
pip install tensorflow[and-cuda]==2.14.0

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
import math
import numpy as np
from tensorflow.keras import backend as K
import gc
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization


# TRAIN_IMG = '../data/train/img/train/'
# VAL_IMG = '../data/train/img/val/'
# TEST_IMG = '../data/test/img/'

TRAIN_IMG = '../daewoong/data_cropped/train/'
VAL_IMG = '../daewoong/data_cropped/val/'
TEST_IMG = '../daewoong/data_cropped/test/'

# 학습, 검증, 분류할 클래스, 이미지 크기, 배치 크기 설정
TRAIN_SAMPLES = 6000
VALIDATION_SAMPLES = 1200
NUM_CLASSES = 4
IMG_WIDTH, IMG_HEIGHT = 224, 224 # image size for MobileNet
BATCH_SIZE = 16 # 64

# 1. 모델 객체 삭제
try:
    del model_final
    print("모델 객체가 삭제되었습니다.")
except NameError:
    print("모델 객체가 이미 삭제되었거나 존재하지 않습니다.")

# 2. 백엔드 초기화
K.clear_session()
print("Keras 백엔드가 초기화되었습니다.")

# 3. 메모리 정리
gc.collect()
print("메모리가 정리되었습니다.")

# # 학습 데이터에 대해서는 데이터 증강 적용 -> 모델의 일반화 성능 향상
# train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, # 입력 포맷팅, normalization(픽셀 값 조정)
#                                    rotation_range=20,
#                                    width_shift_range=0.2,
#                                    height_shift_range=0.2,
#                                    zoom_range=0.2, # 최대 20%까지 무작위 확대
#                                    horizontal_flip=True,
#                                    vertical_flip=True)

# 증강하지 않은 train_datagen
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# 검증 데이터에는 증강 적용 X
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
  # 다른 옵션 없음, 증대 변환을 적용 X -> 실제 검증 환경의 예상 형식 및 분포와 최대한 유사하게 데이터에 대한 모델 평가

# 학습 및 검증 데이터 제너레이터를 설정
# train_datagen을 사용해서 학습 데이터 호출, 모델에 제공
# data augmentation -> 이미지 배치가 에포크마다 모델에 공급
# train_generator는 학습 동안 모델에 데이터를 제공하는 역할을 합니다. train_datagen에 의해 지정된 데이터 증강과

train_generator = train_datagen.flow_from_directory(TRAIN_IMG, # 학습 이미지가 저장된 디렉토리의 경로입니다. 이 경로 내의 폴더 구조는 각 클래스별로 분류되어 있어야 하며, 각 클래스 폴더 내에 해당 클래스의 이미지들이 들어 있습니다.
                                                    target_size=(IMG_WIDTH,  # 모델에 입력될 이미지의 크기 설정, MobileNet 224x224
                                                                 IMG_HEIGHT),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    seed=12345, # 데이터 randomness...
                                                    class_mode='categorical') # 레이블을 원-핫 인코딩 형태로 변환

# val_datagen을 이용해서 검증 데이터를 불러옴
validation_generator = val_datagen.flow_from_directory(
    VAL_IMG,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical')

# Functional
# MobileNet을 기반으로 사용자 정의 모델 구축
# 모든 레이어의 학습 가능 여부를 False로 설정 -< 가중치를 고정
def model_maker():
    # top 층을 제외한 사전학습 모형 호출
    base_model = MobileNet(include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

    # 미세 조정을 위해 상위 레이어들의 학습 가능 설정을 True로 변경
    for layer in base_model.layers[-60:]:  # 마지막 30개 레이어의 학습을 가능하게 설정
        layer.trainable = True

    # Input 클래스를 이용해서 입력 데이터의 형태를 지정
    input1 = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3))
    custom_model = base_model(input1)
    custom_model = GlobalAveragePooling2D()(custom_model) # 각 채널 (7x7 채널)에서 평균값 하나만 추출
        # 기본 네트워크의 출력 feature 맵의 차원(높이 및 너비)을 줄임
    custom_model = Dense(64, activation='relu')(custom_model)
    custom_model = Dense(32, activation='relu')(custom_model)
    predictions = Dense(NUM_CLASSES, activation='softmax')(custom_model)
        # 출력 층, NUM_CLASSES : 예측 클래스 수 = 4
    return Model(inputs=input1, outputs=predictions)

model_final = model_maker()

model_final.summary()

# 손실 함수, 최적화 알고리즘, 평가 지표 설정
model_final.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.001), # 0.0001
              metrics=['acc'])

# EarlyStopping 콜백 설정
early_stopping = EarlyStopping(
    monitor='val_loss',       # 검증 손실(val_loss)을 모니터링
    patience=5,               # 개선되지 않는 epoch 수 (조기 종료 대기 기간)
    restore_best_weights=True # 최상의 가중치를 복원
)

# 모델을 컴파일, 학습 수행
history = model_final.fit(
    train_generator, # 학습데이터 호출
    steps_per_epoch=TRAIN_SAMPLES // BATCH_SIZE, # number of updates, 1 epoch 당 업데이트 횟수 지정
    epochs=20, # 20,
    validation_data=validation_generator, # 검증 데이터 호출
    validation_steps=VALIDATION_SAMPLES // BATCH_SIZE,
    callbacks=[early_stopping])  # EarlyStopping 콜백 추가
      # ImageDataGenerator가 각 batch마다 random하게 BATCH_SIZE (=64)에 해당하는 이미지 생성

# 지정된 디렉토리
save_dir = './models/'
# os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성

# 모델 저장
model_final.save(os.path.join(save_dir, 'model15.h5'))

# # 모델 로드
# model = load_model(os.path.join(save_dir, 'model01.h5'))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['train','val'])
plt.show()

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    TEST_IMG,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    shuffle=False,
    class_mode='categorical')

# Test 데이터셋으로 모델 평가
test_loss, test_accuracy = model_final.evaluate(
    test_generator,  # 테스트 데이터 호출
    steps=test_generator.samples // BATCH_SIZE,  # 테스트 데이터 업데이트 횟수
    verbose=1  # 진행 상황 출력
)

print(f"Test Loss: {test_loss}%")
print(f"Test Accuracy: {test_accuracy}%")

# np.set_printoptions(suppress=True)

# # img_path = '../data/test/img/anger/0h5sfbee84a3ee8e7efdd0f578f83e4df7e491dbcdab482a480a7add89daegs3z.jpg' # anger
# # img_path = '../data/test/img/anger/0ir64e27c17818b295d32b497973b93d343dfb707521b3049d8d71b6589cd9gl8.jpg' # anger2
# # img_path = '../data/test/img/happy/0mc92e557f414f8933d90960b0a867ee96dcbdc6e46dc65d363e3a82bb4a5tyvo.jpg' # happy
# img_path = '../data/test/img/panic/0cal9ede3eabc2ec3303e630d8ec2364fd73404116908101342edfeddccf12c7p.jpg' # panic
# # img_path = '../data/test/img/sadness/0cyn350e7b9f3be037d2b7747cc267375f3cf3f1f36e5372e3ff311f48fdff9g5.jpg' # sadness
# img = image.load_img(img_path, target_size=(224, 224))
# img_array = image.img_to_array(img)
# plt.imshow(img_array/255)
# expanded_img_array = np.expand_dims(img_array, axis=0)
# preprocessed_img = expanded_img_array / 255

# prediction = model_final.predict(preprocessed_img)
# print(np.array(prediction[0]))


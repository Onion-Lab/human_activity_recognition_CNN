'''
Class : Make CNN Model 
Writer : Lee ji hyun
Last Modified Date : 21.07.29
Modification details : Add Comments and Modify Final
'''

#################### Import Library ####################
import loader as loader
import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd
########################################################


# Dimension of images
#######################
img_width  = 41
img_height = 14
channels   = 6
#######################


# Data load
###############################
X_train = loader.load_x_train()
y_train = loader.load_y_train()
X_test = loader.load_x_test()
y_test = loader.load_y_test()
###############################

# Data shuffle
#################################################
tmp = [ [x,y] for x, y in zip(X_train, y_train)]

random.shuffle(tmp)

X_train=[n[0] for n in tmp]
y_train=[n[1] for n in tmp]

X_train = np.array(X_train)
y_train = np.array(y_train)
#################################################



# Train
#####################################################################################################
batch_size = 32
epochs = 30

model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(3, 3), activation='relu', strides=1, padding='same', 
                data_format='channels_last', input_shape=(41,14,9)))
model.add(BatchNormalization())
# 2차원 CNN Layer 추가, filter의 갯수를 8개로 설저으 kernel의 크기는 3*3 activation 함수는 relu, 손실되는 부분이 없게하기 위해 padding은 same으로 설정
# filter의 갯수가 8인 이유는 input shape가 41 * 14 image여서 filtler의 shape가 너무 커지면 안된다고 판단.
# 이후 2^n 형식으로 filter의 크기를 늘려가며 전체적인 image를 확인함

model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', strides=1, padding='same', 
                data_format='channels_last'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
    
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', strides=1, padding='same', 
                data_format='channels_last'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten()) # 평활화 진행
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(7, activation='softmax')) # 7가지 종류를 분류해야하므로 activation 함수 softmax 사용


model.compile(optimizer='Adam', loss="categorical_crossentropy", metrics=["accuracy"]) # 다중 분류 loss 함수 categorical_crossentropy


hist = model.fit(x=X_train, y=y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(X_test,y_test))

model.summary() # 모델 summary 출력

#####################################################################################################


# Check Score
######################################
score = model.evaluate(X_test, y_test)  # test data를 통한 validation 진행
#######################################


ACTIVITIES = {
    0: 'WALKING',
    1: 'WALKING_UPSTAIRS',
    2: 'WALKING_DOWNSTAIRS',
    3: 'SIT TO STANDING',
    4: 'STANDING',
    5: 'LAYING',
    6: 'STANDING TO SIT'
}
score = model.evaluate(X_test, y_test)  # test data를 통한 validation 진행
print(score)
def confusion_matrix(Y_true, Y_pred):
    Y_true = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_true, axis=1)])
    Y_pred = pd.Series([ACTIVITIES[y] for y in np.argmax(Y_pred, axis=1)])

    return pd.crosstab(Y_true, Y_pred, rownames=['True'], colnames=['Pred'])

# Evaluate
print(confusion_matrix(y_test, model.predict(X_test)))
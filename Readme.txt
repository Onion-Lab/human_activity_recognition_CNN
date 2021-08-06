// input shape가 LSTM과 달라 loader를 새로 만들어야했는데 급조한 loader이다보니 코드가 지저분한점 죄송합니다.

< Dataset >
test : test에 이용할 데이터
train : train에 이용할 데이터
activity_labels.txt : Labeling index

body_acc_x_train : 중력가속도를 포함한 몸의 가속도 데이터 (x,y,z 공통)
body_gyro_x_train : calibration 되어있는 각속도 데이터 (x,y,z 공통)
removeG_acc_x_train : 중력가속도를 제거한 몸의 가속도 데이터 (x,y,z 공통)
y_train : train에 이용할 정답 Label

body_acc_x_test : 중력가속도를 포함한 몸의 가속도 데이터 (x,y,z 공통)
body_gyro_x_test : calibration 되어있는 각속도 데이터 (x,y,z 공통)
removeG_acc_x_test : 중력가속도를 제거한 몸의 가속도 데이터 (x,y,z 공통)
y_test : test에 이용할 정답 Label

정석대로라면 CNN에서 사용할 Dataset을 따로 구비해야하나 LSTM에서 확보한 데이터셋을 다시 분할하여
2차원 CNN Dataset 6개를 만드는것은 시간적 낭비라고 판단되어 Data load를 기존 LSTM에서 사용한 데이터셋을 로드 후
input shape에 맞춰 STFT 변환을 한 후 학습을 진행함

< 프로그램 구동방법 >
1. loader.py -> xlsx_ax, xlsx_ay, xlsx_az, xlsx_gx, xlsx_gy, xlsx_gz 변수의 절대경로를 
                    dataset의 Inertial Signals폴더 안의 파일로 경로 지정
                -> y_train, y_test 각각 파일의 절대경로 지정
                -> load_x_train(), load_y_train(), load_x_test(), load_y_test()
                    총 4 개의 함수에서 파일의 절대경로를 지정



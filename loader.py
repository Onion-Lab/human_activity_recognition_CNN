'''
Class : Make CNN Model 
Writer : Cho jun ho
Last Modified Date : 21.07.29
Modification details : Add Comments and Modify Final
'''

#################### Import Library ####################
import numpy as np
import librosa.display
import pandas as pd
import warnings
from tensorflow.keras.utils import to_categorical
warnings.filterwarnings(action='ignore')
########################################################


def load_x_train() :
    all=[]
    xlsx_ax = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\train\\Inertial Signals\\body_acc_x_train.txt', header=None, delim_whitespace=True)
    xlsx_ay = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\train\\Inertial Signals\\body_acc_y_train.txt', header=None, delim_whitespace=True)
    xlsx_az = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\train\\Inertial Signals\\body_acc_z_train.txt', header=None, delim_whitespace=True)
    xlsx_gx = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\train\\Inertial Signals\\body_gyro_x_train.txt', header=None, delim_whitespace=True)
    xlsx_gy = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\train\\Inertial Signals\\body_gyro_y_train.txt', header=None, delim_whitespace=True)
    xlsx_gz = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\train\\Inertial Signals\\body_gyro_z_train.txt', header=None, delim_whitespace=True)
    xlsx_ax_g = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\train\\Inertial Signals\\body_gyro_x_train.txt', header=None, delim_whitespace=True)
    xlsx_ay_g = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\train\\Inertial Signals\\body_gyro_y_train.txt', header=None, delim_whitespace=True)
    xlsx_az_g = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\train\\Inertial Signals\\body_gyro_z_train.txt', header=None, delim_whitespace=True)

    for i in range(len(xlsx_ax)):
        ax = xlsx_ax.loc[[i],:]
        ay = xlsx_ay.loc[[i],:]
        az = xlsx_az.loc[[i],:]
        gx = xlsx_gx.loc[[i],:]
        gy = xlsx_gy.loc[[i],:]
        gz = xlsx_gz.loc[[i],:]
        ax_g = xlsx_ax_g.loc[[i],:]
        ay_g = xlsx_ay_g.loc[[i],:]
        az_g = xlsx_az_g.loc[[i],:]


        ax=np.array(ax)
        ay=np.array(ay)
        az=np.array(az)
        gx=np.array(gx)
        gy=np.array(gy)
        gz=np.array(gz)
        ax_g=np.array(ax_g)
        ay_g=np.array(ay_g)
        az_g=np.array(az_g)

        D1 = np.abs(librosa.stft(ax[0], n_fft=80, hop_length=10))
        D2 = np.abs(librosa.stft(ay[0], n_fft=80, hop_length=10))
        D3 = np.abs(librosa.stft(az[0], n_fft=80, hop_length=10))
        D4 = np.abs(librosa.stft(gx[0], n_fft=80, hop_length=10))
        D5 = np.abs(librosa.stft(gy[0], n_fft=80, hop_length=10))
        D6 = np.abs(librosa.stft(gz[0], n_fft=80, hop_length=10))
        D7 = np.abs(librosa.stft(ax_g[0], n_fft=80, hop_length=10))
        D8 = np.abs(librosa.stft(ay_g[0], n_fft=80, hop_length=10))
        D9 = np.abs(librosa.stft(az_g[0], n_fft=80, hop_length=10))

        stft_=[D1,D2,D3,D4,D5,D6,D7,D8,D9]

        all.append(np.dstack(stft_))


    train_data=np.array(all)

    return train_data
    # return after stft train_data 

def load_y_train() :
    y_train = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\train\\y_train.txt', header=None, delim_whitespace=True,encoding='utf-16').values
    y_train = y_train - 1
    
    y_train = to_categorical(y_train)

    return y_train
    # return y_train


def load_x_test() :
    all=[]
    xlsx_ax = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\test\\Inertial Signals\\body_acc_x_test.txt', header=None, delim_whitespace=True)
    xlsx_ay = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\test\\Inertial Signals\\body_acc_y_test.txt', header=None, delim_whitespace=True)
    xlsx_az = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\test\\Inertial Signals\\body_acc_z_test.txt', header=None, delim_whitespace=True)
    xlsx_gx = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\test\\Inertial Signals\\body_gyro_x_test.txt', header=None, delim_whitespace=True)
    xlsx_gy = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\test\\Inertial Signals\\body_gyro_y_test.txt', header=None, delim_whitespace=True)
    xlsx_gz = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\test\\Inertial Signals\\body_gyro_z_test.txt', header=None, delim_whitespace=True)
    xlsx_ax_g = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\test\\Inertial Signals\\body_gyro_x_test.txt', header=None, delim_whitespace=True)
    xlsx_ay_g = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\test\\Inertial Signals\\body_gyro_y_test.txt', header=None, delim_whitespace=True)
    xlsx_az_g = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\test\\Inertial Signals\\body_gyro_z_test.txt', header=None, delim_whitespace=True)
    for i in range(len(xlsx_ax)):
        ax = xlsx_ax.loc[[i],:]
        ay = xlsx_ay.loc[[i],:]
        az = xlsx_az.loc[[i],:]
        gx = xlsx_gx.loc[[i],:]
        gy = xlsx_gy.loc[[i],:]
        gz = xlsx_gz.loc[[i],:]
        ax_g = xlsx_ax_g.loc[[i],:]
        ay_g = xlsx_ay_g.loc[[i],:]
        az_g = xlsx_az_g.loc[[i],:]


        ax=np.array(ax)
        ay=np.array(ay)
        az=np.array(az)
        gx=np.array(gx)
        gy=np.array(gy)
        gz=np.array(gz)
        ax_g=np.array(ax_g)
        ay_g=np.array(ay_g)
        az_g=np.array(az_g)

        D1 = np.abs(librosa.stft(ax[0], n_fft=80, hop_length=10))
        D2 = np.abs(librosa.stft(ay[0], n_fft=80, hop_length=10))
        D3 = np.abs(librosa.stft(az[0], n_fft=80, hop_length=10))
        D4 = np.abs(librosa.stft(gx[0], n_fft=80, hop_length=10))
        D5 = np.abs(librosa.stft(gy[0], n_fft=80, hop_length=10))
        D6 = np.abs(librosa.stft(gz[0], n_fft=80, hop_length=10))
        D7 = np.abs(librosa.stft(ax_g[0], n_fft=80, hop_length=10))
        D8 = np.abs(librosa.stft(ay_g[0], n_fft=80, hop_length=10))
        D9 = np.abs(librosa.stft(az_g[0], n_fft=80, hop_length=10))

        stft_=[D1,D2,D3,D4,D5,D6,D7,D8,D9]

        all.append(np.dstack(stft_))


    test_data=np.array(all)

    return test_data
    # return after stft test_data 

def load_y_test() :
    y_test = pd.read_csv('C:\\Users\\Geovision-internship\\Downloads\\CNN_Model_Final\\dataset\\test\\y_test.txt', header=None, delim_whitespace=True,encoding='utf-16').values
    y_test = y_test - 1
    
    y_test = to_categorical(y_test)

    return y_test
    # return y_test
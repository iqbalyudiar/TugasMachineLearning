import pandas as pd
import math
from matplotlib import pyplot as plt
# from google.colab import files
# uploaded = files.upload()

x = []
teta1 = [0.3, 0,1, 0.4, 0.2]
teta2 = [0.7, 0.8, 0.5, 0.6]
bias1 = 0.2
bias2 = 0.3
alpha = 0.1
kelas1 = []
kelas2 = []
training = []
#read data from csv using pandas
def readData():
    data = pd.read_csv('data.csv')
    for i, row in data.iterrows():
        # print(row)
        x.append([float(row[i]) for i in range (len(row)-1)])
        if (row[4]=='Iris-setosa'):
            kelas1.append(0)
            kelas2.append(1)
        elif (row[4]=='Iris-versicolor'):
            kelas1.append(1)
            kelas2.append(0)
        elif (row[4]=='Iris-virginica'):
            kelas1.append(1)
            kelas2.append(1)

#target (h)
def target1(xi, teta1i, bias1i):
    res = 0.0
    for i in range(4):
      res += xi[i] * teta1i[i]
    res += bias1i
    return res

def target2(xi, teta2i, bias2i):
    res = 0.0
    for i in range(4):
      res += xi[i] * teta2i[i]
    res += bias2i
    return res

#error
def error1(prediksi1, fact1):
    return (fact1 - prediksi1) ** 2

def error2(prediksi2, fact2):
    return (fact2 - prediksi2) ** 2

#activation function
def sigmoid1(h):
    return 1/(1 + math.exp(-1 * h))

def sigmoid2(h):
    return 1/(1+ math.exp(-1 * h))

def delta_theta1(prediksi1, fact1, xtmp):
    return 2 * (fact1 - prediksi1) * (1 - prediksi1) * prediksi1 * xtmp

def delta_theta2(prediksi2, fact2, xtmp):
    return 2 * (fact2 - prediksi2) * (1 - prediksi2) * prediksi2 * xtmp

def delta_bias1(prediksi1, fact1):
    return 2 * (fact1 - prediksi1) * (1 - prediksi1) * prediksi1

def delta_bias2(prediksi2, fact2):
    return 2* (fact2 - prediksi2) * (1 - prediksi2) * prediksi2

def update_theta1(xi, pred1, fact1):
    for i in range(4):
        teta1[i] += alpha * delta_theta1(prediksi1, kelas1, xi[i])

def update_theta2(xi, pred2, fact2):
    for i in range(4):
        teta2[i] += alpha * delta_theta2(prediksi2, kelas2, xi[i])

def update_bias1(prediksi1, fact1):
    global bias1
    bias1 += alpha * delta_bias1(prediksi1, fact1)

def update_bias2(prediksi2, fact2):
    global bias2
    bias2 += alpha * delta_bias2(prediksi2, fact2)

def update_function(xi, prediksi1, prediksi2, fact1, fact2):
    update_theta1(xi, prediksi1, fact1)
    update_theta2(xi, prediksi2, fact2)
    update_bias1(prediksi1, fact1)
    update_bias2(prediksi2, fact2)

def train1():
    errori = 0.0
    for i in range (149):
        total = target1(x[i], teta1, bias1)
        pred = sigmoid1(total)
        errori += error1(pred, kelas1[i])
    return errori/149

def train2():
    errorj = 0.0
    for j in range(149):
        total = target2(x[j], teta2, bias2)
        pred = sigmoid2(total)
        errorj += error2(pred, kelas2[j])
    return errorj/150

def plot_err(*print_data):
    for data in print_data:
        plt.plot(data[0], label = data[1])
    plt.legend(loc = 'upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()

def start_train1():
    for epoch in range(60):
        training.append(train1())
    plot_err([training, 'Training1'])

def start_train2():
    for epoch in range(60):
        training.append(train2())
    # print(training)
    plot_err([training, 'Training2'])

readData()
start_train1()
start_train2()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import os
from datetime import datetime


def GetFeatures(path='/Users/Epilo/Desktop/COLLEGE/科研/石斛补充实验/', flag=False):
    features = []
    if flag:
        sr = 100
        for i in range(10):
            for root, dirs, files in os.walk("/Users/Epilo/Desktop/COLLEGE/科研/石斛补充实验/" + str(i + 1), topdown=True):
                j = 0
                for name in files:
                    print(os.path.join(root, name))
                    df = pd.read_table(os.path.join(root, name), delimiter='\t', encoding='latin1', dtype=float,
                                       usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
                    df1 = df.drop(len(df) - 1)
                    df2 = df.drop(0)
                    # features 1
                    dfmax = df.max(0)
                    # features 2
                    dfsum = df.sum(0)

                    # features 3 & 4
                    a1 = 1 / 100 / sr
                    l = a1 * df1
                    for k in range(1, len(df2) - 1):
                        l.iloc[k] = (1 - a1) * l.iloc[k - 1] + a1 * (df2.iloc[k - 1] - df1.iloc[k - 1])
                    a1max = l.max(0)
                    a1min = l.min(0)

                    # features 5 & 6
                    a2 = 1 / 10 / sr
                    l = a2 * df1
                    for k in range(1, len(df2) - 1):
                        l.iloc[k] = (1 - a2) * l.iloc[k - 1] + a2 * (df2.iloc[k - 1] - df1.iloc[k - 1])
                    a2max = l.max(0)
                    a2min = l.min(0)

                    # features 7 & 8
                    a3 = 1 / sr
                    l = a3 * df1
                    for k in range(1, len(df2) - 1):
                        l.iloc[k] = (1 - a3) * l.iloc[k - 1] + a3 * (df2.iloc[k - 1] - df1.iloc[k - 1])
                    a3max = l.max(0)
                    a3min = l.min(0)

                    if j == 0 and i == 0:
                        features = np.hstack((i, dfmax, dfsum, a1max, a1min, a2max, a2min, a3max, a3min))
                    else:
                        features = np.vstack((features,
                                              np.hstack((i, dfmax, dfsum, a1max, a1min, a2max, a2min, a3max, a3min))))
                    j = j + 1
        features = pd.DataFrame(features)
        features.to_csv('features.csv')
    else:
        features = pd.read_csv('features.csv')
    print('Loaded Successfully')
    # print(features)
    # print(features.shape)
    # print(features.__class__)
    return features


# -----     START     -----
time_start = datetime.now()

df_features128 = GetFeatures(flag=False)
# np.save('features.npy', df_features128)
features128 = df_features128.values
print('features128.shape = ', features128.shape)
np.random.shuffle(features128)
totallen = len(features128)
print('tlen=', totallen)
sep = int(totallen * 0.7)
train_set = features128[:sep]
test_set = features128[sep:]
print('trainset.shape = ', train_set.shape)

# -----    Data Pre    -----
train_x = train_set[:, 2:]
test_x = test_set[:, 2:]

print('train_x.shape = ', train_x.shape)
print(train_x[0, :])
train_y = train_set[:, 1:2].astype(int)
test_y = test_set[:, 1:2].astype(int)
print(train_y[330, :])

train_y8 = []
train_y4 = []
train_y2 = []
train_y1 = []
for i in range(len(train_y)):
    train_y8 = np.append(train_y8, int(train_y[i] / 8))
    train_y4 = np.append(train_y4, int(train_y[i] % 8 / 4))
    train_y2 = np.append(train_y2, int(train_y[i] % 4 / 2))
    train_y1 = np.append(train_y1, int(train_y[i] % 2))
    # print(train_y8[i], y4[i], y2[i], y1[i])

test_y8 = []
test_y4 = []
test_y2 = []
test_y1 = []
for i in range(len(test_y)):
    test_y8 = np.append(test_y8, int(test_y[i] / 8))
    test_y4 = np.append(test_y4, int(test_y[i] % 8 / 4))
    test_y2 = np.append(test_y2, int(test_y[i] % 4 / 2))
    test_y1 = np.append(test_y1, int(test_y[i] % 2))

# -----    Data Process    -----
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1)
svm.fit(train_x, train_y8)
predict8 = svm.predict(test_x)  # separate y8 1from0

svm.fit(train_x, train_y4)
predict4 = svm.predict(test_x)  # separate y4 1from0

svm.fit(train_x, train_y2)
predict2 = svm.predict(test_x)  # separate y2 1from0

svm.fit(train_x, train_y1)
predict1 = svm.predict(test_x)  # separate y1 1from0

print(predict8.shape)
print(sum(abs(predict8-test_y8)))
print(sum(abs(predict4-test_y4)))
print(sum(abs(predict2-test_y2)))
print(sum(abs(predict1-test_y1)))


# -----    END    -----
time_end = datetime.now()
print('Total time consumed:', time_end - time_start)

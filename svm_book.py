import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def load_data():
    dataset = pd.read_csv('data.csv', usecols=['combined_shot_type', 'minutes_remaining', 'seconds_remaining',
                                               'shot_type', 'shot_zone_area', 'shot_made_flag'])
    return dataset


def convert2onehot(d2onehot):
    # covert data to onehot representation
    return pd.get_dummies(d2onehot)


# Load data
data_df = load_data().dropna()
data_df['time_remaining'] = 60*data_df.loc[:, 'minutes_remaining']+data_df.loc[:, 'seconds_remaining']
data_df = data_df.drop('minutes_remaining', axis=1)
data_df = data_df.drop('seconds_remaining', axis=1)
# print(data_df.head(5))
data = convert2onehot(data_df)
print('Onehot data:\n', data[:5])
print("Num of data: ", data.shape, "\n")
for name in data_df.keys():
    print(name, pd.unique(data_df[name]))

# Preprocessing & separate training sets
data = data.values.astype(np.float32)
np.random.shuffle(data)
sep = int(0.7*len(data))
train_x = data[:sep, 1:]
train_y = 2*data[:sep, 0:1]-1
test_x = data[sep:, 1:]
test_y = 2*data[sep:, 0:1]-1
# x_combined = np.vstack((train_x, test_x))
# y_combined = np.vstack((train_y, test_y))
print('Train_x[1,:]:', train_x[1:10, :])
print('Train_y[1]:', train_y[1:10])
print('Train_y.shape: ', train_y.shape)

# batch processing
batch_size = len(train_x)

# build svm
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1)
svm.fit(train_x, train_y)

# predict & calculate acc
pred = svm.predict(train_x)
print('pred: ', pred)
error = pred-np.transpose(train_y)
print('error: ', error[:10])
count = 0
for i in error.flat:
    if i == 0.:
        count = count+1
    # print(i)
print('acc_n: ', count)
print('acc: ', count/len(pred))

# print(svm.n_support_)
# print(svm.support_)
# print(svm.support_vectors_)

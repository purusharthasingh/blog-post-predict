import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import seed
import os

seed(1)
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import tensorflow

tensorflow.random.set_seed(1)
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.utils import plot_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.vis_utils import plot_model

from sklearn import metrics


df = pd.read_csv('blogData_Train.csv', header=None)

train_x = df.loc[:, :280]
train_y = df.loc[:,280]

test_data_list = []
for file in os.listdir('BlogPostData'):
    if 'test' in file:
        df = pd.read_csv('BlogPostData/'+file, header=None)
        test_data_list.append(df)
test_data = pd.concat(test_data_list, axis=0, ignore_index=True)

test_x = test_data.loc[:, :280]
test_y = test_data.loc[:,280]

normalizer = preprocessing.Normalization()

# adapt to the data
normalizer.adapt(np.array(train_x))
print(normalizer.mean.numpy())

loss = keras.losses.MeanSquaredError() # MeanSquaredError
optim = keras.optimizers.Adam(learning_rate=0.001)

model = Sequential([])
model.add(Dense(128, input_dim=281, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()


plot_model(
    model, to_file='model.png', show_shapes=False, show_dtype=False,
    show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
)

history = model.fit(x=train_x, y=train_y, batch_size=100, epochs=10, verbose=1, validation_split=0.1, shuffle=True, use_multiprocessing=True)

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 100])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    plt.show()
plot_loss(history)

model.evaluate(test_x, test_y, verbose=1)
predicted_y = model.predict(test_x)

print ("Mean Squared Error : ",metrics.mean_squared_error(test_y,predicted_y))
print ("R2 score : ",metrics.r2_score(test_y,predicted_y))

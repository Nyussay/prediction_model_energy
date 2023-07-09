import math
import numpy as np
import pandas as pd
import tensorflow as tf
from pylab import rcParams
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from flask import Flask, jsonify

app = Flask(__name__)

def fixData(df):
    df = df.drop(df.index[[0, 1, 2]])
    df = df.drop('MW', axis=1)
    df.columns = ['Date', 'From', 'to', 'MW']

    dates = df['Date'].tolist()

    days = []
    months = []
    years = []
    for date in dates:
        days.append(date.split('.')[0])
        months.append(date.split('.')[1])
        years.append(date.split('.')[2])

    df['Day'] = days
    df['Month'] = months
    df['Year'] = years

    def fixMW(cols):
        TT = cols
        TT = TT.replace('.', '')
        TT = TT.replace(',', '.')
        return TT

    df['MW'] = df['MW'].apply(fixMW).astype(float)
    df['Day'] = df['Day'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Year'] = df['Year'].astype(int)

    df = df.groupby(['Year', 'Month', 'Day'])['MW'].sum()
    df = pd.DataFrame(df)
    df['Month'] = df.index.get_level_values('Month')
    df['Day'] = df.index.get_level_values('Day')
    df['Year'] = df.index.get_level_values('Year')
    df.reset_index(drop=True, inplace=True)
    return df

def split1(df, train_size=0.75):
    split_index = int(len(df) * train_size)
    train = df[:split_index]
    test = df[split_index:]
    return train, test


def split2(df, look_back=7):
    dataX, dataY = [], []
    for i in range(len(df)-look_back-1):
        a = df.iloc[i:(i+look_back), 0].values
        dataX.append(a)
        dataY.append(df.iloc[i + look_back, 0])
    return np.array(dataX), np.array(dataY)



def Xplot(hist):
    if 'loss' in hist.history and 'val_loss' in hist.history:
        train_loss = hist.history['loss']
        val_loss = hist.history['val_loss']

        print("Train Loss:")
        for epoch, loss in enumerate(train_loss, start=1):
            print(f"Epoch {epoch}: {loss:.2f}")

        print("Validation Loss:")
        for epoch, loss in enumerate(val_loss, start=1):
            print(f"Epoch {epoch}: {loss:.2f}")
    else:
        print("Missing loss values in history dictionary.")






def visualiser(df):
    f, axarr = plt.subplots(1, 1, figsize=(20, 5))
    df = df.drop('Month', axis=1)
    df = df.drop('Day', axis=1)
    df = df.drop('Year', axis=1)
    df = df.values
    df = df.astype('float32')
    plt.plot(df)
    plt.show()
    return df

# Load and preprocess the data
print("Starting data preprocessing...")
Data1 = pd.read_csv('https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2010.csv', sep=';')
Data2 = pd.read_csv('https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2011.csv', sep=';')
Data3 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2012.csv', sep=';')
Data4 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2013.csv', sep=';')
Data5 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2014.csv', sep=';')
Data6 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2015.csv', sep=';')
Data7 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2016.csv', sep=';')
Data8 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2017.csv', sep=';')
Data9 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2018.csv', sep=';')
Data10 = pd.read_csv(
    'https://ws.50hertz.com/web02/api/PhotovoltaicActual/DownloadFile?fileName=2019.csv', sep=';')

# Load the remaining data files

Data1 = fixData(Data1)
Data2 = fixData(Data2)
Data3 = fixData(Data3)
Data4 = fixData(Data4)
Data5 = fixData(Data5)
Data6 = fixData(Data6)
Data7 = fixData(Data7)
Data8 = fixData(Data8)
Data9 = fixData(Data9)
Data10 = fixData(Data10)

print("Data preprocessing completed.")
# Apply the fixData function to the remaining Data variables

print("Concatenating the Data variables into a single DataFrame...")
Data = pd.concat([Data1, Data2, Data3, Data4, Data5, Data6,
                 Data7, Data8, Data9, Data10], ignore_index=True)
print("Data concatenation completed.")

print("Visualizing the Data...")
Data0 = Data

scaler = MinMaxScaler(feature_range=(0, 1))
Data0 = scaler.fit_transform(Data0)
print("Data visualization and scaling completed.")

print("Splitting the Data into train and test sets...")
train, test = split1(Data)
print("Data splitting completed.")

print("Splitting the train and test sets into input and output...")
trainX, trainY = split2(train)
testX, testY = split2(test)
print("Input and output splitting completed.")

print("Reshaping the input data...")
trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], 1)
testX = testX.reshape(testX.shape[0], testX.shape[1], 1)
print("Reshaping completed.")


# Build and train the model
print("Starting model training...")
model = Sequential()
model.add(GRU(units=8, input_shape=(trainX.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))

model.add(GRU(units=16))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='nadam', loss='mse')

model.summary()

print("Training the model...")
history = model.fit(trainX, trainY, validation_split=0.2,
                    epochs=100, batch_size=8, verbose=1)
print("Training completed.")

print(history.history)
Xplot(history)



print("Model training completed.")

print("Performing predictions...")

# Perform predictions
trainPredict= model.predict(trainX)
testPredict = model.predict(testX)

# Inverse scale the predictions
scaler2 = MinMaxScaler()
scaler2.min_, scaler2.scale_ = scaler.min_[0], scaler.scale_[0]

# Set the scaler2.min_ and scaler2.scale_ values based on the scaler used in fixData function

trainPredict = scaler2.inverse_transform(trainPredict)
trainYp = scaler2.inverse_transform([trainY])
testPredict = scaler2.inverse_transform(testPredict)
testYp = scaler2.inverse_transform([testY])

print("Predictions completed.")

rcParams['figure.figsize'] = 20, 5

print("Calculating root mean squared error...")
trainScore = math.sqrt(mean_squared_error(trainYp[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testYp[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

print("Shifting train predictions for plotting...")
trainPredictPlot = np.empty_like(Data0)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[7:len(trainPredict)+7, :] = trainPredict

print("Shifting test predictions for plotting...")
testPredictPlot = np.empty_like(Data0)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(7*2)+1:len(Data0)-1, :] = testPredict

print('Train predictions:')
print(trainPredictPlot)

print('Test predictions:')
print(testPredictPlot)

print("Plotting baseline and predictions...")
plt.plot(scaler.inverse_transform(Data0), color='y')
plt.plot(trainPredictPlot, color='b')
plt.plot(testPredictPlot, color='r')
plt.savefig('prediction_plot.png')  # Save the plot as an image file




# Flask routes
@app.route('/prediction/monthly')
def get_monthly_prediction():

    column_names = ['MW Total']
    df = pd.DataFrame(trainPredict, columns=column_names)

    high_production_days = df.nlargest(5, 'MW Total')['MW Total'].index.tolist()
    low_production_days = df.nsmallest(5, 'MW Total')['MW Total'].index.tolist()

    response = {
        'high_production_days': high_production_days,
        'low_production_days': low_production_days
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()

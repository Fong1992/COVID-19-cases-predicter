import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

#### Read Data ####
date = datetime.datetime.now()
url = 'https://covid.ourworldindata.org/data/ecdc/total_cases.csv'
data = pd.read_csv(url)
data = data[['World']]
x = np.arange(len(data['World'])).reshape(-1, 1)
y = np.array(data['World']).reshape(-1, 1)

#### Training ####
polyFeat = PolynomialFeatures(degree=3)
x = polyFeat.fit_transform(x)
print('-' * 45)
print('Training Data:')
model = linear_model.LinearRegression()
model.fit(x, y)
acc = model.score(x, y)
print(f'Accuracy: {round(acc * 100, 3)}%')

#### Prediction ####
print('-' * 45)
days = 5
pred_days = date + datetime.timedelta(days)
new_x = np.arange(days + len(x)).reshape(-1, 1)
new_x = polyFeat.fit_transform(new_x)
new_y = model.predict(new_x)
print(f'Today:')
print(f'Covid-19 cases [{round(int(y[-1]) / 1000000, 2)}] million on {date.strftime("%Y / %m / %d")}')
print('-' * 45)
print(f'Prediction on date: ')
print(f'Covid-19 cases [{round(int(new_y[-1]) / 1000000, 2)}] million on {pred_days.strftime("%Y / %m / %d")} ')

#### Show plot ####
plt.plot(y, '--b', label='World Covid-19 cases')
plt.plot(new_y, '--r', label='Prediction Covid-19 cases')
plt.legend(loc='upper left')
plt.title('Covid-19 cases curve')
plt.show()

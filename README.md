# Ex.No: 07                                       AUTO REGRESSIVE MODEL
### Date: 04.10.2025
### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

data = pd.read_csv('gold.csv', parse_dates=['Date'], index_col='Date')
result = adfuller(data['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

x = int(0.6 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

lag_order = 25
model = AutoReg(train_data['Close'], lags=lag_order)
model_fit = model.fit()

plot_acf(data['Close'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()

plot_pacf(data['Close'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
mse = mean_squared_error(test_data['Close'], predictions)
print('Mean Squared Error (MSE):', mse)

plt.plot(test_data['Close'], label='Test Data - Gold Close Price')
plt.plot(predictions, label='Predictions - Gold Close Price', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Gold Price')
plt.title('AR Model Predictions vs Test Data (Gold)')
plt.legend()
plt.grid()
plt.show()
```
### OUTPUT:
GIVEN DATA
<img width="682" height="618" alt="1 given data" src="https://github.com/user-attachments/assets/6ae061e2-b6c1-4253-ad29-4df4063759b0" />

ADF test result
<img width="375" height="47" alt="2 ADF " src="https://github.com/user-attachments/assets/5ca8f171-b160-499e-80c5-1fe000f51bcc" />

ACF
<img width="802" height="518" alt="3 ACF" src="https://github.com/user-attachments/assets/828dc2fa-69ad-469c-a41d-47fd96b3fc9b" />

PACF
<img width="756" height="556" alt="4 PACF" src="https://github.com/user-attachments/assets/3bcb3d62-cd72-412a-be72-8872772103e0" />

MSE

<img width="452" height="27" alt="5 MSE" src="https://github.com/user-attachments/assets/7a5f9596-0f65-493e-9c73-0575b10f736a" />

Prediction vs test data
<img width="773" height="557" alt="6 Prediction" src="https://github.com/user-attachments/assets/6dbe76e2-a8cf-45fe-9418-7cc1f406bf5f" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.

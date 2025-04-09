# Time Series Forecasting with XGBoost

This project demonstrates how to use **XGBoost**, to forecast energy consumption from time series data.

---

## 📌 Project Overview
- **Goal:** Predict energy consumption (in megawatts) using historical data.
- **Data:** PJME energy consumption dataset.
- **Approach:**
  - Time series preprocessing
  - Feature engineering from datetime index
  - Modeling using XGBoost Regressor
  - Visualization of predictions

---

## 📂 Project Structure
```
.
├── megawatt_energy_consumption.csv     <- Dataset
├── timeseries_with_xgboost.ipynb       <- Main notebook for analysis
├── readme.md                           <- Project documentation (this file)
```

---

## 🛠️ Tools & Libraries
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `xgboost` - Machine Learning model
- `scikit-learn` - Evaluation metric

---

## 📈 Time Series Patterns Observed
1. Exponential Trend
2. Linear Trend
3. Seasonal Pattern
4. Seasonal Pattern with Linear Growth

---

## 🔧 Data Preparation
### 1. Load Dataset
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv("megawatt_energy_consumption.csv")
```

### 2. Set Datetime as Index
```python
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)
df.plot(style='.', figsize=(15,8), color=color_pal[0], title="Energy Consumption (MW)")
plt.show()
```

### 3. Train/Test Split
```python
train = df.loc[df.index < '2015-01-01']
test  = df.loc[df.index >= '2015-01-01']
```

---

## 🧠 Feature Engineering
Create time-based features:
```python
def create_features(df):
    df = df.copy()
    df['hour']      = df.index.hour
    df['dayofweek'] = df.index.day_of_week
    df['quarter']   = df.index.quarter
    df['month']     = df.index.month
    df['year']      = df.index.year
    df['dayofyear'] = df.index.day_of_year
    return df

train = create_features(train)
test = create_features(test)
```

### 📊 Feature Relationships
Visualizing relationship between features and target:
```python
sns.boxplot(data=df, x='hour', y='PJME_MW')
sns.boxplot(data=df, x='dayofweek', y='PJME_MW')
sns.boxplot(data=df, x='month', y='PJME_MW')
sns.boxplot(data=df, x='year', y='PJME_MW')
```

---

## 🧪 Modeling with XGBoost
```python
features = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear']
target = 'PJME_MW'

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

reg = XGBRegressor(n_estimators=1000, early_stopping_rounds=50, learning_rate=0.01)
reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)
```

---

## 📌 Feature Importance
```python
fi = pd.DataFrame(data=reg.feature_importances_, index=reg.feature_names_in_, columns=['importance'])
fi.sort_values('importance', ascending=False).plot(kind='barh', title='Feature Importance')
plt.show()
```

---

## 📉 Forecasting
```python
test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)

ax = df[['PJME_MW']].plot(figsize=(25,10))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Real Value', 'Prediction Value'])
plt.title('Real Data vs Prediction')
plt.show()
```

### 📅 Weekly Zoom In
```python
ax = df.loc[(df.index > '2018-04-01') & (df.index < '2018-04-08')]['PJME_MW'].plot(figsize=(15,5), title='Week Of Data')
df.loc[(df.index > '2018-04-01') & (df.index < '2018-04-08')]['prediction'].plot(style='.')
```

---

## 🧾 Evaluation
```python
from sklearn.metrics import mean_squared_error
score = mean_squared_error(test['PJME_MW'], test['prediction'])
print(f"MSE Score: {score:.2f}")

# Error by Date
test['error'] = np.abs(test[target] - test['prediction'])
test['date'] = test.index.date
test.groupby('date')['error'].mean().sort_values(ascending=False).head(20)
test.groupby('date')['error'].mean().sort_values().head(20)
```

---

## ✅ Summary
- Built a complete time series forecasting pipeline using **XGBoost**.
- Extracted rich time-based features from datetime index.
- Visualized results and evaluated using MSE.

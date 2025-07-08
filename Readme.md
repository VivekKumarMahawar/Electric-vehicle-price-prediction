# Electric Vehicle Price Prediction 🚗🔋

## 📊 **Project Overview**

This project analyses the **Electric Vehicle Population Data** to perform:

- **Exploratory Data Analysis (EDA)** to derive insights on EV adoption, make, model, electric range, and price distribution.
- **Regression modelling** to predict **Base MSRP (vehicle price)** using machine learning algorithms.

---

## 🔧 **Features Implemented**

- Data Cleaning & Preprocessing
- Outlier Removal using IQR method
- Encoding Categorical Features (Label Encoding & One-Hot Encoding)
- Feature Scaling using StandardScaler
- Exploratory Data Analysis (EDA) with seaborn and matplotlib
- Regression Models:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor
- Model Evaluation using RMSE and R² score
- Feature Importance Analysis

---

## 🗂️ **Dataset**

- **Source:** [Washington State Open Data](https://data.wa.gov/) *(Include exact dataset URL if sharing publicly)*
- **Description:** Contains details of electric vehicles including make, model, vehicle type (BEV/PHEV), electric range, and Base MSRP.

---

## 💻 **Libraries Used**

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
```

---

## 📈 **Results Summary**

| Model             | Validation RMSE | Validation R² |
| ----------------- | --------------- | ------------- |
| Linear Regression | \~13,500        | \~0.77        |
| Random Forest     | \~9,100         | \~0.89        |
| XGBoost           | \~9,400         | \~0.89        |

✅ **Random Forest Regressor** achieved the **highest accuracy** with a **validation R² of \~0.89** and lowest RMSE, indicating it as the best model for predicting EV prices.

---

## 🚀 **Future Scope**

- Include more features like battery capacity and incentives to improve accuracy.
- Perform classification task to categorise vehicle type (BEV/PHEV).
- Deploy models as a web application for real-time price prediction.
- Expand analysis with global datasets for broader applicability.

---

## ✨ **Author**

**Vivek Kumar Mahawar**

---

> **Note:** Replace dataset URL with your exact link if making the repository public.

---

### 📌 **How to Run**

1. Clone this repository
2. Install dependencies using `pip install -r requirements.txt`
3. Open `Electric_Vehicle_Population_Analysis.ipynb` in Jupyter Notebook or VS Code
4. Run all cells sequentially to view EDA and model results

---

Let me know if you want badges or markdown styling for GitHub profile aesthetic today.


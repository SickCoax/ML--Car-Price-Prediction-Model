import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error
import matplotlib.pyplot as plt

df1 = pd.read_csv(r"dataset/car_price_dataset_medium.csv")
x = df1.drop(["Car_ID" , "Price_USD"] , axis=1)
y = df1["Price_USD"]

num_cols = [ 'Model_Year', 'Kilometers_Driven', 'Engine_CC', 'Max_Power_bhp', 'Mileage_kmpl', 'Seats' ]
cat_cols = ['Brand', 'Fuel_Type', 'Transmission','Owner_Type' ]

xtrain , xtest , ytrain , ytest = train_test_split(x , y , test_size=0.2 , random_state=42)

preprocess = ColumnTransformer([("num" , StandardScaler() , num_cols) , ("cat" , OneHotEncoder(drop="first" , handle_unknown="ignore") , cat_cols)])

pipeline = Pipeline([("preprocess" , preprocess) , ("linear" , LinearRegression())])

model = pipeline.fit(xtrain , ytrain)
ypred = model.predict(xtest)
residual = ytest - ypred


# --------------------------
# Evaluation Metric
# --------------------------
print("R2 Score:", r2_score(ytest, ypred))
print("MAE:", mean_absolute_error(ytest, ypred))
print("RMSE:", np.sqrt(mean_squared_error(ytest, ypred)))

# --------------------------
# Residual Plot
# --------------------------
plt.scatter(ypred , residual)
plt.axhline(y=0 , color="red" , linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.grid()
plt.show()


# --------------------------
# Conclusion :
# Residual Plot - Model is not missing a strong nonlinear relationship -- No need to use Polynomial Regression
# Using ElasticNet in main file to reduce errors
# --------------------------
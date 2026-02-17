import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import r2_score , mean_absolute_error , mean_squared_error
import matplotlib.pyplot as plt

df1 = pd.read_csv("car_price_dataset_medium.csv")
x = df1.drop(["Car_ID" , "Price_USD"] , axis=1)
y = df1["Price_USD"]

num_cols = [ 'Model_Year', 'Kilometers_Driven', 'Engine_CC', 'Max_Power_bhp', 'Mileage_kmpl', 'Seats' ]
cat_cols = ['Brand', 'Fuel_Type', 'Transmission','Owner_Type' ]

xtrain , xtest , ytrain , ytest = train_test_split(x , y , test_size=0.2 , random_state=42)

preprocess = ColumnTransformer([("num" , StandardScaler() , num_cols) , ("cat" , OneHotEncoder(drop="first" , handle_unknown="ignore") , cat_cols)])

pipeline = Pipeline([("preprocess" , preprocess) , ("elastic" , ElasticNet(max_iter=10000))])


param_grid = {
    "elastic__alpha": [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1],
    "elastic__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
}


grid = GridSearchCV(pipeline , param_grid , cv=5)

grid.fit(xtrain , ytrain)

model = grid.best_estimator_
ypred = model.predict(xtest)

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
#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install lightgbm')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
from scipy.stats import randint as sp_randint, uniform


# In[3]:


sheet_W = pd.read_excel(r'C:\Users\rahul\Project\OUT policy\supply_chain_results_1.xlsx', sheet_name = 'Wholesaler')
sheet_W = sheet_W[['Replenishment quantity','Beginning inventory','Inventory position', 'Retailer order','Allocated quantity', 'Ending inventory', 'Forecast', 'Order up to level', 'Order quantity', 'Lost sales']]


# In[4]:


data = pd.DataFrame(sheet_W)


# # Correlation

# In[8]:


data.corr()


# # Variance Inflation Factors

# In[9]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif(df):
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(len(df.columns))]
    return vif_data
calculate_vif(data)


# In[10]:


p = data[['Replenishment quantity','Beginning inventory','Inventory position', 'Retailer order','Allocated quantity', 'Ending inventory', 'Forecast', 'Order up to level', 'Lost sales']]
q = data[['Order quantity']]


# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

p_train, p_test, q_train, q_test = train_test_split(p, q, test_size=0.2, random_state=42)
train_data = lgb.Dataset(p_train, label=q_train)
test_data = lgb.Dataset(p_test, label=q_test)
params = {
    'objective': 'regression',
    'metric': '12',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])
q_pred = bst.predict(p_test, num_iteration=bst.best_iteration)
accuracy = accuracy_score(q_test, q_pred.round())
print("Accuracy:", accuracy)


# In[12]:


x = data[['Replenishment quantity','Beginning inventory','Inventory position', 'Retailer order','Allocated quantity', 'Ending inventory', 'Forecast', 'Order up to level', 'Lost sales']]
y = data[['Order quantity']]


# # Splitting the dataset into training set and test set

# In[13]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
print(x_train.shape)
print(x_test.shape)


# # Hyperparameter tuning

# In[15]:


from sklearn.model_selection import RandomizedSearchCV
param_dist = {
    'num_leaves': [20, 31, 50],
    'learning_rate': [0.02, 0.025, 0.03, 0.035, 0.05,0.1],
    'n_estimators': [30,51,70,100],
    'subsample': [0.8, 0.96, 1.0,1.5],
    'colsample_bytree': [0.3, 0.8, 1.0, 2],
    'reg_alpha': [0,1,2],
    'reg_lambda': [0,1,2]
}
lgb_regressor = lgb.LGBMRegressor()
random_search = RandomizedSearchCV(estimator=lgb_regressor, param_distributions=param_dist, n_iter=50, cv=2, scoring='neg_mean_squared_error', random_state=42)
random_search.fit(x_train, y_train)
print("Best parameters:", random_search.best_params_)


# In[16]:


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
x = data[['Replenishment quantity','Beginning inventory','Inventory position', 'Retailer order','Allocated quantity', 'Ending inventory', 'Forecast', 'Order up to level', 'Lost sales']]
y = data[['Order quantity']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

best_params = {
    'num_leaves': 20,
    'learning_rate': 0.03,
    'n_estimators': 51,
    'subsample': 0.96,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 2,
}
# Build the LightGBM model using the best parameters
final_model = lgb.LGBMRegressor(**best_params)

# Train the final model
final_model.fit(x_train, y_train)

# Perform pruning based on feature importance
# Get feature importances
feature_importances = final_model.feature_importances_

# Determine the threshold for feature importance
threshold_percentage = 0.05  # For example, choose 5%
threshold = max(feature_importances) * threshold_percentage

# Prune less important features
pruned_features = x.columns[feature_importances >= threshold]

# Filter the dataset to keep only the pruned features
x_train_pruned = x_train[pruned_features]
x_test_pruned = x_test[pruned_features]


# Re-train the model using pruned features
final_model_pruned = lgb.LGBMRegressor(**best_params)
final_model_pruned.fit(x_train_pruned, y_train)

# Make predictions on the test set using the pruned model
y_pred_pruned = final_model_pruned.predict(x_test_pruned)

# Evaluate the pruned model
mse_pruned = mean_squared_error(y_test, y_pred_pruned)
rmse = np.sqrt(mse_pruned)
r2 = r2_score(y_test, y_pred_pruned)
print("Pruned Model root Mean Squared Error:", rmse)
print("r2:", r2)
n = len(y_test)
p = len(pruned_features)
print(n,p)
adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print("Adjusted R-squared: ", adj_r2)


# In[17]:


print("Features used for prediction after pruning:", pruned_features)


# In[18]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score

x = data[['Replenishment quantity', 'Beginning inventory', 'Inventory position', 'Forecast', 'Order up to level']]
y = data[['Order quantity']]


r2_scores = cross_val_score(final_model_pruned, x, y, cv=10, scoring='r2')
mean_r2 = np.mean(r2_scores)

rmse_scores = cross_val_score(final_model_pruned, x, y, cv=10, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-rmse_scores)
mean_rmse = np.mean(rmse_scores)

mae_scores = cross_val_score(final_model_pruned, x, y, cv=10, scoring='neg_mean_absolute_error')
mae_scores = -mae_scores  
mean_mae = np.mean(mae_scores)

rmse_std = np.std(rmse_scores)
mae_std = np.std(mae_scores)

print("\nCross-validated R^2 scores:")
print(r2_scores)
print(f"\nMean R^2 Score: {mean_r2}")

print("Cross-validated RMSE scores:")
print(rmse_scores)
print(f"\nMean RMSE Score: {mean_rmse}")

print("\nCross-validated MAE scores:")
print(mae_scores)
print(f"\nMean MAE Score: {mean_mae}")

print("Standard deviation of RMSE scores:", rmse_std)
print("Standard deviation of MAE scores:", mae_std)


# In[22]:


y_pred_actual = scaler_y.inverse_transform(y_pred_pruned.reshape(-1, 1))
print("Actual Predicted Values:")
print(y_pred_actual.shape)
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
print(y_test_actual.shape)


# In[23]:


plt.scatter(y_test_actual[0:200], y_pred_actual[0:200], alpha=1)
plt.plot([min(y_test_actual[0:200]), max(y_test_actual[0:200])], [min(y_test_actual[0:200]), max(y_test_actual[0:200])], '-', color='red')
plt.title('Predicted vs. Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')


# In[19]:


import pickle

# Assuming 'trained_model' is your trained model object
with open('W_LGB_final_model_pruned.pkl', 'wb') as file:
    pickle.dump(final_model_pruned, file)


# In[20]:


import pickle

# Load the trained model from the pickle file
with open('W_LGB_final_model_pruned.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# # Making a single value prediction

# In[22]:


import math
input_data = pd.DataFrame({'Replenishment quantity': [18],'Beginning inventory': [36],'Inventory position': [36],'Retailer order': [20], 'Forecast': [17],'Order up to level': [51]})
print(math.ceil(scaler_y.inverse_transform(loaded_model.predict(scaler_x.transform(input_data)).reshape(-1,1))))


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import plotly.express as px


# In[2]:


sheet_R = pd.read_excel(r'C:\Users\rahul\Project\Variant of OUT Policy\supply_chain_results_2.xlsx', sheet_name = 'Factory')
sheet_R = sheet_R[['Replenishment quantity','Beginning inventory','Inventory position', 'Distributor order','Allocated quantity', 'Ending inventory', 'Forecast', 'Order up to level', 'EDR', 'Order quantity', 'Lost sales']]


# In[3]:


data = pd.DataFrame(sheet_R)


# In[4]:


from sklearn.model_selection import train_test_split
x = data[['Replenishment quantity','Beginning inventory','Inventory position', 'Distributor order','Allocated quantity', 'Ending inventory', 'Forecast', 'Order up to level', 'EDR', 'Lost sales']]
y = data[['Order quantity']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[5]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Define the hyperparameter grid for RandomizedSearchCV
param_distributions = {
    'n_estimators': [50, 100, 150, 200],  # Number of boosting rounds
    'learning_rate': uniform(loc=0.01, scale=1.0),  # Learning rate
    'loss': ['linear', 'square', 'exponential']  # Loss function
}

# Initialize AdaBoostRegressor
adaboost = AdaBoostRegressor(random_state=42)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(adaboost, param_distributions, n_iter=10, scoring='neg_mean_squared_error', cv=2, random_state=42)

# Perform RandomizedSearchCV
random_search.fit(x_train, y_train)

# Get the best model
best_adaboost = random_search.best_estimator_

# Print the best parameters
print("Best Parameters:", random_search.best_params_)


# In[7]:


from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

x = data[['Replenishment quantity','Beginning inventory','Inventory position', 'Distributor order','Allocated quantity', 'Ending inventory', 'Forecast', 'Order up to level', 'EDR', 'Lost sales']]
y = data[['Order quantity']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

best_params = {
    'n_estimators': 200,  # Number of boosting rounds
    'learning_rate': 0.068,  # Learning rate
    'loss': 'linear'
}

# Use AdaBoostRegressor instead of XGBRegressor
final_model = AdaBoostRegressor(**best_params)

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
final_model_pruned = AdaBoostRegressor(**best_params)
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


# In[8]:


print("Features used for prediction after pruning:", pruned_features)


# In[9]:


x = data[['Inventory position', 'Forecast', 'Order up to level', 'EDR']]
y = data[['Order quantity']]

from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score


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


# In[9]:


y_pred_actual = scaler_y.inverse_transform(y_pred_pruned.reshape(-1, 1))
print("Actual Predicted Values:")
print(y_pred_actual.shape)
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
print(y_test_actual.shape)


# In[10]:


plt.scatter(y_test_actual[0:200], y_pred_actual[0:200], alpha=1)
plt.plot([min(y_test_actual[0:200]), max(y_test_actual[0:200])], [min(y_test_actual[0:200]), max(y_test_actual[0:200])], '-', color='red')
plt.title('Predicted vs. Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')


# In[10]:


import pickle

# Assuming 'trained_model' is your trained model object
with open('F_ADB_final_model_pruned.pkl', 'wb') as file:
    pickle.dump(final_model_pruned, file)


# In[11]:


import pickle

# Load the trained model from the pickle file
with open('F_ADB_final_model_pruned.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# In[12]:


import math
input_data = pd.DataFrame({'Beginning inventory': [36],'Inventory position': [36], 'Forecast': [17], 'Order up to level': [51], 'EDR': [17]})
print(math.ceil(scaler_y.inverse_transform(loaded_model.predict(scaler_x.transform(input_data)).reshape(-1,1))))


# In[13]:


data.shape


# In[ ]:





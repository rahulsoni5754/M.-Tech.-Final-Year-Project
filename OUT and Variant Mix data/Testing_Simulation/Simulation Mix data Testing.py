#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import math
import pickle

with open(r'C:\Users\rahul\Project\OUT and Variant mix data\Retailer Stage\R_LGB_final_model_pruned.pkl', 'rb') as file:
    retailer_model = pickle.load(file)
    
with open(r'C:\Users\rahul\Project\OUT and Variant mix data\Wholesaler stage\W_LGB_final_model_pruned.pkl', 'rb') as file:
    wholesaler_model = pickle.load(file)

with open(r'C:\Users\rahul\Project\OUT and Variant mix data\Distributor Stage\D_LGB_final_model_pruned.pkl', 'rb') as file:
    distributor_model = pickle.load(file)

with open(r'C:\Users\rahul\Project\OUT and Variant mix data\Factory Stage\F_LGB_final_model_pruned.pkl', 'rb') as file:
    factory_model = pickle.load(file)


Maxit = 45
t = 0

data_retailer = {'Period': [],'Replenishment quantity': [],'Beginning inventory': [],'Inventory position': [],'Customer order': [],
                 'Allocated quantity': [],'Ending inventory': [],'Forecast': [],'Order up to level': [],'EDR': [],'Order quantity': [],'Lost sales': []}

data_wholesaler = {'Period': [],'Replenishment quantity': [],'Beginning inventory': [],'Inventory position': [],'Retailer order': [],
                   'Allocated quantity': [],'Ending inventory': [],'Forecast': [],'Order up to level': [],'EDR': [],'Order quantity': [],'Lost sales': []}

data_distributor = {'Period': [],'Replenishment quantity': [],'Beginning inventory': [],'Inventory position': [],'Wholesaler order': [],
                    'Allocated quantity': [],'Ending inventory': [],'Forecast': [],'Order up to level': [],'EDR': [],'Order quantity': [],'Lost sales': []}

data_factory = {'Period': [],'Replenishment quantity': [],'Beginning inventory': [],'Inventory position': [],'Distributor order': [],
                'Allocated quantity': [],'Ending inventory': [],'Forecast': [],'Order up to level': [],'EDR': [],'Order quantity': [],'Lost sales': []}

demand_values = [10, 18, 18, 19, 15, 12, 22, 18, 29, 31, 12, 20, 14, 20, 16, 25, 22, 21, 20, 21, 21, 21, 21, 28, 21, 21, 27, 21, 19, 21, 16, 25, 15, 12, 8, 18, 17, 20, 22, 15, 14, 14, 28, 17, 25, 17]


if t == 0:
    Retailer_ending_inventory = 60
    Retailer_order_quantity = 0
    Wholesaler_allocated_quantity = 0
    Wholesaler_ending_inventory = 60
    Wholesaler_order_quantity = 0
    Distributor_allocated_quantity = 0
    Distributor_ending_inventory = 60
    Distributor_order_quantity = 0
    Factory_allocated_quantity = 0
    Factory_ending_inventory = 60
    Factory_order_quantity = 0

while t <= Maxit:
    t += 1
    Demand_t = demand_values[t-1]
    
    # Retailer Stage
    Retailer_replenishment_quantity = Wholesaler_allocated_quantity
    Retailer_beginning_inventory = Retailer_ending_inventory + Retailer_replenishment_quantity
    Retailer_inventory_position = Retailer_beginning_inventory + Retailer_order_quantity
    Customer_order = Demand_t
    Retailer_allocated_quantity = min(Retailer_beginning_inventory, Customer_order)
    Retailer_ending_inventory = Retailer_beginning_inventory - Retailer_allocated_quantity
    Retailer_forecast = 20 if t == 1 else (data_retailer['Customer order'][0] if t == 2 else math.ceil((data_retailer['Customer order'][t-2] + data_retailer['Customer order'][t-3]) / 2))
    Retailer_order_up_to_level = 3 * Retailer_forecast
    Retailer_EDR = 1 * Retailer_forecast
    Retailer_features = pd.DataFrame([[Retailer_replenishment_quantity, Retailer_beginning_inventory, Retailer_inventory_position, Retailer_forecast, Retailer_order_up_to_level, Retailer_EDR]], columns=['Retailer_replenishment_quantity', 'Retailer_beginning_inventory','Retailer_inventory_position', 'Retailer_forecast', 'Retailer_order_up_to_level', 'Retailer_EDR'])
    Retailer_order_quantity = math.ceil(retailer_model.predict(Retailer_features))
    Retailer_lost_sales = Customer_order - Retailer_allocated_quantity
    
    # Append data to the Retailer dictionary
    data_retailer['Period'].append(t)
    data_retailer['Replenishment quantity'].append(Retailer_replenishment_quantity)
    data_retailer['Beginning inventory'].append(Retailer_beginning_inventory)
    data_retailer['Inventory position'].append(Retailer_inventory_position)
    data_retailer['Customer order'].append(Customer_order)
    data_retailer['Allocated quantity'].append(Retailer_allocated_quantity)
    data_retailer['Ending inventory'].append(Retailer_ending_inventory)
    data_retailer['Forecast'].append(Retailer_forecast)
    data_retailer['Order up to level'].append(Retailer_order_up_to_level)
    data_retailer['EDR'].append(Retailer_EDR)
    data_retailer['Order quantity'].append(Retailer_order_quantity)
    data_retailer['Lost sales'].append(Retailer_lost_sales)

    # Wholesaler Stage
    Wholesaler_replenishment_quantity = Distributor_allocated_quantity
    Wholesaler_beginning_inventory = Wholesaler_ending_inventory + Wholesaler_replenishment_quantity
    Wholesaler_inventory_position = Wholesaler_beginning_inventory + Wholesaler_order_quantity
    Retailer_order = 0 if t == 1 else (data_retailer['Order quantity'][t-2])
    Wholesaler_allocated_quantity = min(Wholesaler_beginning_inventory, Retailer_order)
    Wholesaler_ending_inventory = Wholesaler_beginning_inventory - Wholesaler_allocated_quantity
    Wholesaler_forecast = 20 if t == 1 else (data_wholesaler['Retailer order'][0] if t == 2 else math.ceil((data_wholesaler['Retailer order'][t-2] + data_wholesaler['Retailer order'][t-3]) / 2))
    Wholesaler_order_up_to_level = 3 * Wholesaler_forecast
    Wholesaler_EDR = 1 * Wholesaler_forecast
    Wholesaler_features = pd.DataFrame([[Wholesaler_inventory_position, Wholesaler_forecast, Wholesaler_EDR]], columns=['Wholesaler_inventory_position', 'Wholesaler_forecast', 'Wholesaler_EDR'])
    Wholesaler_order_quantity = math.ceil(wholesaler_model.predict(Wholesaler_features))
    Wholesaler_lost_sales = Retailer_order - Wholesaler_allocated_quantity

    # Append data to the Wholesaler dictionary
    data_wholesaler['Period'].append(t)
    data_wholesaler['Replenishment quantity'].append(Wholesaler_replenishment_quantity)
    data_wholesaler['Beginning inventory'].append(Wholesaler_beginning_inventory)
    data_wholesaler['Inventory position'].append(Wholesaler_inventory_position)
    data_wholesaler['Retailer order'].append(Retailer_order)
    data_wholesaler['Allocated quantity'].append(Wholesaler_allocated_quantity)
    data_wholesaler['Ending inventory'].append(Wholesaler_ending_inventory)
    data_wholesaler['Forecast'].append(Wholesaler_forecast)
    data_wholesaler['Order up to level'].append(Wholesaler_order_up_to_level)
    data_wholesaler['EDR'].append(Wholesaler_EDR)
    data_wholesaler['Order quantity'].append(Wholesaler_order_quantity)
    data_wholesaler['Lost sales'].append(Wholesaler_lost_sales)

    # Distributor Stage
    Distributor_replenishment_quantity = Factory_allocated_quantity
    Distributor_beginning_inventory = Distributor_ending_inventory + Distributor_replenishment_quantity
    Distributor_inventory_position = Distributor_beginning_inventory + Distributor_order_quantity
    Wholesaler_order = 0 if t == 1 else (data_wholesaler['Order quantity'][t-2])
    Distributor_allocated_quantity = min(Distributor_beginning_inventory, Wholesaler_order)
    Distributor_ending_inventory = Distributor_beginning_inventory - Distributor_allocated_quantity
    Distributor_forecast = 20 if t == 1 else (data_distributor['Wholesaler order'][0] if t == 2 else math.ceil((data_distributor['Wholesaler order'][t-2] + data_distributor['Wholesaler order'][t-3]) / 2))
    Distributor_order_up_to_level = 3 * Distributor_forecast
    Distributor_EDR = 1 * Distributor_forecast
    Distributor_features = pd.DataFrame([[Distributor_inventory_position, Distributor_forecast, Distributor_order_up_to_level, Distributor_EDR]], columns=['Distributor_inventory_position', 'Distributor_forecast', 'Distributor_order_up_to_level', 'Distributor_EDR'])
    Distributor_order_quantity = math.ceil(distributor_model.predict(Distributor_features))
    Distributor_lost_sales = Wholesaler_order - Distributor_allocated_quantity

    # Append data to the Distributor dictionary
    data_distributor['Period'].append(t)
    data_distributor['Replenishment quantity'].append(Distributor_replenishment_quantity)
    data_distributor['Beginning inventory'].append(Distributor_beginning_inventory)
    data_distributor['Inventory position'].append(Distributor_inventory_position)
    data_distributor['Wholesaler order'].append(Wholesaler_order)
    data_distributor['Allocated quantity'].append(Distributor_allocated_quantity)
    data_distributor['Ending inventory'].append(Distributor_ending_inventory)
    data_distributor['Forecast'].append(Distributor_forecast)
    data_distributor['Order up to level'].append(Distributor_order_up_to_level)
    data_distributor['EDR'].append(Distributor_EDR)
    data_distributor['Order quantity'].append(Distributor_order_quantity)
    data_distributor['Lost sales'].append(Distributor_lost_sales)

    # Factory Stage
    Factory_replenishment_quantity = data_factory['Order quantity'][t-3] if t > 2 else 0
    Factory_beginning_inventory = Factory_ending_inventory + Factory_replenishment_quantity
    Factory_inventory_position = Factory_beginning_inventory + Factory_order_quantity
    Distributor_order = 0 if t == 1 else (data_distributor['Order quantity'][t-2])
    Factory_allocated_quantity = min(Factory_beginning_inventory, Distributor_order)
    Factory_ending_inventory = Factory_beginning_inventory - Factory_allocated_quantity
    Factory_forecast = 20 if t == 1 else (data_factory['Distributor order'][0] if t == 2 else math.ceil((data_factory['Distributor order'][t-2] + data_factory['Distributor order'][t-3]) / 2))
    Factory_order_up_to_level = 3 * Factory_forecast
    Factory_EDR = 1 * Factory_forecast
    Factory_features = pd.DataFrame([[Factory_inventory_position, Factory_forecast, Factory_EDR]], columns=['Factory_inventory_position', 'Factory_forecast', 'Factory_EDR'])
    Factory_order_quantity = math.ceil(factory_model.predict(Factory_features))
    Factory_lost_sales = Distributor_order - Factory_allocated_quantity

    # Append data to the Factory dictionary
    data_factory['Period'].append(t)
    data_factory['Replenishment quantity'].append(Factory_replenishment_quantity)
    data_factory['Beginning inventory'].append(Factory_beginning_inventory)
    data_factory['Inventory position'].append(Factory_inventory_position)
    data_factory['Distributor order'].append(Distributor_order)
    data_factory['Allocated quantity'].append(Factory_allocated_quantity)
    data_factory['Ending inventory'].append(Factory_ending_inventory)
    data_factory['Forecast'].append(Factory_forecast)
    data_factory['Order up to level'].append(Factory_order_up_to_level)
    data_factory['EDR'].append(Factory_EDR)
    data_factory['Order quantity'].append(Factory_order_quantity)
    data_factory['Lost sales'].append(Factory_lost_sales)

# Create DataFrames for each stage
df_retailer = pd.DataFrame(data_retailer)
df_wholesaler = pd.DataFrame(data_wholesaler)
df_distributor = pd.DataFrame(data_distributor)
df_factory = pd.DataFrame(data_factory)

# Create an Excel writer object
with pd.ExcelWriter('mix_data_testing_result_6.xlsx') as writer:
    # Write each DataFrame to a separate sheet
    df_retailer.to_excel(writer, sheet_name='Retailer', index=False)
    df_wholesaler.to_excel(writer, sheet_name='Wholesaler', index=False)
    df_distributor.to_excel(writer, sheet_name='Distributor', index=False)
    df_factory.to_excel(writer, sheet_name='Factory', index=False)

#load the excel file    
excel_file = pd.ExcelFile('mix_data_testing_result_6.xlsx')
#print(excel_file)
retailer_sheet = excel_file.parse('Retailer')  
wholesaler_sheet = excel_file.parse('Wholesaler')
distributor_sheet = excel_file.parse('Distributor')
factory_sheet = excel_file.parse('Factory')
print("Retailer Sheet:")
print(retailer_sheet)

print("\nWholesaler Sheet:")
print(wholesaler_sheet)

print("\nDistributor Sheet:")
print(distributor_sheet)

print("\nFactory Sheet:")
print(factory_sheet)


# In[7]:


print(scaler_x_f.n_features_in_)


# In[ ]:





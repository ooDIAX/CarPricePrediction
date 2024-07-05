import streamlit as st
import pandas as pd
from joblib import load
import constants
from xgboost import XGBRegressor

def round_to_hundred(value):
    return round(value // 100) * 100

# Load your trained linear regression model
model = load('model/XGboost.joblib')  # Adjust filename as needed


# Function to predict based on user inputs
def predict_price(brand, fuel_type, transmission, model_year, milage, accident, HP, engine_volume):
    # Create a DataFrame with user inputs
    column_names = list(constants.columns.keys())
    column_types = constants.columns

    # Create the DataFrame with specified types
    df = pd.DataFrame({col: pd.Series(dtype=typ) for col, typ in column_types.items()})

    new_row = {col: False for col in constants.columns}
    new_row['model_year'] = model_year
    new_row['milage'] = milage
    new_row['accident'] = True if accident else False
    new_row['HP'] = HP
    new_row['Engine Volume (L)'] = engine_volume
    new_row[f'brand_{brand}'] = True
    new_row['transmission_True'] = True if transmission == 'Automatic' else False
    new_row['transmission_False'] = not new_row['transmission_True']
    new_row[f'fuel_type_{fuel_type}'] = True

    # Convert new_row to DataFrame and use concat to add it
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Predict using the model
    prediction = model.predict(df)
    return prediction[0]  # Assuming a single prediction is returned

st.title('Car Price Prediction')

# Dropdown for selecting Buyer or Seller
user_type = st.selectbox('Are you a Buyer or a Seller?', ('Buyer', 'Seller'))


brand = st.selectbox('Brand', [''] + constants.brand_list)
fuel_type = st.selectbox('Fuel Type', [''] + constants.fuel_list)
transmission = st.selectbox('Transmission', ['', 'Automatic', 'Manual'])
model_year = st.number_input('Model Year', min_value=1900, max_value=2024, value=2020, step=1)
milage = st.number_input('Milage', min_value=0, max_value = 300000, value=0)
accident = st.radio('Accident', ('Not specified', 'Yes', 'No'))
HP = st.number_input('Horsepower (HP)', min_value=0, format='%d', value=0)
engine_volume = st.number_input('Engine Volume (L)', min_value=0.0, format="%.1f", value=0.0)

if accident == 'Not specified':
    accident = None
elif accident == 'Yes':
    accident = True
else:
    accident = False

if brand == '':
    brand = None
if fuel_type == '':
    fuel_type = None
if transmission == '':
    transmission = None
if model_year == 0:
    model_year = None
if milage == 0:
    milage = None
if HP == 0:
    HP = None
if engine_volume == 0.0:
    engine_volume = None


# Predict button
if st.button('Predict Price'):
    if user_type == 'Seller' and not all([brand, fuel_type, transmission, model_year, milage, HP, engine_volume]):
        st.write('Sellers must fill all fields before predicting.')
    else:
        # Set default values for optional fields if empty
        if user_type == 'Buyer':
            brand1 = brand or 'Toyota'
            fuel_type1 = fuel_type or 'Gasoline'
            transmission1 = transmission or 'Automatic'
            model_year1 = model_year or 2010
            milage1 = milage or 150
            accident1 = accident or False
            HP1 = HP or 100
            engine_volume1 = engine_volume or 2.0
            
            brand2 = brand or 'Toyota'
            fuel_type2 = fuel_type or 'Gasoline'
            transmission2 = transmission or 'Automatic'
            model_year2 = model_year or 2024
            milage2 = milage or 150000
            accident2 = accident or True
            HP2 = HP or 350
            engine_volume2 = engine_volume or 2.0

            prediction1 = predict_price(brand1, fuel_type1, transmission1, model_year1, milage1, accident1, HP1, engine_volume1)
            prediction2 = predict_price(brand2, fuel_type2, transmission2, model_year2, milage2, accident2, HP2, engine_volume2)

            st.write(f'Predicted Price range: ${min(prediction1, prediction2)} - ${max(prediction1, prediction2)}')
        
        else:
            prediction = predict_price(brand, fuel_type, transmission, model_year, milage, accident, HP, engine_volume)
            st.write(f'Predicted Price: ${round_to_hundred(prediction)}')

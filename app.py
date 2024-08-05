import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Load the dataset to get the unique states
dataset = pd.read_csv('dataset/Housing_dataset_train.csv')  # Replace with actual path
unique_states = dataset['loc'].unique()

# Load the saved house price model
model = pickle.load(open('Saved_model/House Price Prediction for Pri-Ame Estate.sav', 'rb'))

# Define title encoding (update this with actual property types from your dataset)
title_encoding = {
    'Apartment': 0, 
    'Penthouse': 1, 
    'Detached House': 2, 
    'Semi-Detached House': 3, 
    'Terraced House': 4, 
    'Bungalow': 5
}

# Set page configuration
st.set_page_config(page_title="House Price Prediction",
                   layout="wide",
                   page_icon="🏡")

# Sidebar for navigation 
st.sidebar.title("Navigation")

# Main content
st.title('House Price Prediction using Machine Learning')

col1, col2, col3 = st.columns(3)

# Dropdown for Location (States from dataset)
with col1:
    Location = st.selectbox('Select State', options=unique_states, index=0)

# Dropdown for Title (Property Type)
with col2:
    Title = st.selectbox('Select Property Type', options=list(title_encoding.keys()), index=0)

# Input fields for numeric data
with col3:
    Bedroom = st.number_input('Number of Bedrooms', step=1, min_value=0)

with col1:
    Bathroom = st.number_input('Number of Bathrooms', step=1, min_value=0)

with col2:
    Parking_space = st.number_input('Number of Parking Spaces', step=1, min_value=0)

# When the predict button is clicked
if st.button('Predict House Price'):
    # Encode the categorical data to match model training preprocessing
    loc_encoded = np.where(unique_states == Location)[0][0]  # Find the index of the selected state
    title_encoded = title_encoding[Title]
    
    # Prepare user input for prediction
    user_input = {
        'loc': [loc_encoded],
        'title': [title_encoded],
        'bathroom': [Bathroom],
        'bedroom': [Bedroom],
        'parking_space': [Parking_space]
    }

    # Convert the user input into a DataFrame
    user_input_df = pd.DataFrame(user_input)
    
    # Predict house price using the loaded model
    try:
        predicted_price = model.predict(user_input_df)

        # Convert the price to Naira (assuming it's in Naira)
        predicted_price_naira = f"₦{predicted_price[0]:,.2f}"

        # Displaying the prediction result
        st.success(f'The Predicted House Price is: {predicted_price_naira}')
    except ValueError as e:
        st.error(f"Error in prediction: {e}")

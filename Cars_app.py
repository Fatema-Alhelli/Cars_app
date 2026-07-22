import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title = "Car Price Prediction", page_icon = "🚗")

## LOAD MODEL ##
@st.cache_resource
def load_pipe():
    return joblib.load('model.pkl')

pipe = load_pipe()

## INPUT FROM USER ##
st.sidebar.header("Enter Car Information")

year = st.sidebar.slider("Year", 1990, 2024, 2015)
engine_hp = st.sidebar.number_input("Engine HP", value=200.0)
engine_cylinders = st.sidebar.number_input("Engine Cylinders", value=4.0)
number_of_doors = st.sidebar.selectbox("Number of Doors", [2.0, 3.0, 4.0])
highway_mpg = st.sidebar.number_input("highway MPG", value=25)
city_mpg = st.sidebar.number_input("city mpg", value=18)
popularity = st.sidebar.number_input("Popularity", value=1000)

make = st.sidebar.text_input("Make", value="BMW")
model_name = st.sidebar.text_input("Model", value="1 Series")
engine_fuel_type = st.sidebar.text_input("Engine Fuel Type", value="regular unleaded")
transmission_type = st.sidebar.selectbox("Transmission Type", ["AUTOMATIC", "MANUAL"])
driven_wheels = st.sidebar.selectbox("Driven_Wheels", ["front wheel drive", "rear wheel drive", "all wheel drive"])
market_category = st.sidebar.text_input("Market Category", value="Luxury")
vehicle_size = st.sidebar.selectbox("Vehicle Size", ["Compact", "Midsize", "Large"])
vehicle_style = st.sidebar.text_input("Vehicle Style", value="Sedan")

## PREPARE DATA ##
new_data = {
    'Make': make, 'Model': model_name, 'Year': year,
    'Engine Fuel Type': engine_fuel_type, 'Engine HP': engine_hp,
    'Engine Cylinders': engine_cylinders, 'Transmission Type': transmission_type,
    'Driven_Wheels': driven_wheels, 'Number of Doors': number_of_doors,
    'Market Category': market_category, 'Vehicle Size': vehicle_size,
    'Vehicle Style': vehicle_style, 'highway MPG': highway_mpg,
    'city mpg': city_mpg, 'Popularity': popularity
}

new_data_df = pd.DataFrame([new_data])

## PREDICTION ##
if st.button("Predict"):
    result = pipe.predict(new_data_df)
    st.write(f"Estimated Price: **${result[0]:,.2f}**")


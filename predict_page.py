import streamlit as st
import numpy as np
import pickle

def load_data():
    with open("boston.pkl","rb") as file:
        data = pickle.load(file)
    return data
data = load_data()

regressor = data["model"]
le_ocean = data["sea"]
scaler = data["scaler"]

def show_prediction():
    st.title("Real Estate Price Prediction")
    st.write("""This project is made by CH.VINEETH RAM""")
    ocean = (
        'NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND'
    )
    oceanp = st.selectbox("Ocean Proximity",ocean)
    age = st.number_input("House age",0)
    rooms = st.number_input("Number of rooms",0)
    brooms = st.number_input("Number of bedrooms",0)
    income = st.number_input("Median income of House hold(in 10k $)",0)
    population = st.number_input("Number of people living in the building",0)
    households = st.number_input("Number of families living in the building",0)
    
    ok = st.button("Predict the price")
    if ok:
        X = np.array([[age,rooms,brooms,population,households,income,oceanp]])
        X[:,6] = le_ocean.transform(X[:,6])
        X = X.astype(float)
        X = scaler.transform(X)
        
        price = regressor.predict(X)
        st.subheader(f"The predicted price of the property is ${price[0]:,.2f}")
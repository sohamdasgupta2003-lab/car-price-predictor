import gzip
import pickle

import streamlit as st
import pandas as pd
import joblib

# --------------------------------
# 1. Load the trained ML model
# --------------------------------

model = joblib.load('car_price_model.pkl.gz')

# Extract the fitted preprocessing pipeline
rf_fitted = model.regressor_

preprocessor = rf_fitted.named_steps['preprocessing']

encoder = preprocessor.named_transformers_['encoder']

# Extract available categories from training data
categories = encoder.categories_

car_names = sorted(categories[0].tolist())
fuel_types = sorted(categories[1].tolist())
seller_types = sorted(categories[2].tolist())
transmissions = sorted(categories[3].tolist())
owners = sorted(categories[4].tolist())


# --------------------------------
# 2. Configure the application
# --------------------------------

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Car Price Predictor")

st.write(
    "Predict the estimated selling price of a used car "
    "using Machine Learning."
)

st.divider()


# --------------------------------
# 3. Create input form
# --------------------------------

st.subheader("Enter Your Car Details")

with st.form("car_prediction_form"):

    car_name = st.selectbox(
        "Car Model",
        car_names
    )

    col1, col2 = st.columns(2)

    with col1:

        year = st.number_input(
            "Manufacturing Year",
            min_value=1980,
            max_value=2026,
            value=2018,
            step=1
        )

        km_driven = st.number_input(
            "Kilometers Driven",
            min_value=1,
            max_value=1000000,
            value=45000,
            step=1000
        )

        fuel = st.selectbox(
            "Fuel Type",
            fuel_types
        )

    with col2:

        seller_type = st.selectbox(
            "Seller Type",
            seller_types
        )

        transmission = st.selectbox(
            "Transmission",
            transmissions
        )

        owner = st.selectbox(
            "Number of Previous Owners",
            owners
        )

    submitted = st.form_submit_button(
        "Predict Car Price",
        use_container_width=True
    )


# --------------------------------
# 4. Make prediction
# --------------------------------

if submitted:

    # Prepare input in the same format
    # used during model training

    input_data = pd.DataFrame([{
        'name': car_name,
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner
    }])

    # Predict using the trained model
    prediction = model.predict(input_data)[0]

    prediction = max(0, prediction)

    # Display result
    st.divider()

    st.success("Prediction completed!")

    st.subheader("Estimated Selling Price")

    st.markdown(
        f"## ₹ {prediction:,.0f}"
    )

    st.caption(
        "This is an ML-based estimate, not a guaranteed market price."
    )

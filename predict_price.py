import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle


# Function for log transformation
def log_transform(value):
    return np.log(value)


html_temp = """
<div style="background-color:tomato;padding:10px">
<h1 style="color:white;text-align:center;">BerliNB crystal ball simulator</h1>
</div>"""
st.markdown(html_temp, unsafe_allow_html=True)

model = pickle.load(open("best_xgb_model.pkl", "rb"))


new_data = {
    "host_since": [3650.0],
    "accommodates": [1.098612],
    "bathrooms": [0.693147],
    "bedrooms": [0.693147],
    "guests_included": [0.693147],
    "minimum_nights": [1.098612],
    "maximum_nights": [7.026427],
    "calculated_host_listings_count": [0.693147],
    "friedrichshain_kreuzberg": [0],
    "mitte": [0],
    "pankow": [1],
    "neukölln": [0],
    "charlottenburg_wilm.": [0],
    "tempelhof___schöneberg": [0],
    "lichtenberg": [0],
    "treptow___köpenick": [0],
    "steglitz___zehlendorf": [0],
    "reinickendorf": [0],
    "marzahn___hellersdorf": [0],
    "spandau": [0],
    "private_room": [0],
    "entire_home_apt": [1],
    "shared_room": [0],
}

host_since = st.number_input("For how long have you been hosting on AirBNB?", 0, 5785)
accommodates = st.number_input("How many guests can the listing accommodate?", 1, 16)
bathrooms = st.number_input("Number of bathrooms", 0, 10)
bedrooms = st.number_input("Number of bedrooms", 0, 12)
guests_included = st.number_input("How many guests are included in the price per night?", 1, 16) 
minimum_nights = st.number_input("Minimum nights", 1, 365)
maximum_nights = st.number_input("Maximum nights", 1, 1000)
calculated_host_listings_count = st.number_input("How many listings do you have in Berlin, including this one?", 1, 100)


# List of districts
districts = [
    "friedrichshain_kreuzberg",
    "mitte",
    "pankow",
    "neukölln",
    "charlottenburg_wilm.",
    "tempelhof___schöneberg",
    "lichtenberg",
    "treptow___köpenick",
    "steglitz___zehlendorf",
    "reinickendorf",
    "marzahn___hellersdorf",
    "spandau",
]

# Dropdown to select the district
selected_district = st.selectbox("District", districts)

# Initialize a dictionary for one-hot encoding of districts
one_hot_district = {district: 0 for district in districts}
one_hot_district[selected_district] = 1

# Convert the district dictionary to a DataFrame
one_hot_district_df = pd.DataFrame([one_hot_district])


# List of room types
room_types = ["private_room", "entire_home_apt", "shared_room"]

# Radio buttons to select the room type
selected_room_type = st.radio("What type of room are you offering?", room_types)

# Initialize a dictionary for one-hot encoding of room types
one_hot_room_type = {room_type: 0 for room_type in room_types}
one_hot_room_type[selected_room_type] = 1

# Convert the room type dictionary to a DataFrame
one_hot_room_df = pd.DataFrame([one_hot_room_type])


new_data = {
    "host_since": [host_since],
    "accommodates": [log_transform(accommodates)],
    "bathrooms": [log_transform(bathrooms)],
    "bedrooms": [log_transform(bedrooms)],
    "guests_included": [log_transform(guests_included)],
    "minimum_nights": [log_transform(minimum_nights)],
    "maximum_nights": [log_transform(maximum_nights)],
    "calculated_host_listings_count": [log_transform(calculated_host_listings_count)],
}
new_data.update(one_hot_district)
new_data.update(one_hot_room_type)

new_data_df = pd.DataFrame(new_data)


if st.button('Predict'):
    prediction = model.predict(new_data_df)
    st.success(
        "The estimated price of your listing is €{} per guest. ".format(
            int(np.exp(prediction[0]))
        )
)

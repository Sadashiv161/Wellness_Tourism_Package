import streamlit as st
import pandas as pd
import joblib

from huggingface_hub import hf_hub_download

# Download model from Hugging Face Model Hub
model_path = hf_hub_download(
    repo_id="sadashivbhatt/Wellness_Tourism_Package",
    filename="tourism_project_Prediction_v1.joblib"
)

model = joblib.load(model_path)

st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Wellness Tourism Package Prediction")

st.write("""
Predict whether a customer is likely to purchase the newly introduced
**Wellness Tourism Package**.
""")

st.header("Customer Details")

col1, col2 = st.columns(2)

with col1:

    Age = st.number_input("Age", 18, 100, 30)

    TypeofContact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Enquiry"]
    )

    CityTier = st.selectbox(
        "City Tier",
        [1,2,3]
    )

    DurationOfPitch = st.number_input(
        "Duration Of Pitch",
        min_value=1,
        max_value=60,
        value=10
    )

    Occupation = st.selectbox(
        "Occupation",
        [
            "Salaried",
            "Small Business",
            "Large Business",
            "Free Lancer"
        ]
    )

    Gender = st.selectbox(
        "Gender",
        ["Male","Female"]
    )

    NumberOfPersonVisiting = st.number_input(
        "Number Of Persons Visiting",
        1,
        10,
        2
    )

    NumberOfFollowups = st.number_input(
        "Number Of Followups",
        0,
        10,
        2
    )

with col2:

    ProductPitched = st.selectbox(
        "Product Pitched",
        [
            "Basic",
            "Standard",
            "Deluxe",
            "Super Deluxe",
            "King"
        ]
    )

    PreferredPropertyStar = st.selectbox(
        "Preferred Property Star",
        [3,4,5]
    )

    MaritalStatus = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced",
            "Unmarried"
        ]
    )

    NumberOfTrips = st.number_input(
        "Number Of Trips",
        0,
        20,
        2
    )

    Passport = st.selectbox(
        "Passport",
        [0,1]
    )

    PitchSatisfactionScore = st.selectbox(
        "Pitch Satisfaction Score",
        [1,2,3,4,5]
    )

    OwnCar = st.selectbox(
        "Own Car",
        [0,1]
    )

    NumberOfChildrenVisiting = st.number_input(
        "Number Of Children Visiting",
        0,
        5,
        0
    )

    Designation = st.selectbox(
        "Designation",
        [
            "Executive",
            "Manager",
            "Senior Manager",
            "AVP",
            "VP"
        ]
    )

    MonthlyIncome = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=500000,
        value=30000
    )

input_df = pd.DataFrame({

    "Age":[Age],
    "TypeofContact":[TypeofContact],
    "CityTier":[CityTier],
    "DurationOfPitch":[DurationOfPitch],
    "Occupation":[Occupation],
    "Gender":[Gender],
    "NumberOfPersonVisiting":[NumberOfPersonVisiting],
    "NumberOfFollowups":[NumberOfFollowups],
    "ProductPitched":[ProductPitched],
    "PreferredPropertyStar":[PreferredPropertyStar],
    "MaritalStatus":[MaritalStatus],
    "NumberOfTrips":[NumberOfTrips],
    "Passport":[Passport],
    "PitchSatisfactionScore":[PitchSatisfactionScore],
    "OwnCar":[OwnCar],
    "NumberOfChildrenVisiting":[NumberOfChildrenVisiting],
    "Designation":[Designation],
    "MonthlyIncome":[MonthlyIncome]

})

if st.button("Predict"):

    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction")

    if prediction == 1:

        st.success(
            "Customer is likely to purchase the Wellness Tourism Package."
        )

    else:

        st.error(
            "Customer is unlikely  to purchase the Wellness Tourism Package."
        )

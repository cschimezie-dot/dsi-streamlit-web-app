

#import libraries 
import streamlit as st
import pandas as pd
import joblib

# load our model pipline object 
model = joblib.load( "model.joblib")

#add title and instructions
st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")


# age input form

# age input form
age = st.number_input(
    label = "01. Enter the cusotmer's credit score", 
    min_value =18, # min age of 18
    max_value = 2000,
    value = 35
    )

# gender input form
gender = st.radio(
    label = "02. Enter the cusotmer's gender",
    options = ['M' ,'F'] 
   )
# credit score input form

credit_score = st.number_input(
    label = "03. Enter the credit score", 
    min_value = 0, 
    max_value = 830,
    value = 500
) 

# submit inputs to model
if st.button("Submit For Prediction"):
    
    # store our data in a dataframe for prediction
    # Added the missing colon after "credit_score"
    new_data = pd.DataFrame({
        "age": [age], 
        "gender": [gender], 
        "credit_score": [credit_score]
    })
    
    # apply model pipeline to the input data and extract probability prediction 
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # output prediction 
    # Added the missing '}' to close the f-string variable
    st.subheader(f"Based on these customer attributes, our model predicts a purchase probability of {pred_proba:.0%}")
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

#Load the trained model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

#Streamlit app
st.title('Customer Churn Prediction')

#Input fields
Geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
Gender = st.selectbox('Gender', label_encoder_gender.classes_)
Age = st.slider('Age', 18, 92)
EstimatedSalary = st.number_input('EstimatedSalary')
Tenure = st.slider('Tenure', 0, 10)
Balance = st.number_input('Balance')
NumOfProducts = st.slider('NumOfProducts', 1, 4)
HasCrCard = st.selectbox('HasCrCard', [0, 1])
IsActiveMember = st.selectbox('IsActiveMember', [0, 1])
credit_score = st.number_input('CreditScore')

#Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [NumOfProducts],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [EstimatedSalary]
})

# One-hot encode Geography
geography_encoded = onehot_encoder_geo.transform([[Geography]]).toarray()
geography_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine with input data
input_data = pd.concat([input_data.reset_index(drop=True), geography_df], axis=1)

#Scale the input data
input_data_scaled = scaler.transform(input_data)

#Make prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

#Display result
st.write(f'Churn Probability: {prediction_proba:.2%}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')
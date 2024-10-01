import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st

# Data collection and analysis
# Load the diabetes dataset
diabetes_dataset = pd.read_csv('diabetes.csv')

# Separate the features and target variable
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the Support Vector Machine Classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

# Streamlit UI
st.title("Prediksi penyakit diabetes Web App")
st.subheader("Masukan data detail")

# Example values for features
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
example_values = [2, 120, 70, 20, 79, 25.0, 0.5, 33]

# User inputs
name = st.text_input("Masukan nama anda", "contoh: John Doe")
input_data = []

for feature, example in zip(feature_names, example_values):
    value = st.number_input(f'Enter your {feature}', value=example)
    input_data.append(value)



if st.button('Enter'):
    input_data_df = pd.DataFrame([input_data], columns=feature_names)
    std_data = scaler.transform(input_data_df)
    prediction = classifier.predict(std_data)
    
    if prediction[0] == 0:
      st.success(f'{name}, tidak diabetes')
    else:
      st.error(f'{name}, terdeteksi diabetes')




# Display accuracy of the model
st.write(f'akurasi data latih: {training_data_accuracy:.2f}')
st.write(f'akurasi test data: {test_data_accuracy:.2f}')

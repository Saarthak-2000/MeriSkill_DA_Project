#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import math
#Spliting Data into Train and Test:
from sklearn.model_selection import train_test_split 
#For Feature Scaling:
from sklearn.preprocessing import StandardScaler 
#Support Vector Machine:
from sklearn.svm import SVC
#Logistic Regression:
from sklearn.linear_model import LogisticRegression
#Evaluation:
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
#For ignoring  warnings:
import warnings
warnings.filterwarnings('ignore')


# In[5]:


data = pd.read_csv(r"C:\Users\HP\Downloads\DIABETES PATIENTS\diabetes.csv")

data


# In[13]:


df = data.copy()


# In[14]:


df.head()


# In[15]:


df.tail()


# In[16]:


df.shape
print("Total Number of Rows in Dataset  :",data.shape[0])
print("Total Number of Columns in Dataset:",data.shape[1])


# In[17]:


df.info()


# In[18]:


plt.figure(figsize=(8, 6))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.title('Missing Values Heatmap')
plt.show()


# In[19]:


df.describe()


# In[20]:


import plotly.graph_objs as go

# Create a list to store the box plot traces
box_traces = []

# Iterate through each column and create a box plot
for column in df.columns:
    if column != 'Outcome':  # Exclude 'Outcome' if it's the target variable
        trace = go.Box(y=df[column], name=column)
        box_traces.append(trace)

# Create a layout
layout = go.Layout(title='Box Plots for Dataset Columns')

# Create a figure and add the traces and layout
fig = go.Figure(data=box_traces, layout=layout)

# Show the figure
fig.show()


# In[21]:


#Create a function to handle Outliers
def remove_outliers(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    lower_limit = Q1 - 1.5 * IQR
    data[column_name] = data[column_name].clip(lower=lower_limit, upper=upper_limit)
    return data


# In[22]:


#Handle outliers using "remove_outliers" function

df = remove_outliers(df, 'Pregnancies')
df = remove_outliers(df, 'Glucose')
df = remove_outliers(df, 'BloodPressure')
df = remove_outliers(df, 'SkinThickness')
df = remove_outliers(df, 'Insulin')
df = remove_outliers(df, 'BMI')


# In[23]:


import plotly.graph_objs as go

# Create a list to store the box plot traces
box_traces = []

for column in df.columns:
    if column != 'Outcome':  # Exclude 'Outcome' if it's the target variable
        trace = go.Box(y=df[column], name=column)
        box_traces.append(trace)

# Create a layout
layout = go.Layout(title='Box Plots for Dataset Columns')

# Create a figure and add the traces and layout
fig = go.Figure(data=box_traces, layout=layout)

# Show the figure
fig.show()


# In[24]:


df['Pregnancies']=round(df['Pregnancies'].astype('int32'))
df['Glucose']=round(df['Glucose'].astype('int32'))
df['Insulin']=round(df['Insulin'].astype('int32'))


# In[25]:


df.info()


# In[26]:


df.columns


# In[27]:


import pandas as pd
import matplotlib.pyplot as plt

# Plot the histogram
fig, ax = plt.subplots(figsize=(10, 10))
df.hist(bins=50, ax=ax)

# Show the plot
plt.show()


# Observations:
# 
# Distributions are mostly skewed to the right
# 
# Small peaks at higher values for glucose, blood pressure, skin thickness, insulin, BMI, and diabetes pedigree function
# 
# Bimodal distribution for outcome variable (diabetes vs. no diabetes)

# In[28]:


# Count the occurrences of each outcome value
outcome_counts = df['Outcome'].value_counts()

# Create a pie chart
plt.figure(figsize=(6, 6))
plt.pie(outcome_counts, labels=['Non-Diabetic', 'Diabetic'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
plt.title('Distribution of Outcomes')
plt.show()


# In[29]:


pregnancies_bins = [0, 1, 4, 8, 12, 16]
pregnancies_labels = ['0', '1-3', '4-7', '8-11', '12-15']
df['PregnanciesGroup'] = pd.cut(df['Pregnancies'], bins=pregnancies_bins, labels=pregnancies_labels)

# Filter the dataset to include only records with 'Outcome' equal to 1 (Diabetic patients)
diabetic_df = df[df['Outcome'] == 1]

# Create a bar chart for Diabetic patients with 'PregnanciesGroup' as the x-axis
plt.figure(figsize=(10, 6))
sns.countplot(data=diabetic_df, x='PregnanciesGroup', order=pregnancies_labels, palette="Set2")
plt.xlabel('Pregnancies group')
plt.ylabel('Count of Diabetic Patients')
plt.title('Count of Diabetic Patients by Pregnancies Group')
plt.show()


# In[30]:


bins = [20, 30, 40, 50, 60, 70, 80, 200]
labels = ['20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)


# In[31]:


# Filter the dataset to include only records with 'Outcome' equal to 1 (Diabetic patients)
diabetic_df = df[df['Outcome'] == 1]

# Create a bar chart for Diabetic patients with age groups
plt.figure(figsize=(10, 6))
sns.countplot(data=diabetic_df, x='AgeGroup', order=labels, palette="Set2")
plt.xlabel('Age group')
plt.ylabel('Count of Diabetic Patients')
plt.title('Count of Diabetic Patients by Age Group')
plt.xticks(rotation=45)
plt.show()


# In[32]:


new_df = df[(df['Outcome'] == 1) & (df['Pregnancies'] > 0)]
# Create a bar chart with 'Outcome' as hue
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=new_df, x='AgeGroup', y='Pregnancies')
plt.xlabel('Age group')
plt.ylabel('Number of pregnancies')
plt.title('Count of Pregnancies with diabetes by Age Group')
plt.xticks(rotation=45)


plt.show()


# In[33]:


# Define the bins and labels for 'BloodPressure'
blood_pressure_bins = [0, 80, 89, 99, 119, 1000]  # Adjust the boundaries as needed
blood_pressure_labels = ['Low', 'Normal', 'Prehypertension', 'Stage 1 hypertension', 'Stage 2 hypertension']

# Create a new column 'BloodPressureCategory' based on the bins and labels
df['BloodPressureCategory'] = pd.cut(df['BloodPressure'], bins=blood_pressure_bins, labels=blood_pressure_labels,right=False)

df.head()


# In[34]:


# Filter the dataset to include only records with 'Outcome' equal to 1 (Diabetic patients)
diabetic_df = df[df['Outcome'] == 1]

# Create a bar chart for Diabetic patients with 'BloodPressureCategory' as the x-axis
plt.figure(figsize=(10, 6))
sns.countplot(data=diabetic_df, x='BloodPressureCategory', order=blood_pressure_labels, palette="Set2")
plt.xlabel('Blood Pressure Category')
plt.ylabel('Count of Diabetic Patients')
plt.title('Count of Diabetic Patients by Blood Pressure Category')
plt.xticks(rotation=45)
plt.show()


# In[35]:


skin_thickness_bins = [0, 20, 30, 40, 50, 100]
skin_thickness_labels = ['Very thin', 'Thin', 'Normal', 'Thick', 'Very thick']

# Create a new column 'SkinThicknessCategory' based on the bins and labels
df['SkinThicknessCategory'] = pd.cut(df['SkinThickness'], bins=skin_thickness_bins, labels=skin_thickness_labels)

# Filter the dataset to include only records with 'Outcome' equal to 1 (Diabetic patients)
diabetic_df = df[df['Outcome'] == 1]

# Create a bar chart for Diabetic patients with 'SkinThicknessCategory' as the x-axis
plt.figure(figsize=(10, 6))
sns.countplot(data=diabetic_df, x='SkinThicknessCategory', order=skin_thickness_labels, palette="Set2")
plt.xlabel('Skin Thickness Category')
plt.ylabel('Count of Diabetic Patients')
plt.title('Count of Diabetic Patients by Skin Thickness Category')
plt.xticks(rotation=45)
plt.show()


# In[36]:


# Define the custom bins and labels for 'BMI'
bmi_bins = [0, 18.5, 24.9, 29.9, 1000]
bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese']

# Create a new column 'BMICategory' based on the custom bins and labels
df['BMICategory'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_labels)

# Filter the dataset to include only records with 'Outcome' equal to 1 (Diabetic patients)
diabetic_df = df[df['Outcome'] == 1]

# Create a bar chart for Diabetic patients with 'BMICategory' as the x-axis
plt.figure(figsize=(10, 6))
sns.countplot(data=diabetic_df, x='BMICategory', order=bmi_labels, palette="Set2")
plt.xlabel('BMI Category')
plt.ylabel('Count of Diabetic Patients')
plt.title('Count of Diabetic Patients by BMI Category')
plt.xticks(rotation=45)
plt.show()


# In[37]:


insulin_bins = [0, 50, 200, 10000]  
insulin_labels = ['Low', 'Medium', 'High']

# Create a new column 'InsulinCategory' based on the custom bins and labels
df['InsulinCategory'] = pd.cut(df['Insulin'], bins=insulin_bins, labels=insulin_labels)

# Filter the dataset to include only records with 'Outcome' equal to 1 (Diabetic patients)
diabetic_df = df[df['Outcome'] == 1]

# Create a bar chart for Diabetic patients with 'InsulinCategory' as the x-axis
plt.figure(figsize=(10, 6))
sns.countplot(data=diabetic_df, x='InsulinCategory', order=insulin_labels, palette="Set2")
plt.xlabel('Insulin Category')
plt.ylabel('Count of Diabetic Patients')
plt.title('Count of Diabetic Patients by Insulin Category')
plt.xticks(rotation=45)
plt.show()


# In[38]:


glucose_bins = [0, 75, 90, 125, 150, 1000]  # Adjust the boundaries as needed
glucose_labels = ['Very low', 'Low', 'Normal', 'Prediabetic', 'High']

# Create a new column 'GlucoseCategory' based on the custom bins and labels
df['GlucoseCategory'] = pd.cut(df['Glucose'], bins=glucose_bins, labels=glucose_labels)

# Filter the dataset to include only records with 'Outcome' equal to 1 (Diabetic patients)
diabetic_df = df[df['Outcome'] == 1]

# Create a bar chart for Diabetic patients with 'GlucoseCategory' as the x-axis
plt.figure(figsize=(10, 6))
sns.countplot(data=diabetic_df, x='GlucoseCategory', order=glucose_labels, palette="Set2")
plt.xlabel('Glucose Category')
plt.ylabel('Count of Diabetic Patients')
plt.title('Count of Diabetic Patients by Glucose Category')
plt.xticks(rotation=45)
plt.show()


# In[39]:


df.columns


# In[40]:


#Independent Variables
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
#Target variable
y = df['Outcome']


# In[41]:


X.head()


# In[42]:


y.head()


# In[43]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[44]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[45]:


X_train


# In[46]:


sv_model = SVC(C= 0.1,kernel='linear',random_state=15)
sv_model.fit(X_train,y_train)

#Prediction on Traing Data
sv_pred_train = sv_model.predict(X_train)
#Prediction on Test Data 
sv_pred_test = sv_model.predict(X_test)


#Evaluation
SVM_Train_Accuracy = accuracy_score(y_train,sv_pred_train)*100
SVM_Test_Accuracy = accuracy_score(y_test,sv_pred_test)*100
SVM_CV = cross_val_score(sv_model,X_test,y_test,cv=5,scoring="accuracy").mean()*100


print(f"Train Accuracy: {SVM_Train_Accuracy:.2f}%")
print(f"Test Accuracy: {SVM_Test_Accuracy:.2f}%")
print(f"cross Validataion Score: {SVM_CV:.2f}%")


# In[47]:


logistic = LogisticRegression(C=100,penalty='l1',solver='liblinear',random_state=16)
logistic.fit(X_train,y_train)


#Prediction on Traing Data
log_pred_train = logistic.predict(X_train)
#Prediction on Test Data 
log_pred_test = logistic.predict(X_test)


log_Train_Accuracy = accuracy_score(y_train,log_pred_train)*100
log_Test_Accuracy = accuracy_score(y_test,log_pred_test)*100
Log_CV = cross_val_score(logistic,X_test,y_test,cv=5,scoring="accuracy").mean()*100


print(f"Train Accuracy: {log_Train_Accuracy:.2f}%")
print(f"Test Accuracy: {log_Test_Accuracy:.2f}%")
print(f"cross Validataion Score: {Log_CV:.2f}%")


# Conclusion
# After evaluating the performance metrics of the models, specifically Support Vector Machine and Logistic Regression, and keeping in mind the objective of maximizing the accuracy in predicting Diabetic Patients, the Logistic Regression model stands out as the most suitable choice.
# 
# As a result, we recommend using Logistic Regression for predicting Diabetic Patients based on the available data and the assessed evaluation metrics.

# Developing a Prediction System

# In[48]:


input_data = (2,174,88,37,120,44.5,0.646,24)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)

prediction = logistic.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# Graphical User Interface

# In[49]:


import tkinter as tk
from tkinter import Label, Entry, Button

# Function to make predictions
def predict():
    input_data = [
        pregnancies_entry.get(),
        glucose_entry.get(),
        blood_pressure_entry.get(),
        skin_thickness_entry.get(),
        insulin_entry.get(),
        bmi_entry.get(),
        pedigree_function_entry.get(),
        age_entry.get()
    ]

    # Convert input_data to a NumPy array and standardize it
    input_data_as_numpy_array = np.asarray(input_data, dtype=float).reshape(1, -1)
    std_input_data = scaler.transform(input_data_as_numpy_array)

    # Make the prediction using sv_model
    prediction = logistic.predict(std_input_data)

    # Interpret the prediction
    result = 'diabetic' if prediction[0] == 1 else 'not diabetic'
    result_label.config(text=f'Prediction: {result}')

# Create the main window
root = tk.Tk()
root.title("Diabetes Prediction")

# Create labels and entry fields for input features
Label(root, text="Pregnancies").grid(row=0, column=0)
pregnancies_entry = Entry(root)
pregnancies_entry.grid(row=0, column=1)

Label(root, text="Glucose").grid(row=1, column=0)
glucose_entry = Entry(root)
glucose_entry.grid(row=1, column=1)

Label(root, text="Blood Pressure").grid(row=2, column=0)
blood_pressure_entry = Entry(root)
blood_pressure_entry.grid(row=2, column=1)

Label(root, text="Skin Thickness").grid(row=3, column=0)
skin_thickness_entry = Entry(root)
skin_thickness_entry.grid(row=3, column=1)

Label(root, text="Insulin").grid(row=4, column=0)
insulin_entry = Entry(root)
insulin_entry.grid(row=4, column=1)

Label(root, text="BMI").grid(row=5, column=0)
bmi_entry = Entry(root)
bmi_entry.grid(row=5, column=1)

Label(root, text="Diabetes Pedigree Function").grid(row=6, column=0)
pedigree_function_entry = Entry(root)
pedigree_function_entry.grid(row=6, column=1)

Label(root, text="Age").grid(row=7, column=0)
age_entry = Entry(root)
age_entry.grid(row=7, column=1)


# Create a button to make predictions
predict_button = Button(root, text="Predict", command=predict)
predict_button.grid(row=8, columnspan=2)

# Create a label to display predictions
result_label = Label(root, text="Prediction: ")
result_label.grid(row=9, columnspan=2)

# Start the main loop
root.mainloop()


# In[ ]:





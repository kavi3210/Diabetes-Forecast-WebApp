from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import seaborn as sns

# load the diabetes dataset
df = pd.read_csv('diabetes.csv')

# group the data by outcome to get a sense of the distribution
diabetes_mean_df = df.groupby('Outcome').mean()

# split the data into input and target variables
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# scale the input variables using StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# create an SVM model with a linear kernel
model = svm.SVC(kernel='linear')

# train the model on the training set
model.fit(X_train, y_train)

# make predictions on the training and testing sets
train_y_pred = model.predict(X_train)
test_y_pred = model.predict(X_test)

# calculate the accuracy of the model on the training and testing sets
train_acc = accuracy_score(train_y_pred, y_train)
test_acc = accuracy_score(test_y_pred, y_test)

# create the Streamlit app
def app():

    img = Image.open("img.jpeg")
    img = img.resize((200,200))
    st.image(img,caption="Diabetes Image",width=200)


    st.title('Diabetes Prediction')

    # create the input form for the user to input new data
    st.sidebar.title('Input Features')
    preg = st.sidebar.slider('Pregnancies', 0, 10, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 72)
        
    skinthickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725, 0.001)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # make a prediction based on the user input
    input_data = [preg, glucose, bp, skinthickness, insulin, bmi, dpf, age]
    user_report_data = {
      'pregnancies':preg,
      'glucose':glucose,
      'bp':bp,
      'skinthickness':skinthickness,
      'insulin':insulin,
      'bmi':bmi,
      'dpf':dpf,
      'age':age
    }
    user_data = pd.DataFrame(user_report_data, index=[0])
    input_data_nparray = np.asarray(input_data)
    reshaped_input_data = input_data_nparray.reshape(1, -1)
    prediction = model.predict(reshaped_input_data)

    # display the prediction to the user
    st.write('Based on the input features, the model predicts:')
    if prediction == 1:
        st.warning('This person has diabetes.')
    else:
        st.success('This person does not have diabetes.')

    st.subheader('Patient Data')
    st.write(user_data)

    # display some summary statistics about the dataset
    st.header('Dataset Summary')
    st.write(df.describe())

    st.header('Distribution by Outcome')
    st.write(diabetes_mean_df)
    
    if prediction==0:
        color = 'blue'
    else:
        color = 'red'

    st.header('Pregnancy count Graph (Others vs Yours)')
    fig_preg = plt.figure()
    ax1 = sns.scatterplot(x = 'Age', y = 'Pregnancies', data = df, hue = 'Outcome', palette = 'Greens')
    ax2 = sns.scatterplot(x = user_data['age'], y = user_data['pregnancies'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,20,2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_preg)

    # Age vs Glucose
    st.header('Glucose Value Graph (Others vs Yours)')
    fig_glucose = plt.figure()
    ax3 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome' , palette='magma')
    ax4 = sns.scatterplot(x = user_data['age'], y = user_data['glucose'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,220,10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_glucose)



# Age vs Bp
    st.header('Blood Pressure Value Graph (Others vs Yours)')
    fig_bp = plt.figure()
    ax5 = sns.scatterplot(x = 'Age', y = 'BloodPressure', data = df, hue = 'Outcome', palette='Reds')
    ax6 = sns.scatterplot(x = user_data['age'], y = user_data['bp'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,130,10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bp)


# Age vs St
    st.header('Skin Thickness Value Graph (Others vs Yours)')
    fig_st = plt.figure()
    ax7 = sns.scatterplot(x = 'Age', y = 'SkinThickness', data = df, hue = 'Outcome', palette='Blues')
    ax8 = sns.scatterplot(x = user_data['age'], y = user_data['skinthickness'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,110,10))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_st)


# Age vs Insulin
    st.header('Insulin Value Graph (Others vs Yours)')
    fig_i = plt.figure()
    ax9 = sns.scatterplot(x = 'Age', y = 'Insulin', data = df, hue = 'Outcome', palette='rocket')
    ax10 = sns.scatterplot(x = user_data['age'], y = user_data['insulin'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,900,50))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_i)


# Age vs BMI
    st.header('BMI Value Graph (Others vs Yours)')
    fig_bmi = plt.figure()
    ax11 = sns.scatterplot(x = 'Age', y = 'BMI', data = df, hue = 'Outcome', palette='rainbow')
    ax12 = sns.scatterplot(x = user_data['age'], y = user_data['bmi'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,70,5))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_bmi)


# Age vs Dpf
    st.header('DPF Value Graph (Others vs Yours)')
    fig_dpf = plt.figure()
    ax13 = sns.scatterplot(x = 'Age', y = 'DiabetesPedigreeFunction', data = df, hue = 'Outcome', palette='YlOrBr')
    ax14 = sns.scatterplot(x = user_data['age'], y = user_data['dpf'], s = 150, color = color)
    plt.xticks(np.arange(10,100,5))
    plt.yticks(np.arange(0,3,0.2))
    plt.title('0 - Healthy & 1 - Unhealthy')
    st.pyplot(fig_dpf)

    # display the model accuracy
    st.header('Model Accuracy')
    st.write(f'Train set accuracy: {train_acc:.2f}')
    st.write(f'Test set accuracy: {test_acc:.2f}')

if __name__ == '__main__':
    app()

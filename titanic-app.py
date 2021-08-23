import streamlit as st
import pandas as pd
import pickle

st.write("""

# 🚢 Titanic Prediction App

Will you survive? Will you perish?
Let's find out!

This app predicts whether a passenger on the Titanic survives or not, based on different features such as Class, Sex, Age, Siblings, Parch and Fare.
\nYou can check out the original data on [Kaggle](https://www.kaggle.com/c/titanic/data).
\nThis app uses a Random Forest Classifier, you can find more information about it [here](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

""")


def user_input_features():
    sex = st.sidebar.selectbox('Gender', ('Female', 'Male'))
    pclass = st.sidebar.slider('Passenger Class', 1, 3, 1)
    age = st.sidebar.slider('Age', 0, 1, 110)
    siblings = st.sidebar.slider('Siblings or spouse onboard', 0, 1, 8)
    parch = st.sidebar.slider('Parents or children onboard', 0, 1, 6)
    fare = st.sidebar.slider('Ticket Fare', 0, 10, 1000)

    data = {
        'Sex': [sex],
        'Pclass': [pclass],
        'Age': [age],
        "SibSp": [siblings],
        "Parch": [parch],
        "Fare": [fare]
            }

    features = pd.DataFrame(data)
    return features

df = user_input_features()

def sex(x):
    if x == 'Male':
        return 1
    else:
        return 0

df['Sex'] = df['Sex'].apply(lambda x: sex(x))


# Load the saved model
pickle_model = pickle.load(open('pickle_model.pkl', 'rb'))
# Predicts
prediction = pickle_model.predict(df)
prediction_proba = pickle_model.predict_proba(df)
#
if prediction[0] == 1:
    output = "You're a survivor! 🥳"
else:
    output = "We've got some bad news for you... 🙁"

st.subheader("Our Prediction")
st.write(output)

st.subheader("Prediction Probability")
st.write(prediction_proba)


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

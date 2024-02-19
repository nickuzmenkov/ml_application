import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

data = load_iris()
model = LogisticRegression(max_iter=1000)
model.fit(data.data, data.target)

st.title("Iris classification")

sepal_width = st.number_input(label="Sepal length", min_value=0)
sepal_height = st.number_input(label="Sepal width", min_value=0)
petal_width = st.number_input(label="Petal length", min_value=0)
petal_height = st.number_input(label="Petal width", min_value=0)

if st.button(label="Get iris class"):
    predict = model.predict([[sepal_width, sepal_height, petal_width, petal_height]])[0]
    st.success(data.target_names[predict])

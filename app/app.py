import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from datetime import date

st.title("Sales Forcasting")

model_path = Path(__file__).parent.parent / "models" / "best_xgboost.pkl"

model = joblib.load(model_path)

st.write("This application predicts weekly sales using a trained XGBoost model.")

st.header("Enter input data")

store = st.number_input("store" ,min_value = 1 , step =1)
dept  = st.number_input("dept" , min_value =1, step=1)
temperature = st.number_input("temperature" )
fuel_price = st.number_input("fule price")
cpi = st.number_input("CPI")
unemployment = st.number_input("unemployment" )
markdown1 = st.number_input("MarkDown1", value=0.0)
markdown2 = st.number_input("MarkDown2", value=0.0)
markdown3 = st.number_input("MarkDown3", value=0.0)
markdown4 = st.number_input("MarkDown4", value=0.0)
markdown5 = st.number_input("MarkDown5", value=0.0)
size = st.number_input("size" ,min_value =  0 ,step = 1)
is_holiday = st.selectbox("Is Holiday???" ,[True , False])
store_type = st.selectbox("store type",  ["A" , "B" , "c"])
input_date = st.date_input("Date",value=date(2012, 1, 1) )


input_data = pd.DataFrame({
    "Store": [store],
    "Dept": [dept],
    "Type": [store_type],
    "IsHoliday": [is_holiday],
    "Temperature": [temperature],
    "Fuel_Price": [fuel_price],
    "MarkDown1": [markdown1],
    "MarkDown2": [markdown2],
    "MarkDown3": [markdown3],
    "MarkDown4": [markdown4],
    "MarkDown5": [markdown5],
    "CPI": [cpi],
    "Unemployment": [unemployment],
    "Size": [size],
    "Year": [input_date.year],
    "Month": [input_date.month],
    "Day": [input_date.day]

})


input_data = pd.get_dummies(input_data,columns=["Type", "Store", "IsHoliday"],drop_first=True)

input_data = input_data.reindex(columns = model.feature_names_in_ , fill_value = 0)
st.write(input_data.shape)


if st.button("Predict Sales"):
    prediction = model.predict(input_data)
    st.success(f"Predict weekly sales : {prediction[0]: ,.2f}")

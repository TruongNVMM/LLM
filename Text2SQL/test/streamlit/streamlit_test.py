import pandas as pd 
import matplotlib.pyplot as plt 
import streamlit as st 

st.title('Phân tích dữ liệu giá nhà')
upload_file = st.file_uploader('Chọn file csv', type=['csv'])

def calculate_average(prices):
    return sum(prices) / len(prices)


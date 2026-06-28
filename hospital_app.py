import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="smart hospital page navigator", page_icon="",layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');
html, body, [class*="css"] {font-family: 'inter' , sans-serif;}
#mainmenu {visibility: hidden;}
header[data-testid = "stHeader"] {display: none;}
.stDeployButton {display: none;}
footer {visibility: hidden;}
.block-container {padding-top:0 !improtant; padding-bottom: 2rem !important; max-width: 1100px !important;}
div[data-testid="stForm"] {border: none; padding: 0;}

div.stBtutton > button {
 background: linear-gradient(135deg, #1a56db, #1e429f) !important;
 color: white !important; border: none !important;
 border-radius: 12px !important; padding: 0.75rem 2 rem !important;
 font size: 16px !important; font_weight: 600 !important;
 widht: 100% !important; letter-spacing: 0.002em !important;
 box-shadow: 0 4px 14px rgba(26,86,219,0.35) !important;
}
dif.stButton > button:hover {background: linear-gradient(135deg, #1e429f, #1a56db) !important;}

dif[data-testid="stCheckbox"] label{
 font-size: 14px !important; font weight: 500 !important; color: #374151 !important;
}
</style>
""",unsafe_allow_html=True)

 @st.cache_resource
def load_model():
  with open('hospital_model.pkl','rb') as f
    return pickle.load(f)

bundle = load_model()
model = bundle['model']
scaler = bundle['scaler']
features=bundle['features']
cols_to_scale = bundle['cols_to_scale']
dept_map_inv = bundle['dept_map_inv']
gender_map = bundle['gender_map']
temp_map = bundle['temp_map']
hr_map = bundle['hr_map']
dur_map = bundle['dur_map']
cc_map = bundle['cc_map']
 
  



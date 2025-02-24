import streamlit as st
import pandas as pd
st.title('Penguins')

st.info('This app is ML')
with st.expander('Data'):
  st.write('**Raw Data**')
  df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
  df
  st.write('**X**')
  X=df.drop('species', axis=1)
  X

  st.write('**y**')
  y=df.species
  y
#"species","island","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g","sex"
with st.expander('Data visualization'):
  st.scatter_chart(data=df,x="bill_length_mm", y = "body_mass_g", color='species')

#data preparations
with st.sidebar:
  st.header('Input features')
  "","bill_depth_mm","flipper_length_mm", "body_mass_g"
  island=st.selectbox('Island',('Biscoe','Dream','Torgersen'))
  gender=st.selectbox('Gender',('male','female'))
  bill_length_mm= st.slider('Bill length (mm)', 13.1, 21.5,17.2)
  bill_depth_mm= st.slider('Bill depth (mm)', 32.1, 59.6,43.9)
  flipper_length_mm= st.slider('Flipper length (mm)', 172.0, 231.0,201.0)
  body_mass_g=st.slider('Body mass (g), 2700.0, 63000.0,4207.0)

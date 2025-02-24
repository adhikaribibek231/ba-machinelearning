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
  # "","bill_depth_mm","flipper_length_mm", "body_mass_g"
  island=st.selectbox('Island',('Biscoe','Dream','Torgersen'))
  gender=st.selectbox('Gender',('male','female'))
  bill_length_mm= st.slider('Bill length (mm)', 32.1, 59.6,43.9)
  

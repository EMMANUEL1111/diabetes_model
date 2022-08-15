
import numpy as np
import pandas as pd
import pickle as pk
import streamlit as st 
model=pk.load(open('svc_new.pkl','rb'))
st.write('** Machine Learning Model to Predict Diabetes Chances of a Patient**')
st.write('Diabetes in the African continent provides challenges both to the researcher and clinician. From a research viewpoint, unusual subgroups of diabetes are encountered in Africa; and the epidemiology of both type 1 and type 2 diabetes differs from many western countries. Clinically, the challenge is to provide care for rapidly expanding numbers of diabetic patients, with severely limited resources.')

st.write('----------')

def diabetes_data(Pregnancies,Glucose ,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    data_input={'Pregnancies':Pregnancies,'Glucose':Glucose,'BloodPressure':BloodPressure,'SkinThickness':SkinThickness,'Insulin':Insulin,'BMI':BMI,'DiabetesPedigreeFunction':DiabetesPedigreeFunction,'Age':Age}
    diabetes_frame=pd.DataFrame(data_input,index=[0])
    pre_diabetes=model.predict(diabetes_frame)
    prob_diabetes=model.predict_proba(diabetes_frame)
    return pre_diabetes, prob_diabetes

def main():
    name=st.sidebar.text_input('Your Name Please')
    st.write('**Welcome **', name)
    
    
    st.title("** ARTIFICIAL INTELLIGENCE  **")
    html="""<div style= "background-color:blue" ;padding : 15px"">
        <h2> <b> --CHECKING DIABETES  CHANCES-- </b> </h2>
    </div>
    """
    st.markdown(html,unsafe_allow_html=True)

    Pregnancies=st.text_input('Pregnacies Test Value  ')
    Glucose=st.text_input('Glucose Test Value ')
    BloodPressure=st.text_input('BloodPressure Test Value ')
    SkinThickness=st.text_input('SkinThickness Test Value ')
    Insulin=st.text_input('Insulin Test Value ')
    BMI=st.text_input('BMI Test Value  ')
    DiabetesPedigreeFunction=st.text_input('DiabetesPedigreeFunction Test Value  ')
    Age=st.text_input('Age of The Patient  ')

    if st.button('Predict Diabetes Chances'):
        predict_diabetes=diabetes_data(Pregnancies,Glucose ,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)[0]
        predict_proba_diabetes=diabetes_data(Pregnancies,Glucose ,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)[1]
        a=np.round((predict_proba_diabetes[0][0]) * 100,1)
        b=np.round((predict_proba_diabetes[0][1]) * 100,1)
        if predict_diabetes==1:
            st.error('THIS PATIENT WILL HAVE DIABETES  ')
            st.write(name,'Your Diabetes  Probablity Chances: NO is {}% , YES is {}%'.format(a,b))
        else:
            st.success('THIS PATIENT WILL NOT HAVE DIABETES ')
            st.write(name,'Your Diabetes Probablity Chances: NO is {}% , YES is {}%'.format(a,b))




if __name__=='__main__':
    main()


st.title("!!! DISCLAIMER ")
st.write('This is an Artificail Intelligence And Its not Totally Accurate')
st.write('_______')
st.write('Machine Learning Model Developed By Emmanauel Oladejo')


st.header('Link to the Github Code')

st.write('https://github.com/EMMANUEL1111/DIABETES/blob/main/DIABETES%20.ipynb')

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 22:04:52 2023

@author: nikitha
"""

import numpy as np
import pickle 
import streamlit as st


loaded_model=pickle.load(open("drug_class_model.sav",'rb'))

#creating a function for classification
def drug_classification(input_data):
    
    numpy_array= np.asarray(input_data)
    
    reshaped = numpy_array.reshape(1,-1)
    
    pred = loaded_model.predict(reshaped)
    
    if (pred== 0):
        return 'Drug Type: DrugY'
    elif(pred== 4):
        return 'Drug Type: DrugX'
    elif(pred== 1):
        return 'Drug Type: DrugA'
    elif(pred== 3):
        return 'Drug Type: DrugC'
    elif(pred== 2):
        return 'Drug Type: DrugB'
        
def main():
    #giving a title
    st.title('Drug Classification Web App')
   
    #getting the input from user
    Age=st.text_input('Age of the person')
    Sex=st.text_input('Type 1 for Male, 0 for Female')
    BloodPressure=st.text_input('BloodPressure value: 0 High, 1 Low, 2 Normal')
    Cholestrol=st.text_input('Cholestrol value: 0 High, 1 Normal')
    Na_to_K=st.text_input('Blood sodium and potassium concentration')
     
    #code for prediction 
    diagnosis=''
    
    #creating a button for prediction
    if st.button('Prescribed Drug Result'):
        diagnosis=drug_classification([Age,Sex,BloodPressure,Cholestrol,Na_to_K])
        
    st.success(diagnosis)
    
if __name__ =='__main__':
    main()
    

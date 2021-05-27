# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 17:37:38 2021

@author: Lara_bis
"""

import streamlit as st

import tensorflow

from keras.preprocessing.sequence import pad_sequences 

from keras.preprocessing.text import Tokenizer

from keras.models import load_model

import os

def main():
    
    st.title("Application Streamlit: Réaliser une analyse de sentiment")
    
    st.header("Choisissez votre modèle")
    
    chemin=os.path.dirname(os.path.abspath(__file__))
    
    def file_selector(chemin):
    
        filenames = os.listdir(chemin)
        
        filenames = [f for f in filenames if f.endswith('.' + "h5")]
        
        selected_filename = st.selectbox('Choisissez votre modèle ', filenames)
        
        return os.path.join(chemin, selected_filename)
    
    chemin = file_selector(chemin)
    
    model = load_model(chemin)
    
    texte = st.text_area('Ecrivez le texte à analyser:')
    
    if texte is not None: 
        
        st.write(texte)
        
        tokenizer = Tokenizer(num_words=5000)
        
        instance = tokenizer.texts_to_sequences(texte)
        
        liste = []
        
        for sublist in instance:
            for item in sublist:
                liste.append(item)
        
        liste = [liste]
    
        instance = pad_sequences(liste, padding='post', maxlen=100)
    
        proba = model.predict(instance)[0][0]
            
        if proba < 0:
            st.write("Le modèle prédit que le commentaire est négatif à hauteur de:",round(abs(proba))*100,"%")
        
        else:
            st.write("Le modèle prédit que le commentaire est positif à hauteur de:",round(abs(proba))*100,"%")  
    
if __name__ == "__main__":
    main()                

import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import numpy as np

# Título e Introducción
st.title("Ayudante para Cardiólogo")
st.write("Realizado por: Juan Borja")
st.markdown("""
Esta aplicación utiliza un modelo de Inteligencia Artificial entrenado con el algoritmo KNN (K-Nearest Neighbors) 
para predecir si una persona tiene o no problemas cardíacos basándose en su edad y nivel de colesterol. 
Los datos fueron normalizados utilizando `MinMaxScaler` y el modelo fue entrenado con un conjunto de datos previamente procesado.
""")

# Cargar el escalador y el modelo
scaler = joblib.load('escalador.bin')
model = joblib.load('modelo_knn.bin')

# Crear pestañas
tab1, tab2 = st.tabs(["Instrucciones y Datos", "Predicción"])

# Pestaña 1: Instrucciones y Datos
with tab1:
    st.header("Instrucciones")
    st.markdown("""
    Para utilizar esta aplicación, siga estos pasos:
    1. Ingrese la **edad** del paciente (entre 18 y 80 años).
    2. Ingrese el nivel de **colesterol** del paciente (entre 50 y 600).
    3. Presione el botón **"Predecir"** en la pestaña de Predicción.
    """)

    # Entrada de datos
    st.header("Datos del Paciente")
    edad = st.number_input("Edad", min_value=18, max_value=80, step=1)
    colesterol = st.number_input("Colesterol", min_value=50, max_value=600, step=1)

# Pestaña 2: Predicción
with tab2:
    st.header("Resultado de la Predicción")
    
    if 'edad' in locals() and 'colesterol' in locals():
        # Botón para realizar la predicción
        if st.button("Predecir"):
            # Crear un DataFrame con los datos ingresados
            datos_entrada = pd.DataFrame({
                'edad': [edad],
                'colesterol': [colesterol]
            })

            # Escalar los datos de entrada
            datos_escalados = scaler.transform(datos_entrada)

            # Realizar la predicción
            prediccion = model.predict(datos_escalados)

            # Mostrar el resultado
            if prediccion[0] == 1:
                st.error("El paciente TIENE problemas cardíacos.")
                st.image("https://s3.amazonaws.com/arc-wordpress-client-uploads/infobae-wp/wp-content/uploads/2017/03/01204001/iStock-506476770.jpg", caption="Problemas Cardíacos Detectados")
            else:
                st.success("El paciente NO tiene problemas cardíacos.")
                st.image("https://s28461.pcdn.co/wp-content/uploads/2017/07/Tu-corazo%CC%81n-consejos-para-mantenerlo-sano-y-fuerte.jpg", caption="Corazón Sano")
    else:
        st.warning("Por favor, ingrese los datos del paciente en la pestaña 'Instrucciones y Datos'.")

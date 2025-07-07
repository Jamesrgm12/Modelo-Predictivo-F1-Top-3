import streamlit as st
import pandas as pd
import numpy as np
import time
import joblib
import pickle

@st.cache_resource
def cargar_modelo():
    return pickle.load(open('modelFinal1.sav','rb'))

def transformar(df):
    
    # One Hot Encoding
    oneHot = joblib.load('oneHE.pkl')

    df_oneHot = df[['constructor_name','nationality_drivers']]
    
    dataOneHot = oneHot.transform(df_oneHot)
    dataOneHot = pd.DataFrame(
        data = dataOneHot,
        columns = oneHot.get_feature_names_out()
    )

    # Target Encoding
    tarEnc = joblib.load('target_encoder.pkl')

    df_tarEnc = df[['driver_name', 'race_name', 'circuit_name']]
    dataTarEnc = tarEnc.transform(df_tarEnc)

    # Min Max Scalar
    minMaxSca = joblib.load('scaler_minmax.pkl')

    df_minMax = df[['q1', 'q2', 'year', 'round']]
    dataMinMax = minMaxSca.transform(df_minMax)
    dataMinMax = pd.DataFrame(
        minMaxSca.transform(df_minMax),
        columns=['q1', 'q2', 'year', 'round'])


    df_ToPredict = pd.concat([dataOneHot,dataTarEnc,dataMinMax],axis=1)

    return df_ToPredict

def prediccion(data):

    modeloF1 = cargar_modelo()

    df_base = pd.DataFrame(
        [data], 
        columns=['nationality_drivers', 'constructor_name','race_name','circuit_name','driver_name','q1','q2','year','round']
    )
    
    df_ToPredict = transformar(df_base)

    resultado = modeloF1.predict(df_ToPredict)

    if(resultado):
        st.success(f"Felicidades {df_base['driver_name'].values[0]}, se encuentra dentro del podio 🏆")
    else:  
        st.error(f"{df_base['driver_name'].values[0]} no ha podido quedar en el podio")
    
def main():
    st.title("Top 3 F1")

    data_cat = pd.read_pickle('input.pickle')
    data_num = pd.read_pickle('df_num.pickle')

    col1, col2 = st.columns(2)

    with col1:
        escuderia = st.selectbox(
            'Escudería',
            list(data_cat['constructor_name'].unique()),
            index=None,
            placeholder="Seleccione la escudería"
        )

        carreraNombre = st.selectbox(
            'Nombre de carrera',
            list(data_cat['race_name'].unique()),
            index=None,
            placeholder="Seleccione el nombre de la carrera"
        )

        timepoQ1 = st.text_input("Ingrese Q1 (en segundos)",placeholder="Ejm: 82.910")

        ronda = st.selectbox(
            'Ronda',
            list(data_num['round'].unique()),
            index=None,
            placeholder="Seleccione la ronda"
        )

    with col2:

        pilotoNombre = st.selectbox(
            'Piloto',
            list(data_cat['driver_name'].unique()),
            index=None,
            placeholder="Seleccione el Piloto"
        )

        circuitoNombre = st.selectbox(
            'Circuito',
            list(data_cat['circuit_name'].unique()),
            index=None,
            placeholder="Seleccione el nombre del circuito"
        )

        tiempoQ2 = st.text_input("Ingrese Q2 (en segundos)",placeholder="Ejm: 82.910")

        anio = st.text_input('Año',placeholder="Ejm: 2018")

    if st.button('Calcular ingreso al podio'):

        if None in [escuderia, carreraNombre, circuitoNombre, pilotoNombre, anio, ronda] or '' in [timepoQ1, tiempoQ2]:
                st.error('Por favor, complete todos los campos antes de continuar', icon="🚨")
                return
        else:
            with st.spinner("Cargando...", show_time=False):
                time.sleep(5)
                
            nacionalidad = data_cat[data_cat["driver_name"] == pilotoNombre]["nationality_drivers"].unique()[0]
            prediccion([nacionalidad, escuderia, carreraNombre, circuitoNombre, pilotoNombre, timepoQ1, tiempoQ2, anio, ronda])

if __name__ == '__main__':
    main()
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
import matplotlib.pyplot as plt

# --- INTERFACE STREAMLIT ---
st.title("📈 IA de Prédiction Boursière & Géopolitique")
st.write("Ce système utilise des neurones LSTM pour le prix et des neurones classiques pour la géopolitique.")

symbol = st.sidebar.text_input("Symbole (ex: BTC-USD, AAPL, GC=F)", "BTC-USD")
news_impact = st.sidebar.slider("Impact Géopolitique (Social/Guerre/News)", -1.0, 1.0, 0.0)
st.sidebar.info("1.0 = Très Positif / -1.0 = Très Négatif (ex: Conflit)")

# --- 1. PRÉPARATION DES DONNÉES (Échantillons - Vidéo 2) ---
@st.cache_data
def load_data(s):
    data = yf.download(s, start="2020-01-01")
    return data

df = load_data(symbol)

# On prépare les prix
scaler = MinMaxScaler()
scaled_prices = scaler.fit_transform(df[['Close']])

# --- 2. CRÉATION DU RÉSEAU DE NEURONES (Couches cachées - Vidéo 2) ---
def build_model():
    # Branche 1 : Analyse l'historique des prix (LSTM)
    input_price = Input(shape=(60, 1))
    x = LSTM(50, return_sequences=False)(input_price)
    
    # Branche 2 : Analyse la donnée Géopolitique (Dense)
    input_geo = Input(shape=(1,))
    y = Dense(10, activation='relu')(input_geo)
    
    # Fusion des deux branches (Comme le cerveau qui croise les infos)
    combined = Concatenate()([x, y])
    
    # Sortie (Prédit le prix - Vidéo 1)
    output = Dense(1)(combined)
    
    model = Model(inputs=[input_price, input_geo], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

# --- 3. ENTRAÎNEMENT ET PRÉDICTION (Backpropagation - Vidéo 1) ---
if st.button('Lancer la prédiction avec IA'):
    with st.spinner('L’IA analyse les graphiques et la géopolitique...'):
        model = build_model()
        
        # On crée un petit entraînement rapide pour l'exemple
        # En réalité, on l'entraînerait sur des milliers de données
        X_p = []
        y_p = []
        for i in range(60, len(scaled_prices)): # Petit échantillon pour aller vite en ligne
            X_p.append(scaled_prices[i-60:i, 0])
            y_p.append(scaled_prices[i, 0])
        
        X_p, y_p = np.array(X_p), np.array(y_p)
        X_p = X_p.reshape(X_p.shape[0], X_p.shape[1], 1)
        
        # Simulation d'un sentiment géopolitique passé aléatoire
        X_g = np.random.uniform(-1, 1, len(X_p))
        
        # Entraînement (L'IA ajuste ses poids)
        model.fit([X_p, X_g], y_p, epochs=5, verbose=0)
        
        # Prédiction pour demain
        last_60_days = scaled_prices[-60:].reshape(1, 60, 1)
        prediction_scaled = model.predict([last_60_days, np.array([[news_impact]])])
        prediction = scaler.inverse_transform(prediction_scaled)
        
        st.metric("Prix Prédit (Demain)", f"{prediction[0][0]:.2f} $")
        st.write("Note : Si vous baissez l'impact géopolitique, la prédiction changera !")
        
        # Graphique
        st.line_chart(df['Close'].tail(100))

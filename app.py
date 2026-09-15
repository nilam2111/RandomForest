import pathlib
import pickle
import streamlit as st


@st.cache_resource
def load_model():
    # Get the folder where app.py is located
    APP_DIR = pathlib.Path(__file__).parent.resolve()
    model_path = APP_DIR / "RandomForest.pkl"

    with open(model_path, "rb") as file:
        return pickle.load(file)


model = load_model()

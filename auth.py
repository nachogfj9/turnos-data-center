import streamlit as st
import pandas as pd
import os
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 4rem;
        padding-bottom: 4rem;
        background-color: #fef4ef;
        min-height: 100vh;
    }

    .stTextInput > div > div > input {
        background-color: #f7f9fc;
    }

    button[kind="primary"] {
        background-color: #ff4b00;
        color: white;
    }

    .stApp {
        background-color: #fef4ef;
    }
    </style>
    """,
    unsafe_allow_html=True
)


USUARIOS_CSV = os.path.join("Data", "usuarios.csv")
CONTRATOS_CSV = os.path.join("Data", "contratos.csv")


def auth_gate():
    if "usuario" in st.session_state and "pep" in st.session_state:
        return  # Ya ha iniciado sesión

    st.title("🔐 Inicio de sesión")

    expediente_input = st.text_input("Expediente", placeholder="Ej: t89640")
    pep_input = st.text_input("PEP del contrato", placeholder="Ej: 123456", type="password")

    if st.button("Iniciar sesión"):
        if not os.path.exists(USUARIOS_CSV):
            st.error("⚠️ Archivo usuarios.csv no encontrado en la carpeta /Data.")
            return

        df_usuarios = pd.read_csv(USUARIOS_CSV)

        usuario = df_usuarios[
            (df_usuarios["Expediente"].astype(str).str.strip() == expediente_input.strip()) &
            (df_usuarios["Pep"].astype(str).str.strip() == pep_input.strip())
        ]

        if not usuario.empty:
            st.session_state["usuario"] = usuario.iloc[0].to_dict()
            st.session_state["pep"] = str(usuario.iloc[0]["Pep"])
            st.success(f"Bienvenido, {usuario.iloc[0]['Nombre']}")
            st.rerun()
        else:
            st.error("❌ Credenciales incorrectas. Verifica expediente y PEP.")



def logout():
    for key in ["usuario", "pep"]:
        st.session_state.pop(key, None)
    st.rerun()

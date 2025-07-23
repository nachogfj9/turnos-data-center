import streamlit as st
import pandas as pd
import os

# Estilos visuales
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

# Rutas de los CSV
USUARIOS_CSV = os.path.join("Data", "usuarios.csv")
CONTRATOS_CSV = os.path.join("Data", "contratos.csv")


import streamlit as st
import pandas as pd
import os

USUARIOS_CSV = os.path.join("Data", "usuarios.csv")
CONTRATOS_CSV = os.path.join("Data", "contratos.csv")

def cargar_datos():
    df_usuarios = pd.read_csv(USUARIOS_CSV)
    df_contratos = pd.read_csv(CONTRATOS_CSV)
    df_usuarios["Expediente"] = df_usuarios["Expediente"].astype(str).str.strip()
    df_usuarios["Pep"] = df_usuarios["Pep"].astype(str).str.strip()
    df_contratos["Pep"] = df_contratos["Pep"].astype(str).str.strip()
    df_contratos["Contrato"] = df_contratos["Contrato"].astype(str).str.strip()
    return df_usuarios, df_contratos

def auth_gate():
    if "usuario" in st.session_state and "pep" in st.session_state and "contrato_nombre" in st.session_state:
        return

    st.title("🔐 Inicia sesión con tu expediente")
    expediente_input = st.text_input("Expediente", placeholder="Ej: a04943")

    if expediente_input:
        df_usuarios, df_contratos = cargar_datos()
        usuarios_filtrados = df_usuarios[df_usuarios["Expediente"] == expediente_input.strip()]

        if usuarios_filtrados.empty:
            st.error("❌ Expediente no encontrado.")
            return

        # Agrupar los PEPs del usuario por contrato
        pep_contratos = []

        for pep in usuarios_filtrados["Pep"].unique():
            fila = df_contratos[df_contratos["Pep"] == pep]
            if not fila.empty:
                contrato = fila["Contrato"].values[0]
                pep_contratos.append((pep, contrato))

        # Agrupar por contrato único
        contratos_dict = {}
        for pep, contrato in pep_contratos:
            if contrato not in contratos_dict:
                contratos_dict[contrato] = pep  # Asociaremos al primer PEP válido encontrado para ese contrato

        if not contratos_dict:
            st.error("❌ No se encontraron contratos asociados a tu expediente.")
            return

        elif len(contratos_dict) == 1:
            # Acceso directo
            contrato, pep = list(contratos_dict.items())[0]
            usuario = usuarios_filtrados[usuarios_filtrados["Pep"] == pep].iloc[0].to_dict()
            iniciar_sesion(usuario, pep, contrato)

        else:
            st.info("Tienes acceso a varios contratos. Selecciona uno:")
            seleccion = st.selectbox("Contratos disponibles:", list(contratos_dict.keys()))
            if st.button("Acceder"):
                pep = contratos_dict[seleccion]
                usuario = usuarios_filtrados[usuarios_filtrados["Pep"] == pep].iloc[0].to_dict()
                iniciar_sesion(usuario, pep, seleccion)


def iniciar_sesion(usuario, pep, contrato):
    st.session_state["usuario"] = usuario
    st.session_state["pep"] = pep
    st.session_state["contrato_nombre"] = contrato

    # Reset de posibles datos previos
    for key in ["df_turnos", "df_parejas"]:
        st.session_state.pop(key, None)

    os.makedirs(f"data_{contrato}", exist_ok=True)

    st.success(f"Bienvenido, {usuario['Nombre']} ({contrato})")
    st.rerun()

def logout():
    for key in ["usuario", "pep", "contrato_nombre", "df_turnos", "df_parejas"]:
        st.session_state.pop(key, None)
    st.rerun()

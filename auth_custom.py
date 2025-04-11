import streamlit as st
import hashlib
import os
import json
import base64

USERS_DB = "usuarios.json"


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def cargar_usuarios():
    if os.path.exists(USERS_DB):
        with open(USERS_DB, "r") as f:
            return json.load(f)
    return {}


def guardar_usuarios(usuarios):
    with open(USERS_DB, "w") as f:
        json.dump(usuarios, f, indent=2)


def estilo_login():
    st.markdown("""
    <style>
    .stApp {
        background-color: #FFBFA3 !important;
    }

    header, footer { visibility: hidden; }

    h1, h2, h3, label,
    .stTextInput label, .stPasswordInput label {
        color: #5A3A2E !important;
    }

    .stTextInput input,
    .stPasswordInput input {
        background-color: white !important;
        color: black !important;
        border-radius: 5px;
    }

    .stButton>button {
        background-color: #FF4B00 !important;
        color: white !important;
        font-weight: bold;
        border-radius: 5px;
    }

    .stButton>button:hover {
        background-color: #FF6A2A !important;
    }

    .main > div {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding-top: 8vh;
    }
    </style>
    """, unsafe_allow_html=True)


def login(config=None):
    usuarios = cargar_usuarios()
    st.title("Inicio de sesión")
    nombre = st.text_input("Nombre de usuario")
    password = st.text_input("Contraseña", type="password")

    if st.button("Iniciar sesión"):
        if nombre in usuarios and usuarios[nombre]["password"] == hash_password(password):
            st.session_state["logged_in"] = True
            st.session_state["user"] = nombre
            st.success(f"Bienvenido, {nombre}")
            st.rerun()
        else:
            st.error("Credenciales incorrectas")


def registro():
    st.subheader("Registrarse")
    usuarios = cargar_usuarios()
    nuevo_usuario = st.text_input("Nuevo usuario")
    nueva_contra = st.text_input("Nueva contraseña", type="password")

    if st.button("Registrar"):
        if nuevo_usuario in usuarios:
            st.warning("Este usuario ya existe.")
        else:
            usuarios[nuevo_usuario] = {"password": hash_password(nueva_contra)}
            guardar_usuarios(usuarios)
            st.success("Usuario creado correctamente")


def auth_gate(config=None):
    if "logged_in" not in st.session_state or not st.session_state.get("logged_in"):
        estilo_login()

        if config and config.get("logo_path") and os.path.exists(config["logo_path"]):
            with open(config["logo_path"], "rb") as f:
                img = f.read()
            logo_base64 = base64.b64encode(img).decode()
            st.markdown(f"""
                <div style="text-align: center; margin-bottom: 2rem;">
                    <img src="data:image/png;base64,{logo_base64}" style="height: 100px;">
                </div>
            """, unsafe_allow_html=True)

        login(config)
        registro()
        st.stop()


def logout():
    st.session_state.pop("user", None)
    st.session_state.pop("logged_in", None)
    st.success("Sesión cerrada.")
    st.rerun()


def cambiar_password():
    usuarios = cargar_usuarios()
    user = st.session_state.get("user")
    if not user:
        st.warning("Debes estar logueado para cambiar la contraseña.")
        return

    st.subheader("Cambiar contraseña")
    actual = st.text_input("Contraseña actual", type="password")
    nueva = st.text_input("Nueva contraseña", type="password")
    confirmar = st.text_input("Confirmar nueva contraseña", type="password")

    if st.button("Actualizar contraseña"):
        if usuarios[user]["password"] != hash_password(actual):
            st.error("Contraseña actual incorrecta.")
        elif nueva != confirmar:
            st.error("La nueva contraseña no coincide.")
        else:
            usuarios[user]["password"] = hash_password(nueva)
            guardar_usuarios(usuarios)
            st.success("Contraseña actualizada correctamente.")

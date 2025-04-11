# editar_calendario.py
import streamlit as st
import pandas as pd
from datetime import datetime
from utils import guardar_calendario_local, cargar_calendario_local

def editar_calendario_tab():
    st.header("🛠️ Edición Manual del Calendario")

    df_turnos, df_parejas = cargar_calendario_local()
    if df_turnos is None:
        st.warning("No hay calendario cargado. Ve a la pestaña 'Calendario y Bajas' para generarlo.")
        return

    id_cols = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "Day", "MesNum"]
    tech_cols = [col for col in df_turnos.columns if col not in id_cols]

    # Modificar turno manualmente
    with st.expander("✏️ Modificar turno manualmente"):
        fecha_mod = st.date_input("Fecha del turno a modificar")
        tecnico_mod = st.selectbox("Técnico", tech_cols)
        turno_nuevo = st.selectbox("Nuevo turno", ["", "M", "T", "N", "R", "V", "D", "M+T", "T+N"])

        if st.button("Aplicar cambio manual"):
            idx = df_turnos[df_turnos["Fecha"] == pd.to_datetime(fecha_mod)].index
            if not idx.empty:
                df_turnos.at[idx[0], tecnico_mod] = turno_nuevo
                guardar_calendario_local(df_turnos, df_parejas)
                st.success(f"Turno actualizado: {tecnico_mod} el {fecha_mod.strftime('%d/%m/%Y')} → {turno_nuevo}")
                st.experimental_rerun()

    # Incorporar técnico nuevo
    with st.expander("➕ Incorporar técnico nuevo a grupo existente"):
        nuevo_tecnico = st.text_input("Nombre del nuevo técnico")
        fecha_inicio_nuevo = st.date_input("Fecha de incorporación", datetime(2025, 5, 1))

        if st.button("Incorporar Técnico"):
            if nuevo_tecnico in df_turnos.columns:
                st.warning(f"{nuevo_tecnico} ya existe en el calendario.")
            else:
                df_turnos[nuevo_tecnico] = ""
                idx_inicio = df_turnos[df_turnos["Fecha"] >= fecha_inicio_nuevo].index
                bloque = df_turnos.loc[idx_inicio[0], "Bloque"]
                grupos_bloque = df_parejas[df_parejas["Bloque"] == bloque]
                grupo_seleccionado = None
                tecnico_modelo = None
                for i, row in grupos_bloque.iterrows():
                    if len(row["Tecnicos"]) < 3:
                        grupo_seleccionado = row["Tecnicos"]
                        tecnico_modelo = grupo_seleccionado[0]
                        df_parejas.at[i, "Tecnicos"] = grupo_seleccionado + [nuevo_tecnico]
                        break
                if grupo_seleccionado is None:
                    grupo_seleccionado = grupos_bloque.iloc[0]["Tecnicos"]
                    tecnico_modelo = grupo_seleccionado[0]
                    df_parejas.at[grupos_bloque.index[0], "Tecnicos"] = grupo_seleccionado + [nuevo_tecnico]

                df_turnos.loc[idx_inicio, nuevo_tecnico] = df_turnos.loc[idx_inicio, tecnico_modelo].values
                guardar_calendario_local(df_turnos, df_parejas)
                st.success(f"Técnico {nuevo_tecnico} incorporado al grupo de {tecnico_modelo} a partir de {fecha_inicio_nuevo.strftime('%d/%m/%Y')}")
                st.experimental_rerun()

    # Bloquear semanas para evitar cambios (simulado con lista)
    with st.expander("🔒 Bloquear semanas (simulado)"):
        if "bloqueos" not in st.session_state:
            st.session_state.bloqueos = []
        semana_bloquear = st.number_input("Semana a bloquear", min_value=1, max_value=52, step=1)
        if st.button("Bloquear semana"):
            if semana_bloquear not in st.session_state.bloqueos:
                st.session_state.bloqueos.append(semana_bloquear)
                st.success(f"Semana {semana_bloquear} bloqueada")

        st.write("Semanas bloqueadas:", st.session_state.bloqueos)

    # Mostrar DataFrame final
    st.write("### Vista rápida del calendario actual")
    st.dataframe(df_turnos.head(20))

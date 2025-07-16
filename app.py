import streamlit as st
import pandas as pd
import numpy as np
import os
import base64
import io
from datetime import datetime, timedelta
from scipy.optimize import linear_sum_assignment
from PIL import Image
from collections import defaultdict
import itertools
import ast  
st.set_page_config(page_title="Guaxen", layout="wide")

###############################################################################
# CONFIGURACIONES POR DEFECTO
###############################################################################
DEFAULT_CONFIG = {
    "logo_path": r"C:/Users/a04943/OneDrive - Serveo/Documentos/TFM/turnos_data_center/Serveo_logo.png",
    "template_file": "MatPol_Plantilla.xlsx",
    "max_hours_year": 1750,
    "repeated_pairs_penalty": 999999,
    "shift_patterns": [
        ["M", "M", "M", "M", "M", "D", "D"],
        ["T", "T", "T", "T", "T", "D", "D"],
        ["N", "N", "N", "N", "N", "T+N", "T+N"],
        ["R", "R", "R", "R", "R", "R", "R"],
        ["V", "V", "V", "V", "V", "M+T", "M+T"]
    ],
    "duracion_bloque_semanas": 5,
    "weak_threshold": 0.3,
    "overlap_strong": 0.7,
    "overlap_mid": 0.3,
    "weight_coverage": 0.45,
    "weight_overlap": 0.20,
    "weight_expertise": 0.35,
    "baja_short_days_threshold": 7,
    "shift_hours": {
        "M": 8, "T": 8, "N": 8,
        "R": 0, "V": 0, "D": 0,
        "M+T": 12, "T+N": 12
    },
    "shift_colors": {
        "M": "#00B0F0",
        "T": "#00B050",
        "N": "#C00000",
        "R": "#FFC000",
        "V": "#BFBFBF",
        "D": "#BFBFBF",
        "M+T": "#92D050",
        "T+N": "#FF5050"
    },
    "technicians_per_shift": 2,
    "skill_types": [
        {"Tipo": "electricidad", "Importancia": 3},
        {"Tipo": "climatizacion", "Importancia": 2}
    ]
}

DAYS_OF_WEEK = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
TURNOS_DISPONIBLES = ["M", "T", "N", "R", "V", "D", "M+T", "T+N"]

###############################################################################
# 0. GESTIÓN DE CONFIGURACIÓN EN SESSION_STATE
###############################################################################
def init_config():
    if "CONFIG" not in st.session_state:
        st.session_state["CONFIG"] = DEFAULT_CONFIG.copy()

def get_config():
    return st.session_state["CONFIG"]

###############################################################################
# 1. FUNCIONES DE CARGA/GUARDADO LOCAL
###############################################################################
def get_contract_folder():
    pep = st.session_state.get("pep", "default")
    folder = f"data_{pep}"
    os.makedirs(folder, exist_ok=True)
    return folder

def guardar_calendario_local(df_turnos, df_grupos):
    folder = get_contract_folder()
    df_turnos.to_csv(os.path.join(folder, "calendario_turnos.csv"), index=False)
    df_grupos["Tecnicos"] = df_grupos["Tecnicos"].apply(str)
    df_grupos.to_csv(os.path.join(folder, "calendario_parejas.csv"), index=False)

def cargar_calendario_local():
    folder = get_contract_folder()
    file_turnos = os.path.join(folder, "calendario_turnos.csv")
    file_parejas = os.path.join(folder, "calendario_parejas.csv")
    if os.path.exists(file_turnos) and os.path.exists(file_parejas):
        df_turnos = pd.read_csv(file_turnos, parse_dates=["Fecha"])
        df_grupos = pd.read_csv(file_parejas)
        df_grupos["Tecnicos"] = df_grupos["Tecnicos"].apply(lambda x: ast.literal_eval(x))
        return df_turnos, df_grupos
    return None, None

###############################################################################
# 2. FUNCIONES DE CÁLCULO DE CALENDARIO Y MATRIZ DE HABILIDADES
###############################################################################
def cargar_y_preparar_matriz(file, num_tecnicos=10):
    xls = pd.ExcelFile(file)
    # Leer la hoja "Habilidades": sin encabezados, omitiendo las dos primeras filas
    df_hab = pd.read_excel(xls, sheet_name="Habilidades", header=None, skiprows=2)
    df_hab.columns = ["Tipo", "Habilidad"]

    # Construir el mapeo base: para cada tipo, se listan las habilidades (normalizadas)
    category_to_skills = defaultdict(list)
    for _, row in df_hab.iterrows():
        tipo = str(row["Tipo"]).strip().lower()
        habilidad = str(row["Habilidad"]).strip().upper()
        category_to_skills[tipo].append(habilidad)
    # Obtener la configuración de tipos de habilidades
    config = get_config()
    tipo_to_weight = {}
    for entry in config.get("skill_types", []):
        t = str(entry["Tipo"]).strip().lower()
        w = entry["Importancia"]
        tipo_to_weight[t] = w
    # Si algún tipo no está configurado, se usará 1 por defecto.
    skill_to_weight = {}
    for tipo, skills in category_to_skills.items():
        peso = tipo_to_weight.get(tipo, 1)
        for skill in skills:
            skill_to_weight[skill] = peso

    # Leer la hoja "Matriz Polivalencia"
    df_pol_raw = pd.read_excel(xls, sheet_name="Matriz Polivalencia", header=None)
    df_pol_raw.columns = df_pol_raw.iloc[3]  # Fila 4 (índice 3) como cabecera
    df_pol = df_pol_raw.iloc[4:].reset_index(drop=True)
    df_pol.rename(columns={df_pol.columns[0]: "Nombre"}, inplace=True)
    # Filtrar técnicos (primeros 10)
    df_pol = df_pol[df_pol["Nombre"].notna()].iloc[:num_tecnicos]

    # Asignar nombres de columnas a la matriz a partir de la lista de habilidades leída
    lista_habilidades = df_hab["Habilidad"].str.upper().str.strip().tolist()
    num_skill_cols = df_pol.shape[1] - 1  # Restar columna "Nombre"
    lista_habilidades = lista_habilidades[:num_skill_cols]
    columnas_df = ["Nombre"] + lista_habilidades
    df_pol.columns = columnas_df

    # Procesar la matriz: convertir a numérico, rellenar NaN y normalizar
    df_skills = df_pol.drop(columns=["Nombre"]).apply(pd.to_numeric, errors='coerce').fillna(0)
    df_normalizado = df_skills / 4.0  # Según código original

    # Construir el vector de pesos usando el diccionario skill_to_weight
    pesos = np.ones(df_normalizado.shape[1])
    for i, col in enumerate(df_normalizado.columns):
        pesos[i] = skill_to_weight.get(str(col).upper(), 1)
    df_ponderado = df_normalizado * pesos
    df_ponderado.insert(0, "Nombre", df_pol["Nombre"])

    return df_ponderado, skill_to_weight

def calcular_complementariedad(df, skill_to_weight):
    config = get_config()
    n = len(df)
    matriz = np.zeros((n, n))
    w_thr = config["weak_threshold"]
    ov_strong = config["overlap_strong"]
    ov_mid = config["overlap_mid"]
    w_cov = config["weight_coverage"]
    w_ov = config["weight_overlap"]
    w_exp = config["weight_expertise"]
    for i in range(n):
        for j in range(i+1, n):
            skills_i = df.iloc[i, 1:].values
            skills_j = df.iloc[j, 1:].values
            # Cobertura
            coverage_score = 0
            potential_coverage = 0
            for si, sj in zip(skills_i, skills_j):
                if si < w_thr or sj < w_thr:
                    potential_coverage += 1
                    coverage_score += max(si, sj)
            coverage_score = coverage_score / max(potential_coverage, 1)
            # Conocimiento común
            overlap_score = 0
            for si, sj in zip(skills_i, skills_j):
                if si > ov_strong and sj > ov_strong:
                    overlap_score += 0.7
                elif si > ov_mid and sj > ov_mid:
                    overlap_score += 1.0
            overlap_score = overlap_score / len(skills_i)
            # Experiencia / Conocimiento especializado
            expertise_score = 0
            max_possible = 0
            for idx_col, col_name in enumerate(df.columns[1:]):
                max_skill = max(skills_i[idx_col], skills_j[idx_col])
                weight = skill_to_weight.get(str(col_name).upper(), 1)
                expertise_score += max_skill * weight
                max_possible += weight
            expertise_score = expertise_score / max_possible
            total_score = w_cov * coverage_score + w_ov * overlap_score + w_exp * expertise_score
            matriz[i, j] = total_score
            matriz[j, i] = total_score
    return matriz

def emparejar_optimo_con_grupos(df_ponderado, matriz_compl, skill_to_weight, techs_per_shift, grupos_anteriores):
    """
    Empareja técnicos en grupos de 2, con un técnico de refuerzo si es necesario.
    """
    num_tecnicos = len(df_ponderado)
    distribucion = calcular_distribucion_optima(num_tecnicos)
    
    grupos = []
    tecnicos_solos = []
    
    cost_matrix = np.zeros((num_tecnicos, num_tecnicos))
    for i in range(num_tecnicos):
        for j in range(i+1, num_tecnicos):
            penalty = get_config()["repeated_pairs_penalty"]
            cost = 1 - matriz_compl[i, j]
            # grupos_anteriores debe ser una lista de sets con nombres (estables)
            pareja_actual = (df_ponderado.iloc[i]['Nombre'], df_ponderado.iloc[j]['Nombre'])
            pareja_actual_inv = (df_ponderado.iloc[j]['Nombre'], df_ponderado.iloc[i]['Nombre'])
            if grupos_anteriores and (pareja_actual in grupos_anteriores or pareja_actual_inv in grupos_anteriores):
                cost += max(penalty - 1, 0)  # Penalización relativa, cero si penalty es 1
            cost_matrix[i,j] = cost_matrix[j,i] = cost
    
    grupos_2 = distribucion[0][1] if distribucion[0][0] == 2 else 0
    if grupos_2 > 0:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        pares = list(zip(row_ind, col_ind))
        pares_unicos = []
        for i, j in pares:
            if i < j and len(pares_unicos) < grupos_2:
                pares_unicos.append((i, j))
        for i, j in pares_unicos:
            grupo = [df_ponderado.index[i], df_ponderado.index[j]]
            grupos.append({"Tecnicos": grupo})
            grupos_anteriores.append((i,j))
    
    if len(distribucion) > 1 and distribucion[1][0] == 1:
        tecnicos_asignados = set()
        for grupo in grupos:
            tecnicos_asignados.update(grupo["Tecnicos"])
        for i in range(num_tecnicos):
            if df_ponderado.index[i] not in tecnicos_asignados:
                tecnicos_solos.append(df_ponderado.index[i])
                break
    
    return pd.DataFrame(grupos), tecnicos_solos

def calculate_group_scores(df, indices_grupo, skill_to_weight):
    config = get_config()
    skills = [df.iloc[i, 1:].values for i in indices_grupo]
    num_skills = len(skills[0])
    num_techs = len(indices_grupo)

    # --- Complementariedad ---
    coverage_score = 0
    possible_slots = 0
    for skill_pos in range(num_skills):
        skill_values = [s[skill_pos] for s in skills]
        if any(v < config["weak_threshold"] for v in skill_values):
            possible_slots += 1
            coverage_score += max(skill_values)
    coverage_score = coverage_score / possible_slots if possible_slots > 0 else 1.0
    coverage_score = min(coverage_score, 1.0)

    # --- Conocimiento Común ---
    shared_skills = 0
    for skill_pos in range(num_skills):
        skill_values = [s[skill_pos] for s in skills]
        if all(v >= config["overlap_mid"] for v in skill_values):
            shared_skills += 1
    overlap_score = shared_skills / num_skills
    overlap_score = min(overlap_score, 1.0)

    # --- Especialización ---
    total_skill_score = 0
    max_skill_score = 0
    for skill_idx, skill_name in enumerate(df.columns[1:]):
        weight = skill_to_weight.get(str(skill_name).upper(), 1)
        skill_values = [s[skill_idx] for s in skills]
        min_value = min(skill_values)
        total_skill_score += min_value * weight
        max_skill_score += weight
    expertise_score = total_skill_score / max_skill_score
    expertise_score = min(expertise_score, 1.0)

    # --- Score total ponderado ---
    total_score = (
        config["weight_coverage"] * coverage_score +
        config["weight_overlap"] * overlap_score +
        config["weight_expertise"] * expertise_score
    )

    return {
        "coverage": coverage_score,
        "overlap": overlap_score,
        "expertise": expertise_score,
        "total": total_score
    }



###############################################################################
# 3. GENERACIÓN DEL CALENDARIO Y EMPAREJAMIENTO DE GRUPOS
###############################################################################
def asignar_semanas_personalizadas(df):
    df = df.copy()
    fechas = df["Fecha"]
    
    # Día de la semana del 1 de enero (lunes=0, domingo=6)
    inicio_ano = fechas.min()
    dia_semana_inicio = inicio_ano.weekday()

    # Días que quedan hasta el domingo (inclusive)
    dias_primera_semana = 6 - dia_semana_inicio + 1

    # Asignar semana 1 a esos primeros días
    semanas = []
    for i, fecha in enumerate(fechas):
        delta_dias = (fecha - inicio_ano).days
        if delta_dias < dias_primera_semana:
            semanas.append(1)
        else:
            semanas.append(2 + (delta_dias - dias_primera_semana) // 7)
    
    df["Semana"] = semanas
    return df

def optimizar_reten_semanal(df_turnos):
    config = get_config()
    df = df_turnos.copy()
    
    # Inicializar contador de R acumuladas por técnico
    cols_id = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "MesNum"]
    if "Day" not in df.columns:
        df["Day"] = df["Fecha"].dt.day
    cols_tec = [col for col in df.columns if col not in cols_id + ["Day"]]
    horas_R = {tec: 0 for tec in cols_tec}

    # Agrupar por semanas
    df = asignar_semanas_personalizadas(df)
    semanas = df["Semana"].unique()
    
    for semana in semanas:
        df_semana = df[df["Semana"] == semana]
        idxs_semana = df_semana.index

        # Técnicos que tienen R esa semana
        tecnicos_con_R = []
        for tec in cols_tec:
            if any(df.loc[idxs_semana, tec].apply(clean_shift) == "R"):
                tecnicos_con_R.append(tec)

        if tecnicos_con_R:
            # Elegir técnico con menos horas de R acumuladas
            tecnico_reten = min(tecnicos_con_R, key=lambda t: horas_R[t])

            # Asignar R a este técnico y V al resto
            for tec in tecnicos_con_R:
                for idx in idxs_semana:
                    if clean_shift(df.at[idx, tec]) == "R":
                        if tec == tecnico_reten:
                            df.at[idx, tec] = "R"  # mantener
                            horas_R[tec] += config["shift_hours"]["R"]
                        else:
                            df.at[idx, tec] = "V"  # vacaciones
    df = df.drop(columns=["Semana"])
    return df


def generar_calendario_anual_grupos(file, anio=2025, num_bloques=2, penalizar_repetidas=True, num_tecnicos=10, techs_per_shift=2):
    config = get_config()
    df_ponderado, skill_to_weight = cargar_y_preparar_matriz(file, num_tecnicos=num_tecnicos)
    
    fecha_inicial = datetime(anio, 1, 1)
    fecha_final = datetime(anio, 12, 31)
    total_days = (fecha_final - fecha_inicial).days + 1
    
    # Calcula la duración exacta de cada bloque
    dias_por_bloque = total_days // num_bloques
    dias_extra = total_days % num_bloques  # Para distribuir equitativamente los días restantes
    
    fechas = pd.date_range(start=fecha_inicial, periods=total_days, freq='D')
    
    df_turnos_anual = pd.DataFrame({
        "Fecha": fechas,
        "Día": pd.Series(fechas).dt.strftime("%A"),
        "Día Corto": pd.Series(fechas).dt.strftime("%a"),
        "MesNum": pd.Series(fechas).dt.month,
        "Mes": pd.Series(fechas).dt.month.map({
            1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo",
            6: "junio", 7: "julio", 8: "agosto", 9: "septiembre",
            10: "octubre", 11: "noviembre", 12: "diciembre"
        })
    })
    
    df_grupos_anual = pd.DataFrame()
    grupos_anteriores = []
    
    current_day = 0  # contador de días para asignar
    
    for bloque_idx in range(num_bloques):
        if bloque_idx > 0:
            ultimos_turnos_previos = {
                tecnico: df_turnos_anual.loc[current_day - 1, tecnico]
                for tecnico in df_ponderado["Nombre"]
                if tecnico in df_turnos_anual.columns
            }
        else:
            ultimos_turnos_previos = {}

        # Añadir un día extra si quedan días adicionales por repartir
        current_block_days = dias_por_bloque + (1 if bloque_idx < dias_extra else 0)
        
        matriz_compl = calcular_complementariedad(df_ponderado, skill_to_weight)
        df_grupos, tecnicos_solos = emparejar_optimo_con_grupos(
            df_ponderado,
            matriz_compl,
            skill_to_weight,
            techs_per_shift,
            grupos_anteriores
        )
        df_grupos["Bloque"] = bloque_idx + 1
        df_grupos_anual = pd.concat([df_grupos_anual, df_grupos], ignore_index=True)
        
        # Asignar turnos para cada grupo
        for idx, row in df_grupos.iterrows():
            tecnicos_grupo = [t for t in row["Tecnicos"] if "dummy" not in str(t)]
            # Buscar patrones válidos que NO empiecen con "M" si algún técnico venía de "N" o "T+N"
            evitar_m = any(
                clean_shift(ultimos_turnos_previos.get(t, "")) in ["N", "T+N"]
                for t in tecnicos_grupo
            )
            for patron_idx in range(len(config["shift_patterns"])):
                primer_turno = config["shift_patterns"][patron_idx][0]
                if evitar_m and primer_turno == "M":
                    continue  # saltar patrones que empiecen por mañana si venían de noche
                else:
                    break  # primer patrón válido encontrado
            if not tecnicos_grupo:
                continue
            turnos_diarios = []
            for day_offset in range(current_block_days):
                dia_actual = fecha_inicial + timedelta(days=current_day + day_offset)
                semana = dia_actual.isocalendar()[1] - 1
                patron_idx = (semana + idx) % len(config["shift_patterns"])
                dia_semana = dia_actual.weekday()
                turno = config["shift_patterns"][patron_idx][dia_semana]
                turnos_diarios.append(turno)
            for tecnico in tecnicos_grupo:
                df_turnos_anual.loc[current_day:current_day + current_block_days - 1, tecnico] = turnos_diarios
        
        df_turnos_anual.loc[current_day:current_day + current_block_days - 1, "Bloque"] = bloque_idx + 1
        
        current_day += current_block_days
    
    df_turnos_anual = df_turnos_anual.fillna("")
    df_turnos_anual = optimizar_reten_semanal(df_turnos_anual)
    return df_turnos_anual, df_grupos_anual

    return df_turnos_anual, df_grupos_anual

def seleccionar_mejor_grupo(tecnicos_disponibles, tam_grupo, matriz_compl, df_ponderado, grupos_anteriores=None):
    if grupos_anteriores is None:
        grupos_anteriores = []
    
    mejor_grupo = None
    mejor_score = float('-inf')
    umbral_minimo = 0.3
    
    if tam_grupo == 1:
        scores_individuales = [(i, df_ponderado.iloc[i, 1:].mean()) for i in tecnicos_disponibles]
        return [max(scores_individuales, key=lambda x: x[1])[0]]
    
    for combinacion in itertools.combinations(tecnicos_disponibles, tam_grupo):
        score = sum(matriz_compl[i][j] for i, j in itertools.combinations(combinacion, 2))
        num_pares = len(list(itertools.combinations(combinacion, 2)))
        score_normalizado = score / num_pares
        if score_normalizado < umbral_minimo:
            continue
        parejas_actuales = set(itertools.combinations(combinacion, 2))
        parejas_anteriores = set()
        for grupo_anterior in grupos_anteriores:
            parejas_anteriores.update(itertools.combinations(grupo_anterior, 2))
        parejas_repetidas = parejas_actuales.intersection(parejas_anteriores)
        if parejas_repetidas:
            penalizacion = len(parejas_repetidas) * 0.2
            score_normalizado -= penalizacion
        if score_normalizado > mejor_score:
            mejor_score = score_normalizado
            mejor_grupo = list(combinacion)
    
    if mejor_grupo is None:
        for combinacion in itertools.combinations(tecnicos_disponibles, tam_grupo):
            score = sum(matriz_compl[i][j] for i, j in itertools.combinations(combinacion, 2))
            if score > mejor_score:
                mejor_score = score
                mejor_grupo = list(combinacion)
    
    return mejor_grupo

def generar_calendario_anual_grupos_mixtos(file, anio=2025, num_bloques=2, distribucion=None, num_tecnicos=10):
    config = get_config()
    df_ponderado, skill_to_weight = cargar_y_preparar_matriz(file, num_tecnicos=num_tecnicos)
    
    fecha_inicial = datetime(anio, 1, 1)
    fecha_final = datetime(anio, 12, 31)
    total_days = (fecha_final - fecha_inicial).days + 1
    
    dias_por_bloque = total_days // num_bloques
    dias_extra = total_days % num_bloques
    
    fechas = pd.date_range(start=fecha_inicial, periods=total_days, freq='D')
    
    df_turnos_anual = pd.DataFrame({
        "Fecha": fechas,
        "Día": pd.Series(fechas).dt.strftime("%A"),
        "Día Corto": pd.Series(fechas).dt.strftime("%a"),
        "MesNum": pd.Series(fechas).dt.month,
        "Mes": pd.Series(fechas).dt.month.map({
            1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo",
            6: "junio", 7: "julio", 8: "agosto", 9: "septiembre",
            10: "octubre", 11: "noviembre", 12: "diciembre"
        })
    })
    
    df_grupos_anual = pd.DataFrame()
    grupos_anteriores = []
    
    current_day = 0
    
    for bloque_idx in range(num_bloques):
        current_block_days = dias_por_bloque + (1 if bloque_idx < dias_extra else 0)
        
        matriz_compl_block = calcular_complementariedad(df_ponderado, skill_to_weight)
        df_grupos_bloque = pd.DataFrame()
        
        tecnicos_disponibles = set(range(num_tecnicos))
        
        for tam_grupo, num_grupos in distribucion:
            for _ in range(num_grupos):
                if len(tecnicos_disponibles) >= tam_grupo:
                    grupo_actual = seleccionar_mejor_grupo(
                        list(tecnicos_disponibles),
                        tam_grupo,
                        matriz_compl_block,
                        df_ponderado,
                        grupos_anteriores
                    )
                    scores = calculate_group_scores(df_ponderado, grupo_actual, skill_to_weight)
                    nombres_tecnicos = [df_ponderado.iloc[i]["Nombre"] for i in grupo_actual]
                    df_grupos_bloque = pd.concat([df_grupos_bloque, pd.DataFrame([{
                        "Tecnicos": nombres_tecnicos,
                        "Complementariedad": scores["coverage"],
                        "Conocimiento Común": scores["overlap"],
                        "Conocimiento Especializado": scores["expertise"],
                        "Compatibilidad Total": scores["total"],
                        "Bloque": bloque_idx + 1
                    }])], ignore_index=True)
                    tecnicos_disponibles -= set(grupo_actual)
                    grupos_anteriores.append(grupo_actual)
        
        df_grupos_anual = pd.concat([df_grupos_anual, df_grupos_bloque], ignore_index=True)
        
        for idx, row in df_grupos_bloque.iterrows():
            tecnicos_grupo = row["Tecnicos"]
            turnos_diarios = []
            for day_offset in range(current_block_days):
                dia_actual = fecha_inicial + timedelta(days=current_day + day_offset)
                semana = dia_actual.isocalendar()[1] - 1
                patron_idx = (semana + idx) % len(config["shift_patterns"])
                dia_semana = dia_actual.weekday()
                turno = config["shift_patterns"][patron_idx][dia_semana]
                turnos_diarios.append(turno)
            
            for tecnico in tecnicos_grupo:
                df_turnos_anual.loc[current_day:current_day + current_block_days - 1, tecnico] = turnos_diarios
        
        df_turnos_anual.loc[current_day:current_day + current_block_days - 1, "Bloque"] = bloque_idx + 1
        
        current_day += current_block_days
    
    df_turnos_anual = df_turnos_anual.fillna("")
    df_turnos_anual = optimizar_reten_semanal(df_turnos_anual)
    return df_turnos_anual, df_grupos_anual

###############################################################################
# 4. LÓGICA DE INCORPORACIÓN
###############################################################################
def incorporar_tecnico_en_mayo(df_turnos, df_parejas, nuevo_tecnico, fecha_incorporacion):
    # Asegurarse de que el técnico no existe
    if nuevo_tecnico in df_turnos.columns:
        st.warning(f"{nuevo_tecnico} ya existe en el calendario.")
        return df_turnos

    df = df_turnos.copy()
    df[nuevo_tecnico] = ""  # Crear la columna vacía

    # Buscar la fecha de inicio
    fecha_inicio = pd.to_datetime(fecha_incorporacion)
    idx_inicio = df[df["Fecha"] >= fecha_inicio].index

    # Buscar un grupo existente al que agregarlo (usamos el de ese bloque)
    bloque = df.loc[idx_inicio[0], "Bloque"]
    grupos_en_bloque = df_parejas[df_parejas["Bloque"] == bloque]
    
    if grupos_en_bloque.empty:
        st.error("No se encontraron grupos en ese bloque.")
        return df

    # Escoger el primer grupo con menos de 3 técnicos (por simplicidad)
    for idx, row in grupos_en_bloque.iterrows():
        if len(row["Tecnicos"]) < 3:
            grupo_seleccionado = row["Tecnicos"]
            tecnico_modelo = grupo_seleccionado[0]  # Copiamos sus turnos
            break
    else:
        st.warning("Todos los grupos ya tienen 3 técnicos. Se agregará a uno igualmente.")
        grupo_seleccionado = grupos_en_bloque.iloc[0]["Tecnicos"]
        tecnico_modelo = grupo_seleccionado[0]

    # Copiar turnos del técnico modelo desde la fecha de incorporación
    df.loc[idx_inicio, nuevo_tecnico] = df.loc[idx_inicio, tecnico_modelo].values

    # Actualizar df_parejas también
    for i, row in df_parejas.iterrows():
        if row["Bloque"] == bloque and tecnico_modelo in row["Tecnicos"]:
            df_parejas.at[i, "Tecnicos"] = row["Tecnicos"] + [nuevo_tecnico]
            break

    return df

###############################################################################
# 4. LÓGICA DE BAJA
###############################################################################
def clean_shift(shift_value):
    if pd.isna(shift_value):
        return None
    return str(shift_value).split(" ")[0].strip()

def calcular_horas_extras_acumuladas(df, config):
    id_cols = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "MesNum"]
    tech_cols = [c for c in df.columns if c not in id_cols]
    
    df_melt = df.melt(id_vars=id_cols, 
                      value_vars=tech_cols, 
                      var_name="Técnico", 
                      value_name="Turno")
    
    df_melt[["Shift", "CoveringBaja", "NeedsSupport"]] = df_melt["Turno"].apply(lambda x: pd.Series(parse_shift_and_coverage(x)))
    df_melt["HorasExtras"] = df_melt.apply(lambda row: config["shift_hours"].get(row["Shift"], 0) if row["CoveringBaja"] else 0, axis=1)
    return df_melt.groupby("Técnico")["HorasExtras"].sum().to_dict()


def seleccionar_reten_menos_horas(turnos_df, dia_idx, candidatos, shift, horas_extras, config):
    disponibles = [t for t in candidatos if clean_shift(turnos_df.at[dia_idx, t]) == "R"]
    if not disponibles:
        return None
    return min(disponibles, key=lambda t: horas_extras.get(t, 0))

def aplicar_baja(df_turnos, tecnico_baja, fecha_inicio_baja, fecha_fin_baja, ret_pref=None):
    config = get_config()
    dt_inicio = datetime.strptime(fecha_inicio_baja, "%d/%m/%Y")
    dt_fin = datetime.strptime(fecha_fin_baja, "%d/%m/%Y")
    start_idx = df_turnos.index[df_turnos["Fecha"] == dt_inicio]
    end_idx = df_turnos.index[df_turnos["Fecha"] == dt_fin]

    if start_idx.empty or end_idx.empty:
        st.error("Fecha de inicio o fin no encontrada en el calendario.")
        return df_turnos

    start_idx = start_idx[0]
    end_idx = end_idx[0]
    dias_baja = (dt_fin - dt_inicio).days + 1

    id_cols = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "Day", "MesNum"]
    cols_tec = [c for c in df_turnos.columns if c not in id_cols]
    is_short_baja = (dias_baja <= config["baja_short_days_threshold"])

    horas_extras = calcular_horas_extras_acumuladas(df_turnos, config)
    
    for idx in range(start_idx, end_idx + 1):
        turno_orig = df_turnos.at[idx, tecnico_baja]
        if pd.isna(turno_orig) or turno_orig == "":
            continue
        df_turnos.at[idx, tecnico_baja] = f"{turno_orig} (BAJA)"
        if ret_pref:
            ret_seleccionado = ret_pref
        else:
            ret_seleccionado = seleccionar_reten_menos_horas(df_turnos, idx, cols_tec, turno_orig, horas_extras, config)
        if ret_seleccionado:
            turno_base = clean_shift(turno_orig)
            if is_short_baja:
                 if turno_base in ["N", "T+N", "M+T", "M", "T"]:
                    df_turnos.at[idx, ret_seleccionado] = f"{turno_base} (H.EXT)"
            else:
                if turno_base in ["T","N", "T+N", "M+T"]:
                    df_turnos.at[idx, ret_seleccionado] = f"{turno_base} (H.EXT)"
    
    return df_turnos

###############################################################################
# 4. LÓGICA DE REFUERZO
###############################################################################
def aplicar_grupo_refuerzo(df_turnos, fecha_inicio, dias, turno_requerido, num_tecnicos, tecnicos_seleccionados=None):
    config = get_config()
    fecha_inicio_dt = datetime.strptime(fecha_inicio, "%d/%m/%Y")
    
    for dia in range(dias):
        fecha_actual = fecha_inicio_dt + timedelta(days=dia)
        idx_fecha = df_turnos.index[df_turnos["Fecha"] == fecha_actual]
        if idx_fecha.empty:
            continue
        idx_fecha = idx_fecha[0]

        if tecnicos_seleccionados:
            tecnicos_asignados = 0
            for tec in tecnicos_seleccionados:
                if clean_shift(df_turnos.at[idx_fecha, tec]) in ["R", "V", "D"]:
                    df_turnos.at[idx_fecha, tec] = f"{turno_requerido} (REFUERZO)"
                    tecnicos_asignados += 1
                    if tecnicos_asignados >= num_tecnicos:
                        break
            if tecnicos_asignados < num_tecnicos:
                st.warning(f"Solo se asignaron {tecnicos_asignados} refuerzos manuales el {fecha_actual.strftime('%d/%m/%Y')}.")
        else:
            # Lógica automática actual
            cols_excluir = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "MesNum"]
            tecnicos = [col for col in df_turnos.columns if col not in cols_excluir]
            disponibles_R = [tec for tec in tecnicos if clean_shift(df_turnos.at[idx_fecha, tec]) == "R"]
            disponibles_VD = [tec for tec in tecnicos if clean_shift(df_turnos.at[idx_fecha, tec]) in ["V", "D"]]
            tecnicos_asignados = []
            for grupo in [disponibles_R, disponibles_VD]:
                for tecnico in grupo:
                    if len(tecnicos_asignados) < num_tecnicos:
                        df_turnos.at[idx_fecha, tecnico] = f"{turno_requerido} (REFUERZO)"
                        tecnicos_asignados.append(tecnico)
                    else:
                        break
                if len(tecnicos_asignados) >= num_tecnicos:
                    break
            if len(tecnicos_asignados) < num_tecnicos:
                st.warning(f"No hay suficientes técnicos disponibles el {fecha_actual.strftime('%d/%m/%Y')}.")

    return df_turnos


###############################################################################
# 5. COLORES Y VISUALIZACIÓN
###############################################################################
SHIFT_COLORS = {
    "M": "#00B0F0",
    "T": "#00B050",
    "N": "#C00000",
    "R": "#FFC000",
    "V": "#BFBFBF",
    "D": "#BFBFBF",
    "M+T": "#92D050",
    "T+N": "#FF5050"
}

def color_cell(shift_value, day_of_week=None):
    config = get_config()
    if isinstance(shift_value, str) and "(BAJA)" in shift_value:
        return 'background-color: #FF4B00; color: white; font-weight: bold'
    if day_of_week in ["Sat", "Sun"]:
        return 'background-color: #FFBFA3; color: black; font-weight: bold'
    base_shift = clean_shift(shift_value)
    if base_shift in config["shift_colors"]:
        return f'background-color: {config["shift_colors"][base_shift]}; color: white; font-weight: bold'
    return ''

def parse_shift_and_coverage(turno_str):
    if pd.isna(turno_str) or turno_str == "":
        return None, False, False
    
    shift_str = str(turno_str).strip()
    covering = "(H.EXT)" in shift_str or "(REFUERZO)" in shift_str
    needs_support = "(BAJA)" in shift_str
    base = shift_str.split(" ")[0]
    return base, covering, needs_support

def mostrar_calendario_mensual_editable(df, selected_month):
    config = get_config()
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Day"] = df["Fecha"].dt.day
    df_month = df[df["Mes"].str.lower() == selected_month.lower()].copy()
    if df_month.empty:
        st.warning(f"No hay datos para el mes de {selected_month.capitalize()}.")
        return
    day_map = dict(zip(df_month["Day"], df_month["Día Corto"]))
    st.markdown(f"""
        <div style="background-color: #FF4B00;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;">
            <h2 style="text-align: center; color: white;">
                Calendario de {selected_month.capitalize()}
            </h2>
        </div>
    """, unsafe_allow_html=True)
    id_cols = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "Day", "MesNum"]
    tech_cols = [col for col in df.columns if col not in id_cols]
    days_in_month = sorted(df_month["Day"].unique())
    data = []
    for tech in tech_cols:
        row_data = {"Técnico": tech}
        for day in days_in_month:
            day_data = df_month[df_month["Day"] == day][tech].iloc[0] if any(df_month["Day"] == day) else ""
            row_data[str(day)] = day_data
        data.append(row_data)
    calendar_df = pd.DataFrame(data)
    calendar_df.set_index("Técnico", inplace=True)
    def style_func(val, day_of_week):
        return color_cell(val, day_of_week)
    styles = []
    for col in calendar_df.columns:
        day_of_week = day_map.get(int(col), "")
        styles.append({
            'selector': f'td:nth-child({list(calendar_df.columns).index(col) + 1})',
            'props': [('background-color', style_func(calendar_df[col].iloc[0], day_of_week))]
        })
    
    st.write("### Calendario Mensual (Turnos)")
    editable_calendar_df = calendar_df.reset_index()
    edited_df = st.data_editor(
        editable_calendar_df,
        height=550,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            col: st.column_config.TextColumn(width="small") for col in editable_calendar_df.columns if col != "Técnico"
        }
    )
    # Si necesitas seguir usando el DataFrame editado:
    edited_df.set_index("Técnico", inplace=True)

    df_melt = df_month.melt(id_vars=id_cols, value_vars=tech_cols, 
                           var_name="Técnico", value_name="Turno")
    df_melt[["Shift", "CoveringBaja", "NeedsSupport"]] = df_melt["Turno"].apply(lambda x: pd.Series(parse_shift_and_coverage(x)))
    df_counts = df_melt.groupby(["Técnico", "Shift", "CoveringBaja", "NeedsSupport"]).size().reset_index(name="Cantidad")
    df_counts["TurnoCompleto"] = df_counts.apply(lambda row: f"{row['Shift']} (Cubriendo)" if row["CoveringBaja"] else row["Shift"], axis=1)
    df_counts_pivot = df_counts.pivot_table(
        index="Técnico",
        columns="TurnoCompleto",
        values="Cantidad",
        fill_value=0
    ).reset_index()
    st.write("### Resumen de Turnos (Mensual)")
    st.dataframe(df_counts_pivot, height=200)
    df_melt["HorasExtras"] = df_melt.apply(
        lambda r: config["shift_hours"].get(r["Shift"], 0) if r["CoveringBaja"] else 0,
        axis=1
    )
    df_extras = df_melt.groupby("Técnico")["HorasExtras"].sum().reset_index()
    df_extras.columns = ["Técnico", "Horas Extras (Baja)"]
    st.write("### Horas Extras (cubrimiento de bajas) - Mensual")
    st.dataframe(df_extras, height=200)

def mostrar_calendario_mensual(df, selected_month):
    config = get_config()
    df = df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Day"] = df["Fecha"].dt.day
    df_month = df[df["Mes"].str.lower() == selected_month.lower()].copy()
    if df_month.empty:
        st.warning(f"No hay datos para el mes de {selected_month.capitalize()}.")
        return
    day_map = dict(zip(df_month["Day"], df_month["Día Corto"]))
    st.markdown(f"""
        <div style="background-color: #FF4B00;
                    padding: 10px;
                    border-radius: 5px;
                    margin-bottom: 20px;">
            <h2 style="text-align: center; color: white;">
                Calendario de {selected_month.capitalize()}
            </h2>
        </div>
    """, unsafe_allow_html=True)
    id_cols = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "Day", "MesNum"]
    tech_cols = [col for col in df.columns if col not in id_cols]
    days_in_month = sorted(df_month["Day"].unique())
    data = []
    for tech in tech_cols:
        row_data = {"Técnico": tech}
        for day in days_in_month:
            day_data = df_month[df_month["Day"] == day][tech].iloc[0] if any(df_month["Day"] == day) else ""
            row_data[str(day)] = day_data
        data.append(row_data)
    calendar_df = pd.DataFrame(data)
    calendar_df.set_index("Técnico", inplace=True)
    def style_func(val, day_of_week):
        return color_cell(val, day_of_week)
    styles = []
    for col in calendar_df.columns:
        day_of_week = day_map.get(int(col), "")
        styles.append({
            'selector': f'td:nth-child({list(calendar_df.columns).index(col) + 1})',
            'props': [('background-color', style_func(calendar_df[col].iloc[0], day_of_week))]
        })
    
    styled_df = calendar_df.style.apply(lambda x: [style_func(v, day_map.get(int(x.name), "")) for v in x])
    st.write("### Calendario Mensual (Turnos)")
    st.dataframe(styled_df, height=500)
    df_melt = df_month.melt(id_vars=id_cols, value_vars=tech_cols, 
                           var_name="Técnico", value_name="Turno")
    df_melt[["Shift", "CoveringBaja", "NeedsSupport"]] = df_melt["Turno"].apply(lambda x: pd.Series(parse_shift_and_coverage(x)))
    df_counts = df_melt.groupby(["Técnico", "Shift", "CoveringBaja", "NeedsSupport"]).size().reset_index(name="Cantidad")
    df_counts["TurnoCompleto"] = df_counts.apply(lambda row: f"{row['Shift']} (Cubriendo)" if row["CoveringBaja"] else row["Shift"], axis=1)
    df_counts_pivot = df_counts.pivot_table(
        index="Técnico",
        columns="TurnoCompleto",
        values="Cantidad",
        fill_value=0
    ).reset_index()
    st.write("### Resumen de Turnos (Mensual)")
    st.dataframe(df_counts_pivot, height=200)
    df_melt["HorasExtras"] = df_melt.apply(
        lambda r: config["shift_hours"].get(r["Shift"], 0) if r["CoveringBaja"] else 0,
        axis=1
    )
    df_extras = df_melt.groupby("Técnico")["HorasExtras"].sum().reset_index()
    df_extras.columns = ["Técnico", "Horas Extras (Baja)"]
    st.write("### Horas Extras (cubrimiento de bajas) - Mensual")
    st.dataframe(df_extras, height=200)

def mostrar_resumen_anual(df):
    config = get_config()
    id_cols = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "MesNum", "Day"]
    if "Day" not in df.columns:
        df["Day"] = df["Fecha"].dt.day
    tech_cols = [c for c in df.columns if c not in id_cols]

    df_melt = df.melt(id_vars=id_cols, value_vars=tech_cols, var_name="Técnico", value_name="Turno")
    df_melt[["Shift", "CoveringBaja", "NeedsSupport"]] = df_melt["Turno"].apply(lambda x: pd.Series(parse_shift_and_coverage(x)))

    # Cantidad de turnos normales
    df_counts = df_melt.groupby(["Técnico", "Shift"]).size().reset_index(name="Cantidad de Turnos")
    df_counts_pivot = df_counts.pivot(index="Técnico", columns="Shift", values="Cantidad de Turnos").fillna(0)
    st.write("### Resumen de Turnos (Anual)")
    st.dataframe(df_counts_pivot, height=250)

    # Horas Extras (ahora correctamente incluyendo bajas y refuerzos)
    df_melt["HorasExtras"] = df_melt.apply(
        lambda row: config["shift_hours"].get(row["Shift"], 0) if row["CoveringBaja"] else 0,
        axis=1
    )
    df_extras = df_melt.groupby("Técnico")["HorasExtras"].sum().reset_index()
    df_extras.columns = ["Técnico", "Horas Extras (Baja + Refuerzo)"]
    st.write("### Horas Extras Anuales (Baja + Refuerzo)")
    st.dataframe(df_extras, height=250)


def mostrar_resumen_anual_horas_sep_cobertura(df):
    horas_turno_detalle = {
    "M": {"diurnas": 8, "nocturnas": 0},
    "T": {"diurnas": 7, "nocturnas": 1},
    "N": {"diurnas": 0, "nocturnas": 8},
    "M+T": {"diurnas": 12, "nocturnas": 0},
    "T+N": {"diurnas": 4, "nocturnas": 8},
    "R": {"diurnas": 0, "nocturnas": 0},
    "V": {"diurnas": 0, "nocturnas": 0},
    "D": {"diurnas": 0, "nocturnas": 0}
    }
    config = get_config()
    id_cols = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "MesNum", "Day"]
    if "Day" not in df.columns:
        df["Day"] = df["Fecha"].dt.day
    tech_cols = [c for c in df.columns if c not in id_cols]
    df_melt = df.melt(id_vars=id_cols, value_vars=tech_cols, var_name="Técnico", value_name="Turno")
    df_melt[["Shift", "CoveringBaja", "NeedsSupport"]] = df_melt["Turno"].apply(lambda x: pd.Series(parse_shift_and_coverage(x)))
    df_melt["HorasTurno"] = df_melt["Shift"].apply(lambda x: config["shift_hours"].get(x, 0) if x else 0)
    df_melt["HorasDiurnas"] = df_melt.apply(lambda r: horas_turno_detalle.get(r["Shift"], {"diurnas": 0})["diurnas"] if not r["CoveringBaja"] else 0, axis=1)
    df_melt["HorasNocturnas"] = df_melt.apply(lambda r: horas_turno_detalle.get(r["Shift"], {"nocturnas": 0})["nocturnas"] if not r["CoveringBaja"] else 0, axis=1)
    df_melt["HorasExtrasDiurnas"] = df_melt.apply(lambda r: horas_turno_detalle.get(r["Shift"], {"diurnas": 0})["diurnas"] if r["CoveringBaja"] else 0, axis=1)
    df_melt["HorasExtrasNocturnas"] = df_melt.apply(lambda r: horas_turno_detalle.get(r["Shift"], {"nocturnas": 0})["nocturnas"] if r["CoveringBaja"] else 0, axis=1)
    df_final = df_melt.groupby("Técnico").agg({
    "HorasDiurnas": "sum",
    "HorasNocturnas": "sum",
    "HorasExtrasDiurnas": "sum",
    "HorasExtrasNocturnas": "sum"
    }).reset_index()
    df_final["TotalHoras"] = df_final[["HorasDiurnas", "HorasNocturnas", "HorasExtrasDiurnas", "HorasExtrasNocturnas"]].sum(axis=1)
    max_hours = config["max_hours_year"]
    def color_hours(val):
        if val >= max_hours:
            return 'background-color: #FF4B00; color: white; font-weight: bold'
        else:
            return 'background-color: lightgreen; color: black; font-weight: bold'
    styled_df = df_final.style.applymap(color_hours, subset=["TotalHoras"])
    st.write("### Resumen Anual de Horas (Separando Cobertura)")
    st.dataframe(styled_df, height=300)

from datetime import datetime

def mostrar_resumen_hasta_hoy(df):
    config = get_config()
    hoy = pd.Timestamp(datetime.today().date())
    fecha_str = hoy.strftime("%d/%m/%Y")

    id_cols = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "MesNum", "Day"]
    if "Day" not in df.columns:
        df["Day"] = df["Fecha"].dt.day
    tech_cols = [c for c in df.columns if c not in id_cols]

    df_filtrado = df[df["Fecha"] <= hoy]
    
    if df[df["Fecha"] <= hoy].empty:
        st.warning("El calendario comienza en el futuro. No hay datos acumulados hasta hoy.")
        return

    df_melt = df_filtrado.melt(id_vars=id_cols, value_vars=tech_cols, var_name="Técnico", value_name="Turno")
    df_melt[["Shift", "CoveringBaja", "NeedsSupport"]] = df_melt["Turno"].apply(lambda x: pd.Series(parse_shift_and_coverage(x)))
    
    df_counts = df_melt.groupby(["Técnico", "Shift"]).size().reset_index(name="Cantidad de Turnos")
    df_counts_pivot = df_counts.pivot(index="Técnico", columns="Shift", values="Cantidad de Turnos").fillna(0)

    st.write(f"### 📊 Resumen de Turnos (Acumulado hasta {fecha_str})")
    st.dataframe(df_counts_pivot, height=250)

    df_melt["HorasExtras"] = df_melt.apply(
        lambda row: config["shift_hours"].get(row["Shift"], 0) if row["CoveringBaja"] else 0,
        axis=1
    )
    df_extras = df_melt.groupby("Técnico")["HorasExtras"].sum().reset_index()
    df_extras.columns = ["Técnico", "Horas Extras Acumuladas"]
    st.write(f"### ⏱️ Horas Extras (Baja/Refuerzo) hasta {fecha_str}")
    st.dataframe(df_extras, height=250)

def mostrar_resumen_hasta_hoy_horas_sep(df):
    horas_turno_detalle = {
    "M": {"diurnas": 8, "nocturnas": 0},
    "T": {"diurnas": 7, "nocturnas": 1},
    "N": {"diurnas": 0, "nocturnas": 8},
    "M+T": {"diurnas": 12, "nocturnas": 0},
    "T+N": {"diurnas": 4, "nocturnas": 8},
    "R": {"diurnas": 0, "nocturnas": 0},
    "V": {"diurnas": 0, "nocturnas": 0},
    "D": {"diurnas": 0, "nocturnas": 0}
    }
    config = get_config()
    hoy = pd.Timestamp(datetime.today().date())
    fecha_str = hoy.strftime("%d/%m/%Y")

    id_cols = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "MesNum", "Day"]
    if "Day" not in df.columns:
        df["Day"] = df["Fecha"].dt.day
    df_filtrado = df[df["Fecha"] <= hoy]
    tech_cols = [c for c in df.columns if c not in id_cols]

    if df[df["Fecha"] <= hoy].empty:
        st.warning("El calendario comienza en el futuro. No hay datos acumulados hasta hoy.")
        return
    
    df_melt = df_filtrado.melt(id_vars=id_cols, value_vars=tech_cols, var_name="Técnico", value_name="Turno")
    df_melt[["Shift", "CoveringBaja", "NeedsSupport"]] = df_melt["Turno"].apply(lambda x: pd.Series(parse_shift_and_coverage(x)))
    df_melt["HorasTurno"] = df_melt["Shift"].apply(lambda x: config["shift_hours"].get(x, 0) if x else 0)
    df_melt["HorasDiurnas"] = df_melt.apply(lambda r: horas_turno_detalle.get(r["Shift"], {"diurnas": 0})["diurnas"] if not r["CoveringBaja"] else 0, axis=1)
    df_melt["HorasNocturnas"] = df_melt.apply(lambda r: horas_turno_detalle.get(r["Shift"], {"nocturnas": 0})["nocturnas"] if not r["CoveringBaja"] else 0, axis=1)
    df_melt["HorasExtrasDiurnas"] = df_melt.apply(lambda r: horas_turno_detalle.get(r["Shift"], {"diurnas": 0})["diurnas"] if r["CoveringBaja"] else 0, axis=1)
    df_melt["HorasExtrasNocturnas"] = df_melt.apply(lambda r: horas_turno_detalle.get(r["Shift"], {"nocturnas": 0})["nocturnas"] if r["CoveringBaja"] else 0, axis=1)

    df_final = df_melt.groupby("Técnico").agg({
    "HorasDiurnas": "sum",
    "HorasNocturnas": "sum",
    "HorasExtrasDiurnas": "sum",
    "HorasExtrasNocturnas": "sum"
    }).reset_index()
    df_final["TotalHoras"] = df_final[["HorasDiurnas", "HorasNocturnas", "HorasExtrasDiurnas", "HorasExtrasNocturnas"]].sum(axis=1)

    max_hours = config["max_hours_year"]
    def color_hours(val):
        if val >= max_hours:
            return 'background-color: #FF4B00; color: white; font-weight: bold'
        else:
            return 'background-color: lightgreen; color: black; font-weight: bold'

    styled_df = df_final.style.applymap(color_hours, subset=["TotalHoras"])
    st.write(f"### 📈 Resumen de Horas (hasta {fecha_str})")
    st.dataframe(styled_df, height=300)

###############################################################################
# 6. VISUALIZACIÓN DE PAREJAS
###############################################################################

import plotly.graph_objects as go

def mostrar_gauge_chart(valor, titulo, color="#FF4B00"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor * 100,
        number={'suffix': "%"},
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': titulo},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 60], 'color': "#FFB3B3"},
                {'range': [60, 85], 'color': "#C6C6EF"},
                {'range': [85, 100], 'color': "#FFE699"}
            ]
        }
    ))
    fig.update_layout(margin=dict(l=1, r=1, t=30, b=0), height=300)
    return fig



def mostrar_parejas_en_cards_por_bloque(df_grupos):
    st.markdown("""
        <style>
        .group-card {
            background-color: #FFFFFF;
            border-radius: 10px;
            border: 2px solid #FF4B00;
            margin-bottom: 20px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .group-card h4 {
            color: #FF4B00;
            margin-top: 0;
        }
        .group-card .score {
            float: right;
            font-size: 24px;
            color: #FF4B00;
            font-weight: bold;
        }
        .group-card ul {
            list-style: none;
            padding-left: 0;
        }
        .group-card li {
            margin-bottom: 5px;
        }
        .group-card p {
            margin: 0.2rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    if df_grupos.empty:
        st.info("No hay grupos generados.")
        return
    
    if "Tecnicos" not in df_grupos.columns:
        st.error("La columna 'Tecnicos' no está presente en el DataFrame.")
        return
    
    df_grupos["Tecnicos"] = df_grupos["Tecnicos"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    bloques_unicos = sorted(df_grupos["Bloque"].unique())
    
    for bloque in bloques_unicos:
        st.markdown(f"## Bloque {bloque}")
        df_bloque = df_grupos[df_grupos["Bloque"] == bloque].reset_index(drop=True)
        for i in range(0, len(df_bloque), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(df_bloque):
                    row = df_bloque.iloc[i + j]
                    tecnicos = row["Tecnicos"]
                    tecnicos_html = ""
                    for idx, tec in enumerate(tecnicos, 1):
                        tecnicos_html += f"<p><strong>Técnico {idx}:</strong> {tec}</p>"
                    
                    if len(tecnicos) > 1:
                        coverage_score = row["Complementariedad"] * 100
                        overlap_score = row["Conocimiento Común"] * 100
                        expertise_score = row["Conocimiento Especializado"] * 100
                        total_score = row["Compatibilidad Total"] * 100
                        
                        card_html = f"""
                        <div class="group-card">
                          <div class="score">{total_score:.0f}%</div>
                          <h4>Grupo óptimo</h4>
                          {tecnicos_html}
                          <ul>
                            <li><strong>Complementariedad:</strong> {coverage_score:.0f}%</li>
                            <li><strong>Conocimiento común:</strong> {overlap_score:.0f}%</li>
                            <li><strong>Especialización:</strong> {expertise_score:.0f}%</li>
                          </ul>
                        </div>
                        """
                    else:
                        card_html = f"""
                        <div class="group-card">
                          <h4>Técnico Solo</h4>
                          {tecnicos_html}
                          <ul>
                            <li><em>Este técnico trabajará solo y recibirá apoyo cuando sea necesario</em></li>
                          </ul>
                        </div>
                        """
                    
                    with cols[j]:
                        with st.container():
                            # Estilo visual tipo tarjeta
                            st.markdown(f"""
                                <div style="background-color: white; border: 2px solid #FF4B00;
                                            border-radius: 10px; padding: 20px; margin-bottom: 10px;">
                                    <h4 style="color:#FF4B00; margin-top:0;">Grupo de {len(tecnicos)} técnicos</h4>
                                    {''.join([f"<p><strong>Técnico {i+1}:</strong> {tec}</p>" for i, tec in enumerate(tecnicos)])}
                                    <ul>
                                        <li><strong>Complementariedad:</strong> {coverage_score:.0f}%</li>
                                        <li><strong>Conocimiento común:</strong> {overlap_score:.0f}%</li>
                                        <li><strong>Especialización:</strong> {expertise_score:.0f}%</li>
                                    </ul>
                                    <p style="text-align:right; font-size: 24px; color: #FF4B00; font-weight: bold;">{total_score:.0f}%</p>
                                </div>
                            """, unsafe_allow_html=True)

                            # Gráficos dentro del contenedor
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.plotly_chart(
                                    mostrar_gauge_chart(coverage_score / 100, "Complementariedad", color="#00B0F0"),
                                    use_container_width=True,
                                    key=f"gauge_comp_{bloque}_{i+j}"
                                )
                            with col2:
                                st.plotly_chart(
                                    mostrar_gauge_chart(overlap_score / 100, "Conocimiento Común", color="#00B050"),
                                    use_container_width=True,
                                    key=f"gauge_comm_{bloque}_{i+j}"
                                )
                            with col3:
                                st.plotly_chart(
                                    mostrar_gauge_chart(expertise_score / 100, "Especialización", color="#C00000"),
                                    use_container_width=True,
                                    key=f"gauge_expert_{bloque}_{i+j}"
                                )


###############################################################################
# 7. EDITAR SHIFT_PATTERNS (sin data_editor)
###############################################################################
def editar_shift_patterns_sin_data_editor():
    st.markdown("### Editar Patrones de Turnos (Manual)")
    st.write("Modifica, patrón por patrón, los turnos de cada día (Lunes..Domingo).")
    header_cols = st.columns(8)
    header_cols[0].markdown("**Turno**")
    for i, day in enumerate(DAYS_OF_WEEK):
        header_cols[i+1].markdown(f"**{day}**")
    config = get_config()
    shift_patterns = config["shift_patterns"]
    new_patterns = []
    for idx, pattern in enumerate(shift_patterns):
        row_cols = st.columns(8)
        row_cols[0].markdown(f"**Turno {idx+1}**")
        row_vals = []
        for j, turno in enumerate(pattern):
            new_val = row_cols[j+1].selectbox(
                label=f"pattern_{idx}_day_{j}",
                options=TURNOS_DISPONIBLES,
                index=TURNOS_DISPONIBLES.index(turno) if turno in TURNOS_DISPONIBLES else 0,
                label_visibility="collapsed"
            )
            row_vals.append(new_val)
        new_patterns.append(row_vals)
    if st.button("Guardar Cambios en Shift Patterns"):
        config["shift_patterns"] = new_patterns
        st.success("shift_patterns actualizado correctamente.")
        st.write("Nuevos shift_patterns:")
        st.write(new_patterns)

###############################################################################
# 8. CONFIGURACIONES AVANZADAS
###############################################################################
def configuraciones_avanzadas_tab():
    st.header("Configuraciones Avanzadas")
    editar_shift_patterns_sin_data_editor()
    
    st.subheader("Otras Variables Clave")
    config = get_config()
    
    new_baja_short = st.number_input("baja_short_days_threshold",
                                     min_value=1,
                                     value=config["baja_short_days_threshold"],
                                     step=1)
    if new_baja_short != config["baja_short_days_threshold"]:
        config["baja_short_days_threshold"] = new_baja_short
        st.success("Actualizado baja_short_days_threshold.")

    new_penalty = st.number_input("repeated_pairs_penalty",
                                  min_value=1.0,
                                  value=float(config["repeated_pairs_penalty"]),
                                  step=1.0)
    if new_penalty != config["repeated_pairs_penalty"]:
        config["repeated_pairs_penalty"] = new_penalty
        st.success("Actualizado repeated_pairs_penalty.")

    new_max_hours = st.number_input("max_hours_year",
                                    min_value=1,
                                    value=config["max_hours_year"],
                                    step=10)
    if new_max_hours != config["max_hours_year"]:
        config["max_hours_year"] = new_max_hours
        st.success("Actualizado max_hours_year.")

    new_wc = st.slider("Importancia Complementaridad", 0.0, 1.0, config["weight_coverage"], 0.05)
    new_wo = st.slider("Importancia Conocimiento Común", 0.0, 1.0, config["weight_overlap"], 0.05)
    new_we = st.slider("Importancia Experiencia", 0.0, 1.0, config["weight_expertise"], 0.05)

    if (new_wc + new_wo + new_we) > 1.0:
        st.warning("La suma de los pesos supera 1.0 (revisa si es correcto).")

    if (new_wc != config["weight_coverage"] or
        new_wo != config["weight_overlap"] or
        new_we != config["weight_expertise"]):
        config["weight_coverage"] = new_wc
        config["weight_overlap"] = new_wo
        config["weight_expertise"] = new_we
        st.success("Pesos de la fórmula de compatibilidad actualizados.")

    st.subheader("Configuración de Tipos de Habilidades (Edición Manual)")
    st.write("""
        Define el **nombre del tipo** de habilidad y su **importancia (peso)**.
        Por ejemplo, 'electricidad' puede tener un peso de 3 y 'climatizacion' 2.
        - Usa **Eliminar** para quitar una fila.
        - Usa **Añadir Fila** para crear una nueva.
        - Al finalizar, pulsa **Guardar Cambios**.
    """)

    if "skill_types" not in config:
        config["skill_types"] = [
            {"Tipo": "electricidad", "Importancia": 3},
            {"Tipo": "climatizacion", "Importancia": 2}
        ]

    if "skill_types_temp" not in st.session_state:
        st.session_state.skill_types_temp = [dict(x) for x in config["skill_types"]]

    i = 0
    while i < len(st.session_state.skill_types_temp):
        skill_row = st.session_state.skill_types_temp[i]
        cols = st.columns([3, 2, 1])
        new_tipo = cols[0].text_input(
            label="Tipo",
            value=skill_row["Tipo"],
            key=f"tipo_{i}"
        )
        new_importancia = cols[1].number_input(
            label="Importancia",
            min_value=1,
            value=skill_row["Importancia"],
            step=1,
            key=f"importancia_{i}"
        )
        remove_btn = cols[2].button("Eliminar", key=f"remove_{i}")
        if remove_btn:
            st.session_state.skill_types_temp.pop(i)
            raise st.runtime.scriptrunner.script_runner.RerunException(st.runtime.scriptrunner.script_runner.RerunData(None))
        else:
            st.session_state.skill_types_temp[i]["Tipo"] = new_tipo
            st.session_state.skill_types_temp[i]["Importancia"] = new_importancia
            i += 1

    if st.button("Añadir Fila"):
        st.session_state.skill_types_temp.append({"Tipo": "", "Importancia": 1})
        raise st.runtime.scriptrunner.script_runner.RerunException(st.runtime.scriptrunner.script_runner.RerunData(None))

    if st.button("Guardar Cambios en Tipos de Habilidades"):
        config["skill_types"] = st.session_state.skill_types_temp
        st.success("Configuración de tipos de habilidades actualizada.")
        st.write("Nuevos valores:")
        st.write(config["skill_types"])

###############################################################################
# 9. LOGIN 
###############################################################################
import streamlit as st
import os
import base64
import auth as auth

###############################################################################
# 9. STREAMLIT APP PRINCIPAL
###############################################################################
def main():
    init_config()
    config = get_config()

    st.markdown(
    """
    <script>
      document.title = "Guaxen";       // aquí pones exactamente lo que quieras
    </script>
    """,
    unsafe_allow_html=True
)
    init_config()

    auth.auth_gate()
    st.markdown("""
    <style>
    body { background-color: #FFF5F0; }
    .main, .block-container { background-color: #FFF5F0; padding: 2rem; border-radius: 0.5rem; }
    header, .css-18e3th9 { background-color: #FF4B00 !important; }
    h1 { font-size: 3rem; color: #FF4B00; font-weight: 900; margin-bottom: 0.5rem; }
    h2, h4 { color: #FF4B00; font-weight: 700; }
    h3 { color: #333333; margin-top: 0; }
    .logo-container { display: flex; align-items: center; justify-content: left; margin-bottom: 1rem; }
    .logo-container img { height: 80px; margin-right: 1rem; }
    .stTabs [role="tablist"] button { background-color: transparent; color: #FF4B00; border: none; font-weight: 600; padding: 0.6rem 1rem; }
    .stTabs [role="tablist"] button:hover { background-color: #FFF5F0; cursor: pointer; }
    .stTabs [role="tablist"] button[aria-selected="true"] { color: #FF4B00; background-color: transparent; border-bottom: 3px solid #FF4B00; font-weight: 700; padding-bottom: 0.4rem; }
    .stButton button { background-color: #FF4B00; color: #FFFFFF; border-radius: 4px; font-weight: bold; border: 1px solid #FF4B00; padding: 0.6rem 1rem; }
    .stButton button:hover { background-color: #FF6A2A; border: 1px solid #FF6A2A; }
    </style>
    """, unsafe_allow_html=True)

    if "usuario" not in st.session_state or "pep" not in st.session_state:
        st.stop()  # Esto detiene toda la ejecución aquí
    if st.sidebar.button("Cerrar sesión"):
        auth.logout()
    init_config()
    config = get_config()
    if config["logo_path"] and os.path.exists(config["logo_path"]):
        with open(config["logo_path"], "rb") as f:
            img = f.read()
        logo_base64 = base64.b64encode(img).decode()
        st.markdown(f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{logo_base64}" alt="Logo">
        </div>
        """, unsafe_allow_html=True)
    st.markdown("""
    <h1>Guaxen: Gestión de Turnos Automatizada</h1>
    <h3>Optimiza la asignación de turnos de forma automática y sencilla.</h3>
    """, unsafe_allow_html=True)
    tabs = st.tabs(["Inicio", "Calendario y Bajas", "Parejas Generadas", "Configuraciones Avanzadas"])
    with tabs[0]:
        st.header("¡Bienvenido!")
        st.write("""
       Esta aplicación ha sido diseñada para ayudarte a organizar y optimizar la asignación de turnos en tu empresa de forma automática y sencilla. A continuación, te explicamos paso a paso cómo funciona y cómo puedes empezar:

        ## Funcionalidades Principales
        - **Emparejamiento Inteligente de Técnicos:** Se generan automáticamente emparejamientos de técnicos basados en sus habilidades, organizados en bloques para maximizar la complementariedad y eficiencia. Se puede seleccionar las habilidades más importantes.
        - **Generación de Calendario Anual:** Crea un calendario de turnos para todo el año, con una visualización mensual que facilita la planificación.
        - **Calendario Moficiable**: Una vez creado el calendario, es posible realizar las modificaciones que se deseen sobre el mismo en el apartado pertinente.
        - **Incorporaciones:** Permite incorporar técnicos durante el año e incorporarlo a los turnos sin modificar el del resto.
        - **Gestión de Bajas:** Administra las ausencias de los técnicos. Puedes seleccionar el técnico que estará de baja, la fecha de inicio y la duración de la ausencia. La aplicación se encargará de redistribuir los turnos automáticamente.
        - **Resumen de Turnos y Horas:** Obtén informes detallados de turnos asignados y horas trabajadas, a nivel mensual, anual yt acumulado hasta la fecha.
        - **Configuración Avanzada:** Personaliza parámetros clave, como la importancia de cada tipo de habilidad, patrones de turnos, y otros ajustes relevantes.

        ## Pasos para Empezar

        1. **Sube la Matriz de Polivalencia:**
        - Dirígete a la pestaña **"Calendario y Bajas"**.
        - Sube el archivo Excel que contiene la matriz de polivalencia. Este documento debe incluir la información de habilidades de los técnicos.
        2. **Genera el Calendario:**
        - Ingresa el año deseado y el número de bloques (cambios de parejas).
        - Haz clic en **"Generar Turnos"**. La aplicación procesará el Excel y generará automáticamente el calendario de turnos.
        3. **Visualiza y Navega:**
        - Utiliza las flechas (◀️ y ▶️) para navegar por los diferentes meses del calendario.
        - Consulta los resúmenes de turnos y horas, así como el detalle de los emparejamientos de técnicos.
        4. **Gestiona las Bajas:**
        - Selecciona el técnico que necesite baja, la fecha de inicio y la duración de la ausencia.
        - Opcionalmente, selecciona un retén (técnico que cubrirá la baja).
        - Pulsa **"Aplicar Baja"** para actualizar el calendario.
        5. **Configura los Parámetros Avanzados:**
        - En la pestaña **"Configuraciones Avanzadas"** podrás ajustar aspectos como la importancia de cada tipo de habilidad (por ejemplo, definir que "electricidad" tiene mayor importancia que "climatización"), así como los patrones de turnos y otros parámetros.
        - Estos ajustes te permitirán adaptar la aplicación a las necesidades específicas de tu organización.

        ## Recomendaciones

        - **Guardar Cambios:** Una vez generado el calendario, la aplicación guarda los datos localmente. Si cierras o recargas la aplicación, se recuperará el último calendario guardado.
        - **Forzar Recálculo:** Si deseas empezar desde cero, utiliza el botón **"Forzar recálculo"** para borrar los archivos locales y generar un nuevo calendario.
        - **Consulta la Documentación:** Si tienes alguna duda, revisa las instrucciones en cada pestaña o contacta al administrador para más detalles.

        ¡Explora todas las funcionalidades y optimiza la gestión de turnos en tu organización!
    
        """)
        if os.path.exists(config["template_file"]):
            with open(config["template_file"], "rb") as file:
                plantilla_data = file.read()
            st.download_button(
                label="Descargar plantilla Excel",
                data=plantilla_data,
                file_name=config["template_file"],
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("No se encontró la plantilla local. Sube tu propio Excel de polivalencia.")
    with tabs[1]:
        st.header("Calendario y Bajas")
        if st.button("Forzar recálculo"):
            folder = get_contract_folder()
            try:
                os.remove(os.path.join(folder, "calendario_turnos.csv"))
            except FileNotFoundError:
                pass
            try:
                os.remove(os.path.join(folder, "calendario_parejas.csv"))
            except FileNotFoundError:
                pass
            st.session_state.pop("df_turnos", None)
            st.session_state.pop("df_parejas", None)
            st.warning("Archivos locales borrados. Se recargará la app.")
            raise st.runtime.scriptrunner.script_runner.RerunException(st.runtime.scriptrunner.script_runner.RerunData(None))
        if 'df_turnos' not in st.session_state and 'df_parejas' not in st.session_state:
            df_turnos_local, df_parejas_local = cargar_calendario_local()
            if df_turnos_local is not None and df_parejas_local is not None:
                st.session_state.df_turnos = df_turnos_local
                st.session_state.df_parejas = df_parejas_local
                st.success("Calendario cargado desde disco (CSV).")
            else:
                st.info("No hay calendario guardado. Sube un Excel y genera uno nuevo.")
        if 'df_turnos' not in st.session_state:
            st.subheader("Generar Calendario")
            uploaded_file = st.file_uploader("Sube la matriz de polivalencia (Excel)", type=["xlsx"])
            anio = st.number_input("Año para el calendario", value=2025, step=1)
            num_tecnicos = st.number_input("Número total de técnicos", min_value=1, value=10, step=1)
            num_bloques = st.number_input("Número de cambios de grupos (bloques)", value=2, step=1)
            distribucion = calcular_distribucion_optima(num_tecnicos)
            mostrar_explicacion_distribucion(distribucion, num_tecnicos)
            usar_recomendacion = st.checkbox("Usar distribución recomendada", value=True)
            
            if not usar_recomendacion:
                st.warning("""
                Si decides no usar la distribución recomendada, ten en cuenta que:
                - Los grupos deberían mantener una jerarquía de tamaños.
                - Evita mezclar grupos con diferencias de más de un integrante.
                - Los técnicos solos solo son recomendables cuando trabajas con parejas.
                """)
                techs_per_shift = st.number_input(
                    "Número de técnicos por turno",
                    min_value=1,
                    max_value=num_tecnicos,
                    value=distribucion[0][0],
                    step=1,
                    help="Especifica cuántos técnicos trabajarán juntos en cada turno"
                )
            else:
                techs_per_shift = distribucion[0][0]
            
            if uploaded_file is not None and st.button("Generar Turnos"):
                with st.spinner("Generando calendario..."):
                    try:
                        config = get_config()
                        config["technicians_per_shift"] = techs_per_shift
                        
                        if usar_recomendacion:
                            df_turnos, df_parejas = generar_calendario_anual_grupos_mixtos(
                                uploaded_file,
                                anio=anio,
                                num_bloques=num_bloques,
                                distribucion=distribucion,
                                num_tecnicos=num_tecnicos
                            )
                        else:
                            df_turnos, df_parejas = generar_calendario_anual_grupos(
                                uploaded_file,
                                anio=anio,
                                num_bloques=num_bloques,
                                penalizar_repetidas=True,
                                num_tecnicos=num_tecnicos,
                                techs_per_shift=techs_per_shift
                            )
                        
                        st.session_state.df_turnos = df_turnos
                        st.session_state.df_parejas = df_parejas
                        guardar_calendario_local(df_turnos, df_parejas)
                        st.success("¡Calendario generado y guardado correctamente!")
                        raise st.runtime.scriptrunner.script_runner.RerunException(st.runtime.scriptrunner.script_runner.RerunData(None))
                    except Exception as e:
                        st.error(f"Error generando calendario: {e}")
        else:
            df = st.session_state.df_turnos
            meses = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
            if "current_month_index" not in st.session_state:
                hoy = datetime.today()
                st.session_state.current_month_index = hoy.month - 1  # mes actual (0 = enero, 11 = diciembre)
            col1, col2, col3 = st.columns([1,2,1])
            with col1:
                if st.button("◀️", key="month_prev"):
                    if st.session_state.current_month_index > 0:
                        st.session_state.current_month_index -= 1
            with col3:
                if st.button("▶️", key="month_next"):
                    if st.session_state.current_month_index < 11:
                        st.session_state.current_month_index += 1
            current_month = meses[st.session_state.current_month_index]
            mostrar_calendario_mensual(df, current_month)
            st.markdown("---")
            mostrar_resumen_anual(df)
            st.markdown("---")
            mostrar_resumen_anual_horas_sep_cobertura(df)
            st.markdown("---")
            mostrar_resumen_hasta_hoy(df)
            mostrar_resumen_hasta_hoy_horas_sep(df)
            st.markdown("---")
            with st.expander("🧮 Editor manual estilo calendario mensual"):
                df_mes = df[df["Mes"].str.lower() == current_month.lower()].copy()
                df_mes["Day"] = df_mes["Fecha"].dt.day

                dias_mes = sorted(df_mes["Day"].unique())
                id_cols = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "MesNum", "Day"]
                tech_cols = [c for c in df_mes.columns if c not in id_cols]

                opciones_turno = ["", "M", "T", "N", "R", "V", "D", "M+T", "T+N"]

                # Construir tabla calendario estilo horizontal (técnico x día)
                data = []
                for tecnico in tech_cols:
                    fila = {"Técnico": tecnico}
                    for dia in dias_mes:
                        turno = df_mes[df_mes["Day"] == dia][tecnico].values[0]
                        fila[str(dia)] = turno
                    data.append(fila)

                df_editor = pd.DataFrame(data)
                df_editor = df_editor.set_index("Técnico").reset_index()

                # Editor tipo Excel horizontal
                st.markdown("🎨 **Edita turnos directamente en la tabla** (como Excel):")
                edited_df = st.data_editor(
                    df_editor,
                    num_rows="fixed",
                    use_container_width=True,
                    height=550,
                    column_config={
                        col: st.column_config.SelectboxColumn(
                            label=col,
                            options=opciones_turno,
                            width="small"
                        ) for col in df_editor.columns if col != "Técnico"
                    }
                )

                # Guardar cambios en el DataFrame original
                if st.button("💾 Guardar cambios del mes (manual)"):
                    for _, row in edited_df.iterrows():
                        tecnico = row["Técnico"]
                        for dia in dias_mes:
                            df_mes.loc[df_mes["Day"] == dia, tecnico] = row[str(dia)]

                    df.update(df_mes)
                    st.session_state.df_turnos = df
                    guardar_calendario_local(df, st.session_state.df_parejas)
                    st.success(f"Cambios guardados para {current_month.capitalize()}.")
                    raise st.rerun()




            with st.expander("➕ Incorporar técnico nuevo"):
                nuevo_tecnico = st.text_input("Nombre del nuevo técnico")
                fecha_inicio_nuevo = st.date_input("Fecha de incorporación", datetime(2025, 5, 1))

                if st.button("Incorporar Técnico"):
                    df_nuevo = incorporar_tecnico_en_mayo(
                        st.session_state.df_turnos,
                        st.session_state.df_parejas,
                        nuevo_tecnico,
                        fecha_inicio_nuevo
                        )
                    st.session_state.df_turnos = df_nuevo
                    guardar_calendario_local(df_nuevo, st.session_state.df_parejas)
                    st.success(f"Técnico {nuevo_tecnico} incorporado correctamente.")
                    raise st.runtime.scriptrunner.script_runner.RerunException(st.runtime.scriptrunner.script_runner.RerunData(None))

            st.subheader("Aplicar Baja")
            id_cols = ["Fecha", "Día", "Mes", "Día Corto", "Bloque", "Day", "MesNum"]
            tech_cols = [c for c in df.columns if c not in id_cols]
            if tech_cols:
                tecnico_baja = st.selectbox("Técnico a dar de baja", tech_cols)
                fecha_inicio_baja = st.date_input("Fecha de inicio de baja", datetime(2025, 1, 1))
                fecha_fin_baja = st.date_input("Fecha de fin de baja", datetime(2025, 1, 5))
                ret_options = ["(ninguno)"] + tech_cols
                ret_pref_sel = st.selectbox("Retén preferido (opcional)", ret_options)
                ret_pref = None if ret_pref_sel == "(ninguno)" else ret_pref_sel
                if st.button("Aplicar Baja"):
                    fecha_inicio_str = fecha_inicio_baja.strftime("%d/%m/%Y")
                    fecha_fin_str = fecha_fin_baja.strftime("%d/%m/%Y")
                    df_mod = aplicar_baja(df.copy(), tecnico_baja, fecha_inicio_str, fecha_fin_str, ret_pref)
                    st.session_state.df_turnos = df_mod
                    guardar_calendario_local(st.session_state.df_turnos, st.session_state.df_parejas)
                    st.success("Baja aplicada. Calendario y archivo local actualizados.")
                    raise st.runtime.scriptrunner.script_runner.RerunException(st.runtime.scriptrunner.script_runner.RerunData(None))
            st.subheader("Aplicar Grupo de Refuerzo")
            fecha_inicio = st.date_input("Fecha de inicio refuerzo", datetime(2025, 1, 1))
            dias_refuerzo = st.number_input("Días de refuerzo", min_value=1, max_value=14, value=7)
            turno_refuerzo = st.selectbox("Turno requerido", ["M", "T", "N", "M+T", "T+N"])
            num_tecnicos_extra = st.number_input("Número de técnicos extra requeridos", min_value=1, value=2)

            tecnicos_disponibles = [col for col in df.columns if col not in id_cols]
            tecnicos_manuales = st.multiselect("Selecciona técnicos específicos para refuerzo (opcional)", tecnicos_disponibles)

            if st.button("Asignar Grupo de Refuerzo"):
                fecha_inicio_str = fecha_inicio.strftime("%d/%m/%Y")
                df_modificado = aplicar_grupo_refuerzo(
                    st.session_state.df_turnos.copy(), 
                    fecha_inicio_str, 
                    dias_refuerzo, 
                    turno_refuerzo, 
                    num_tecnicos_extra,
                    tecnicos_seleccionados=tecnicos_manuales if tecnicos_manuales else None
                )
                st.session_state.df_turnos = df_modificado
                guardar_calendario_local(st.session_state.df_turnos, st.session_state.df_parejas)
                st.success("Grupo de refuerzo aplicado con éxito.")
                raise st.runtime.scriptrunner.script_runner.RerunException(st.runtime.scriptrunner.script_runner.RerunData(None))

    with tabs[2]:
        st.header("Emparejamientos (Parejas) Generadas")
        if 'df_parejas' in st.session_state:
            mostrar_parejas_en_cards_por_bloque(st.session_state.df_parejas)
        else:
            st.warning("No se han generado parejas aún. Ve a 'Calendario y Bajas' para generar el calendario.")
    with tabs[3]:
        configuraciones_avanzadas_tab()



def calcular_distribucion_optima(num_tecnicos):
    """
    Calcula la distribución óptima en 5 grupos.
    """
    tam_base = num_tecnicos // 5
    tecnicos_restantes = num_tecnicos % 5
    
    if tam_base == 0:
        grupos_con_uno = 5 - num_tecnicos
        return [(1, num_tecnicos), (1, grupos_con_uno)]
    
    distribucion = []
    
    if tecnicos_restantes > 0:
        distribucion.append((tam_base + 1, tecnicos_restantes))
    
    grupos_base = 5 - tecnicos_restantes
    if grupos_base > 0:
        distribucion.append((tam_base, grupos_base))
    
    return distribucion

def mostrar_explicacion_distribucion(distribucion, num_tecnicos):
    st.write("### Distribución recomendada:")
    for tam_grupo, num_grupos in distribucion:
        if tam_grupo == 4:
            st.write(f"- {num_grupos} grupo{'s' if num_grupos > 1 else ''} de 4 técnicos")
        elif tam_grupo == 3:
            st.write(f"- {num_grupos} grupo{'s' if num_grupos > 1 else ''} de 3 técnicos")
        elif tam_grupo == 2:
            st.write(f"- {num_grupos} grupo{'s' if num_grupos > 1 else ''} de 2 técnicos")
        else:
            st.write(f"- {num_grupos} técnico{'s' if num_grupos > 1 else ''} como refuerzo")
    st.info(f"""
        Esta distribución se ha calculado siguiendo estas reglas:
        - Siempre hay exactamente 5 grupos.
        - Los grupos se equilibran en número de personas.
        - Se maximiza el tamaño de los grupos dentro de estas restricciones.
        - Los técnicos de refuerzo trabajarán en turno R para apoyar a los grupos principales.
        """)

if __name__ == "__main__":
    main()

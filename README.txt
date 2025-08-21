# 🔄 Guaxen – Optimización de Turnos en Data Centers  

## 📌 Descripción  
**Guaxen** es una aplicación web desarrollada en **Streamlit** que automatiza la planificación de turnos en centros de datos, optimizando el **emparejamiento de técnicos** y la **asignación de turnos** en entornos 24/7.  

El proyecto surge de un reto real en un **Data Center**, donde la planificación manual generaba:  
- ❌ Emparejamientos ineficientes  
- ⏳ Alta carga administrativa para supervisores  
- ⚠️ Riesgo de errores humanos  

Con esta herramienta se consigue una **planificación ágil, justa y transparente**, reduciendo la intervención manual y optimizando la operación del centro.  

## ⚡ Funcionalidades principales  

✅ **Login seguro** con control por contrato  
✅ **Gestión multi-fase** (ej. Fase 1 y Fase 2 del Data Center)  
✅ **Emparejamiento óptimo** de técnicos con el **Método Húngaro** de optimización  
✅ **Matriz de polivalencia dinámica**, con mayor peso en especialidades críticas (Eléctrica, Climatización)  
✅ **Calendario anual interactivo** con opción de edición manual  
✅ **Gestión de bajas y refuerzos**, respetando descansos y cargas de trabajo  
✅ **Incorporación de técnicos a mitad de año** sin alterar turnos previos  
✅ **Interfaz intuitiva** con estadísticas y visualización en tarjetas  

---

## 🛠️ Tecnologías utilizadas  

- [Streamlit](https://streamlit.io/) – Framework para la interfaz web  
- [Pandas](https://pandas.pydata.org/) – Gestión de datos y calendarios  
- [NetworkX](https://networkx.org/) – Modelado de grafos para emparejamiento  
- [SciPy](https://scipy.org/) – Algoritmos de optimización (Método Húngaro, etc.)  
- [Python](https://www.python.org/) – Lógica principal del sistema  



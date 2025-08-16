# 📦 Importaciones
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
from streamlit_lottie import st_lottie
import shap
import statsmodels.api as sm
from st_aggrid import GridOptionsBuilder, AgGrid
import base64

# Paleta de colores
#colores = {
#    "fondo": "#2A2853",
#    "texto": "#282DB8",
#    "aprobado": "#99E1F9",
#    "no_aprobado": "#D9C2F6",
#    "barra": "#886FE7",
#    "borde": "#7079ED"
#}



# Fuente futurista Orbitron + fondo global
st.markdown("""
    <style>
        /* Fondo global azul Pantone 201747 */
        body, .css-18e3th9, .css-1outpf7 {
            background-color: #001747 !important;
        }

        html, body {
            font-family: 'Orbitron', sans-serif;
        }

        h1, h2, h3, h4 {
            color: #99E1F9;  /* texto más visible sobre azul oscuro */
        }

        div[data-testid="stTabs"] button {
            color: #D9C2F6 !important;
            font-family: 'Orbitron', sans-serif;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #282DB8 !important;
            font-weight: bold;
            border-bottom: 2px solid #7079ED;
        }

        .stButton > button {
            background-color: #886FE7;
            color: white;
            border: 2px solid #7079ED;
            font-family: 'Orbitron', sans-serif;
            transition: 0.3s ease;
        }

        .stButton > button:hover {
            background-color: #99E1F9;
            color: #282DB8;
        }
    </style>
""", unsafe_allow_html=True)


# ⚙️ Configuración de la página
st.set_page_config(page_title="Evaluación de microcréditos", layout="wide")

# 🧠 Cargar modelo
try:
    rf_model = joblib.load("src/modelo_bfr.pkl")
    columnas_modelo_original = getattr(rf_model, "feature_names_in_", None)
    if columnas_modelo_original is None:
        st.error("⚠️ El modelo no contiene información sobre las columnas esperadas.")
        st.stop()
    # Limpiar nombres igual que en el CSV
    columnas_modelo = [col.strip().lower().replace(" ", "_") for col in columnas_modelo_original]
except FileNotFoundError:
    st.error("❌ No se encontró el archivo del modelo 'modelo_bfr.pkl'.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error al cargar el modelo: {e}")
    st.stop()

# 🖼️ Pestañas de navegación
tab1, tab2, tab3 = st.tabs(["Inicio", "Cargar CSV", "Evaluación Individual"])

def funcion_card(titulo, descripcion):
    st.markdown(f"""
        <!-- Fuentes -->
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500&family=Oswald:wght@600;700&display=swap" rel="stylesheet">
        
        <div style='
            background-color:#D9C2F6;
            padding:20px;
            border-radius:20px;
            margin:15px auto;
            box-shadow:0 4px 12px rgba(0,0,0,0.1);
            max-width:500px;
            text-align:center;
        '>
            <h4 style='
                font-family: "Oswald", sans-serif;
                color:#282DB8;
                font-size:22px;
                font-weight:700;  /* Más fuerte y destacado */
                margin-bottom:10px;
            '>{titulo}</h4>
            <p style='
                font-family: "Montserrat", sans-serif;
                font-size:16px;
                color:black;
                margin:0;
                line-height:1.5;
            '>{descripcion}</p>
        </div>
    """, unsafe_allow_html=True)


# 🏠 TAB 1: Inicio
with tab1:
        # Leer la imagen local
    with open("src/fondo_micro_banner.jpg", "rb") as f:
        data = f.read()
    img_base64 = base64.b64encode(data).decode()
    
    # Banner con titulo fuente Oswald, márgenes y dimensiones personalizadas
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@700&display=swap" rel="stylesheet">
        <div style="
            position: relative; 
            text-align: center; 
            color: white; 
            margin: 0px auto;  /* margen superior e inferior */
            width: 100%;           /* ancho del banner */
            max-width: 1200px;    /* ancho máximo */
            height: 200px;        /* altura del banner */
            overflow: hidden;
        ">
            <img src="data:image/jpeg;base64,{img_base64}" 
                alt="Banner" 
                style="
                    width: 100%; 
                    height: 100%; 
                    object-fit: cover; 
                    filter: brightness(70%);
                ">
            <h1 style="
                position: absolute; 
                top: 50%; 
                left: 50%; 
                transform: translate(-50%, -50%); 
                font-size: 48px; 
                font-weight: bold; 
                font-family: 'Oswald', sans-serif;
                margin: 0;
            ">
                Predicción de Riesgo Crediticio
            </h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
        <h1 style='
            font-family: "Montserrat", sans-serif;
            font-size:20px;
            color:black;
            margin: 20px auto;
        '>
        Hemos diseñado tres opciones principales para que la gestión de riesgos sea más eficiente y adaptada a tus necesidades
        </h3>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        funcion_card(
            "Subir archivos CSV",
            "Sube un archivo con múltiples clientes para análisis masivo y rápido."
        )

    with col2:
        funcion_card(
            "Evaluación individual",
            "Evalúa el riesgo de un cliente específico ingresando sus datos."
    )

        # Ruta de la imagen local
    image_path = "src/AI-credit-Scoring-2.jpg"

    # Leer y codificar la imagen
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()

    # Insertar en el markdown
    st.markdown(
        f"""
        <div style='
            margin-top: 30px;
            text-align: center;
        '>
            <img src="data:image/jpeg;base64,{encoded_image}" 
                style='width:100%; max-width:1200px; height:auto'>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 🧾 Pie de página
    st.markdown("""
    <hr style="border:1px solid #ccc">
    <p style='text-align: center; color: gray; font-size: 12px;'>
    Desarrollado por Alejandra, Marta y Jose Luis • Evaluación de Microcréditos © 2025
    </p>
    """, unsafe_allow_html=True)

# 🏠 TAB 2: Cargar CSV y evaluación
with tab2:
    st.header("Carga de datos")
    archivo_csv = st.file_uploader("Carga el archivo CSV de los potenciales clientes para ejecutar las predicciones", type=["csv"])

    if archivo_csv is not None:
        # Leer CSV
        df_clientes = pd.read_csv(archivo_csv)

        # Nombres legibles para vista previa
        df_clientes_visual = df_clientes.copy()
        df_clientes_visual.columns = df_clientes_visual.columns.str.replace("_", " ").str.title()

        columnas_modelo = getattr(rf_model, "feature_names_in_", None)
        faltantes = [col for col in columnas_modelo if col not in df_clientes.columns]
        if faltantes:
            st.error(f"Faltan columnas necesarias para el modelo: {', '.join(faltantes)}")
            st.stop()

        # Convertir columnas numéricas
        for col in columnas_modelo:
            df_clientes[col] = pd.to_numeric(df_clientes[col], errors="coerce")

        st.success("Archivo cargado correctamente")

        with st.expander("Vista previa del archivo CSV"):
            st.dataframe(df_clientes_visual.head())

        # Predicciones
        try:
            prob = rf_model.predict_proba(df_clientes)[:, 1]
        except Exception as e:
            st.error(f"Error al ejecutar el modelo: {e}")
            st.stop()

        # Construcción de resultados
        df_resultado = df_clientes.copy()
        df_resultado["probabilidad impago"] = prob
        df_resultado["Riesgo"] = np.where(df_resultado["probabilidad impago"] < 0.8, "No riesgo", "Riesgo")

        # Botón de descarga
        csv = df_resultado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar predicciones .CSV",
            data=csv,
            file_name='predicciones.csv',
            mime='text/csv'
        )

        # ---------------------------
        # 1️⃣ Visión general
        # ---------------------------
        with st.expander("1. Visión general"):
            # Vista previa de resultados
            st.subheader("Vista previa de resultados con predicciones")
            st.dataframe(df_resultado.head())

            st.subheader("Distribución de probabilidad de impago")
            fig_hist = px.histogram(df_resultado, 
            x="probabilidad impago", 
            nbins=20,
            color="Riesgo", 
            title="Distribución de probabilidad de impago",
            color_discrete_map={"Riesgo": "#D81B60", "No riesgo": "#1E88E5"}  # <-- colores personalizados
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Conteo de clientes por riesgo")
            conteo_riesgo = df_resultado["Riesgo"].value_counts().reset_index()
            conteo_riesgo.columns = ["Riesgo", "Cantidad"]
            st.dataframe(conteo_riesgo)

            st.subheader("Promedio de características por tipo de cliente")
            # quitamos columna cliente para que no contamine promedios
            df_promedios = df_resultado.drop(columns=["probabilidad impago"], errors="ignore").groupby("Riesgo").mean().round(2)
            st.dataframe(df_promedios)

        # ---------------------------
        # 2️⃣ Segmentación y patrones
        # ---------------------------
        with st.expander("2. Segmentación y patrones"):
            # Importancia de variables
            importancias = rf_model.feature_importances_
            features = rf_model.feature_names_in_
            df_importancia = pd.DataFrame({"Variable": features, "Importancia": importancias})

            top_n = st.number_input("Selecciona cuántas variables importantes mostrar", min_value=1,
                                    max_value=len(df_importancia), value=10)
            df_importancia_sorted = df_importancia.sort_values(by="Importancia", ascending=False).head(top_n)

            # Ajustar tamaño dinámico del gráfico
            height_chart = 400 if top_n <= 15 else top_n * 25

            fig_imp = px.bar(df_importancia_sorted, x="Importancia", y="Variable", orientation="h",
                             title=f"Top {top_n} variables más importantes",
                             height=height_chart)
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})  # asegura orden descendente
            st.plotly_chart(fig_imp, use_container_width=True)

            # Gráfico de dependencia para variables numéricas
            numeric_cols = df_resultado.select_dtypes(include=["float64", "int64"]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ["probabilidad impago"]]
            if numeric_cols:
                variable_num = st.selectbox("Selecciona variable numérica para ver dependencia", numeric_cols)
                if variable_num:
                    fig_dep = px.scatter(
                            df_resultado,
                            x=variable_num,
                            y="probabilidad impago",
                            color="Riesgo",
                            title=f"Dependencia de {variable_num} con probabilidad de impago",
                            color_discrete_map={"Riesgo": "#D81B60", "No riesgo": "#1E88E5"})  # <-- colores personalizados
                    st.plotly_chart(fig_dep, use_container_width=True)

            # Gráficos de frecuencia para variables no numéricas vs Riesgo
            non_numeric_cols = df_resultado.select_dtypes(exclude=["float64", "int64"]).columns.tolist()
            non_numeric_cols = [c for c in non_numeric_cols if c != "Riesgo"]
            if non_numeric_cols:
                variable_cat = st.selectbox("Selecciona variable categórica para ver frecuencia por riesgo", non_numeric_cols)
                if variable_cat:
                    df_freq_riesgo = df_resultado.groupby([variable_cat, "Riesgo"]).size().reset_index(name="Cantidad")
                    fig_cat = px.bar(df_freq_riesgo, x=variable_cat, y="Cantidad", color="Riesgo",
                        barmode="group", text="Cantidad",
                        title=f"Frecuencia de {variable_cat} por riesgo",
                        color_discrete_map={"Riesgo": "#D81B60", "No riesgo": "#1E88E5"} ) # <-- colores personalizados
                    st.plotly_chart(fig_cat, use_container_width=True)

        # ---------------------------
        # 3️⃣ Detalle individual
        # ---------------------------
        with st.expander("Detalle individual"):
            st.subheader("Explicación de predicción individual")

            cliente_idx = st.number_input("Selecciona índice de cliente", 0, len(df_clientes)-1, 0)

            if st.button("Generar explicación para este cliente"):
                try:
                    # Crear explainer (probabilidades o raw, según tu preferencia)
                    explainer = shap.TreeExplainer(rf_model, model_output="raw")

                    # Calcular explicaciones (esto ya devuelve un Explanation)
                    shap_values = explainer(df_clientes)

                    # 🔹 Seleccionar solo la instancia del cliente
                    shap_cliente = shap_values[cliente_idx]

                    # 🔹 Si es clasificación binaria, quedarse con la clase 1
                    if shap_cliente.values.ndim == 2:  
                        # shap_cliente.values es (n_features, n_classes)
                        shap_cliente = shap.Explanation(
                            values=shap_cliente.values[:, 1],
                            base_values=shap_cliente.base_values[1],
                            data=shap_cliente.data,
                            feature_names=shap_cliente.feature_names
                        )

                    # Force plot usando API nueva
                    fig = shap.plots.force(shap_cliente, matplotlib=False)

                    # Renderizar en Streamlit
                    import streamlit.components.v1 as components
                    shap_html = f"<head>{shap.getjs()}</head><body>{fig.html()}</body>"
                    components.html(shap_html, height=300)

                except Exception as e:
                    st.error(f"Error en SHAP: {e}")

            # 🧾 Pie de página
            st.markdown("""
            <hr style="border:1px solid #ccc">
            <p style='text-align: center; color: gray; font-size: 12px;'>
            Desarrollado por Alejandra, Marta y Jose Luis • Evaluación de Microcréditos © 2025
            </p>
            """, unsafe_allow_html=True)

# 📤 TAB 3: Evaluación Individual
with tab3:
    st.header("Evaluación Individual")
    
    # Contenedor para mostrar el resultado arriba
    resultado_container = st.container()

    st.write("Introduce los datos del potencial cliente para ejecutar la predicción.")

    # Creación de campos numéricos
    valor_credito = st.number_input("Valor del crédito", min_value=0, key="valor_credito")
    suma_ingresos = st.number_input("Suma de ingresos", min_value=0, key="suma_ingresos")
    antiguedad_ocupacion = st.number_input("Antigüedad en ocupación (años)", min_value=0, key="antiguedad_ocupacion")
    plazo_solicitado = st.number_input("Plazo solicitado (días)", min_value=0, key="plazo_solicitado")
    tamano_empresa = st.number_input("Tamaño de la empresa", min_value=0, key="tamano_empresa")
    estrato = st.number_input("Estrato", min_value=0, key="estrato")
    periocidad_pago = st.number_input("Periodicidad de pago", min_value=0, key="periocidad_pago")

    # Campos categóricos binarios
    tipo_pago = st.selectbox("Tipo de pago", ["0 (sin dato)", "sustitución", "pago_total"], key="tipo_pago")
    tipo_pago_encoded = {
        "tipo_pago_0": int(tipo_pago == "0 (sin dato)"),
        "tipo_pago_sustitucion": int(tipo_pago == "sustitución")
    }

    estado_civil = st.selectbox("Estado civil", ["soltero", "casado", "union libre"], key="estado_civil")
    estado_civil_encoded = {
        "estado_civil_soltero": int(estado_civil == "soltero"),
        "estado_civil_casado": int(estado_civil == "casado"),
        "estado_civil_union libre": int(estado_civil == "union libre")
    }

    ciudad = st.selectbox("Selecciona tu ciudad", ["bogota", "medellin", "cali", "otras"], key="ciudad")
    ciudad_encoded = {
        "ciudad_bogota": int(ciudad == "bogota"),
        "ciudad_medellin": int(ciudad == "medellin"),
        "ciudad_otras": int(ciudad == "otras")
    }

    operador_cel = st.selectbox("Operador de celular", ["claro", "movistar", "tigo"], key="operador_cel")
    operador_cel_encoded = {
        "operador_cel_claro": int(operador_cel == "claro"),
        "operador_cel_movistar": int(operador_cel == "movistar"),
        "operador_cel_tigo": int(operador_cel == "tigo")
    }

    horario_contacto = st.selectbox("Horario de contacto preferido", ["mañana", "otro", "desconocido"], key="horario_contacto")
    horario_contacto_encoded = {
        "horario_contacto_manana": int(horario_contacto == "mañana"),
        "horario_contacto_otro": int(horario_contacto == "otro"),
        "horario_contacto_desconocido": int(horario_contacto == "desconocido")
    }

    sector_ocupacion = st.selectbox("Sector ocupación", ["servicios", "industria", "salud_educacion", "otros"], key="sector_ocupacion")
    sector_ocupacion_encoded = {
        "sector_ocupacion_servicios": int(sector_ocupacion == "servicios"),
        "sector_ocupacion_industria": int(sector_ocupacion == "industria"),
        "sector_ocupacion_salud_educacion": int(sector_ocupacion == "salud_educacion"),
        "sector_ocupacion_otros": int(sector_ocupacion == "otros")
    }

    fuente_origen = st.selectbox("Fuente de origen", ["visita_google", "otro"], key="fuente_origen")
    fuente_origen_encoded = {
        "fuente_origen_visita_google": int(fuente_origen == "visita_google")
    }

    tipo_plan_celular = st.selectbox("Tipo de plan celular", ["prepago", "postpago"], key="tipo_plan_celular")
    tipo_plan_encoded = {
        "tipo_plan_celular_prepago": int(tipo_plan_celular == "prepago")
    }

    ocupacion = st.selectbox("Ocupación", ["empleado_estable", "otro"], key="ocupacion")
    ocupacion_encoded = {
        "ocupacion_empleado_estable": int(ocupacion == "empleado_estable")    
    }

# Construir el DataFrame con los datos introducidos
    datos = {
        "valor_credito": [valor_credito],
        "suma_ingresos": [suma_ingresos],
        "antiguedad_ocupacion": [antiguedad_ocupacion],
        "plazo_solicitado": [plazo_solicitado],
        "tamano_empresa": [tamano_empresa],
        "estrato": [estrato],
        "periocidad_pago": [periocidad_pago],
        **tipo_pago_encoded,
        **estado_civil_encoded,
        **ciudad_encoded,
        **operador_cel_encoded,
        **horario_contacto_encoded,
        **sector_ocupacion_encoded,
        **fuente_origen_encoded,
        **tipo_plan_encoded,
        **ocupacion_encoded
    }

    df_clientes = pd.DataFrame(datos)

    columnas_ordenadas = [
    "valor_credito", "tipo_pago_0", "suma_ingresos", "antiguedad_ocupacion",
    "plazo_solicitado", "tamano_empresa", "estrato", "tipo_pago_sustitucion",
    "periocidad_pago", "horario_contacto_desconocido", "operador_cel_claro",
    "sector_ocupacion_servicios", "ciudad_otras", "estado_civil_soltero",
    "operador_cel_movistar", "ciudad_bogota", "operador_cel_tigo",
    "sector_ocupacion_otros", "estado_civil_union libre",
    "fuente_origen_visita_google", "horario_contacto_manana",
    "estado_civil_casado", "sector_ocupacion_industria",
    "tipo_plan_celular_prepago", "sector_ocupacion_salud_educacion",
    "horario_contacto_otro", "ocupacion_empleado_estable", "ciudad_medellin"
]

# Reordenar columnas
    df_clientes = df_clientes[columnas_ordenadas]

    # Botón de predicción
    if st.button("Ejecutar predicción"):
        try:
            # 🔮 Predicción
            pred = rf_model.predict(df_clientes)
            prob = rf_model.predict_proba(df_clientes)[:, 1]

            # 📊 Resultados
            df_resultado = df_clientes.copy()
            df_resultado["aprobado"] = ["No" if p == 1 else "Sí" for p in pred]
            df_resultado["probabilidad impago"] = (prob * 100).round(2)

            resultado = df_resultado.iloc[0]

            # Mostrar animación y resultado en la parte superior
            with resultado_container:
                if resultado["aprobado"] == "Sí":
                    st.markdown(f"""
                    <div style="background-color:#e6f4ea; padding:20px; border-radius:10px; border-left:5px solid #34a853;">
                        <h3 style="color:#34a853;">✅ Cliente aprobado</h3>
                        <p style="font-size:16px;">Probabilidad de impago: <strong>{resultado['probabilidad impago']}%</strong></p>
                        <p style="color:#555;">El perfil cumple con los criterios financieros establecidos.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color:#fce8e6; padding:20px; border-radius:10px; border-left:5px solid #ea4335;">
                        <h3 style="color:#ea4335;">❌ Cliente no aprobado</h3>
                        <p style="font-size:16px;">Probabilidad de impago: <strong>{resultado['probabilidad impago']}%</strong></p>
                        <p style="color:#555;">El perfil presenta un riesgo financiero elevado.</p>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error al ejecutar el modelo: {e}")
            st.stop()

    # 🧾 Pie de página
    st.markdown("""
    <hr style="border:1px solid #ccc">
    <p style='text-align: center; color: gray; font-size: 12px;'>
    Desarrollado por Alejandra, Marta y Jose Luis • Evaluación de Microcréditos © 2025
    </p>
    """, unsafe_allow_html=True)
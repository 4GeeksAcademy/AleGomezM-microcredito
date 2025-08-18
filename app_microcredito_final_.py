# 📦 Importaciones
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import shap
from st_aggrid import AgGrid
import base64

# 🎨 Paleta de colores (centralizada)
colores = {
    "fondo": "#2A2853",
    "texto": "#282DB8",
    "aprobado": "#99E1F9",
    "no_aprobado": "#D9C2F6",
    "barra": "#D9C2F6",
    "borde": "#7079ED"
}

# 🎨 Estilos globales
st.markdown(f"""
    <style>
        html, body {{
            font-family: 'Orbitron', sans-serif;
        }}

        h1, h2, h3, h4 {{
            color: {colores["aprobado"]};
        }}

        /* Tabs no seleccionados */
        div[data-testid="stTabs"] button p {{
            font-size: 16px !important;
            color: #886FE7 !important;
            'Montserrat', sans-serif !important;
        }}

        /* Eliminar la línea roja nativa */
        div[data-testid="stTabs"] button[aria-selected="true"]::after {{
            content: none !important;
        }}

        /* Texto del tab seleccionado */
        div[data-testid="stTabs"] button[aria-selected="true"] p {{
            font-size: 18px !important;
            color: #282DB8 !important;
            font-weight: bold !important;
        }}

        /* 📌 Botones unificados */
        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{
            background-color: {colores["barra"]};
            color: black !important;
            border: 2px solid {colores["borde"]};
            font-family: 'Orbitron', sans-serif;
            font-size: 18px !important;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            transition: 0.3s ease;
            cursor: pointer;
        }}

        .stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {{
            background-color: {colores["aprobado"]};
            color: {colores["texto"]} !important;
            border: 2px solid {colores["texto"]};
        }}
    </style>
""", unsafe_allow_html=True)

# ⚙️ Configuración de la página
st.set_page_config(page_title="Evaluación de microcréditos", layout="wide")

# 🧠 Cargar modelo
try:
    rf_model = joblib.load("src/modelo_bfr_f.pkl")
    columnas_modelo_original = getattr(rf_model, "feature_names_in_", None)
    if columnas_modelo_original is None:
        st.error("⚠️ El modelo no contiene información sobre las columnas esperadas.")
        st.stop()
    # Limpiar nombres igual que en el CSV
    columnas_modelo = [col.strip().lower().replace(" ", "_") for col in columnas_modelo_original]
except FileNotFoundError:
    st.error("❌ No se encontró el archivo del modelo 'modelo_bfr_f.pkl'.")
    st.stop()
except Exception as e:
    st.error(f"⚠️ Error al cargar el modelo: {e}")
    st.stop()

# Función para convertir imagen a base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Ruta de tu imagen
image_path = "src/Purple Modern Banner2.jpg"
img_base64_ = get_base64_image(image_path)

# Mostrar con margen superior e inferior
st.markdown(
    f"""
    <div style="margin-top:0px; margin-bottom:10px;">
        <img src="data:image/png;base64,{img_base64_}" style="width:100%;">
    </div>
    """,
    unsafe_allow_html=True
)


# 🖼️ Pestañas de navegación
tab1, tab2, tab3 = st.tabs(["Inicio", "Cargar CSV", "Evaluación Individual"])

# 📌 Card reutilizable
def funcion_card(titulo, descripcion):
    st.markdown(f"""
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500&family=Oswald:wght@600;700&display=swap" rel="stylesheet">
        <div style='
            background-color:{colores["no_aprobado"]};
            padding:10px;
            border-radius:10px;
            margin:10px auto;
            box-shadow:0 4px 12px rgba(0,0,0,0.1);
            max-width:600px;
            text-align:center;
        '>
            <h4 style='
                font-family: "Oswald", sans-serif;
                color:{colores["texto"]};
                font-size:26px;
                font-weight:700;
                margin-bottom:10px;
            '>{titulo}</h4>
            <p style='
                font-family: "Montserrat", sans-serif;
                font-size:16px;
                color:black;
                margin:10px;
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
            height: 200px;        /* altura del banner */
            overflow: hidden;
            margin: 0;         /* quitar márgenes */
        ">
            <img src="data:image/jpeg;base64,{img_base64}" 
                alt="Banner" 
                style="
                    width: 100%; 
                    height: 100%; 
                    object-fit: cover; 
                    filter: brightness(90%);
                ">
            <h1 style="
                position: absolute; 
                top: 50%; 
                left: 50%; 
                transform: translate(-50%, -50%); 
                font-size: 68px; 
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
        <p style='
                font-family: "Montserrat", sans-serif;
                font-size:16px;
                color:black;
                margin:30px;
                line-height:1.5;
        '>
        Nuestra plataforma está diseñada para que evalúes el riesgo crediticio de tus clientes en tiempo real, utilizando modelos avanzados de Machine Learning garantizamos decisiones más seguras y basadas en datos.
        </p>
        """, unsafe_allow_html=True)


    st.markdown("""
        <h1 style='
            font-family: "Montserrat", sans-serif;
            font-size:20px;
            color:black;
            margin:30px;
        '>
        Ofrecemos dos opciones principales para que la gestión de riesgos sea más eficiente y adaptada a tus necesidades
        </h3>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        funcion_card(
            "Cargar archivos CSV",
            "Sube un archivo con múltiples clientes para análisis masivo y rápido."
        )

    with col2:
        funcion_card(
            "Evaluación individual",
            "Evalúa el riesgo de un cliente específico ingresando sus datos."
    )

        # Ruta de la imagen local
    image_path = "src/AI-credit-Scoring-4.jpg"

    # Leer y codificar la imagen
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()

    # Insertar en el markdown
    st.markdown(
        f"""
        <div style='
            margin-top: 80px;
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

        # 🔁 Mapeos originales e inversos
        map_periocidad_pago = {'semanal': 0, 'quincenal': 1, 'mensual': 2}
        map_tamano_empresa = {
            'de 1 a 10': 0, 'de 11 a 25': 1, 'de 26 a 50': 2,
            'de 51 a 100': 3, 'de 100 a 500': 4, 'mas de 500': 5
        }
        inv_map_periocidad_pago = {v: k for k, v in map_periocidad_pago.items()}
        inv_map_tamano_empresa = {v: k for k, v in map_tamano_empresa.items()}

        # Validar columnas del modelo
        columnas_modelo = getattr(rf_model, "feature_names_in_", None)
        faltantes = [col for col in columnas_modelo if col not in df_clientes.columns]
        if faltantes:
            st.error(f"Faltan columnas necesarias para el modelo: {', '.join(faltantes)}")
            st.stop()

        # Convertir columnas del modelo a numérico (para predicción)
        for col in columnas_modelo:
            df_clientes[col] = pd.to_numeric(df_clientes[col], errors="coerce")

        # Crear versión visual con valores legibles
        df_clientes_visual = df_clientes.copy()
        for col, mapa in {
            "periocidad_pago": inv_map_periocidad_pago,
            "tamano_empresa": inv_map_tamano_empresa
        }.items():
            if col in df_clientes_visual.columns:
                df_clientes_visual[col] = df_clientes_visual[col].map(mapa)

        df_clientes_visual.columns = df_clientes_visual.columns.str.replace("_", " ").str.title()

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
        df_resultado["Riesgo"] = np.where(df_resultado["probabilidad impago"] < 0.758, "No riesgo", "Riesgo")

        # Aplicar mapeos legibles si los valores son numéricos
        for col, mapa in {
            "periocidad_pago": inv_map_periocidad_pago,
            "tamano_empresa": inv_map_tamano_empresa
        }.items():
            if col in df_resultado.columns and pd.api.types.is_numeric_dtype(df_resultado[col]):
                df_resultado[col] = df_resultado[col].map(mapa)
            df_resultado[col] = df_resultado[col].astype("category")

        # Crear versión visual
        df_resultado_visual = df_resultado.copy()
        df_resultado_visual.columns = df_resultado_visual.columns.str.replace("_", " ").str.title()

        # Botón de descarga
        csv = df_resultado_visual.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar predicciones .CSV",
            data=csv,
            file_name='predicciones.csv',
            mime='text/csv'
        )

        # ---------------------------
        # 1️⃣ Visión general
        # ---------------------------
        with st.expander("1. Visión general de predicciones"):
            st.subheader("Vista previa de resultados con predicciones")
            st.dataframe(df_resultado_visual.head())

            st.subheader("Distribución de probabilidad de impago")
            fig_hist = px.histogram(df_resultado_visual, 
                x="Probabilidad Impago", 
                nbins=20,
                color="Riesgo", 
                title="Distribución de probabilidad de impago",
                color_discrete_map={"Riesgo": "#D81B60", "No riesgo": "#1E88E5"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.subheader("Conteo de clientes por riesgo")
            conteo_riesgo = df_resultado_visual["Riesgo"].value_counts().reset_index()
            conteo_riesgo.columns = ["Riesgo", "Cantidad"]
            st.dataframe(conteo_riesgo)

            st.subheader("Resumen de características por tipo de cliente")
            numericas = df_resultado.select_dtypes(include=["float64", "int64"]).columns.tolist()
            categoricas = df_resultado.select_dtypes(exclude=["float64", "int64"]).columns.tolist()
            categoricas = [c for c in categoricas if c != "Riesgo"]

            df_num = df_resultado.groupby("Riesgo")[numericas].mean().round(2)
            df_cat = df_resultado.groupby("Riesgo")[categoricas].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
            df_resumen = pd.concat([df_num, df_cat], axis=1)
            df_resumen.columns = df_resumen.columns.str.replace("_", " ").str.title()
            st.dataframe(df_resumen)

        # ---------------------------
        # 2️⃣ Segmentación y patrones
        # ---------------------------
        with st.expander("2. Segmentación y patrones"):
            if 'df_resultado_visual' not in locals():
                st.warning("⚠️ Primero carga un archivo CSV válido en la pestaña anterior.")
                st.stop()

            # Importancia de variables
            importancias = rf_model.feature_importances_
            features = rf_model.feature_names_in_
            # Normalizar a porcentaje
            importancias_pct = 100 * (importancias / importancias.sum())
            features_legibles = [f.replace("_", " ").title() for f in features]
            df_importancia = pd.DataFrame({"Variable": features_legibles, "Importancia": importancias_pct})

            top_n = st.number_input("Selecciona cuántas variables importantes mostrar", min_value=1,
                                    max_value=len(df_importancia), value=10)
            df_importancia_sorted = df_importancia.sort_values(by="Importancia", ascending=False).head(top_n)

            height_chart = 400 if top_n <= 15 else top_n * 25

            fig_imp = px.bar(df_importancia_sorted, x="Importancia", y="Variable", orientation="h",
                            title=f"Top de variables más importantes",
                            height=height_chart)
            fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_imp, use_container_width=True)

            # Gráfico de dependencia para variables numéricas
            numeric_cols = df_resultado_visual.select_dtypes(include=["float64", "int64"]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col not in ["Probabilidad Impago"]]
            if numeric_cols:
                variable_num = st.selectbox("Selecciona variable numérica para ver dependencia", numeric_cols)
                if variable_num:
                    # Si es la columna Relacion Credito Ingresos -> aplicar filtro percentil 1-99
                    if variable_num == "Relacion Credito Ingresos":
                        q_low = df_resultado_visual[variable_num].quantile(0.01)
                        q_high = df_resultado_visual[variable_num].quantile(0.995)

                        df_plot = df_resultado_visual[
                            (df_resultado_visual[variable_num] >= q_low) &
                            (df_resultado_visual[variable_num] <= q_high)
                        ]
                    else:
                        # Para las demás columnas usar todos los datos
                        df_plot = df_resultado_visual

                    # 📊 Scatter
                    fig_dep = px.scatter(
                        df_plot,
                        x=variable_num,
                        y="Probabilidad Impago",
                        color="Riesgo",
                        title=f"Dependencia de {variable_num} con probabilidad de impago",
                        color_discrete_map={"Riesgo": "#D81B60", "No riesgo": "#1E88E5"}
                    )
                    st.plotly_chart(fig_dep, use_container_width=True)

            # Gráficos de frecuencia para variables no numéricas vs Riesgo
            non_numeric_cols = df_resultado_visual.select_dtypes(exclude=["float64", "int64"]).columns.tolist()
            non_numeric_cols = [c for c in non_numeric_cols if c != "Riesgo"]
            if non_numeric_cols:
                variable_cat = st.selectbox("Selecciona variable categórica para ver frecuencia por riesgo", non_numeric_cols)
                if variable_cat:
                    df_freq_riesgo = df_resultado_visual.groupby([variable_cat, "Riesgo"]).size().reset_index(name="Cantidad")
                    fig_cat = px.bar(df_freq_riesgo, x=variable_cat, y="Cantidad", color="Riesgo",
                        barmode="group", text="Cantidad",
                        title=f"Frecuencia de {variable_cat} por riesgo",
                        color_discrete_map={"Riesgo": "#D81B60", "No riesgo": "#1E88E5"} )
                    st.plotly_chart(fig_cat, use_container_width=True)

        # ---------------------------
        # 3️⃣ Detalle individual
        # ---------------------------
        with st.expander("Detalle individual"):
            if 'df_resultado_visual' not in locals():
                st.warning("⚠️ Primero carga un archivo CSV válido en la pestaña anterior.")
                st.stop()

            st.subheader("Explicación de predicción individual")
            cliente_idx = st.number_input("Selecciona índice de cliente", 0, len(df_clientes)-1, 0)

            if st.button("Generar explicación para este cliente"):
                try:
                    with st.spinner("Generando explicación para este cliente..."):
                        # Tomar solo la fila del cliente seleccionado
                        cliente_data = df_clientes.iloc[[cliente_idx]]

                        # Crear explainer y calcular SHAP solo para este cliente
                        explainer = shap.TreeExplainer(rf_model, model_output="raw")
                        shap_values_cliente = explainer(cliente_data)
                        shap_cliente = shap_values_cliente[0]

                        # Ajuste si es clasificación multiclase
                        if shap_cliente.values.ndim == 2:
                            shap_cliente = shap.Explanation(
                                values=shap_cliente.values[:, 1],
                                base_values=shap_cliente.base_values[1],
                                data=shap_cliente.data,
                                feature_names=[f.replace("_", " ").title() for f in shap_cliente.feature_names]
                            )
                        # Predicción del cliente
                        riesgo_cliente = rf_model.predict_proba(cliente_data)[:, 1][0]

                        # Mostrar mensaje según umbral
                        if riesgo_cliente >= 0.758:
                            st.error(f"Cliente NO APROBADO (Riesgo: {riesgo_cliente:.2f})")
                        else:
                            st.success(f"Cliente APROBADO (Riesgo: {riesgo_cliente:.2f})")

                        # Generar force plot
                        fig = shap.plots.force(shap_cliente, matplotlib=False)
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
    st.write("Introduce los datos del potencial cliente para ejecutar la predicción.")

    # --- Variables numéricas ---
    valor_credito = st.number_input("Valor del crédito", min_value=0, key="valor_credito")
    suma_ingresos = st.number_input("Suma de ingresos", min_value=0, key="suma_ingresos")
    antiguedad_ocupacion = st.number_input("Antigüedad en ocupación (años)", min_value=0, key="antiguedad_ocupacion")
    plazo_solicitado = st.number_input("Plazo solicitado (días)", min_value=0, key="plazo_solicitado")
    estrato = st.number_input("Estrato", min_value=0, max_value=6, key="estrato")

    # --- Mapeos ---
    map_periocidad_pago = {
        'semanal': 0,
        'quincenal': 1,
        'mensual': 2
    }
    map_tamano_empresa = {
        'de 1 a 10': 0,
        'de 11 a 25': 1,
        'de 26 a 50': 2,
        'de 51 a 100': 3,
        'de 100 a 500': 4,
        'mas de 500': 5
    }

    # --- Selectbox con "Selecciona..." ---
    periocidad_pago_str = st.selectbox("Periodicidad de pago", ["Selecciona..."] + list(map_periocidad_pago.keys()))
    tamano_empresa_str = st.selectbox("Tamaño de la empresa", ["Selecciona..." ] + list(map_tamano_empresa.keys()))

    tipo_pago = st.selectbox("Tipo de pago", ["Selecciona...", "0 (sin dato)", "sustitución", "pago_total"])
    estado_civil = st.selectbox("Estado civil", ["Selecciona...", "soltero", "casado", "union libre"])
    ciudad = st.selectbox("Selecciona tu ciudad", ["Selecciona...", "bogota", "medellin", "otras"])
    operador_cel = st.selectbox("Operador de celular", ["Selecciona...", "claro", "movistar", "tigo", "otro"])
    horario_contacto = st.selectbox("Horario de contacto preferido", ["Selecciona...", "mañana", "otro", "desconocido"])
    sector_ocupacion = st.selectbox("Sector ocupación", ["Selecciona...", "servicios", "industria", "salud_educacion", "otros"])
    fuente_origen = st.selectbox("Fuente de origen", ["Selecciona...", "visita_google", "otro"])
    tipo_plan_celular = st.selectbox("Tipo de plan celular", ["Selecciona...", "prepago", "postpago"])
    ocupacion = st.selectbox("Ocupación", ["Selecciona...", "empleado_estable", "otro"])

    # --- Validación ---
    campos_validos = all([
        valor_credito > 0,
        suma_ingresos > 0,
        antiguedad_ocupacion > 0,
        plazo_solicitado > 0,
        estrato >= 0,
        periocidad_pago_str != "Selecciona...",
        tamano_empresa_str != "Selecciona...",
        tipo_pago != "Selecciona...",
        estado_civil != "Selecciona...",
        ciudad != "Selecciona...",
        operador_cel != "Selecciona...",
        horario_contacto != "Selecciona...",
        sector_ocupacion != "Selecciona...",
        fuente_origen != "Selecciona...",
        tipo_plan_celular != "Selecciona...",
        ocupacion != "Selecciona..."
    ])

    # --- Aplicar mapeo ---
    periocidad_pago = map_periocidad_pago.get(periocidad_pago_str, -1)
    tamano_empresa = map_tamano_empresa.get(tamano_empresa_str, -1)

    # --- Encoding categóricos ---
    tipo_pago_encoded = {
        "tipo_pago_0": int(tipo_pago == "0 (sin dato)"),
        "tipo_pago_sustitucion": int(tipo_pago == "sustitución")
    }

    estado_civil_encoded = {
        "estado_civil_soltero": int(estado_civil == "soltero"),
        "estado_civil_casado": int(estado_civil == "casado"),
        "estado_civil_union libre": int(estado_civil == "union libre")
    }

    ciudad_encoded = {
        "ciudad_bogota": int(ciudad == "bogota"),
        "ciudad_medellin": int(ciudad == "medellin"),
        "ciudad_otras": int(ciudad == "otras")
    }

    operador_cel_encoded = {
        "operador_cel_claro": int(operador_cel == "claro"),
        "operador_cel_movistar": int(operador_cel == "movistar"),
        "operador_cel_tigo": int(operador_cel == "tigo"),
        "operador_cel_otro": int(operador_cel == "otro")
    }

    horario_contacto_encoded = {
        "horario_contacto_manana": int(horario_contacto == "mañana"),
        "horario_contacto_otro": int(horario_contacto == "otro"),
        "horario_contacto_desconocido": int(horario_contacto == "desconocido")
    }

    sector_ocupacion_encoded = {
        "sector_ocupacion_servicios": int(sector_ocupacion == "servicios"),
        "sector_ocupacion_industria": int(sector_ocupacion == "industria"),
        "sector_ocupacion_salud_educacion": int(sector_ocupacion == "salud_educacion"),
        "sector_ocupacion_otros": int(sector_ocupacion == "otros")
    }

    fuente_origen_encoded = {
        "fuente_origen_visita_google": int(fuente_origen == "visita_google")
    }

    tipo_plan_encoded = {
        "tipo_plan_celular_prepago": int(tipo_plan_celular == "prepago")
    }

    ocupacion_encoded = {
        "ocupacion_empleado_estable": int(ocupacion == "empleado_estable")    
    }

    # --- Construcción DataFrame ---
    datos = {
        "relacion_credito_ingresos": [valor_credito / suma_ingresos if suma_ingresos > 0 else 0],
        "valor_credito": [valor_credito],
        **tipo_pago_encoded,
        "suma_ingresos": [suma_ingresos],
        "antiguedad_ocupacion": [antiguedad_ocupacion],
        "plazo_solicitado": [plazo_solicitado],
        "tamano_empresa": [tamano_empresa],
        "estrato": [estrato],
        "periocidad_pago": [periocidad_pago],
        **horario_contacto_encoded,
        **operador_cel_encoded,
        **sector_ocupacion_encoded,
        **ciudad_encoded,
        **estado_civil_encoded,
        **fuente_origen_encoded,
        **tipo_plan_encoded,
        **ocupacion_encoded
    }

    df_clientes = pd.DataFrame(datos)

    # Asegurarse de que las columnas coincidan con las del modelo
    columnas_modelo = rf_model.feature_names_in_
    df_clientes = df_clientes.reindex(columns=columnas_modelo, fill_value=0)

    # Ahora sí hacer la predicción
    prob = rf_model.predict_proba(df_clientes)[:, 1]

    # --- Mostrar tabla preview ---
    st.dataframe(df_clientes)

    # --- Botón predicción ---
    if st.button("Ejecutar predicción"):
        if not campos_validos:
            st.warning("⚠️ Por favor, completa todos los campos antes de continuar.")
            st.stop()
        try:
            prob = rf_model.predict_proba(df_clientes)[:, 1]
            df_resultado = df_clientes.copy()
            df_resultado["probabilidad impago"] = (prob * 100).round(2)
            df_resultado["aprobado"] = ["No" if p >= 0.758 else "Sí" for p in prob]
            resultado = df_resultado.iloc[0]

            # --- Mostrar resultado ---
            if resultado["aprobado"] == "Sí":
                st.markdown(f"""<div style="background:#e6ffed; padding:15px; border-radius:10px;">✅ Cliente aprobado con probabilidad de impago: {resultado['probabilidad impago']}%</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div style="background:#ffe6e6; padding:15px; border-radius:10px;">❌ Cliente NO aprobado. Probabilidad de impago: {resultado['probabilidad impago']}%</div>""", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"⚠️ Error al ejecutar el modelo: {e}")
            st.stop()

        # Pie de página
        st.markdown("""
        <hr style="border:1px solid #ccc">
        <p style='text-align: center; color: gray; font-size: 12px;'>
        Desarrollado por Alejandra, Marta y Jose Luis • Evaluación de Microcréditos © 2025
        </p>
        """, unsafe_allow_html=True)
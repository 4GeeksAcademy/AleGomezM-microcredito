import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Configuración de la página
st.set_page_config(page_title="Evaluación de Microcréditos", layout="wide")

st.markdown("""
    <style>
    /* Fondo general */
    .main {
        background-color: #f5f7fa;
    }

    /* Títulos */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Botones */
    .stButton > button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: bold;
    }

    /* Métricas */
    .stMetric {
        background-color: #ecf0f1;
        border-radius: 10px;
        padding: 10px;
    }

    /* Tab headers */
    div[data-testid="stTabs"] button {
        background-color: #dfe6e9;
        color: #2c3e50;
        font-weight: bold;
        border-radius: 5px;
    }

    /* DataFrame scroll */
    .stDataFrame {
        border: 1px solid #bdc3c7;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Cargar modelo
rf_model = joblib.load("modelo_bfr.pkl")
columnas_modelo = rf_model.feature_names_in_

# Título principal
st.title("Evaluación de clientes para microcrédito")

# Pestañas
tab1, tab2, tab3 = st.tabs(["Cargar CSV", "Resultados", "Análisis"])

# TAB 1: Cargar CSV
with tab1:
    st.header("Carga de datos")
    archivo_csv = st.file_uploader("Sube tu archivo CSV con datos de clientes", type=["csv"])

    if archivo_csv is not None:
        df_clientes = pd.read_csv(archivo_csv)
        st.success("Archivo cargado correctamente")
        st.dataframe(df_clientes.head())

        # Codificación
        df_encoded = pd.get_dummies(df_clientes)
        for col in columnas_modelo:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[columnas_modelo]

        # Validación de columnas esperadas
        columnas_faltantes = [col for col in columnas_modelo if col not in df_encoded.columns]
        if columnas_faltantes:
            st.warning(f"Faltan columnas esperadas en el CSV: {', '.join(columnas_faltantes)}")

        # Predicción
        pred = rf_model.predict(df_encoded)
        prob = rf_model.predict_proba(df_encoded)[:, 1]

        # Resultados
        df_resultado = df_clientes.copy()
        df_resultado["¿Apto para crédito?"] = ["Sí" if p == 1 else "No" for p in pred]
        df_resultado["Probabilidad (%)"] = (prob * 100).round(2)

        # Guardar en sesión
        st.session_state.df_resultado = df_resultado

# TAB 2: Resultados
with tab2:
    if "df_resultado" in st.session_state:
        df_resultado = st.session_state.df_resultado

        st.header("Resultados de evaluación")
        st.dataframe(df_resultado)

        # Métricas
        total = len(df_resultado)
        aptos = sum(df_resultado["¿Apto para crédito?"] == "Sí")
        riesgo_medio = df_resultado["Probabilidad (%)"].mean().round(2)

        st.markdown("Resumen ejecutivo")
        col1, col2, col3 = st.columns(3)
        col1.metric("Clientes evaluados", total)
        col2.metric("Aprobados", aptos)
        col3.metric("Riesgo promedio", f"{riesgo_medio}%")

        # Botones de descarga
        st.subheader("Descargar resultados")
        aptos_df = df_resultado[df_resultado["¿Apto para crédito?"] == "Sí"]
        no_aptos_df = df_resultado[df_resultado["¿Apto para crédito?"] == "No"]

        st.download_button("Clientes aptos", aptos_df.to_csv(index=False).encode("utf-8"),
                           file_name="clientes_aprobados.csv", mime="text/csv")
        st.download_button("Clientes no aptos", no_aptos_df.to_csv(index=False).encode("utf-8"),
                           file_name="clientes_rechazados.csv", mime="text/csv")
        st.download_button("Todos los resultados", df_resultado.to_csv(index=False).encode("utf-8"),
                           file_name="evaluacion_completa.csv", mime="text/csv")
    else:
        st.info("Primero sube un archivo en la pestaña 'Cargar CSV'.")

# TAB 3: Análisis
with tab3:
    if "df_resultado" in st.session_state:
        df_resultado = st.session_state.df_resultado

        st.header("Análisis de riesgo")

        # Filtro por probabilidad
        umbral = st.slider("Filtrar por probabilidad mínima (%)", 0, 100, 50)
        filtrados = df_resultado[df_resultado["Probabilidad (%)"] >= umbral]
        st.dataframe(filtrados)

        # Histograma
        fig = px.histogram(df_resultado, x="Probabilidad (%)", color="¿Apto para crédito?",
                           nbins=20, title="Distribución de probabilidad de aprobación")
        st.plotly_chart(fig, use_container_width=True)

        # Reconstruir estado civil
        df_resultado["estado_civil"] = df_resultado.apply(lambda row: (
            "Soltero" if row["estado_civil_soltero"] else
            "Unión libre" if row["estado_civil_union libre"] else
            "Casado" if row["estado_civil_casado"] else
            "Otro"), axis=1)

        # Reconstruir sector ocupación
        df_resultado["sector_ocupacion"] = df_resultado.apply(lambda row: (
            "Servicios" if row["sector_ocupacion_servicios"] else
            "Industria" if row["sector_ocupacion_industria"] else
            "Salud/Educación" if row["sector_ocupacion_salud_educacion"] else
            "Otros"), axis=1)

        st.subheader("Análisis por grupo")

        # Selección de variable de agrupación
        columnas_categoricas = df_resultado.select_dtypes(include="object").columns.tolist()
        grupo = st.selectbox("Selecciona variable para agrupar", ["estado_civil", "sector_ocupacion", "tamano_empresa", "estrato"])

        if "¿Apto para crédito?" in df_resultado.columns:
            df_resultado["Aprobado"] = df_resultado["¿Apto para crédito?"].apply(lambda x: 1 if x == "Sí" else 0)
        else:
            st.error("La columna '¿Apto para crédito?' no existe en el DataFrame.")


        # Agrupación y métricas
        if grupo in df_resultado.columns:
            df_resultado["Cliente"] = 1  # Cada fila representa un cliente

            df_grupo = df_resultado.groupby(grupo).agg({
                "Probabilidad (%)": "mean",
                "Aprobado": "mean",
                "Cliente": "sum"
            }).rename(columns={
                "Probabilidad (%)": "Riesgo promedio",
                "Aprobado": "% Aprobados",
                "Cliente": "Clientes"
            }).reset_index()

            df_grupo["% Aprobados"] = (df_grupo["% Aprobados"] * 100).round(2)
            df_grupo["Riesgo promedio"] = df_grupo["Riesgo promedio"].round(2)

            # Gráfico
            fig = px.bar(
                df_grupo,
                x=grupo,
                y="% Aprobados",
                color="Riesgo promedio",
                text="Clientes",
                title=f"Aprobación y riesgo por {grupo}",
                color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)

            # Tabla
            st.dataframe(df_grupo.style.format({
                "Riesgo promedio": "{:.2f}%",
                "% Aprobados": "{:.2f}%",
                "Clientes": "{:,}"}))
        else:
            st.warning(f"La columna '{grupo}' no está disponible en los datos.")

        # Gráfico de dispersión
        st.subheader("Relación entre ingreso y probabilidad de aprobación")
        if "suma_ingresos" in df_resultado.columns:
            fig3 = px.scatter(df_resultado, x="suma_ingresos", y="Probabilidad (%)",
                            color="¿Apto para crédito?", title="Ingreso vs Probabilidad")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("La columna 'income' no está disponible en los datos.")

        # Resumen estadístico por grupo
        st.subheader("Resumen estadístico por grupo")
        columnas_numericas = df_resultado.select_dtypes(include="number").columns.tolist()
        resumen = df_resultado.groupby("¿Apto para crédito?")[columnas_numericas].agg(["mean", "std", "min", "max"])
        st.dataframe(resumen)

        # Importancia de variables
        st.subheader("Importancia de variables en el modelo")
        importancias = rf_model.feature_importances_
        features = rf_model.feature_names_in_

        # Crear DataFrame ordenado
        df_importancia = pd.DataFrame({
            "Variable": features,
            "Importancia": importancias
        }).sort_values(by="Importancia", ascending=True)

        # Selector de número de variables
        top_n = st.slider("Selecciona cuántas variables mostrar", 5, len(df_importancia), 10)
        df_top = df_importancia.tail(top_n)

        # Estilo visual con seaborn
        fig2, ax = plt.subplots(figsize=(8, len(df_importancia) * 0.4))
        sns.barplot(
            x="Importancia",
            y="Variable",
            data=df_top,
            hue="Variable", # Esto asigna color por variable.
            dodge=False,    # Evita que se separen las barras.
            palette="viridis",
            ax=ax,
            legend=False)  

        ax.set_title("Importancia de variables", fontsize=14)
        ax.set_xlabel("Importancia", fontsize=12)
        ax.set_ylabel("Variable", fontsize=12)
        st.pyplot(fig2)
        import io

        buf = io.BytesIO()
        fig2.savefig(buf, format="png")
        st.download_button("Descargar gráfico", buf, file_name="importancia_variables.png",
                            mime="image/png")

    else:
        st.info("Primero sube un archivo en la pestaña 'Cargar CSV'.")

st.markdown("""
<hr style="border:1px solid #ccc">
<p style='text-align: center; color: gray; font-size: 12px;'>
Desarrollado por Alejandra, Marta y Jose Luis • Evaluación de Microcréditos © 2025
</p>
""", unsafe_allow_html=True)
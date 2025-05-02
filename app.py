import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch
import os
import sys

# Configuración de página
st.set_page_config(
    page_title="Detección de Objetos con YOLOv5 🔍",
    page_icon="🧠",
    layout="wide"
)

# Estilos personalizados: fondo oscuro y texto claro
st.markdown("""
    <style>
        html, body, .stApp {
            background-color: #B3BAC4;
            color: #E0E0E0;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3, h4, h5, h6, .stTitle, .stHeader {
            text-align: center;
            color: #FFEB3B;
        }
        .stSidebar > div:first-child {
            background-color: #1E1E1E;
        }
        .stButton>button {
            background-color: #FF5722;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
        .stMarkdown, .stDataFrame, .stCaption {
            text-align: center;
        }
        .css-1cpxqw2 {  /* para inputs */
            color: white !important;
        }
        .block-container {
            padding-left: 5%;
            padding-right: 5%;
        }
    </style>
""", unsafe_allow_html=True)

# Función para cargar modelo
@st.cache_resource
def load_yolov5_model(model_path='yolov5s.pt'):
    try:
        import yolov5
        try:
            model = yolov5.load(model_path, weights_only=False)
            return model
        except TypeError:
            model = yolov5.load(model_path)
            return model
        except:
            st.warning("Intentando método alternativo de carga...")
            current_dir = os.path.dirname(os.path.abspath(__file__))
            if current_dir not in sys.path:
                sys.path.append(current_dir)
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
            Recomendaciones:
            1. `pip install torch==1.12.0 torchvision==0.13.0`
            2. `pip install yolov5`
            3. Verifica si el archivo 'yolov5s.pt' está en el directorio correcto
        """)
        return None

# Encabezado
st.title("🔍 Detección de Objetos con YOLOv5")
st.markdown("Analiza imágenes capturadas desde tu cámara usando inteligencia artificial. 🤖")

# Cargar modelo
with st.spinner("⏳ Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

if model:
    # Parámetros en la barra lateral
    st.sidebar.title("⚙️ Parámetros de Detección")
    model.conf = st.sidebar.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
    model.iou = st.sidebar.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
    st.sidebar.caption(f"Confianza: {model.conf:.2f} | IoU: {model.iou:.2f}")

    st.sidebar.subheader("🔧 Opciones avanzadas")
    try:
        model.agnostic = st.sidebar.checkbox('NMS class-agnostic', False)
        model.multi_label = st.sidebar.checkbox('Múltiples etiquetas por caja', False)
        model.max_det = st.sidebar.number_input('Detecciones máximas', 10, 2000, 1000, 10)
    except:
        st.sidebar.warning("⚠️ Algunas opciones avanzadas no están disponibles")

    # Entrada desde la cámara
    picture = st.camera_input("📸 Captura una imagen")

    if picture:
        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        with st.spinner("🎯 Detectando objetos..."):
            try:
                results = model(cv2_img)
            except Exception as e:
                st.error(f"❌ Error durante la detección: {str(e)}")
                st.stop()

        # Mostrar resultados
        predictions = results.pred[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🖼 Imagen con detecciones")
            results.render()
            st.image(cv2_img, channels='BGR', use_column_width=True)

        with col2:
            st.subheader("📊 Objetos detectados")
            label_names = model.names
            category_count = {}

            for category in categories:
                idx = int(category.item()) if hasattr(category, 'item') else int(category)
                category_count[idx] = category_count.get(idx, 0) + 1

            data = []
            for idx, count in category_count.items():
                label = label_names[idx]
                confidence = scores[categories == idx].mean().item()
                data.append({
                    "Categoría": label,
                    "Cantidad": count,
                    "Confianza promedio": f"{confidence:.2f}"
                })

            if data:
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index('Categoría')['Cantidad'])
            else:
                st.info("🤷 No se detectaron objetos con la configuración actual.")
                st.caption("⬅️ Ajusta la confianza en la barra lateral.")

else:
    st.error("🚫 El modelo no se cargó correctamente.")

# Pie de página
st.markdown("---")
st.caption("📌 App desarrollada con Streamlit, PyTorch y YOLOv5 | © 2025")

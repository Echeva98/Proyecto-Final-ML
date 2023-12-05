import streamlit as st
import model_evaluation

def main():
    st.title("Clasificación de Plantas - Resultados del Modelo")

    model = cargar_modelo()

    test_data, test_labels, class_names = cargar_datos_prueba()

    report, confusion_mat = model_evaluation.evaluate_model(model, test_data, test_labels, class_names)

    # Mostrar métricas
    st.subheader("Métricas de Rendimiento:")
    st.text(report)

    # Mostrar matriz de confusión
    st.subheader("Matriz de Confusión:")
    st.text(confusion_mat)


if __name__ == "__main__":
    main()

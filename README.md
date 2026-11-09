# PredictLab — Proyecto final de predicción de dengue pediátrico

## Archivo principal de entrenamiento
**`Modelo_Predictivo_Dengue.ipynb`** es el notebook principal. Ábrelo en Jupyter Notebook o Visual Studio Code y ejecuta las celdas en orden. El notebook:
1. Carga `MASTER_CHART_F1000Research.xlsx`.
2. Revisa los datos faltantes y distribuciones.
3. Prepara los cinco indicadores seleccionados.
4. Divide los datos en entrenamiento y prueba.
5. Entrena regresión logística y red neuronal artificial.
6. Calcula métricas, matrices de confusión y curvas ROC.
7. Guarda los modelos entrenados en `models/modelos_entrenados.joblib`.
8. Crea/actualiza `dengue.db` y guarda el informe JSON.

## Ejecutar en Windows
1. Descomprime el ZIP completo.
2. Abre la carpeta `Proyecto_Dengue_PredictLab` en Visual Studio Code.
3. Abre una terminal y ejecuta:
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements-notebook.txt
   ```
4. Abre `Modelo_Predictivo_Dengue.ipynb`, selecciona el kernel de Python del entorno `venv` y ejecuta todas las celdas.
5. Cuando termine el notebook, ejecuta:
   ```powershell
   python app.py
   ```
6. Abre http://127.0.0.1:5000

> El ZIP incluye los modelos ya entrenados y la base SQLite generada. Puedes iniciar la aplicación web directamente después de instalar dependencias; ejecutar el notebook vuelve a entrenar los modelos y actualiza los artefactos.

## Archivos principales
- `Modelo_Predictivo_Dengue.ipynb`: notebook de entrenamiento y evaluación.
- `MASTER_CHART_F1000Research.xlsx`: base de datos original.
- `models/modelos_entrenados.joblib`: modelos ya entrenados que consume Flask.
- `dengue.db`: base SQLite con 302 registros válidos, métricas y tabla de historial.
- `app.py`: aplicación web.
- `templates/` y `static/`: interfaz web.
- `informe_entrenamiento.json`: métricas y matrices.
- `plantilla_prediccion_lote.csv`: ejemplo para probar predicción por lote.
- `articulo_referencia.pdf` y `diccionario_datos_original.docx`: fuentes suministradas.

## Variables de entrada del modelo
- `BP_AT_ADMISSION`: hipotensión.
- `PCV`: hematocrito / packed cell volume.
- `PLATELET_COUNT`: recuento de plaquetas.
- `HIGHEST_WBC`: recuento máximo de leucocitos.
- `ALT`: alanina aminotransferasa.

La variable objetivo es la primera columna del Excel: código 1 = dengue no complicado y código 2 = dengue complicado.

## Nota metodológica sobre presión arterial
El README/diccionario adjunto describe las categorías 1 y 2, pero no especifica de forma inequívoca cuál representa hipotensión. En el Excel, el código 2 aparece en 38 registros, cantidad que coincide con los 38 casos con hipotensión descritos en la Tabla 1 del artículo (9 no complicados + 29 complicados). Por ello el notebook interpreta código 2 como hipotensión y código 1 como ausencia de hipotensión. Es una inferencia por correspondencia de frecuencias; debe explicitarse en la sustentación.

## Resultados de la evaluación
El notebook muestra los resultados al ejecutarse y actualiza `informe_entrenamiento.json`. La evaluación utiliza una partición estratificada única (75 % entrenamiento / 25 % prueba, `random_state=42`); no es validación externa.

## Advertencia
Aplicación académica. No usar para diagnóstico, tratamiento, triage ni decisiones clínicas. El artículo fuente indica que el sistema de puntuación requiere validación antes de su aplicación clínica.

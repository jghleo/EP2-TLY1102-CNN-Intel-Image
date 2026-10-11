# ¿Cómo ejecutar este proyecto? — Guía rápida

**Proyecto EP2 TLY1102 — CNN para clasificación de escenarios naturales (Intel Image Classification)**

Este paquete trae todo lo que pide la pauta: el **cuaderno** (`.ipynb`), el **informe técnico**
(`informe.md`), el **README**, los **requisitos** y la **estructura de carpetas** profesional.

```
EP2_TLY1102_CNN_Intel_Image_ENTREGA/
├── README.md
├── informe.md
├── requirements.txt
├── .gitignore
├── data/          <- aquí va el dataset (ver más abajo)
├── notebooks/     <- el cuaderno
├── models/        <- aquí se guardan los modelos entrenados
├── images/        <- figuras (EDA ya incluidas)
└── src/
```

---

## Opción A — Google Colab (recomendado, usa GPU gratis)

1. Sube la carpeta del proyecto a tu Google Drive (o súbela a tu GitHub y usa el botón
   **Open in Colab**).
2. Abre `notebooks/EP2_TLY1102_CNN_Intel_Image.ipynb` en Colab.
3. Activa la GPU: **Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)**.
4. Menú **Entorno de ejecución → Ejecutar todo**.
5. El cuaderno **descarga el dataset solo** desde Google Drive y guarda las figuras en `images/`
   y los modelos en `models/`.
6. Al terminar, guarda el cuaderno **con las salidas**: **Archivo → Descargar → .ipynb**.
7. Copia la **tabla que imprime la última celda** en la sección *Resultados* de `informe.md`.

> ⏱️ Tiempo estimado en Colab con GPU T4: **~15–30 min**.
> Sin GPU (CPU) puede tardar **varias horas**.

---

## Opción B — En tu computador (local)

### 1) Instala las dependencias

```bash
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Consigue el dataset

Descarga **Intel Image Classification** desde Kaggle:
<https://www.kaggle.com/datasets/puneet6060/intel-image-classification>

Descomprime de modo que quede así (ojo: el dataset viene con un nivel anidado):

```
data/
├── seg_train/seg_train/{buildings, forest, glacier, mountain, sea, street}/*.jpg
├── seg_test/seg_test/{buildings, forest, glacier, mountain, sea, street}/*.jpg
└── seg_pred/seg_pred/*.jpg
```

> Alternativa: si NO pones el dataset en `data/`, el cuaderno intenta bajarlo solo desde Google
> Drive (necesita el paquete `gdown` e internet).

### 3) Ejecuta el cuaderno

```bash
jupyter notebook notebooks/EP2_TLY1102_CNN_Intel_Image.ipynb
```

Luego: **Kernel → Restart & Run All**.

---

## Qué calcula y guarda el cuaderno

- **EDA**: distribución de clases, ejemplos por clase y tamaños (`images/`).
- **CNN**: entrenamiento con curvas de aprendizaje, métricas (accuracy, precision, recall, F1),
  matriz de confusión, ROC/AUC, mapas de características y activación por clase.
- **MLP**: el mismo experimento con el modelo de la EP1 (mismos datos) para comparar.
- **Modelos**: `models/cnn_intel_image.keras` y `models/mlp_intel_image.keras`.
- **Resumen final**: una tabla Markdown lista para pegar en `informe.md`.

---

## Preguntas frecuentes

- **¿Se demora mucho?** Con GPU ~15–30 min. En CPU, varias horas (sube `EPOCHS` o baja
  `N_POR_CLASE` / `IMG_SIZE` en la celda de configuración si necesitas que sea más rápido).
- **¿Falta el dataset?** El cuaderno lo descarga solo desde Drive; si falla, usa Kaggle.
- **¿Dónde están las figuras?** En `images/` (se regeneran cada vez que ejecutas el cuaderno).
- **¿No tengo GPU?** Funciona igual en CPU, solo más lento. Reduce `IMG_SIZE` a 64 y
  `N_POR_CLASE` a 500 para una prueba rápida.

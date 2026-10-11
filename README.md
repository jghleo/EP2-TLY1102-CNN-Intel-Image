# Clasificación de escenarios naturales con una CNN — Intel Image Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jghleo/EP2-TLY1102-CNN-Intel-Image/blob/main/notebooks/EP2_TLY1102_CNN_Intel_Image.ipynb)

**Evaluación Parcial N°2 — TLY1102 Técnicas Avanzadas de Machine Learning I**
**Sección:** 0001D
**Integrantes:** Javier Garcia, Leandro Marin, Andres Zuñiga

Red Neuronal Convolucional (CNN) en Keras/TensorFlow que clasifica fotografías de paisajes y
escenas urbanas en **6 categorías** (`buildings`, `forest`, `glacier`, `mountain`, `sea`,
`street`), comparada contra el **Perceptrón Multicapa (MLP)** de la Evaluación Parcial N°1.

---

## Descripción

Este proyecto es la **continuación directa** de la EP1 (mismo dataset, mismo problema). El objetivo
es demostrar experimentalmente que una **CNN aprovecha mejor la estructura espacial** de la imagen
que un MLP, el cual —al aplanar la imagen— pierde la relación entre píxeles vecinos y por eso
alcanzó solo un 47,7 % de accuracy en la EP1.

El cuaderno entrena y compara ambos modelos bajo **condiciones idénticas** (misma partición,
resolución y aumento de datos), y visualiza **mapas de características** para explicar por qué la
CNN funciona mejor.

### Variante propia del grupo

- Muestra de **1.300 imágenes por clase** para entrenamiento (semilla fija `SEED=42`).
- Imágenes redimensionadas a **128×128 px**.
- Partición **80/20 estratificada** (entrenamiento/validación) + set `seg_test` intacto (3.000 imgs).
- Data augmentation (flip, rotación, zoom, contraste) y regularización (BatchNorm, Dropout).

## Dataset

**Intel Image Classification** (Intel / Kaggle). Contiene imágenes de paisajes y escenas urbanas en
6 clases.

| Split | Uso | Imágenes |
|---|---|---|
| `seg_train` | entrenamiento + validación | ~14.034 |
| `seg_test` | evaluación final independiente | 3.000 |
| `seg_pred` | imágenes sin etiqueta (demo) | ~7.300 |

La carpeta `data/` **no se sube al repositorio** (por tamaño). El cuaderno la obtiene
automáticamente desde Google Drive al ejecutarse (o descárgala desde
[Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)).

## Estructura del proyecto

```
EP2-TLY1102-CNN-Intel-Image/
├── README.md
├── informe.md                     # Informe técnico (Markdown)
├── requirements.txt
├── .gitignore
├── data/                          # Dataset (no versionado)
│   ├── seg_train/
│   ├── seg_test/
│   └── seg_pred/
├── notebooks/
│   └── EP2_TLY1102_CNN_Intel_Image.ipynb
├── models/                        # Modelos entrenados (.keras)
├── images/                        # Figuras del informe (EDA, curvas, feature maps, ...)
└── src/                           # (espacio para utilidades/scripts)
```

## Cómo ejecutar

### Google Colab (recomendado — usa GPU)

1. Abre el cuaderno con el botón **Open in Colab** de arriba.
2. Activa la GPU: *Entorno de ejecución → Cambiar tipo de entorno → GPU (T4)*.
3. Menú **Entorno de ejecución → Ejecutar todo**.
4. El cuaderno descarga el dataset desde Google Drive y guarda todas las figuras en `images/` y
   los modelos en `models/`.

### Local

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Opción A: colocar el dataset descomprimido en data/ y ejecutar el cuaderno.
# Opción B: dejar que el cuaderno lo descargue desde Google Drive (requiere gdown e internet).

jupyter notebook notebooks/EP2_TLY1102_CNN_Intel_Image.ipynb
```

> **Nota de entrega:** tras ejecutar el cuaderno en Colab, guarda el `.ipynb` **con las salidas**
> (*Archivo → Descargar → .ipynb*) y copia la tabla impresa por la **celda final** en la sección de
> resultados de `informe.md`.

## Contenido del cuaderno

El cuaderno sigue la estructura de la pauta:

| Sección | Contenido |
|---|---|
| 1. Introducción | Problema, dataset, variante, objetivo y por qué comparar con el MLP |
| 2. Carga y EDA | Descarga, balance de clases, ejemplos, tamaños |
| 3. Preparación de datos | Redimensionado, normalización, formato matricial, split, augmentation |
| 4. Diseño de la CNN | Arquitectura, capas, activaciones, pooling, pérdida, optimizador |
| 5. Entrenamiento | Callbacks y curvas de aprendizaje |
| 6. Evaluación | Accuracy, Precision, Recall, F1, matriz de confusión, ROC/AUC |
| 7. Comparación MLP vs CNN | Desempeño, parámetros, tiempo y generalización |
| 8. Feature maps | Visualización de mapas de características y activación por clase |
| 9. Análisis de errores | Aciertos, errores y clases difíciles |
| 10. Conclusiones | Ventajas de la CNN, limitaciones y mejoras |

## Resultados

Se completan al ejecutar el cuaderno (columna *CNN*). La EP1 reportó **47,7 %** de accuracy con el
MLP sobre 800 imágenes/clase a 64 px.

| Modelo | Accuracy | F1 (weighted) | Parámetros | Tiempo (min) |
|---|---|---|---|---|
| MLP | _por completar_ | _por completar_ | _por completar_ | _por completar_ |
| CNN | _por completar_ | _por completar_ | _por completar_ | _por completar_ |

## Declaración de uso de IA

Se utilizó Inteligencia Artificial como **apoyo** para la redacción y organización del cuaderno y
del informe. Todas las decisiones técnicas y la validación de resultados fueron realizadas por el
equipo. Referencia: <https://bibliotecas.duoc.cl/ia>

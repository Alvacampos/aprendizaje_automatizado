# Default of Credit Card Clients – EDA y Modelado

**Evaluación Integradora – Aprendizaje Automatizado**
**Universidad:** Universidad Blas Pascal
**Alumno:** Gonzalo Alvarez Campos

---

## Pregunta de negocio

> ¿Es posible anticipar qué clientes de tarjetas de crédito entrarán en *default* (falta de pago) el mes siguiente?

## Dataset

| | |
|---|---|
| **Nombre** | Default of Credit Card Clients |
| **Fuente** | [UCI Machine Learning Repository (id 350)](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) – Yeh, I. C. & Lien, C. H. (2009) |
| **Registros** | 30.000 clientes de tarjetas de crédito en Taiwán (abril–septiembre 2005) |
| **Variables** | 23 explicativas + 1 objetivo (`DEFAULT`) |

Grupos de variables:

- **Datos del cliente:** `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`.
- **Estado de pago mensual:** `PAY_1` … `PAY_6` (‑1 = al día, 1–9 = meses de atraso).
- **Montos facturados:** `BILL_AMT1` … `BILL_AMT6` (NT$).
- **Montos pagados:** `PAY_AMT1` … `PAY_AMT6` (NT$).
- **Objetivo:** `DEFAULT` (1 = no paga el mes siguiente).

### Justificación de la elección

El dataset admite los tres enfoques pedidos por la consigna:

- **Clasificación:** predecir `DEFAULT` (binaria).
- **Regresión:** estimar el límite de crédito `LIMIT_BAL` (sin usar `DEFAULT`, para evitar fuga de información).
- **Clustering:** segmentar clientes según su comportamiento de pago y facturación.

---

## Estructura del repositorio

| Archivo | Descripción |
|---|---|
| [EDA_default_tarjetas.ipynb](EDA_default_tarjetas.ipynb) | Notebook principal (Google Colab) con código, salidas y explicaciones. |
| [eda_default_tarjetas.py](eda_default_tarjetas.py) | Exportación del notebook a script Python. |

Al ejecutarse, el notebook genera además `dataset_final.csv` (dataset limpio y codificado).

---

## Parte 1 – Análisis exploratorio de datos (Módulos 1 y 2)

1. **Tipos de variables:** todas vienen como `int64`, pero `SEX`, `EDUCATION`, `MARRIAGE` y `PAY_X` son códigos categóricos, no magnitudes.
2. **Calidad de datos:**
   - Sin valores nulos; 35 filas duplicadas (0,12 %).
   - Códigos no documentados: `EDUCATION` 0, 5, 6 (≈1 %) y `MARRIAGE` 0 → se agrupan en "otros".
   - `PAY_X` con valores ‑2 y 0 (58 % en `PAY_1`): se toman como categorías válidas.
   - Facturas negativas (saldo a favor, ≈2 % mensual) y 2.115 clientes con deuda mayor al límite: comportamientos reales, no errores.
   - Outliers en montos (8–9 %) que se conservan: son clientes de alto consumo/pago.
3. **Distribuciones:**
   - Clases desbalanceadas: 22 % default vs. 78 % no default → la *accuracy* no es una métrica adecuada.
   - Montos con fuerte asimetría a derecha (media ≫ mediana).
   - Población: 60 % mujeres, 82 % con estudios universitarios o de posgrado, mitad entre 28 y 41 años.
4. **Relaciones:**
   - `PAY_1` es la variable más asociada al default (r = 0,33): ~13–17 % de default al día, 34 % con un mes de atraso, >50 % con dos o más.
   - `LIMIT_BAL` se relaciona negativamente (r = ‑0,15).
   - `BILL_AMT1`…`BILL_AMT6` muy correlacionadas entre sí (0,89–0,95) → redundancia.
   - Edad, sexo, estado civil y educación aportan diferencias pequeñas.
5. **Conclusiones:** el comportamiento de pago reciente es el mejor predictor; se planifica limpiar códigos, reducir redundancia y excluir `SEX` por criterio ético.

---

## Parte 2 – Modelado, evaluación y comparación (Módulos 3 y 4)

Enfoque elegido: **clasificación** de `DEFAULT`.

### Preparación de datos

- Eliminación de las 35 filas duplicadas (evita que una misma fila quede en train y test).
- Agrupación de códigos no documentados en "otros".
- Exclusión de `SEX` (criterio ético) y de `BILL_AMT2`–`BILL_AMT6` (redundancia).
- *One‑hot encoding* de `EDUCATION` y `MARRIAGE`; `PAY_X` se mantienen ordinales.
- División 80/20 estratificada (`random_state=42`).
- `StandardScaler` ajustado solo con train, dentro de un *pipeline* para la Regresión Logística.

Dataset final: **29.965 filas × 21 columnas**.

### Modelos

| Modelo | Configuración |
|---|---|
| Regresión Logística | `class_weight='balanced'`, escalado previo |
| Random Forest | 100 árboles, `class_weight='balanced'`, `max_depth=5` |

- Validación cruzada estratificada de 5 *folds* con F1 como métrica.
- `GridSearchCV` sobre `max_depth`: sin límite el modelo sobreajusta (F1 train 0,998 vs. validación 0,459). Se elige profundidad 5 porque rinde igual que 10 (0,543 vs. 0,544) con mucha menor brecha train–validación.

### Resultados sobre test

| | Regresión Logística | Random Forest |
|---|---|---|
| Accuracy | 0,685 | 0,773 |
| Precision | 0,374 | 0,489 |
| Recall | 0,629 | 0,590 |
| **F1** | 0,469 | **0,535** |
| Buenos clientes rechazados | 1.397 (30 %) | 817 (17,5 %) |
| Interpretabilidad | Alta | Baja |
| Necesita escalado | Sí | No |
| Capta relaciones no lineales | No | Sí |

Ambos modelos coinciden en que `PAY_1` es la variable más importante; en Random Forest las variables de atraso suman ≈75 % de la importancia.

### Conclusiones

- **Modelo elegido:** Random Forest (profundidad 5). Mejor F1 y 580 buenos clientes rechazados menos, con un recall similar y sin overfitting (diferencia de F1 train–test < 0,02).
- **Antes de producción:** ajustar el umbral de decisión según costos del negocio, incorporar explicabilidad individual y reentrenar periódicamente.
- **Limitaciones:** el mejor modelo deja pasar el 41 % de los defaults; el techo parece estar en la información disponible (solo 6 meses de historial, sin ingresos ni deudas externas).
- **Sesgos del dataset:** datos de Taiwán 2005, muestra no representativa, sesgo de selección (solo clientes aprobados) y `LIMIT_BAL` refleja decisiones previas del banco.

---

## Cómo ejecutarlo

1. Abrir [EDA_default_tarjetas.ipynb](EDA_default_tarjetas.ipynb) en Google Colab o Jupyter.
2. Ejecutar todas las celdas. El dataset se descarga automáticamente con `ucimlrepo`.

Dependencias principales: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `ucimlrepo`.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ucimlrepo
```

> La última celda usa `google.colab.files` para descargar `dataset_final.csv`; fuera de Colab puede omitirse.

---

## Uso de IA

Este trabajo fue realizado con la guía de Claude (Anthropic). Claude propuso el código y los borradores de las interpretaciones; el alumno ejecutó el análisis, revisó y adaptó los textos, y tomó las decisiones finales sobre el dataset, los modelos y las conclusiones.

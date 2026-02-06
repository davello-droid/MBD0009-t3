# MBD0009-t3
**Problem Set 03: Forecasting y Pricing (MBD0009)**

Repositorio con scripts por pregunta (1 archivo `.py` por pregunta), más dependencias (`requirements.txt`) y un `main.py` opcional para ejecutar todas.

---
## 1) Estructura del Proyecto
```text
Tarea01/
|-- main.py                     # Script principal para ejecutar todo
|-- Pregunta_01.py              # Descomposición Temporal e Ingeniería de Variables
|-- Pregunta_02.py              # Validación y Métricas de Negocio
|-- Pregunta_03.py              # Selección de Algoritmos y Casos Especiales
|-- Pregunta_04.py              # Implementación de Forecasting con MLForecast
|-- Pregunta_05.py              # Estimación de Elasticidad en E-commerce
|-- Pregunta_06.py              # Pricing Dinámico y Discriminación de Precios
|-- Pregunta_07.py              # Predecir precios
|-- requirements.txt            # Dependencias del proyecto
|-- README.md                   # Este archivo
|-- data/                       # Carpeta de datos (descargas + archivos locales)
```

---
## 2) Instalación (Windows / PowerShell)

```powershell
# 1) Crear entorno
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Instalar dependencias
python -m pip install -U pip
python -m pip install -r requirements.txt
```

> Si PowerShell bloquea la activación del venv:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 3) Datasets (carpeta `data/`)
### 3.1 M5 Forecasting (Kaggle) — usado en Pregunta 4
Descarga con Kaggle CLI (requiere credenciales).

**Usando variables de entorno (PowerShell)**
#es necesario generar API token en https://www.kaggle.com/settings, el siguiente comando crea una variable de entorno temporal:
```powershell
$env:KAGGLE_API_TOKEN="PEGA_TU_TOKEN"
```
Luego descarga:
# crear carpeta data si no existe
```powershell
# crear carpeta data si no existe
New-Item -ItemType Directory -Force -Path .\data | Out-Null

# M5 competition files

kaggle competitions download -c m5-forecasting-accuracy -p .\data
Expand-Archive .\data\m5-forecasting-accuracy.zip -DestinationPath .\data -Force
```
### 3.2 Online Retail II (Kaggle) — usado en Pregunta 5
```powershell
kaggle datasets download -d mashlyn/online-retail-ii-uci -p .\data
Expand-Archive .\data\online-retail-ii-uci.zip -DestinationPath .\data -Force
```

### 3.3 Precios propiedades (Excel) — usado en Pregunta 7
El archivo debe estar en:
- `data/Precios_PS03.xlsx`

Contiene 2 hojas:
- `Estudiantes-train-test`
- `Estudiantes-target`


---

## 4) Ejecución (1 script por pregunta)

> Ejecutar desde la raíz del repo (misma carpeta donde están los `.py`).

```powershell
python Pregunta_01.py
python Pregunta_02.py
python Pregunta_03.py
python Pregunta_04.py
python Pregunta_05.py
python Pregunta_06.py
python Pregunta_07.py
```

---

## 5) Outputs esperados (para corrección)

### Pregunta 4
- Genera métricas comparativas (MAE/RMSE/MAPE) para LGBM vs XGB y un gráfico de predicción.

### Pregunta 7 (entregable obligatorio)
- Genera: `Grupo_2_PS03Q07.xlsx`
- Formato: **solo dos columnas** (`titulo_propiedad`, `precio_uf`)
- Ubicación: raíz del repo (o donde se ejecute el script)

---
## 6) Datasets utilizados (resumen)

| Archivo | Descripción |
|---|---|
| `sales_train_evaluation.csv` | Ventas diarias por producto (M5) |
| `calendar.csv` | Calendario, eventos y SNAP (M5) |
| `sell_prices.csv` | Precios semanales (M5) |
| `online_retail_II.csv` | Transacciones retailer online UK (2009–2011) |
| `Precios_PS03.xlsx` | `Estudiantes-train-test` (train) y `Estudiantes-target` (predicción) |

---
## Autoría
**Grupo 2 – MBD0009**
- Felipe Valdivia
- Daniel Avello
- Roberto Sepúlveda
- Roberto Sepulveda
---

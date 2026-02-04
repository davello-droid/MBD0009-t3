# MBD0009-t3
Problem Set 03: Forecasting y Pricing
## Estructura del Proyecto
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
```

---
### Instalación de Dependencias

```bash
#Librerias y entorno en windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
#es necesario registrar token de Kaggle para descarga de dataset usados en pregunta 4, el siguiente comando crea una variable de entorno temporal:
$env:KAGGLE_API_TOKEN="PEGA_TU_TOKEN"
#ejecuta el siguiente comando para descargar dataset de kaggle en carpeta "data":
kaggle competitions download -c m5-forecasting-accuracy -p .\data
Expand-Archive .\data\m5-forecasting-accuracy.zip -DestinationPath .\data -Force

```
## Datasets Utilizados
| Archivo | Descripción |
|---------|-------------|
| `sales_train_evaluation.csv` |  Ventas diarias por producto (30,490 series × 1,941 días) |
| `calendar.csv` | Información de fechas, eventos y SNAP |
| `sell_prices.csv` | Precios semanales por producto y tienda |

---
## Autor

Grupo 2 - MBD0009
- Felipe Valdivia
- Daniel Avello
- Roberto Sepulveda
---


# Pregunta 1:  Descomposición Temporal e Ingeniería de Variables
def run() -> None:
    print("=" * 150)
    print("PARTE I: FORECASTING\n")
    print("Pregunta 1:  Descomposición Temporal e Ingeniería de Variables")
    print("=" * 150)
    print("""
    1.a) Data Leakage con lag_0
            Pregunta: Un analista junior propone utilizar lag_0 (la venta del mismo día) como feature porque "tiene la correlación más alta con el target". 
            Explica por qué esta práctica constituye Data Leakage y cuál sería el impacto en las métricas de backtesting vs. la operación real.

            Respuesta:
            El Data leakage ocurre cuando información del futuro o del target se filtra inadvertidamente hacia las
            features de entrenamiento, provocando que el modelo aprenda patrones que no estarán disponbles en producción.

            ¿Por que lag_0 constituye Data Leakage?
            1. Definicion del problema: lag_0 representa las ventas del mismo día que queremos predecir. En términos matematicos:
                    * Si nuestro target es Yt (ventas del día t)
                    * Entonces lag_0 = Yt (exactamente el mismo valor)
            2. El problema fundamental: En el momento de hacer la predicción (Ej: la noche anterior o temprano en la mañana) 
                No conocemos las ventas del día t porque aún no han ocurrido, Usar esta información es como "ver el futiuro". 
            3. Correlación perfecta engañosa:  La correlación de lag_0 con el target es 1.0 (o cercana a 1.0 su hay ruidao), pero 
                esta correlación es espuria porque estamois correlacionando una variable consigo misma.
            
            Impacto en las métricas de backtesting vs. la operación real:
            -------------------------------------------------------------------
            | Aspecto       | Backtesting (con leakage) | Operación Real      |
            |---------------|---------------------------|---------------------|
            | RMSE          | Muy bajo (~0 o cercano)   | Extremadamente alto |
            | MAPE          | ~0%                       | Puede ser >50%      |
            | R²            | ~1.0 (perfecto)           | Negativo o muy bajo |
            | Confianza     | Falsa sensación de éxito  | Fracaso total       |
            -------------------------------------------------------------------

            Ejemplo numérico ilustrativo:

            Backtesting con lag_0:
                - RMSE: 0.5 unidades
                - MAPE: 0.1%
                - R²: 0.99

            Producción (sin lag_0 disponible):
                - RMSE: 150 unidades
                - MAPE: 35%
                - R²: 0.45

            Consecuencias operativas:
                - Sobrestock o quiebres de inventario por predicciones incorrectas
                - Pérdida de credibilidad del equipo de data science
                - Decisiones de negocio erróneas basadas en métricas infladas artificialmente
            
            Solución correcta:
                - Usar únicamente lags con horizonte ≥ 1 (como lag_1, lag_7) que representen información realmente 
                disponible al momento de la predicción.              
    """)
    print("=" * 50)
    print("Demostración del Data Leakage")
    print("=" * 50)
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

    # Simulación de datos
    np.random.seed(42)
    n = 365
    ventas = 100 + np.cumsum(np.random.randn(n) * 5) + 20 * np.sin(np.arange(n) * 2 * np.pi / 7)

    df = pd.DataFrame({
        'ventas': ventas,
        'lag_0': ventas,  # LEAKAGE!
        'lag_1': pd.Series(ventas).shift(1),
        'lag_7': pd.Series(ventas).shift(7)
    })
    df = df.dropna()

    # Split temporal
    train = df.iloc[:300]
    test = df.iloc[300:]

    # Modelo CON leakage (lag_0)
    model_leak = LinearRegression()
    model_leak.fit(train[['lag_0']], train['ventas'])
    pred_leak = model_leak.predict(test[['lag_0']])

    # Modelo SIN leakage (lag_1, lag_7)
    model_ok = LinearRegression()
    model_ok.fit(train[['lag_1', 'lag_7']], train['ventas'])
    pred_ok = model_ok.predict(test[['lag_1', 'lag_7']])


    print("MODELO CON DATA LEAKAGE (lag_0):")
    print(f"  RMSE: {np.sqrt(mean_squared_error(test['ventas'], pred_leak)):.4f}")
    print(f"  MAPE: {mean_absolute_percentage_error(test['ventas'], pred_leak)*100:.4f}%")
    print()
    print("MODELO SIN DATA LEAKAGE (lag_1, lag_7):")
    print(f"  RMSE: {np.sqrt(mean_squared_error(test['ventas'], pred_ok)):.4f}")
    print(f"  MAPE: {mean_absolute_percentage_error(test['ventas'], pred_ok)*100:.4f}%")
    print("\n⚠️ En producción, lag_0 NO está disponible, por lo que el modelo con leakage fallará completamente.\n")
    print("=" * 150)
    #================================== Fin Pregunta 1.a ==================================

    print("""
    1.b) Interpretación de Tendencia con Apertura de Tiendas
        Pregunta: El componente de Tendencia (Tt) muestra un crecimiento sostenido del 15% anual. Sin embargo, 
        el equipo de finanzas advierte que este crecimiento se debe principalmente a la apertura de nuevas tiendas 
        (factor externo) y no a un aumento orgánico de la demanda por tienda. 
        ¿Cómo afecta esto la interpretación del forecast? Propón una estrategia para aislar la tendencia "real" de demanda.

        Respuesta:
            Impacto en la Interpretación del Forecast:
            1. Confusión de señales: La tendencia del 15% combina dos efectos muy diferentes:
            - Crecimiento extensivo: Nuevas tiendas (factor controlable por la empresa)
            - Crecimiento intensivo: Mayor demanda por tienda existente (factor de mercado)

            2. Problemas para la planificación:
            - Si la empresa no planea abrir más tiendas, el forecast sobreestimará la demanda
            - Si sí planea abrir tiendas, necesita saber cuánto crecimiento esperar por tienda
            - La asignación de inventario por tienda será incorrecta

            3. Riesgo de decisiones erróneas:
            - Sobreinversión en inventario central
            - Subinversión en capacidad por tienda
            - Evaluación incorrecta del desempeño de tiendas existentes

            Estrategia para Aislar la Tendencia "Real" de Demanda:
                Método 1 : Normalización por Numero de Tiendas 
                    Vn,t = Ventas Normalizadas en el periodo t
                    Vt,t = Ventas Totales en el periodo t
                    Nt,t = Numero de Tiendas en el periodo t
                    
                            Vn,t = Vt,t 
                                ----
                                Nt,t
                            
                Donde Nt,t es el numero de tiendas activas en el periodo t

                Método 2: Same-Store Sales (SSS)
                    Analizar únicamente tiendas que han operado durante todo el período de análisis:
                        * Las ventas totales del sistema en el período t (SSSt) se calculan como la suma 
                        de las ventas de todas las tiendas maduras en ese mismo período
                    Una tienda se considera madura cuando ha estado operativa por al menos 12 meses.
                
                Método 3: Modelo Jerárquico con Variable Exógena
                    Incluir el número de tiendas como regresor en el modelo:}

                        Ŷt=f(features temporales) + β * Nt
                
                Método 4: Descomposición en Dos Etapas
                    Paso 1: Descomponer ventas por tienda promedio
                            ventas_por_tienda = ventas_total / n_tiendas
                    
                    Paso 2: Aplicar STL a ventas_por_tienda
                            T_real, S, R = STL(ventas_por_tienda)

                    Paso 3: Para forecast total, multiplicar por n_tiendas futuras
                            forecast_total = forecast_por_tienda * n_tiendas_proyectado

            Recomendación:
                Usar una combinación de Método 2 (SSS) para diagnóstico y Método 3 (modelo con variable exógena) para forecast, ya que permite:
                    - Entender la demanda orgánica real (SSS)
                    - Proyectar con diferentes escenarios de expansión (modelo paramétrico)        
    """)
    print("=" * 50)
    print("Demostración: Aislando la tendencia real")
    print("=" * 50)

    # Simulación de datos
    np.random.seed(42)
    dias = 365 * 2  # 2 años

    # Número de tiendas: comienza con 10, abre 1 cada 2 meses
    n_tiendas = np.floor(10 + np.arange(dias) / 60).astype(int)

    # Demanda por tienda: crecimiento orgánico del 3% anual + estacionalidad
    demanda_por_tienda = 100 * (1 + 0.03/365) ** np.arange(dias) + 15 * np.sin(np.arange(dias) * 2 * np.pi / 7)
    demanda_por_tienda += np.random.randn(dias) * 5

    # Ventas totales = demanda_por_tienda * n_tiendas
    ventas_totales = demanda_por_tienda * n_tiendas

    # Crear DataFrame
    df_demo = pd.DataFrame({
        'fecha': pd.date_range('2024-01-01', periods=dias),
        'ventas_totales': ventas_totales,
        'n_tiendas': n_tiendas,
        'ventas_por_tienda': demanda_por_tienda
    })
    # Cálculo de tasas de crecimiento
    crecimiento_total = (df_demo['ventas_totales'].iloc[-30:].mean() / df_demo['ventas_totales'].iloc[:30].mean() - 1) * 100
    crecimiento_por_tienda = (df_demo['ventas_por_tienda'].iloc[-30:].mean() / df_demo['ventas_por_tienda'].iloc[:30].mean() - 1) * 100

    print(f"\nCrecimiento aparente (ventas totales): {crecimiento_total:.1f}% en 2 años")
    print(f"Crecimiento real (ventas por tienda): {crecimiento_por_tienda:.1f}% en 2 años")
    print(f"\n→ Diferencia explicada por apertura de nuevas tiendas: {crecimiento_total - crecimiento_por_tienda:.1f} puntos porcentuales")
    print("=" * 150)
    #================================== Fin Pregunta 1.b ==================================
    print("""
    1c) Codificación Cíclica con Funciones Trigonométricas
        Pregunta: La variable `dia_semana` codificada como entero (0-6) presenta un problema conceptual. 
        Explica por qué la codificación numérica directa es inadecuada para series temporales cíclicas y describe 
        matemáticamente cómo implementar una codificación cíclica usando funciones trigonométricas.

        Respuesta:
            ¿Por qué la codificación numérica directa (0-6) es inadecuada?
                1. Pérdida de continuidad cíclica:
                    - Con codificación entera: Lunes(0) → Domingo(6) → Lunes(0)
                    - El modelo interpreta que la "distancia" entre Domingo(6) y Lunes(0) es 6 unidades
                    - En realidad, la distancia temporal es solo 1 día (son días consecutivos)

                2. Imposición de orden artificial:
                    - El modelo asume que Jueves(3) + Jueves(3) = Domingo(6), lo cual no tiene sentido
                    - También asume que Miércoles está "más cerca" de Viernes que de Lunes, cuando todos son equidistantes

                3. Problema geométrico:
                    - Los días de la semana forman un círculo, no una línea
                    - Una representación lineal no captura esta topología

        Solución: Codificación Cíclica con Funciones Trigonométricas
            Se debe transformar cada valor cíclico en coordenadas de un círculo unitario usando seno y coseno:
                Formula:

                    sin_feature = sin(2 * π * x / P)
                    cos_feature = cos(2 * π * x / P)

                Donde:
                    x = valor cíclico (0, 1, 2, ...)
                    P = período del ciclo (7 para días de la semana, 12 para meses, 24 para horas)

                Para día de la semana (P = 7):
                    
                        dia_sin = sin(2 * π * dia_semana / 7)
                        dia_cos = cos(2 * π * dia_semana / 7)

        ¿Por qué funciona?
            1. Continuidad preservada: Los valores forman un círculo continuo donde Domingo y Lunes están adyacentes
            2. Distancias correctas: La distancia euclidiana entre puntos en el círculo refleja la distancia temporal real
            3. Sin ordenamiento artificial: Ningún día es "mayor" o "menor" que otro en términos absolutos   
        
        Tabla de transformación para día de semana:
        -----------------------------------------------
        | Día       | Valor | sin(2πx/7) | cos(2πx/7) |
        |-----------|-------|------------|------------|
        | Lunes     | 0     | 0.000      | 1.000      |
        | Martes    | 1     | 0.782      | 0.623      |
        | Miércoles | 2     | 0.975      | -0.223     |
        | Jueves    | 3     | 0.434      | -0.901     |
        | Viernes   | 4     | -0.434     | -0.901     |
        | Sábado    | 5     | -0.975     | -0.223     |
        | Domingo   | 6     | -0.782     | 0.623      |
        -----------------------------------------------
    """)
    print("=" * 50)
    print("Demostración: Implementación de codificación cíclica")
    print("=" * 50)

    import matplotlib.pyplot as plt
    import numpy as np

    def codificacion_ciclica(valores, periodo):
        """
        Convierte valores cíclicos a coordenadas seno/coseno.
        
        Parámetros:
        -----------
        valores : array-like
            Valores numéricos (0, 1, ..., periodo-1)
        periodo : int
            Longitud del ciclo (7 para días, 12 para meses, etc.)
        
        Retorna:
        --------
        sin_vals, cos_vals : tuple de arrays
        """
        valores = np.array(valores)
        sin_vals = np.sin(2 * np.pi * valores / periodo)
        cos_vals = np.cos(2 * np.pi * valores / periodo)
        return sin_vals, cos_vals

    # Aplicación a días de la semana
    dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    valores_dia = np.arange(7)

    sin_dia, cos_dia = codificacion_ciclica(valores_dia, 7)

    # Tabla de resultados
    print("Codificación Cíclica - Días de la Semana")
    print("=" * 50)
    print(f"{'Día':<12} {'Valor':<8} {'sin(2πx/7)':<12} {'cos(2πx/7)'}")
    print("-" * 50)
    for i, dia in enumerate(dias):
        print(f"{dia:<12} {i:<8} {sin_dia[i]:<12.4f} {cos_dia[i]:.4f}")

    # Demostrar distancias
    print("\nDistancias Euclidianas entre días:")
    print("-" * 40)
    # Distancia Lunes-Domingo (codificación lineal)
    dist_lineal = abs(0 - 6)
    # Distancia Lunes-Domingo (codificación cíclica)
    dist_ciclica = np.sqrt((cos_dia[0] - cos_dia[6])**2 + (sin_dia[0] - sin_dia[6])**2)
    # Distancia Lunes-Martes (codificación cíclica)  
    dist_lun_mar = np.sqrt((cos_dia[0] - cos_dia[1])**2 + (sin_dia[0] - sin_dia[1])**2)

    print(f"Lunes-Domingo (lineal): {dist_lineal} unidades")
    print(f"Lunes-Domingo (cíclica): {dist_ciclica:.4f} unidades")
    print(f"Lunes-Martes (cíclica): {dist_lun_mar:.4f} unidades")
    print(f"\n→ En codificación cíclica, Lunes-Domingo ≈ Lunes-Martes ✓")

    print("=" * 50)
    print("Implementación práctica en un DataFrame")
    print("=" * 50)
    import pandas as pd

    # Crear datos de ejemplo
    fechas = pd.date_range('2024-01-01', periods=30, freq='D')
    df_ejemplo = pd.DataFrame({'fecha': fechas})

    # Codificación original (problemática)
    df_ejemplo['dia_semana'] = df_ejemplo['fecha'].dt.dayofweek
    df_ejemplo['mes'] = df_ejemplo['fecha'].dt.month

    # Codificación cíclica (correcta)
    # Para día de semana (período = 7)
    df_ejemplo['dia_sin'] = np.sin(2 * np.pi * df_ejemplo['dia_semana'] / 7)
    df_ejemplo['dia_cos'] = np.cos(2 * np.pi * df_ejemplo['dia_semana'] / 7)

    # Para mes (período = 12)
    df_ejemplo['mes_sin'] = np.sin(2 * np.pi * (df_ejemplo['mes'] - 1) / 12)
    df_ejemplo['mes_cos'] = np.cos(2 * np.pi * (df_ejemplo['mes'] - 1) / 12)

    print("DataFrame con codificación cíclica:")
    print(df_ejemplo[['fecha', 'dia_semana', 'dia_sin', 'dia_cos', 'mes', 'mes_sin', 'mes_cos']].head(10))
    print("=" * 150)
    #================================== Fin Pregunta 1.c ==================================
    print("""
    1.d) Residuos Post-Feriados y Variables Exógenas

        Pregunta: Al analizar los residuos ($R_t$), el equipo detecta que los días posteriores a feriados nacionales 
        (como el "Día de los Difuntos") muestran residuos sistemáticamente positivos de +25%. ¿Qué tipo de variable exógena 
        deberían incorporar y cómo impactaría esto en la decisión de qué modelo utilizar (estadístico vs. ML)?

        Respuesta:
            Diagnóstico del Problema:
                Los residuos sistemáticamente positivos (+25%) en días post-feriados indican que:
                    1. Existe un patrón predecible no capturado por la descomposición STL
                    2. La estacionalidad semanal regular no contempla estos eventos especiales
                    3. Hay una demanda latente diferida que se materializa después del feriado

            Variables Exógenas a Incorporar:
                    1. Variable binaria post-feriado:              
                        df['post_feriado'] = df['es_feriado'].shift(1).fillna(0)

                    2. Ventana extendida post-feriado (si el efecto persiste):
                        df['post_feriado_1d'] = df['es_feriado'].shift(1)
                            df['post_feriado_2d'] = df['es_feriado'].shift(2)

                    3. Tipo de feriado (si hay diferencias):
                        df['post_feriado_consumo'] = df['feriado_consumo'].shift(1)  # Navidad, etc.
                        df['post_feriado_cultural'] = df['feriado_cultural'].shift(1)  # Día Difuntos

                    4. Indicador pre-feriado (posible efecto anticipación):
                        df['pre_feriado'] = df['es_feriado'].shift(-1).fillna(0)
                    5. Interacciones con día de semana:
                        df['feriado_lunes'] = df['post_feriado'] * (df['dia_semana'] == 0)
    
            Impacto en la Decisión: Modelo Estadístico vs. ML
                --------------------------------------------------------------------------------------------------
                | Criterio           | Modelo Estadístico (SARIMAX)      | Modelo ML (XGBoost, LightGBM)         |
                |--------------------|-----------------------------------|---------------------------------------|
                | Manejo de exógenas | Requiere especificación explícita | Aprende automáticamente interacciones |
                | Interacciones      | Deben definirse manualmente       | Descubre patrones no lineales         |
                | Interpretabilidad  | Coeficientes claros               | Requiere SHAP/LIME                    |
                | Datos necesarios   | Menos datos, más parsimonioso     | Más datos para mejor performance      |
                | Tipos de feriados  | Limitado a features manuales      | Puede diferenciar automáticamente     |
                --------------------------------------------------------------------------------------------------

            Recomendación:
                Dado el contexto descrito, se recomienda usar un modelo de Machine Learning por las siguientes razones:
                    1. Complejidad de las interacciones: Los efectos post-feriado pueden variar según:
                        - Tipo de feriado (religioso, cívico, comercial)
                        - Día de la semana en que cae
                        - Si forma un "puente" con el fin de semana
                        - Mes del año

                    2. No linealidades: El efecto del +25% puede no ser constante:
                        - Podría ser +30% después de Navidad pero +15% después de otros feriados
                        - Modelos como XGBoost capturan esto sin especificación manual

                    3. Escalabilidad: Ecuador tiene múltiples feriados nacionales y regionales; un modelo ML puede manejar la 
                        complejidad sin explosión de features

                Sin embargo, un enfoque híbrido es óptimo:
                    - Usar SARIMAX como baseline interpretable
                    - Usar ML (XGBoost/LightGBM) para capturar no linealidades
                    - Comparar performance en validación cruzada temporal

                Modelos específicos recomendados:
                    - SARIMAX con regresores exógenos (si se prefiere interpretabilidad)
                    - Prophet (incluye manejo nativo de feriados)
                    - LightGBM/XGBoost (si se tiene suficiente historia y se busca mejor accuracy)
    """)
    print("=" * 50)
    print("Demostración: Implementación de variables exógenas para feriados")
    print("=" * 50)
    import pandas as pd
    import numpy as np

    # Crear calendario de ejemplo (Ecuador)
    fechas = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    df_feriados = pd.DataFrame({'fecha': fechas})

    # Feriados nacionales de Ecuador 2024 (simplificado)
    feriados_ecuador = [
        '2024-01-01',  # Año Nuevo
        '2024-02-12',  # Carnaval
        '2024-02-13',  # Carnaval
        '2024-03-29',  # Viernes Santo
        '2024-05-01',  # Día del Trabajo
        '2024-05-24',  # Batalla de Pichincha
        '2024-08-10',  # Primer Grito Independencia
        '2024-10-09',  # Independencia de Guayaquil
        '2024-11-02',  # Día de los Difuntos
        '2024-11-03',  # Independencia de Cuenca
        '2024-12-25',  # Navidad
    ]
    feriados_ecuador = pd.to_datetime(feriados_ecuador)

    # Crear variables exógenas
    df_feriados['es_feriado'] = df_feriados['fecha'].isin(feriados_ecuador).astype(int)

    # Variables post-feriado
    df_feriados['post_feriado_1d'] = df_feriados['es_feriado'].shift(1).fillna(0).astype(int)
    df_feriados['post_feriado_2d'] = df_feriados['es_feriado'].shift(2).fillna(0).astype(int)

    # Variable pre-feriado
    df_feriados['pre_feriado'] = df_feriados['es_feriado'].shift(-1).fillna(0).astype(int)

    # Puente (día entre feriado y fin de semana)
    df_feriados['dia_semana'] = df_feriados['fecha'].dt.dayofweek
    df_feriados['es_puente'] = (
        (df_feriados['pre_feriado'] == 1) & 
        (df_feriados['dia_semana'] == 4)  # Viernes antes de feriado
    ).astype(int) | (
        (df_feriados['post_feriado_1d'] == 1) & 
        (df_feriados['dia_semana'] == 0)  # Lunes después de feriado
    ).astype(int)

    # Mostrar ejemplo alrededor del Día de los Difuntos
    mask = (df_feriados['fecha'] >= '2024-10-30') & (df_feriados['fecha'] <= '2024-11-06')
    print("Variables exógenas alrededor del Día de los Difuntos (2024):")
    print("=" * 80)
    cols = ['fecha', 'dia_semana', 'es_feriado', 'post_feriado_1d', 'post_feriado_2d', 'pre_feriado', 'es_puente']
    print(df_feriados.loc[mask, cols].to_string(index=False))


    print("=" * 50)
    print("Demostración: Comparación de modelos con y sin variables de feriados")
    print("=" * 50)

    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import warnings
    warnings.filterwarnings('ignore')

    # Simular ventas con efecto post-feriado
    np.random.seed(42)
    n_dias = len(df_feriados)

    # Base: tendencia + estacionalidad semanal + ruido
    base = 1000 + np.arange(n_dias) * 0.5
    estacionalidad = 50 * np.sin(2 * np.pi * df_feriados['dia_semana'] / 7)
    ruido = np.random.randn(n_dias) * 30

    # Efecto real post-feriado: +25% en días siguientes a feriados
    efecto_post_feriado = df_feriados['post_feriado_1d'] * 250  # +25% aprox

    df_feriados['ventas'] = base + estacionalidad + ruido + efecto_post_feriado

    # Preparar features
    # Codificación cíclica del día de semana
    df_feriados['dia_sin'] = np.sin(2 * np.pi * df_feriados['dia_semana'] / 7)
    df_feriados['dia_cos'] = np.cos(2 * np.pi * df_feriados['dia_semana'] / 7)

    # Features sin feriados
    features_basicas = ['dia_sin', 'dia_cos']

    # Features con feriados
    features_completas = ['dia_sin', 'dia_cos', 'es_feriado', 'post_feriado_1d', 
                        'post_feriado_2d', 'pre_feriado']

    # Train/test split temporal
    train = df_feriados.iloc[:300]
    test = df_feriados.iloc[300:]

    # Modelo 1: Sin variables de feriados
    model_basico = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_basico.fit(train[features_basicas], train['ventas'])
    pred_basico = model_basico.predict(test[features_basicas])

    # Modelo 2: Con variables de feriados
    model_completo = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_completo.fit(train[features_completas], train['ventas'])
    pred_completo = model_completo.predict(test[features_completas])

    # Métricas
    print("Comparación de Modelos")
    print("=" * 60)
    print(f"\nModelo SIN variables de feriados:")
    print(f"  MAE: {mean_absolute_error(test['ventas'], pred_basico):.2f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(test['ventas'], pred_basico)):.2f}")

    print(f"\nModelo CON variables de feriados:")
    print(f"  MAE: {mean_absolute_error(test['ventas'], pred_completo):.2f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(test['ventas'], pred_completo)):.2f}")

    # Mejora
    mejora_mae = (mean_absolute_error(test['ventas'], pred_basico) - 
                mean_absolute_error(test['ventas'], pred_completo))
    print(f"\n→ Mejora en MAE al incluir feriados: {mejora_mae:.2f} unidades")
    print(f"→ Reducción porcentual del error: {mejora_mae/mean_absolute_error(test['ventas'], pred_basico)*100:.1f}%")
    print("=" * 150)
    #================================== Fin Pregunta 1.d ==================================
    print("""
    Resumen de Conceptos Clave:
    ------------------------------------------------------------------------------------------------------------------------------------------------
    | Concepto                 | Problema                                               | Solución                                                 |
    |--------------------------|--------------------------------------------------------|----------------------------------------------------------|
    |   Data Leakage           | Usar información no disponible en producción           | Usar solo lags ≥ 1, validar con backtesting temporal     |
    |   Tendencia Confundida   | Mezclar crecimiento por expansión con demanda orgánica | Normalizar por número de tiendas o usar Same-Store Sales |
    |   Codificación Cíclica   | Representación lineal de variables circulares          | Transformar a (sin, cos) preservando continuidad         |
    |   Efectos Feriados       | Patrones especiales no capturados por STL              | Crear variables exógenas, considerar modelos ML          |
    ------------------------------------------------------------------------------------------------------------------------------------------------
    """)

if __name__ == "__main__":
    run()
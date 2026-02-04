# Pregunta 3: Selección de Algoritmos y Casos Especiales
def run() -> None:
    print("=" * 150)
    print("PARTE I: FORECASTING\n")
    print("Pregunta 3: Selección de Algoritmos y Casos Especiales")
    print("=" * 150)
    print("""
    3.a) Modelos para Demanda Intermitente - Long Tail
        Pregunta: Para los productos del segmento C (Long Tail), el equipo entrena un modelo ARIMA tradicional pero obtiene 
        predicciones siempre cercanas a la media histórica (aproximadamente 0.3 unidades/día), lo cual es inútil para planificación.

            - Explica por qué los modelos continuos (ARIMA, ETS) fallan con demanda intermitente.
            - Describe el modelo de Croston y explica cómo separa el problema en dos componentes.
            - ¿Qué distribución de probabilidad sería más apropiada para modelar la ocurrencia de demanda esporádica?

        Respuesta:
        
        ¿Por qué ARIMA y ETS fallan con demanda intermitente?
            1. Supuestos violados:
                ----------------------------------------------------------------------------
                | Supuesto del Modelo   | Realidad de Demanda Intermitente                 |
                ----------------------------------------------------------------------------
                | Distribución continua | Distribución mixta (ceros + valores positivos)   |
                | Errores normales      | Distribución altamente sesgada                   |
                | Varianza constante    | Varianza dependiente del nivel (0 vs positivo)   |
                | Autocorrelación suave | Patrones erráticos, picos esporádicos            |
                ----------------------------------------------------------------------------
        
            2. Problema de la media:
                Si un producto vende 3 unidades una vez cada 10 días:
                - Datos: [0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
                - Media: 0.3 unidades/día       
                - ARIMA predice ~ 0,3 simple
            
                Pero operacionalmente:
                    - 0.3 unidades/día es **imposible de ejecutar**
                    - ¿Cómo reponer 0.3 unidades? ¿Cuándo?
                    - Necesitamos saber: ¿Cuándo habrá demanda? y ¿Cuántas unidades?

            3. Pronósticos inútiles:
                Forecast ARIMA: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, ...]
                Realidad      : [0,   0,   0,   5,   0,   0,   0,   ...]
                                            ↑ El modelo NUNCA predice el pico
                                    
        El Modelo de Croston

                Idea clave: Separar el problema en dos componentes independientes:

                    formula:
                                Demanda =  ¿Cuándo?         x           ¿Cuánto?
                                        ---\/---                     ---\/---
                                Intervalo entre Demandas        Tamaño cuando ocurre
        
        Componentes del modelo:

            1. Qt : Intervalo entre demandas (frecuencias)
                - Tiempo (en período) desde la última demanda no-cero
                - Actualizado solo cuando ocurre una demanda
                - Suavizando exponencial: Q^t =  alfa * Qt + (1 - alfa) + Q^t -1
            
            2. Zt : Tamaño de la demanda (intensidad)
                - Cantidad vendida cuando hay demanda (exclurendo ceros)
                - Suavizando exponencial: Z^t =  alfa * Zt + (1 - alfa) + Z^t -1
            
            Forecast de Croston:

                        Y^ = Z^ 
                            ---
                            Q^
            Donde:
                - Z^ : estimación del tamaño promedio de demanda cuando ocurre
                - Q^ : estimación del intervalo promedio entre demandas
            
            Ejemplo numérico:

                Serie: [0, 0, 3, 0, 0, 0, 5, 0, 0, 2] 

                Demandas no cero: [3, 5, 2] → z̄ = 3.33
                Intervalos: [3, 4, 3] días → q̄ = 3.33

                Forecast Croston: 3.33 / 3.33 = 1.0 unidad/día promedio

                Interpretación operacional:
                    - Esperamos demanda cada ~3.3 días
                    - Cuando ocurre, será de ~3.3 unidades
                    - Política: reponer 10 unidades cada 10 días
        
        Distribución de Probabilidad para Demanda Esporádica

            Opciones recomendadas:
                -----------------------------------------------------------------------------------
                | Distribución          | Uso                                 | Parámetros        |
                |-----------------------|-------------------------------------|-------------------|
                | Poisson               | Eventos raros, conteo entero        | λ (tasa promedio) |
                | Binomial Negativa     | Sobre-dispersión (varianza > media) | r, p              |
                | Zero-Inflated Poisson | Exceso de ceros estructural         | λ, π              |
                -----------------------------------------------------------------------------------
        
        Recomendación: Distribución de Poisson

                    P(X=k) = λ^K * e^(-λ)
                            ------------
                                K!
            
                Donde:
                    - λ (lambda): tasa promedio de demanda diaria
                    - k: número de demandas            
            
            Justificación:
                1. Modela eventos discretos (ventas son unidades enteras)
                2. Apropiada para eventos raros e independientes
                3. Natural para procesos de conteo (llegadas de clientes)
                4. Permite calcular P(demanda > 0) = 1 - e^(-λ)

            Para casos con sobre-dispersión: Usar Binomial Negativa que permite varianza > media
    """)
    print("=" * 50)
    print(" Implementación del Modelo de Croston")
    print("=" * 50)
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def croston_method(demand, alpha=0.1):
        """
        Implementación del método de Croston para demanda intermitente.
        
        Parámetros:
        -----------
        demand : array-like
            Serie temporal de demanda (incluye ceros)
        alpha : float
            Parámetro de suavizado (0 < alpha < 1)
        
        Retorna:
        --------
        dict con forecast y componentes
        """
        demand = np.array(demand)
        n = len(demand)
        
        # Inicializar con primera demanda no cero
        first_nonzero = np.where(demand > 0)[0]
        if len(first_nonzero) == 0:
            return {'forecast': 0, 'z_hat': 0, 'q_hat': np.inf}
        
        # Componentes de Croston
        z_hat = demand[first_nonzero[0]]  # Tamaño de demanda inicial
        q_hat = first_nonzero[0] + 1 if first_nonzero[0] > 0 else 1  # Intervalo inicial
        
        q = 0  # Contador de períodos desde última demanda
        z_history = [z_hat]
        q_history = [q_hat]
        
        for t in range(n):
            q += 1
            if demand[t] > 0:
                # Actualizar solo cuando hay demanda
                z_hat = alpha * demand[t] + (1 - alpha) * z_hat
                q_hat = alpha * q + (1 - alpha) * q_hat
                q = 0  # Reiniciar contador
            z_history.append(z_hat)
            q_history.append(q_hat)
        
        # Forecast = z / q
        forecast = z_hat / q_hat if q_hat > 0 else 0
        
        return {
            'forecast': forecast,
            'z_hat': z_hat,  # Tamaño esperado de demanda
            'q_hat': q_hat,  # Intervalo esperado
            'z_history': z_history,
            'q_history': q_history
        }

    # Simular demanda intermitente
    np.random.seed(42)
    n_dias = 100

    # Generar demanda esporádica (promedio 1 venta cada 5 días, tamaño ~4 unidades)
    ocurrencias = np.random.binomial(1, 0.2, n_dias)  # 20% prob de demanda
    tamanos = np.random.poisson(4, n_dias)  # Tamaño si hay demanda
    demanda = ocurrencias * tamanos

    # Aplicar Croston
    resultado = croston_method(demanda, alpha=0.1)

    # Comparar con promedio simple (lo que haría ARIMA)
    media_simple = demanda.mean()

    print("=" * 60)
    print("COMPARACIÓN: ARIMA (Media) vs CROSTON")
    print("=" * 60)
    print(f"\nDatos de demanda intermitente:")
    print(f"  Total de días: {n_dias}")
    print(f"  Días con demanda > 0: {(demanda > 0).sum()} ({(demanda > 0).mean()*100:.1f}%)")
    print(f"  Días con demanda = 0: {(demanda == 0).sum()} ({(demanda == 0).mean()*100:.1f}%)")

    print(f"\n📊 Pronósticos:")
    print(f"  ARIMA/Media simple: {media_simple:.2f} unidades/día")
    print(f"  Croston: {resultado['forecast']:.2f} unidades/día")

    print(f"\n📊 Componentes de Croston:")
    print(f"  Tamaño esperado (z̄): {resultado['z_hat']:.2f} unidades cuando hay demanda")
    print(f"  Intervalo esperado (q̄): {resultado['q_hat']:.2f} días entre demandas")

    print(f"\n💡 Interpretación operacional:")
    print(f"  → Esperamos demanda cada ~{resultado['q_hat']:.1f} días")
    print(f"  → Cuando ocurre, será de ~{resultado['z_hat']:.1f} unidades")
    print(f"  → Sugerencia: reponer {int(resultado['z_hat'] * 2):.0f} unidades cada {int(resultado['q_hat'] * 2):.0f} días")

    print("=" * 50)
    print(" Visualización de demanda intermitente y Croston")
    print("=" * 50)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Panel 1: Demanda observada vs forecasts
    ax1 = axes[0]
    ax1.bar(range(n_dias), demanda, color='steelblue', alpha=0.7, label='Demanda real')
    ax1.axhline(media_simple, color='red', linestyle='--', linewidth=2, label=f'ARIMA/Media = {media_simple:.2f}')
    ax1.axhline(resultado['forecast'], color='green', linestyle='-', linewidth=2, label=f'Croston = {resultado["forecast"]:.2f}')
    ax1.set_xlabel('Día')
    ax1.set_ylabel('Demanda (unidades)')
    ax1.set_title('Demanda Intermitente: ARIMA vs Croston', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, 50)  # Mostrar primeros 50 días

    # Panel 2: Componentes de Croston
    ax2 = axes[1]
    ax2_twin = ax2.twinx()

    ax2.plot(resultado['z_history'], 'b-', linewidth=2, label='Tamaño (z)', alpha=0.8)
    ax2_twin.plot(resultado['q_history'], 'orange', linewidth=2, label='Intervalo (q)', alpha=0.8)

    ax2.set_xlabel('Período')
    ax2.set_ylabel('Tamaño de demanda (z)', color='blue')
    ax2_twin.set_ylabel('Intervalo entre demandas (q)', color='orange')
    ax2.set_title('Evolución de Componentes de Croston', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='orange')

    # Leyenda combinada
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.tight_layout()
    plt.show()

    print("=" * 60)
    print("Distribuciones para demanda esporádica")
    print("=" * 60)
    from scipy import stats
    import matplotlib.pyplot as plt
    import numpy as np

    # Parámetros
    lambda_poisson = 0.8  # Demanda promedio baja
    x = np.arange(0, 10)

    # Distribuciones
    poisson_pmf = stats.poisson.pmf(x, lambda_poisson)
    nbinom_pmf = stats.nbinom.pmf(x, n=2, p=0.7)  # Binomial Negativa

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Poisson
    axes[0].bar(x, poisson_pmf, color='steelblue', alpha=0.7)
    axes[0].set_title(f'Distribución Poisson (λ = {lambda_poisson})', fontsize=12)
    axes[0].set_xlabel('Demanda (unidades)')
    axes[0].set_ylabel('Probabilidad')
    axes[0].annotate(f'P(X=0) = {poisson_pmf[0]:.2f}', xy=(0, poisson_pmf[0]), 
                    xytext=(2, poisson_pmf[0]), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='red'))

    # Binomial Negativa
    axes[1].bar(x, nbinom_pmf, color='coral', alpha=0.7)
    axes[1].set_title('Distribución Binomial Negativa (sobre-dispersión)', fontsize=12)
    axes[1].set_xlabel('Demanda (unidades)')
    axes[1].set_ylabel('Probabilidad')

    plt.tight_layout()
    plt.show()

    print("Comparación de distribuciones para demanda esporádica:")
    print("=" * 50)
    print(f"\nPoisson (λ={lambda_poisson}):")
    print(f"  P(Demanda = 0): {poisson_pmf[0]:.2%}")
    print(f"  P(Demanda > 0): {1-poisson_pmf[0]:.2%}")
    print(f"  Media: {lambda_poisson}")
    print(f"  Varianza: {lambda_poisson} (igual a media)")

    print(f"\nBinomial Negativa:")
    print(f"  P(Demanda = 0): {nbinom_pmf[0]:.2%}")
    print(f"  Media: {stats.nbinom.mean(n=2, p=0.7):.2f}")
    print(f"  Varianza: {stats.nbinom.var(n=2, p=0.7):.2f} (mayor que media = sobre-dispersión)")
    print("=" * 150)
    #==================================== Fin de la Pregunta 3.a ====================================
    print("""
    3.b) NeuralForecast (NHITS) para Todo el Catálogo
        Pregunta: Se propone utilizar NeuralForecast (NHITS) para todo el catálogo argumentando que "deep learning siempre es mejor". Argumenta a favor o en contra considerando:
            - Tamaño de la historia disponible (2 años de datos diarios)
            - Costo computacional para 50,000 series
            - Capacidad de capturar dependencias de largo plazo vs. complejidad añadida
        
        Respuesta:
            Análisis de la Propuesta "Mi posición: EN CONTRA de usar NHITS para todo el catálogo, A FAVOR de usarlo selectivamente."

            1. Tamaño de Historia (2 años = ~730 días)
                -------------------------------------------------------------------------------
                | Segmento          | Datos Efectivos                 | Suficiencia para NHITS |
                |-------------------|---------------------------------|------------------------|
                | A (Alto volumen)  | 730 puntos densos               | ✅ Suficiente para DL  |
                | B (Medio volumen) | 730 puntos con patrones         | ⚠️ Marginal            |
                | C (Long Tail)     | ~150 puntos no-cero (730 × 20%) | ❌ Insuficiente        |
                -------------------------------------------------------------------------------

            Problema:
                - NHITS requiere miles de observaciones para aprender patrones complejos
                - Para segmento C (70% del catálogo), solo hay ~150 observaciones útiles
                - Riesgo de overfitting severo en productos de baja rotación

            2. Costo Computacional
                Estimación de recursos:

                    50,000 SKUs × 730 días = 36.5 millones de observaciones
                
                Entrenamiento NHITS (estimado):
                    - GPU A100: ~4-8 horas para entrenar modelo global
                    - Costo cloud: ~$50-100 por entrenamiento
                    - Re-entrenamiento semanal: ~$2,500-5,000/año

                Alternativa con LightGBM:
                    - CPU standard: ~30 minutos
                    - Costo: ~$1 por entrenamiento
                    - Re-entrenamiento semanal: ~$52/año
                
                Ratio costo/beneficio:
                    - NHITS cuesta **50-100x más** que modelos tradicionales
                    - ¿La mejora en accuracy justifica este costo?
            
            3. Dependencias de Largo Plazo
                    ------------------------------------------------------------------------------------------------
                    | Aspecto              | Ventaja NHITS                      | Limitación                       |
                    |----------------------|------------------------------------|----------------------------------|
                    | Estacionalidad anual | Puede capturar s/t anuales         | Con 2 años, solo 2 ciclos        |
                    | Tendencias           | Aprende tendencias no lineales     | Riesgo de extrapolación errónea  |
                    | Interacciones        | Detecta interacciones cross-series | Requiere muchos datos            |
                    | Holidays             | Puede aprender efectos especiales  | Solo 2 observaciones por feriado |
                    ------------------------------------------------------------------------------------------------
                
                Problema fundamental:
                    - Con 2 años, solo hay **2 observaciones** de cada evento anual (Navidad, Black Friday, etc.)
                    - Un modelo estadístico simple con regresores explícitos puede ser igual o más efectivo
                
            Recomendación: Estrategia Híbrida por Segmento
                    ----------------------------------------------------------------------------------------------------
                    | Segmento | % Catálogo | SKUs   | Modelo Recomendado | Justificación                              |
                    |----------|------------|--------|--------------------|--------------------------------------------|
                    | A        | 5%         | 2,500  | NHITS o N-BEATS    | Datos suficientes, alto impacto en revenue |
                    | B        | 25%        | 12,500 | LightGBM o Prophet | Balance accuracy/costo                     |
                    | C        | 70%        | 35,000 | Croston o SBA      | Modelos para demanda intermitente          |
                    ----------------------------------------------------------------------------------------------------
            
            Beneficios de la estrategia híbrida:
                1. Optimización de recursos: DL solo donde genera valor
                2. Mantenibilidad: Menos complejidad en producción
                3. Interpretabilidad: Modelos simples para diagnóstico
                4. Robustez: Menos riesgo de fallas catastróficas
            
        Conclusión
            "Deep Learning siempre es mejor" es FALSO en forecasting de retail.

        La realidad:
            - 70% del catálogo (Long Tail) no tiene suficientes datos para DL
            - El costo computacional no se justifica para todos los SKUs
            - Modelos más simples son más interpretables y mantenibles
            - El benchmark M5 (Kaggle) mostró que LightGBM compite con (y a veces supera) redes neuronales

        "The right model for the right data, not the fanciest model for all data."
    """)
    print("=" * 60)
    print("Análisis de costo-beneficio para selección de modelos")
    print("=" * 60)
    import pandas as pd
    import matplotlib.pyplot as plt

    # Datos del catálogo
    catalogo = pd.DataFrame({
        'Segmento': ['A (Alto)', 'B (Medio)', 'C (Long Tail)'],
        'Porcentaje': [5, 25, 70],
        'SKUs': [2500, 12500, 35000],
        'Obs_utiles_por_SKU': [730, 500, 150],
        'Contribucion_revenue': [40, 35, 25]  # % del revenue
    })

    # Costos estimados por modelo (USD/año)
    costos = {
        'NHITS': {'entrenamiento': 100, 'inferencia': 20, 'mantenimiento': 50},
        'LightGBM': {'entrenamiento': 5, 'inferencia': 1, 'mantenimiento': 10},
        'Croston': {'entrenamiento': 0.5, 'inferencia': 0.1, 'mantenimiento': 2}
    }

    print("=" * 70)
    print("ANÁLISIS DE COSTO-BENEFICIO: NHITS vs Estrategia Híbrida")
    print("=" * 70)

    # Opción 1: NHITS para todo
    costo_nhits_todo = 50000 * sum(costos['NHITS'].values()) / 1000  # Escalar
    print(f"\n📊 Opción 1: NHITS para 50,000 SKUs")
    print(f"  Costo anual estimado: ${costo_nhits_todo:,.0f}")

    # Opción 2: Híbrido
    costo_hibrido = (
        2500 * sum(costos['NHITS'].values()) / 1000 +     # Segmento A
        12500 * sum(costos['LightGBM'].values()) / 1000 + # Segmento B
        35000 * sum(costos['Croston'].values()) / 1000    # Segmento C
    )
    print(f"\n📊 Opción 2: Estrategia Híbrida")
    print(f"  Segmento A (2,500 SKUs): NHITS")
    print(f"  Segmento B (12,500 SKUs): LightGBM")
    print(f"  Segmento C (35,000 SKUs): Croston")
    print(f"  Costo anual estimado: ${costo_hibrido:,.0f}")

    print(f"\n💰 Ahorro con estrategia híbrida: ${costo_nhits_todo - costo_hibrido:,.0f} ({(1-costo_hibrido/costo_nhits_todo)*100:.0f}%)")

    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Gráfico 1: Distribución del catálogo
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    axes[0].pie(catalogo['SKUs'], labels=catalogo['Segmento'], autopct='%1.0f%%',
            colors=colors, explode=[0.05, 0, 0])
    axes[0].set_title('Distribución del Catálogo (50,000 SKUs)', fontsize=12)

    # Gráfico 2: Comparación de costos
    x = ['NHITS\n(Todo)', 'Híbrido']
    y = [costo_nhits_todo, costo_hibrido]
    bars = axes[1].bar(x, y, color=['#e74c3c', '#2ecc71'])
    axes[1].set_ylabel('Costo Anual (USD)')
    axes[1].set_title('Comparación de Costos Anuales', fontsize=12)
    for bar, val in zip(bars, y):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'${val:,.0f}', ha='center', fontsize=11)

    plt.tight_layout()
    plt.show()
    print("=" * 150)
    #==================================== Fin de la Pregunta 3.b ====================================
    print("""
    3.c) Forecast para Black Friday con Poca Historia
        Pregunta: El equipo necesita generar un forecast para la semana del Black Friday, pero solo tienen 2 observaciones 
        históricas de este evento.    

            - ¿Por qué los modelos puramente estadísticos tendrían dificultades extremas?
            - Propón una estrategia híbrida combinando modelo base + variables exógenas + ajuste experto.
            - ¿Cómo medirías el éxito del forecast post-evento?

        Respuesta:
                ¿Por qué los modelos estadísticos puros fallan con Black Friday?

                    1. Insuficiencia estadística:
                        - Solo 2 observaciones = cero grados de libertad para estimar varianza
                        - Imposible distinguir señal de ruido
                        - Intervalos de confianza serían infinitamente amplios

                    2. Heterogeneidad del evento:
                        - Cada Black Friday es diferente (ofertas, día de la semana, competencia)
                        - 2 puntos no capturan esta variabilidad

                    3. Efectos no estacionarios:
                        - El "efecto Black Friday" cambia año a año
                        - La cultura de consumo evoluciona (ej: pre-pandemia vs post-pandemia)
                    
                    4. Modelos de series de tiempo:
                            -------------------------------------------------------
                            | Modelo | Problema con Black Friday                   |
                            |--------|---------------------------------------------|
                            | ARIMA  | No tiene mecanismo para eventos únicos      |     
                            | ETS    | Estacionalidad se basa en ciclos regulares  |
                            | Prophet| Requiere múltiples ocurrencias para ajustar |
                            | NHITS  | Necesita patrones recurrentes para aprender |
                            --------------------------------------------------------

                Estrategia Híbrida Propuesta
                    Componente 1: Modelo Base (Baseline)
                        Forecast para una "semana normal" del mismo período:
                            # Usar modelo entrenado (Prophet, LightGBM, etc.)
                            # Predecir como si no hubiera Black Friday
                            baseline_forecast = modelo.predict(semana_black_friday)
                    
                    Componente 2: Variable Exógena de Evento
                        Crear indicador binario y estimar efecto:
                            df['es_black_friday'] = df['fecha'].isin(fechas_black_friday).astype(int)
                            # Estimar uplift histórico promedio
                            # (limitado pero mejor que nada)
                            uplift_historico = (
                                df[df['es_black_friday'] == 1]['ventas'].mean() /
                                df[df['es_black_friday'] == 0]['ventas'].mean()
                            )
                    
                    Componente 3: Ajuste Experto (Uplift Factor)
                        Incorporar conocimiento de negocio:

                    Inputs del equipo comercial:
                        - Presupuesto de marketing: +15% vs año anterior
                        - Descuentos planificados: 40% promedio (vs 35% año pasado)
                        - Categorías en promoción: 20 (vs 15 año pasado)
                        - Competencia: Amazon tiene sale el mismo día

                    Ajuste experto: multiplicador de 1.2 - 1.5 sobre histórico
            
        Fórmula Final:

                        FORECASTbf = BASELINE * UPLIFT_historico * FACTOR_experto
            
            Ejemplo:
                    Baseline (semana normal): 10,000 unidades
                    Uplift histórico (2 años): 2.5x promedio
                    Factor experto: 1.3 (más marketing este año)

                    Forecast BF = 10,000 * 2.5 * 1.3 = 32,500 unidades
        
        ¿Cómo medir el éxito sin backtesting convencional?
            
            1. Métricas Operacionales Post-Evento:
                    ---------------------------------------------------------------------------------------
                    | Métrica                   | Definición                         | Objetivo           |
                    |---------------------------|------------------------------------|--------------------|
                    | Fill Rate                 | % de demanda satisfecha            | > 95%              |
                    | Quiebre de Stock          | # SKUs sin inventario              | < 5%               |
                    | Sobre-stock Post-BF       | Inventario excedente día siguiente | < 10% del forecast |
                    | Ventas Perdidas Estimadas | Demanda no satisfecha * precio     | Minimizar          |
                    ---------------------------------------------------------------------------------------
            
            2. Análisis de Error Ponderado por Impacto:
                    
                    WMAPEbf = E|Yt - Y^i| * MARGENi
                            ---------------------
                                E(Yi * MARGENi)

                Ponderar por margen para priorizar productos más rentables.
            
            3. Análisis de Sesgo Direccional:

                    Si Forecast > Real: Sobre-stock (costo = almacenamiento)
                    Si Forecast < Real: Sub-stock (costo = ventas perdidas * 6)

                    Dado el ratio de costos 1:6, es MEJOR errar por exceso.
            
            4. Comparación con Benchmark Naive:
                
                Benchmark = Ventas BF del año pasado * (1 + crecimiento_anual)

                Si |Error_modelo| < |Error_benchmark|:
                    El modelo agregó valor
            
            5. Registro para Mejora Futura:

                    Documentar exhaustivamente:
                    - Forecast vs Real por categoría
                    - Factores externos (clima, competencia, macro)
                    - Efectividad de promociones

                Esto genera la tercera observación para el próximo año.
    """)
    print("=" * 70)
    print("Implementación de estrategia híbrida para Black Friday")
    print("=" * 70)
    import pandas as pd
    import numpy as np

    def forecast_black_friday(baseline, uplift_historico, factor_experto, 
                            incertidumbre=0.2):
        """
        Genera forecast para Black Friday con estrategia híbrida.
        
        Parámetros:
        -----------
        baseline : float
            Forecast para semana normal (sin evento)
        uplift_historico : float
            Multiplicador promedio de BF históricos (ej: 2.5 = +150%)
        factor_experto : float
            Ajuste del equipo comercial (ej: 1.2 = +20% vs histórico)
        incertidumbre : float
            Nivel de incertidumbre para intervalos (default 20%)
        
        Retorna:
        --------
        dict con forecast puntual e intervalos
        """
        # Forecast puntual
        forecast_puntual = baseline * uplift_historico * factor_experto
        
        # Intervalos (dado que solo hay 2 observaciones, usamos heurística)
        forecast_bajo = forecast_puntual * (1 - incertidumbre)
        forecast_alto = forecast_puntual * (1 + incertidumbre)
        
        # Escenarios
        escenario_conservador = baseline * uplift_historico * 0.9  # -10% factor
        escenario_optimista = baseline * uplift_historico * factor_experto * 1.15  # +15%
        
        return {
            'forecast_puntual': forecast_puntual,
            'intervalo_bajo': forecast_bajo,
            'intervalo_alto': forecast_alto,
            'escenario_conservador': escenario_conservador,
            'escenario_optimista': escenario_optimista,
            'componentes': {
                'baseline': baseline,
                'uplift_historico': uplift_historico,
                'factor_experto': factor_experto
            }
        }

    # Ejemplo de aplicación
    print("=" * 60)
    print("FORECAST BLACK FRIDAY - ESTRATEGIA HÍBRIDA")
    print("=" * 60)

    # Inputs
    baseline_normal = 10000  # Ventas semana normal
    uplift_bf = 2.5  # Histórico: BF vende 2.5x más
    factor_comercial = 1.3  # Este año: +30% marketing

    # Generar forecast
    resultado = forecast_black_friday(baseline_normal, uplift_bf, factor_comercial)

    print(f"\n📊 Inputs:")
    print(f"  Baseline (semana normal): {baseline_normal:,} unidades")
    print(f"  Uplift histórico BF: {uplift_bf}x")
    print(f"  Factor experto: {factor_comercial}x")

    print(f"\n📈 Forecast Black Friday:")
    print(f"  Puntual: {resultado['forecast_puntual']:,.0f} unidades")
    print(f"  Intervalo: [{resultado['intervalo_bajo']:,.0f} - {resultado['intervalo_alto']:,.0f}]")

    print(f"\n📋 Escenarios para planificación:")
    print(f"  Conservador (plan mínimo): {resultado['escenario_conservador']:,.0f} unidades")
    print(f"  Base (plan operativo): {resultado['forecast_puntual']:,.0f} unidades")
    print(f"  Optimista (capacidad máxima): {resultado['escenario_optimista']:,.0f} unidades")

    print(f"\n💡 Recomendación de inventario:")
    print(f"  Stock objetivo: {resultado['escenario_optimista']:,.0f} unidades")
    print(f"  (Errar por exceso es 6x menos costoso que por defecto)")

    print("=" * 70)
    print("# Análisis post-evento Black Friday")
    print("=" * 70)
    import pandas as pd
    import numpy as np

    def evaluar_forecast_post_evento(forecast, real, inventario_inicial,
                                    costo_quiebre=3.0, costo_sobrestock=0.5):
        """
        Evalúa el éxito del forecast después de Black Friday.
        
        Parámetros:
        -----------
        forecast : float
            Unidades pronosticadas
        real : float
            Unidades vendidas realmente
        inventario_inicial : float
            Stock disponible al inicio del evento
        costo_quiebre : float
            Costo por unidad no vendida por falta de stock
        costo_sobrestock : float
            Costo por unidad excedente post-evento
        """
        # Métricas de error
        error_absoluto = abs(forecast - real)
        error_porcentual = error_absoluto / real * 100
        bias = (forecast - real) / real * 100
        
        # Métricas operacionales
        demanda_satisfecha = min(real, inventario_inicial)
        fill_rate = demanda_satisfecha / real * 100
        
        # Costos
        if inventario_inicial < real:  # Quiebre
            ventas_perdidas = real - inventario_inicial
            costo_total = ventas_perdidas * costo_quiebre
            tipo_error = "Sub-stock (QUIEBRE)"
        else:  # Sobre-stock
            excedente = inventario_inicial - real
            costo_total = excedente * costo_sobrestock
            tipo_error = "Sobre-stock"
        
        return {
            'forecast': forecast,
            'real': real,
            'inventario': inventario_inicial,
            'error_absoluto': error_absoluto,
            'error_porcentual': error_porcentual,
            'bias': bias,
            'fill_rate': fill_rate,
            'costo_total': costo_total,
            'tipo_error': tipo_error
        }

    # Simular diferentes escenarios post-evento
    print("=" * 70)
    print("EVALUACIÓN POST-EVENTO BLACK FRIDAY")
    print("=" * 70)

    escenarios = [
        {'nombre': 'Escenario A: Forecast preciso', 'forecast': 32500, 'real': 31000, 'inventario': 35000},
        {'nombre': 'Escenario B: Sub-estimación', 'forecast': 25000, 'real': 35000, 'inventario': 27000},
        {'nombre': 'Escenario C: Sobre-estimación', 'forecast': 40000, 'real': 28000, 'inventario': 42000},
    ]

    resultados = []
    for esc in escenarios:
        r = evaluar_forecast_post_evento(esc['forecast'], esc['real'], esc['inventario'])
        r['nombre'] = esc['nombre']
        resultados.append(r)
        
        print(f"\n{esc['nombre']}:")
        print(f"  Forecast: {r['forecast']:,} | Real: {r['real']:,} | Inventario: {r['inventario']:,}")
        print(f"  Error: {r['error_porcentual']:.1f}% | Bias: {r['bias']:+.1f}%")
        print(f"  Fill Rate: {r['fill_rate']:.1f}%")
        print(f"  Tipo: {r['tipo_error']}")
        print(f"  Costo: ${r['costo_total']:,.2f}")

    print("\n" + "=" * 70)
    print("📊 COMPARACIÓN DE COSTOS:")
    df_resultados = pd.DataFrame(resultados)[['nombre', 'error_porcentual', 'fill_rate', 'tipo_error', 'costo_total']]
    df_resultados.columns = ['Escenario', 'Error %', 'Fill Rate %', 'Tipo', 'Costo $']
    print(df_resultados.to_string(index=False))

    print("\n💡 Conclusión: El Escenario B (sub-estimación) tiene MAYOR costo")
    print("   a pesar de tener menor error porcentual que C, debido a la")
    print("   asimetría de costos (quiebre = 6x sobre-stock).")
    print("=" * 150)
    #============================================ Fin Pregunta 3.c ============================================
    print("""
    Resumen de Pregunta 3
    ----------------------------------------------------------------------------------------------------------------------
    | Concepto                   | Problema                                  | Solución                                  |
    |----------------------------|-------------------------------------------|-------------------------------------------|
    | Demanda Intermitente       | ARIMA predice siempre la media (inútil)   | Croston: separar frecuencia e intensidad  |
    | Deep Learning para todo    | Costoso y sobreajuste en Long Tail        | Estrategia híbrida por segmento           |
    | Eventos con poca historia  | No hay suficientes datos para estadística | Modelo base + exógenas + juicio experto   |
    | Evaluación sin backtesting | Solo 2 observaciones históricas           | Métricas operacionales post-evento        |
    ----------------------------------------------------------------------------------------------------------------------

    Fórmulas Clave:

            Croston:  

                        Y^ = Z^ 
                            ---
                            Q^
            

            Forecast Black Friday:

                        FORECASTbf = BASELINE * UPLIFT_historico * FACTOR_experto
    """)
    print("=" * 150)
if __name__ == "__main__":
    run()
# Pregunta 2: Validación y Métricas de Negocio
def run() -> None:
    print("=" * 150)
    print("PARTE I: FORECASTING\n")
    print("Pregunta 2: Validación y Métricas de Negocio")
    print("=" * 150)
    print("""
    2.a) Interpretación del Bias Negativo y Costo Operacional
        Pregunta: A pesar de tener el menor WAPE, el modelo LightGBM presenta un Bias de -8%. Interpreta este resultado en 
        términos operacionales: ¿el modelo tiende a sobre-stockear o sub-stockear? Calcula el costo esperado adicional del 
        sesgo si el volumen promedio semanal es de 10,000 unidades.

        Respuesta:
            Interpretación del Bias = -8%:
                El Bias (sesgo) mide la tendencia sistemática del modelo a sobreestimar o subestimar:
                
                Formula:
                                Bias= E(Ŷt- Yt)
                                    --------- x 100%
                                    E(Yt)

                    - Bias Positivo(+) : El modelo predice más de lo real --> Sobrestock
                    - Bias Negativo(-): El modelo predice menos de lo real --> Substock
                                
                Con Bias = -8%:
                    - LightGBM subestima sistemáticamente las ventas en un 8%
                    - Operacionalmente: TIENDE A SUB-STOCKEAR
                    - Se compra/produce menos inventario del necesario
                    - Resultado: quiebres de stock recurrentes

            Cálculo del Costo Esperado Adicional:
                Datos:
                    - Volumen promedio semanal: 10,000 unidades
                    - Bias: -8% (subestimación)
                    - Costo de quiebre: $3.00 por unidad

                Cálculo:

                    1. Unidades subestimads por semana:

                                Unidades_faltantes = 10.000 x 8% = 800 unidades / semana

                    2. Costo semanal por sesgo:

                                Costo_semanal = 800 x $3,00 = $2.400 / semana

                    3. Costo anual proyectado:

                                Costo_anual = 2.400 x 52 semanas = $124.800 / año
                
                Comparación con modelos alternativos:
                    ----------------------------------------------------------------------------------------
                    | Modelo   | Bias | Tipo         | Unidades Afectadas | Costo Unitario | Costo Semanal |
                    |----------|------|--------------|--------------------|----------------|---------------|
                    | LightGBM | -8%  | Quiebre      |         800        |     $3.00      |    $2,400     |
                    | Prophet  | +1%  | Sobre-stock  |         100        |     $0.50      |    $50        |
                    | Baseline | +2%  | Sobre-stock  |         200        |     $0.50      |    $100       |
                    ----------------------------------------------------------------------------------------

                Conclusión: A pesar de tener mejor WAPE, LightGBM es el modelo más costoso operacionalmente 
                debido a su sesgo negativo y la asimetría de costos (quiebre 6x más caro que sobre-stock). 
    """)
    print("=" * 50)
    print("Demostración: Análisis de Costo por Sesgo (BIAS)")
    print("=" * 50)
    import pandas as pd
    import numpy as np

    # Parámetros
    volumen_semanal = 10000  # unidades
    costo_sobrestock = 0.50  # $/unidad
    costo_quiebre = 3.00     # $/unidad

    # Modelos y sus bias
    modelos = {
        'Baseline (Naive)': {'MAE': 45, 'WAPE': 18.5, 'Bias': 0.02},
        'LightGBM': {'MAE': 32, 'WAPE': 12.1, 'Bias': -0.08},
        'Prophet': {'MAE': 38, 'WAPE': 14.3, 'Bias': 0.01}
    }

    print("=" * 70)
    print("ANÁLISIS DE COSTO POR SESGO (BIAS)")
    print("=" * 70)
    print(f"\nVolumen semanal: {volumen_semanal:,} unidades")
    print(f"Costo sobre-stock: ${costo_sobrestock:.2f}/unidad")
    print(f"Costo quiebre: ${costo_quiebre:.2f}/unidad")
    print(f"\nRatio de asimetría: {costo_quiebre/costo_sobrestock:.0f}x (quiebre vs sobre-stock)")
    print("\n" + "-" * 70)

    resultados = []
    for nombre, metricas in modelos.items():
        bias = metricas['Bias']
        unidades_afectadas = abs(bias) * volumen_semanal
        
        if bias < 0:  # Subestimación -> Quiebre
            tipo = "Quiebre (sub-stock)"
            costo_unitario = costo_quiebre
        else:  # Sobreestimación -> Sobre-stock
            tipo = "Sobre-stock"
            costo_unitario = costo_sobrestock
        
        costo_semanal = unidades_afectadas * costo_unitario
        costo_anual = costo_semanal * 52
        
        resultados.append({
            'Modelo': nombre,
            'Bias': f"{bias:+.0%}",
            'Tipo': tipo,
            'Unidades/Semana': int(unidades_afectadas),
            'Costo Semanal': f"${costo_semanal:,.0f}",
            'Costo Anual': f"${costo_anual:,.0f}"
        })
        
        print(f"\n{nombre}:")
        print(f"  Bias: {bias:+.0%} → {tipo}")
        print(f"  Unidades afectadas: {unidades_afectadas:,.0f}/semana")
        print(f"  Costo semanal: ${costo_semanal:,.2f}")
        print(f"  Costo anual proyectado: ${costo_anual:,.2f}")

    print("\n" + "=" * 70)
    print("\n📊 RESUMEN COMPARATIVO:")
    df_resultados = pd.DataFrame(resultados)
    print(df_resultados.to_string(index=False))

    print("\n⚠️  CONCLUSIÓN: LightGBM tiene el menor WAPE pero el MAYOR costo operacional debido al sesgo negativo y la asimetría de costos.")
    print("=" * 150)
    #================================== Fin Pregunta 2.a ==================================
    print("""
    2.b) Validación Cruzada para Series Temporales
        Pregunta: El equipo de ML propone usar validación cruzada K-Fold tradicional (aleatorio) para optimizar hiperparámetros. 
        Explica por qué esta estrategia es incorrecta para series temporales y describe con detalle la diferencia entre 
        Expanding Window y Sliding Window. ¿En qué escenario de retail preferirías cada una?

        Respuesta:
            ¿Por qué K-Fold tradicional es incorrecto para series temporales?
            El K-Fold aleatorio viola el principio fundamental de las series temporales: la causalidad temporal.

            Problemas específicos:
                1. Data Leakage Temporal:
                    - K-Fold mezcla datos pasados y futuros aleatoriamente
                    - El modelo puede "entrenar en el futuro" y "validar en el pasado"
                    - Ejemplo: Entrena con datos de diciembre 2024, valida con octubre 2024

                2. Ruptura de autocorrelación:
                    - Las observaciones consecutivas están correlacionadas
                    - Al separar aleatoriamente, se rompe esta estructura
                    - La varianza de los errores de validación se subestima

                3. Métricas infladas artificialmente:
                    - El modelo "ve" patrones cercanos al punto de validación
                    - RMSE/MAE en validación << RMSE/MAE en producción real

                4. No simula el escenario real:
                    - En producción, SIEMPRE predecimos el futuro con datos del pasado
                    - Nunca tenemos acceso a información futura
            
            Estrategias Correctas: Expanding Window vs. Sliding Window

                Expanding Window (Ventana Expansiva)
                Fold 1: [████████░░░░░░░░░░░░] Train → [██] Test
                Fold 2: [████████████░░░░░░░░] Train → [██] Test  
                Fold 3: [████████████████░░░░] Train → [██] Test
                Fold 4: [████████████████████] Train → [██] Test
                        ←────── Tiempo ──────→
            
            Características:
                - La ventana de entrenamiento crece con cada fold
                - Incluye toda la historia disponible hasta el punto de corte
                - La ventana de test es fija y se desplaza hacia adelante

            Ventajas:
                - Maximiza uso de datos históricos
                - Captura tendencias de largo plazo
                - Simula escenario real de "re-entrenamiento periódico con toda la historia"
            
            Sliding Window (Ventana Deslizante)
                Fold 1: [████████░░░░░░░░░░░░] Train → [██] Test
                Fold 2: [░░████████░░░░░░░░░░] Train → [██] Test  
                Fold 3: [░░░░████████░░░░░░░░] Train → [██] Test
                Fold 4: [░░░░░░████████░░░░░░] Train → [██] Test
                        ←────── Tiempo ──────→
            
            Características:
                - La ventana de entrenamiento tiene tamaño fijo
                - Se "desliza" hacia adelante, descartando datos antiguos
                - Tanto train como test se mueven en el tiempo

            Ventajas:
                - Captura cambios de régimen (concept drift)
                - Más robusto cuando datos antiguos son irrelevantes
                - Computacionalmente más eficiente
        
        ¿Cuándo usar cada estrategia en Retail?
    ------------------------------------------------------------------------------------------------------------------------------------------------
    | Escenario                                        | Estrategia Recomendada | Justificación                                                     |
    |--------------------------------------------------|------------------------|-------------------------------------------------------------------|
    | Productos estables (alimentos básicos, limpieza) | Expanding Window       | Patrones estacionales consistentes, historia larga es valiosa     |
    | Moda/Tendencias (ropa, electrónica)              | Sliding Window         | Tendencias cambian rápido, datos de 2 años atrás son irrelevantes |
    | Post-pandemia/Post-crisis                        | Sliding Window         | El comportamiento pre-crisis ya no aplica                         |
    | Categorías nuevas (<1 año de historia)           | Expanding Window       | Necesitamos toda la data disponible                               |
    | Productos con ciclo de vida corto                | Sliding Window         | Solo datos recientes son representativos                          |
    | Forecast a largo plazo (12+ meses)               | Expanding Window       | Necesitamos capturar estacionalidad completa                      |
    ------------------------------------------------------------------------------------------------------------------------------------------------
    """)
    print("=" * 50)
    print("Visualización de las estrategias de validación")
    print("=" * 50)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Configuración común
    n_points = 20
    n_folds = 4
    test_size = 2

    colors = {'train': '#3498db', 'test': '#e74c3c', 'unused': '#ecf0f1'}

    # ============ Panel 1: K-Fold Tradicional (INCORRECTO) ============
    ax1 = axes[0]
    ax1.set_title('K-Fold Tradicional (INCORRECTO para Series Temporales)', fontsize=12, fontweight='bold')

    np.random.seed(42)
    for fold in range(n_folds):
        y = n_folds - fold - 1
        # Simular selección aleatoria
        test_indices = np.random.choice(n_points, test_size * 2, replace=False)
        for i in range(n_points):
            color = colors['test'] if i in test_indices else colors['train']
            ax1.barh(y, 1, left=i, color=color, edgecolor='white', height=0.6)
        ax1.text(-1.5, y, f'Fold {fold+1}', va='center', ha='right', fontsize=10)

    ax1.axvline(x=n_points*0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.text(n_points*0.5, n_folds + 0.3, '⚠️ Futuro mezclado con pasado', ha='center', color='red', fontsize=10)
    ax1.set_xlim(-3, n_points + 1)
    ax1.set_ylim(-0.5, n_folds + 0.5)
    ax1.set_xlabel('Tiempo →')
    ax1.set_yticks([])

    # ============ Panel 2: Expanding Window (CORRECTO) ============
    ax2 = axes[1]
    ax2.set_title('Expanding Window (CORRECTO)', fontsize=12, fontweight='bold')

    initial_train = 8
    for fold in range(n_folds):
        y = n_folds - fold - 1
        train_end = initial_train + fold * test_size
        test_start = train_end
        test_end = test_start + test_size
        
        for i in range(n_points):
            if i < train_end:
                color = colors['train']
            elif i < test_end:
                color = colors['test']
            else:
                color = colors['unused']
            ax2.barh(y, 1, left=i, color=color, edgecolor='white', height=0.6)
        ax2.text(-1.5, y, f'Fold {fold+1}', va='center', ha='right', fontsize=10)

    ax2.annotate('', xy=(initial_train + 3*test_size, -0.8), xytext=(0, -0.8),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax2.text(7, -1.2, 'Ventana de train CRECE', ha='center', color='green', fontsize=10)
    ax2.set_xlim(-3, n_points + 1)
    ax2.set_ylim(-1.5, n_folds + 0.5)
    ax2.set_xlabel('Tiempo →')
    ax2.set_yticks([])

    # ============ Panel 3: Sliding Window (CORRECTO) ============
    ax3 = axes[2]
    ax3.set_title('Sliding Window (CORRECTO)', fontsize=12, fontweight='bold')

    window_size = 8
    for fold in range(n_folds):
        y = n_folds - fold - 1
        train_start = fold * test_size
        train_end = train_start + window_size
        test_start = train_end
        test_end = test_start + test_size
        
        for i in range(n_points):
            if train_start <= i < train_end:
                color = colors['train']
            elif test_start <= i < test_end:
                color = colors['test']
            else:
                color = colors['unused']
            ax3.barh(y, 1, left=i, color=color, edgecolor='white', height=0.6)
        ax3.text(-1.5, y, f'Fold {fold+1}', va='center', ha='right', fontsize=10)

    ax3.annotate('', xy=(3*test_size + window_size, -0.8), xytext=(3*test_size, -0.8),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax3.text(10, -1.2, 'Tamaño FIJO se desliza', ha='center', color='purple', fontsize=10)
    ax3.set_xlim(-3, n_points + 1)
    ax3.set_ylim(-1.5, n_folds + 0.5)
    ax3.set_xlabel('Tiempo →')
    ax3.set_yticks([])

    # Leyenda
    train_patch = mpatches.Patch(color=colors['train'], label='Entrenamiento')
    test_patch = mpatches.Patch(color=colors['test'], label='Validación/Test')
    unused_patch = mpatches.Patch(color=colors['unused'], label='No usado')
    fig.legend(handles=[train_patch, test_patch, unused_patch], loc='upper right', ncol=3)

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

    print("=" * 50)
    print("Implementación práctica con sklearn")
    print("=" * 50)
    from sklearn.model_selection import TimeSeriesSplit
    import pandas as pd
    import numpy as np

    # Datos de ejemplo
    n = 100
    dates = pd.date_range('2023-01-01', periods=n, freq='W')
    data = pd.DataFrame({'fecha': dates, 'ventas': np.random.randn(n).cumsum() + 100})

    print("IMPLEMENTACIÓN DE VALIDACIÓN TEMPORAL CON SKLEARN")
    print("=" * 60)

    # TimeSeriesSplit (Expanding Window nativo)
    tscv = TimeSeriesSplit(n_splits=5)

    print("\n1. TimeSeriesSplit (Expanding Window):")
    print("-" * 60)
    for fold, (train_idx, test_idx) in enumerate(tscv.split(data), 1):
        print(f"Fold {fold}: Train [{train_idx[0]:3d}-{train_idx[-1]:3d}] ({len(train_idx):3d} obs) | "
            f"Test [{test_idx[0]:3d}-{test_idx[-1]:3d}] ({len(test_idx):3d} obs)")

    # Sliding Window personalizado
    def sliding_window_split(data, window_size, test_size, n_splits):
        """Generador de índices para Sliding Window CV."""
        n = len(data)
        for i in range(n_splits):
            train_start = i * test_size
            train_end = train_start + window_size
            test_start = train_end
            test_end = test_start + test_size
            
            if test_end > n:
                break
                
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx

    print("\n2. Sliding Window (Ventana Fija de 52 semanas):")
    print("-" * 60)
    for fold, (train_idx, test_idx) in enumerate(sliding_window_split(data, window_size=52, test_size=8, n_splits=5), 1):
        print(f"Fold {fold}: Train [{train_idx[0]:3d}-{train_idx[-1]:3d}] ({len(train_idx):3d} obs) | "
            f"Test [{test_idx[0]:3d}-{test_idx[-1]:3d}] ({len(test_idx):3d} obs)")
    print("=" * 150)
    #================================== Fin Pregunta 2.b ==================================
    print("""
    2.c) Ceros Censurados por Quiebre de Stock
        Pregunta: Al analizar los datos históricos, descubres que 15% de los registros de "ventas = 0" 
        corresponden a días donde el producto estuvo en quiebre de stock (inventario = 0).     
            - Explica cómo este fenómeno sesga sistemáticamente el forecast hacia valores más bajos.
            - Propón al menos dos estrategias técnicas para mitigar este problema (una basada en data augmentation y otra en modelado).
            - ¿Por qué es este un problema ético además de técnico? Relaciona con la pérdida de ventas atribuible al sistema de forecast.
        
        Respuesta:
            ¿Cómo sesga el fenómeno de ceros censurados el forecast?
                El problema de ceros censurados (censored zeros) ocurre cuando:
                    
                    formula:
                            Ventas_observadas= min(Demanda_real, Inventario_disponible)
                            
                    Cuando inventario = 0, observamos ventas = 0, pero:
                        - No significa que la demanda fue cero
                        - La demanda real pudo ser 50, 100 o más unidades
                        - Esta demanda no se observa (está censurada)

                    Mecanismo del sesgo:
                        1. Entrenamiento contaminado:
                            - El modelo "aprende" que ciertos días tienen ventas = 0
                            - Asocia patrones (día de la semana, temporada, etc.) con demanda cero
                            - En realidad, esos patrones corresponden a quiebres, no a baja demanda

                        2. Ciclo de retroalimentación negativa: 
                            
                            Quiebre → Ventas=0 → Modelo predice bajo → Menos reposición → Más quiebres
                            
                        3. Subestimación sistemática:
                            - Con 15% de ceros por quiebre, el modelo subestima la media real
                            - Si demanda real promedio = 100 unidades, con 15% ceros censurados:
                                    
                                    Media observada ~ 100 x 0,85 = 85 unidades                    
                            
                            - Subestimación del 15% solo por este efecto
        

            Estrategias de Mitigación:
                Estrategia 1: Data Augmentation - Imputación de Demanda Censurada
                    
                    Método: Reemplazar los ceros censurados con estimaciones de la demanda real
                            # Identificar días con quiebre
                            df['es_quiebre'] = (df['ventas'] == 0) & (df['inventario'] == 0)

                            # Opción A: Imputar con promedio histórico del mismo día/semana
                            df.loc[df['es_quiebre'], 'ventas_corregidas'] = df.groupby('dia_semana')['ventas'].transform(
                                lambda x: x[x > 0].mean()
                            )  

                            # Opción B: Imputar con modelo de demanda (sin días de quiebre)
                            modelo_demanda = entrenar_modelo(df[~df['es_quiebre']])
                            df.loc[df['es_quiebre'], 'ventas_corregidas'] = modelo_demanda.predict(df[df['es_quiebre']])

                Variantes:
                    - Imputación por percentil (P75 o P90 para ser conservador)
                    - Imputación múltiple (generar varios escenarios)
                    - KNN-imputation usando días similares sin quiebre

            Estrategia 2: Modelado - Modelo Tobit o Zero-Inflated
                    
                    Modelo Tobit (Censored Regression):
                        Modela explícitamente la censura:
                                Yt = Xt beta + epsilon_t (demanda latente)
                                Yt = max(0,min(Yt,inventariot)) (ventas observadas)

                                ```python
                                from statsmodels.regression.linear_model import OLS
                                from statsmodels.discrete.truncreg import TruncatedReg  # O usar Tobit

                                # Indicar cuáles observaciones están censuradas
                                df['censurado'] = df['inventario'] == 0

                                # Modelo que considera la censura
                                # (implementación conceptual - requiere librerías especializadas)
            
                    Modelo Zero-Inflated:
                        Dos componentes:
                            1. Modelo binario: Probabilidad de quiebre vs. no quiebre
                            2. Modelo continuo: Demanda dado que hay stock disponible

                                P(y=0) = P(quiebre) + P(demanda = 0 | no quiebre)
                                P(y>0) = P(no quiebre) * P(demanda > 0 | no quiebre)
            
            Estrategia 3 (Bonus): Feature Engineering con Variable de Censura
                            ```python
                            # Agregar indicador de días de quiebre reciente
                            df['quiebre_ayer'] = (df['inventario'].shift(1) == 0).astype(int)
                            df['dias_desde_quiebre'] = (df['inventario'] > 0).groupby(
                                (df['inventario'] == 0).cumsum()
                            ).cumcount()

                            # El modelo puede aprender a "compensar" los efectos de quiebres
        
        Dimensión Ética del Problema:
            ¿Por qué es un problema ético además de técnico?

                1. **Ciclo de exclusión de productos menos rentables:**
                    - Productos con poco margen → menor reposición → más quiebres → parecen tener baja demanda → **se descatalogan**
                    - Pero la demanda real existía; solo estaba censurada

                2. **Pérdida de ventas atribuible al sistema:**
                    - Si el forecast subestima por ceros censurados → menos inventario → más quiebres → profecía autocumplida
                    - **El sistema de forecast causa las ventas perdidas que luego "predice"**

                3. **Responsabilidad del equipo de Data Science:**
                    - Ignorar el problema de censura es **negligencia técnica**
                    - El equipo tiene la responsabilidad de identificar y comunicar esta limitación
                    - Reportar solo WAPE/MAE sin mencionar el sesgo por censura es **engañoso**

                4. Impacto en stakeholders:
                    - **Clientes:** No encuentran productos que quieren comprar
                    - **Proveedores:** Reciben menos pedidos de los que deberían
                    - **Tiendas:** Pierden ventas y clientes
                    - **Empleados:** Metas de ventas inalcanzables por falta de stock
                
            Cuantificación del impacto ético:
                Si 15% de días tienen quiebre y la demanda promedio es 100 unidades:
                    - Demanda real: 100 unidades/día
                    - Forecast sesgado: 85 unidades/día
                    - Pérdida diaria: 15 unidades × $3.00 = $45/día por SKU
                    - Con 500 SKUs: $45 × 500 = $22,500/día = $8.2M/año
            Esta pérdida es ATRIBUIBLE AL SISTEMA de forecast que no corrige la censura.                  
    """)
    print("=" * 50)
    print("Demostración del efecto de ceros censurados")
    print("=" * 50)
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error

    np.random.seed(42)

    # Simular demanda real y ventas censuradas
    n_dias = 365
    demanda_real = 100 + 20 * np.sin(np.arange(n_dias) * 2 * np.pi / 7) + np.random.randn(n_dias) * 15
    demanda_real = np.maximum(demanda_real, 0)  # No negativa

    # Simular inventario (a veces hay quiebre)
    inventario = np.random.choice([0, 1000], size=n_dias, p=[0.15, 0.85])  # 15% quiebres

    # Ventas observadas = min(demanda, inventario)
    ventas_observadas = np.minimum(demanda_real, inventario)

    # Crear DataFrame
    df_censura = pd.DataFrame({
        'dia': np.arange(n_dias),
        'dia_semana': np.arange(n_dias) % 7,
        'demanda_real': demanda_real,
        'inventario': inventario,
        'ventas_observadas': ventas_observadas,
        'es_quiebre': inventario == 0
    })

    # Modelo 1: Sin corrección (usa ventas observadas)
    X = df_censura[['dia', 'dia_semana']].values
    model_sin_correccion = LinearRegression()
    model_sin_correccion.fit(X, df_censura['ventas_observadas'])
    pred_sin_correccion = model_sin_correccion.predict(X)

    # Modelo 2: Con corrección (excluye días de quiebre del entrenamiento)
    df_sin_quiebre = df_censura[~df_censura['es_quiebre']]
    model_con_correccion = LinearRegression()
    model_con_correccion.fit(df_sin_quiebre[['dia', 'dia_semana']].values, df_sin_quiebre['ventas_observadas'])
    pred_con_correccion = model_con_correccion.predict(X)

    # Visualización
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Panel 1: Comparación de series
    ax1 = axes[0]
    dias_quiebre = df_censura[df_censura['es_quiebre']]['dia']
    ax1.plot(df_censura['dia'], df_censura['demanda_real'], 'b-', label='Demanda Real', alpha=0.7)
    ax1.plot(df_censura['dia'], df_censura['ventas_observadas'], 'g-', label='Ventas Observadas', alpha=0.5)
    ax1.scatter(dias_quiebre, [0]*len(dias_quiebre), color='red', s=20, label='Quiebres (ceros censurados)', zorder=5)
    ax1.set_xlabel('Día')
    ax1.set_ylabel('Unidades')
    ax1.set_title('Demanda Real vs. Ventas Observadas (con Quiebres)', fontsize=12)
    ax1.legend()
    ax1.set_xlim(0, 100)  # Mostrar solo primeros 100 días para claridad

    # Panel 2: Comparación de forecasts
    ax2 = axes[1]
    ax2.plot(df_censura['dia'], df_censura['demanda_real'], 'b-', label='Demanda Real', alpha=0.5)
    ax2.plot(df_censura['dia'], pred_sin_correccion, 'r--', label='Forecast SIN corrección', linewidth=2)
    ax2.plot(df_censura['dia'], pred_con_correccion, 'g-', label='Forecast CON corrección', linewidth=2)
    ax2.set_xlabel('Día')
    ax2.set_ylabel('Unidades')
    ax2.set_title('Impacto de la Corrección por Ceros Censurados', fontsize=12)
    ax2.legend()
    ax2.set_xlim(0, 100)

    plt.tight_layout()
    plt.show()

    # Métricas
    print("\n" + "=" * 60)
    print("COMPARACIÓN DE ERRORES (vs. Demanda Real)")
    print("=" * 60)
    print(f"\nModelo SIN corrección por censura:")
    print(f"  MAE: {mean_absolute_error(df_censura['demanda_real'], pred_sin_correccion):.2f} unidades")
    print(f"  Bias: {(pred_sin_correccion.mean() - df_censura['demanda_real'].mean()):.2f} unidades")

    print(f"\nModelo CON corrección (excluyendo quiebres):")
    print(f"  MAE: {mean_absolute_error(df_censura['demanda_real'], pred_con_correccion):.2f} unidades")
    print(f"  Bias: {(pred_con_correccion.mean() - df_censura['demanda_real'].mean()):.2f} unidades")

    print(f"\n📊 Estadísticas:")
    print(f"  Demanda real promedio: {df_censura['demanda_real'].mean():.2f}")
    print(f"  Ventas observadas promedio: {df_censura['ventas_observadas'].mean():.2f}")
    print(f"  Subestimación por censura: {(1 - df_censura['ventas_observadas'].mean()/df_censura['demanda_real'].mean())*100:.1f}%")

    print("=" * 50)
    print("Estrategia de imputación para ceros censurados")
    print("=" * 50)
    import pandas as pd
    import numpy as np

    def imputar_ceros_censurados(df, columna_ventas, columna_inventario, metodo='media_dia'):
        """
        Imputa valores para ceros censurados (quiebres de stock).
        
        Parámetros:
        -----------
        df : DataFrame
            DataFrame con datos de ventas e inventario
        columna_ventas : str
            Nombre de la columna de ventas
        columna_inventario : str
            Nombre de la columna de inventario
        metodo : str
            'media_dia': Imputar con el promedio del mismo día de la semana
            'percentil_75': Imputar con el percentil 75 del mismo día
            'interpolacion': Interpolación lineal
        
        Retorna:
        --------
        Series con ventas corregidas
        """
        df = df.copy()
        
        # Identificar ceros censurados
        mask_quiebre = (df[columna_ventas] == 0) & (df[columna_inventario] == 0)
        
        # Datos sin quiebres para calcular estadísticas
        df_ok = df[~mask_quiebre]
        
        if metodo == 'media_dia':
            # Calcular media por día de la semana (solo de días sin quiebre)
            media_por_dia = df_ok.groupby('dia_semana')[columna_ventas].mean()
            df['ventas_corregidas'] = df[columna_ventas].copy()
            for dia in range(7):
                mask = mask_quiebre & (df['dia_semana'] == dia)
                df.loc[mask, 'ventas_corregidas'] = media_por_dia[dia]
        
        elif metodo == 'percentil_75':
            # Usar percentil 75 (más conservador)
            p75_por_dia = df_ok.groupby('dia_semana')[columna_ventas].quantile(0.75)
            df['ventas_corregidas'] = df[columna_ventas].copy()
            for dia in range(7):
                mask = mask_quiebre & (df['dia_semana'] == dia)
                df.loc[mask, 'ventas_corregidas'] = p75_por_dia[dia]
        
        elif metodo == 'interpolacion':
            # Reemplazar ceros con NaN e interpolar
            df['ventas_corregidas'] = df[columna_ventas].copy()
            df.loc[mask_quiebre, 'ventas_corregidas'] = np.nan
            df['ventas_corregidas'] = df['ventas_corregidas'].interpolate(method='linear')
        
        return df['ventas_corregidas']

    # Aplicar imputación
    df_censura['ventas_imputadas_media'] = imputar_ceros_censurados(
        df_censura, 'ventas_observadas', 'inventario', metodo='media_dia'
    )
    df_censura['ventas_imputadas_p75'] = imputar_ceros_censurados(
        df_censura, 'ventas_observadas', 'inventario', metodo='percentil_75'
    )

    # Comparar resultados
    print("COMPARACIÓN DE ESTRATEGIAS DE IMPUTACIÓN")
    print("=" * 60)
    print(f"\nDemanda real promedio: {df_censura['demanda_real'].mean():.2f}")
    print(f"Ventas observadas promedio: {df_censura['ventas_observadas'].mean():.2f}")
    print(f"Ventas imputadas (media): {df_censura['ventas_imputadas_media'].mean():.2f}")
    print(f"Ventas imputadas (P75): {df_censura['ventas_imputadas_p75'].mean():.2f}")

    print(f"\n📊 Error vs. Demanda Real:")
    print(f"  Sin corrección: {abs(df_censura['ventas_observadas'].mean() - df_censura['demanda_real'].mean()):.2f} unidades de sesgo")
    print(f"  Con media: {abs(df_censura['ventas_imputadas_media'].mean() - df_censura['demanda_real'].mean()):.2f} unidades de sesgo")
    print(f"  Con P75: {abs(df_censura['ventas_imputadas_p75'].mean() - df_censura['demanda_real'].mean()):.2f} unidades de sesgo")
    print("=" * 150)
    #================================== Fin Pregunta 2.c ==================================
    print("""
    Resumen de Pregunta 2
    ------------------------------------------------------------------------------------------------------------------------------------
    | Concepto             | Problema                                      | Solución Clave                                            |
    |----------------------|-----------------------------------------------|-----------------------------------------------------------|
    | Bias vs. Accuracy    | Mejor WAPE ≠ menor costo operacional          | Considerar asimetría de costos (quiebre vs. sobre-stock)  |
    | Validación Temporal  | K-Fold filtra información futura              | Usar Expanding/Sliding Window                             |
    | Ceros Censurados     | Quiebres contaminan histórico                 | Imputar demanda o modelado Tobit                          |
    | Ética en Forecasting | El sistema puede causar pérdidas              | Transparencia + corrección del sesgo                      |
    ------------------------------------------------------------------------------------------------------------------------------------

    Fórmulas Clave:++
                        Costo por Bias = |Bias| * Volumen * COSTOunitario

                        WAPE = E|Yt - Ŷt|
                            ----------
                                E|Yt|
                        
                        Bias = E(Ŷt . Yt)
                            ----------
                                E(Yt)
    """)
if __name__ == "__main__":
    run()
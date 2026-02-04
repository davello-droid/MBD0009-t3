import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
def run() -> None:
    # Configuración de visualización
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    warnings.filterwarnings('ignore')

    print("="*80)
    print("INICIO: Pregunta 05 - Análisis de Elasticidad y Estrategia de Precios")
    print("="*80)

    # -------------------------------------------------------------------------
    # a) Preparación de datos para análisis de precios
    # -------------------------------------------------------------------------
    print("\n[a] Preparación de datos...")
    
    # Cargar dataset
    try:
        df = pd.read_csv('data/online_retail_II.csv')
    except FileNotFoundError:
        # Intento alternativo si el archivo está en unicode/latin1 a veces pasa con este dataset
        try:
            df = pd.read_csv('data/online_retail_II.csv', encoding='ISO-8859-1')
        except FileNotFoundError:
            print("ERROR: No se encontró el archivo 'data/online_retail_II.csv'. Asegúrate de que existe.")
            return

    # 1. Filtra solo transacciones de UK
    df_uk = df[df['Country'] == 'United Kingdom'].copy()
    
    # 2. Elimina devoluciones (Quantity < 0 o Invoice empieza con 'C')
    # Convertir Invoice a string para asegurar que startswith funcione
    df_uk['Invoice'] = df_uk['Invoice'].astype(str)
    df_uk = df_uk[~df_uk['Invoice'].str.startswith('C')]
    df_uk = df_uk[df_uk['Quantity'] > 0]
    
    # 3. Elimina registros con Price <= 0 o Customer ID nulo
    df_uk = df_uk[(df_uk['Price'] > 0) & (df_uk['Customer ID'].notnull())]
    
    # Preprocesamiento de fechas
    df_uk['InvoiceDate'] = pd.to_datetime(df_uk['InvoiceDate'])
    # Crear variable de semana (usando ISO calendar week)
    df_uk['Year'] = df_uk['InvoiceDate'].dt.isocalendar().year
    df_uk['Week'] = df_uk['InvoiceDate'].dt.isocalendar().week
    # Crear un índice de tiempo continuo para ordenar correctamente (Year-Week)
    df_uk['TimeIndex'] = df_uk['Year'] * 100 + df_uk['Week']
    
    # Month para el modelo parte C
    df_uk['Month'] = df_uk['InvoiceDate'].dt.month

    # 4. Selecciona los 10 productos más vendidos
    top_products_series = df_uk.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False).head(10)
    top_stock_codes = top_products_series.index.tolist()
    
    print(f"\nTop 10 Productos seleccionados (StockCode): {top_stock_codes}")
    
    # Filtrar dataset solo para estos productos
    df_top = df_uk[df_uk['StockCode'].isin(top_stock_codes)].copy()
    
    # 5. Agrega los datos a nivel producto-semana
    # Se calcula precio promedio ponderado o simple? El enunciado dice "precio_promedio".
    # Usaremos el promedio simple de los precios en esa semana para ese producto, 
    # o mejor: Total Revenue / Total Quantity para obtener el precio efectivo promedio ponderado.
    # Dado que es un análisis de elasticidad, el precio unitario promedio es lo estándar.
    
    df_agg = df_top.groupby(['StockCode', 'Description', 'Year', 'Week', 'TimeIndex', 'Month']).agg(
        cantidad_total=('Quantity', 'sum'),
        precio_promedio=('Price', 'mean'), # Precio promedio observado
        n_transacciones=('Invoice', 'nunique')
    ).reset_index()
    
    # Ordenar
    df_agg = df_agg.sort_values(['StockCode', 'TimeIndex'])
    
    # ENTREGABLE A: Tabla resumen
    print("\n--- Entregable (a): Resumen de Ventas Top 10 Productos ---")
    resumen_a = df_agg.groupby(['StockCode', 'Description'])['cantidad_total'].sum().sort_values(ascending=False).reset_index()
    print(resumen_a)
    
    # -------------------------------------------------------------------------
    # b) Estimación de elasticidad naive (Log-Log)
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("[b] Estimación de elasticidad naive (Modelo Log-Log)")
    print("-" * 80)
    
    results_list = []
    
    for stock_code in top_stock_codes:
        sub_df = df_agg[df_agg['StockCode'] == stock_code].copy()
        
        # Transformación logarítmica
        sub_df['ln_Q'] = np.log(sub_df['cantidad_total'])
        sub_df['ln_P'] = np.log(sub_df['precio_promedio'])
        
        # Modelo OLS: ln(Q) = alpha + beta * ln(P)
        mod = smf.ols("ln_Q ~ ln_P", data=sub_df)
        res = mod.fit()
        
        # Guardar descripción (usamos la primera que aparezca pues es constante por stockcode usualmente)
        desc = sub_df['Description'].iloc[0]
        
        results_list.append({
            'StockCode': stock_code,
            'Description': desc,
            'Beta (Elasticidad)': res.params['ln_P'],
            'P-value': res.pvalues['ln_P'],
            'R-squared': res.rsquared
        })
        
    results_b = pd.DataFrame(results_list)
    
    # ENTREGABLE B: Tabla de resultados
    print("\n--- Entregable (b): Resultados de Elasticidad Naive ---")
    print(results_b[['StockCode', 'Description', 'Beta (Elasticidad)', 'P-value', 'R-squared']])
    
    # Análisis de resultados
    significativos = results_b[results_b['P-value'] < 0.05]
    elasticos = significativos[abs(significativos['Beta (Elasticidad)']) > 1]
    inelasticos = significativos[abs(significativos['Beta (Elasticidad)']) <= 1]
    
    print(f"\nANÁLISIS (b):")
    print(f"Productos con elasticidad significativa (p < 0.05): {len(significativos)} de 10")
    print(f"De los significativos: {len(elasticos)} son elásticos (|Beta| > 1) y {len(inelasticos)} son inelásticos.")

    # -------------------------------------------------------------------------
    # c) Modelo con efectos temporales (Pooled OLS con efectos fijos)
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("[c] Modelo Pooled con Efectos Temporales")
    print("-" * 80)
    
    # Preparar datos para pooled regression
    df_pooled = df_agg.copy()
    df_pooled['ln_Q'] = np.log(df_pooled['cantidad_total'])
    df_pooled['ln_P'] = np.log(df_pooled['precio_promedio'])
    
    # Modelo: ln(Q_it) = Beta * ln(P_it) + Gamma_i (StockCode) + Delta_t (Month)
    # Usamos C() para indicar variables categóricas
    formula_pooled = "ln_Q ~ ln_P + C(StockCode) + C(Month)"
    
    mod_pooled = smf.ols(formula_pooled, data=df_pooled)
    res_pooled = mod_pooled.fit(cov_type='HC1') # Errores estándar robustos
    
    beta_pooled = res_pooled.params['ln_P']
    promedio_beta_individual = results_b['Beta (Elasticidad)'].mean()
    
    print(f"\nResultados Modelo Pooled con Efectos Fijos (Producto y Mes):")
    print(f"Beta Estimado (Elasticidad Pooled): {beta_pooled:.4f}")
    print(f"Promedio de Betas Individuales (Naive): {promedio_beta_individual:.4f}")
    print(res_pooled.summary().tables[1]) # Mostrar tabla parcial de coeficientes si es muy larga
    
    print("\nANÁLISIS (c):")
    print("Comparación: El beta pooled representa una elasticidad promedio controlando por")
    print("diferencias inherentes entre productos y estacionalidad mensual.")
    print("Si |Beta Pooled| > |Promedio Naive|, sugiere que al controlar por estacionalidad (ej: alta demanda en Navidad")
    print("independiente del precio), la sensibilidad al precio se aísla mejor.")
    print("La estacionalidad positiva (alta demanda) suele correlacionarse con precios altos o constantes,")
    print("lo que podría sesgar la estimación naive hacia cero (menos elástica) si no se controla.")

    # -------------------------------------------------------------------------
    # d) Recomendación de pricing
    # -------------------------------------------------------------------------
    print("\n" + "-"*80)
    print("[d] Recomendación de Pricing y Análisis Crítico")
    print("-" * 80)
    
    # 1. Margen óptimo (Regla de Lerner): (P - C) / P = -1 / Beta
    # Margen % Óptimo = -1 / Beta
    # Asumimos que usamos la elasticidad pooled (beta_pooled) estimad en (c)
    
    elasticidad_c = beta_pooled
    
    # Cuidado: Si la demanda no es elástica (|e| < 1), el margen óptimo matemático es > 100% o indefinido 
    # para maximización de beneficios simple (monopolio), lo que implica subir precios indefinidamente hasta que sea elástica.
    
    print(f"Elasticidad estimada (Modelo c): {elasticidad_c:.4f}")
    
    margen_optimo = np.nan
    recomendacion = ""
    
    if elasticidad_c > -1 and elasticidad_c < 0:
        # Inelástica (-0.5 por ejemplo) -> |e| < 1. 
        # Lerner: Margen = 1 / |e| > 1 (mayor al 100%? No tiene sentido físico directo sin restricciones)
        # Interpretación económica: Si es inelástica, conviene subir precios para aumentar ingresos y margen.
        margen_optimo_teorico = -1 / elasticidad_c
        recomendacion = "La demanda es INELÁSTICA (|e| < 1). La empresa debería SUBIR precios."
        margen_msg = f"{margen_optimo_teorico:.2%} (Teórico > 100%, implica subir precios hasta zona elástica)"
    elif elasticidad_c <= -1:
        # Elástica (-1.5 por ejemplo) -> |e| > 1
        margen_optimo = -1 / elasticidad_c
        recomendacion = f"Comparando con margen actual (40%): "
        if margen_optimo > 0.40:
             recomendacion += "El margen óptimo es MAYOR. Debería SUBIR precios."
        else:
             recomendacion += "El margen óptimo es MENOR. Debería BAJAR precios."
        margen_msg = f"{margen_optimo:.2%}"
    else:
        # Elasticidad positiva (Bien Giffen o error de datos/modelo)
        margen_msg = "N/A (Elasticidad positiva)"
        recomendacion = "Revisar modelo, elasticidad positiva no permite cálculo estándar de Lerner."

    print(f"Margen Actual: 40.00%")
    print(f"Margen Óptimo Sugerido (Lerner): {margen_msg}")
    print(f"Sugerencia: {recomendacion}")
    
    print("\nANÁLISIS CRÍTICO (Limitaciones):")
    print("-" * 60)
    analisis_texto = """
    1. Endogeneidad: El precio no es exógeno. Es probable que la empresa fije precios reaccionando a la 
       demanda esperada, o que precios y demanda se muevan juntos por factores omitidos (calidad, marketing). 
       Esto sesga el estimador de elasticidad (simultaneidad).
    2. Falta de variación experimental: Sin A/B tests o cambios de costos exógenos, estamos capturando 
       correlaciones de equilibrio de mercado, no la curva de demanda pura.
    3. Estrategia Robusta Propuesta: Implementar un diseño experimental (A/B testing) aleatorizando precios 
       para productos similares o en diferentes regiones geográficas para romper la endogeneidad y aislar 
       el efecto causal del precio en la cantidad. Instrumentalizar precios con costos de insumos también sería útil.
    """
    print(analisis_texto)
    print("="*80)

if __name__ == "__main__":
    main()
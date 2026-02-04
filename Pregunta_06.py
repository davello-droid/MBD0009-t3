# -*- coding: utf-8 -*-
"""T3_P6

#Tarea 3 - Pregunta 06

- Daniel Avello
- Reberto Sepúlveda
- Felipe Valdivia
"""
def run() -> None:
    # a)

    #a.1) Calcular margen de contribución actual (%)
    precio = 3500
    cv = 2100
    mu = round(precio - cv, 0)
    mp = round(mu / precio, 4)

    print("#"*80)
    print("\n")
    print("a.1) Calcular margen de contribución actual (%)")
    print("\n")
    print("Datos: ")
    print(f"Precio = ${precio}")
    print(f"Costo variable = ${cv}")
    print("\n")
    print(f"Margen unitario = precio - costo variable = ${mu}")
    print(f"Margen (%) = Margen_unitario / Precio = {mp} %")
    print("\n")
    print(f"Por lo tanto, el margen de contribución actual es del {mp*100} %.")
    print("\n")
    print("#"*80)

    #a.2) Calcular contribución diaria total

    viaje_dia_ejecutivo = 12500
    viaje_dia_casual = 17500
    viaje_dia_nocturno = 10000
    viaje_dia_aeropuerto = 10000
    total_viajes_dia = viaje_dia_ejecutivo + viaje_dia_casual + viaje_dia_nocturno + viaje_dia_aeropuerto
    cd = round(total_viajes_dia * mu, 0)

    print("#"*80)
    print("\n")
    print("a.2) Calcular contribución diaria total")
    print("\n")
    print("Datos: ")
    print(f"Cantidad de viajes ejecutivos por día = {viaje_dia_ejecutivo}")
    print(f"Cantidad de viajes casuales por día = {viaje_dia_casual}")
    print(f"Cantidad de viajes nocturnos por día = {viaje_dia_nocturno}")
    print(f"Cantidad de viajes a aeropuertos por día = {viaje_dia_aeropuerto}")
    print(f"Total de viajes por día = sum(viaje_i) = {total_viajes_dia}")
    print(f"Margen unitario promedio por viaje = ${mu}")
    print("\n")
    print(f"Contribución diaria promedio = Total_viajes_dia * Margen = {cd}")
    print("\n")
    print(f"Por lo tanto, la contribución diaria total es de ${cd}.")
    print("\n")
    print("#"*80)

    #a.3) Calcular margen óptimo por segmento

    b_ejecutivo = -0.6
    b_casual = -1.8
    b_nocturno = -1.2
    b_aeropuerto = -0.4
    mg_op_ejecutivo = round(-1/b_ejecutivo, 4)
    mg_op_casual = round(-1/b_casual, 4)
    mg_op_nocturno = round(-1/b_nocturno, 4)
    mg_op_aeropuerto = round(-1/b_aeropuerto, 4)

    print("#"*80)
    print("\n")
    print("a.3) Calcular margen óptimo por segmento")
    print("\n")
    print("Datos: ")
    print(f"Elasticidad ejecutivo = {b_ejecutivo}")
    print(f"Elasticidad casual = {b_casual}")
    print(f"Elasticidad nocturno = {b_nocturno}")
    print(f"Elasticidad aeropuerto = {b_aeropuerto}")
    print("\n")
    print(f"Margen óptimo = -1/beta")
    print("\n")
    print(f"Margen óptimo viaje ejecutivo = {mg_op_ejecutivo*100:.2f} %")
    print(f"Margen óptimo viaje casual = {mg_op_casual*100:.2f} %")
    print(f"Margen óptimo viaje nocturno = {mg_op_nocturno*100:.2f} %")
    print(f"Margen óptimo viaje aeropuerto = {mg_op_aeropuerto*100:.2f} %")
    print("\n")
    print(f"Por lo tanto, como todos los segmentos tienen un margen óptimo superior al margen total actual ({mp*100}%), se concluye que todos los segmentos están siendo subexplotados.")
    print("\n")
    print("#"*80)

    #b)

    #1.b) Calcular precio óptimo por segmento
    Q_perdida = 0.3
    p_op_ejecutivo = round(precio * (1+(Q_perdida/(-b_ejecutivo))), 4)
    p_op_casual = round(cv * (abs(b_casual)/(abs(b_casual)-1)), 4)
    p_op_nocturno = round(cv * (abs(b_nocturno)/(abs(b_nocturno)-1)), 4)
    p_op_aeropuerto = round(precio * (1+(Q_perdida/(-b_aeropuerto))), 4)

    print("#"*80)
    print("\n")
    print("b.1) Calcular precio óptimo por segmento")
    print("\n")
    print("Datos: ")
    print(f"% disminución viajes inelásticos = {Q_perdida*100}%")
    print(f"Costo unitario = ${cv}")
    print(f"Elasticidad ejecutivo = {b_ejecutivo}")
    print(f"Elasticidad casual = {b_casual}")
    print(f"Elasticidad nocturno = {b_nocturno}")
    print(f"Elasticidad aeropuerto = {b_aeropuerto}")
    print("\n")
    print(f"Precio óptimo = Margen_unitario * (|beta|/(|beta|-1))")
    print(f"Precio óptimo viajes inelásticos = beta = delta(# viajes) / delta(precio)")
    print("\n")
    print(f"Precio óptimo ejecutivo = ${p_op_ejecutivo:.0f}*")
    print(f"Precio óptimo casual = ${p_op_casual:.0f}")
    print(f"Precio óptimo nocturno = ${p_op_nocturno:.0f}")
    print(f"Precio óptimo aeropuerto = ${p_op_aeropuerto:.0f}*")
    print("\n")
    print(f"Los precios óptimos son ${p_op_casual:.0f} para viaje casual, y ${p_op_nocturno:.0f} para viajes nocturnos.")
    print(f"Dado que en los segmentos 'ejecutivo' y 'aeropuerto' las elasticidades varían entre (0,1), entonces la demanda es inelástica.")
    print(f"Así, no es posible aplicar la fórmula de Lerner para estos casos. La empresa podría aumentar el precio de los viajes infinitamente y la cantidad de viajes no variarían considerablemente.")
    print(f"La estimación de estos precios se realiza asumiendo un {Q_perdida*100}% de disminución en viajes. Luego, los precios nuevos son ${p_op_ejecutivo:.0f} para viajes ejecutivos, y ${p_op_aeropuerto:.0f} para viajes al aeropuerto.")
    print("\n")
    print("#"*80)

    #2.b) Expresar cada precio como multiplicador del precio original
    mult_ejecutivo = round(p_op_ejecutivo / precio, 4)
    mult_casual = round(p_op_casual / precio, 4)
    mult_nocturno = round(p_op_nocturno / precio, 4)
    mult_aeropuerto = round(p_op_aeropuerto / precio, 4)

    print("#"*80)
    print("\n")
    print("b.2) Expresar cada precio como multiplicador del precio original")
    print("\n")
    print("Datos: ")
    print(f"Precio original = ${precio}")
    print(f"Precio óptimo ejecutivo = ${p_op_ejecutivo:.0f}")
    print(f"Precio óptimo casual = ${p_op_casual:.0f}")
    print(f"Precio óptimo nocturno = ${p_op_nocturno:.0f}")
    print(f"Precio óptimo aeropuerto = ${p_op_aeropuerto:.0f}")
    print("\n")
    print(f"Multiplicador del precio = Precio_optimo / Precio_original")
    print("\n")
    print(f"Multiplicador ejecutivo = {mult_ejecutivo:.2f}")
    print(f"Multiplicador casual = {mult_casual:.2f}")
    print(f"Multiplicador nocturno = {mult_nocturno:.2f}")
    print(f"Multiplicador aeropuerto = {mult_aeropuerto:.2f}")
    print("\n")
    print("#"*80)

    #3.b) Nueva contribución total diaria
    nuevo_viaje_dia_ejecutivo = round(viaje_dia_ejecutivo*(1-Q_perdida), 0)
    nuevo_viaje_dia_casual = max(0, round(viaje_dia_casual * (1+(b_casual*((p_op_casual-precio)/precio))), 0))
    nuevo_viaje_dia_nocturno = max(0, round(viaje_dia_nocturno * (1+(b_nocturno*((p_op_nocturno-precio)/precio))), 0))
    nuevo_viaje_dia_aeropuerto = round(viaje_dia_aeropuerto*(1-Q_perdida), 0)

    nuevo_mu_ejecutivo = p_op_ejecutivo - cv
    nuevo_mu_casual = p_op_casual - cv
    nuevo_mu_nocturno = p_op_nocturno - cv
    nuevo_mu_aeropuerto = p_op_aeropuerto - cv

    contr_ejecutivo = round(nuevo_viaje_dia_ejecutivo * nuevo_mu_ejecutivo, 0)
    contr_casual = round(nuevo_viaje_dia_casual * nuevo_mu_casual, 0)
    contr_nocturno = round(nuevo_viaje_dia_nocturno * nuevo_mu_nocturno, 0)
    contr_aeropuerto = round(nuevo_viaje_dia_aeropuerto * nuevo_mu_aeropuerto, 0)

    cd_corto_plazo = round(viaje_dia_ejecutivo * nuevo_mu_ejecutivo, 0) + round(viaje_dia_casual * nuevo_mu_casual, 0) + round(viaje_dia_nocturno * nuevo_mu_nocturno, 0) + round(viaje_dia_aeropuerto * nuevo_mu_aeropuerto, 0)
    nuevo_cd = contr_ejecutivo + contr_casual + contr_nocturno + contr_aeropuerto

    print("#"*80)
    print("\n")
    print("b.3) Nueva contribución diaria")
    print("\n")
    print("Datos: ")
    print(f"Costo unitario = ${cv}")
    print(f"Nuevo precio viaje ejecutivo = ${p_op_ejecutivo:.0f}")
    print(f"Nuevo precio viaje casual = ${p_op_casual:.0f}")
    print(f"Nuevo precio viaje nocturno = ${p_op_nocturno:.0f}")
    print(f"Nuevo precio viaje aeropuerto = ${p_op_aeropuerto:.0f}")
    print("\n")
    print(f"Nuevo margen ejecutivo = ${nuevo_mu_ejecutivo:.0f}")
    print(f"Nuevo margen casual = ${nuevo_mu_casual:.0f}")
    print(f"Nuevo margen nocturno = ${nuevo_mu_nocturno:.0f}")
    print(f"Nuevo margen aeropuerto = ${nuevo_mu_aeropuerto:.0f}")
    print("\n")
    print(f"Nueva cantidad viajes ejecutivos = {nuevo_viaje_dia_ejecutivo:.0f}")
    print(f"Nueva cantidad viajes casuales = {nuevo_viaje_dia_casual:.0f}")
    print(f"Nueva cantidad viajes nocturnos = {nuevo_viaje_dia_nocturno:.0f}")
    print(f"Nueva cantidad viajes aeropuerto = {nuevo_viaje_dia_aeropuerto:.0f}")
    print("\n")
    print(f"Nueva contribución total en el corto plazo = ${cd_corto_plazo:.0f}")
    print(f"Nueva contribución total en el largo plazo = ${nuevo_cd:.0f}")
    print("\n")
    print("#"*80)

    #4.b) Incremento porcentual en contribución

    print("#"*80)
    print("\n")
    print("b.4) Incremento porcentual en contribución")
    print("\n")
    print("Datos: ")
    print(f"Contribución original = {cd}")
    print(f"Nueva contribución diaria = {nuevo_cd:.0f}")
    print("\n")
    print(f"Incremento porcentual = {(nuevo_cd - cd)/cd * 100:.2f}%")
    print("\n")
    print("#"*80)

    #c)

    #1.c) Estrategia de comunicación

    print("#"*80)
    print("\n")
    print("c.1) Estrategia de comunicación")
    print("\n")
    print("Para evitar una sensación de rechazo generalizada con los usuarios, se proponen las siguientes medidas:")
    print("- Mostrar mensaje de aumento de precio por alta demanda de conductores en la zona, con un multiplicador de, por ejemplo, 1.x veces más por porcentaje de demanda adicional.")
    print("- Comentar el hecho de que poner a disposición mas conductores en la zona implica mayor costo pero reduce las tasas de espera.")
    print("- Proponer una medida tal que una parte del aumento del viaje es una mejora en la remuneración del conductor.")
    print("- Proponer un límite máximo a los aumentos de precio.")
    print("- Proponer opciones al usuario, como ofrecer la opción de pagar mas por esperar menos, o bien desplazarse a sectores de menor demanda para disminuir el precio.")
    print("- Ofrecer descuentos y cupones para ciertos viajes, como los de hora punta.")
    print("\n")
    print("#"*80)

    #2.c) Multiplicador máximo del surge

    print("#"*80)
    print("\n")
    print("c.2) Multiplicador máximo del surge")
    print("\n")
    print("Se propone un multiplicador máximo de surge pricing de entre 1.5x - 2.0x. Esto se fundamenta en lo siguiente:")
    print("- Económicamente, los segmentos mas inelásticos (ejecutivos, aeropuerto) no perderán demanda de manera tán drástica al aumentar los precios.")
    print("- En los segmentos más elásticos (casual, nocturno), se puede argumentar por el tramo horario y seguridad de las personas.")
    print("- Parte de las ganacias en las tarifas aumentadas deben ir como remuneración al conductor, entendiendo que es justo para él.")
    print("- Hay que considerar que aumentar precios se entenderá como abuso para los usuarios.")
    print("- Aumentar el precio a más del doble se entenderá como un extremo poco ético por parte de la empresa.")
    print("- Argumentar los aumentos por demanda, horario y distinciones en el viaje (respecto a el precio base) mejora la relación con los usuarios, dándoles la opción de elegir si desean pagar más por cambiar algunas características de un viaje estándar.")
    print("\n")
    print("De lo anterior, es correcto decir que se puede argumentar aumentos de precios de entre 1.5x - 2.0x dependiendo del caso, como mejorar tiempos de espera, evitar rechazo emocional al tener un límite claro e implementar el concepto de justicia y seguridad para el conductor.")
    print("\n")
    print("#"*80)

    #3.c) Evitar el surge

    print("#"*80)
    print("\n")
    print("c.3) Evitar el surge pricing")
    print("\n")
    print("Si los usuarios aprenden a evitar el surge pricing, por ejemplo, esperando más por un viaje de menos costo, entonces la elasticidad sube. Esto demuestra la sensibilidad al precio.")
    print("Adicionalmente, los usuarios podrían optar por no viajar o cambiar de aplicación (ir a la competencia).")
    print("Al aumentar la elasticidad, implica que el precio óptimo del surge pricing baja, esto por la regla de lerner. Aumentar 'epsilon' impplica que disminuye '1/epsilon'.")
    print("Por lo tanto, lo que debería hacerse en estos casos es focalizar el pricing a segmentos puntuales y hacerlos mas moderados.")
    print("\n")
    print("#"*80)

    #d)

    #1.d) Discriminación de precios

    print("#"*80)
    print("\n")
    print("d.1) Discriminación de precios")
    print("\n")
    print("En la teoría económica, cada consumidor tiene una disposición a pagar un monto por un viaje. Este monto puede variar para cada persona / usuario.")
    print("Si se cobra un precio estándar a todos por un viaje, hay usuarios a los que no se les cobra lo que están dispuestos a pagar (pérdida) y hay otros a los que se les cobra más de lo que están dispuestos (no realizan viaje).")
    print("Entonces, lo que busca hacer el modelo de ML es precisamente ajustar el precio de cada viaje a un modelo de oferta y demanda que represente la realidad de cada usuario.")
    print("Esto maximizaría la utilidad total, asumiendo toda la información de los clientes (o la mayor parte), pagando lo que estén dispuestos por un viaje. La restricción en este caso debe ser que cada viaje debe suplir al menos el costo de cada uno, para obtener ganancia. En caso contrario, no conviene realizarlo.")
    print("En resumen, esta técnica capturaría todos los excedentes de los usuarios.")
    print("\n")
    print("#"*80)

    #2.d) Problemas éticos

    print("#"*80)
    print("\n")
    print("d.2) Problemas éticos")
    print("\n")
    print("Algunos problemas éticos serían:")
    print("- Usuarios similares pagan precios distintos. Esto se podría entender como asignación arbitraria de precios, lo que podría tarer problemas regulatorios.")
    print("- Abuso de poder y vulnerabilidad. Un precio por poca batería, un viaje urgente o un viaje de noche demostraría que se está abusando del usuario.")
    print("- Podría darse el caso de que el modelo de ML no asigne viajes a usuarios con proxy de poco ingreso (modelo celular) porque pagan menos que otros sectores. Esto se traduce en discrimincación para grupos mas vulnerables.")
    print("\n")
    print("#"*80)

    #3.d) Normas legales en chile

    print("#"*80)
    print("\n")
    print("d.3) Normas legales en chile")
    print("\n")
    print("Actualmente, existe la ley 19.496 - Ley del consumidor. En este se expresa:")
    print("- El derecho a contar con información veraz y oportuna sobre los bienes y servicios ofrecidos, precio y condiciones de contratación.")
    print("- No discriminación arbitraria por parte de las empresas proveedoras.")
    print("\n")
    print("El uso de surge pricing con ML podría eventualmente caer en contradicción de uno de estos puntos, como la razón del precio cobrado o la discriminación de usuarios por parte del modelo.")
    print("\n")
    print("Por otro lado, la nueva ley 21.719 de datos personales, que entra en vigencia el 01.12.2026, estipula que el tratamiento de los datos requiere consentimiento libre, informado, específico e inequívoco.")
    print("No se podrán utilizar los datos para fines que puedan derivar en discriminación del servicio, esto podría traer consecuencias legales.")
    print("\n")
    print("#"*80)

    #4.d) Límites entre optimización y explotación

    print("#"*80)
    print("\n")
    print("d.4) Límites entre optimización y explotación")
    print("\n")
    print("El límite se marca cuando la búsqueda de equilibrio se centra entre quienes pueden pagar más por un servicio estándar en condiciones normales, y la explotación es abusar de ciertas características de un viaje estándar.")
    print("En general, abusar de un usuario sería cobrar mas en condiciones perjudiciales para el/ella, como cobrar mas de noche, por un viaje urgente, por alguna catástrofe, por poca batería, entre otros. Usar esta información para cobrar más no es ético, pues abusa del nivel de información a sabiendas de que el usuario probablemente pague mas por estar seguro.")
    print("Si se comienza a abusar de manera generalizada, la prensa y las redes sociales podrían iniciar campañas contra la empresa, dejando en evidencia comportamientos que el modelo aprendió y no necesariamente reflejan la visión de la empresa.")
    print("Así, siempre es importante estar atento a las sugerencias de los usuarios, proporcionando opciones de mejora, buscando alternativas y nuevos servicios.")
    print("\n")
    print("#"*80)
if __name__ == "__main__":
    run()
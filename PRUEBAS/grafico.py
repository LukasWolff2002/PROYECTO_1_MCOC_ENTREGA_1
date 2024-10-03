#importar variables de codigo.py

#Dibujar las redes de flujo en python, las cuadriculas deben ser de 5x5 mm
#Abajo tiene que ser curvo
#Intentar hacer en autocad

from variables import caso_1, caso_2, caso_3, num_equipotenciales
from matplotlib import pyplot as plt
from lineas import agregar_linea_horizontal, agregar_linea_vertical
from lineas_flujo import agregar_red_de_flujo, agregar_red_de_flujo_recta, agregar_red_de_flujo_fondo
from lineas_equipotenciales import agregar_lineas_equipotenciales, extraer_pendientes, extraer_puntos, graficar_lineas_con_pendientes, agregar_lineas_equipotenciales_fondo


# Dimensiones de la hoja A4 en milímetros
ancho_a4_mm = 210
alto_a4_mm = 150

# Tamaño de la cuadrícula en milímetros
tamanio_cuadricula_mm = 5

def crear_hoja_cuadriculada(ancho, alto, tamanio_cuadricula):
    # Configurar la figura para que tenga el tamaño de una hoja A4 en pulgadas (1 pulgada = 25.4 mm)
    fig, ax = plt.subplots(figsize=(ancho / 25.4, alto / 25.4), dpi=100)
    
    # Eliminar márgenes
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Dibujar líneas verticales
    for x in range(0, ancho + tamanio_cuadricula, tamanio_cuadricula):
        ax.axvline(x, color='gray', linewidth=0.5)

    # Dibujar líneas horizontales
    for y in range(0, alto + tamanio_cuadricula, tamanio_cuadricula):
        ax.axhline(y, color='gray', linewidth=0.5)

    # Establecer los límites de la gráfica
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, alto)

    # Eliminar marcas de los ejes
    ax.set_xticks([])
    ax.set_yticks([])

    return ax

def graficar(caso, nombre, altura_base):
    d = (caso['d']*1000) / 200
    ax = crear_hoja_cuadriculada(ancho_a4_mm, alto_a4_mm, tamanio_cuadricula_mm)
    #agregar_linea_horizontal(ax, 0.5 + altura_base, 0, 210, 'black')  # Linea de fondo
    C1 = (caso['c1'] * 1000) / 200  # Convierto a escala 1:200
    agregar_linea_horizontal(ax, C1 + altura_base, 0, 105, 'black')  # Linea de C1
    C2 = (caso['c2'] * 1000) / 200  # Convierto a escala 1:200
    agregar_linea_horizontal(ax, C2 + altura_base, 105, 105, 'black')  # Linea de C2
    B1 = (caso['b1']*1000)/200 #Convierto a escala 1:200
    B1 = B1 + C1 #Sumo la altura de C1
    agregar_linea_horizontal(ax, B1+altura_base, 0, 105, 'blue') #Linea de B1
    B2 = (caso['b2']*1000)/200 #Convierto a escala 1:200
    B2 = B2 + C2 #Sumo la altura de C2
    agregar_linea_horizontal(ax, B2+altura_base, 105, 105, 'blue') #Linea de B2
    A1 = (caso['a1']*1000)/200 #Convierto a escala 1:200
    A1 = A1 + B1 #Sumo la altura de B1
    agregar_linea_vertical(ax, 105, C2+altura_base-d, A1+altura_base) #Linea de A1

    agregar_linea_horizontal(ax, 0.1+altura_base, 0, 210, 'black') #Linea de A1
    # Dibujar una curva de flujo con dos puntos de control
    punto_1 =(C2-d)/4
    punto_2 = punto_1*2
    punto_3 = punto_1*3



    bezier_path_sup = agregar_red_de_flujo(ax, (0, 0), (52.5, 0), (105, 0), (157.5, 0), (210, 0), altura_base, False, grosor=0)
    bezier_path_ver = agregar_red_de_flujo(ax, (105, 0), (105, (C2-d)/2), (105, C2-d), (105, ((C2-d)*3)/2), (105, 2*(C2-d)), altura_base, False, grosor=0)
    pendientes_inicio, coordenadas_inicio = agregar_lineas_equipotenciales(ax, bezier_path_sup,8, longitud=10, color='red', grosor=0)
    pendientes_VER, coordenadas_VER = agregar_lineas_equipotenciales(ax, bezier_path_ver,8, longitud=10, color='red', grosor=0)

    bezier_path_ataguia = agregar_red_de_flujo(ax, (105, C1), (105, (C2-d)), (105, C2-d), (105, (C2-d)), (105, C2-0.5), altura_base, False, grosor=0)
    pendientes_ataguia, coordenadas_ataguia = agregar_lineas_equipotenciales(ax, bezier_path_ataguia,num_equipotenciales, longitud=10, color='red', grosor=0)

    #fondo = agregar_red_de_flujo_recta(ax, (0.5, C1), (0.5, 0), (105, 0), (210, 0), (210, C2), altura_base, True)
    #agregar_lineas_equipotenciales_rectas(ax, fondo, num_equipotenciales, longitud=10, color='green', grosor=5)
    
    #TRBAJAR AQUIIIII
    #fondo = agregar_red_de_flujo_fondo(ax, (0.5, C1), (0.5, C1/5), (0.5, 0), (105/5, 0), (105, 0),altura_base, False)
    #p_fondo, coordenadas_fondo = agregar_lineas_equipotenciales_fondo(ax, fondo,int(num_equipotenciales/2), longitud=10, color='red', grosor=1)

    distancia_1 = coordenadas_inicio['punto_1'][0]+5
    distancia_2 = coordenadas_inicio['punto_2'][0] + 7
    distancia_3 = coordenadas_inicio['punto_3'][0]-5

    ver_1 = coordenadas_VER['punto_1'][1]-altura_base
    ver_2 = coordenadas_VER['punto_2'][1]-altura_base+2.5
    ver_3 = coordenadas_VER['punto_3'][1]-altura_base 

    '''
    bezier_path_1 = agregar_red_de_flujo(ax, (26.25, C1), (52.5,punto_1 ),(105, punto_1), (157.5, punto_1), (183.75, C2), altura_base, True)
    bezier_path_2 = agregar_red_de_flujo(ax, (26.25*2, C1), (65.625,punto_2 ), (105, punto_2), (144.375, punto_2),(183.75-26.25, C2), altura_base, True)
    bezier_path_3 = agregar_red_de_flujo(ax, (26.25*3, C1), (75.75,punto_3 ), (105, punto_3), (134.25, punto_3), (183.75-(26.25*2), C2), altura_base, True)
    '''
    
    bezier_path_1 = agregar_red_de_flujo(ax, (105-distancia_3, C1), (52.5,ver_1 ),(105, ver_1), (157.5, ver_1), (105+distancia_3, C2), altura_base, False)
    bezier_path_2 = agregar_red_de_flujo(ax, (105-distancia_2, C1), (65.625,ver_2 ), (105, ver_2), (144.375, ver_2),(105+distancia_2, C2), altura_base, False)
    bezier_path_3 = agregar_red_de_flujo(ax, (105-distancia_1, C1), (75.75,ver_3 ), (105, ver_3), (134.25, ver_3), (105+distancia_1, C2), altura_base, False)
    bezier_path_4 = agregar_red_de_flujo(ax, (0, 0), (52.5, 0), (105, 0), (157.5, 0), (210, 0), altura_base, False, grosor=0)

    # Dibujar las líneas equipotenciales
    pendientes_1, coordenadas_1 = agregar_lineas_equipotenciales(ax, bezier_path_1, num_equipotenciales, longitud=10, color='green', grosor=0)
    pendientes_2, coordenadas_2 = agregar_lineas_equipotenciales(ax, bezier_path_2, num_equipotenciales, longitud=10, color='green', grosor=0)
    pendientes_3, coordenadas_3 = agregar_lineas_equipotenciales(ax, bezier_path_3, num_equipotenciales, longitud=10, color='green', grosor=0)
    pendientes_4, coordenadas_4 = agregar_lineas_equipotenciales(ax, bezier_path_4, num_equipotenciales, longitud=10, color='green', grosor=0)
    

    # Obtener las claves del diccionario
    claves = list(pendientes_4.keys())
    # Calcular el punto de la mitad
    mitad = len(claves) // 2

    # Asignar valores de +10 para la primera mitad
    for i in range(mitad):
        pendientes_4[claves[i]] = 10.0

    # Asignar valores de -10 para la segunda mitad
    for i in range(mitad, len(claves)):
        pendientes_4[claves[i]] = -10.0

    #Ajusto el grafico manualemnte
    punto_2 = 'punto_' +str(num_equipotenciales-2)
    coordenadas_4['punto_1'] = (0, C1/3)
    coordenadas_4[punto_2] = (ancho_a4_mm, C2/4)

    pendientes = [pendientes_4, pendientes_1, pendientes_2, pendientes_3, pendientes_ataguia]
    coordenadas = [coordenadas_4, coordenadas_1, coordenadas_2, coordenadas_3, coordenadas_ataguia]

    
    # Llamar a la función para graficar las líneas
    graficar_lineas_con_pendientes(ax, coordenadas, pendientes, color='green', grosor=1)

    # Guardar la figura usando el objeto ax
    #plt.savefig(f"{nombre}.jpg", format='jpg', bbox_inches='tight', pad_inches=0)

    return ax, pendientes, coordenadas

#Ejemplo de uso
#graficar(caso_1, 'caso_1', 0)
# graficar(caso_2, 'caso_2', 50)
# graficar(caso_3, 'caso_3', 50)


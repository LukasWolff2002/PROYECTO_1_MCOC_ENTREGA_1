from scipy.interpolate import CubicHermiteSpline, CubicSpline
import numpy as np


def agregar_lineas_equipotenciales(ax, bezier_path, num_equipotenciales=5, longitud=10, color='green', grosor=1.5):

    # Diccionario para almacenar la pendiente en cada punto y coordenadas
    pendientes_diccionario = {}
    coordenadas_diccionario = {}

    # Dividir la curva en puntos equidistantes para colocar las equipotenciales
    num_puntos = len(bezier_path)
    indices_equipotenciales = np.linspace(0, num_puntos - 1, num_equipotenciales).astype(int)
    i = 0
    for idx in indices_equipotenciales:
        # Punto en la curva donde vamos a colocar la línea equipotencial
        x, y = bezier_path[idx]
        
        # Derivada numérica para calcular la pendiente de la curva en ese punto
        if idx > 0:
            dx = bezier_path[idx][0] - bezier_path[idx - 1][0]
            dy = bezier_path[idx][1] - bezier_path[idx - 1][1]
        else:
            # Si estamos en el primer punto, tomamos la derivada con el siguiente
            dx = bezier_path[idx + 1][0] - bezier_path[idx][0]
            dy = bezier_path[idx + 1][1] - bezier_path[idx][1]

        # Pendiente de la curva de flujo
        pendiente_flujo = dy / dx if dx != 0 else np.inf
        
        # Pendiente de la línea equipotencial, que es el negativo recíproco
        if pendiente_flujo != 0 and pendiente_flujo != np.inf:
            pendiente_equipotencial = -1 / pendiente_flujo
        elif pendiente_flujo == 0:
            pendiente_equipotencial = np.inf  # Línea equipotencial será vertical
        else:
            pendiente_equipotencial = 0  # Línea equipotencial será horizontal

        # Almacenar la pendiente en el diccionario
        pendientes_diccionario[f'punto_{i}'] = -1/pendiente_flujo

        # Almacenar las coordenadas en el diccionario
        coordenadas_diccionario[f'punto_{i}'] = (x, y)
        
        # Calcular los puntos de la línea equipotencial
        if pendiente_equipotencial == np.inf:
            # Línea equipotencial vertical
            x_values = [x, x]
            y_values = [y - longitud / 2, y + longitud / 2]
        elif pendiente_equipotencial == 0:
            # Línea equipotencial horizontal
            x_values = [x - longitud / 2, x + longitud / 2]
            y_values = [y, y]
        else:
            # General case
            delta_x = longitud / (2 * np.sqrt(1 + pendiente_equipotencial**2))
            delta_y = pendiente_equipotencial * delta_x
            x_values = [x - delta_x, x + delta_x]
            y_values = [y - delta_y, y + delta_y]
        
        # Dibujar la línea equipotencial
        ax.plot(x_values, y_values, color=color, linewidth=grosor)
        i += 1

    return pendientes_diccionario, coordenadas_diccionario

def agregar_lineas_equipotenciales_fondo(ax, bezier_path, num_equipotenciales=5, longitud=10, color='green', grosor=1):
    # Encontrar el punto más alto de la función
    max_y = np.max(bezier_path[:, 1])
    max_index = np.argmax(bezier_path[:, 1])
    punto_alto = bezier_path[max_index]

    # Espaciado inicial y factor de disminución
    espaciado_inicial = 10
    factor_disminucion = 0.8
    
    # Diccionario para almacenar la pendiente en cada punto y coordenadas
    pendientes_diccionario = {}
    coordenadas_diccionario = {}
    
    # Calcular las líneas equipotenciales
    for i in range(num_equipotenciales):
        # Espaciado decreciente
        espaciado_actual = espaciado_inicial * (factor_disminucion ** i)
        
        # Coordenadas del punto para la línea equipotencial
        x, y = punto_alto
        
        # Derivada numérica para calcular la pendiente de la curva en ese punto
        if max_index > 0 and max_index < len(bezier_path) - 1:
            dx = bezier_path[max_index + 1][0] - bezier_path[max_index - 1][0]
            dy = bezier_path[max_index + 1][1] - bezier_path[max_index - 1][1]
        elif max_index == 0:
            # Si estamos en el primer punto, tomamos la derivada con el siguiente
            dx = bezier_path[max_index + 1][0] - bezier_path[max_index][0]
            dy = bezier_path[max_index + 1][1] - bezier_path[max_index][1]
        else:
            # Si estamos en el último punto, tomamos la derivada con el anterior
            dx = bezier_path[max_index][0] - bezier_path[max_index - 1][0]
            dy = bezier_path[max_index][1] - bezier_path[max_index - 1][1]
        
        pendiente_flujo = dy / dx if dx != 0 else np.inf
        
        # Pendiente de la línea equipotencial, que es el negativo recíproco
        if pendiente_flujo != 0 and pendiente_flujo != np.inf:
            pendiente_equipotencial = -1 / pendiente_flujo
        elif pendiente_flujo == 0:
            pendiente_equipotencial = np.inf  # Línea equipotencial será vertical
        else:
            pendiente_equipotencial = 0  # Línea equipotencial será horizontal
        
        # Almacenar la pendiente en el diccionario
        pendientes_diccionario[f'punto_{i}'] = -1/pendiente_flujo
        
        # Almacenar las coordenadas en el diccionario
        coordenadas_diccionario[f'punto_{i}'] = (x, y)
        
        # Calcular los puntos de la línea equipotencial
        if pendiente_equipotencial == np.inf:
            # Línea equipotencial vertical
            x_values = [x, x]
            y_values = [y - espaciado_actual / 2, y + espaciado_actual / 2]
        elif pendiente_equipotencial == 0:
            # Línea equipotencial horizontal
            x_values = [x - espaciado_actual / 2, x + espaciado_actual / 2]
            y_values = [y, y]
        else:
            # General case
            delta_x = longitud / (2 * np.sqrt(1 + pendiente_equipotencial**2))
            delta_y = pendiente_equipotencial * delta_x
            x_values = [x - delta_x, x + delta_x]
            y_values = [y - delta_y, y + delta_y]
        
        # Dibujar la línea equipotencial
        ax.plot(x_values, y_values, color=color, linewidth=grosor)
    
    return pendientes_diccionario, coordenadas_diccionario

'''
def agregar_lineas_equipotenciales(ax, bezier_path, num_equipotenciales=5, longitud=10, color='green', grosor=1):
    # Diccionario para almacenar la pendiente en cada punto y coordenadas
    pendientes_diccionario = {}
    coordenadas_diccionario = {}

    # Dividir la curva en puntos equidistantes para colocar las equipotenciales
    num_puntos = len(bezier_path)
    indices_equipotenciales = np.linspace(0, num_puntos - 1, num_equipotenciales).astype(int)

    # Calcular el punto medio de la curva
    x_medio, y_medio = bezier_path[num_puntos // 2]

    i = 0
    for idx in indices_equipotenciales:
        # Punto en la curva donde vamos a colocar la línea equipotencial
        x, y = bezier_path[idx]
        
        # Derivada numérica para calcular la pendiente de la curva en ese punto
        if idx > 0:
            dx = bezier_path[idx][0] - bezier_path[idx - 1][0]
            dy = bezier_path[idx][1] - bezier_path[idx - 1][1]
        else:
            # Si estamos en el primer punto, tomamos la derivada con el siguiente
            dx = bezier_path[idx + 1][0] - bezier_path[idx][0]
            dy = bezier_path[idx + 1][1] - bezier_path[idx][1]

        # Pendiente de la curva de flujo
        pendiente_flujo = dy / dx if dx != 0 else np.inf
        
        # Pendiente de la línea equipotencial, que es el negativo recíproco
        if pendiente_flujo != 0 and pendiente_flujo != np.inf:
            pendiente_equipotencial = -1 / pendiente_flujo
        elif pendiente_flujo == 0:
            pendiente_equipotencial = np.inf  # Línea equipotencial será vertical
        else:
            pendiente_equipotencial = 0  # Línea equipotencial será horizontal

        # Almacenar la pendiente en el diccionario
        pendientes_diccionario[f'punto_{i}'] = pendiente_equipotencial

        # Almacenar las coordenadas en el diccionario
        coordenadas_diccionario[f'punto_{i}'] = (x, y)
        
        # Calcular la distancia al punto medio
        distancia_al_medio = np.sqrt((x - x_medio)**2 + (y - y_medio)**2)
        
        # Modificar la longitud de la línea equipotencial según la distancia al punto medio
        longitud_modificada = longitud / (1 + distancia_al_medio)
        
        # Calcular los puntos de la línea equipotencial
        if pendiente_equipotencial == np.inf:
            # Línea equipotencial vertical
            x_values = [x, x]
            y_values = [y - longitud_modificada / 2, y + longitud_modificada / 2]
        elif pendiente_equipotencial == 0:
            # Línea equipotencial horizontal
            x_values = [x - longitud_modificada / 2, x + longitud_modificada / 2]
            y_values = [y, y]
        else:
            # Caso general
            delta_x = longitud_modificada / (2 * np.sqrt(1 + pendiente_equipotencial**2))
            delta_y = pendiente_equipotencial * delta_x
            x_values = [x - delta_x, x + delta_x]
            y_values = [y - delta_y, y + delta_y]
        
        # Dibujar la línea equipotencial
        ax.plot(x_values, y_values, color=color, linewidth=grosor)
        i += 1

    # Retornar los diccionarios con pendientes y coordenadas
    return pendientes_diccionario, coordenadas_diccionario
'''

'''
def agregar_lineas_equipotenciales(ax, bezier_path, num_equipotenciales=5, longitud=10, color='green', grosor=1):
    # Diccionario para almacenar la pendiente en cada punto y coordenadas
    pendientes_diccionario = {}
    coordenadas_diccionario = {}

    # Definir el número de puntos y el punto intermedio
    num_puntos = len(bezier_path)
    mid_idx = num_puntos // 2  # Índice del punto central

    # Crear un factor de escala que genera más puntos cerca del centro
    scale_factor = np.linspace(0, 1, num=num_equipotenciales//2 + 1)
    scale_factor = np.sqrt(scale_factor)  # Aumentar la densidad cerca del centro

    # Parte 1: Distribuir líneas desde el inicio hacia el centro
    for i, s in enumerate(scale_factor[:-1]):  # Ignoramos el último valor porque ya será el punto central
        idx = int(s * mid_idx)
        # Punto en la curva donde se coloca la línea equipotencial
        x, y = bezier_path[idx]
        
        # Derivada numérica para calcular la pendiente de la curva en ese punto
        if idx > 0:
            dx = bezier_path[idx][0] - bezier_path[idx - 1][0]
            dy = bezier_path[idx][1] - bezier_path[idx - 1][1]
        else:
            dx = bezier_path[idx + 1][0] - bezier_path[idx][0]
            dy = bezier_path[idx + 1][1] - bezier_path[idx][1]

        # Pendiente de la curva de flujo
        pendiente_flujo = dy / dx if dx != 0 else np.inf
        
        # Pendiente de la línea equipotencial, que es el negativo recíproco
        if pendiente_flujo != 0 and pendiente_flujo != np.inf:
            pendiente_equipotencial = -1 / pendiente_flujo
        elif pendiente_flujo == 0:
            pendiente_equipotencial = np.inf  # Línea equipotencial será vertical
        else:
            pendiente_equipotencial = 0  # Línea equipotencial será horizontal

        # Almacenar la pendiente y las coordenadas en el diccionario
        pendientes_diccionario[f'punto_{i}'] = -1 / pendiente_flujo
        coordenadas_diccionario[f'punto_{i}'] = (x, y)

        # Calcular los puntos de la línea equipotencial
        if pendiente_equipotencial == np.inf:
            x_values = [x, x]
            y_values = [y - longitud / 2, y + longitud / 2]
        elif pendiente_equipotencial == 0:
            x_values = [x - longitud / 2, x + longitud / 2]
            y_values = [y, y]
        else:
            delta_x = longitud / (2 * np.sqrt(1 + pendiente_equipotencial**2))
            delta_y = pendiente_equipotencial * delta_x
            x_values = [x - delta_x, x + delta_x]
            y_values = [y - delta_y, y + delta_y]

        # Dibujar la línea equipotencial
        ax.plot(x_values, y_values, color=color, linewidth=grosor)

    # Parte 2: Distribuir líneas desde el final hacia el centro
    for i, s in enumerate(scale_factor[:-1]):  # Ignoramos el último valor
        idx = num_puntos - 1 - int(s * (num_puntos - mid_idx - 1))
        # Punto en la curva donde se coloca la línea equipotencial
        x, y = bezier_path[idx]
        
        # Derivada numérica para calcular la pendiente de la curva en ese punto
        if idx < num_puntos - 1:
            dx = bezier_path[idx][0] - bezier_path[idx + 1][0]
            dy = bezier_path[idx][1] - bezier_path[idx + 1][1]
        else:
            dx = bezier_path[idx - 1][0] - bezier_path[idx][0]
            dy = bezier_path[idx - 1][1] - bezier_path[idx][1]

        # Pendiente de la curva de flujo
        pendiente_flujo = dy / dx if dx != 0 else np.inf
        
        # Pendiente de la línea equipotencial, que es el negativo recíproco
        if pendiente_flujo != 0 and pendiente_flujo != np.inf:
            pendiente_equipotencial = -1 / pendiente_flujo
        elif pendiente_flujo == 0:
            pendiente_equipotencial = np.inf  # Línea equipotencial será vertical
        else:
            pendiente_equipotencial = 0  # Línea equipotencial será horizontal

        # Almacenar la pendiente y las coordenadas en el diccionario
        pendientes_diccionario[f'punto_{i + len(scale_factor)}'] = -1 / pendiente_flujo
        coordenadas_diccionario[f'punto_{i + len(scale_factor)}'] = (x, y)

        # Calcular los puntos de la línea equipotencial
        if pendiente_equipotencial == np.inf:
            x_values = [x, x]
            y_values = [y - longitud / 2, y + longitud / 2]
        elif pendiente_equipotencial == 0:
            x_values = [x - longitud / 2, x + longitud / 2]
            y_values = [y, y]
        else:
            delta_x = longitud / (2 * np.sqrt(1 + pendiente_equipotencial**2))
            delta_y = pendiente_equipotencial * delta_x
            x_values = [x - delta_x, x + delta_x]
            y_values = [y - delta_y, y + delta_y]

        # Dibujar la línea equipotencial
        ax.plot(x_values, y_values, color=color, linewidth=grosor)

    return pendientes_diccionario, coordenadas_diccionario
'''

def extraer_puntos(lista_diccionarios, i):
    lista_coordenadas = []

    for diccionario in lista_diccionarios:
        # Verificar si el diccionario contiene 'punto_1'
        if f'punto_{i}' in diccionario:
            if isinstance(diccionario[f'punto_{i}'], tuple):
                # Si es una tupla, son coordenadas
                lista_coordenadas.append(diccionario[f'punto_{i}'])
            
    return lista_coordenadas

def extraer_pendientes(lista_diccionarios, i):
    lista_pendientes = []
   
    for diccionario in lista_diccionarios:
        # Verificar si el diccionario contiene 'punto_1'
        if f'punto_{i}' in diccionario:
            if isinstance(diccionario[f'punto_{i}'], float):
                # Si es un flotante, es una pendiente
                lista_pendientes.append(diccionario[f'punto_{i}'])
            
    return lista_pendientes


def graficar_lineas_con_pendientes(ax, coordenadas, pendientes, color='blue', grosor=2):

    for i in range(len(coordenadas[0])):

        if i == 0:
            continue

        if i == len(coordenadas[0])-1:
            break

        # Extraer las coordenadas y pendientes de cada punto
        coor = extraer_puntos(coordenadas , i)
        m = extraer_pendientes(pendientes, i)


        x_coords = [c[0] for c in coor]
        y_coords = [c[1] for c in coor]

        # Checkeo que x sea en orden ascendente
        if x_coords != sorted(x_coords):
            x_coords = x_coords[::-1]  # Invertir la lista si no está en orden ascendente
            y_coords = y_coords[::-1]

        # Crear el spline cúbico que pase por los puntos sin considerar pendientes
        spline = CubicSpline(x_coords, y_coords)

        # Generar puntos adicionales para graficar la curva suavemente
        x_new = np.linspace(min(x_coords), max(x_coords), 100)
        y_new = spline(x_new)

        # Dibujar la curva en el gráfico
        ax.plot(x_new, y_new, color=color, linewidth=grosor)

        i += 1

'''
def graficar_lineas_con_pendientes(ax, coordenadas, pendientes, color='blue', grosor=1):

    for i in range(len(coordenadas[0])):

        if i == 0:
            continue

        if i == len(coordenadas[0])-1:
            break

        # Extraer las coordenadas y pendientes de cada punto
        coor = extraer_puntos(coordenadas , i)
        m = extraer_pendientes(pendientes, i)

        print(coor)
 

        # Extraer las coordenadas X e Y de las coordenadas
        x_coords = [c[0] for c in coor]
        y_coords = [c[1] for c in coor]


        #Checkeo que x sea en orden ascendente
        if x_coords != sorted(x_coords):
            x_coords = x_coords [::-1]  # Invertir la lista si no está en orden ascendente
            y_coords = y_coords [::-1]
            m = m [::-1]

        print('x coord', x_coords)
        print('y coord', y_coords)
        print('m',m)

        t = np.arange(len(x_coords))  # t: 0, 1, 2, ..., len(x_coords) - 1

        # Crear el spline cúbico hermítico que respeta las pendientes
        spline_x = CubicHermiteSpline(t, x_coords, dydx=m)  # Spline para x
        spline_y = CubicHermiteSpline(t, y_coords, dydx=m)  # Spline para y
        
        # Generar puntos adicionales para graficar la curva suavemente
        t_new = np.linspace(0, len(x_coords) - 1, 100)
        x_new = spline_x(t_new)
        y_new = spline_y(t_new)

        # Dibujar la curva en el gráfico
        ax.plot(x_new, y_new, color=color, linewidth=grosor)

        i += 1

'''

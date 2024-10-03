import numpy as np
from scipy.interpolate import CubicHermiteSpline

def agregar_red_de_flujo(ax, inicio, control1, intermedio, control2, fin, altura_base, display,grosor=1.5, color='blue'):
    # Convertir puntos a numpy arrays
    p0 = np.array(inicio)
    p1 = np.array(control1)
    p2 = np.array(control2)
    p3 = np.array(fin)
    p_intermedio = np.array(intermedio)

    # Ajustar control1 y control2 para tener pendiente casi vertical
    control_vertical_1 = np.array([p0[0], p1[1]])  # Mantener x constante en p0
    control_vertical_2 = np.array([p3[0], p2[1]])  # Mantener x constante en p3

    # Definir t
    t = np.linspace(0, 1, 100)

    # Primera curva Bezier desde inicio (p0) hasta intermedio (p_intermedio) con pendiente casi vertical en p0
    bezier_path_1 = (1 - t)[:, None]**2 * p0 + 2 * (1 - t)[:, None] * t[:, None] * control_vertical_1 + t[:, None]**2 * p_intermedio

    # Segunda curva Bezier desde intermedio (p_intermedio) hasta fin (p3) con pendiente casi vertical en p3
    bezier_path_2 = (1 - t)[:, None]**2 * p_intermedio + 2 * (1 - t)[:, None] * t[:, None] * control_vertical_2 + t[:, None]**2 * p3

    # Ajustar la altura base en ambos caminos
    bezier_path_1[:, 1] += altura_base
    bezier_path_2[:, 1] += altura_base

    # Combinar ambos segmentos de curva
    bezier_path_combined = np.vstack([bezier_path_1, bezier_path_2])

    # Dibujar ambas curvas
    ax.plot(bezier_path_1[:, 0], bezier_path_1[:, 1], color=color, linewidth=grosor)
    ax.plot(bezier_path_2[:, 0], bezier_path_2[:, 1], color=color, linewidth=grosor)

    if display:

        # Dibujar puntos de control
        ax.scatter([p0[0], control_vertical_1[0], control_vertical_2[0], p3[0], p_intermedio[0]], 
                [p0[1] + altura_base, control_vertical_1[1] + altura_base, control_vertical_2[1] + altura_base, p3[1] + altura_base, p_intermedio[1] + altura_base],
                color='red', label='Puntos de Control', zorder=5)

        ax.legend()

    # Devolver el camino de la curva combinado
    return bezier_path_combined

def agregar_red_de_flujo_fondo(ax, inicio, intermedio1, control1, intermedio2, intermedio3, altura_base, display, color='blue', grosor=1):

    # Convertir puntos a numpy arrays
    p0 = np.array(inicio)
    p1 = np.array(intermedio1)
    p2 = np.array(control1)
    p3 = np.array(intermedio2)
    p4 = np.array(intermedio3)
   

    # Ajustar la altura base
    p0[1] += altura_base
    p1[1] += altura_base
    p2[1] += altura_base
    p3[1] += altura_base
    p4[1] += altura_base
   

    # Definir t para interpolación
    t = np.linspace(0, 1, 100)

    # Curva de Bézier cúbica desde inicio hasta intermedio1 pasando por control1
    bezier_path_1 = (1 - t)[:, None]**3 * p0 + 3 * (1 - t)[:, None]**2 * t[:, None] * p1 + \
                    3 * (1 - t)[:, None] * t[:, None]**2 * p2 + t[:, None]**3 * p3

    # Curva de Bézier lineal desde intermedio2 hasta intermedio3
    bezier_path_2 = (1 - t)[:, None] * p3 + t[:, None] * p4


    # Combinar las trayectorias
    bezier_path_combined = np.vstack([bezier_path_1, bezier_path_2])

    # Dibujar las líneas
    ax.plot(bezier_path_combined[:, 0], bezier_path_combined[:, 1], color=color, linewidth=grosor)


    # Devolver el camino combinado
    return bezier_path_combined






def agregar_red_de_flujo_recta(ax, inicio, control1, intermedio, control2, fin, altura_base, display=False, color='blue', grosor=5):
    # Convertir puntos a numpy arrays
    p0 = np.array(inicio)
    p1 = np.array(control1)
    p2 = np.array(control2)
    p3 = np.array(fin)
    p_intermedio = np.array(intermedio)

    # Ajustar la altura base
    p0[1] += altura_base
    p1[1] += altura_base
    p2[1] += altura_base
    p3[1] += altura_base
    p_intermedio[1] += altura_base

    # Crear las líneas rectas entre los puntos
    line_path = np.vstack([p0, p1, p_intermedio, p2, p3])

    # Dibujar las líneas rectas
    ax.plot(line_path[:, 0], line_path[:, 1], color=color, linewidth=grosor)

    if display:
        # Dibujar puntos de control
        ax.scatter([p0[0], p1[0], p_intermedio[0], p2[0], p3[0]],
                   [p0[1], p1[1], p_intermedio[1], p2[1], p3[1]],
                   color='red', label='Puntos de Control', zorder=5)

        ax.legend()

    # Devolver el camino de las líneas rectas
    return line_path




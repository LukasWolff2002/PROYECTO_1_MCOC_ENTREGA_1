import numpy as np
import matplotlib.pyplot as plt
from variables import caso_1, caso_2, caso_3, caso_ejemplo, k
import pandas as pd

#Las grillas deben ser cuadradas, lo cual no se esta cumpliendo actualmente
#En los lugares impermeables, el gradiente debe ser 0



def laplace (caso, nombre, grilla, factor_correccion, factor):
    print(f'calculando {nombre}')
    C2 = caso['c1']
    C2_real = caso['c2']
    C1 = caso['c1']
    B2 = caso['b2']
    B1 = caso['b1']
    D = caso['d']
    D = C2_real - D
    k = caso['k']

    nx, ny = grilla, grilla  # Tamaño de la grilla
    lx, ly = C1, C1  # Dimensiones físicas del dominio
    dx, dy = lx / (nx - 1), ly / (ny - 1)  # Espaciado en x e y
    #print(dx,dy)

    #Defino algunas cosas
    Potencial_incial = (B1)*100 #Ingresar la altura de agua en cm
    Potencial_final = (B2)*100 #Ingresar la altura de agua en cm

    #Transformaciones a grillas
    #c2 esta en metros, por lo tanto, al pasar a grilla
    c2 = int((1-(C2_real/ly))*ny)
    #print(c2)
    D = int((1-(D/ly))*ny)

    # Inicializamos el potencial como una matriz de ceros
    potential = np.ones((ny, nx))
    K = k*np.ones((ny, nx))

    # Condiciones de borde: potencial alto en algunos bordes
    #potential[nx//2:, -1] = 100     # Borde izquierdo: impermeable (potencial bajo)
    #potential[0, :nx//2] = 30     # Borde superior: potencial alto
    #potential[:nx//2, -1] = 0     # Borde izquierdo: impermeable (potencial bajo)
    #potential[-1, :] = 0    # Borde inferior: impermeable
    #potential[:, 0] = 0    # Borde derecho: impermeable (potencial bajo)

    if C2 != ly:

        potential[0, nx//2:] = Potencial_incial     # Borde superior: potencial alto
        potential[c2-1, :ny//2] = Potencial_final   # Borde superior: potencial alto

        potential[:, -1] = 0     # Borde izquierdo: impermeable (potencial bajo)
        K[:, -1] = 1e-15     # Borde izquierdo: impermeable (permeabilidad baja)
        potential[-1, :] = 0    # Borde inferior: impermeable
        K[-1, :] = 1e-15     # Borde inferior: impermeable
        potential[:, 0] = 0    # Borde derecho: impermeable (potencial bajo)
        K[:, 0] = 1e-15     # Borde derecho: impermeable
        potential[:D, ny//2] = 0    #Ataguia
        K[:D, ny//2] = 1e-15     #Ataguia
        potential[:c2-1, :nx//2] = np.nan     # Borde superior: potencial alto
        K[:c2-1, :nx//2] = 1e-15      # Borde superior: potencial alto

    else:

        potential[0, nx//2:] = Potencial_incial     # Borde superior: potencial alto
        potential[0:c2, :ny//2] = Potencial_final   # Borde superior: potencial alto

        potential[:, -1] = 0     # Borde izquierdo: impermeable (potencial bajo)
        K[:, -1] = 1e-15      # Borde izquierdo: impermeable (permeabilidad baja)
        potential[-1:, :] = 0    # Borde inferior: impermeable
        K[-1:, :] = 1e-15     # Borde inferior: impermeable
        potential[:, 0] = 0    # Borde derecho: impermeable (potencial bajo)
        K[:, 0] = 1e-15     # Borde derecho: impermeable
        potential[:D, ny//2] = 0    #Ataguia
        K[:D, ny//2] = 1e-15     #Ataguia
        K[:c2-1, :nx//2] = 1e-15      # Borde superior: potencial alto

    #print(pd.DataFrame(potential))
    #print('')
    #print(pd.DataFrame(K))

    # Resolución iterativa de la ecuación de Laplace
    tolerance = 1e-6
    max_iterations = 100000

    print(pd.DataFrame(potential))

    for it in range(max_iterations):
        potential_old = potential.copy()
    
        for i in range(ny):
            for j in range(nx):
                # Verificar si estamos en una región sin agua (np.nan)
                if np.isnan(potential[i, j]):
                    continue  # Saltamos las celdas sin agua

                # Verificar si estamos en un borde
                if i == 0 or i == ny-1 or j == 0 or j == nx-1:
                    # Si estamos en un borde y el potencial es 0, no hacemos nada
                    if potential[i, j] == 0:
                        continue

                # Verificar si estamos en la línea vertical de nx//2 y hasta ny//2
                elif j == nx//2 and i < D-1:
                    # Si estamos en la línea vertical, consideramos que es una barrera con potencial 0
                    potential[i, j] = 0
                    continue

                elif i == ny-1:  # Borde inferior
                    potential[i, j] = 0
                    continue

                elif i<= c2-1 and j < ny//2:  # Borde superior
                    potential[i, j] = Potencial_final
                    continue

                

                else:
                    # Si estamos en una celda adyacente a un borde, manejamos las actualizaciones evitando las celdas con potencial 0
                    suma_potenciales = 0
                    vecinos_validos = 0

                    # Revisamos el vecino de arriba
                    if i > 0 and not np.isnan(potential[i-1, j]) and potential[i-1, j] > 1:
                        suma_potenciales += potential[i-1, j]
                        vecinos_validos += 1

                    # Revisamos el vecino de abajo
                    #Agregar dos condiciones, si estoy bajo o sobre la ataguia
                    #if i < nx-1 and potential[i+1, j] > 1 and j < 10:
                    #    print(f'se cumple la condicion con {i=}')
                    #    suma_potenciales += potential[i+1, j]
                    #    vecinos_validos += 1
                    

                    if i < ny-2 and potential[i+1, j] > 1:
                        #print(f'se cumple la condicion con {i=}')
                        suma_potenciales += potential[i+1, j]
                        vecinos_validos += 1

                        

                    #----------------------

                    # Revisamos el vecino de la izquierda
                    if j > 0 and potential[i, j-1] > 1:
                        suma_potenciales += potential[i, j-1]
                        vecinos_validos += 1

                    #if j > 0 and potential[i, j-1] > 2 and i >= 30:
                    #    suma_potenciales += potential[i, j-1]
                    #    vecinos_validos += 1

                    # Revisamos el vecino de la derecha
                    if j < nx-1 and potential[i, j+1] > 1:
                        suma_potenciales += potential[i, j+1]
                        vecinos_validos += 1

                    # Actualizamos el potencial si tenemos al menos un vecino válido
                    if vecinos_validos > 0:
                        potential[i, j] = suma_potenciales / vecinos_validos



        
        # Comprobamos la convergencia
        if np.max(np.abs(potential - potential_old)) < tolerance:
            print(f"Convergencia alcanzada en {it} iteraciones")
            break

    print(pd.DataFrame(potential))

    # Calcular el gradiente del potencial (flujo de velocidad)
    dy, dx = np.gradient(potential, dy, dx)

    #print(f'El flujo de agua en x es {dx}')


    #tengo que multiplicar la velocidad por K
    velocity_x = -dx * K 
    velocity_y = -dy * K

    #Obtengo las velocidades de salida:
    salida_velocity_y = velocity_y[c2]

    #print(pd.DataFrame(potential))
            
    #Ok, la salida se de en el vector de velocidad y, donde debe ser el valor negativo

    #print('')
    #print(f'Flujo de salida en y es{salida_velocity_y}')

    #Ahora saco los vectores de salida
    #print('')
    salida_velocity_y = salida_velocity_y[1:ny//2]
    #print(f'Flujo de salida en y es{salida_velocity_y}')

    # Calcular el flujo de salida
    #El espaciado de cada grilla es
    espaciado = lx/grilla
    #print(f'El espaciado es {espaciado}')
    #print('')
    q = ((np.sum(salida_velocity_y)*espaciado*-1)/(C1*10/2))
    print(f'El flujo de salida es {q = } m/s')

    # Crear la grilla de coordenadas para visualizar
    x = np.linspace(-lx / 2, lx / 2, nx)
    y = np.linspace(-ly / 2, ly / 2, ny)
    X, Y = np.meshgrid(x, y)

    # Invertir los valores de los ejes X y Y
    X_inverted = -X
    Y_inverted = -Y


    # Graficar el potencial y las líneas de flujo con streamplot
    plt.figure(figsize=(20, 5))

    # Subplot para el potencial
    plt.subplot(1, 3, 1)
    plt.contourf(X_inverted, Y_inverted, K, levels=50, cmap='summer')

    plt.title("Distribución Matriz K")
    plt.xticks([])
    plt.yticks([])

    # Crear una máscara para evitar regiones donde el potencial es 0
    masked_potential = np.ma.masked_where(K < 1e-10, potential)

    # Subplot para el potencial
    plt.subplot(1, 3, 2)
    plt.contourf(X_inverted, Y_inverted, masked_potential/100, levels=18, cmap='Blues')
    plt.colorbar(label="Potential")
    plt.title("Matriz de Potencial")
    plt.xticks([])
    plt.yticks([])

    

    # Subplot para las líneas de flujo (streamlines)
    plt.subplot(1, 3, 3)
    plt.streamplot(X, Y, velocity_x, velocity_y, linewidth=1, cmap='viridis')
    contour_lines = plt.contour(X, Y, masked_potential/100, levels=12, colors='red')
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f')  # Etiquetas de las líneas
    plt.title("Lineas Equipotenciales y FLujo")
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(f"laplace_{nombre}.jpg", format='jpg', bbox_inches='tight', pad_inches=0)

    print(f'listo {nombre}')
    print('')

laplace(caso_1, 'caso_1', 10, False, 0) #14 levels y 10 en el factor de correccion
#laplace(caso_2, 'caso_2', 60, False, 0) #14 levels y 10 en el factor de correccion
#laplace(caso_3, 'caso_3', 60, True, 30)

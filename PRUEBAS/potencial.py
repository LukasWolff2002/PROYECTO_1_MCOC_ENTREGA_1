import numpy as np
import matplotlib.pyplot as plt

# Parámetros
nx, ny = 50, 50  # Tamaño de la grilla
lx, ly = 4, 4  # Dimensiones físicas del dominio
dx, dy = lx / (nx - 1), ly / (ny - 1)  # Espaciado en x e y

# Inicializamos el potencial como una matriz de ceros
potential = np.zeros((ny, nx))

# Condiciones de borde: potencial alto en un lado y bajo en el otro
potential[:, 0] = 0  # Borde izquierdo: potencial bajo
potential[nx//2:, 0] = 15  # Borde derecho: potencial alto
potential[0, :] = 30  # Borde derecho: potencial alto

#Defino una barrera vertical transversal al flujo
#barrier_x = nx // 2  # Posición x de la barrera
#barrier_y_start, barrier_y_end = ny // 4, 3 * ny // 4  # Altura de la barrera
#potential[barrier_y_start:barrier_y_end, barrier_x] = 0  # Mantener el potencial constante en la barrera

#Defino una barrera horizontal transversal al flujo
#barrier_y = ny // 2  # Posición y de la barrera
#barrier_x_start, barrier_x_end = nx // 4, 3 * nx // 4  # Ancho de la barrera
#potential[barrier_y, barrier_x_start:barrier_x_end] = 0  # Mantener el potencial constante en la barrera

# Definir una barrera vertical en el centro del dominio, longitudinal al flujo
# Definir una barrera vertical en el centro del dominio
barrier_x = 0  # Posición x de la barrera
barrier_y_start, barrier_y_end = ny, ny // 2 # Altura de la barrera

# Asignamos un valor fijo a las celdas de la barrera para bloquear el paso
potential[barrier_y_start:barrier_y_end, barrier_x] = 0  # Valor fijo





# Resolución iterativa de la ecuación de Laplace
tolerance = 1e-6
max_iterations = 100
for it in range(max_iterations):
    potential_old = potential.copy()
    
    # Actualizamos el potencial en cada punto, evitando la barrera
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            #if i >= barrier_y_start and i < barrier_y_end and j == barrier_x: Para una barrera  vertical transversal al flujo
            #if i == barrier_y and j >= barrier_x_start and j < barrier_x_end: #Para una barrera  horizontal transversal al flujo
            if i >= barrier_y_start and i < barrier_y_end and j == barrier_x:
                continue  # Saltar la actualización en las barreras
            potential[i, j] = 0.25 * (potential[i, j-1] +  # Izquierda
                                      potential[i, j+1] +  # Derecha
                                      potential[i-1, j] +  # Arriba
                                      potential[i+1, j])   # Abajo
    
    # Comprobamos la convergencia
    if np.max(np.abs(potential - potential_old)) < tolerance:
        print(f"Convergencia alcanzada en {it} iteraciones")
        break

# Calcular el gradiente del potencial (flujo de velocidad)
dy, dx = np.gradient(potential)

# Asignar 0 al gradiente en la barrera para que el flujo no la atraviese
dy[barrier_y_start:barrier_y_end, barrier_x] = 0
dx[barrier_y_start:barrier_y_end, barrier_x] = 0


velocity_x = -dx  # La velocidad es el negativo del gradiente del potencial
velocity_y = -dy

# Crear la grilla de coordenadas para visualizar
x = np.linspace(-lx / 2, lx / 2, nx)
y = np.linspace(-ly / 2, ly / 2, ny)
X, Y = np.meshgrid(x, y)

# Graficar el potencial y las líneas de flujo con streamplot
plt.figure(figsize=(10, 5))

# Subplot para el potencial
plt.subplot(1, 2, 1)
plt.contourf(X, Y, potential, levels=50, cmap='viridis')
plt.colorbar(label="Potential")
plt.title("Potential with Barrier")

# Subplot para las líneas de flujo (streamlines)
plt.subplot(1, 2, 2)
plt.streamplot(X, Y, velocity_x, velocity_y, color=potential, linewidth=1, cmap='viridis')
plt.title("Flow lines with Barrier")

plt.tight_layout()
plt.savefig(f"prueba.jpg", format='jpg', bbox_inches='tight', pad_inches=0)


import numpy as np
import matplotlib.pyplot as plt

# Parámetros
grilla = 50
nx, ny = grilla, grilla  # Tamaño de la grilla
lx, ly = 8, 4  # Dimensiones físicas del dominio
dx, dy = lx / (nx - 1), ly / (ny - 1)  # Espaciado en x e y

# Inicializamos el potencial como una matriz de ceros
potential = np.ones((ny, nx))

# Condiciones de borde: potencial alto en algunos bordes
#potential[nx//2:, -1] = 100     # Borde izquierdo: impermeable (potencial bajo)
#potential[0, :nx//2] = 30     # Borde superior: potencial alto
#potential[:nx//2, -1] = 0     # Borde izquierdo: impermeable (potencial bajo)
#potential[-1, :] = 0    # Borde inferior: impermeable
#potential[:, 0] = 0    # Borde derecho: impermeable (potencial bajo)

potential[0, nx//2:] = 100     # Borde superior: potencial alto
potential[nx//4, :nx//2] = 10     # Borde superior: potencial alto
potential[:nx//4, :nx//2] = 0     # Borde superior: potencial alto
potential[:, -1] = 0     # Borde izquierdo: impermeable (potencial bajo)
potential[-1, :] = 0    # Borde inferior: impermeable
potential[:, 0] = 0    # Borde derecho: impermeable (potencial bajo)
potential[:nx//2, ny//2] = 0    # Borde superior: impermeable

print(potential)

# Resolución iterativa de la ecuación de Laplace
tolerance = 1e-6
max_iterations = 510

for it in range(max_iterations):
    potential_old = potential.copy()
    
    for i in range(ny):
        for j in range(nx):
            # Verificar si estamos en un borde
            if i == 0 or i == ny-1 or j == 0 or j == nx-1:
                # Si estamos en un borde y el potencial es 0, no hacemos nada
                if potential[i, j] == 0:
                    continue

            # Verificar si estamos en la línea vertical de nx//2 y hasta ny//2
            elif j == nx//2 and i < ny//2:
                # Si estamos en la línea vertical, consideramos que es una barrera con potencial 0
                potential[i, j] = 0
                continue

            else:
                # Si estamos en una celda adyacente a un borde, manejamos las actualizaciones evitando las celdas con potencial 0
                suma_potenciales = 0
                vecinos_validos = 0

                # Revisamos el vecino de arriba
                if i > 0 and potential[i-1, j] > 1:
                    suma_potenciales += potential[i-1, j]
                    vecinos_validos += 1

                # Revisamos el vecino de abajo
                if i < ny-1 and potential[i+1, j] > 1:
                    suma_potenciales += potential[i+1, j]
                    vecinos_validos += 1

                # Revisamos el vecino de la izquierda
                if j > 0 and potential[i, j-1] > 1:
                    suma_potenciales += potential[i, j-1]
                    vecinos_validos += 1

                # Revisamos el vecino de la derecha
                if j < nx-1 and potential[i, j+1] >1:
                    suma_potenciales += potential[i, j+1]
                    vecinos_validos += 1

                # Actualizamos el potencial si tenemos al menos un vecino válido
                if vecinos_validos > 0:
                    potential[i, j] = suma_potenciales / vecinos_validos



    
    # Comprobamos la convergencia
    if np.max(np.abs(potential - potential_old)) < tolerance:
        print(f"Convergencia alcanzada en {it} iteraciones")
        break

# Calcular el gradiente del potencial (flujo de velocidad)
dy, dx = np.gradient(potential, dy, dx)

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
plt.title("Potential with Impermeable Boundaries")

# Subplot para las líneas de flujo (streamlines)
plt.subplot(1, 2, 2)
plt.streamplot(X, Y, velocity_x, velocity_y, color=potential, linewidth=1, cmap='viridis')
plt.title("Flow lines with Impermeable Boundaries")

plt.tight_layout()
plt.savefig(f"prueba_impermeable_bordes_adjacentes.jpg", format='jpg', bbox_inches='tight', pad_inches=0)

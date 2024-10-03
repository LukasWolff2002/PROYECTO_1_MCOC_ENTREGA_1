import numpy as np
import matplotlib.pyplot as plt

# Parámetros
grilla = 39
nx, ny = grilla, grilla  # Tamaño de la grilla
lx, ly = 4, 2  # Dimensiones físicas del dominio
dx, dy = lx / (nx - 1), ly / (ny - 1)  # Espaciado en x e y

# Inicializamos el potencial como una matriz de unos
potential = np.ones((ny, nx))
K = np.ones((ny, nx))

# Condiciones de borde: potencial alto en algunos bordes
potential[0, nx//2:] = 20     # Potencial alto en el borde superior derecho
potential[0, :nx//2] = 10  # Segundo potencial alto definido

# Potenciales cero para barreras impermeables
potential[:, -1] = 0     # Borde izquierdo: impermeable (potencial bajo)
K[:, -1] = 0.001     # Borde izquierdo: impermeable (permeabilidad baja)

potential[-1, :] = 0    # Borde inferior: impermeable
K[-1, :] = 0.001    # Borde inferior: impermeable

potential[:, 0] = 0    # Borde derecho: impermeable
K[:, 0] = 0.001    # Borde derecho: impermeable

# Ajustamos permeabilidades muy bajas para impedir flujo a través del borde
K[-1, :] = 0.001    # Fondo impermeable

# Resolución iterativa de la ecuación de Laplace
tolerance = 1e-6
max_iterations = 100

for it in range(max_iterations):
    potential_old = potential.copy()
    
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            if np.isnan(potential[i, j]) or K[i, j] <= 0.001:
                continue  # Ignorar celdas NaN o de permeabilidad baja

            suma_potenciales = 0
            vecinos_validos = 0

            # Vecinos: arriba, abajo, izquierda, derecha
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < ny and 0 <= nj < nx and K[ni, nj] > 0.001 and potential[ni, nj] != 0:
                    suma_potenciales += potential[ni, nj]
                    vecinos_validos += 1

            if vecinos_validos > 0:
                potential[i, j] = suma_potenciales / vecinos_validos

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
plt.streamplot(X, Y, velocity_x, velocity_y, density=2, color='magenta')
plt.title("Flow lines with High Potentials Only")

plt.tight_layout()
plt.show()

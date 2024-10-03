import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

# Generar una grilla rectangular centrada en (0, 0)
def make_grid(Lx=4, Ly=5, nx=50, ny=51):
    x = np.linspace(0, Lx, num=nx) - Lx/2
    y = np.linspace(0, Ly, num=ny) - Ly/2
    X, Y = np.meshgrid(x, y)
    return X, Y

# Crear la matriz de impermeabilidad (1 para permeable, 0 para impermeable)
def create_impermeability_matrix(grid, y_ataguia):
    """Crear una matriz de impermeabilidad."""
    X, Y = grid
    I = np.ones_like(X)  # Todo el dominio es permeable por defecto
    I[Y >= y_ataguia] = 0  # Bloqueamos el flujo por encima de la ataguía
    return I

# Modificar la función de ensamblado para incluir la matriz de impermeabilidad
def assemble_and_solve_with_impermeability(grid, x, y, nx, ny, impermeability):
    """Ensamblar y resolver el flujo potencial con matriz de impermeabilidad."""
    X, Y = grid
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    # Definir el sistema de ecuaciones
    A = np.zeros((X.size, X.size))  # Matriz de coeficientes
    B = np.zeros(X.size)  # Vector lado derecho

    # Mapeo de coordenadas a índices planos
    def get_index(i, j):
        return i * X.shape[1] + j

    # Ensamblar la matriz de ecuaciones tomando en cuenta la impermeabilidad
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            idx = get_index(i, j)
            if impermeability[i, j] == 1:  # Solo trabajar en celdas permeables
                # Si no es un borde, aplicar la discretización de diferencias finitas
                if i > 0 and i < X.shape[0] - 1 and j > 0 and j < X.shape[1] - 1:
                    A[idx, idx] = -4
                    A[idx, get_index(i+1, j)] = 1  # Abajo
                    A[idx, get_index(i-1, j)] = 1  # Arriba
                    A[idx, get_index(i, j+1)] = 1  # Derecha
                    A[idx, get_index(i, j-1)] = 1  # Izquierda
                else:
                    # Borde: condición de Dirichlet (mantener potencial)
                    A[idx, idx] = 1
                    B[idx] = 0  # Potencial en las fronteras

            else:
                # Celda impermeable: no actualizar el potencial
                A[idx, idx] = 1
                B[idx] = 0  # Potencial fijo a 0 en áreas impermeables

    # Resolver el sistema de ecuaciones
    phi = np.linalg.solve(A, B)
    return phi.reshape(X.shape)

# Generar la grilla
grid = make_grid(nx=50, ny=51)
X, Y = grid

# Definir el círculo paramétrico
t = np.linspace(0, 2 * np.pi, num=35, endpoint=False)
x, y = np.cos(t), np.sin(t)
nx, ny = x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2)

# Definir la altura de la ataguía
y_ataguia = 0.5

# Crear la matriz de impermeabilidad
impermeability = create_impermeability_matrix(grid, y_ataguia)

# Resolver el flujo potencial
phi_mapped = assemble_and_solve_with_impermeability(grid, x, y, nx, ny, impermeability)

# Visualizar el potencial y las líneas de flujo
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

# Potencial
m = ax1.pcolormesh(X, Y, phi_mapped, shading='auto')
plt.colorbar(m, ax=ax1, label='Potencial')
ax1.contour(X, Y, phi_mapped, colors='k')
ax1.axis('equal')
ax1.set_title("Potencial con ataguía")

# Gradiente (flujo)
u, v = np.gradient(phi_mapped)
ax2.quiver(X, Y, v, u)
ax2.set_title("Velocidad (gradiente del potencial)")
ax2.axis('equal')

# Líneas de flujo
fig, ax = plt.subplots()
ax.streamplot(X, Y, v, u)
ax.axis('equal')
ax.set_title("Líneas de flujo")

plt.show()

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Definir las dimensiones del rectángulo y los parámetros de la grilla
Lx = 1
Ly = 1
Nx = 10
Ny = 10
nx = Nx + 1
ny = Ny + 1
dx = Lx / Nx
dy = Ly / Ny

# Definir los valores de x e y en la grilla
x = np.arange(0, Nx + 1) * dx
y = np.arange(0, Ny + 1) * dy

# Definir las condiciones de borde (boundary_index)
boundary_index = np.concatenate([
    np.arange(0, nx),  # Bottom
    np.arange(0, ny * nx, nx),  # Left
    np.arange((ny - 1) * nx, ny * nx),  # Top
    np.arange(nx - 1, ny * nx, nx)  # Right
])

# Crear las diagonales
diagonals = [
    4 * np.ones(nx * ny),  # Diagonal principal
    -1 * np.ones(nx * ny - 1),  # Diagonal inferior
    -1 * np.ones(nx * ny - 1),  # Diagonal superior
    -1 * np.ones(nx * ny - nx),  # Diagonal nx posiciones por debajo
    -1 * np.ones(nx * ny - nx)  # Diagonal nx posiciones por encima
]

# Crear la matriz A usando spdiags
A = sp.diags(diagonals, [0, -1, 1, -nx, nx], shape=(nx * ny, nx * ny)).tolil()

# Asignar la matriz identidad a las filas en boundary_index
I = sp.eye(nx * ny).tolil()
A[boundary_index, :] = I[boundary_index, :]

# Convertir A a formato CSR para una resolución eficiente
A = A.tocsr()

# Definir la matriz b y establecer las condiciones de frontera
b = np.zeros((nx, ny))
b[:, 0] = 0 #left
b[0, :] = 0 #top
b[:, ny - 1] = 0 #right
b[nx - 1, :] = 4 * x * (1 - x) #bottom

#voy a probar poner una ataguia
# b[nx//2:, ny//2] = 1
print(b)
b = b.reshape(nx * ny, 1)

# Resolver la ecuación de Laplace usando la eliminación gaussiana
Phi = spla.spsolve(A, b)


# Remodelar Phi a una matriz de tamaño (nx, ny)
Phi = Phi.reshape(nx, ny)


# Generar la cuadrícula para el gráfico
X, Y = np.meshgrid(x, y)

# Definir los niveles de contorno
v = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]


# Graficar los contornos
plt.contour(X, Y, Phi, v, colors='black')
plt.axis('equal')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)
plt.title(r'Titulo', fontsize=14)
plt.savefig(f"prueba.jpg", format='jpg', bbox_inches='tight', pad_inches=0)
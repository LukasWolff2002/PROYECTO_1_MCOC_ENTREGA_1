import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

# Definir la función que genera la grilla
def make_grid(Lx=4, Ly=5, nx=50, ny=51):
    """Generar una grilla rectangular centrada en (0, 0)."""
    x = np.linspace(0, Lx, num=nx) - Lx/2
    y = np.linspace(0, Ly, num=ny) - Ly/2
    X, Y = np.meshgrid(x, y)
    return X, Y

grid = make_grid(nx=25, ny=26)

# Crear etiquetas para definir las zonas donde se bloqueará el flujo
def make_tags_with_ataguia(grid, x, y, y_ataguia):
    """Crear una grilla con etiquetas que definen cada tipo de celda."""
    X, Y = grid
    r = np.arange(X.shape[0])
    c = np.arange(X.shape[1])
    C, R = np.meshgrid(c, r)
    
    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    EPS = 1e-5
    
    # Rasterizamos los puntos del contorno del círculo
    grid_coords = rasterize_points_to_grid_coords(x, y, grid)
    grid_coords_set = set((r, c) for (r, c) in grid_coords)
    
    # Creamos un polígono y un set de coordenadas
    mask = polygon(np.array(grid_coords)[:, 0], np.array(grid_coords)[:, 1], X.shape)
    mask_set = set((r, c) for r, c in zip(*mask))

    # Definimos las etiquetas para la grilla
    tags = np.zeros_like(R, dtype=int) * np.nan

    for r, c in zip(R.flatten(), C.flatten()):
        # Barrera en la parte superior, sobre la ataguía
        if Y[r, c] >= y_ataguia:
            tags[r, c] = 8  # Etiqueta de la ataguía (barrera)
        # Lado izquierdo de la grilla
        elif abs(X[r, c] - xmin) < EPS:
            tags[r, c] = 1
        # Lado derecho de la grilla
        elif abs(X[r, c] - xmax) < EPS:
            tags[r, c] = 2
        # Fondo de la grilla
        elif abs(Y[r, c] - ymin) < EPS:
            tags[r, c] = 3
        # Parte superior de la grilla
        elif abs(Y[r, c] - ymax) < EPS:
            tags[r, c] = 4
        # Interior del dominio debajo de la ataguía
        elif (r, c) in mask_set and Y[r, c] < y_ataguia:
            # Borde del círculo
            if (r, c) in grid_coords_set:
                tags[r, c] = 5
            # Volumen interior
            else:
                tags[r, c] = 6
        else:
            tags[r, c] = 7

    for r, c in grid_coords:
        tags[r, c] = 5
    return tags

# Modificar la función de ensamblado para tener en cuenta la ataguía
def assemble_and_solve_with_ataguia(grid, x, y, nx, ny, tags, y_ataguia):
    """Ensamblar y resolver el flujo potencial con una ataguía horizontal."""
    grid_coords = rasterize_points_to_grid_coords(x, y, grid)
    rasterized_vector_field = rasterize_vectors_to_grid(x, y, nx, ny, grid)

    assert grid_coords.shape[0] == rasterized_vector_field.shape[0]

    rc2normal = {}
    for rc, n in zip(grid_coords, rasterized_vector_field):
        rc2normal[(rc[0], rc[1])] = n

    X, Y = grid
    r = np.arange(X.shape[0])
    c = np.arange(X.shape[1])
    C, R = np.meshgrid(c, r)

    has_unknown = (tags != 6) & (tags != 8)  # Excluir áreas de la ataguía
    rcs = np.c_[R[has_unknown].flatten(), C[has_unknown].flatten()]

    rc2id = {}
    id2rc = {}

    for ind, (r, c) in enumerate(rcs):
        rc2id[(r, c)] = ind
        id2rc[ind] = (r, c)

    A = np.zeros((rcs.shape[0], rcs.shape[0]))
    B = np.zeros((rcs.shape[0],))
    v0 = np.array([1., 0])  # Velocidad en el lado izquierdo
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    for (r, c) in rc2id:
        this_point = rc2id[(r, c)]
        if tags[r, c] == 1:
            id_right = rc2id[(r, c+1)]
            id_left = rc2id[(r, c)]
            A[this_point, id_left] += 1/dx
            A[this_point, id_right] += -1/dx
            B[this_point] = -v0[0]
        elif tags[r, c] == 2:
            id_right = rc2id[(r, c)]
            id_left = rc2id[(r, c-1)]
            A[this_point, id_right] += 1/dx
            A[this_point, id_left] += -1/dx
            B[this_point] = v0[0]
        elif tags[r, c] == 4:
            id_top = rc2id[(r, c)]
            id_bottom = rc2id[(r-1, c)]
            A[this_point, id_top] += 1/dy
            A[this_point, id_bottom] += -1/dy
            B[this_point] = 0
        elif tags[r, c] == 3:
            id_top = rc2id[(r, c)]
            id_bottom = rc2id[(r+1, c)]
            A[this_point, id_top] += 1/dy
            A[this_point, id_bottom] += -1/dy
            B[this_point] = 0
        elif tags[r, c] == 5:
            nxx, nyy = rc2normal[(r, c)]
            id_center = rc2id[(r, c)]
            if (r-1, c) in rc2id:
                id_bottom = rc2id[(r-1, c)]
                A[this_point, id_center] += 1/(dy) * nyy
                A[this_point, id_bottom] += -1/(dy) * nyy
            else:
                id_top = rc2id[(r+1, c)]
                A[this_point, id_center] += -1/(dy) * nyy
                A[this_point, id_top] += 1/(dy) * nyy

            if (r, c+1) in rc2id:
                id_right = rc2id[(r, c+1)]
                A[this_point, id_right] += 1/(dx) * nxx
                A[this_point, id_center] += -1/(dx) * nxx
            else:
                id_left = rc2id[(r, c-1)]
                A[this_point, id_center] += 1/(dx) * nxx
                A[this_point, id_left] += -1/(dx) * nxx
        elif tags[r, c] == 6:
            id_center = rc2id[(r, c)]
            A[this_point, id_center] = 1
            B[this_point] = 0
        else:
            id_bottom = rc2id[(r-1, c)]
            id_top = rc2id[(r+1, c)]
            id_right = rc2id[(r, c+1)]
            id_left = rc2id[(r, c-1)]
            id_center = rc2id[(r, c)]
            A[this_point, id_top] += -1/dy
            A[this_point, id_bottom] += -1/dy
            A[this_point, id_right] += -1/dx
            A[this_point, id_left] += -1/dx
            A[this_point, id_center] += (2/dx + 2/dy)

    phi_sol = np.linalg.solve(A, B)
    phi_mapped = np.zeros_like(tags) * np.nan
    for _id, rc in id2rc.items():
        phi_mapped[rc[0], rc[1]] = phi_sol[_id]
    return phi_mapped

# Definir el círculo paramétrico y la ataguía
t = np.linspace(0, 2 * np.pi, num=35, endpoint=False)
x, y = np.cos(t), np.sin(t)
nx, ny = x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2)
y_ataguia = 0  # Altura de la ataguía

# Generar etiquetas
tags = make_tags_with_ataguia(grid, x, y, y_ataguia)

# Resolver el flujo potencial con la ataguía
phi_mapped = assemble_and_solve_with_ataguia(grid, x, y, nx, ny, tags, y_ataguia)

# Visualizar el potencial y las líneas de flujo
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
m = ax1.pcolormesh(grid[0], grid[1], phi_mapped, shading='Gouraud')
ax1.contour(grid[0], grid[1], phi_mapped)
plt.colorbar(m, ax=ax1)
ax1.axis('equal')
ax1.set_title("Flujo con ataguía")

u, v = np.gradient(phi_mapped)
ax2.quiver(grid[0], grid[1], v, u)
ax2.set_title("Velocidad (gradiente del potencial)")
ax2.axis('equal')

# Streamlines para mostrar el flujo alrededor de la ataguía
fig, ax = plt.subplots()
ax.streamplot(grid[0], grid[1], v, u)
ax.set_title("Líneas de flujo con ataguía")
ax.axis('equal')

plt.show()

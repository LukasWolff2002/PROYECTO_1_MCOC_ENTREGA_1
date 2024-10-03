import numpy as np

def make_grid(Lx=4, Ly=5, nx=50, ny=51):
    """Make a rectangular grid, with coordinates centered on (0, 0)."""
    x = np.linspace(0, Lx, num=nx) - Lx/2
    y = np.linspace(0, Ly, num=ny) - Ly/2
    X, Y = np.meshgrid(x, y)
    return X, Y

grid = make_grid(nx=25, ny=26)


import matplotlib.pyplot as plt

R = np.sqrt(grid[0]**2 + grid[1]**2)

fig, ax = plt.subplots()
m = ax.pcolormesh(grid[0], grid[1], R)
plt.colorbar(m, ax=ax, label='R')
ax.axis('equal')

# define a parametric circle
t = np.linspace(0, 2 * np.pi, num=35, endpoint=False)
x, y = np.cos(t), np.sin(t)
nx, ny = x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2)

fig, ax = plt.subplots()
m = ax.pcolormesh(grid[0], grid[1], R)
plt.colorbar(m, ax=ax, label='R')
ax.plot(x, y, 'ok')
ax.quiver(x, y, nx, ny)
ax.axis('equal')

from skimage import draw

def transform_grid_coords_to_xy(grid_coords, grid):
    """Transform rc coords to xy coords from the grid."""
    X, Y = grid
    x_disc = np.array([X[rc[0], rc[1]] for rc in grid_coords])
    y_disc = np.array([Y[rc[0], rc[1]] for rc in grid_coords])
    return x_disc, y_disc

def rasterize_points_to_grid_coords(x, y, grid):
    """Rasterizes a discrete set of (x, y) points to a grid. Returns (r, c) coordinates."""
    X, Y = grid
    X_flat, Y_flat = X.flatten(), Y.flatten()

    r = np.arange(X.shape[0])
    c = np.arange(X.shape[1])
    C, R = np.meshgrid(c, r)
    R_flat, C_flat = R.flatten(), C.flatten()

    coords = np.c_[X_flat, Y_flat]
    coords_rc = np.c_[R_flat, C_flat]
    
    # first part: discretize each point from x and y while avoiding duplicates
    rcs = []
    discrete_coords = []
    already_seen_points = set()
    for xx, yy in zip(x, y):
        dist = ((coords - np.array([xx, yy]).reshape(1, -1))**2).sum(axis=1)
        argmin = dist.argmin()
        if argmin not in already_seen_points:            
            rc = coords_rc[argmin]
            rcs.append(rc)
            discrete_coords.append(coords[argmin])
            already_seen_points.add(argmin)

    discrete_coords = np.array(discrete_coords)
    x_disc, y_disc = transform_grid_coords_to_xy(rcs, grid)
    
    # sanity check
    assert np.allclose(np.c_[x_disc, y_disc], discrete_coords)

    # second part: use the Bresenham algorithm to discretize each segment
    N = len(rcs)
    grid_coords = []
    for start, stop in zip(range(N), range(1, N+1)):
        if stop >= N:
            stop = stop % N
        r0, c0 = rcs[start]
        r1, c1 = rcs[stop]
        rr, cc = draw.line(r0, c0, r1, c1)
        grid_coords.extend([r,c] for r, c in zip(rr[:-1], cc[:-1]))
        
    return np.array(grid_coords)

grid_coords = rasterize_points_to_grid_coords(x, y, grid)
x_disc, y_disc = transform_grid_coords_to_xy(grid_coords, grid)

fig, ax = plt.subplots()
m = ax.pcolormesh(grid[0], grid[1], np.sqrt(grid[0]**2 + grid[1]**2))
plt.colorbar(m, ax=ax, label='R')
ax.plot(x, y, 'o')
ax.plot(x_disc, y_disc, '-x')
ax.set_title(f'points on curve: {x.size}, points on discrete curve: {x_disc.size}')
ax.axis('equal')

def rasterize_vectors_to_grid(x, y, nx, ny, grid):
    """Rasterizes a discrete set of (x, y) points and vectors (nx, ny) to a grid. 
    Returns (nx_disc, ny_disc) vector field."""

    X, Y = grid
    X_flat, Y_flat = X.flatten(), Y.flatten()

    r = np.arange(X.shape[0])
    c = np.arange(X.shape[1])
    C, R = np.meshgrid(c, r)
    R_flat, C_flat = R.flatten(), C.flatten()

    coords = np.c_[X_flat, Y_flat]
    coords_rc = np.c_[R_flat, C_flat]
    
    # first part: discretize each point from x and y while avoiding duplicates
    rcs = []
    discrete_coords = []
    discrete_vectors = []
    already_seen_points = set()
    for xx, yy, nxx, nyy in zip(x, y, nx, ny):
        dist = ((coords - np.array([xx, yy]).reshape(1, -1))**2).sum(axis=1)
        argmin = dist.argmin()
        if argmin not in already_seen_points:            
            rc = coords_rc[argmin]
            rcs.append(rc)
            discrete_coords.append(coords[argmin])
            discrete_vectors.append([nxx, nyy])
            already_seen_points.add(argmin)

    discrete_coords = np.array(discrete_coords)
    x_disc, y_disc = transform_grid_coords_to_xy(rcs, grid)
    
    # sanity check
    assert np.allclose(np.c_[x_disc, y_disc], discrete_coords)

    # second part: use the Bresenham algorithm to discretize each segment
    N = len(rcs)
    grid_coords = []
    rasterized_vector_field = []
    for start, stop in zip(range(N), range(1, N+1)):
        if stop >= N:
            stop = stop % N
        r0, c0 = rcs[start]
        r1, c1 = rcs[stop]
        nx0, ny0 = discrete_vectors[start]
        nx1, ny1 = discrete_vectors[stop]
        
        rr, cc = draw.line(r0, c0, r1, c1)
        N_interp = len(rr) 
        for i in range(N_interp - 1):
            alpha = i/N_interp
            grid_coords.append([rr[i], cc[i]])
            rasterized_vector_field.append([(1 - alpha) * nx0 + alpha * nx1,
                                            (1 - alpha) * ny0 + alpha * ny1])
    return np.array(rasterized_vector_field)

grid_coords = rasterize_points_to_grid_coords(x, y, grid)
x_disc, y_disc = transform_grid_coords_to_xy(grid_coords, grid)
rasterized_vector_field = rasterize_vectors_to_grid(x, y, nx, ny, grid)

assert rasterized_vector_field.shape[0] == x_disc.size

fig, ax = plt.subplots()
m = ax.pcolormesh(grid[0], grid[1], np.sqrt(grid[0]**2 + grid[1]**2))
plt.colorbar(m, ax=ax, label='R')
ax.plot(x, y, 'o')
ax.plot(x_disc, y_disc, '-x')
ax.quiver(x_disc, y_disc, rasterized_vector_field[:, 0], rasterized_vector_field[:, 1])
ax.set_title(f'points on curve: {x.size}, points on discrete curve: {x_disc.size}')
ax.axis('equal')

def make_data_and_plot(n, ax):
    t = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
    x, y = np.cos(t), np.sin(t)
    nx, ny = x / np.sqrt(x**2 + y**2), y / np.sqrt(x**2 + y**2)
    grid_coords = rasterize_points_to_grid_coords(x, y, grid)
    x_disc, y_disc = transform_grid_coords_to_xy(grid_coords, grid)
    rasterized_vector_field = rasterize_vectors_to_grid(x, y, nx, ny, grid)
    # plotting
    ax.plot(x, y, 'o')
    ax.plot(x_disc, y_disc, '-x')
    ax.quiver(x_disc, y_disc, rasterized_vector_field[:, 0], rasterized_vector_field[:, 1])

n_points = [3, 7, 18, 44]

fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))

for ax, n in zip(axes.ravel(), n_points):
    m = ax.pcolormesh(grid[0], grid[1], np.sqrt(grid[0]**2 + grid[1]**2))
    make_data_and_plot(n, ax)
    plt.colorbar(m, ax=ax, label='R')
    ax.axis('equal')



from skimage.draw import polygon


def make_tags(grid, x, y):
    """Create a grid with tags defining each cell's type."""
    X, Y = grid
    r = np.arange(X.shape[0])
    c = np.arange(X.shape[1])
    C, R = np.meshgrid(c, r)

    xmin, xmax = X.min(), X.max()
    ymin, ymax = Y.min(), Y.max()
    EPS = 1e-5

    grid_coords = rasterize_points_to_grid_coords(x, y, grid)
    grid_coords_set = set((r,c) for (r,c) in grid_coords)
    mask = polygon(np.array(grid_coords)[:, 0], np.array(grid_coords)[:, 1], X.shape)
    mask_set = set((r, c) for r, c in zip(*mask))

    tags = np.zeros_like(R, dtype=int) * np.nan

    for r, c in zip(R.flatten(), C.flatten()):
        # left boundary
        if abs(X[r, c] - xmin) < EPS:
            tags[r, c] = 1
        # right boundary
        elif abs(X[r, c] - xmax) < EPS:
            tags[r, c] = 2
        # bottom boundary
        elif abs(Y[r, c] - ymin) < EPS:
            tags[r, c] = 3
        # top boundary
        elif abs(Y[r, c] - ymax) < EPS:
            tags[r, c] = 4
        # interior
        elif (r, c) in mask_set:
            # boundary
            if (r, c) in  grid_coords_set:
                tags[r, c] = 5
            # volume inside
            else:
                tags[r, c] = 6
        else:
            tags[r, c] = 7

    for r,c in grid_coords:
        tags[r, c] = 5
    return tags

tags = make_tags(grid, x, y)

fig, ax = plt.subplots()
m = ax.pcolormesh(grid[0], grid[1], tags, shading='Gouraud')
plt.colorbar(m, ax=ax)
ax.axis('equal')
ax.set_title(f"unique values: {np.unique(tags[~np.isnan(tags)])}")

X, Y = grid
r = np.arange(X.shape[0])
c = np.arange(X.shape[1])
C, R = np.meshgrid(c, r)

has_unknown = np.ones_like(tags, dtype=bool)
rcs = np.c_[R[has_unknown].flatten(), C[has_unknown].flatten()]

# Let's create some mappings to simplify the mapping.

rc2id = {}
id2rc = {}

for ind, (r, c) in enumerate(rcs):
    rc2id[(r, c)] = ind
    id2rc[ind] = (r, c)

# coef matrix
A = np.zeros((rcs.shape[0], rcs.shape[0]))
# rhs vector
B = np.zeros((rcs.shape[0],))
# v0
v0 = np.array([1., 0])
# grid params
dx = 1
dy = 1

for (r, c) in rc2id:
    this_point = rc2id[(r, c)]
    # left edge
    if tags[r, c] == 1:
        id_right = rc2id[(r, c+1)]
        id_left = rc2id[(r, c)]
        A[this_point, id_left] += 1/dx
        A[this_point, id_right] += -1/dx
        B[this_point] = -v0[0]
    # right edge
    elif tags[r, c] == 2:
        id_right = rc2id[(r, c)]
        id_left = rc2id[(r, c-1)]
        A[this_point, id_right] += 1/dx
        A[this_point, id_left] += -1/dx
        B[this_point] = v0[0]
    # top edge
    elif tags[r, c] == 4:
        id_top = rc2id[(r, c)]
        id_bottom = rc2id[(r-1, c)]
        A[this_point, id_top] += 1/dy
        A[this_point, id_bottom] += -1/dy
        B[this_point] = 0
    # bottom edge
    elif tags[r, c] == 3:
        id_top = rc2id[(r, c)]
        id_bottom = rc2id[(r+1, c)]
        A[this_point, id_top] += 1/dy
        A[this_point, id_bottom] += -1/dy
        B[this_point] = 0
    # interior boundary, interior volume and exterior volume
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

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
m = ax1.pcolormesh(grid[0], grid[1], phi_mapped, shading='Gouraud')
ax1.contour(grid[0], grid[1], phi_mapped, linestyles='dotted', colors="black")
plt.colorbar(m, ax=ax1)
ax1.axis('equal')
ax1.set_title("potential")

u, v = np.gradient(phi_mapped)
ax2.quiver(X, Y, v, u)
ax2.set_title("velocity (grad. of potential)")
ax2.axis('equal')

fig, ax = plt.subplots()
ax.streamplot(X, Y, v, u)
ax.axis('equal')

def assemble_and_solve(grid, x, y, nx, ny, tags):
    """Assembles and solves the potential flow from a curve (x, y, nx, ny) and on the given grid."""
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

    has_unknown = (tags != 6)
    rcs = np.c_[R[has_unknown].flatten(), C[has_unknown].flatten()]

    # Let's create some mappings to simplify the loops.

    rc2id = {}
    id2rc = {}

    for ind, (r, c) in enumerate(rcs):
        rc2id[(r, c)] = ind
        id2rc[ind] = (r, c)

    # coef matrix
    A = np.zeros((rcs.shape[0], rcs.shape[0]))
    # rhs vector
    B = np.zeros((rcs.shape[0],))
    # v0
    v0 = np.array([1., 0])
    # grid params
    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]

    for (r, c) in rc2id:
        this_point = rc2id[(r, c)]
        # left edge
        if tags[r, c] == 1:
            id_right = rc2id[(r, c+1)]
            id_left = rc2id[(r, c)]
            A[this_point, id_left] += 1/dx
            A[this_point, id_right] += -1/dx
            B[this_point] = -v0[0]
        # right edge
        elif tags[r, c] == 2:
            id_right = rc2id[(r, c)]
            id_left = rc2id[(r, c-1)]
            A[this_point, id_right] += 1/dx
            A[this_point, id_left] += -1/dx
            B[this_point] = v0[0]
        # top edge
        elif tags[r, c] == 4:
            id_top = rc2id[(r, c)]
            id_bottom = rc2id[(r-1, c)]
            A[this_point, id_top] += 1/dy
            A[this_point, id_bottom] += -1/dy
            B[this_point] = 0
        # bottom edge
        elif tags[r, c] == 3:
            id_top = rc2id[(r, c)]
            id_bottom = rc2id[(r+1, c)]
            A[this_point, id_top] += 1/dy
            A[this_point, id_bottom] += -1/dy
            B[this_point] = 0
        # point on the interior boundary
        elif tags[r, c] == 5:
            nxx, nyy = rc2normal[(r, c)]
            id_center = rc2id[(r, c)]
            # vertical gradient
            if (r-1, c) in rc2id:
                id_bottom = rc2id[(r-1, c)]
                A[this_point, id_center] += 1/(dy) * nyy
                A[this_point, id_bottom] += -1/(dy) * nyy
            else:
                id_top = rc2id[(r+1, c)]
                A[this_point, id_center] += -1/(dy) * nyy
                A[this_point, id_top] += 1/(dy) * nyy

            # horizontal gradient
            if (r, c+1) in rc2id:
                id_right = rc2id[(r, c+1)]
                A[this_point, id_right] += 1/(dx) * nxx
                A[this_point, id_center] += -1/(dx) * nxx
            else:
                id_left = rc2id[(r, c-1)]
                A[this_point, id_center] += 1/(dx) * nxx
                A[this_point, id_left] += -1/(dx) * nxx

        # inside the volume
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

phi_mapped = assemble_and_solve(grid, x, y, nx, ny, tags)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
m = ax1.pcolormesh(grid[0], grid[1], phi_mapped, shading='Gouraud')
ax1.contour(grid[0], grid[1], phi_mapped)
plt.colorbar(m, ax=ax1)
ax1.axis('equal')
ax1.set_title("potential");

u, v = np.gradient(phi_mapped)
ax2.quiver(X, Y, v, u)
ax2.set_title("velocity (grad. of potential)");
ax2.axis('equal')

fig, ax = plt.subplots()
ax.streamplot(X, Y, v, u)
ax.axis('equal')

plt.show()
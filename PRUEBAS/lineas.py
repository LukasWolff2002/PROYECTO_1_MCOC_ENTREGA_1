def agregar_linea_horizontal(ax, y, x_inicio, ancho_linea, color, grosor=2):
    ax.hlines(y, x_inicio, x_inicio + ancho_linea, colors=color, linewidth=grosor)

def agregar_linea_vertical(ax, x, y_inicio, altura, color='black', grosor=3):
    ax.vlines(x, y_inicio, y_inicio + altura, colors=color, linewidth=grosor)
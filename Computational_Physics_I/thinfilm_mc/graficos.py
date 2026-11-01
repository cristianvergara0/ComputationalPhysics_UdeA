"""Figuras reutilizables de los cuadernos."""

import numpy as np


def mapa_calor(ax, matriz, etiquetas_x, etiquetas_y, titulo, cmap="viridis", formato="{:.3f}",
               etiqueta_barra="", vmin=None, vmax=None, negritas=None):
    """Dibuja una matriz [fila, columna] como mapa de calor con el valor impreso en cada celda.

    Las filas van de abajo hacia arriba. ``negritas`` es una matriz booleana opcional que
    marca con negrita las celdas destacadas (por ejemplo, las de interacción apreciable).
    """
    imagen = ax.imshow(matriz, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(matriz.shape[1]))
    ax.set_xticklabels(etiquetas_x)
    ax.set_yticks(range(matriz.shape[0]))
    ax.set_yticklabels(etiquetas_y)
    ax.set_title(titulo)
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            valor = matriz[i, j]
            if not np.isfinite(valor):
                continue
            rgba = imagen.cmap(imagen.norm(valor))
            luminancia = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                j, i, formato.format(valor), ha="center", va="center", fontsize=7.5,
                color="black" if luminancia > 0.55 else "white",
                fontweight="bold" if (negritas is not None and negritas[i, j]) else "normal",
            )
    ax.figure.colorbar(imagen, ax=ax, label=etiqueta_barra)
    return imagen


def linea_rd(ax, matriz, rugosidad_rd):
    """Dibuja la curva de nivel W = W_RD si cae dentro del intervalo de la matriz; devuelve si la dibujó."""
    if np.nanmin(matriz) < rugosidad_rd < np.nanmax(matriz):
        ax.contour(matriz, levels=[rugosidad_rd], colors="white", linestyles="--", linewidths=1.5)
        return True
    return False


def texto_rd(dibujada, rugosidad_rd):
    """Fragmento del título que explica la línea de nivel de RD, o que indica que no cae en el mapa."""
    if dibujada:
        return rf"línea blanca: $W = W_{{RD}}$ = {rugosidad_rd:.3f}"
    return rf"$W_{{RD}}$ = {rugosidad_rd:.3f} queda fuera del intervalo de $W$"

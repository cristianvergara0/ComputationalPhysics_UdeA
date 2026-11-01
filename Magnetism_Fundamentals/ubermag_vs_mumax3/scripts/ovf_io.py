"""Lectura y escritura de archivos OVF 2.0 (el formato de campos de MuMax3 y OOMMF).

MuMax3 escribe por defecto OVF 2.0 binario de 4 bytes: una cabecera de líneas que empiezan con
'#', un valor de control (1234567.0 en 4 bytes, 123456789012345.0 en 8 bytes) y los datos con x
variando más rápido, luego y, luego z. read_ovf devuelve el campo como un arreglo
(nx, ny, nz, valuedim), con el mismo orden de índices que discretisedfield (Field.array).

write_ovf existe para las pruebas de tests/: escribe campos sintéticos con el mismo formato.
"""
from pathlib import Path

import numpy as np

_CHECK = {4: 1234567.0, 8: 123456789012345.0}


def read_ovf(path):
    """Devuelve un arreglo (nx, ny, nz, valuedim) con el campo de un archivo OVF 2.0."""
    raw = Path(path).read_bytes()
    header = {}
    pos = 0
    kind = None
    while pos < len(raw):
        end = raw.index(b"\n", pos)
        line = raw[pos:end].decode("latin-1").strip()
        pos = end + 1
        if line.lower().startswith("# begin: data"):
            kind = line.split(":", 1)[1].strip().lower()   # "data binary 4", "data binary 8" o "data text"
            break
        if line.startswith("#") and ":" in line:
            key, val = line[1:].split(":", 1)
            header[key.strip().lower()] = val.strip()
    if kind is None:
        raise ValueError(f"{path}: no se encontró la sección de datos")
    nx, ny, nz = (int(header[k]) for k in ("xnodes", "ynodes", "znodes"))
    dim = int(header.get("valuedim", 3))
    count = nx * ny * nz * dim
    if kind.endswith("binary 4") or kind.endswith("binary 8"):
        size = 4 if kind.endswith("4") else 8
        dtype = "<f4" if size == 4 else "<f8"
        check = np.frombuffer(raw, dtype, 1, pos)[0]
        if check != _CHECK[size]:
            raise ValueError(f"{path}: valor de control {check}, se esperaba {_CHECK[size]}")
        data = np.frombuffer(raw, dtype, count, pos + size).astype(float)
    elif kind.endswith("text"):
        stop = raw.index(b"# End: Data", pos)
        data = np.array(raw[pos:stop].decode("latin-1").split(), dtype=float)[:count]
    else:
        raise ValueError(f"{path}: formato de datos no reconocido ({kind})")
    if data.size != count:
        raise ValueError(f"{path}: se esperaban {count} valores y hay {data.size}")
    return data.reshape(nz, ny, nx, dim).transpose(2, 1, 0, 3)


def write_ovf(path, field, cell=(5e-9, 5e-9, 5e-9), fmt="binary 4", title="m"):
    """Escribe un campo (nx, ny, nz, valuedim) en OVF 2.0 ("binary 4", "binary 8" o "text")."""
    field = np.asarray(field, dtype=float)
    nx, ny, nz, dim = field.shape
    head = [
        "# OOMMF OVF 2.0", "# Segment count: 1", "# Begin: Segment", "# Begin: Header",
        f"# Title: {title}", "# meshtype: rectangular", "# meshunit: m",
        "# xmin: 0", "# ymin: 0", "# zmin: 0",
        f"# xmax: {nx * cell[0]}", f"# ymax: {ny * cell[1]}", f"# zmax: {nz * cell[2]}",
        f"# valuedim: {dim}", "# valuelabels: m_x m_y m_z", "# valueunits: 1 1 1",
        f"# xbase: {cell[0] / 2}", f"# ybase: {cell[1] / 2}", f"# zbase: {cell[2] / 2}",
        f"# xnodes: {nx}", f"# ynodes: {ny}", f"# znodes: {nz}",
        f"# xstepsize: {cell[0]}", f"# ystepsize: {cell[1]}", f"# zstepsize: {cell[2]}",
        "# End: Header", f"# Begin: Data {fmt}",
    ]
    flat = field.transpose(2, 1, 0, 3).reshape(-1)   # x más rápido, luego y, luego z
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("latin-1"))
        if fmt.startswith("binary"):
            size = int(fmt.split()[1])
            dtype = "<f4" if size == 4 else "<f8"
            f.write(np.array([_CHECK[size]], dtype).tobytes())
            f.write(flat.astype(dtype).tobytes())
            f.write(b"\n")
        else:
            f.write(("\n".join(" ".join(f"{v:.17g}" for v in row) for row in flat.reshape(-1, dim)) + "\n").encode())
        f.write(f"# End: Data {fmt}\n# End: Segment\n".encode("latin-1"))

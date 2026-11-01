| Código | Celdas | Mediana (s) | Mín. (s) | Máx. (s) |
|---|---:|---:|---:|---:|
| Ubermag+OOMMF: md.drive | 72 | 6.116 | 6.08 | 6.19 |
| Ubermag+OOMMF: md.drive | 576 | 6.145 | 6.11 | 6.20 |
| Ubermag+OOMMF: md.drive | 4608 | 6.214 | 6.20 | 6.31 |
| Ubermag+OOMMF: md.drive | 15552 | 6.230 | 6.18 | 6.23 |
| Ubermag+OOMMF: md.drive | 36864 | 6.318 | 6.28 | 6.33 |
| Ubermag+OOMMF: md.drive | 72000 | 6.482 | 6.42 | 6.56 |
| MuMax3: proceso con Minimize() | 72 | 0.127 | 0.12 | 0.13 |
| MuMax3: proceso con Minimize() | 576 | 0.127 | 0.12 | 0.13 |
| MuMax3: proceso con Minimize() | 4608 | 0.129 | 0.13 | 0.13 |
| MuMax3: proceso con Minimize() | 15552 | 0.143 | 0.14 | 0.16 |
| MuMax3: proceso con Minimize() | 36864 | 0.163 | 0.16 | 0.17 |
| MuMax3: proceso con Minimize() | 72000 | 0.199 | 0.20 | 0.21 |
| MuMax3: proceso sin Minimize() | 72 | 0.110 | 0.11 | 0.11 |
| MuMax3: proceso sin Minimize() | 576 | 0.116 | 0.11 | 0.12 |
| MuMax3: proceso sin Minimize() | 4608 | 0.128 | 0.12 | 0.14 |
| MuMax3: proceso sin Minimize() | 15552 | 0.128 | 0.13 | 0.20 |
| MuMax3: proceso sin Minimize() | 36864 | 0.151 | 0.15 | 0.24 |
| MuMax3: proceso sin Minimize() | 72000 | 0.193 | 0.18 | 0.20 |
| MuMax3: Minimize() (diferencia) | 72 | 0.017 |  |  |
| MuMax3: Minimize() (diferencia) | 576 | 0.010 |  |  |
| MuMax3: Minimize() (diferencia) | 4608 | 0.000 |  |  |
| MuMax3: Minimize() (diferencia) | 15552 | 0.015 |  |  |
| MuMax3: Minimize() (diferencia) | 36864 | 0.013 |  |  |
| MuMax3: Minimize() (diferencia) | 72000 | 0.006 |  |  |

Ajuste t = t0 + c N (mínimos cuadrados sobre las medianas):

| Código | t0 (s) | c (µs por celda) |
|---|---:|---:|
| Ubermag+OOMMF: md.drive | 6.151 ± 0.015 | 4.636 ± 0.438 |
| MuMax3: proceso con Minimize() | 0.126 ± 0.001 | 1.019 ± 0.022 |
| MuMax3: Minimize() (diferencia) | 0.011 ± 0.004 | -0.044 ± 0.106 |

Entorno: python=3.12.14; so=Windows-11-10.0.26300-SP0; discretisedfield=0.92.0; micromagneticmodel=0.65.0; oommfc=0.66.0; gpu=NVIDIA GeForce RTX 5060 Laptop GPU, 621.01
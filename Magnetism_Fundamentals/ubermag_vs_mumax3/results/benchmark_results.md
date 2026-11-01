| Caso | Enfoque | Mediana (s) | Min (s) | Max (s) |
|---|---|---:|---:|---:|
| micromagnetic_energy_terms_skyrmion | Ubermag+OOMMF (notebook completo) | 89.76 | 89.70 | 91.27 |
| micromagnetic_energy_terms_skyrmion | MuMax3 (script completo) | 3.48 | 3.37 | 3.58 |
| torque_magnetometry_uniaxial_box | Ubermag+OOMMF (notebook completo) | 470.40 | 470.03 | 471.19 |
| torque_magnetometry_uniaxial_box | MuMax3 (script completo) | 1.49 | 1.48 | 1.57 |

Factor Ubermag/MuMax3 en micromagnetic_energy_terms_skyrmion: 25.8x

Factor Ubermag/MuMax3 en torque_magnetometry_uniaxial_box: 316.0x

Entorno: python=3.12.14; so=Windows-11-10.0.26300-SP0; cpu=Intel(R) Core(TM) i7-14650HX; cpu_id=Intel64 Family 6 Model 183 Stepping 1, GenuineIntel; ubermag=0.1; discretisedfield=0.92.0; micromagneticmodel=0.65.0; oommfc=0.66.0; mag2exp=0.64.0; gpu=NVIDIA GeForce RTX 5060 Laptop GPU, 621.01
MuMax3: //mumax 3.12 [windows_amd64 go1.26.0(gc) CUDA-12.9]
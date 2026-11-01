# Manual de laboratorio

Guía para medir y analizar las oscilaciones de un imán colgado de un resorte sobre una bobina. Cubre el montaje, la toma de datos con un ESP32 y el uso de `spring_magnet_analyzer.py`.

1. [Qué se mide](#1-qué-se-mide)
2. [Material](#2-material)
3. [Montaje y cableado](#3-montaje-y-cableado)
4. [Programar el ESP32](#4-programar-el-esp32)
5. [Instalar el programa](#5-instalar-el-programa)
6. [Calibrar el ADC](#6-calibrar-el-adc)
7. [Medir](#7-medir)
8. [Configurar y ejecutar](#8-configurar-y-ejecutar)
9. [Leer los resultados](#9-leer-los-resultados)
10. [Límites del método y cómo reducirlos](#10-límites-del-método-y-cómo-reducirlos)
11. [Problemas frecuentes](#11-problemas-frecuentes)
12. [Formato de los archivos](#12-formato-de-los-archivos)

## 1. Qué se mide

El imán oscila en vertical. Mientras se mueve, el flujo magnético $\Phi$ que atraviesa la bobina de $N$ vueltas cambia y, por la ley de Faraday, entre sus terminales aparece una fem

$$\varepsilon(t) = -N\,\frac{d\Phi}{dt}.$$

Del registro de $\varepsilon(t)$ el programa extrae tres cosas:

- La frecuencia dominante $f_0$ del espectro. Si cada ciclo produce un solo pulso positivo, es la frecuencia de oscilación del imán; si el equilibrio del imán está en el plano de la bobina, no lo es (sección 10).
- La tasa de decaimiento de la amplitud $\gamma$, con $A(t) = A_0\,e^{-\gamma t}$, a partir de cómo decrecen los pulsos.
- La integral de la fem de un lóbulo, $\int \varepsilon\,dt = N\,\Delta\Phi$, con $\Delta\Phi$ el cambio de flujo entre los dos extremos del recorrido del imán en ese semiciclo. Con el número de vueltas y el radio $R$ de la bobina se obtiene $B_\text{eq} = |\int \varepsilon\,dt| \,/\, (N \pi R^2)$, la variación del campo medio sobre la bobina entre esos dos extremos. $B_\text{eq}$ crece con la amplitud de la oscilación y no es el campo del imán en un punto.

Para un oscilador con amortiguamiento viscoso, $m\ddot{x} + b\dot{x} + kx = 0$, se cumple $\gamma = b/(2m)$, $\omega_0^2 = k/m$ y $\omega_1^2 = \omega_0^2 - \gamma^2$, con $\omega_1 = 2\pi f_0$ la frecuencia angular medida. Las medidas de $f_0$ y $\gamma$ dan dos ecuaciones para tres incógnitas ($m$, $k$, $b$): si se conoce una, se obtienen las otras dos.

La medición no distingue la causa de las pérdidas. $\gamma$ incluye la resistencia del aire, la fricción interna del resorte y cualquier frenado por corrientes inducidas en la bobina.

El notebook de `notebooks/` deduce estas relaciones, explica cada paso del análisis y lo comprueba con registros simulados.

## 2. Material

- Resorte, soporte rígido e imán.
- Bobina de la que se conozca el número de vueltas y el radio.
- ESP32 y cable USB.
- Computador con Python 3.9 o superior.
- Balanza (g), regla o calibre (mm) y multímetro.
- Fuente de tensión continua variable, o resistencias para armar un divisor de tensión, para calibrar el ADC.
- Osciloscopio (recomendado).

Valores aproximados del montaje del proyecto: imán anular de 1 cm de grosor, 4 cm de radio exterior y 2 cm de radio interior; bobina de unas 3400 a 3500 vueltas y 41.5 mm de radio exterior. Mide los de tu montaje.

## 3. Montaje y cableado

1. Fija la bobina sobre la mesa con su eje vertical.
2. Sujeta el resorte a un soporte rígido y cuelga el imán de él, de modo que oscile en vertical sobre el centro de la bobina sin tocarla.
3. Conecta un terminal de la bobina al pin analógico del ESP32 y el otro a GND.

Antes de conectar la bobina al ESP32, mide la fem con un osciloscopio. En el montaje del proyecto los picos eran de unas décimas de voltio. Las entradas del ESP32 solo admiten tensiones dentro de un rango estrecho alrededor de 0 a 3.3 V (los límites exactos están en la hoja de datos de tu módulo, en las características eléctricas, *Absolute Maximum Ratings*), y una fem negativa de más de unas décimas de voltio o una positiva de varios voltios puede dañar el pin. Si la tuya lo supera, intercala una resistencia en serie de 1 kΩ o más y diodos de pinza a GND y a 3.3 V, o usa el circuito de polarización de la sección 6. Un divisor de tensión reduce la amplitud, pero no protege del lado negativo.

El sketch lee el pin que define `analogInPin` (2 en el sketch incluido). Comprueba en el esquema de pines de tu placa que sea una entrada del ADC. En el ESP32 clásico, GPIO2 pertenece al ADC2, que no se puede usar mientras el Wi-Fi está activo (el sketch no lo activa), y además es un pin de arranque (*strapping*) según la hoja de datos del módulo. Los GPIO 32 a 39 pertenecen al ADC1, que no tiene esas limitaciones: con ese chip es preferible usar uno de ellos y cambiar `analogInPin`. En los ESP32-S2, S3 y C3 los canales del ADC están en otros pines. Los intervalos de tensión del ADC que cita la sección 6 son los de la guía de Espressif para el ESP32 clásico; en las otras variantes son distintos, y conviene consultar la guía de tu chip.

Espressif indica que el ADC del ESP32 es sensible al ruido y sugiere conectar un capacitor cerámico de 100 nF a la entrada del ADC.

El ADC no mide tensiones negativas: lee 0 cuando la fem es negativa. Con esta conexión directa se registra solo el lóbulo positivo de cada ciclo. La sección 6 explica cómo registrar los dos.

## 4. Programar el ESP32

1. Instala el Arduino IDE y el soporte para placas ESP32.
2. Abre `firmware/esp32_adc_serial/esp32_adc_serial.ino`. El archivo debe quedar dentro de una carpeta con el mismo nombre.
3. Elige la placa y el puerto, y sube el sketch.
4. Abre el monitor serie a 115200 baudios. Deben aparecer unas 1000 lecturas por segundo, enteros entre 0 y 4095.
5. Cierra el monitor serie. Solo un programa a la vez puede abrir el puerto, y si el monitor sigue abierto el análisis no podrá conectarse.

El sketch no fija la frecuencia de muestreo con un temporizador, sino con `delay(1)`. El programa de análisis la estima como número de muestras entre duración de la captura y la guarda con el registro.

## 5. Instalar el programa

```
pip install -r requirements.txt
pip install pyserial
python spring_magnet_analyzer.py --version
python -m unittest discover -s tests
```

`pyserial` solo hace falta para adquirir datos con el ESP32. Para analizar archivos y para la demo no se necesita. `--version` muestra las versiones de Python y de las bibliotecas, útil al describir un problema.

## 6. Calibrar el ADC

El programa convierte cuentas en voltios con $V_\text{scale} = V_\text{ref} / 4095$ y $V_\text{ref} = 3.3$ V. Ese factor es nominal. Según Espressif, la tensión de referencia del ADC es de 1100 mV por diseño, pero puede estar entre 1000 y 1200 mV según el chip. $V_\text{scale}$ afecta a $B_\text{eq}$ y a la integral de la fem, pero no a $f_0$, $\gamma$, $k$ ni $b$.

### Calibración de un punto

1. Aplica a la entrada del ADC una tensión continua estable del orden de la señal (décimas de voltio), con una fuente o con un divisor de tensión.
2. Mídela con el multímetro. Es `v_cal_V`.
3. Registra unos segundos con el ESP32 y promedia las lecturas. Es `adc_cal_cuentas`. Con `"tipo": "serie"` el registro queda guardado en `resultados/<fecha_hora>/medicion_raw_<fecha_hora>.csv` aunque el análisis falle por no haber oscilación; se promedia la columna `adc`.
4. Escribe ambos valores en la sección `adc` de `config.json`. El programa usará `v_cal_V / adc_cal_cuentas` como $V_\text{scale}$.

Una calibración de un punto supone que el ADC lee 0 con 0 V y que la lectura es proporcional a la tensión. Fija la pendiente, pero si el ADC tiene un desplazamiento o una zona muerta, o su respuesta no es lineal, la escala sale sesgada. En ese caso usa varias tensiones.

### Calibración con varias tensiones y linealidad

Repite el procedimiento anterior con 4 a 8 tensiones entre 0.15 V y el máximo de tu señal, y escribe cada par en `adc.puntos_cal`, con la tensión en voltios y la lectura media en cuentas:

```json
"puntos_cal": [[0.20, 248], [0.30, 372], [0.40, 496], [0.50, 620]]
```

Los números son un ejemplo. El programa ajusta una recta, tensión = pendiente × cuentas + ordenada, y usa la pendiente como $V_\text{scale}$. La ordenada es la tensión que el ADC no registra: si es mayor que 10 mV, el programa avisa de una posible zona muerta. Si los puntos se apartan de la recta más del 2 % del rango, avisa de que la respuesta no es lineal. Usa esta forma o la de un punto, no las dos. Si las lecturas son 0 hasta cierta tensión, o la pendiente cambia, el ADC no responde linealmente en ese tramo.

Esto importa porque la señal es pequeña. Según la documentación de Espressif, el ADC da sus mejores resultados dentro de un intervalo de tensión recomendado que, con la atenuación por defecto de Arduino (11 dB), va de 150 a 2450 mV, y con 0 dB va de 100 a 950 mV. Con pulsos de 250 mV, buena parte de cada lóbulo queda por debajo de 150 mV.

En el simulador, una zona muerta del ADC (tensiones bajo un umbral leídas como 0) con pulsos de 250 mV produce este error:

| Zona muerta | $B_\text{eq}$ respecto al valor real | $\gamma$ respecto al valor real | ¿Avisa el programa? |
|---|---|---|---|
| 0 mV | 1.00 | 0.99 | no hace falta |
| 20 mV | 0.87 | 1.23 | no |
| 50 mV | 0.69 | 1.77 | sí, una advertencia |
| 100 mV | 0.43 | 3.45 | sí, dos advertencias |

Una zona muerta de solo 20 mV, menos de la décima parte del pico, ya sesga $\gamma$ en un 23 % sin que el programa lo detecte.

### Si la respuesta no es lineal

Lo más eficaz es llevar la señal al intervalo recomendado del ADC con un circuito que le sume una tensión continua y, si hace falta, la amplifique. Una opción es sumar 1.65 V, el centro del intervalo de 0 a 3.3 V, con un capacitor que bloquee la componente continua de la bobina. Ese circuito no se probó en este proyecto. Con él se digitalizan los dos lóbulos de cada ciclo, y el programa lo admite sin cambios: reconoce la señal bipolar por el tamaño de su parte negativa, toma la mediana de las lecturas como nivel de reposo y analiza los lóbulos de ambas polaridades.

La línea comentada del sketch, `analogSetAttenuation(ADC_0db)`, cambia el intervalo recomendado a 100 - 950 mV. Mejora la resolución, pero el límite inferior sigue siendo 100 mV.

## 7. Medir

1. Anota el número de vueltas y el radio de la bobina. Pesa el cuerpo que oscila (imán y soporte, sin el resorte) y, si puedes, el resorte aparte. Como comprobación, mide también la constante estática: cuelga el cuerpo, mide el alargamiento $\Delta x$ y calcula $k = mg/\Delta x$ con $g = 9.81$ m/s². Si escribes esa $k$ junto con la masa, el programa compara ambos datos con la medición.
2. Con el ESP32 conectado y el imán quieto, las lecturas deben quedar en valores bajos y estables. Si hay saltos grandes, revisa la conexión y la tierra común.
3. En `config.json`, pon `"tipo": "serie"` y, en `fuente.duracion_s`, los segundos que debe durar la captura (40 por defecto). El programa lee datos del puerto hasta que se cumple ese tiempo; con el sketch incluido llegan unas mil lecturas por segundo, algo menos en la práctica, y la frecuencia de muestreo se calcula como lecturas entre segundos de captura.
4. Ejecuta el programa. Abrir el puerto reinicia la mayoría de las placas, así que primero espera un segundo a que el ESP32 arranque y después avisa con una cuenta regresiva (`fuente.cuenta_regresiva_s`, 5 s por defecto). Sostén el imán quieto en vertical y, cuando aparezca «Capturando», suéltalo desde el reposo con una amplitud pequeña. Evita que se balancee de lado: espera unos segundos a que se amortigüe antes de empezar.
5. La captura debe durar lo bastante para que la amplitud baje al menos una tercera parte, es decir, $\gamma T \geq 0.4$ con $T$ la duración. Con $\gamma \approx 0.022$ s⁻¹, son unos 18 s; los 40 s por defecto dan margen. El programa exige al menos 5 ciclos y avisa si $\gamma T < 0.1$, es decir, si la amplitud baja menos de un 10 % aproximadamente.
6. Haz tres registros y compara $\gamma$. Si difieren mucho entre sí, el resultado no es fiable. $B_\text{eq}$ solo es comparable entre registros soltados desde la misma amplitud, porque crece con ella.
7. Anota la fecha, los datos del montaje, la amplitud inicial aproximada y cualquier incidencia.

La amplitud pequeña importa para $\gamma$ (sección 10): cuanto más se acerca el imán a la bobina, más cambia la forma de los pulsos a medida que la oscilación decae.

## 8. Configurar y ejecutar

Todo se define en `config.json`. Las claves que empiezan con guion bajo son comentarios. Los números van sin comillas y con punto decimal. En Windows, las rutas se escriben con barras normales (`C:/datos/ensayo.csv`). Una clave mal escrita produce un error que propone la correcta.

Se ejecuta con:

```
python spring_magnet_analyzer.py
```

Otras formas de llamarlo:

| Orden | Efecto |
|---|---|
| `python spring_magnet_analyzer.py --config otro.json` | usa otro archivo de configuración |
| `python spring_magnet_analyzer.py --crear-config` | crea un `config.json` de ejemplo; no sobrescribe uno existente |
| `python spring_magnet_analyzer.py --version` | muestra las versiones del programa y de las bibliotecas |

### Claves de `config.json`

| Sección y clave | Unidad | Descripción | Por defecto |
|---|---|---|---|
| `fuente.tipo` | | `demo` (datos simulados), `archivo` (registro guardado) o `serie` (adquirir con el ESP32) | `demo` |
| `fuente.archivo` | ruta | Registro `medicion_raw_<fecha>.csv`. Obligatorio con `archivo`. Las rutas relativas se cuentan desde la carpeta del `config.json` | |
| `fuente.fs_Hz` | Hz | Frecuencia de muestreo. Solo hace falta si junto al registro no está su `meta_<fecha>.json` | |
| `fuente.puerto` | | Puerto serie (por ejemplo `COM3` en Windows o `/dev/ttyUSB0` en Linux). Con `null` se usa el único puerto serie disponible | `null` |
| `fuente.baudrate` | baudios | Debe coincidir con `Serial.begin()` del sketch | 115200 |
| `fuente.duracion_s` | s | Duración del registro: el tiempo de captura en modo `serie` y la duración de la señal en modo `demo`. Entre 1 y 3600 | 40 |
| `fuente.cuenta_regresiva_s` | s | Aviso previo a la captura en modo `serie`, para preparar el imán; 0 lo quita | 5 |
| `adc.v_ref_V` | V | Referencia nominal del ADC | 3.3 |
| `adc.v_cal_V` | V | Tensión continua de calibración | |
| `adc.adc_cal_cuentas` | cuentas | Lectura media con esa tensión. `v_cal_V` y `adc_cal_cuentas` van juntos o ninguno | |
| `adc.puntos_cal` | | Calibración con varias tensiones: lista de pares `[tensión en V, cuentas]`, al menos dos. No se combina con `v_cal_V` | |
| `montaje.vueltas` | | Vueltas de la bobina | |
| `montaje.radio_mm` | mm | Radio de las vueltas de la bobina | |
| `montaje.masa_g` | g | Masa que oscila: imán y soporte, sin el resorte | |
| `montaje.masa_resorte_g` | g | Masa del resorte; si se da, la masa efectiva es la masa más un tercio de la del resorte | |
| `montaje.k_N_m` | N/m | Constante del resorte | |
| `montaje.b_kg_s` | kg/s | Coeficiente de amortiguamiento | |
| `analisis.f_min_Hz` | Hz | Se ignora el espectro por debajo de esta frecuencia | 0.2 |
| `analisis.kernel_mediano` | muestras | Ventana del filtro mediano de la envolvente | 101 |
| `analisis.umbral_sigma` | | Se descartan los valores con un residuo mayor que este número de desviaciones típicas en el ajuste de $\gamma$ | 2.5 |
| `analisis.frac_prominencia` | | Prominencia mínima de un pico de la envolvente, como fracción de su máximo | 0.04 |
| `demo.amplitud_mm` | mm | Amplitud inicial de la oscilación simulada | 12 |
| `demo.altura_mm` | mm | Altura del equilibrio sobre la bobina en la simulación | 35 |
| `demo.zona_muerta_mV` | mV | Zona muerta del ADC simulado | 0 |
| `demo.semilla` | | Semilla del ruido simulado | 1 |
| `salida.carpeta` | ruta | Carpeta de resultados | `resultados` |
| `salida.graficas` | | `true` o `false` | `true` |
| `salida.mostrar_graficas` | | Abre además las gráficas en pantalla | `false` |

Los datos de `montaje` que no se conozcan se dejan en `null`. Con datos reales, $k$, $b$ y $B_\text{eq}$ solo se calculan si se escriben los del montaje, y el informe indica lo que falta. En modo demo, lo que quede en `null` se toma del simulador (3400 vueltas, 41.5 mm de radio y 200 g).

### Los tres modos

En modo demo (`"tipo": "demo"`) el programa simula un registro con valores conocidos y compara al final lo calculado con lo verdadero. Sirve para probarlo sin hardware y para ver cómo responden los métodos ante una zona muerta o una amplitud grande (`demo.zona_muerta_mV`, `demo.amplitud_mm`). La duración de la señal simulada es `fuente.duracion_s`.

En modo archivo (`"tipo": "archivo"`) analiza un `medicion_raw_<fecha>.csv`. Lee la frecuencia de muestreo de su `meta_<fecha>.json`, que debe estar en la misma carpeta. Sirve también para repetir el análisis de un registro propio con otros datos del montaje.

En modo serie (`"tipo": "serie"`) conecta con el ESP32, avisa con una cuenta regresiva, captura durante `duracion_s` segundos, guarda el registro crudo y lo analiza. Sin `puerto`, usa el único puerto serie disponible; si hay varios, el error los lista. Si el análisis falla, el registro queda guardado y puede analizarse después en modo archivo.

### Qué se guarda

Cada ejecución crea `resultados/<fecha_hora>/` con:

| Archivo | Contenido |
|---|---|
| `informe_<fecha_hora>.txt` | El informe que se muestra en pantalla |
| `resultados_<fecha_hora>.json` | Los mismos resultados, para procesarlos con otros programas |
| `01_senal.png` | Señal completa y ampliación de los primeros segundos |
| `02_espectro.png` | Espectro de amplitud con $f_0$ y el ancho de la línea |
| `03_decaimiento.png` | Decaimiento de la altura de los pulsos y del área de los lóbulos, con sus ajustes |
| `04_lobulo.png` | El lóbulo usado para la integral de la fem |
| `medicion_raw_<fecha_hora>.csv` y `meta_<fecha_hora>.json` | El registro crudo y su frecuencia de muestreo (modos demo y serie) |
| `config_usada.json` | Copia del `config.json` que produjo estos resultados |

## 9. Leer los resultados

### Qué se calcula según lo que se sepa

Del registro siempre salen $f_0$, $\gamma$ y la integral de la fem. El resto depende de los datos de `montaje`:

| Datos dados | Resultado |
|---|---|
| ninguno | $\omega_0$, factor de calidad $Q = \omega_0/(2\gamma)$, decremento logarítmico $2\pi\gamma/\omega_1$, tiempo de decaimiento $1/\gamma$ |
| `masa_g` | además $k = m\,\omega_0^2$ y $b = 2m\gamma$ |
| `masa_g` y `masa_resorte_g` | lo mismo con la masa efectiva $m + m_\text{resorte}/3$ |
| `k_N_m` | masa efectiva $k/\omega_0^2$ y $b$ |
| `b_kg_s` | masa efectiva $b/(2\gamma)$ y $k$ |
| dos o tres entre masa, $k$ y $b$ | además, una comprobación de consistencia entre los datos dados y la medición (concuerdan si difieren menos del 10 %) |
| `vueltas` | variación de flujo por vuelta $\Delta\Phi$ |
| `vueltas` y `radio_mm` | variación del campo medio $B_\text{eq}$ |

La masa efectiva $m + m_\text{resorte}/3$ es la aproximación habitual para un resorte uniforme cuya masa es pequeña frente a la del cuerpo. Lo que el programa no puede calcular lo lista en la sección «No se pudo calcular», con el dato que falta.

### Las cifras del informe

La incertidumbre de $f_0$ es la mitad del ancho de la línea espectral a media amplitud, que con la ventana de Hann vale $1/T$ incluso para una sinusoide pura ($T$ es la duración del registro). Es una cota conservadora: con un registro de 40 s, el error de $f_0$ por cuantización del espectro es del orden de 0.002 Hz. Esa resolución espectral se propaga a $\omega_0$, $Q$, el decremento logarítmico y $k$; como $k \propto \omega_0^2$, su incertidumbre relativa es el doble de la de $f_0$ (un 3.4 % con $f_0 = 1.45$ Hz y 40 s). Las demás incertidumbres del informe son estadísticas.

El valor de $\gamma$ marcado con `(*)` es el que usan los cálculos siguientes. Sale del decaimiento del área de cada lóbulo cuando hay al menos 6 lóbulos completos; si no, de la altura de los pulsos. La otra estimación se muestra como comprobación. La incertidumbre es el error estándar del ajuste de $\ln A$ contra $t$. Un $R^2$ alto no garantiza que $\gamma$ sea correcta (sección 10).

La integral de la fem es la del lóbulo completo de mayor área, en valor absoluto. Un lóbulo cortado por el inicio o el final del registro se descarta.

$B_\text{eq}$ es $|\int \varepsilon\,dt| / (N\pi R^2)$, la variación del campo medio sobre la bobina entre los dos extremos del recorrido del imán en ese semiciclo. Crece con la amplitud: con el mismo imán, una oscilación de la mitad de amplitud da aproximadamente la mitad de $B_\text{eq}$, así que solo se pueden comparar registros soltados desde la misma amplitud. Su incertidumbre incluye solo el ruido de lectura, estimado dentro del lóbulo, y es mucho menor que las incertidumbres sistemáticas.

### Advertencias

| Aviso | Qué significa y qué hacer |
|---|---|
| Se interpolaron X de N muestras | Hubo líneas ilegibles en el puerto serie. Revisa el cable y la velocidad. |
| Hay lecturas en el fondo de escala del ADC | La señal se satura. Reduce su amplitud. |
| La señal está activa más de la mitad del tiempo | La mediana ya no es el nivel de reposo. Ocurre con pulsos muy anchos en una señal unipolar; con una señal bipolar no se emite. |
| Hay un pico en f0/2 | La frecuencia de oscilación podría ser la mitad de la detectada. Comprueba con el osciloscopio si cada ciclo da uno o dos pulsos. |
| El lóbulo de mayor área dura menos del 75 % de medio periodo | Posible zona muerta del ADC, o el imán cruza el plano de la bobina. Revisa la sección 6. |
| La altura de los pulsos decae más rápido que su área | La forma de los pulsos cambia durante el registro. Usa una amplitud menor. |
| La altura de los pulsos decae más lento que su área | Posible zona muerta del ADC. Revisa la sección 6. |
| El ajuste exponencial es pobre | El decaimiento no es exponencial limpio o hay interferencias. |
| gamma <= 0 | La amplitud no decae en el registro. |
| La amplitud decae solo X % | El registro es corto para estimar $\gamma$. Hazlo más largo. |
| Hay menos de 6 lóbulos completos | $\gamma$ se estima solo con la altura de los pulsos. |
| `montaje.masa_resorte_g` se ignora | Se dio la masa del resorte sin la masa del cuerpo. |
| La recta de calibración corta el eje de tensión en X mV | El ADC no registra los primeros milivoltios (zona muerta). Revisa la sección 6. |
| Los puntos de calibración se apartan hasta X mV de una recta | La respuesta del ADC no es lineal. Revisa la sección 6. |

Que no haya advertencias no demuestra que el resultado sea correcto: la zona muerta pequeña de la sección 6 no se detecta, y tampoco el caso en que el equilibrio del imán está justo en el plano de la bobina (sección 10).

## 10. Límites del método y cómo reducirlos

1. Con la conexión directa, el ADC descarta la fem negativa y solo se registra un lóbulo por ciclo. La integral de un lóbulo es $N$ veces el cambio de flujo entre dos extremos consecutivos del flujo, que es lo que se interpreta como $B_\text{eq}$. Como ese cambio depende de la amplitud de la oscilación, $B_\text{eq}$ no es una propiedad del imán. La mejora de la sección 6 recupera los dos lóbulos.
2. La respuesta del ADC es la fuente de error más importante que el programa no puede corregir. La referencia varía entre chips, la respuesta cerca de 0 V puede no ser lineal y una zona muerta sesga $B_\text{eq}$ y $\gamma$ de forma fuerte (tabla de la sección 6). Los pulsos de 250 mV de los ejemplos quedan en gran parte bajo el intervalo recomendado de 150 mV.
3. $\gamma$ se obtiene de cómo decaen los pulsos, y eso coincide con el decaimiento de la amplitud solo si los pulsos conservan su forma. Si el imán se acerca a la bobina, al disminuir la amplitud el imán se acerca menos y los pulsos cambian de altura más rápido que la amplitud. En 20 simulaciones con un dipolo ideal y distintas alturas y amplitudes, $\gamma$ por altura de pulsos quedó entre +1 % y +38 % sobre el valor real (media +12 %), con $R^2$ superior a 0.99 en todos los casos. Por área de los lóbulos quedó entre −21 % y +18 % (media +3 %). Ninguno de los dos métodos es mejor en todas las geometrías, y los dos sesgos crecen aproximadamente con el cuadrado de la amplitud; la sección 10 del notebook explica por qué. Para reducir el sesgo, mantén la amplitud de oscilación pequeña frente a la distancia entre el imán y la bobina. El simulador trata el imán como un dipolo puntual: con un imán anular de tamaño comparable a su distancia a la bobina, las magnitudes de estos sesgos no se trasladan, aunque su signo sí. La altura de los pulsos se mide sobre la envolvente suavizada de la señal, que cerca de los extremos del registro se aparta del máximo real de cada pulso; el ajuste descarta esos picos como atípicos.
4. $B_\text{eq}$ se calcula como si el campo fuera uniforme sobre el área de la bobina. Si `radio_mm` es el radio exterior de la bobina, el área media de las vueltas es menor y $B_\text{eq}$ queda subestimado. Con un imán anular de tamaño comparable a su distancia a la bobina, el campo no es uniforme sobre ella y $B_\text{eq}$ es solo un valor medio equivalente.
5. $B_\text{eq}$ es proporcional a $1/(N R^2)$, así que los errores en las vueltas y el radio pasan a $B_\text{eq}$. $k$ y $b$ son proporcionales a la masa. Un error del 1 % en la frecuencia de muestreo produce un 2 % en $k$ y un 1 % en $B_\text{eq}$; por eso el programa la estima a partir de la duración de la captura y no usa el valor nominal del sketch.
6. Si el equilibrio del imán está en el plano de la bobina, la fem va como $x\,\dot{x}$ y oscila al doble de la frecuencia mecánica: $f_0$ sale el doble y $k$, cuatro veces mayor. Si el equilibrio queda algo desplazado, aparece el aviso de f0/2; con una alineación casi perfecta, no. Para descartarlo, comprueba con el osciloscopio que cada ciclo da un solo pulso, o escribe la constante $k$ medida de forma estática junto a la masa: la comprobación de consistencia lo detecta.
7. El programa se probó con registros simulados que conocen sus valores y, para la captura por el puerto serie, con un ESP32 simulado. No se ha contrastado con mediciones nuevas ni con el ESP32 conectado.

## 11. Problemas frecuentes

| Mensaje | Causa probable | Qué hacer |
|---|---|---|
| `config.json: error de sintaxis en la línea N` | Coma sobrante, coma decimal, comillas simples o barras invertidas en una ruta | Corrige la línea indicada; el mensaje da una pista |
| `la clave 'montaje.vuletas' no existe. ¿Quisiste decir 'vueltas'?` | Clave mal escrita | Usa el nombre sugerido |
| `No se pudo abrir COM3` | Otro programa tiene el puerto abierto, o el puerto es otro | Cierra el monitor serie; el mensaje lista los puertos detectados |
| `No se detectó ningún puerto serie` | El ESP32 no está conectado o falta el driver de su chip USB-serie | Conéctalo, instala el driver o escribe `fuente.puerto` |
| `Hay varios puertos serie (...)` | Hay más de un dispositivo conectado | Escribe el del ESP32 en `fuente.puerto` |
| `La escala del ADC resulta ... no es razonable` | Pares de `adc.puntos_cal` escritos al revés o con un dato erróneo | Cada par es `[tensión en V, cuentas]` |
| `Para adquirir datos hace falta pyserial` | Falta la biblioteca | `pip install pyserial` |
| `El ESP32 dejó de enviar datos` | Cable suelto, reinicio de la placa o velocidad distinta | Revisa el cable y que `baudrate` coincida con el sketch |
| `No se encontró el meta_<fecha>.json` | El registro está sin su archivo de metadatos | Escribe `fuente.fs_Hz` |
| `No se detecta una oscilación clara` | Señal demasiado pequeña, bobina desconectada o sin tierra común | Revisa el cableado y mira la señal con el osciloscopio |
| `contiene N ciclos; se necesitan al menos 5` | Registro corto | Aumenta `fuente.duracion_s` |
| `No hay ningún lóbulo completo` | La señal nunca vuelve al reposo o el registro es muy corto | Revisa el nivel de reposo y la duración |
| `fs` distinta de 1000 Hz en el informe | El sketch o el puerto van más lentos | Es normal una diferencia pequeña; si es grande, revisa el sketch |

## 12. Formato de los archivos

El registro crudo (`medicion_raw_<fecha>.csv`) tiene dos columnas, `index` y `adc`, con una fila por muestra. `adc` son las cuentas del ADC; una lectura ilegible se guarda como `nan`.

```
index,adc
1,0
2,3
```

Los metadatos (`meta_<fecha>.json`) incluyen `fs_est_Hz`, la frecuencia de muestreo con la que se reconstruye el tiempo; en modo serie, también `puerto`, `baudrate` y `duracion_pedida_s`.

Los resultados (`resultados_<fecha_hora>.json`) tienen los bloques `medido`, `deducido`, `montaje_dado`, `chequeos`, `faltantes` y, en la demo, `comparacion_simulador`. Las unidades van en el nombre de cada clave (`f0_Hz`, `gamma_1_s`, `k_N_m`); `B_T` es el valor de $B_\text{eq}$ en teslas.

## Referencias

Espressif Systems. (s. f.). *Analog to digital converter (ADC)*. ESP-IDF Programming Guide, release v4.4 (ESP32). https://docs.espressif.com/projects/esp-idf/en/release-v4.4/api-reference/peripherals/adc.html

Espressif Systems. (s. f.). *ESP32-WROOM-32 datasheet*. https://www.espressif.com/sites/default/files/documentation/esp32-wroom-32_datasheet_en.pdf

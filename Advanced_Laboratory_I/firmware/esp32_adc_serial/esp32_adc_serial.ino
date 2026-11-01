// Lectura de la fem inducida en la bobina con el ADC del ESP32.
// Envía una lectura por línea (0 a 4095) por el puerto serie, a 115200 baudios y unas 1000 veces por segundo.
// El programa de análisis estima la frecuencia de muestreo real a partir de la duración de la captura.

const int analogInPin = 2;   // Pin de entrada analógica. Debe coincidir con el cableado de la bobina.

void setup() {
  Serial.begin(115200);
  // Arduino usa por defecto ADC_11db, con un intervalo de tensión recomendado de 150 a 2450 mV.
  // Con ADC_0db el intervalo recomendado es de 100 a 950 mV. Para probarlo, quita las barras de la línea siguiente:
  // analogSetAttenuation(ADC_0db);
}

void loop() {
  Serial.println(analogRead(analogInPin));
  delay(1);   // unos 1000 Hz
}

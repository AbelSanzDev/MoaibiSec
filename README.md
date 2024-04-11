# Sistema de Reconocimiento Facial

Este sistema de reconocimiento facial consta de tres partes principales: Almacenamiento de Rostros Locales, Entrenamiento del Modelo de Reconocimiento Facial e Identificación Facial en Tiempo Real.

## Almacenamiento de Rostros Locales

Este script de Python utiliza OpenCV para detectar rostros mediante una cámara web y almacenar las imágenes capturadas localmente en carpetas específicas por usuario.

### Requisitos

- Python 3.x
- OpenCV
- imutils

### Instalación de Dependencias

Para ejecutar este script, necesita instalar OpenCV y imutils. Puede hacerlo mediante pip:

pip install opencv-python imutils

# Uso

- Asegúrese de tener una cámara web conectada a su ordenador.
- Coloque el script en una carpeta y ejecute el script desde la línea de comando o un IDE.
- Ingrese el nombre del usuario cuando se le solicite. Esto creará una carpeta para ese usuario y almacenará allí las imágenes de su rostro.
- Las imágenes de los rostros detectados se guardan en formato .jpg con un número de secuencia como nombre de archivo.

## Características

- Detecta rostros en tiempo real usando la cámara web.
- Almacena las imágenes capturadas en carpetas basadas en el nombre de entrada del usuario.
- Capacidad para almacenar hasta 600 imágenes de rostros por sesión.

## Configuración

El script utiliza un clasificador en cascada Haar para la detección de rostros, configurado para detectar rostros frontales.

## Notas Adicionales

- El script terminará automáticamente después de guardar 600 imágenes o si se presiona la tecla ESC.
- Asegúrese de tener suficiente espacio en disco para almacenar todas las imágenes.

---

# Entrenamiento del Modelo de Reconocimiento Facial

Script para entrenar un modelo de reconocimiento facial usando EigenFaces con imágenes almacenadas localmente.

## Requisitos

- Python 3.x
- OpenCV
- Numpy

## Instalación de Dependencias

Instale las dependencias necesarias con pip:

pip install numpy opencv-python

# Uso

- Organice las imágenes de rostros en subdirectorios dentro de la carpeta Data, nombrados según cada persona.
- Ejecute el script para procesar estas imágenes y entrenar el modelo de reconocimiento facial.
- El modelo entrenado se guarda en el archivo modeloLBPHFace.xml.

## Características

- Utiliza el algoritmo EigenFace para el entrenamiento del modelo.
- Procesa múltiples imágenes y etiquetas asociadas a cada persona.
- Guarda el modelo entrenado para su uso en aplicaciones de reconocimiento facial.

## Configuración

El script lee imágenes en escala de grises para el entrenamiento y las etiquetas correspondientes se generan automáticamente según el directorio.

## Notas Adicionales

- Asegúrese de tener imágenes de buena calidad y bien etiquetadas para un entrenamiento efectivo del modelo.

---

# Identificación Facial en Tiempo Real

Este script utiliza un modelo de reconocimiento facial previamente entrenado para identificar rostros en tiempo real a través de una cámara web.

## Requisitos

- Python 3.x
- OpenCV

# Uso

- Asegúrese de que el modelo modeloLBPHFace.xml esté presente en el directorio del script.
- Conecte una cámara web a su computadora y ejecute el script.
- El script detectará rostros y utilizará el modelo para identificar personas, mostrando el nombre o marcando como desconocido.

## Características

- Identifica rostros en tiempo real utilizando un modelo de reconocimiento facial.
- Muestra información de identificación directamente en el video capturado.
- Funciona con cualquier cámara web compatible con OpenCV.

## Configuración

El script utiliza un clasificador Haar para detectar rostros en las imágenes capturadas por la cámara.

## Notas Adicionales

- La precisión de la identificación puede variar según la calidad del modelo y las condiciones de iluminación.

## Autor

Abel Sanchez Urrea

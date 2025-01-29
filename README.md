# Parpadeo
Programa que mide el parpadeo

## 1. Importación de librerías y configuración inicial

```python
import cv2
import mediapipe as mp
import numpy as np
import os

# Suprimir logs de TensorFlow para evitar advertencias innecesarias
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
```

Se importan las librerías necesarias:
**cv2**: Para el procesamiento de video con OpenCV.
**mediapipe**: Para detectar y rastrear los ojos mediante FaceMesh.
**numpy**: Para cálculos matemáticos y manejo de arrays.
**os**: Para manipular configuraciones del sistema.
Se configuran los logs de TensorFlow para evitar mensajes innecesarios en la consola.

## 2. Cargar video
```python
cap = cv2.VideoCapture(r'D:\UDG\25A Engame\Seminario Analisis del movimiento\Parpadeos.mp4')

if not cap.isOpened():
    print("Error: No se pudo abrir el video.")
    exit()
```
*cv2.VideoCapture(r'RUTA_DEL_VIDEO')*
Carga el video desde la ruta especificada en la computadora.
*cap.isOpened()*
Verifica si el video se abrió correctamente.
Si no se puede abrir (por una ruta incorrecta o un archivo corrupto), el programa muestra un error y **se cierra** *(exit()).*

##  3. Configuración de MediaPipe FaceMesh
```python
mp_face_mesh = mp.solutions.face_mesh

# Índices de los puntos clave de los ojos (según MediaPipe)
index_left_eye = [33, 160, 158, 133, 153, 144]
index_right_eye = [362, 385, 387, 263, 373, 380]

# Parámetros de detección de parpadeo
EAR_THRESHOLD = 0.2  # Umbral para detectar si el ojo está cerrado
CONSEC_FRAMES = 10   # Cuántos frames consecutivos deben cumplir el umbral para contar un parpadeo

# Variables de control
blink_counter = 0  # Contador de parpadeos detectados
frame_counter = 0  # Contador de frames en los que se detecta ojo cerrado
ear_history = []  # Historial de valores EAR para suavizar detección
MAX_HISTORY = 10  # Número de frames que se considerarán para suavizar el EAR
```
**Se inicializa** *mp_face_mesh*, **el modelo de detección de puntos faciales.
Los puntos de los ojos se definen con listas** *(index_left_eye, index_right_eye).*

Estos números corresponden a los **índices de los puntos de referencia** en el modelo FaceMesh.
Se usará para extraer las coordenadas de los ojos.

**Parámetros para detección de parpadeo:**

*EAR_THRESHOLD* = 0.2: Si el valor EAR cae por debajo de 0.2, se considera que el ojo está cerrado.

*CONSEC_FRAMES* = 10: Un parpadeo solo se cuenta si el ojo permanece cerrado durante al menos 10 frames consecutivos.

##  4. Cálculo del EAR (Eye Aspect Ratio)
```python
def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    return (A + B) / (2.0 * C)
```
Esta función calcula el EAR (Eye Aspect Ratio), que mide la proporción de apertura del ojo.
Cómo se calcula el EAR:
A y B: Distancias verticales del ojo.
C: Distancia horizontal del ojo.
**¿Por qué se usa esta métrica?**

Cuando el ojo está **abierto**, el EAR es alto.
Cuando el ojo está **cerrado**, el EAR es bajo.
Si el EAR baja del **umbral** por varios frames, se considera un parpadeo.

## 5. Procesamiento del Video
```python
with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video procesado por completo.")
            break
```
**Se inicializa el modelo FaceMesh** *(mp_face_mesh.FaceMesh()*).
*static_image_mode=False*: Se ejecuta en tiempo real.
*max_num_faces=1*: Solo detecta un rostro.
*refine_landmarks=True*: Mejora la precisión en ojos y labios.
**Se inicia un bucle para procesar cada frame del video.**
Si *cap.read()* no puede leer un frame (porque el video terminó), se detiene el bucle

## 6. Conversión del Frame a RGB y Detección de Ojos
```python
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
```
**El frame se voltea horizontalmente** *(cv2.flip(frame, 1))*
Esto permite que la imagen refleje la vista de un espejo.
**El frame se convierte a RGB** *(cv2.COLOR_BGR2RGB)*
MediaPipe necesita imágenes en formato RGB (OpenCV usa BGR por defecto).
*face_mesh.process(frame_rgb)*
Detecta la cara y devuelve las posiciones de los puntos faciales.

## 7. Mostrar EAR y Número de Parpadeos en Pantalla
```python
                    cv2.putText(frame, f"EAR: {ear_smoothed:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(frame, f"Parpadeos: {blink_counter}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
```
Se usa **cv2.putText**() para dibujar el **EAR y el número de parpadeos** en la imagen.
Se elige un color amarillo *(0, 255, 255)* para que sea visible.

## Finalización y Liberación de Recursos
```python
           
cap.release()
cv2.destroyAllWindows()
```
*cap.release()*: Libera la memoria del video.
*cv2.destroyAllWindows()*: Cierra todas las ventanas de OpenCV.

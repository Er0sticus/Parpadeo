# Parpadeo
Programa que mide el parpadeo

1. Importación de librerías y configuración inicial
import cv2
import mediapipe as mp
import numpy as np
import os

# Suprimir logs de TensorFlow para evitar advertencias innecesarias
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import cv2
import os
import numpy as np

# Ruta al directorio donde se almacenan los datos de los rostros.
dataPath = r"C:\Users\abels\SecureMoaibi\Reconocimiento Facial\Data"
# Lista de subdirectorios en dataPath, cada uno correspondiente a una persona.
peopleList = os.listdir(dataPath)
print("Lista de personas: ", peopleList)

# Inicialización de listas para almacenar etiquetas y datos de los rostros.
labels = []
facesData = []
label = 0

# Recorre cada directorio correspondiente a una persona.
for nameDir in peopleList:
    personPath = dataPath + "/" + nameDir
    print("Leyendo las imágenes")

    # Recorre cada archivo de imagen en el directorio de la persona.
    for fileName in os.listdir(personPath):
        print("Rostros: ", nameDir + "/" + fileName)
        # Añade la etiqueta correspondiente al rostro.
        labels.append(label)
        # Lee la imagen en escala de grises y la añade a la lista de datos de rostros.
        facesData.append(cv2.imread(personPath + "/" + fileName, 0))

    # Incrementa la etiqueta para la siguiente persona.
    label += 1

# Crea un reconocedor de rostros usando EigenFaces.
face_recognizer = cv2.face.EigenFaceRecognizer_create()
# Entrenando el reconocedor de rostros con los datos y etiquetas.
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Guarda el modelo entrenado en un archivo XML.
face_recognizer.write("modeloLBPHFace.xml")
print("Modelo almacenado...")

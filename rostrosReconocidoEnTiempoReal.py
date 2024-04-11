import cv2
import os

# Ruta donde se almacenan los datos de entrenamiento de los rostros.
dataPath = r"C:\Users\abels\SecureMoaibi\Reconocimiento Facial\Data"
# Lista de todos los subdirectorios (que contienen imágenes de rostros).
imagePaths = os.listdir(dataPath)
print("imagePaths=", imagePaths)

# Cargar el modelo de reconocimiento facial previamente entrenado.
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read("modeloLBPHFace.xml")

# Iniciar la captura de video desde la cámara web.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Cargar el clasificador preentrenado para detección de rostros frontales.
faceClassif = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    # Leer un frame de la cámara web.
    ret, frame = cap.read()
    if not ret:
        break
    # Convertir el frame a escala de grises para el procesamiento.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = gray.copy()

    # Detectar rostros en el frame.
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        # Extraer el rostro detectado del frame.
        rostro = auxFrame[y : y + h, x : x + w]
        # Redimensionar el rostro para el reconocimiento.
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
        # Predecir el rostro utilizando el modelo entrenado.
        result = face_recognizer.predict(rostro)

        # Mostrar el resultado de la predicción.
        cv2.putText(
            frame,
            "{}".format(result),
            (x, y - 5),
            1,
            1.3,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Verificar el umbral de confianza del reconocimiento.
        if result[1] < 5700:
            # Si el rostro es reconocido, mostrar el nombre correspondiente.
            cv2.putText(
                frame,
                "{}".format(imagePaths[result[0]]),
                (x, y - 25),
                2,
                1.1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            # Si el rostro no es reconocido, etiquetarlo como desconocido.
            cv2.putText(
                frame, "Desconocido", (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA
            )
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Mostrar el frame en una ventana.
    cv2.imshow("frame", frame)
    # Esperar una tecla para salir.
    k = cv2.waitKey(1)
    if k == 27:
        break

# Liberar la cámara y cerrar todas las ventanas.
cap.release()
cv2.destroyAllWindows()

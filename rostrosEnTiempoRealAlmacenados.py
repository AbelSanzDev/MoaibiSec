import cv2
import os
import imutils

# Directorio donde se guardan las imágenes de los usuarios.
path = os.listdir("Reconocimiento Facial/Data")

print("Nombre de persona")
# Solicita al usuario el nombre de la persona para crear o usar un directorio personalizado.
personName = input()

# Ruta base donde se almacenan los datos.
dataPath = r"C:\Users\abels\SecureMoaibi\Reconocimiento Facial\Data"
# Ruta completa del directorio de la persona.
personPath = os.path.join(dataPath, personName)

# Crea un directorio para la nueva persona si no existe.
if not os.path.exists(personPath):
    print("Carpeta creada:", personPath)
    os.makedirs(personPath)

# Configura la cámara para capturar video.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Clasificadores para detección de rostro y cuerpo.
faceClassif = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
bodyClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
count_face = 0  # Contador de imágenes de rostros capturadas.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensiona el frame a 640 de ancho para un procesamiento más rápido.
    frame = imutils.resize(frame, width=640)
    # Convierte el frame a escala de grises para la detección.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    # Detecta cuerpos en el frame (actualmente no utilizado).
    # bodies = bodyClassif.detectMultiScale(gray, 1.3, 5)

    # Detecta rostros en el frame.
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    # Procesa cada rostro detectado.
    for x, y, w, h in faces:
        # Dibuja un rectángulo alrededor del rostro.
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Extrae y redimensiona el rostro detectado.
        face = auxFrame[y : y + h, x : x + w]
        face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
        # Guarda el rostro en el directorio de la persona.
        cv2.imwrite(personPath + f"/face_{count_face}.jpg", face)
        count_face += 1

    # Muestra el frame en una ventana.
    cv2.imshow("frame", frame)

    # Termina el bucle si el usuario presiona 'ESC' o se alcanza el límite de capturas.
    k = cv2.waitKey(1)
    if k == 27 or count_face >= 600:
        break

# Libera la cámara y cierra todas las ventanas.
cap.release()
cv2.destroyAllWindows()

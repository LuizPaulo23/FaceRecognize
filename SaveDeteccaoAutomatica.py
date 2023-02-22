import cv2
import os
import time

video_capture = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

counter = 0  # Inicializa o contador de imagens
max_images = 1000000  # Define o número máximo de imagens a serem salvas
interval = 1  # Define o intervalo de tempo para salvar cada imagem (em segundos)
start_time = time.time()  # Obtém o tempo atual

while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (30,50)
    )

    for(x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # Verifica se o número máximo de imagens já foi atingido
        if counter >= max_images:
            break

        # Verifica se já passou o intervalo de tempo para salvar a próxima imagem
        if time.time() - start_time >= interval:
            counter += 1
            filename = "detected_face_{:04d}.jpg".format(counter)  # Gera o nome do arquivo com um número sequencial
            cv2.imwrite(filename, frame[y:y+h, x:x+w])  # Salva a imagem
            start_time = time.time()  # Reseta o tempo

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or counter >= max_images:
        break

video_capture.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

classificador = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
classificadorOlho = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")

camera = cv2.VideoCapture(0)
amostra = 1
numeroAmostras = 25
nome = input("Digite seu nome: ")
largura, altura = 220, 220
print("Capturando...")


while(True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    facesDetectadas = classificador.detectMultiScale(imagemCinza, 
                                                        scaleFactor=1.5,
                                                        minSize=(100, 100)
                                                    )
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

        regiao = imagem[y:y + a, x:x + l]
        regiaoCinza = cv2.cvtColor(regiao, cv2.COLOR_RGB2GRAY)
        olhos = classificadorOlho.detectMultiScale(regiaoCinza)

        for (ox, oy, ol, oa) in olhos:
            cv2.rectangle(regiao, (ox, ol), (ox + ol, oy + oa), (0, 255, 0), 2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                if np.average(imagemCinza) > 90:
                    imagemFace = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
                    cv2.imwrite("fotos/pessoa." + str(nome) + "." + str(amostra) + ".jpg", imagemFace)
                    print("Foto capturada com sucesso")
                    amostra += 1

    cv2.imshow("Face", imagem)
    cv2.waitKey(1)
    if(amostra >= numeroAmostras + 1):
        break

camera.release()
cv2.destroyAllWindows()
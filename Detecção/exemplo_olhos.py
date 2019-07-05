import cv2

classificadorFace = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
classificadorOlhos = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')

imagem = cv2.imread("imagens/beatles.jpg")
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detectadas = classificadorFace.detectMultiScale(imagemCinza)

for (x, y, l, a) in detectadas:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
    
    regiao  = imagem[y:y + a, x:x + l]
    regiaoCinza = cv2.cvtColor(regiao, cv2.COLOR_RGB2GRAY)
    olhos = classificadorOlhos.detectMultiScale(regiaoCinza, scaleFactor=1.15, minNeighbors=2)
    for (ox, oy, ol, oa) in olhos:
        cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (255, 0, 255), 2)

        

cv2.imshow("Faces e olhos" ,imagem)

cv2.waitKey()
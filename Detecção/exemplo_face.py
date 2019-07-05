import cv2

classificador = cv2.CascadeClassifier(
    'cascades/haarcascade_frontalface_default.xml')
imagem = cv2.imread('imagens/pessoas3.jpg')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor=1.21, minNeighbors=5, minSize=(40, 40))

for (x, y, l, a) in facesDetectadas:
    cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,0,255), 2)

cv2.imshow("Faces", imagem)
cv2.waitKey()
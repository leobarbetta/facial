import cv2

classificadorCarro = cv2.CascadeClassifier("cascades/cars.xml")

imagem = cv2.imread("imagens/carro3.jpg")
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

detectado = classificadorCarro.detectMultiScale(imagemCinza, scaleFactor=1.05, minNeighbors=4)

for (x, y, l, a) in detectado:
    imagem = cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)

cv2.imshow("Encontrado", imagem)
cv2.waitKey()
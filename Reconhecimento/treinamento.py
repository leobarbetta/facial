import cv2
import os
import numpy as np


eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()


def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []
    for caminho in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminho), cv2.COLOR_BGR2RGB)
        id = int(os.path.split(caminho)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagemFace)
    return np.array(ids), faces

ids, faces =  getImagemComId()
print("Treinando...")

eigenface.train(faces, ids)
eigenface.write("classificadorEigen.yml")

fisherface.train(faces, ids)
fisherface.write("classificadorFisher.yml")

lbph.train(faces, ids)
lbph.write("classificadorLBPH.yml")


print("Treinado!")
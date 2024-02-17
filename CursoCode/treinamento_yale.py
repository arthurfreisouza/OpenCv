import cv2 as cv
import os
import numpy as np
from PIL import Image

eigenface = cv.face.EigenFaceRecognizer_create()
fisherface = cv.face.FisherFaceRecognizer_create()
lbph = cv.face.LBPHFaceRecognizer_create()



def getImagemComId():
    caminhos = [os.path.join('treinamento', f) for f in os.listdir('treinamento')]
    #print(caminhos)
    faces = []
    ids =  []
    for caminhoImagem in caminhos: # Lendo todas as imagens que estão na pasta de fotos.
        imagemFace = Image.open(caminhoImagem).convert('L')# Lendo a imagem e a convertendo para a escala de cinza.
        imagemNP = np.array(imagemFace, 'uint8')
        id = int(os.path.split(caminhoImagem)[1].split(".")[0].replace("subject", ""))
        #print(id)
        ids.append(id)
        faces.append(imagemNP)
        #cv.imshow("Face", imagemFace)
        #cv.waitKey(10)
    return np.array(ids), faces


ids, faces = getImagemComId()
print(ids)
print(len(ids))
print(faces)

# Efetuando o treinamento.
eigenface.train(faces, ids)
eigenface.write('classificadorEigenYale.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisherYale.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPHYale.yml')


print(f' Treinamento realizado...')
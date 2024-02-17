import cv2 as cv
import os
import numpy as np

eigenface = cv.face.EigenFaceRecognizer_create()
fisherface = cv.face.FisherFaceRecognizer_create()
lbph = cv.face.LBPHFaceRecognizer_create()



def getImagemComId():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    #print(caminhos)
    faces = []
    ids =  []
    for caminhoImagem in caminhos: # Lendo todas as imagens que estão na pasta de fotos.
        imagemFace = cv.cvtColor(cv.imread(caminhoImagem), cv.COLOR_BGR2GRAY)# Lendo a imagem,
        id = int(os.path.split(caminhoImagem)[-1].split('.')[1])
        #print(id)
        ids.append(id)
        faces.append(imagemFace)
        #cv.imshow("Face", imagemFace)
        #cv.waitKey(10)
    return np.array(ids), faces


ids, faces = getImagemComId()
print(ids)
print(len(ids))
print(faces)

# Efetuando o treinamento.
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')


print(f' Treinamento realizado...')
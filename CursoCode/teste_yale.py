import cv2 as cv
from PIL import Image
import numpy as np
import os

# Usando o EigenFaces para reconhecer as faces.
detectorFace = cv.CascadeClassifier("haarcascade_frontalface_default.xml") # Um modelo ja treinado para detectar faces.
#reconhecedor = cv.face.EigenFaceRecognizer_create () # Criando uma variável que irá reconhecer faces.
#reconhecedor.read("classificadorEigenYale.yml") # Treinando com a minha face.
#reconhecedor = cv.face.FisherFaceRecognizer_create () # Criando uma variável que irá reconhecer faces.
#reconhecedor.read("classificadorFisherYale.yml") # Treinando com a minha face.
reconhecedor = cv.face.LBPHFaceRecognizer_create () # Criando uma variável que irá reconhecer faces.
reconhecedor.read("classificadorLBPHYale.yml") # Treinando com a minha face.
largura, altura = 220, 220
font = cv.FONT_HERSHEY_COMPLEX_SMALL
camera = cv.VideoCapture(0)
total_acertos = 0
percentual_acertos = 0.0
total_confianca = 0.0



caminhos = [os.path.join('teste', f) for f in os.listdir('teste')]
for caminhoImagem in caminhos: # Lendo todas as imagens que estão na pasta de fotos.
       imagemFace = Image.open(caminhoImagem).convert('L')# Lendo a imagem e a convertendo para a escala de cinza.
       imagemFaceNP = np.array(imagemFace, 'uint8')
       facesDetectadas = detectorFace.detectMultiScale(imagemFaceNP)
       for (x, y, l, a) in facesDetectadas:
           cv.rectangle(imagemFaceNP, (x,y), (x + l, y + a), (0, 0, 255), 2)
           cv.imshow("Face", imagemFaceNP)
           cv.waitKey(1000)

           id_previsto, confianca = reconhecedor.predict(imagemFaceNP)
           id_atual = int(os.path.split(caminhoImagem)[1].split('.')[0].replace("subject", ""))
           print(str(id_atual) + " foi previsto como : " + str(id_previsto) + " - " + str(confianca))
           if id_previsto == id_atual:
                 total_acertos = total_acertos + 1
                 total_confianca = total_confianca + confianca

percentual_acertos = (total_acertos / 30) * 100
total_confianca = total_confianca / total_acertos
print("Percentual de acerto : " + str(percentual_acertos))
print("Total de confiança : " + str(total_confianca)) 
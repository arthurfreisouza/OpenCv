import cv2 as cv
import numpy as np

camera = cv.VideoCapture(0) # Capturando o video.
classificador = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
classificadorOlho = cv.CascadeClassifier("haarcascade_eye.xml")
amostra = 1
numeroAmostras = 25
id = input('Digite seu identificador : ')
largura, altura = 220, 220
print('Capturando as faces ... ')





while True:
    conectado,imagem = camera.read()# Retornará bool e o video.
    imagemCinza = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)

    # Detectando multiplas faces através de 1 classificador já treinado para detectar faces na escala de cinza.
    facesDetectadas = classificador.detectMultiScale(imagemCinza,
                                                     scaleFactor = 1.5,
                                                     minSize = (150, 150))
    #print(np.average(imagemCinza))

    for (x, y, l, a) in facesDetectadas:# A variável faces detectadas conterá 1 retângulo das coordenadas dos rostos encontrados.
        cv.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)# Criando o retangulo que identifica o rosto.
        regiao = imagem[y : y + a, x : x + l]
        regiaoCinzaOlho = cv.cvtColor(regiao, cv.COLOR_BGR2GRAY)
        olhosDetectados = classificadorOlho.detectMultiScale(regiaoCinzaOlho) # Já é 1 classificador treinado com várias imagens de olho.
        for (ox, oy, ol, oa) in olhosDetectados: # Detectando os olhos dentro de 1 cara.
            cv.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255, 0), 2) # Criando o retangulo do olho
            # Salvando as fotos no formato 220 x 220 na pasta fotos apenas se tiver identificado o rosto e o olho...
            if cv.waitKey(1) & 0xFF == ord('q'):
                if np.average(imagemCinza) > 110: # Irei tirar a foto apenas se a foto estiver iluminada o suficiente...
                    imageFace = cv.resize(imagemCinza[y: y + a, x: x + l], (largura, altura))
                    cv.imwrite("fotos/pessoa." + str(id) + "." + str(amostra) + ".jpg", imageFace)
                    print("Foto " + str(amostra) + "capturada com sucesso.")
                    amostra = amostra + 1 
    
    cv.imshow("Face", imagem) # Mostrando a imagem na tela...
    cv.waitKey(1)
    if amostra >= numeroAmostras + 1: # Loop para sair do while.
        break
    
print("Faces capturadas com sucesso.")
camera.release()
cv.destroyAllWindows()
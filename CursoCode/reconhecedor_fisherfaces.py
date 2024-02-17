import cv2 as cv
# Usando o EigenFaces para reconhecer as faces.
detectorFace = cv.CascadeClassifier("haarcascade_frontalface_default.xml") # Um modelo ja treinado para detectar faces.
reconhecedor = cv.face.FisherFaceRecognizer_create () # Criando uma variável que irá reconhecer faces.
reconhecedor.read("classificadorFisher.yml") # Treinando com a minha face.
largura, altura = 220, 220
font = cv.FONT_HERSHEY_COMPLEX_SMALL
camera = cv.VideoCapture(0)

while True:
    conectado, imagem = camera.read()
    imagemCinza = cv.cvtColor(imagem, cv.COLOR_BGR2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza, 
                                                    scaleFactor = 1.5,
                                                     minSize=(30,30))
    
    for (x, y, l, a) in facesDetectadas: # Detectando minha face e a mostrando.
        imagemFace = cv.resize(imagemCinza[y : y + a, x : x + l], (largura, altura))
        cv.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
        id, confianca = reconhecedor.predict(imagemFace)
        nome = None
        if id == 1:
            nome = 'Arthur'
        elif id == 2:
            nome = 'Denise'
            
        if nome is not None:
            cv.putText(imagem, nome, (x, y + (a + 30)), font, 2, (0, 0, 255))
            cv.putText(imagem, str(confianca), (x, y + (a + 50)), font, 1, (0, 0, 255))
    
    cv.imshow("Face", imagem)
    if cv.waitKey(1) == ord('q'):
        break


camera.release()
cv.destroyAllWindows()
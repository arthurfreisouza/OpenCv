import cv2 as cv
import mediapipe as mp # Biblioteca que contém ferramentas e bibliotecas de ML e AI.
import os
import time
import pygame

pygame.mixer.init() # Para a reprodução de audios.
def play_audio(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

video = cv.VideoCapture(0)

hand = mp.solutions.hands # Iniciando o detector de mãos.
Hand = hand.Hands(max_num_hands = 1) # Criando a variável para a detecção de mãos.
mpDraw = mp.solutions.drawing_utils

while True:
    
    check, img = video.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB) # Convertendo imagens para o formato RGB.
    results = Hand.process(imgRGB) # Detectando e processando as mãos na imagem.
    handsPoints = results.multi_hand_landmarks # Os pontos onde há as mãos na imagem.
    h, w, _ = img.shape
    pontos = []
    if handsPoints: # Executará apenas se a variável não estiver vazia.
        time.sleep(0.05)
        for points in handsPoints:
            mpDraw.draw_landmarks(img, points, hand.HAND_CONNECTIONS) # Desenhando as conexões das mãos.
            for id, cord in enumerate(points.landmark):
                cx, cy = int(cord.x * w), int(cord.y * h)
                cv.putText(img, str(id), (cx, cy + 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2) # Colocando os pontos na imagem.
                pontos.append((cx, cy))
        dedos = [8, 12, 16, 20]
        contador = 0
        if points: # Logica das coordenadas.
            if pontos[4][0] < pontos[2][0]:
                contador = contador + 1
            for x in dedos:
                if pontos[x][1] < pontos[x - 2][1]:
                    contador = contador + 1


            if contador == 1:
                caminho_video = "/home/arthur/Desktop/ex_python/OpenCv/audio01"
                if os.path.isfile(caminho_video):
                    play_audio(caminho_video)
                    #time.sleep(2)
                else:
                    raise Exception("There is not a file to open!!")
            elif contador == 2:
                caminho_video = "/home/arthur/Desktop/ex_python/OpenCv/audio02"
                if os.path.isfile(caminho_video):
                    play_audio(caminho_video)
                    #time.sleep(2)
                else:
                    raise Exception("There is not a file to open!!")
            elif contador == 3:
                caminho_video = "/home/arthur/Desktop/ex_python/OpenCv/audio03"
                if os.path.isfile(caminho_video):
                    play_audio(caminho_video)
                    #time.sleep(2)
                else:
                    raise Exception("There is not a file to open!!")
            elif contador == 4:
                caminho_video = "/home/arthur/Desktop/ex_python/OpenCv/audio04"
                if os.path.isfile(caminho_video):
                    play_audio(caminho_video)
                    #time.sleep(2)
                else:
                    raise Exception("There is not a file to open!!")
            elif contador == 5:
                caminho_video = "/home/arthur/Desktop/ex_python/OpenCv/audio05"
                if os.path.isfile(caminho_video):
                    play_audio(caminho_video)
                    #time.sleep(2)
                else:
                    raise Exception("There is not a file to open!!")
            else:
                continue

        cv.rectangle(img, (80, 10), (200, 100), (255, 0, 0), -1)
        cv.putText(img, str(contador), (100, 100), cv.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 5)

        print(contador)   
    cv.imshow("Image", img)
    key = cv.waitKey(1)
    if key == 27: # ESC
            break
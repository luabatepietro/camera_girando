import numpy as np

# Instalar a biblioteca cv2 pode ser um pouco demorado. Não deixe para ultima hora!
import cv2 as cv

def matriz_transformacao(anglo, centro_x, centro_y, cisalhamento):
    theta = np.radians(anglo)

    origem = np.array([
        [1, 0, -centro_x],
        [0, 1, -centro_y],
        [0, 0, 1]
    ])

    rotacao = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1],
    ])

    cisalhament = np.array([
        [1, cisalhamento, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    meio = np.array([
        [1, cisalhamento, centro_x],
        [cisalhamento, 1, centro_y],
        [0, 0, 1]
    ])

    matriz_trans = np.dot(meio, np.dot(cisalhament, np.dot(rotacao, origem)))

    return matriz_trans

def aplica_transformacao(image, matriz_trans):
    
    altura, largura = image.shape[:2] 
    
    imagem_transformada = np.zeros_like(image)
    
    for y in range(altura):  
        for x in range(largura):  

            pixel = np.array([x, y, 1]) 
            
            nova_pos = np.dot(matriz_trans, pixel)
            
            novo_x, novo_y = int(nova_pos[0]), int(nova_pos[1])
            
            if 0 <= novo_x < largura and 0 <= novo_y < altura:
                imagem_transformada[novo_y, novo_x] = image[y, x]
    
    return imagem_transformada


def run():
    # Essa função abre a câmera. Depois desta linha, a luz de câmera (se seu computador tiver) deve ligar.
    cap = cv.VideoCapture(0)

    # Aqui, defino a largura e a altura da imagem com a qual quero trabalhar.
    # Dica: imagens menores precisam de menos processamento!!!
    width = 320
    height = 240
    anglo = 0
    cisalhamento = 0.0

    # Talvez o programa não consiga abrir a câmera. Verifique se há outros dispositivos acessando sua câmera!
    if not cap.isOpened():
        print("Não consegui abrir a câmera!")
        exit()

    # Esse loop é igual a um loop de jogo: ele encerra quando apertamos 'q' no teclado.
    while True:
        # Captura um frame da câmera
        ret, frame = cap.read()

        # A variável `ret` indica se conseguimos capturar um frame
        if not ret:
            print("Não consegui capturar frame!")
            break

        # Mudo o tamanho do meu frame para reduzir o processamento necessário
        # nas próximas etapas
        frame = cv.resize(frame, (width,height), interpolation =cv.INTER_AREA)

        centro_x, centro_y = width // 2, height // 2


        if cv.waitKey(1) == ord('c'):
            cisalhamento += 0.1


        matriz_trans = matriz_transformacao(anglo, centro_x, centro_y, cisalhamento)
        frame_girando = aplica_transformacao(frame, matriz_trans)


        # A variável image é um np.array com shape=(width, height, colors)
        image = np.array(frame).astype(float)/255

        # Agora, mostrar a imagem na tela!
        cv.imshow('Minha Imagem!', frame_girando)

        anglo+=1
        
        # Se aperto 'q', encerro o loop
        if cv.waitKey(1) == ord('q'):
            break

    # Ao sair do loop, vamos devolver cuidadosamente os recursos ao sistema!
    cap.release()
    cv.destroyAllWindows()

run()

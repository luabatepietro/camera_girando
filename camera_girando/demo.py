import numpy as np
import cv2 as cv
import itertools

def criar_indices(min_i, max_i, min_j, max_j):
    L = list(itertools.product(range(min_i, max_i), range(min_j, max_j)))
    idx_i = np.array([e[0] for e in L])
    idx_j = np.array([e[1] for e in L])
    idx = np.vstack((idx_i, idx_j))
    return idx

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

    meio = np.array([
        [1, 0, centro_x],
        [0, 1, centro_y],
        [0, 0, 1]
    ])

    cis = np.array([
        [1, cisalhamento, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    matriz_trans = np.dot(meio, np.dot(cis, np.dot(rotacao, origem)))

    return matriz_trans

def run():
    # Abrindo a câmera
    cap = cv.VideoCapture(0)

    # Definir a largura e a altura da imagem
    width = 550
    height = 500
    anglo = 0
    cis = 0

    centro_x, centro_y = width // 2 - 25, height // 2 + 25

    if not cap.isOpened():
        print("Não consegui abrir a câmera!")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Não consegui capturar frame!")
            break

        frame = cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
        image = np.array(frame).astype(float)/255
        imagem_ = np.zeros_like(image)

        anglo += 0.05  # Aumentando o ângulo de rotação

        # Matriz de transformação (rotação)
        Y = matriz_transformacao(anglo, centro_x, centro_y, cis)

        # Criando os índices de transformação
        Xd = criar_indices(0, image.shape[0], 0, image.shape[1])
        Xd = np.vstack((Xd, np.ones(Xd.shape[1])))

        # Transformar os índices de acordo com a rotação
        X = np.linalg.inv(Y) @ Xd
        X = X.astype(int)
        Xd = Xd.astype(int)


        # Aplicando o clipping corretamente nos índices transformados
        X[0, :] = np.clip(X[0, :], 0, image.shape[0] - 1)
        X[1, :] = np.clip(X[1, :], 0, image.shape[1] - 1)

        # Mapeamento de volta para a imagem rotacionada
        imagem_[Xd[0, :], Xd[1, :], :] = image[X[0, :], X[1, :], :]

        # Exibindo a imagem na tela
        cv.imshow('Minha Imagem Girando!', imagem_)

        a = cv.waitKey(1)
        if a == ord('q'):
            break
        if a == ord('c'):
            cis += 0.01



    cap.release()
    cv.destroyAllWindows()

run()

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

def main():
    cap = cv.VideoCapture(0)

    width = 350
    height = 300
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

        anglo += 0.2 

        Y = matriz_transformacao(anglo, centro_x, centro_y, cis)

        X = criar_indices(0, image.shape[0], 0, image.shape[1])
        X = np.vstack((X, np.ones(X.shape[1])))

        X_ = np.linalg.inv(Y) @ X

        X_ = np.round(X_).astype(int)
        X = X.astype(int)

        X_[0, :] = np.clip(X_[0, :], 0, image.shape[0] - 1)
        X_[1, :] = np.clip(X_[1, :], 0, image.shape[1] - 1)

        imagem_[X[0, :], X[1, :], :] = image[X_[0, :], X_[1, :], :]

        cv.imshow('Minha Imagem Girando!', imagem_)

        a = cv.waitKey(1) 
        if a == ord('q'):
            break
        if a == ord('c'):
            cis += 0.01
        if a == ord('r'):
            anglo += 0.2
        if a == ord('n'):
            anglo = 0


    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
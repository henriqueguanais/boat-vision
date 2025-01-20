import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

# lista de imagens da camera para a calibracao
images_left = glob.glob('zed_cam/images_calibration/left/*L.jpg')
images_right = glob.glob('zed_cam/images_calibration/right/*R.jpg')

# conversao das imagens para escala de cinza
images_left = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in images_left]
images_right = [cv2.imread(img, cv2.IMREAD_GRAYSCALE) for img in images_right]
print(len(images_left), len(images_right))
# definicao do tamanho do ChessBoard
CBROW = 6
CBCOL = 9

# listas que armazenam os pontos 3D e 2D
objpoints = [] # pontos 3D do mundo real
left_points, right_points = [], [] # pontos 2D do plano da imagem
 
for left_image, right_image in zip(images_left, images_right):
    # procura os cantos do ChessBoard
    ret_left, corners_left = cv2.findChessboardCorners(left_image, (CBCOL,CBROW))
    ret_right, corners_right = cv2.findChessboardCorners(right_image, (CBCOL,CBROW))
    if ret_left and ret_right:
        left_points.append(corners_left)
        right_points.append(corners_right)
    else:
        print("Não foi possível encontrar os cantos do ChessBoard nas imagens.")

# prepara a matriz de pontos 3D do ChessBoard
objp = np.zeros((CBROW * CBCOL, 3), np.float32)
objp[:, :2] = np.mgrid[0:CBCOL, 0:CBROW].T.reshape(-1, 2)
objpoints = [objp] * len(left_points)

# calibracao da camera estereo
err, Kl, Dl, Kr, Dr, R, T, E, F = cv2.stereoCalibrate(objpoints, left_points, right_points, 
                                                      None, None, None, None, (images_left[0].shape[1], images_left[0].shape[0]), flags=0)
print('Left camera:')
print(Kl)
print('Left camera distortion:')
print(Dl)
print('Right camera:')
print(Kr)
print('Right camera distortion:')
print(Dr)
print('Rotation matrix:')
print(R)
print('Translation:')
print(T)
print('Error:')
print(err)

# salvando os parametros da calibracao em um arquivo .yaml
fs = cv2.FileStorage('zed_cam/data/calibration.yaml', cv2.FILE_STORAGE_WRITE)
fs.write('Kl', Kl)
fs.write('Dl', Dl)
fs.write('Kr', Kr)
fs.write('Dr', Dr)
fs.write('R', R)
fs.write('T', T)
fs.release()
print('Calibration data salva.')


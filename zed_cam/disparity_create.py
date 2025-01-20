import cv2
import matplotlib.pyplot as plt
import numpy as np

# extrair os parametros intrinsecos da camera
calibration_dir = "zed_cam/data/calibration.yaml"
calibration_file = cv2.FileStorage(calibration_dir, cv2.FILE_STORAGE_READ)
Kl = np.array(calibration_file.getNode('Kl').mat())
Dl = np.array(calibration_file.getNode('Dl').mat())
Kr = np.array(calibration_file.getNode('Kr').mat())
Dr = np.array(calibration_file.getNode('Dr').mat())
R = np.array(calibration_file.getNode('R').mat())
T = np.array(calibration_file.getNode('T').mat())   
calibration_file.release()

focal_length = Kl[0, 0]
baseline = T[0, 0]/-1000

imgL = cv2.imread("zed_cam/images_test/2025-01-19-163633L.jpg")
imgR = cv2.imread("zed_cam/images_test/2025-01-19-163633R.jpg")

# plota as imagens do par estereo
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB))
plt.title("Left Image", fontsize=20)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB))
plt.title("Right Image", fontsize=20)
plt.axis('off')
plt.show()

imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# retificacao
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(Kl, Dl, Kr, Dr, (imgL.shape[1], imgL.shape[0]), R, T)

# mapas de remapeamento
map1_left, map2_left = cv2.initUndistortRectifyMap(Kl, Dl, R1, P1, (imgL.shape[1], imgL.shape[0]), cv2.CV_32FC1)
map1_right, map2_right = cv2.initUndistortRectifyMap(Kr, Dr, R2, P2, (imgR.shape[1], imgR.shape[0]), cv2.CV_32FC1)

# aplicar remapeamento para retificar as imagens
rectified_left = cv2.remap(imgL, map1_left, map2_left, cv2.INTER_LINEAR)
rectified_right = cv2.remap(imgR, map1_right, map2_right, cv2.INTER_LINEAR)

# juntar as imagens normais e retificadas, para visualizacao
image_normal = np.hstack((imgL, imgR))
image_combined = np.hstack((rectified_left, rectified_right))

# desenhar as linhas epipolares
for i in range(0, image_combined.shape[1], 50):
    cv2.line(image_combined, (0, i), (image_combined.shape[1], i), (0, 0, 0), 4)
    cv2.line(image_normal, (0, i), (image_normal.shape[1], i), (0, 0, 0), 4)
    
plt.figure(figsize=(10, 5))
plt.imshow(image_normal, cmap='gray')
plt.title('Unrectified Images')
plt.axis('off')
plt.show()

plt.figure(figsize=(10, 5))
plt.imshow(image_combined, cmap='gray')
plt.title('Rectified Images')
plt.axis('off')
plt.show()
window_size = 15                 
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=160,             
    blockSize=5,
    P1=8 * 1 * window_size ** 2,    
    P2=32 * 1 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disparity_map = stereo.compute(rectified_left, rectified_right).astype(np.float32)/16

plt.imshow(disparity_map, cmap='gray')
plt.axis('off')
plt.show()

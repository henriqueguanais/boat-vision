import cv2
import matplotlib.pyplot as plt
import numpy as np

# extrair os parametros intrinsecos da camera
calibration_dir = "zed_cam/data/calibration.yaml"
calibration_file = cv2.FileStorage(calibration_dir, cv2.FILE_STORAGE_READ)
Kl = np.array(calibration_file.getNode('Kl').mat())
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

stereo = cv2.StereoSGBM_create()
disparity_map = stereo.compute(imgL, imgR).astype(np.float32)/16

plt.imshow(disparity_map, cmap='gray')
plt.axis('off')
plt.show()

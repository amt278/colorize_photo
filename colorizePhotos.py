import imp
from cv2 import COLOR_BGR2LAB
import numpy as np 
import cv2
from PIL import Image

prototxt_path = 'model/colorization_deploy_v2.prototxt'
model_path = 'model/colorization_release_v2.caffemodel'
kernel_path = 'model/pts_in_hull.npy'
image_path = 'model/lion.jpg'

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path) 
points = np.load(kernel_path) 

points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1,313], 2.606, dtype="float32")] 

bw_image = cv2.imread(image_path) # read image in BGR format
normalize_image = bw_image.astype("float32") / 255.0 
lab = cv2.cvtColor(normalize_image, cv2.COLOR_BGR2LAB) 

resized_image = cv2.resize(lab, (224, 224)) 
l = cv2.split(resized_image)[0]
l -= 50 

net.setInput(cv2.dnn.blobFromImage(l)) 
ab = net.forward()[0, :, :, :].transpose((1,2,0)) 

ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0])) 
l = cv2.split(lab)[0]

colorized_image = np.concatenate((l[:, :, np.newaxis], ab), axis=2) 
colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_LAB2BGR)
colorized_image = (255.0 * colorized_image).astype("uint8") 

cv2.imwrite("colorized.png", colorized_image)
cv2.imshow("b&w image", bw_image)
cv2.imshow("colorized image", colorized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

from matplotlib.colors import hsv_to_rgb
from matplotlib import colors
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
import glob

# Especificar ruta a la carpeta de los archivos
ruta = "Lungs/*.png"

# Función de procesado de las imágenes
def extpulmones(im):

	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

	# Histograma
	# hist_count, hist_bins = np.histogram(im_rgb, 256, [0,256])
	# plt.bar(range(0,256), hist_count)
	# plt.show()

	# Media del histograma sin contar el 0
	med = np.sum(im_gray)
	tamaño = len(im_gray[im_gray>0])
	media = med/tamaño
	print("Media histograma imagen: ", media)

	# Cambio oscuridad
	if media>100:
		ma_mo3 = im_hsv[:,:,2]>160
	else:
		ma_mo3 = im_hsv[:,:,2]>0

	ma_mo3 = ma_mo3.astype(int)
	im_rgb[:,:,1] = im_rgb[:,:,1]*ma_mo3

	# Selección de canal verde
	ca_gre = im_rgb[:,:,1]

	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(ca_gre)

	# Binarizar la imagen para detección de contornos
	ret, thr = cv2.threshold(cl1, 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	# Detección de contornos
	contours, hierarchy = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)

	# Dibujar la máscara de unos sobre la forma de los pulmones
	mas1 = np.zeros(im_gray.shape, np.uint8)
	cv2.drawContours(mas1, contours, 1, 1, thickness=-1)
	mas1 = cv2.morphologyEx(mas1, cv2.MORPH_CLOSE, np.ones((35,35)))
	mas2 = np.zeros(im_gray.shape, np.uint8)
	cv2.drawContours(mas2, contours, 2, 1, thickness=-1)
	mas2 = cv2.morphologyEx(mas2, cv2.MORPH_CLOSE, np.ones((35,35)))
	maskto = mas1 + mas2
	
	# Segmentar la imagen de los pulmones
	im_final = maskto * im_gray

	return im_final

# Imágenes de la carpeta a utilizar
images = []
for file in glob.glob(ruta):
	images.append(cv2.imread(file))

# Fotografías procesadas de toda la carpeta
for im in images:
	im_final = extpulmones(im)
	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	plt.rcParams["figure.figsize"] = [12,8]
	f, ax = plt.subplots(1,2)
	ax[0].imshow(im_gray, cmap='gray',vmin=0, vmax=255)
	ax[1].imshow(im_final, cmap='gray',vmin=0, vmax=255)
	plt.show()
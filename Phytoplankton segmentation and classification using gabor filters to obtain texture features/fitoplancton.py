from matplotlib.colors import hsv_to_rgb
from matplotlib import colors
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTE
import pickle
import numpy as np
import pandas as pd
import cv2
import glob
import math

sacar_imagenes = "si"
images = []
ruta = ""
for file in glob.glob(ruta):
	images.append(cv2.imread(file))

# Clasificador entrenado
with open('fitoplancton_model.pkl', 'rb') as fid:
    clf = pickle.load(fid)

# Crear un banco de filtros
filters = []
ksize = 31
for theta in np.arange(0, np.pi, np.pi / 16):
    kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    kern /= 1.5*kern.sum()
    filters.append(kern)

# Función para formar el vector de caracteristicas de textura
# 0 aeruginosa, 1 flos-aque y 2 woronichinia
def gabor_values(im, filters):
	feature_vector = []
	for index,f in enumerate(filters):
	    conv = cv2.filter2D(im, -1, f)
	    # Calcular estadísticas para el vector
	    mean = np.mean(conv)
	    var = np.var(conv)
	    feature_vector.append(mean)
	    feature_vector.append(var)
	        
	    # Distribucion de colores de la imagen
	    histogram, _ = np.histogram(conv, 100)
	    # Probabilidades de ocurrencia de cada color
	    histogram = histogram.astype(float)/ (conv.shape[0]*conv.shape[1])
	    # Formula entropia
	    H = -np.sum(histogram*np.log2(histogram + np.finfo(float).eps))
	    feature_vector.append(H)

	# Calculamos tambien las matrices de concurrencia
	cm = greycomatrix(im, [1, 2], [0, np.pi/4, np.pi/2, 3*np.pi/4], normed=True, symmetric=True)
	props = greycoprops(cm, 'contrast')
	vector = np.reshape(props, (1, props.shape[0]*props.shape[1]))
	props2 = greycoprops(cm, 'energy')
	vector2 = np.reshape(props2, (1, props2.shape[0]*props2.shape[1]))
	props3 = greycoprops(cm, 'homogeneity')
	vector3 = np.reshape(props3, (1, props3.shape[0]*props3.shape[1]))
	props4 = greycoprops(cm, 'correlation')
	vector4 = np.reshape(props4, (1, props4.shape[0]*props4.shape[1]))

	#Concatenación
	feature_vector.extend(vector[0]) 
	feature_vector.extend(vector2[0])
	feature_vector.extend(vector3[0])
	feature_vector.extend(vector4[0])

	return feature_vector

# Función para clasificar, vector texturas de la imágen y clasificación
def clasificar_texturas(im, clf):
	vec_propio = np.asarray(gabor_values(im, filters)).reshape(1,-1)
	prediccion = clf.predict(vec_propio)
	return prediccion

# Inicialización de la estimación del volumen para todas las especies en todas las imágenes del rack
anabaena = 0
microcystis = 0
woronichinia = 0

contador = 0
for im in images:
	# Progreso análisis
	contador += 1
	print(contador, "imágenes de", len(images))

	# Inicialización valores por imagen
	ana = 0
	micro = 0
	woro = 0

	im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
	canal = im_rgb[:,:,2]  # Canal en segundo procesado
	satura = im_hsv[:,:,1] # Selección canal 1 procesado
	
	# Preprocesado para correcta segmentación utilizando el canal s del hsv(por las características de las imágenes)
	khat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (159,159))
	kclose = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55,55))
	kopen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35))
	filtroga = cv2.GaussianBlur(satura, (21, 21), 0) #Filtro ruido general
	
	# Remarcar los bordes(secciones desenfocadas tienen nivel de s más bajo)
	filtrosa = cv2.medianBlur(filtroga, 5)
	diff = filtroga - filtrosa
	satura2 = filtroga + diff

	# Top hat + un filtrado de medias para rebajar ruido del fondo espúreo
	satura2 = cv2.morphologyEx(satura2, cv2.MORPH_TOPHAT, khat)
	satura2 = cv2.blur(satura2, (15,15))

	# Umbralización con treshold bajo
	ret, thr1 = cv2.threshold(satura2, 20, 255, cv2.THRESH_BINARY)
	thr1 = cv2.morphologyEx(thr1, cv2.MORPH_CLOSE, kclose) #Cierre para asegurarnos bien que cojemos toda la región
	thr1 = cv2.morphologyEx(thr1, cv2.MORPH_OPEN, kopen)
	
	# Detección de contornos
	contours, hierarchy = cv2.findContours(thr1,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)

	# Selección contornos válidos
	conta = 0
	for con in contours:
		small = cv2.contourArea(con) # < 25000 área eliminar
		if small < 25000:
			break
		conta += 1

	contours = contours[0:conta]

	# Máscara para dibujar contornos
	out = im_rgb * 1 # Copiamos la imagen original para no modificarla
	out2 = np.zeros(satura.shape, np.uint8)

	# Tratamiento
	for con in range(0,len(contours)):
	
		# Máscara para la región a analizar
		mask1 = np.zeros(satura.shape, np.uint8)
		cv2.drawContours(mask1, contours, con, 1, thickness=-1)

		# Cálculos para descartar especie
		thr2 = thr1 == 255
		thr2 = thr2.astype(int)
		satura3 = thr2 * mask1 # Eliminar burbujas, etc
		satura4 = satura3 + mask1
		if con == 0:
			savisu = satura4
		vacio = len(satura4[satura4==1])
		lleno = len(satura4[satura4==2])
		porcen = vacio / (lleno + vacio)

		if porcen > 0.25: # Descartar anabaena
			cv2.drawContours(out, contours, con, (0,0,255), 3)
			cv2.drawContours(out2, contours, con, 1, -1)
			ana = ana + lleno
			anabaena = anabaena + lleno
			continue

		# Calcular circularidad y excentricidad para ver si tenemos muchos objetos o uno
		((x,y),(w,h), angle) = cv2.minAreaRect(contours[con])
		Are = cv2.contourArea(contours[con])
		Per = cv2.arcLength(contours[con], True)
		Cir = (4*math.pi*Are)/Per**2
		Exc = h/w

		if Cir < 0.51 and Exc > 0.65: # Comprobar si son varios juntos o uno (posibilidad de varios)
			newimage = satura * mask1
			fnew = cv2.medianBlur(newimage, 11)
			ret, thr3 = cv2.threshold(fnew, 75, 255, cv2.THRESH_BINARY)
			kclose2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (47,47))
			thr3 = cv2.morphologyEx(thr3, cv2.MORPH_CLOSE, kclose2)

			# Contornos finos
			contours2, hierarchy = cv2.findContours(thr3,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
			contours2 = sorted(contours2, key=cv2.contourArea, reverse=True)

			# Selección contornos válidos
			conta = 0
			for con in contours2:
				small = cv2.contourArea(con) # < 40000 área eliminar
				if small < 40000:
					break
				conta += 1

			contours2 = contours2[0:conta]
			
			for conto in range(0,len(contours2)):
				# Máscara para la región a analizar apertura muy grande con kernel circular para relleno
				mask2 = np.zeros(satura.shape, np.uint8)
				cv2.drawContours(mask2, contours2, conto, 1, thickness=-1)

				# Dibujar contornos
				cv2.drawContours(out, contours2, conto, (0,0,255), 3)
				cv2.drawContours(out2, contours2, conto, 1, -1)

				kclose3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (137,137))
				mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kclose3)
				mask4 = canal * mask2

				# Extraer imágen para usar texturas
				x,y,w,h = cv2.boundingRect(contours2[conto])
				imtextura = im_gray[y:y+h, x:x+w]

				if sacar_imagenes == "si":
					plt.rcParams["figure.figsize"] = [12,8]
					f, ax = plt.subplots(1,2)
					ax[0].imshow(im_gray, cmap='gray',vmin=0, vmax=255)
					ax[1].imshow(imtextura, cmap='gray', vmin=0, vmax=255)
					plt.show()

				prediccion = clasificar_texturas(imtextura, clf)[0]

				if prediccion == 0:
					print("Detectado Microcystis")
					micro = micro + len(mask2[mask2==1])
					microcystis = microcystis + len(mask2[mask2==1])
				else:
					print("Detectado Woronichinia")
					woro = woro + len(mask2[mask2==1])
					woronichinia = woronichinia + len(mask2[mask2==1])

		else:
			# Extraer imágen para usar texturas
			cv2.drawContours(out, contours, con, (0,0,255), 3)
			cv2.drawContours(out2, contours, con, 1, -1)

			x,y,w,h = cv2.boundingRect(contours[con])
			imtextura = im_gray[y:y+h, x:x+w]

			if sacar_imagenes == "si":
				plt.rcParams["figure.figsize"] = [12,8]
				f, ax = plt.subplots(1,2)
				ax[0].imshow(im_gray, cmap='gray',vmin=0, vmax=255)
				ax[1].imshow(imtextura, cmap='gray', vmin=0, vmax=255)
				plt.show()

			prediccion = clasificar_texturas(imtextura, clf)[0]

			if prediccion == 0:
				print("Detectado Microcystis")
				micro = micro + len(mask1[mask1==1])
				microcystis = microcystis + len(mask1[mask1==1])
			else:
				print("Detectado Woronichinia")
				woro = woro + len(mask1[mask1==1])
				woronichinia = woronichinia + len(mask1[mask1==1])

	
	# Impresión imágenes procesadas
	if sacar_imagenes == "si":
		plt.rcParams["figure.figsize"] = [12,8]
		f, ax = plt.subplots(2,3)
		ax[0,0].imshow(im_gray, cmap='gray',vmin=0, vmax=255)
		ax[0,1].imshow(satura, cmap='gray',vmin=0, vmax=255)
		ax[0,2].imshow(satura2, cmap='gray',vmin=0, vmax=255)
		ax[1,0].imshow(savisu, cmap='gray',vmin=0, vmax=2)
		ax[1,1].imshow(out, cmap='gray',vmin=0, vmax=255)
		ax[1,2].imshow(out2, cmap='gray',vmin=0, vmax=1)
		plt.show()

	# Imprimir los volúmenes en la imagen
	# ima_ana = round(ana/(im_gray.shape[0]*im_gray.shape[1]), 2)
	# print("P.Vol Anabaena spiroides en imagen =", ima_ana)
	# ima_micro = round(micro/(im_gray.shape[0]*im_gray.shape[1]), 2)
	# print("P.Vol Microcystis spp. en imagen =", ima_micro)
	# ima_woro = round(woro/(im_gray.shape[0]*im_gray.shape[1]), 2)
	# print("P.Vol Woronichinia en imagen =", ima_woro)

# Imprimir el promedio entre todas las imágenes
print("\nEstimación en el total de las imágenes:")
total = ((im_gray.shape[0]*im_gray.shape[1])) * len(images)
ima_ana = round(anabaena/total, 2)
print("P.Vol Anabaena spiroides total =", ima_ana)
ima_micro = round(microcystis/total, 2)
print("P.Vol Microcystis spp. total =", ima_micro)
ima_woro = round(woronichinia/total, 2)
print("P.Vol Woronichinia total =", ima_woro)
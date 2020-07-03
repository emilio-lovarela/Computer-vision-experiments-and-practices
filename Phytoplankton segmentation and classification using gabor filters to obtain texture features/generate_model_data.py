from matplotlib.colors import hsv_to_rgb
from matplotlib import colors
from matplotlib import pyplot as plt
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from skimage.feature import greycomatrix, greycoprops
import numpy as np
import pandas as pd
import cv2
import glob
import math

images = []
ruta = ""
for file in glob.glob(ruta):
	images.append(cv2.imread(file))

# Crear un banco de filtros
	filters = []
	ksize = 31
	for theta in np.arange(0, np.pi, np.pi / 16):
	    kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
	    kern /= 1.5*kern.sum()
	    filters.append(kern)

# Función usada para calcular los valores de texturas
texturas_val = []
salida_cuen = 0

# Aeruginosa 0 y 2 Woronichinia
def gabor_values(im):
	salida = [0,0,0,0,0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
	global salida_cuen
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

	# Concatenación
	feature_vector.extend(vector[0]) 
	feature_vector.extend(vector2[0])
	feature_vector.extend(vector3[0])
	feature_vector.extend(vector4[0])
	feature_vector.append(salida[salida_cuen])
	salida_cuen = salida_cuen + 1

	global textura
	texturas_val.append(feature_vector)

contador = 0
for im in images:
	# Progreso análisis
	contador += 1
	print(contador, "imágenes de", len(images))

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

	# Dibujar contornos
	out = im_rgb * 1 # Copiamos la imagen original para no modificarla
	cv2.drawContours(out, contours, -1, (0,0,255), 3)

	# Tratamiento
	for con in range(0,len(contours)):
	
		# Máscara para la región a analizar
		mask1 = np.zeros(satura.shape, np.uint8)
		cv2.drawContours(mask1, contours, con, 1, thickness=-1)

		# Cálculos para descartar especie
		thr2 = thr1 == 255
		thr2 = thr2.astype(int)
		satura3 = thr2 * mask1 #eliminar burbujas, etc
		satura4 = satura3 + mask1
		vacio = len(satura4[satura4==1])
		lleno = len(satura4[satura4==2])
		porcen = vacio / (lleno + vacio)

		if porcen > 0.25: #Descartar anbaena
			continue

		# Calcular circularidad y excentricidad para ver si tenemos muchos objetos o uno
		((x,y),(w,h), angle) = cv2.minAreaRect(contours[con])
		Are = cv2.contourArea(contours[con])
		Per = cv2.arcLength(contours[con], True)
		Cir = (4*math.pi*Are)/Per**2
		Exc = h/w

		if Cir < 0.51 and Exc > 0.65: #Comprobar si son varios juntos o uno (posibilidad de varios)
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
			
			# Dibujar contornos
			out2 = im_rgb * 1 # Copiamos la imagen original para no modificarla
			cv2.drawContours(out2, contours2, -1, (0,0,255), 3)

			for conto in range(0,len(contours2)):
				# Máscara para la región a analizar apertura muy grande con kernel circular para relleno
				mask2 = np.zeros(satura.shape, np.uint8)
				cv2.drawContours(mask2, contours2, conto, 1, thickness=-1)

				kclose3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (137,137))
				mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kclose3)
				mask4 = canal * mask2

				# Extraer imágen para usar texturas
				x,y,w,h = cv2.boundingRect(contours2[conto])
				imtextura = im_gray[y:y+h, x:x+w]

				gabor_values(imtextura)

		else:
			# Extraer imágen para usar texturas
			x,y,w,h = cv2.boundingRect(contours[con])
			imtextura = im_gray[y:y+h, x:x+w]

			gabor_values(imtextura)

	salir = "salir"
	if contador == 20:
		break

df = pd.DataFrame(texturas_val)
df.to_csv('Fitoplancton_features.csv', index=False, header=False)
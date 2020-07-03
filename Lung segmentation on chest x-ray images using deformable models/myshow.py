import SimpleITK as sitk
from math import ceil
import matplotlib.pyplot as plt
import sys

from ipywidgets import interact, fixed
from ipywidgets import widgets

def show_images(images, titles=None):
	def scroll_images(N_imagen, images, titles):
		img = images[N_imagen - 1]
		if str(type(img)) == "<class 'SimpleITK.SimpleITK.Image'>":
			plt.imshow(sitk.GetArrayViewFromImage(img),cmap=plt.cm.Greys_r);
		else:
			plt.imshow(img, cmap=plt.cm.Greys_r);
		plt.axis('off')
		if titles != None:
			plt.title(titles[N_imagen - 1])
		plt.show()

	x = len(images)
	interact(scroll_images, N_imagen=(1,x,1), images=fixed(images), titles=fixed(titles));

def show_multimages(*images, titles=None):
	def scroll_images(N_imagen, images, titles):
		img = images[N_imagen - 1]
		number = ceil(len(img)/3)

		plt.rcParams["figure.figsize"] = [12,8]
		f, ax = plt.subplots(number,3)

		i = 0
		j = 0
		for element in img:
			if str(type(element)) == "<class 'SimpleITK.SimpleITK.Image'>":
				ax[j,i].imshow(sitk.GetArrayViewFromImage(element),cmap=plt.cm.Greys_r);
			else:
				ax[j,i].imshow(element, cmap=plt.cm.Greys_r);
			i += 1
			if i == 3:
				i = 0
				j += 1

		plt.axis('off')
		if titles != None:
			plt.suptitle(titles[N_imagen - 1])
		plt.show()

	x = len(images)
	if len(images[0]) == 1:
		print("ERROR just one Element in list")
		sys.exit()
	elif x == 1:
		print("Running show_images mode")
		show_images(images[0], titles)
		return
	interact(scroll_images, N_imagen=(1,x,1), images=fixed(images), titles=fixed(titles));
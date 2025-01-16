from tkinter import filedialog
import tkinter as tk
import cv2 
import PIL.Image, PIL.ImageTk
import numpy as np


MARGIN = 5
MAXDIM = 400

class App():
	#specify path of image files
	def  __init__(self, window, window_title, image_path = 'town.jpg'):
		#Initialize values for Methods
		self.color_quantization = 1
		self.line_size = 3
		self.blur_value = 3

		############### CREATE GUI ################
		self.window = window
		self.window.title(window_title) 

		# Using cv2.imread() method
		img = cv2.imread(image_path)

		#Load an image using OpenCV
		self.cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.NEWcv_img = self.cv_img.copy()
	
		#Get the image dimensions
		self.height, self.width, channels = self.cv_img.shape
		print('height', self.height)
		print('width', self.width)
		
		#Create a FRAME
		self.frame1 = tk.Frame(self.window, width=self.width, height=self.height, bg='gray')
		self.frame1.pack(fill=tk.BOTH)

		#Create a CANVAS for original image
		self.canvas0 = tk.Canvas(self.frame1, width=MAXDIM, height=MAXDIM+(2*MARGIN), bg='gray')
		self.canvas0.pack(side=tk.LEFT)

		#Create a CANVAS for changing image
		self.canvas1 = tk.Canvas(self.frame1, width=MAXDIM, height=MAXDIM+(2*MARGIN), bg='orange')
		self.canvas1.pack(side=tk.RIGHT)

		# Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
		self.photoOG = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
		self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))

        # Add a PhotoImage to the Canvas (original)
		self.canvas0.create_image(MAXDIM//2, MAXDIM//2, image=self.photoOG)
        
        # Add a PhotoImage to the Canvas (changing effects)
		self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
        
        # Write labels for both images, font/size can be changed
		self.canvas0.create_text(MAXDIM//2, MAXDIM,font="Tahoma 16",text="Original Photo")
		self.canvas1.create_text(MAXDIM//2, MAXDIM,font="Tahoma 16",text="Modified Photo")
        
# ##############################################################################################
# ################################   PARAMETER TOOLBAR   #######################################
# ##############################################################################################

        # Create a FRAME that can fit the below features
		self.frame2 = tk.Frame(self.window, width=self.width, height=200, bg='gray')
		self.frame2.pack(side=tk.BOTTOM, fill=tk.BOTH)

		# Create a SCALE that lets the user adjust the brightness of the image
		self.brightness_slider = tk.Scale(self.frame2, from_=1, to=255, orient=tk.HORIZONTAL, showvalue=1, resolution = 1,command = self.adjust_brightness, length=300, sliderlength=20, label="Brigthness", font="Tahoma 9")
		self.brightness_slider.place(relx=0.05, rely=0.02, relwidth=0.4, relheight=0.35)

		# Create a SCALE that lets the user apply a High Boost Filter to the image
		self.hbf_slider = tk.Scale(self.frame2, from_= 1, to=3, orient=tk.HORIZONTAL, resolution=0.1, showvalue=1, command = self.highBoostFilter, length=100, sliderlength=20, label="High Boost Filter", font="Tahoma 9")
		self.hbf_slider.place(relx=0.5, rely=0.02, relwidth=0.2, relheight=0.35)

		#Create a SCALE that lets the user change the color qunantization amount
		self.color_quantization_slider = tk.Scale(self.frame2, from_= 1, to= 20, orient=tk.HORIZONTAL, resolution=1, showvalue=1, command = self.change_color_quanitization, length=400, sliderlength=20, label="Color Quanitization", font="Tahoma 9")
		self.color_quantization_slider.place(relx=0.05, rely=0.48, relwidth=0.3, relheight=0.4)

		#Create a SCALE that lets the user change the edge line size amount
		self.edge_line_size_slider = tk.Scale(self.frame2, from_= 3, to= 13, orient=tk.HORIZONTAL, showvalue=0, command = self.change_edge_line_size, length=400, sliderlength=20, label="Edge Line Size", font="Tahoma 9")
		self.edge_line_size_slider.place(relx=0.4, rely=0.48, relwidth=0.2, relheight=0.4)

		#Create a SCALE that lets the user change the edge mask blur amount
		self.edge_blur_slider = tk.Scale(self.frame2, from_= 3, to= 13, orient=tk.HORIZONTAL, showvalue=0, command = self.change_edge_blur_value, length=400, sliderlength=20, label="Edge Blur Value", font="Tahoma 9")
		self.edge_blur_slider.place(relx=0.7, rely=0.48, relwidth=0.2, relheight=0.4)

		self.window.mainloop()

##############################################################################################
#################################  CALLBACK FUNCTIONS  #######################################
##############################################################################################

#'''#################################  OTHER FUNCTIONS  ###############################'''#
    # define other callback functions here
    ############# ADJUST BRIGHTNESS ##############
	def adjust_brightness(self, value):
		value = self.brightness_slider.get()  # get value from the corresponding scale
		hsv = cv2.cvtColor(self.NEWcv_img, cv2.COLOR_RGB2HSV)
		h, s, v = cv2.split(hsv)

		lim = 255 - value
		v[v > lim] = 255
		v[v <= lim] += value

		final_hsv = cv2.merge((h, s, v))
		NEWcv_img_modify = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
		self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(NEWcv_img_modify))
		self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
	
	############# APPLY HIGH BOOST FILTERING ##############
	def highBoostFilter(self, A):
		#Kernel that will slide over the image (convolution)
		A = self.hbf_slider.get()  # get value from the corresponding scale
		kernel = np.array([[-1 , -1 , -1] , [-1 , 8+A, -1] ,[-1 , -1 , -1]])
		NEWcv_img_modify = cv2.filter2D(self.NEWcv_img, -1 , kernel = kernel)
		self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(NEWcv_img_modify))
		self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)

	############# METHODS ##############
	### Below methods are used to change inputs for the diffferent functions in the make_cartoon_img function
	def change_color_quanitization(self, k):	
		self.color_quantization = int(k)
		#print('color Quanitization value ', self.color_quantization)
		self.make_cartoon_img()

	def change_edge_line_size(self, k):
		if (int(k) % 2) == 0:
			k = int(k) - 1
		self.line_size = int(k)
		#print('line size ', self.line_size)
		self.make_cartoon_img()

	def change_edge_blur_value(self, k):
		if (int(k) % 2) == 0:
			k = int(k) - 1
		self.blur_value = int(k)
		#print('blur value ', self.blur_value)
		self.make_cartoon_img()

	############# APPLY CARTOON IMAGE ##############
	def make_cartoon_img(self):
		def edge_mask(img, line_size, blur_value):
			gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
			gray_blur = cv2.medianBlur(gray, blur_value)
			edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, C = 2)
			return edges
	
	
		def color_quantization(img, k):
			# Transform the image
			data = np.float32(img).reshape((-1, 3))

			# Determine criteria
			criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

			# Implementing K-Means
			ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
			center = np.uint8(center)
			result = center[label.flatten()]
			result = result.reshape(img.shape)
			return result

		def bilateral_filter(img):
			blurred = cv2.bilateralFilter(img, d=7, sigmaColor=100,sigmaSpace=200)
			return blurred

		image_edges = edge_mask(self.NEWcv_img, self.line_size, self.blur_value)
		image_reduced_color = color_quantization(self.NEWcv_img, self.color_quantization)
		image_reduced_noise = bilateral_filter(image_reduced_color)

		cartoon = cv2.bitwise_and(image_reduced_noise, image_reduced_noise, mask = image_edges)
		self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cartoon))
		self.canvas1.create_image(MAXDIM//2, MAXDIM//2, image=self.photo, anchor=tk.CENTER)
##############################################################################################
# Create a window and pass it to the Application object

App(tk.Tk(), "FINAL PROJECT")
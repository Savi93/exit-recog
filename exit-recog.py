import cv2
import numpy as np
import imutils
from playsound import playsound

import tkinter as tk
from PIL import ImageTk, Image
import tkinter.filedialog
import tkinter.messagebox

#The method finds out which one of the edges is the direction vertex of the arrow:
#graphically, it is the point which has the max. distance to the closest neighbour
def find_arrow_point(approx):
    arr_dist = []
    
    for point1 in approx:
        min_dist = 9999999
        p1_x = point1[0][0]
        p1_y = point1[0][1]
        for point2 in approx:
            p2_x = point2[0][0]
            p2_y = point2[0][1]
            dist_x = abs(p1_x - p2_x)
            dist_y = abs(p1_y - p2_y)
            res_dist = dist_x + dist_y
            
            if(res_dist != 0 and res_dist < min_dist):
                min_dist = res_dist
                
        arr_dist.append(min_dist)
        
    arr_index = arr_dist.index(max(arr_dist))
    return(approx[arr_index][0][0], approx[arr_index][0][1])

#Finds the edge points by approximating the contour of the polygon; epsilon value empirically estimated.  
def find_edge_points(cont, epsilon):
    peri = cv2.arcLength(cont, True)
    area = cv2.contourArea(cont)
    return cv2.approxPolyDP(cont, epsilon * peri, True)

#Draw the edge points on the provided image
def draw_edge_points(approx, image):
    for i in approx:
        x, y = i.ravel()
        cv2.circle(image,(x,y), 3, (0, 0, 0), -1)

#Find the center of the contour
def find_center(cont):
    M = cv2.moments(cont)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY

#Method to check whether the contour is a rectangle; an approximated polygon is a rectengle, if:
#Has 4 vertices and is convex
def is_rectangle(cont): 
    approx = find_edge_points(cont, 0.05)
        
    if(len(approx) == 4 and cv2.isContourConvex(approx)):
        return True

    return False

#Method to check whether the contour is an arrow; an approximated polygon is an arrow if:
#Has 7 or 9 vertices (two types of arrows), is a concave polygon and is similar to the two samples of the arrows.
def is_arrow(cont, cont_arrow, cont_arrow_thin): 
    approx = find_edge_points(cont, 0.02)
    comp1 = cv2.matchShapes(cont, cont_arrow, 1, 0.0)
    comp2 = cv2.matchShapes(cont, cont_arrow_thin, 1, 0.0)
        
    if((len(approx) == 7 or len(approx) == 9) and(comp1 <= 1.5 or comp2 <= 1.5) and not(cv2.isContourConvex(approx))):
        return True

    return False

#Method used to find the biggest contour in the list; it differs whether we are searching the shield or the arrow; we have to
#check also whether the contour area is actually a rectangle (shield) or an arrow and verify the actual size against the original image
def find_biggest_contour(contours, option, original_image):
	max_cont = []
	max_size = 0

	if(option == 0):
		for cont in contours:
			size = cv2.contourArea(cont)
			height_orig, width_orig = original_image.shape[:2]
			if(is_rectangle(cont) and size >=  (height_orig * width_orig / 900)):
				if(size > max_size):
					max_size = size
					max_cont = cont

		return max_cont	

	else:
		cont_thin, cont_right = comparison_arrows('arrow_right.png', 'arrow_thin.png')

		for cont in contours:
			size = cv2.contourArea(cont)
			if(is_arrow(cont, cont_right, cont_thin)):
				if(size > max_size):
					max_size = size
					max_cont = cont

		return max_cont

#Methods used to draw a circle at position x, y
def draw_point(x, y, image):
    cv2.circle(image, (x, y), 4, (255, 0, 0), -1)

#Return the contours of two arrows which will be used for comparison purposes
def comparison_arrows(image1, image2):
    image_right = cv2.imread(image1)
    image_thin = cv2.imread(image2)

    image_thin_g = cv2.cvtColor(image_thin, cv2.COLOR_BGR2GRAY)
    image_right_g = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    
    right_filter = cv2.GaussianBlur(image_right_g,(5,5),0)
    thin_filter = cv2.GaussianBlur(image_thin_g,(5,5),0)

    ret_right, thresh_right = cv2.threshold(right_filter,180,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret_thin, thresh_thin = cv2.threshold(thin_filter,180,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    kernel = np.ones((3,3),np.uint8)
    
    thresh_thin = cv2.erode(thresh_thin, kernel)
    thresh_right = cv2.erode(thresh_right, kernel)

    contours_thin, hierarchy_thin = cv2.findContours(thresh_thin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours_right, hierarchy_right = cv2.findContours(thresh_right, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return contours_thin[0], contours_right[0]

#Check direction of the arrow; we use simple mathematical operations to check the position of the direction vertex of the arrow in respect
#to the center of the arrow
def check_direction(arrow_point, center_point):
    arrow_x = arrow_point[0]
    arrow_y = arrow_point[1]

    center_x  = center_point[0]
    center_y = center_point[1]

    dist_x = abs(arrow_x - center_x)
    dist_y = abs(arrow_y - center_y)

    if(dist_x > dist_y):
        if(arrow_x - center_x < 0):
        	playsound('left.mp3', block = False)
        else:
            playsound('right.mp3', block = False)
    else:
        if(arrow_y - center_y < 0):
            playsound('up.mp3', block = False)
        else:
            playsound('down.mp3', block = False)
            
def startProcessing(path):
	#Loads the image.
	image = cv2.imread(path)
    #Filters with a LPF in order to remove noise.
	image_fil = cv2.GaussianBlur(image, (5, 5), 0)
	#Converts the image from BGR to HSV (Hue-Saturation-Brightness), in order to have linear range of the colors.
	image_hsv = cv2.cvtColor(image_fil, cv2.COLOR_BGR2HSV)
	#Define a binary mask that detects the green color range
	global green_mask
	green_mask = cv2.inRange(image_hsv, np.array([33,40,40]), np.array([86,255,255]))
	#Dilate the margin of the green mask
	kernel = np.ones((3,3),np.uint8)
	green_mask = cv2.dilate(green_mask, kernel)
	#Apply AND bitwise operator on the original image by using the green mask matrix
	global res_mask
	res_mask = cv2.bitwise_and(image, image, mask = green_mask)
	#Extract only the gray version out from the resulting image
	res_mask_gray = cv2.cvtColor(res_mask, cv2.COLOR_BGR2GRAY)
	#Find all the contours in the image
	contours, hierarchy = cv2.findContours(res_mask_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	try:
		#Extract only the biggest contour out frome the image (green shield?)
		biggest_rect = find_biggest_contour(contours, 0, image)
		#Save coordinates of the rectangle reppresenting the biggest contour
		x, y, w, h = cv2.boundingRect(biggest_rect)
		#Cutoff the rectangle (if green shield exists, should extract it from the original image)
		global crop_img
		crop_img = image[y :y + h, x : x + w]
		#Resize the cropped image to have height = 250px
		crop_img = imutils.resize(crop_img, height=250)
		#Filters with a LPF in order to remove noise.
		crop_img_fil = cv2.GaussianBlur(crop_img,(3,3),0)
		#Convert the cropped image into the gray version
		gray_crop = cv2.cvtColor(crop_img_fil, cv2.COLOR_BGR2GRAY)
		#Apply binary thresholding on the cropped image
		global thresh_crop
		ret, thresh_crop = cv2.threshold(gray_crop,180,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		#Erode (remove some borders) from the thresholded image
		kernel = np.ones((3,3),np.uint8)
		thresh_crop = cv2.erode(thresh_crop, kernel)
		#Find the contours into the cropped image
		contours, hierarchy = cv2.findContours(thresh_crop, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

		crop_img_out = crop_img.copy()

		#If no contours are found, raise exception and signal it
		if(len(contours) > 0):
			#Draw contours in red color
			cv2.drawContours(crop_img_out, contours, -1, (0,0,255), 1)
			#Find the biggest contour according to some specific criteria
			max_cont = find_biggest_contour(contours, 1, None)

			#If not big contour is recognized, raise exception and signal it
			if(len(max_cont) > 0):
				#Find the edge points of the arrow (use polygon approximation algorithm)
			    approx = find_edge_points(max_cont, 0.02)
			    #Draw the contour of the arrow in green
			    cv2.drawContours(crop_img_out, max_cont, -1, (0,255,0), 1)
			    #Draw the edge points of the arrow
			    draw_edge_points(approx, crop_img_out)
			    #Find the geometric center of the arrow
			    cX, cY = find_center(approx)
			    #Draw the center point of the arrow
			    draw_point(cX, cY, crop_img_out)
			    #Find the x, y coordinates of the edge of the arrow
			    arr_x, arr_y = find_arrow_point(approx)
			    #Draw the edge of the arrow
			    draw_point(arr_x, arr_y, crop_img_out)
			    #Check the direction of the arrow
			    check_direction((arr_x, arr_y), (cX, cY))

			    plotProcessedImageGUI(crop_img_out)

			else:
			    raise Exception()
		else:
			raise Exception()

	except:
		noProcessedImageGUI()
		playsound('not-found.mp3', block = False)


############################# GUI STARTS HERE ##################################

def plotProcessingInfo():
	images = [green_mask, res_mask, crop_img, thresh_crop];
	j = 0

	for img in images:
		cv2.imshow('Processing step ' + str(j), cv2.resize(img, (300, 300)))
		j += 1
		cv2.waitKey(50)

def plotOriginalImageGUI(path):
    img = Image.open(path)
    img_res = img.resize((450, 300), Image.ANTIALIAS)
    im_res = ImageTk.PhotoImage(img_res)
    panel_orig.configure(image=im_res, width = 0, height = 0)
    panel_orig.image = im_res

def noProcessedImageGUI():
	panel_found.configure(image = '', text = "NO emergency exit found!", width = 65, height = 20)
	panel_found.image = ''

def plotProcessedImageGUI(image):
    b,g,r = cv2.split(image)
    im_rgb = cv2.merge((r,g,b))
    im_rgb = Image.fromarray(im_rgb)
    im_res = im_rgb.resize((450, 300), Image.ANTIALIAS)
    im_final = ImageTk.PhotoImage(im_res)
    panel_found.configure(image=im_final, width = 0, height = 0)
    panel_found.image = im_final

    global button_info
    button_info["state"] = "normal"
    
def selectImageGUI():
	global panel_orig
	global panel_found
	empty = False
        
	path = ""
	path = tk.filedialog.askopenfilename(title = "Select file", filetypes=[("JPG files", "*.jpg"), ("JPG files", "*.jpeg"), ("PNG files", "*.png")])
	
	if(str(path) == ""):
		tkinter.messagebox.showerror(title="Error", message="No file was selected!")
	else:
		button_info["state"] = "disabled"
		plotOriginalImageGUI(path)
		startProcessing(path)

def initializeGUI():
	window = tk.Tk()
	window.config(bg = "white")
	window.resizable(width=False, height=False)
	window.title("Emergency exit recognizer")
	window.geometry("1100x650")

	label_title = tk.Label(master= window, text="Emergency exit recognizer", bg = "white", fg = "gray30")
	label_title.config(font=("Helvetica", 25, "italic bold"))
	label_title.pack()

	global panel_orig
	global panel_found
	panel_orig = tk.Label(image = None, borderwidth=2, relief="groove", text = "Upload an image for processing...", width = 65, height = 25)
	panel_orig.pack(side="left", padx=2, pady=5)
	panel_found = tk.Label(image = None, borderwidth=2, relief="groove", text = "Processed image here...", width = 65, height = 25)
	panel_found.pack(side="right", padx=2, pady=5)

	global button_image
	button_image = tk.Button(master=window, text ="Select image", bg = "gray30", fg = "white", command = selectImageGUI)
	button_image.pack(anchor = "s", side = "left", padx="0", pady="10")
	global button_info
	button_info = tk.Button(master=window, text ="Show info", bg = "gray30", fg = "white", state = "disabled", command = plotProcessingInfo)
	button_info.pack(anchor = "s", side = "right", padx="2", pady="10")

	window.mainloop()

initializeGUI()


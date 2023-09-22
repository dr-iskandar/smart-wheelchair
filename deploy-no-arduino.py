### importing required libraries
import torch
import cv2
import time
import re
import numpy as np
import easyocr

##### DEFINING GLOBAL VARIABLE
EASY_OCR = easyocr.Reader(['en'], gpu=True) ### initiating easyocr
OCR_TH = 0.2
# NAMA_RUANGAN = str(input('Nama Ruangan : '))
# NAMA_RUANGAN = NAMA_RUANGAN.upper()

# time.sleep(1)

### -------------------------------------- function to run detection ---------------------------------------------------------
def detectx (frame, model):
	frame = [frame]
	print(f"[INFO] Detecting. . . ")
	results = model(frame)
	labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

	return labels, cordinates

#### ---------------------------- function to recognize license plate --------------------------------------


# function to recognize license plate numbers using Tesseract OCR
def filter_text(region, ocr_result, region_threshold):
	rectangle_size = region.shape[0]*region.shape[1]
	
	plate = [] 
	# print(ocr_result)
	for result in ocr_result:
		length = np.sum(np.subtract(result[0][1], result[0][0]))
		height = np.sum(np.subtract(result[0][2], result[0][1]))
		
		if length*height / rectangle_size > region_threshold:
			plate.append(result[1])
	return plate


def recognize_plate_easyocr(img, coords,reader,region_threshold):
	# separate coordinates from box
	xmin, ymin, xmax, ymax = coords
	# get the subimage that makes up the bounded region and take an additional 5 pixels on each side
	# nplate = img[int(ymin)-5:int(ymax)+5, int(xmin)-5:int(xmax)+5]
	nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] ### cropping the number plate from the whole image

	ocr_result = reader.readtext(nplate)
	# char_result = ocr_result[0][1]

	text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)
	# print(f'------ini text{text}')

	if len(text) ==1:
		text = text[0].upper()
	return text, ocr_result


### to filter out wrong detections 



### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results,frame,classes):

	"""
	--> This function takes results, frame and classes
	--> results: contains labels and coordinates predicted by model on the given frame
	--> classes: contains the strting labels
	"""
	global plate_num
	global bbox_area
	labels, cord = results
	n = len(labels)
	x_shape, y_shape = frame.shape[1], frame.shape[0]
	print(f"[INFO] Total {n} detections. . . ")
	print(f"[INFO] Looping through all detections. . . ")


	### looping through the detections
	for i in range(n):
		row = cord[i]
		if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
			print(f"[INFO] Extracting BBox coordinates. . . ")
			x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
			text_d = classes[int(labels[i])]
			# cv2.imwrite("./output/dp.jpg",frame[int(y1):int(y2), int(x1):int(x2)])

			coords = [x1,y1,x2,y2]
			bbox_area  = (x2-x1)*(y2-y1) 

			# plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)
			text, plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)

			cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
			cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background
			if plate_num != []:
				cv2.putText(frame, f"{plate_num[0][1]}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
				return frame, plate_num, bbox_area
			
			# cv2.imwrite("./output/np.jpg",frame[int(y1)-25:int(y2)+25, int(x1)-25:int(x2)+25])
	return frame, [(0, 0, 0)] ,0
### ---------------------------------------------- Main function -----------------------------------------------------

def main(img_path=None, vid_path=None, vid_out = None):
	string = ''
	print(f"[INFO] Loading model... ")
	## loading the custom trained model
	model =  torch.hub.load('./yolov5', 'custom', source ='local', path='best.pt',force_reload=True) ### The repo is stored locally

	classes = model.names ### class names in string format
	# print(f'---- ini classes {classes}')
	

	### --------------- for detection on image --------------------
	if img_path != None:
		print(f"[INFO] Working with image: {img_path}")
		# img_out_name = f"./output/result_{img_path.split('/')[-1]}"

		frame = cv2.imread(img_path) ### reading the image
		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
		
		results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    
		frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
		frame, num_plate, area_bbox = plot_boxes(results, frame,classes = classes)
		print(f'[INFO] Character Result : {num_plate[0][1]}')
		print(f'[INFO] Area : {area_bbox}')

		

		cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result

		while True:
			# frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

			cv2.imshow("img_only", frame)

			if cv2.waitKey(5) & 0xFF == ord('q'):
				print(f"[INFO] Exiting. . . ")

				cv2.imwrite("final_output/image_capture.jpg",frame) ## if you want to save he output result.
				break

	### --------------- for detection on video --------------------
	elif vid_path !=None:
		print(f"[INFO] Working with video: {vid_path}")

		## reading the video
		cap = cv2.VideoCapture(vid_path)

		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = int(cap.get(cv2.CAP_PROP_FPS))


		# if vid_out: ### creating the video writer if video output path is given

			# by default VideoCapture returns float instead of int
			# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			# fps = int(cap.get(cv2.CAP_PROP_FPS))
			# codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
			# out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

		# assert cap.isOpened()
		frame_no = 1

		cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
		while True:
			# start_time = time.time()
			ret, frame = cap.read()
			if ret  and frame_no %1 == 0:
				print(f"[INFO] Working with frame {frame_no} ")

				frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

				results = detectx(frame, model = model)
				frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
				frame, num_plate, area_bbox = plot_boxes(results, frame, classes=classes)
				print(f'[INFO] Character Result : {num_plate[0][1]}')
				print(f'[INFO] Area : {area_bbox}')

				print('--------------------------------------------------------------------------------')
					# if num_plate[0][1] == NAMA_RUANGAN:
					#     string = ''
				
				cv2.imshow("vid_out", frame)
				# if vid_out:
				#     print(f"[INFO] Saving output video. . . ")
				#     out.write(frame)
		
				if cv2.waitKey(5) & 0xFF == ord('q'):
					cv2.imwrite('final_output/video_to_image_capture.jpg',frame)
					break
				frame_no += 1
		
		print(f"[INFO] Cleaning up. . . ")
		### releaseing the writer
		# out.release()
		
		## closing all windows
		cv2.destroyAllWindows()



### -------------------  calling the main function-------------------------------


main(vid_path="./test_footages/output3.mp4") ### for custom video
# main(vid_path=0) #### for webcam
# main(img_path="./test_images/papan0.png") ## for image
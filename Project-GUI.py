from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import object_detection as od
import imageio
import cv2
from PIL import Image, ImageTk
from tkinter import ttk
from tracking_scripts.centroidtracker import CentroidTracker
from tracking_scripts.trackableobject import TrackableObject
from imutils.video import FPS
from tkinter.font import BOLD, Font
import numpy as np
import imutils
import dlib
import os
from os import path
import classifier
from datetime import datetime
import trafficshape
import trafficLightColor

class Window(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master
      
       
       
        self.pack(fill=BOTH, expand=2)


        menu = Menu(self.master)
        self.master.config(menu=menu)

        file = Menu(menu)
        file.add_command(label="Open", command=self.open_file)
        file.add_command(label="Exit", command=self.client_exit)
        menu.add_cascade(label="File", menu=file)
        
        load = Image.open("eight.jpg")
        render = ImageTk.PhotoImage(load)
        img = Label(self, image=render,width=800,height=600)
      
        img.image = render
        img.place(x=0, y=0)
        
        self.bold25 = Font(self.master, size=25)
        
        l = Label(text = "Signal Light Violation System",font=('Helvetica', 30),fg="black")
       
        l.place(relx=0.5,rely= 0.2,anchor = CENTER)
        
        OButton = Button(text="Open CCTV", command=self.open_file)
        OButton.place(relx=0.5,rely= 0.5, anchor = CENTER)
        
        
        
        
       
            
        
    
        
       
       

    def open_file(self):
        self.filename = filedialog.askopenfilename()

        cap = cv2.VideoCapture(self.filename)

        reader = imageio.get_reader(self.filename)
        fps = reader.get_meta_data()['fps'] 

        ret, image = cap.read()
        root.withdraw()
        self.main_process()
        
    def show_image(self, frame):
        self.imgSize = Image.fromarray(frame)
        self.tkimage =  ImageTk.PhotoImage(self.imgSize)
       
        self.w, self.h = (1366, 768)
        

        self.canvas = Canvas(master = root, width = self.w, height = self.h)
        self.canvas.create_image(0, 0, image=self.tkimage, anchor='nw')
        self.canvas.pack()

       

        

   

    def client_exit(self):
        exit()
    def openfile():
        return filedialog.askopenfilename()

    
    

    def main_process(self):

        input = self.filename

       # cap = cv2.VideoCapture(input)

        #reader = imageio.get_reader(input)
        #fps = reader.get_meta_data()['fps']    
        #writer = imageio.get_writer('E:\PROJECT\Traffic-Signal-Violation-Detection-System-master\Traffic-Signal-Violation-Detection-System-master\Resources\output\output.mp4', fps = fps)
       

        base="yolo-coco"
        labelsPath = os.path.sep.join([base, "coco.names"])
        LABELS = open(labelsPath).read().strip().split("\n")
        weightsPath = os.path.sep.join([base, "yolov3.weights"])
        configPath = os.path.sep.join([base, "yolov3.cfg"])
        
        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] #retrieve yolo layers
        
        print("[INFO] opening video file...")
        vs = cv2.VideoCapture(input)
        
        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        W = None
        H = None
        
        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        ct = CentroidTracker(maxDisappeared=10, maxDistance=50)
        trackers = []
        trackableObjects = {}
        TRAFFIC_LIGHT_CONFIDENT_VALUE=5000
 
        # initialize the total number of frames processed thus far, along
        # with the total number of objects that have moved either up or down
        totalFrames = 0
        totalUp = 0
        sys_init = False
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
            dtype="uint8")
        # start the frames per second throughput estimator
        fps = FPS().start()
        light_color=""       
        color=""
        
        
        # loop over frames from the video stream
        while True:
        	# grab the next frame and handle if we are reading from either
        	# VideoCapture or VideoStream
        	frame = vs.read()
        	frame = frame[1]
        
        	# if we are viewing a video and we did not grab a frame then we
        	# have reached the end of the video
        	if input is not None and frame is None:
        		break
        	# resize the frame to have a maximum width of 500 pixels (the
        	# less data we have, the faster we can process it), then convert
        	# the frame from BGR to RGB for dlib
        	if sys_init == False:
                    #traff_sel_roi = cv2.selectROI("Select Traffic Light", frame, False, False)
                    #cv2.destroyWindow("Select Traffic Light")
                    #traffic_light = [(traff_sel_roi[0], traff_sel_roi[1]), (traff_sel_roi[0] + traff_sel_roi[2], traff_sel_roi[1] + traff_sel_roi[3])]
        		if not path.exists(input+'ROI1.txt'):
                                
                                sel_roi = cv2.selectROI("Select Monitor Line", frame, False, False)
                                cv2.destroyWindow("Select Monitor Line")
                                mon_line = [(sel_roi[0], sel_roi[1]), (sel_roi[0] + sel_roi[2], sel_roi[1] + sel_roi[3])]
                                fp = open(input+"ROI1.txt", 'w')
                                for i in sel_roi:
                                        fp.write('{}\n'.format(str(i)))
                                fp.close()
    
        		if not path.exists(input+'traffic.txt'):    
                
                                tra_roi = cv2.selectROI("Select Traffic",frame, False, False)
                                cv2.destroyWindow("Select Traffic light")
                                mon_line = [(sel_roi[0], sel_roi[1]), (sel_roi[0] + sel_roi[2], sel_roi[1] + sel_roi[3])]
                                fp = open(input+"traffic.txt", 'w')
                                for i in tra_roi:
                                        fp.write('{}\n'.format(str(i)))
                                fp.close() 
                   
        	sys_init = True
        
        	fp = open(input+'ROI1.txt', 'r')
        	ROI = []
        
        	for line in fp:
        		line = line.strip()
        		ROI.append(int(line))
        
        	mon_line = [(ROI[0],ROI[1]),(ROI[0]+ROI[2],ROI[1]+ROI[3])]
           
    
           
        	fp = open(input+'traffic.txt', 'r')
        	TOI = []
        	for line in fp:
        		line = line.strip()
        		TOI.append(int(line))
		   
        	traffic_ligh=frame[int(TOI[1]):int(TOI[1]+TOI[3]), int(TOI[0]):int(TOI[0]+TOI[2])]					
        	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        	traffic_light=0
        	# if the frame dimensions are empty, set them
        	if W is None or H is None:
        		(H, W) = frame.shape[:2]
        
        	status = "Waiting"
        	rects = []
        
        	# check to see if we should run a more computationally expensive
        	# object detection method to aid our tracker
        	if totalFrames %10 == 0:
        		# set the status and initialize our new set of object trackers
        		status = "Detecting"
        		trackers = []
        
        		# convert the frame to a blob and pass the blob through the
        		# network and obtain the detections
        		blob = cv2.dnn.blobFromImage(frame,1 / 255.0, (416, 416),swapRB=True, crop=False)
        		net.setInput(blob)
        		layeroutputs = net.forward(ln)
        		confidences=[]
        		boxes=[]
        		classID=[]
        		classes=["car","motorbike","truck"]
        		# loop over the detections
        		for layer in layeroutputs:
        			for i, detection in enumerate(layer):
        
        				class_scores=detection[5:]
        				confidence = detection[4]
        				class_id=np.argmax(class_scores)
        				class_score=class_scores[class_id]
        				if LABELS[class_id] not in classes:
        					continue
        				if (confidence)>0.5:
        
        					confidences.append(float(confidence))
        					BOX=detection[0:4]*np.array([W,H,W,H])
        					(centerX,centerY,Width,Height)=BOX.astype("int")
        
        					startX=int(centerX-(Width/2))
        					startY=int(centerY-(Height/2))
        					boxes.append([startX,startY,int(Width),int(Height)])
        					classID.append(class_id)
        
        		idxs=cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.5)
        		if len(idxs) > 0:
        			# loop over the indexes we are keeping
        			for i in idxs.flatten():
        				# extract the bounding box coordinates
        				(x, y) = (boxes[i][0], boxes[i][1])
        				(w, h) = (boxes[i][2], boxes[i][3])
        				if (classID[i] == 9):
        
        					
        					xlight,ylight,wlight,hlight=(x,y,w,h)
        				endX =  x + w
        				endY = y + h
        				# construct a dlib rectangle object from the bounding
        				# box coordinates and then start the dlib correlation
        				# tracker
        				tracker = dlib.correlation_tracker()
        				rect = dlib.rectangle(x, y, endX, endY)
        				tracker.start_track(rgb, rect)
        
        
        				# add the tracker to our list of trackers so we can
        				# utilize it during skip frames
        				
        				trackers.append((tracker,LABELS[classID[i]]))
        
        	# otherwise, we should utilize our object *trackers* rather than
        	# object *detectors* to obtain a higher frame processing throughput
        	else:
        		# loop over the trackers
        		for tracker,id in trackers:
        			# set the status of our system to be 'tracking' rather
        			# than 'waiting' or 'detecting'
        			status = "Tracking"
        			# update the tracker and grab the updated position
        			tracker.update(rgb)
        			pos = tracker.get_position()
        			# unpack the position object
        			startX = int(pos.left())
        			startY = int(pos.top())
        			endX = int(pos.right())
        			endY = int(pos.bottom())
        
        			rects.append((startX, startY, endX, endY,id))
                                
        	# draw a horizontal line in the center of the frame -- once an
        	# object crosses this line we will determine whether they jumped the red light or not
        	cv2.line(frame, mon_line[0], mon_line[1], (0, 0, 255), thickness=1)
        	# use the centroid tracker to associate the (1) old object
        	# centroids with (2) the newly computed object centroids
                    
        	#light = frame[ylight:ylight + hlight, xlight:xlight + wlight]
            
        	hsv = cv2.cvtColor(traffic_ligh, cv2.COLOR_BGR2HSV)#light = frame[ylight:ylight + hlight, xlight:xlight + wlight]            
            
        	sum_saturation = np.sum(hsv[:,:,1]) # Sum the brightness values#light = frame[ylight:ylight + hlight, xlight:xlight + wlight]
        	area = 32*32#light = frame[ylight:ylight + hlight, xlight:xlight + wlight]
        	avg_saturation = sum_saturation / area#light = frame[ylight:ylight + hlight, xlight:xlight + wlight]
        	sat_low = int(avg_saturation * 1.3)#light = frame[ylight:ylight + hlight, xlight:xlight + wlight]
        	val_low = 140#light = frame[ylight:ylight + hlight, xlight:xlight + wlight]
        	lower_red1 = np.array([150,sat_low,val_low])#b, g, r=cv2.split(traffic_ligh)        	
        	upper_red1 = np.array([180,255,255])#b, g, r=cv2.split(traffic_ligh)

        	lower_green = np.array([70,sat_low,val_low])#b, g, r=cv2.split(traffic_ligh)
        	upper_green = np.array([100,255,255])#b, g, r=cv2.split(traffic_ligh)
        	lower_yellow = np.array([10,sat_low,val_low])#b, g, r=cv2.split(traffic_ligh)
        	upper_yellow = np.array([60,255,255])#b, g, r=cv2.split(traffic_ligh)
        	maskr = cv2.inRange(hsv, lower_red1, upper_red1)#b, g, r=cv2.split(traffic_ligh)            
        	#b, g, r=cv2.split(traffic_ligh)
            #b, g, r = cv2.split(traffic_ligh)
        	maskg = cv2.inRange(hsv, lower_green, upper_green)#traffic_ligh=cv2.merge([r,g,b])
        	masky = cv2.inRange(hsv, lower_yellow, upper_yellow)#traffic_ligh=cv2.merge([r,g,b])
        	#traffic_ligh=cv2.merge([r,g,b])        
        	r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80,param1=50, param2=10, minRadius=0, maxRadius=30)#maskr = cv2.add(mask1, mask2)
        	y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, 60,param1=50, param2=10, minRadius=0, maxRadius=30)#maskr = cv2.add(mask1, mask2)
        	g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 30,param1=50, param2=5, minRadius=0, maxRadius=30)#maskr = cv2.add(mask1, mask2)
        	if r_circles is not None:
        		light_color="Red"                     
        		color="red"        	
        	if y_circles is not None:
        		color="orange"
        	if g_circles is not None:
        		color="green"                   
        	
        	#b, g, r = cv2.split(traffic_ligh)
        	#traffic_ligh=cv2.merge([r,g,b])
        	#if(trafficLightColor.estimate_label(traffic_ligh)=="Red"):
        	#	color="Red"
        	#	light_color="Red"
        	#if(trafficLightColor.estimate_label(traffic_ligh)=="Yellow"):
        	#	color="Orange"
        	#if(trafficLightColor.estimate_label(traffic_ligh)=="Green"):
        	#	color="Green"
        	cv2.putText(frame, color, (TOI[0], TOI[1]), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        	objects= ct.update(rects)
        	labels = ct.labels
        	boundingboxes=ct.boundingbox
        	
        	# loop over the tracked objects
        	#traffic_light_crop = frame[int(traff_sel_roi[1]):int(traff_sel_roi[1]+traff_sel_roi[3]), int(traff_sel_roi[0]):int(traff_sel_roi[0]+traff_sel_roi[2])]
        	
        	
        	for (objectID, centroid) in objects.items():
                        
        		box = boundingboxes.get(objectID)
        		
        		#traffic_color=trafficshape.detect(frame)
        		box = boundingboxes.get(objectID)
        		text = "ID {}".format(objectID)
        		#draw bounding box for each object
        		#traffic_light_crop = frame[int(traff_sel_roi[1]):int(traff_sel_roi[1]+traff_sel_roi[3]), int(traff_sel_roi[0]):int(traff_sel_roi[0]+traff_sel_roi[2])]
               
        		#hsv_traffic_light_crop = cv2.cvtColor(traffic_light_crop, cv2.COLOR_BGR2HSV)
        		#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#low_red = np.array([161, 155, 84], np.uint8)
        		#high_red = np.array([179, 255, 255], np.uint8)
        		#traffic_signal_mask = cv2.inRange(hsv_traffic_light_crop, low_red, high_red)
        	
                
        		#if np.sum(traffic_signal_mask) > TRAFFIC_LIGHT_CONFIDENT_VALUE:
            	#		color='red'   
        		#if color=="red":
        		#	light_color="Red"
        
        		#if color == "yellow":
        		#	light_color = "Yellow"
        		#if color == "green":
        		#	light_color = "Green"
        		cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(0, 255, 0), thickness=1)
        		# check to see if a trackable object exists for the current
        		# object ID
        		to = trackableObjects.get(objectID, None)
        
        		# if there is no existing trackable object, create one
        		if to is None:
        			to = TrackableObject(objectID, centroid)
        
        		# otherwise, there is a trackable object so we can utilize it
        		# to determine direction
        		else:
        			# the difference between the y-coordinate of the *current*
        			# centroid and the mean of *previous* centroids will tell
        			# us in which direction the object is moving (negative for
        			# 'up' and positive for 'down')
        			y = [c[1] for c in to.centroids]
        			direction = centroid[1] - np.mean(y)
        			#when the light turns red, save the first position for each vehicle
        			if light_color=="Red" and to.firstpos==0:
        				to.firstpos=centroid[1]
        			to.centroids.append(centroid)
        
        			if to.counted==True : 
                        #when the vehicle passed the line, we mark it with red color bounding box.
                        
        				cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=2)
        				
        					
        				
        
        			# check to see if the object has been counted or not and the first position is below the line
        			if not to.counted and to.firstpos > (mon_line[0][1]+mon_line[1][1])/2  :
        
        				if direction < -3 and centroid[1] < (mon_line[0][1]+mon_line[1][1])/2 and light_color=="Red" :
        					totalUp += 1
        					cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color=(0, 0, 255), thickness=1)
        					traffic_light = frame[box[1]:box[3],box[0]:box[2]]
        					save_dir = "./img/{:DATE_%d-%m-%Y_TIME_%H-%M-%S-%f}_Normal.png".format(datetime.now())
                            			
        					cv2.imwrite(save_dir, traffic_light)
        					to.counted = True
        
        		# store the trackable object in our dictionary
        		trackableObjects[objectID] = to
        	# construct a tuple of information we will be displaying on the
        	# frame
        	info = [
        		("Violation", totalUp),
        	    ]
        
        	# loop over the info tuples and draw them on our frame
        	for (i, (k, v)) in enumerate(info):
        		text = "{}: {}".format(k, v)
        		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
        			cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        	# show the output frame
        	#self.show_image(frame)
        	cv2.imshow('Traffic',frame)
        	key = cv2.waitKey(1) & 0xFF
        	if key == ord("q"):
        		break
        	# increment the total number of frames processed thus far and
        	# then update the FPS counter
        	totalFrames += 1
        	fps.update()
        	
        
        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        
        root.deiconify()
        # close any open windows
       
        cv2.destroyAllWindows()   
       

root = Tk()

app = Window(root)


       
root.geometry("%dx%d"%(600, 400))
root.resizable(0, 0)
root.title("Traffic Violation")
root.configure(bg="red")



root.mainloop()

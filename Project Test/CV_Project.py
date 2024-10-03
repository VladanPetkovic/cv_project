import cv2
import cv2.data



img = cv2.imread(r'test.jpg') 
if img is None:
    print('Wrong path:')
    exit

img = cv2.resize(img, (720, 640)) 
frame = img.copy() 

# ------------ Model for Age detection --------# 


# Model requirements for image 
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
        '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 
#model_mean = (78.4263377603, 87.7689143744, 114.895847746) 


# ------------- Model for face detection---------# 
face_detector_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# converting to grayscale 
img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

# -------------detecting the faces--------------# 
faces = face_detector_cascade.detectMultiScale(img_gray, 1.3, 5) 

# If no faces our detected 
#if not faces: 
#    print('No face detected')
#   #skip picture
    
#else: 
    # --------- Bounding Face ---------# 
for (x, y, w, h) in faces: 
    framed = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imshow('framed face', framed)
    cv2.waitKey(0)

    crop_img = img[y:y+h, x:x+w]
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)

        # ----- Image preprocessing --------#                    



        # -------Age Prediction---------# 
        
        
        
        
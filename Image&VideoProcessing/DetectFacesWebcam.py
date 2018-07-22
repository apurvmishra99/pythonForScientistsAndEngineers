import cv2
import sys


cascPath = "/home/apu35711/anaconda3/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename = WEBCAM.log, level = log.info)

if len(sys.argv) < 2:
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture(sys.argv[1])

while True:
    #frame-by-frame capturing
    ret, image = video_capture.read()
    if not ret:
        break
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5, (30,30))

    for x,y,w,h in faces:
        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.imshow("Faces found", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
video_capture.release()
cv2.destroyAllWindows()
    




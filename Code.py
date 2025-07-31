import cv2
import mediapipe as mp

#Initialize Haar cascades
face_cas=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
eye_cas=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
smile_cas=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_smile.xml')

#Initialize Mediapipe for hand detection
mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
hands=mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5)

#Function to count fingers
def count_fingers(hand_landmarks):
    tips=[8,12,16,20]  #Index,Middle,Ring,Pinky
    count=sum(1 for i in range(4) if hand_landmarks.landmark[tips[i]].y < hand_landmarks.landmark[tips[i] - 2].y)

    #Check if thumb is extended
    thumb_tip=hand_landmarks.landmark[4].x
    thumb_ip=hand_landmarks.landmark[3].x  #Joint before thumb tip
    is_thumb_extended = thumb_tip < thumb_ip if hand_landmarks.landmark[0].x > thumb_tip else thumb_tip > thumb_ip
    if count==4 and is_thumb_extended:
        return "Five"

    return ["No Fingers","One","Two","Three","Four"][count]

#Start video capture
cap=cv2.VideoCapture(0)
while cap.isOpened():
    _,img=cap.read()
    if not _:
        break

    #Converting Color image to Gray
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Face Detection
    faces=face_cas.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=5)

    for (x, y, w, h) in faces:
        #Decimal Code for Blue is (255,255,0)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        #Region of Interest
        roi_gray,roi_color=gray[y:y+h,x:x+w], img[y:y+h,x:x+w]
        
        #Eye Detection and Status
        eyes=eye_cas.detectMultiScale(roi_gray,scaleFactor=1.1,minNeighbors=10,minSize=(20, 20))
        left_status,right_status=("Open","Open") if len(eyes)>=2 else ("Closed","Open" if len(eyes)==1 else "Closed")

        #Smile Detection and Status
        smiles=smile_cas.detectMultiScale(roi_gray,scaleFactor=1.7,minNeighbors=25)
        smile_status="Smiling" if len(smiles)>0 else "Not Smiling"

        #Black stroke for better visibility
        text=f"Left Eye: {left_status}, Right Eye: {right_status}, {smile_status}"
        cv2.putText(img,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),4)  #Black border
        cv2.putText(img,text,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)  #White text

    #Hand Detection
    results=hands.process(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            gesture=count_fingers(hand_landmarks)

            # Black stroke for hand gesture
            cv2.putText(img,gesture,(20,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),3) #Black border
            cv2.putText(img,gesture,(20,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2) #White text

    cv2.imshow('Face & Hand Detection',img)
    if cv2.waitKey(1)!=-1:
        break

cap.release()
cv2.destroyAllWindows()

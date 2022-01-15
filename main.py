import cv2
import mediapipe as mp
import numpy as np


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#capturing webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()


# Pose detection
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence = 0.7, min_tracking_confidence = 0.7) as pose:

    while cap.isOpened():
        ret, frame = cap.read()


    #rendor stuff
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #make detection
        results = pose.process(image)



        image.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # showing detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        cv2.imshow("Webcam", image)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
#developing

cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence = 0.7, min_tracking_confidence = 0.7) as pose:

    while cap.isOpened():
        ret, frame = cap.read()


    #rendor stuff
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #make detection
        results = pose.process(image)



        image.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass


        # showing detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        cv2.imshow("Webcam", image)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


#detection
# for lndmrk in mp_pose.PoseLandmark:
#     print(lndmrk)
landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]


#Calculating angle

def angle_calc(a,b,c):
    a = np.array(a) #shoulder to elbow
    b = np.array(b) #elbow
    c = np.array(c) #elbow to shoulder

    radians = np.arctan2(c[1]-b[1], c[0]- b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                    #distance formula
    #  y-coordinates             c[1] = y2 b[1] = y1 ,
    #  x-coordinates             c[0] = x2  b[0] = x1
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0:
        angle = 360-angle
    return angle

shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
print(angle_calc(shoulder,elbow,wrist))




cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence = 0.7, min_tracking_confidence = 0.7) as pose:
    counter = 0
    stage = None

    while cap.isOpened():
        ret, frame = cap.read()


    #rendor stuff
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        #make detection
        results = pose.process(image)



        image.flags.writeable = True
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            #coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            #angle Calculation
            ang = angle_calc(shoulder, elbow, wrist)
            angle = round(ang, 2)

            #showing on screen
            cv2.putText(image, "Angle: "+str(angle), tuple(np.multiply(elbow,[1140, 780]).astype(int)), cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2, cv2.LINE_AA )


            #Repetition counter
            if angle > 165 :
                stage = "down"
            if angle < 30 and stage == "down":
                stage = "up"
                counter += 1
                print(counter)
        except:
            pass


        # showing detection
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        #showing on screen
        cv2.putText(image,"Repititions: ", (15,12), cv2.FONT_HERSHEY_PLAIN, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, "Repititions: q"+str(counter), (10,60), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2 , cv2.LINE_AA )


        cv2.imshow("Webcam", image)

        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()




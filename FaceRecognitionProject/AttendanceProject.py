import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from deepface import DeepFace  # Import DeepFace for emotion detection
import matplotlib.pyplot as plt  # Importing matplotlib

path = 'ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name, emotion):
    with open('Attendance.csv', 'r+') as f:
        mydataList = f.readlines()
        nameList = []
        for line in mydataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%d/%m/%Y %H:%M:%S")
            f.writelines(f'{name},{dtString},{emotion}\n')  # Log emotion along with attendance

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

            # Unpack face location
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale the coordinates

            # Crop the face from the frame for emotion detection
            face = img[y1:y2, x1:x2]

            # Detect emotion using DeepFace
            emotion_predictions = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = emotion_predictions[0]['dominant_emotion']

            # Log attendance with the detected emotion
            markAttendance(name, emotion)

            # Draw the face bounding box and label with emotion
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f'{name} ({emotion})', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Use matplotlib to display the image instead of cv2.imshow
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib
    plt.imshow(img_rgb)
    plt.axis('off')  # Turn off axis
    plt.show(block=False)  # Display without blocking the rest of the code execution
    plt.pause(0.001)  # Pause to allow for display updates

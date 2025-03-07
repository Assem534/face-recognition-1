import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

path = 'images'
images = []
classNames = []

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to find encodings
def findEncodings(images):
    encodList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Ensure face was detected before appending
            encodList.append(encode[0])
    return encodList

def markAttendans(name):
    with open('attendans.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]  # Read all names first

        if name not in nameList:  # Now check properly
            now = datetime.now()
            dtString = now.strftime('%H:%M')  # Fixed format (removed ":")
            f.write(f'\n{name},{dtString}')  # Use f.write instead of writelines for a single line

encodingListKnow = findEncodings(images)
print("Encoding images: Done")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 1, 1)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeface, locface in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodingListKnow, encodeface)
        faceDis = face_recognition.face_distance(encodingListKnow, encodeface)

        if any(matches):  # If at least one match is found
            matchindex = np.argmin(faceDis)  # Get the best match
            name = classNames[matchindex].upper()
        else:
            name = "UNKNOWN"

        # Draw bounding box and label
        y1, x2, y2, x1 = locface
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, name, (x1, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)

        if name != "UNKNOWN":  # Only mark attendance for known faces
            markAttendans(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

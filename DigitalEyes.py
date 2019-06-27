import time
import dlib
import cv2
import os
from imutils import paths
from imutils.video import VideoStream
import face_recognition
import imutils
import pickle

def add_person():
    person_name = input('What is the name of the new person: ').lower()
    folder = 'dataset' +'\\'+ person_name                              
    if not os.path.exists(folder):
        input("I will now take 20 pictures. Press ENTER when ready.")       
        os.makedirs(folder)
        video = cv2.VideoCapture(0)
        detector = dlib.get_frontal_face_detector()
        counter = 1
        timer = 0

        cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)                    

        while counter < 21:
            ret, frame = video.read()
            if counter == 1:
                time.sleep(4)
            else:
                time.sleep(0.5)

            frame = cv2.flip(frame,1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)    
            if len(rects):
                face_im = cv2.resize(gray, (0,0),fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
                cv2.imwrite(folder+'/'+str(counter)+'.jpg',face_im)
                print('Images Saved:' + str(counter))
                counter += 1
                cv2.imshow('Saved Face', face_im)
            cv2.imshow('Video Feed', frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break
        video.release()
        cv2.destroyAllWindows()
    else:
        print("This name already exists.")
        
    print("Training the model...\n")
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images("dataset"))

    knownEncodings = []
    knownNames = []

    for (i, imagePath) in enumerate(imagePaths):
        print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]

        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb,model='hog')

        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            knownEncodings.append(encoding)
            knownNames.append(name)

    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open("encodings.pickle", "wb")
    f.write(pickle.dumps(data))
    f.close()
    print("Done..")
        
def live():
    print("[INFO] loading encodings...")
    data = pickle.loads(open("encodings.pickle", "rb").read())

    print("[INFO] starting video stream...")
    cap = cv2.VideoCapture(0)
    time.sleep(2.0)

    process_this_frame = True

    while True:
        ret,frame = cap.read()
        frame = cv2.flip(frame,1)
        # Resize frame of video to 1/10 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb = small_frame[:, :, ::-1]
        r = frame.shape[1] / float(rgb.shape[1])

        # Only process every other frame of video to save time
        if process_this_frame:
            boxes = face_recognition.face_locations(rgb,model='hog')
            encodings = face_recognition.face_encodings(rgb, boxes)
            names = []

            for encoding in encodings:
                matches = face_recognition.compare_faces(data["encodings"],encoding)
                name = "Unknown"

                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}

                    for i in matchedIdxs:
                        name = data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                names.append(name)
        process_this_frame = not process_this_frame

        for ((top, right, bottom, left), name) in zip(boxes, names):
            top = int(top*r)
            right = int(right*r)
            bottom = int(bottom*r)
            left = int(left*r)

            cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2,cv2.FILLED)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame,name,(left, y),cv2.FONT_HERSHEY_DUPLEX,0.75,(0,255,0),2)

        cv2.imshow("Frame", frame)
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    while True:
        print("Select one of the options below: ")
        print('Press 1 for adding a new face')
        print('Press 2 for the live recognition')
        print('Press 3 to exit')

        choice = int(input())

        if choice > 3 or choice < 1:
            print('Please select a valid choice')
        if choice == 1:
            add_person()
        elif choice == 2:
            live()
        elif choice == 3:
            print('You choose to exit!')
            break

        cv2.destroyAllWindows()
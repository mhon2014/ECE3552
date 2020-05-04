import cv2
import numpy as np
# import pafy

# https://talhassner.github.io/home/publication/2015_CVPR age and gender files


cap = cv2.VideoCapture(0)


cap.set(3, 480) #set width of the frame
cap.set(4, 640) #set height of the frame

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
#list of outputs
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

#list of outputs for gender
gender_list = ['Male', 'Female']

#load models, caffe framework
def load_caffe_models():
    age_net = cv2.dnn.readNetFromCaffe('deploy_age.prototxt', 'age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('deploy_gender.prototxt', 'gender_net.caffemodel')
    return(age_net, gender_net)

def video_detector(age_net, gender_net):
    #define the font
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        #read the frames/images
        ret, image = cap.read()
        
        #face classifier, from openCV
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        #turn imaget to gray
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #detect faces and returns x,y,w,h
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        #check how many faces we found
        if(len(faces)>0):
            print("Found {} faces".format(str(len(faces))))

        for (x, y, w, h )in faces:
            #draw boxes around faces
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
            #Get Face 
            face_img = image[y:y+h, h:h+w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
           
           
            #Predict Gender
            #Set input
            gender_net.setInput(blob)
            #Feed forward
            gender_preds = gender_net.forward()
            #get the first argument which has the highest probability
            gender = gender_list[gender_preds[0].argmax()]
            #trying to get gender probability
            gender_probability = gender_preds[0].max()*100
            print("Gender : " + gender)
            print(gender_probability)


            #Predict Age
            age_net.setInput(blob)
            #feed foward
            age_preds = age_net.forward()
            #get highest output
            age = age_list[age_preds[0].argmax()]
            #get probability
            age_probability = age_preds[0].max()*100
            print("Age Range: " + age)
            print(age_probability)


            overlay_text = "%s %.2f %s %.2f" % (gender, gender_probability ,age, age_probability)
            # overlay_text = "%s %s" % (gender,age)

            #put text over the frames
            cv2.putText(image, overlay_text, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('frame', image)  
        
        #press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

if __name__ == "__main__":
    age_net, gender_net = load_caffe_models()
    video_detector(age_net, gender_net)
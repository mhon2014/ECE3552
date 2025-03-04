# import numpy as np
# import cv2

# cap = cv2.VideoCapture(0)

# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

import cv2

cap = cv2.VideoCapture(0)

# cap.set(3, 480) #set width of the frame
# cap.set(4, 640) #set height of

def video_detector():
    while (True):
        ret, frame = cap.read()
        
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow('frame', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    
    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_detector()
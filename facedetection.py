import time
import cv2
import dlib
from matplotlib import pyplot as plt

image_path = './image/multiface.jpg'

def read_from_image(url, k = 0):
    # read
    img = cv2.imread(url, 3)
    # show
    # cv2.imshow('Image01', img)      
    # plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # plt.show()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img

# read_from_image(image_path)

def read_from_camera():
    cap = cv2.VideoCapture(0)
    i = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('pic'+ str(i) +'.png', frame)
            i += 1
        if (cv2.waitKey(1) & 0xFF == ord('q')) or i == 1:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    img = cv2.imread('pic0.png', 3)
    return img

# read_from_camera()

def text_on_image(image, text, x, y, w, h, b=255, g=255, r=255):
    # image = cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 1)
    cv2.putText(image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (h,g,r), 2)

def recognition_face(image):
    # Khai báo việc sử dụng các hàm của dlib
    hog_face_detector = dlib.get_frontal_face_detector()
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./model/mmod_human_face_detector.dat')

    # Thực hiện xác định bằng HOG và SVM
    start = time.time()
    faces_hog = hog_face_detector(image, 1)
    end = time.time()
    print_hog = "Hog + SVM" + str(end-start)

    # Vẽ một đường bao màu xanh lá xung quanh các khuôn mặt được xác định ra bởi HOG + SVM
    for face in faces_hog:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
        img = cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
        text_on_image(img, print_hog, x, y, w, h, 0, 0, 255)

    # Thực hiện xác định bằng CNN
    start = time.time()
    faces_cnn = cnn_face_detector(image, 1)
    end = time.time()
    print_cnn = "CNN" + str(end-start)
    # Vẽ một đường bao đỏ xung quanh các khuôn mặt được xác định bởi CNN
    for face in faces_cnn:
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y

        cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        img = cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
        text_on_image(img, print_cnn, x, y, w, h, 0, 255, 0)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# image = read_from_image(image_path)
image = read_from_camera()
recognition_face(image)

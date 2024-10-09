import sys
import cv2
import cv2.data


def display_image(img, description):
    cv2.imshow(description, img)
    cv2.waitKey()  # waits for key-tap to close the window
    cv2.destroyAllWindows()  # closes the window


def get_gray_image(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_by_height(img, target_height):
    (actual_height, actual_width) = img.shape[:2]

    aspect_ratio = actual_width / actual_height

    new_height = target_height
    new_width = int(target_height * aspect_ratio)
    return cv2.resize(img, (new_width, new_height))


def get_image_from_path(file_path):
    img = cv2.imread(file_path)
    if img is None:
        print('Error: Image not found or unable to load.')
        sys.exit(1)

    return img


def get_framed_cropped_face_image(grey_image):
    # ------------- Model for face detection---------#
    face_detector_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    # -------------detecting the faces--------------#
    faces = face_detector_cascade.detectMultiScale(grey_image, 1.3, 5)
    # If no faces our detected
    # if not faces:
    #    print('No face detected')
    #   #skip picture
    # --------- Bounding Face ---------#
    for (x, y, w, h) in faces:
        framed_image = cv2.rectangle(grey_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        display_image(framed_image, 'framed face')

        cropped_image = grey_image[y:y + h, x:x + w]
        return cropped_image  # TODO: change later on to return array (too large?)

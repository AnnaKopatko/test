import numpy as np
import cv2
from utils import order_points, order_contours

import cv2
import numpy as np

"""
The plan of the program:
1.Read the video and the image, init the video writer
2.While we go through the video first of all we find the contours on screen
3. We need to order the contours so we work with only one during the video
2. Then we turn it into the box with 4 points and order the points so we could use them in a wrap function
3.We use findHomography to map my picture into the video
"""

def main():
    cap = cv2.VideoCapture("/home/anna/PycharmProjects/test_problem/video.mp4")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    size = (frame_width, frame_height)
    my_image = cv2.imread("/home/anna/PycharmProjects/test_problem/my_image.JPG")
    result = cv2.VideoWriter('filename.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             50, size)

    while True:
        success, image = cap.read()
        if success:
            #i will find the correct contours by color
            #first of all we need to convert tha image to hsv
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            my_image = my_image.astype(np.uint8)

            #create a green mask
            mask = cv2.inRange(hsv_image, (50, 50, 50), (85, 255, 255))

            imask = mask > 0
            green = np.zeros_like(image, np.uint8)
            green[imask] = image[imask]

            #then we can find the contours on the simpler image

            gray_image = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
            _, threshold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(threshold, kernel, iterations=3)
            #find the contours, sort them and exclude the biggest one
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours= order_contours(contours)
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) <300000.0]

            #create a box with 4 points
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)

            #we need to init four old points witch would be just the shape of my image and the new points
            my_image = cv2.cvtColor(my_image, cv2.COLOR_BGR2RGB)
            my_image_size = my_image.shape
            new_points = order_points(box)
            old_points = np.float32([[0, 0], [my_image_size[1] + 1, 0], [my_image_size[1]+1, my_image_size[0]+1], [0, my_image_size[0]+1]])

            #first we find the matrix that would map the image to a new set of points, then wrap the image
            matrix, status = cv2.findHomography(old_points, new_points)
            warped = cv2.warpPerspective(my_image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

            #fill the area of the new points woth a color as a base
            cv2.fillConvexPoly(image, new_points.astype(int), (0, 255, 0))

            #sum the images
            image = image + warped
            #cv2.imshow('Display', image)
            result.write(image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    #close and release everything
    cap.release()
    result.release()
    cv2.destroyAllWindows()

    print("The video was successfully saved")

if __name__ == '__main__':
    main()
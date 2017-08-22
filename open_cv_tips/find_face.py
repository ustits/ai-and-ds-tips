"""
Inspired by:
http://docs.opencv.org/trunk/d7/d8b/tutorial_py_face_detection.html
http://www.pyimagesearch.com/2016/06/20/detecting-cats-in-images-with-opencv/
"""
import cv2

cat_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalcatface.xml')

cat_img = cv2.imread('images/cat_example.jpg')
gray = cv2.cvtColor(cat_img, cv2.COLOR_BGR2GRAY)

cat_faces = cat_cascade.detectMultiScale(gray, 1.3, 5)
for (i, (x, y, w, h)) in enumerate(cat_faces):
    cv2.rectangle(cat_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.putText(cat_img, "Cat #{}".format(i + 1), (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)


cv2.imshow('img', cat_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# display using matplotlib
# plt.imshow(img, cmap='gray')
# plt.show()

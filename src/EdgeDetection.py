import cv2 as cv2
import sys

sys.path.append('../')
from lib.sobel_operator import SobelOperator


class EdgeDetection:
    def __init__(self, image, approach):
        self._image = image
        self._approach = approach

        self._output = None
    
    def detect(self):
        if self._approach == "sobel_operator":
            edgeDetector = SobelOperator(self._image)
            self._output = edgeDetector.proceed()

        return self._output


if __name__ == "__main__" :
    image_path = "../sample/image1.jpg"
    approach = "sobel_operator"

    image = cv2.imread(image_path)
    detector = EdgeDetection(image, approach)
    output_image = detector.detect()

    cv2.imwrite("../sample/edge_image1.png", output_image)

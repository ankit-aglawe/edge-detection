import numpy as np 
import cv2 as cv2 


class SobelOperator:
    def __init__(self, image, average=False):
        self._image = image
        self._average = average

        self.__filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        self.__vertical_kernel = np.flip(self.__filter.T, axis=0)
        self.__horizontal_kernel = self.__filter

        self.__vertical_edge = None
        self.__horizontal_edge = None

        self._image = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)

    def proceed(self):
        self.__vertical_edge = self._get_vertical_edge()
        self.__horizontal_edge = self._get_horizontal_edge()

        print(self.__vertical_edge.shape)
        print(self.__horizontal_edge.shape)

        edgeImage = self._calculate_magnitude()

        return edgeImage

    def _calculate_magnitude(self):
        gradientMagnitude = np.sqrt(np.square(self.__vertical_edge) + np.square(self.__horizontal_edge))
        gradientMagnitude *= 255.0 / gradientMagnitude.max()

        return gradientMagnitude

    def _get_vertical_edge(self):

        imgRow, imgCol = self._image.shape
        kernelRow, kernelCol = self.__vertical_kernel.shape

        outputImage = np.zeros(self._image.shape)

        padHeight, padWidth = int((kernelRow -1)/2), int((kernelCol -1)/2)

        paddedImage = np.zeros((imgRow + (2 * padHeight), imgCol + (2 * padWidth)))

        paddedImage[padHeight:paddedImage.shape[0] - padHeight, padWidth:paddedImage.shape[1] - padWidth] = self._image
        
        for row in range(imgRow):
            for col in range(imgCol):
                outputImage[row, col] = np.sum(self.__vertical_kernel*paddedImage[row:row + kernelRow, col:col + kernelCol])
                if self._average:
                    outputImage[row, col] /= self.__vertical_kernel.shape[0] * self.__vertical_kernel.shape[1]

        return outputImage


    def _get_horizontal_edge(self):

        imgRow, imgCol = self._image.shape
        kernelRow, kernelCol = self.__horizontal_kernel.shape

        outputImage = np.zeros(self._image.shape)

        padHeight, padWidth = int((kernelRow -1)/2), int((kernelCol -1)/2)

        paddedImage = np.zeros((imgRow + (2 * padHeight), imgCol + (2 * padWidth)))

        paddedImage[padHeight:paddedImage.shape[0] - padHeight, padWidth:paddedImage.shape[1] - padWidth] = self._image
        
        for row in range(imgRow):
            for col in range(imgCol):
                outputImage[row, col] = np.sum(self.__horizontal_kernel*paddedImage[row:row + kernelRow, col:col + kernelCol])
                if self._average:
                    outputImage[row, col] /= self.__horizontal_kernel.shape[0] * self.__horizontal_kernel.shape[1]

        return outputImage


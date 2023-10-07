import cv2

class ImageIOs:
    def __init__(self, verbose) -> None:
        self.verbose = verbose
        pass
    def load_image(self, img_path: str, size = None):
        if size == None:
            size = (256, 256)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        if self.verbose:
            print(img.shape)

        return img
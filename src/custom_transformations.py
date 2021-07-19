import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import cv2
from skimage import io, transform

class ContrastTransform:
    """Adjust contrast of an Image.

    Args:
        contrast_factor (float) – How much to adjust the contrast. Can be any non negative number.
                                    0 gives a solid gray image,
                                    1 gives the original image while
                                    2 increases the contrast by a factor of 2.

    Source: https://pytorch.org/docs/stable/torchvision/transforms.html
    """

    def __init__(self, contrast_factor):
        self.contrast_factor = contrast_factor

    def __call__(self, img):
        return TF.adjust_contrast(img, self.contrast_factor)


class GammaTransform:
    """Perform gamma correction on an image.

    Args:
        gamma (float) – Non negative real number.
                        gamma larger than 1 make the shadows darker, while
                        gamma smaller than 1 make dark regions lighter.
        gain (float) – The constant multiplier.

    Source: https://pytorch.org/docs/stable/torchvision/transforms.html
    """

    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img):
        return TF.adjust_gamma(img, self.gamma, self.gain)


class BrightnessTransform:
    """Adjust brigthness.

    Args:
        brightness_factor (float) - 0 gives a black image,
                                    1 gives the original image while
                                    2 increases the brightness by a factor of 2.
    """

    def __init__(self, brightness_factor):
        self.brightness_factor = brightness_factor

    def __call__(self, img):
        return TF.adjust_brightness(img, self.brightness_factor)


class TranslateTransform:
    """Translate image horizontally and vertically.

    Args:
        translate (list, tuple) - horizontal and vertical translations
    """

    def __init__(self, translate):
        """
        Choose default values for other args than translate.
        """
        self.angle = 0
        # Typechecking of translate arg is already done in affine func
        self.translate = translate
        self.scale = 1
        self.shear = 0

    def __call__(self, img):
        return TF.affine(img, self.angle, self.translate, self.scale, self.shear)


class PadGrayTransform(object):
    """Pads the image with black to a square and then resizes and grayscales it and outputs it as a numpy array.

    Args:
        output_size (int): Desired output size of image (width=height)
        sample (numpy array): Image as a numpy array
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img):

        h, w = img.shape[:2]
        if h > w:
            lpad = int((h - w) / 2)
            img = cv2.copyMakeBorder(img, 0, 0, lpad, lpad, cv2.BORDER_CONSTANT)
        else:
            lpad = int((w - h) / 2)
            img = cv2.copyMakeBorder(img, lpad, lpad, 0, 0, cv2.BORDER_CONSTANT)

        return cv2.resize(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_AREA,
        )


class SquishTransform(object):
    """Squishes the image to given size and grayscales it and outputs it as a numpy array.
    REMARK: It could highly affect the quality of the image and possibly deform the image significantly

    Args:
        output_size (int): Desired output size of image (width=height)
        sample (numpy array): Image as a numpy array
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, img):

        img = transform.resize(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (self.output_size, self.output_size)
        )

        return img


class NaiveGrayTransform(object):
    """Crops, resizes, grayscales the image and outputs it as a numpy array.
    This is the same as transforms.Compose([transforms.CenterCrop(center_crop_size), transforms.Resize(img_size, Image.LANCZOS), transforms.Grayscale()] with PIL Image.

    Args:
        center_crop_size (int): The size to which images will be center cropped
        output_size (int): Desired output size of image (width=height)
        sample (numpy array): Image as a numpy array
    """

    def __init__(self, center_crop_size, output_size):
        self.center_crop_size = center_crop_size
        self.output_size = output_size

    def __call__(self, img):

        center_y = int(np.floor(img.shape[0] / 2))
        center_x = int(np.floor(img.shape[1] / 2))

        crop_half = int(np.floor(self.center_crop_size / 2))

        if self.center_crop_size % 2 == 0:

            y_top = center_y - crop_half
            y_bottom = center_y + crop_half

            x_left = center_x - crop_half
            x_right = center_x + crop_half
        else:
            y_top = center_y - crop_half
            y_bottom = center_y + crop_half + 1

            x_left = center_x - crop_half
            x_right = center_x + crop_half + 1

        return cv2.resize(
            cv2.cvtColor(img[y_top:y_bottom, x_left:x_right, :], cv2.COLOR_BGR2GRAY),
            (self.output_size, self.output_size),
            interpolation=cv2.INTER_AREA,
        )

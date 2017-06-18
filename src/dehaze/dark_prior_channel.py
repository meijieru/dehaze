from __future__ import print_function
from __future__ import division

import cv2
import numpy as np


def statistic(t):
    print("Shape:{}, Dtype: {}, Min: {}, Max: {}, Avg: {}".format(
        t.shape, t.dtype, t.min(), t.max(), t.sum() / t.size))


class DarkPriorChannelDehaze(object):
    """Dehaze using dark prior channel method from Kaiming He etc.

    Reference:
        [Single Image Haze Removal using Dark Channel Prior] \
                (http://kaiminghe.com/publications/cvpr09.pdf)
        [Guided Image Filtering] \
                (http://kaiminghe.com/publications/pami12guidedfilter.pdf)

    Args:
        wsize (int, optional): size of window for dark channel
        radius (int, optional): size of window for guided filter
        t_min (float, optional): minimum value of transmission
        ratio (float, optional): ratio of pixels used to estimate
            the atmosphere, refer to paper for details
        omega (float, optional): percantage of haze to be removed
        refine (bool, optional): whether to refine origin transmission
            estimation using guided filter
    """

    def __init__(self,
                 wsize=15,
                 radius=40,
                 t_min=0.1,
                 ratio=0.001,
                 omega=0.95,
                 refine=True):
        self.wsize = wsize
        self.radius = radius
        self.t_min = t_min
        self.ratio = ratio
        self.omega = omega
        self.refine = refine

    def dark_channel(self, img):
        """Get dark channel of an RGB image.

        Args:
            img (np.ndarray): [h, w, 3] size

        Return:
            An `np.ndarray` of size [h, w, 1]
        """
        h, w = img.shape[:2]
        img = np.min(img, axis=2, keepdims=False)
        padded = np.pad(img, ((self.wsize // 2, self.wsize // 2),
                              (self.wsize // 2, self.wsize // 2)), 'edge')

        img_distract = np.zeros([*img.shape, self.wsize**2])
        for i in range(self.wsize):
            for j in range(self.wsize):
                img_distract[..., i * self.wsize + j] = padded[i:i + h, j:j + w]

        return np.min(img_distract, axis=2, keepdims=True)

    def atmosphere(self, img, img_dark):
        """Estimate the atmosphere light condition.

        Args:
            img (np.ndarray): [h, w, 3] size
            img_dark (np.ndarray): [h, w, 1] size dark channel of the img

        Return:
            An `np.ndarray` of size [3]
        """
        img = img.reshape([-1, 3])
        img_dark = img_dark.flatten()
        top_k_index = np.argsort(img_dark)[-int(img_dark.size * self.ratio):]
        return np.max(np.take(img, top_k_index, axis=0), axis=0)

    def transmission(self, img, at, img_dark, omega=0.95):
        """Estimiate the transmission.

        Args:
            img (np.ndarray): [h, w, 3] size
            at (np.ndarray): [3] size atmosphere
            img_dark (np.ndarray): [h, w, 1] size dark channel of the img

        Return:
            An `np.ndarray` of size [h, w, 1]
        """
        t = 1.0 - self.omega * self.dark_channel(img / at)
        return t

    def soft_mat(self):
        """Refine the estimated `t` by matting techniques.

        NOTE: not implemented
        """
        raise NotImplementedError(
            "soft_mat is deprecated, guided_filter instead")

    def guided_filter(self, img, img_guide, epsilon=0.0001):
        """Smooth filter which keep edge property.

        Args:
            img (np.ndarray): [h, w] size
            img_guided (np.ndarray): [h, w] size guide image
            epsilon (float, optional): avoid `ZeroDivisionError`

        Return:
            A smoothed version of img, `np.ndarray` of size [h, w, 1]
        """
        g_mean = cv2.boxFilter(img_guide, cv2.CV_32F, (self.radius,
                                                       self.radius))
        g_corr = cv2.boxFilter(img_guide * img_guide, cv2.CV_32F, (self.radius,
                                                                   self.radius))
        i_mean = cv2.boxFilter(img, cv2.CV_32F, (self.radius, self.radius))
        gi_corr = cv2.boxFilter(img * img_guide, cv2.CV_32F, (self.radius,
                                                              self.radius))

        g_var = g_corr - g_mean * g_mean
        gi_cov = gi_corr - i_mean * g_mean

        a = gi_cov / (g_var + epsilon)
        b = i_mean - a * g_mean

        mean_a = cv2.boxFilter(a, cv2.CV_32F, (self.radius, self.radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (self.radius, self.radius))

        img = mean_a * img_guide + mean_b
        return img

    def reconstruct(self, img, at, t):
        """Get the final dehazed img."""
        return (img - at) / t + at

    def __call__(self, img):
        img = img.astype(np.float32)
        img_dark = self.dark_channel(img)
        at = self.atmosphere(img, img_dark)
        t = self.transmission(img, at, img_dark)

        if self.refine:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            t = self.guided_filter(t.reshape(img_gray.shape),
                                   img_gray).reshape(t.shape)

        t = np.maximum(self.t_min, t)
        return self.reconstruct(img, at, t)

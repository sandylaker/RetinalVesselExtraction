import numpy as np
import os
import glob
from PIL import Image
from scipy.ndimage import map_coordinates, gaussian_filter
import matplotlib.pyplot as plt


class DataAugmentation:

    def __init__(self, alpha=1000, sigma=30, spline_order=1, mode='nearest',
                 random_state=np.random):
        """
        Elastic deformation of image as described in [Simard2003]_...[Simard2003] Simard,
        Steinkraus and Platt, "Best Practices for Convolutional Neural Networks applied to Visual
        Document Analysis", in Proc. of the International Conference on Document Analysis and
        Recognition, 2003.

        Based on https://gist.github.com/oeway/2e3b989e0343f0884388ed7ed82eb3b0

        :param alpha: (float, optional) - the scale factor
        :param sigma: (float, optional) - the sigma of the gaussian filter
        :param spline_order: (int, optional) - the order of the spline interpolation
        :param mode: ({‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap}, optional) - the mode
                    parameter determines how the input array is extended beyond its boundaries
        :param random_state: (np.random.RandomState, optional) - the random state
        """
        self.alpha = alpha
        self.sigma = sigma
        self.spline_order = spline_order
        self.mode = mode
        self.random_state = random_state

    def __call__(self, n_augmentation=10):
        """
        Augment the training dataset by applying the elastic transformation. The augmented images
        and targets will be saved to the 'augment' directory. The prefix of the augmented images
        (targets) are composed of the prefix of the training images (targets) and the index of
        augmentation. For example, the first augmented image from '21_training.tif' will be named
        '2101_training.tif'

        :param n_augmentation: (int, optional) - Number of augmentation applied to each image (
                                target)
        """
        root = '../data/'
        save_image_dir = os.path.join(root, 'augment/images/')
        save_target_dir = os.path.join(root, 'augment/1st_manual/')

        train_image_paths = sorted(glob.glob(os.path.join(root, 'training/images/*_training.tif')))
        train_target_paths = sorted(glob.glob(os.path.join(root,
                                                           'training/1st_manual/*_manual1.gif')))
        train_mask_paths = sorted(glob.glob(os.path.join(root,
                                                         'training/mask/*_training_mask.gif')))

        for i, (image_path, target_path, mask_path) in enumerate(zip(
                train_image_paths, train_target_paths,train_mask_paths)):
            image = Image.open(image_path)
            target = Image.open(target_path)
            mask = Image.open(mask_path)

            for j in range(n_augmentation):
                image_trans, target_trans = self._elastic_transform(image, target, mask)

                # if the image has prefix 21, the the augmented images will be named with
                # prefix 2101 - 2110 if n_augmentation equals 10
                prefix_scale = 10 ** len(str(n_augmentation))
                image_name = '{}_training.tif'.format((i+21) * prefix_scale +
                                                             j + 1)
                image_trans.save(os.path.join(save_image_dir, image_name))

                target_name = '{}_manual1.gif'.format((i+21) * prefix_scale
                                                               + j + 1)
                target_trans.save(os.path.join(save_target_dir, target_name))
            print('{}th image and target are successfully augmented'.format(i+1))

    def _elastic_transform(self, image, target, mask):
        """
        apply the elastic transform to both image and target, and then apply the mask onto both
        transformed image and target

        :param image: (PIL.Image) - input image with 3 channels
        :param target: (PIL.Image) - input target with 1 channel
        :param mask: (PIL.Image) - mask with 1 channel, which is made up of 255s and 0s
        :return: tuple of (PIL.Image, PIL.Image) - tuple of (transformed image, transformed target)
        """
        image = np.asarray(image)
        target = np.asarray(target)
        mask = np.asarray(mask)
        assert image.ndim == 3
        if not self._check_shapes(image, target, mask):
            raise ValueError('image, target and mask have different shapes, image.shape: {}, '
                             'targe.shape: {}, mask.shape: {}'.format(image.shape, target.shape,
                                                                      mask.shape))
        # shape H*W*C
        shape = image.shape[:2]

        dx = gaussian_filter((self.random_state.rand(*shape) * 2 - 1),
                             self.sigma, mode='constant', cval=0) * self.alpha
        dy = gaussian_filter((self.random_state.rand(*shape) * 2 - 1),
                             self.sigma, mode='constant', cval=0) * self.alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = [np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))]
        image_trans = np.empty_like(image)
        target_trans = np.empty_like(target)
        for i in range(image.shape[2]):
            image_trans[:, :, i] = map_coordinates(image[:, :, i], indices, order=self.spline_order,
                                                   mode=self.mode).reshape(shape)

        target_trans[:, :] = map_coordinates(target[:, :], indices, order=self.spline_order,
                                       mode=self.mode).reshape(shape)

        # mask is made up of 255s and 0s, convert it to boolean array
        mask = np.array(mask == 255)
        image_trans = image_trans * (np.tile(mask[..., np.newaxis], [1, 1, 3]))
        target_trans = target_trans * mask

        return Image.fromarray(image_trans), Image.fromarray(target_trans)

    def _check_shapes(self, image, target, mask):
        return all([image.shape[:2] == target.shape, image.shape[:2] == mask.shape])


data_augmentation = DataAugmentation(alpha=3000, sigma=30, spline_order=3)
data_augmentation(n_augmentation=10)


# if __name__ == '__main__':
#     image = Image.open('../data/training/images/21_training.tif')
#     target = Image.open('../data/training/1st_manual/21_manual1.gif')
#     mask = Image.open('../data/training/mask/21_training_mask.gif')
#
#     aug = DataAugmentation(alpha=3000, sigma=30, spline_order=3)
#     image_trans, target_trans = aug._elastic_transform(image, target, mask)
#     fig = plt.figure()
#     fig.add_subplot(2, 2, 1)
#     plt.imshow(image)
#     fig.add_subplot(2, 2, 2)
#     plt.imshow(target)
#     fig.add_subplot(2, 2, 3)
#     plt.imshow(image_trans)
#     fig.add_subplot(2, 2, 4)
#     plt.imshow(target_trans)
#     plt.show()

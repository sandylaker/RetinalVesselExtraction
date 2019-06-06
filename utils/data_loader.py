import os
import torch
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils


class RetinaDataSet(Dataset):

    def __init__(
            self,
            root=None,
            train=True,
            augment=False,
            transform=None,
            shuffle=True,
            random_state=None):
        """
        The Dataset of Retinal Vessel Extraction Challenge

        :param root: (str, optional) - root directory, by default '../data/'
        :param train: (bool, optional) - if True, load the training data, otherwise load the test
                                        data.
        :param augment: (bool, optional) - If True, include the augmented data
        :param transform: (callable, optional) - a function or callable class, which has the form
                            transform(image, target=None) that performs the same transformation on
                            image (and target), and return the transformed image (and target).
                            By default, toTensor() is applied to image (and target)
        :param shuffle: (bool, optional) - If True, shuffle the data
        :param random_state: (np.random.RandomState, optional) - the specified random state
        """
        self.train = train

        if not root:
            self.root = '../data/'
        else:
            self.root = root

        if self.train:
            self.image_paths = glob.glob(
                os.path.join(self.root, 'training/images/*_training.tif'))
            if augment:
                self.image_paths = self.image_paths + glob.glob(
                    os.path.join(self.root, 'augment/images/*_training.tif')
                )
            self.image_paths = sorted(self.image_paths)

            self.target_paths = glob.glob(
                os.path.join(
                    self.root,
                    'training/1st_manual/*_manual1.gif'))
            if augment:
                self.target_paths = self.target_paths + glob.glob(
                    os.path.join(self.root, 'augment/1st_manual/*_manual1.gif')
                )
            self.target_paths = sorted(self.target_paths)

            if shuffle:
                self.image_paths, self.target_paths = self._shuffle_paths(
                    self.image_paths, self.target_paths, random_state=random_state)
        else:
            self.image_paths = sorted(
                glob.glob(
                    os.path.join(
                        self.root,
                        'test/images/*_test.tif')))
            if shuffle:
                self.image_paths = self._shuffle_paths(
                    self.image_paths, random_state=random_state)

        if not transform:
            self.transform = self._default_tranform
        else:
            self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])

        if self.train:
            target = Image.open(self.target_paths[index])
            return self.transform(image, target)
        else:
            return self.transform(image)

    def __len__(self):
        return len(self.image_paths)

    def _default_tranform(self, image, target=None):
        to_tensor = transforms.ToTensor()
        if target:
            return to_tensor(image), to_tensor(target)
        else:
            return to_tensor(image)

    def _shuffle_paths(
            self,
            image_paths,
            target_paths=None,
            random_state=None):
        """

        shuffle the list of paths of images (and targets)

        :param image_paths: list of strings, paths of images
        :param target_paths: list of targets, paths of targets
        :param random_state: numpy.random.RandomState, specify the random state
        :return: in train mode: return a tuple of (list of image paths, list of target paths);
                 in evaluation mode: return a list of image paths
        """
        if not random_state:
            random_state = np.random.RandomState()
        if not target_paths:
            random_state.shuffle(image_paths)
            return image_paths

        zipped = list(zip(image_paths, target_paths))
        random_state.shuffle(zipped)
        unzipped = zip(*zipped)
        image_paths, target_paths = unzipped
        return list(image_paths), list(target_paths)


class TrainValidationSplit:

    def __init__(self, train_size=0.6):
        if train_size >= 1.0 or train_size <= 0:
            raise ValueError(
                'train_size can only be a float in interval (0, 1)')
        self.train_size = train_size

    def __call__(self, dataset):
        total_len = len(dataset)
        train_len = int(self.train_size * total_len)
        valid_len = total_len - train_len
        train_dataset, valid_dataset = random_split(
            dataset, [train_len, valid_len])
        return train_dataset, valid_dataset


if __name__ == '__main__':
    r = RetinaDataSet(train=True, shuffle=False, augment=True,
                      random_state=np.random.RandomState(
                          12))
    print(r.image_paths)
    print(r.target_paths)
    print(len(r))
    image, target = r[-2]
    print(torch.max(target))
    print(image.size(), target.size())
    to_PIL = transforms.ToPILImage()
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(to_PIL(image))
    fig.add_subplot(1, 2, 2)
    plt.imshow(to_PIL(target))
    plt.show()

    # Spliter = TrainValidationSplit(train_size=0.6)
    # train_dataset, valid_dataset = Spliter(r)
    # print(len(train_dataset), len(valid_dataset))
    # image, target = valid_dataset[0]
    # to_PIL = transforms.ToPILImage()
    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(to_PIL(image))
    # fig.add_subplot(1, 2, 2)
    # plt.imshow(to_PIL(target))
    # plt.show()

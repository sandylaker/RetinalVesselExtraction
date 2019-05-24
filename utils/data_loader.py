import os
import torch
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class RetinaDataSet(Dataset):

    def __init__(self, root=None, train=True, train_size=1.0,
                 transform=None, shuffle=True, random_state=None):
        self.train = train

        if train_size <= 0 or train_size > 1:
            raise ValueError('train_size can only be in the interval (0,1]')
        if not root:
            self.root = '../data/'
        else:
            self.root = root

        if self.train:
            self.image_paths = sorted(glob.glob(os.path.join(self.root,
                                                        'training/images/*_training.tif')))
            self.target_paths = sorted(glob.glob(os.path.join(self.root,
                                                      'training/1st_manual/*_manual1.gif')))
            if shuffle:
                self.image_paths, self.target_paths = self._shuffle_paths(
                    self.image_paths, self.target_paths, random_state=random_state)

            self.image_paths = self.image_paths[:int(train_size * len(self.target_paths))]
            self.target_paths = self.target_paths[:int(train_size * len(self.target_paths))]
        else:
            self.image_paths = sorted(glob.glob(os.path.join(self.root,
                                                             'test/images/*_test.tif')))
            if shuffle:
                self.image_paths = self._shuffle_paths(self.image_paths, random_state=random_state)
            

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
        transform = transforms.ToTensor()
        if target:
            return transform(image), transform(target)
        else:
            return transform(image)

    def _shuffle_paths(self, image_paths, target_paths=None, random_state=None):
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





if __name__ == '__main__':
    r = RetinaDataSet(train_size=0.5, train=True, shuffle=False,
                      random_state=np.random.RandomState(
        123))
    print(r.image_paths)
    print(len(r))
    image, target = r[0]
    print(image.size(), target.size())
    to_PIL = transforms.ToPILImage()
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(to_PIL(image))
    fig.add_subplot(1, 2, 2)
    plt.imshow(to_PIL(target))
    plt.show()
    # print(r.target_paths)

import torch
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from utils.data_loader import RetinaDataSet


def show_images(images, targets=None, predicts=None, **kwargs):
    to_PIL = transforms.ToPILImage()
    image_grid = make_grid(images, **kwargs)
    f1 = plt.figure()
    plt.title('Images')
    plt.imshow(to_PIL(image_grid))
    if targets:
        targets_grid = make_grid(targets, **kwargs)
        f2 =plt.figure()
        plt.title('Targets')
        plt.imshow(to_PIL(targets_grid))
    if predicts:
        predicts_grid = make_grid(predicts, **kwargs)
        f3 = plt.figure()
        plt.title('Predicts')
        plt.imshow(to_PIL(predicts_grid))
    plt.show()


if __name__ == '__main__':
    r = RetinaDataSet()
    images = []
    targets = []
    for i in range(len(r)):
        image, target = r[i]
        images.append(image)
        targets.append(target)
    print(len(images))
    print(len(targets))
    show_images(images, targets, nrow=5)
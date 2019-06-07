import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import *
from src import UNet
import os

def predict(model:UNet, root=None, threshold=0.5):
    """
    """
    if model.pad is None:
        print('UNet has no padding layer')
    else:
        print(model.pad.padding)
    if root is None:
        root = '../data/predict/'

    test_dataset = RetinaDataSet(train=False,
                                 shuffle=False,
                                 )
    # set batch_size = 1 to make naming the predicted image easier
    test_loader = DataLoader(test_dataset, batch_size=1)
    for i, test_data in enumerate(test_loader):
        predict_name = str(i+1) + '.png'
        store_path = os.path.join(root, predict_name)

        image, mask = test_data

        logits = model(image)
        proba = torch.sigmoid(logits)
        predicted = (proba < threshold).type(torch.float)

        # predicted is 4d Tensor, (1, 1, H, W)
        predicted = (predicted * mask)[0]

        to_PIL = transforms.ToPILImage()
        predicted = to_PIL(predicted)

        predicted.save(store_path)
        print('Predicted image {} saved'.format(i+1))




if __name__ =='__main__':
    # unet = UNet(in_channels=3, padding=(117, 118, 108, 108))
    # predict(model=unet)
    from PIL import Image
    img = Image.open('../data/predict/1.png')
    img = 1 - (np.asarray(img) == 255).astype(int)
    print(np.unique(img))
    img = Image.fromarray((img * 255).astype(float))
    plt.imshow(img)
    plt.colorbar()
    plt.show()

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import *
from src.unet import UNet
from src.unet_plusplus import UNetPlusPlus
import os


def predict(model, root=None, threshold=0.5, prune_level=4, device='gpu'):
    """
    """
    if device == 'gpu':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(device)
    model.to(device)
    if model.pad is None:
        print('UNet or UNet++ has no padding layer')
    else:
        print(model.pad.padding)
    if root is None:
        root = '../data/predict/'

    test_dataset = RetinaDataSet(train=False,
                                 shuffle=False,
                                 )
    # set batch_size = 1 to make naming the predicted image easier
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    with torch.no_grad():
        model.eval()
        for i, test_data in enumerate(test_loader):
            predict_name = str(i+1) + '.png'
            store_path = os.path.join(root, predict_name)

            image, mask = test_data[0].type(torch.float).to(device), test_data[1].type(torch.float).to(device)

            if isinstance(model, UNet):
                logits = model(image, train_mode=False)
                print('Model: UNet')
            elif isinstance(model, UNetPlusPlus):
                logits = model(image, train_mode=False, prune_level=prune_level)
                print('Model: UNet++')
            else:
                raise ValueError('Invalid Model: should be either UNet or UNet++,'
                                 'but got {}'.format(type(model).__name__))
            proba = torch.sigmoid(logits)
            predicted = (proba > threshold).type(torch.float)

            # predicted is 4d Tensor, (1, 1, H, W)
            predicted = (predicted * mask)[0]

            to_PIL = transforms.ToPILImage()
            # copy the predicted Tensor to CPU
            predicted_copy = predicted.cpu()
            predicted_image = to_PIL(predicted_copy)

            predicted_image.save(store_path)
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

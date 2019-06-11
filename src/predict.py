import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import *
from src import UNet
import os

def predict(model:UNet, root=None, threshold=0.5, valid_loader=None):
    """
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.eval()
    model.to(device)
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

        image, mask = test_data[0].type(torch.float).to(device), test_data[1].type(torch.float).to(device)

        logits = model(image, train_mode=False)
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
        
    # predict on validation set and save the false negative part along with the ground truths
    for i, valid_data in enumerate(valid_loader):
        fn_name = str(i+1) + '_fn.png'
        fn_path = os.path.join('../data/false_negative/', fn_name)
        tg_path = os.path.join('../data/false_negative/', '{}_target.png'.format(i+1))
        
        image, target = valid_data[0].type(torch.float).to(device), valid_data[1].type(torch.float).to(device)
        
        logits = model(image, train_mode=False)
        proba = torch.sigmoid(logits)
        predicted = (proba > threshold).type(torch.float)
        false_negative = (target - predicted * target)[0]
        to_PIL = transforms.ToPILImage()
        fn = false_negative.cpu()
        fn = to_PIL(fn)
        fn.save(fn_path)
        print('False negative image {} saved'.format(i+1))
        
        tg = target[0].cpu()
        tg = to_PIL(tg)
        tg.save(tg_path)
        print('Target image {} of valid data saved'.format(i+1))


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

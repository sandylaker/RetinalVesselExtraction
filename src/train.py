import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import *
from src import UNet
import os

def train(**kwargs):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    n_epochs = kwargs['n_epochs']
    lr = kwargs['learning_rate']
    batch_size = kwargs['batch_size']
    weight_decay = kwargs['weight_decay']

    dataset = RetinaDataSet(train=True, augment=True)
    train_dataset, valid_dataset = TrainValidationSplit(train_size=0.8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model = UNet(in_channels=3,
                 n_classes=2,
                 return_logits=True,
                 padding=(117, 118, 108, 108),
                 pad_value=0)
    model.train()
    model.to(device)

    criterion = BCEWithLogitsLoss2d()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            images, targets = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            output_logits = model(images)
            loss = criterion(output_logits, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print every 5 mini-batches
            if (i+1) % 5 == 0:
                print('[%d, %d] loss: %f' % (epoch + 1, i + 1, running_loss/5))
                running_loss = 0

        # save state every epochs
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_state_dict': criterion.state_dict(),
        }, os.path.join('./', 'check_point'))

    print('finish training')

    # validation
    score = DiceScoreWithLogits()
    dice_score_list = []
    total = 0

    with torch.no_grad():
        # set the model to evaluation mode
        model.eval()

        for valid_data in valid_dataset:
            total += 1
            images, targets = valid_data[0].to(device), valid_data[1].to(device)
            output_logits = model(images)
            dice_score_list.append(score(output_logits, targets))

    print('Mean Dice Score: %f' % torch.tensor(dice_score_list, dtype=torch.float32).mean().item())


if __name__ == '__main__':
    kwargs = {
        'n_epochs': 10,
        'learning_rate': 0.01,
        'weight_decay': 0.001,
        'batch_size': 10
    }
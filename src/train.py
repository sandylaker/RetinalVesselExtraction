import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import *
from src import UNet
import os

def prepare_training_data(train_size=0.8, batch_size=3, random_state=None):
    dataset = RetinaDataSet(train=True, augment=True, random_state=random_state)
    train_valid_splitter = TrainValidationSplit(train_size)
    train_dataset, valid_dataset = train_valid_splitter(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, valid_loader


def train(train_loader, valid_loader, resume=False, n_epochs=30, lr=0.001, weight_decay=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = UNet(in_channels=3,
                 n_classes=2,
                 return_logits=True,
                 padding=(117, 118, 108, 108),
                 pad_value=0)

    criterion = BCEWithLogitsLoss2d()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if not resume:
        model.train()
        model.to(device)
    else:
        check_point = torch.load('check_point')
        model.load_state_dict(check_point['model_state_dict'])
        optimizer.load_state_dict(check_point['optimizer_state_dict'])
        model.train()
        model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    for epoch in range(n_epochs):
        print('Epoch: {}'.format(epoch + 1))

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

    score = DiceScoreWithLogits()
    dice_score_list = []

    with torch.no_grad():
        # set the model to evaluation mode
        model.eval()

        for valid_data in valid_loader:
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
    train(**kwargs)
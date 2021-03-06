import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import *
from src.unet import UNet
from src.unet_plusplus import UNetPlusPlus
import os

def prepare_training_data(train_size=0.8, batch_size=3, random_state=None):
    dataset = RetinaDataSet(train=True, augment=True, random_state=random_state)
    train_valid_splitter = TrainValidationSplit(train_size)
    train_dataset, valid_dataset = train_valid_splitter(dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    return train_loader, valid_loader


def train_unet(train_loader,
               valid_loader,
               resume=False,
               n_epochs=30,
               lr=0.001,
               weight_decay=0.001,
               loss_type='soft_dice',
               add_out_layers=False,
               ):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    padding = (117, 118, 108, 108)
    model = UNet(in_channels=3,
                 n_classes=2,
                 return_logits=True,
                 padding=padding,
                 pad_value=0,
                 add_out_layers=add_out_layers)

    criterion =_generate_loss(loss_type)
    # add loss for low-res outputs at /2, /4, /8, /16
    if add_out_layers:
        criterion_2 = _generate_loss('bce')
        criterion_4 = _generate_loss('bce')
        criterion_8 = _generate_loss('bce')
        criterion_16 = _generate_loss('bce')

    if not resume:
        model.train()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        check_point = torch.load('../check_point/check_point')
        model.load_state_dict(check_point['model_state_dict'])
        model.train()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.load_state_dict(check_point['optimizer_state_dict'])

    # decay coefficients (weights) of low-res losses /16, /8, /4, /2
    coefs = [0.5 ** k for k in range(3, -1, -1)]
    
    for epoch in range(n_epochs):
        print('Epoch: {}'.format(epoch + 1))
        
        coefs = [0.5 * coef for coef in coefs]
        
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            images, targets = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            output_logits = model(images)
            if not add_out_layers:
                # in this case, the output_logits will be a Tensor
                loss_1 = criterion(output_logits, targets)
                loss_2 = torch.tensor(0.0).type_as(loss_1)
                loss_4 = torch.tensor(0.0).type_as(loss_1)
                loss_8 = torch.tensor(0.0).type_as(loss_1)
                loss_16 = torch.tensor(0.0).type_as(loss_1)
                loss_1.backward()
            else:
                # pad the targets and down sampling, in oder to align them with low-res outputs
                targets_padded = F.pad(targets, list(padding), mode='constant', value=0)
                # down sample to resolution /2
                targets_2 = F.max_pool2d(targets_padded, kernel_size=2, stride=2)
                targets_4 = F.max_pool2d(targets_2, kernel_size=2, stride=2)
                targets_8 = F.max_pool2d(targets_4, kernel_size=2, stride=2)
                targets_16 = F.max_pool2d(targets_8, kernel_size=2, stride=2)
                
                loss_16 = criterion_16(output_logits[4], targets_16)
                loss_8 = criterion_8(output_logits[3], targets_8)
                loss_4 = criterion_4(output_logits[2], targets_4)
                loss_2 = criterion_2(output_logits[1], targets_2)
                loss_1 = criterion(output_logits[0], targets)
                losses = sum([coefs[0] * loss_16, 
                              coefs[1] * loss_8, 
                              coefs[2] * loss_4, 
                              coefs[3] * loss_2, 
                              loss_1])
                losses.backward()
                
            optimizer.step()

            # print statistics
            running_loss += loss_1.item()
            # print every 5 mini-batches
            if (i+1) % 5 == 0:
                print('[%d, %d] loss: %f' % (epoch + 1, i + 1, running_loss/5))
                running_loss = 0

        # save state every epochs
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join('../check_point/', 'check_point'))

    print('finish training')
    validate(model, valid_loader)


def train_unet_plusplus(train_loader,
                        valid_loader,
                        resume=False,
                        n_epochs=30,
                        lr=0.001,
                        weight_decay=0.001,
                        loss_type='combined',
          ):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    padding = (117, 118, 108, 108)
    model = UNetPlusPlus(in_channels=3,
                         n_classes=2,
                         return_logits=True,
                         padding=padding,
                         pad_value=0)

    criterion0_4 = _generate_loss(loss_type)
    criterion0_3 = _generate_loss(loss_type)
    criterion0_2 = _generate_loss(loss_type)
    criterion0_1 = _generate_loss(loss_type)

    if not resume:
        model.train()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        check_point = torch.load('../check_point/check_point')
        model.load_state_dict(check_point['model_state_dict'])
        model.train()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.load_state_dict(check_point['optimizer_state_dict'])

    for epoch in range(n_epochs):
        print('Epoch: {}'.format(epoch + 1))

        running_loss = 0.0
        for i, data in enumerate(train_loader):
            images, targets = data[0].to(device), data[1].to(device)



            optimizer.zero_grad()

            # output is a list [output0_1, output0_2, output0_3, output0_4]
            output_logits = model(images, train_mode=True)

            loss0_1 = criterion0_1(output_logits[0], targets)
            loss0_2 = criterion0_2(output_logits[1], targets)
            loss0_3 = criterion0_3(output_logits[2], targets)
            loss0_4 = criterion0_4(output_logits[3], targets)
            losses = sum([loss0_1, loss0_2, loss0_3, loss0_4])
            # losses = criterion0_4(output_logits, targets)
            losses.backward()

            optimizer.step()

            # print statistics
            running_loss += losses.item()
            # print every 5 mini-batches
            if (i + 1) % 5 == 0:
                print('[%d, %d] loss: %f' % (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0

        # save state every epochs
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join('../check_point/', 'check_point'))
        
    validate(model, valid_loader, prune_level=4)
    print('finish training')
    
def validate(model, valid_loader, prune_level=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    # test on validation set
    score = DiceScoreWithLogits()
    dice_score_list = []

    with torch.no_grad():
        # set the model to evaluation mode
        model.eval()

        for i, valid_data in enumerate(valid_loader):
            images, targets = valid_data[0].to(device), valid_data[1].to(device)
            if isinstance(model, UNetPlusPlus):
                output_logits = model(images, train_mode=False, prune_level=prune_level)
            elif isinstance(model, UNet):
                output_logits = model(images, train_mode=False)
            else:
                raise ValueError('Invalid Model')
            dice_score_list.append(score(output_logits, targets))

    print('Mean Dice Score: %f' % torch.tensor(dice_score_list, dtype=torch.float32).mean().item())


def _generate_loss(loss_type, **kwargs):
    if loss_type == 'bce':
        criterion = BCEWithLogitsLoss2d(**kwargs)
    elif loss_type == 'soft_dice':
        criterion = SoftDiceLoss(**kwargs)
    elif loss_type == 'combined':
        criterion = CombinedLoss(**kwargs)
    else:
        raise ValueError("loss type can be 'bce' or 'soft_dice or 'combined'")

    return criterion

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import json
from torch.utils.tensorboard import SummaryWriter
import pdb

import torch.nn as nn
from settings import process_args
from models import ResNet18, Net
from dataset import *


use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
torch.backends.cudnn.benchmark = True


def train_epoch(model, current_epoch, criterion, criterion_cross_entropy, optimizer, dataset, args, log):
    running_triplet_loss = 0
    running_cross_loss = 0
    total_loss = 0
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True,
                                              num_workers=0)
    num_iters = len(trainloader)
    conf = torch.zeros((args['n_classes'], args['n_classes']))
    for i, data in enumerate(trainloader):
        anchor, positive, negative, label = data #return anc, pos, neg (pair) (person1_img, person2_img) 1,2yes 3 no  = data
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative= negative.to(device)
        optimizer.zero_grad()

        prediction, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)


        triplet_loss = criterion(anchor_features, positive_features, negative_features)
        cross_loss = criterion_cross_entropy(prediction, label)
        loss = triplet_loss + cross_loss
        loss.backward()
        optimizer.step()

        running_cross_loss += cross_loss.item()
        running_triplet_loss += triplet_loss.item()
        total_loss += loss.item()

        if (i % args['log_freq']) == 0:
            log.add_scalar('cross_entropy_loss', running_cross_loss, epoch * num_iters + i)
            acc = conf.diagonal().sum() / (conf.sum() + 0.00001)
            log.add_scalar('training_acc', acc, epoch * num_iters + i)
            log.add_scalar('triplet_loss', running_triplet_loss, epoch * num_iters + i)
            print('Epoch: {:03d} \t Iteration: {:03d}/{:03d} \t Trielet_loss: {:.04f}  \t Cross_entropy_loss: {:.04f} \t Acc: {:.04f}'.format(current_epoch, i + 1,
                                                                                                       num_iters,running_triplet_loss / (i + 1),
                                                                                                        running_cross_loss / (
                                                                                                        i + 1),acc))
            running_triplet_loss = 0
            running_cross_loss = 0
        # if (i % args['save_freq']) == 0:
        #     torch.save(net.state_dict(), os.path + "Epoch" + str(epoch) + '.pth')

    return total_loss/num_iters, model

def val_epoch(model, current_epoch ,criterion, val_dataset, args):
    with torch.no_grad():
        running_triplet_loss = 0
        running_cross_loss = 0
        total_loss = 0
        trainloader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=True,
                                                  num_workers=0)
        num_iters = len(trainloader)
        conf = torch.zeros((args['n_classes'], args['n_classes']))
        for i, data in enumerate(trainloader):
            anchor, positive, negative, label = data  # return anc, pos, neg (pair) (person1_img, person2_img) 1,2yes 3 no  = data
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            optimizer.zero_grad()

            prediction, anchor_features = model(anchor)
            _, positive_features = model(positive)
            _, negative_features = model(negative)

            triplet_loss = criterion(anchor_features, positive_features, negative_features)
            cross_loss = criterion_cross_entropy(prediction, label)
            loss = triplet_loss + cross_loss
            loss.backward()
            optimizer.step()

            running_cross_loss += cross_loss.item()
            running_triplet_loss += triplet_loss.item()
            total_loss += loss.item()

            if (i % args['log_freq']) == 0:
                log.add_scalar('cross_entropy_loss', running_cross_loss, epoch * num_iters + i)
                acc = conf.diagonal().sum() / (conf.sum() + 0.00001)
                log.add_scalar('training_acc', acc, epoch * num_iters + i)
                log.add_scalar('triplet_loss', running_triplet_loss, epoch * num_iters + i)
                print(
                    'Epoch: {:03d} \t Iteration: {:03d}/{:03d} \t Trielet_loss: {:.04f}  \t Cross_entropy_loss: {:.04f} \t Acc: {:.04f}'.format(
                        current_epoch, i + 1,
                        num_iters, running_triplet_loss / (i + 1),
                        running_cross_loss / (
                                i + 1), acc))
                running_triplet_loss = 0
                running_cross_loss = 0



    return total_loss/num_iters, correct/num_iters


if __name__ == '__main__':
    args = process_args()
    log = SummaryWriter(os.path.join(args['log_out_dir'], '{:s}'.format(args['exp_name'])))
    if not os.path.exists(os.path.join(args['check_point_out_dir'], args['exp_name'])):
        os.makedirs(os.path.join(args['check_point_out_dir'], args['exp_name']))
    if not os.path.exists(args['runargs_out_dir']):
        os.makedirs(args['runargs_out_dir'])
    with open(os.path.join(args['runargs_out_dir'], '{:s}.json'.format(args['exp_name'])), 'wt') as fp:
        json.dump(args, fp, indent=2)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    #dataset = SimulationMaskedFacesDataset_Classification(split_file='train.txt', transforms=train_transform)

    #pdb.set_trace()
    r2r_train_dataset = RealMaskedFacesDataset_Classification(split_file='toy.txt', transforms=train_transform)
    r2r_val_dataset = RealMaskedFacesDataset_Classification(split_file='toyval.txt', transforms=val_transform)

    current_best_val_acc = 0
    model = ResNet18()
    model = model.to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    criterion = criterion.to(device)
    criterion_cross_entropy = nn.CrossEntropyLoss()
    criterion_cross_entropy = criterion_cross_entropy.to((device))
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['moment'])

    for epoch in range(args['num_epoch']):
        train_loss, model = train_epoch(model, epoch, criterion, criterion_cross_entropy, optimizer, r2r_train_dataset, args, log)
        val_loss, val_acc = val_epoch(model, epoch, criterion, r2r_val_dataset, args)

        if val_acc > current_best_val_acc:
            current_best_val_acc = val_acc
            name = 'best_so_far.pth'
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_loss,
                        'val_acc': val_acc
                        }, os.path.join(args['check_point_out_dir'], args['exp_name'], name))
            print('Model checkpoint written! :{:s}'.format(name))
        log.close()










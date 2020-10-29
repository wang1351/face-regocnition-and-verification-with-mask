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
from triplet_dataset import *


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
    sum_0 = 0
    sum_1 = 0
    total_0 = 0
    total_1 = 0
    for i, data in enumerate(trainloader):
        anchor, positive, negative, labels = data
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative= negative.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        prediction, anchor_features = model(anchor)
        _, positive_features = model(positive)
        _, negative_features = model(negative)

        with torch.no_grad():
            preds = torch.argmax(prediction, dim=1)
            for idx in range(args['batch_size']):
                if labels[idx] == 0:
                    if preds[idx] == 0:
                        sum_0 += 1
                    total_0 += 1
                else:
                    if preds[idx] == 1:
                        sum_1 += 1
                    total_1 += 1

        triplet_loss = criterion(anchor_features, positive_features, negative_features)
        cross_loss = criterion_cross_entropy(prediction, labels)
        loss = triplet_loss + cross_loss
        loss.backward()
        optimizer.step()

        running_cross_loss += cross_loss.item()
        running_triplet_loss += triplet_loss.item()
        total_loss += loss.item()

        if (i % args['log_freq']) == 0:
            log.add_scalar('cross_entropy_loss', running_cross_loss, epoch * num_iters + i)
            acc = (sum_0 + sum_1) / (total_0 + total_1)
            macro_acc = (sum_0 / total_0 + sum_1 / total_1) / 2
            log.add_scalar('training_micro_acc', acc, epoch * num_iters + i)
            log.add_scalar('training_macro_acc', macro_acc, epoch * num_iters + i)
            log.add_scalar('triplet_loss', running_triplet_loss, epoch * num_iters + i)
            print('Epoch: {:03d} \t Iteration: {:03d}/{:03d} \t Trielet_loss: {:.04f}  \t Cross_entropy_loss: {:.04f} \t Micro_acc: {:.04f} \t Macro_acc: {:.04f}'.format(current_epoch, i + 1,
                                                                                                       num_iters,running_triplet_loss / (i + 1),
                                                                                                        running_cross_loss / (
                                                                                                        i + 1),acc, macro_acc))
            running_triplet_loss = 0
            running_cross_loss = 0

    return total_loss/num_iters, model

def val_epoch(model, current_epoch ,criterion, val_dataset, args):
    with torch.no_grad():
        running_triplet_loss = 0
        running_cross_loss = 0
        total_loss = 0
        trainloader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=True,
                                                  num_workers=0)
        num_iters = len(trainloader)
        sum_0 = 0
        sum_1 = 0
        total_0 = 0
        total_1 = 0
        for i, data in enumerate(trainloader):
            anchor, positive, negative, labels = data  # return anc, pos, neg (pair) (person1_img, person2_img) 1,2yes 3 no  = data
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            prediction, anchor_features = model(anchor)
            preds = torch.argmax(prediction, dim=1)
            for idx in range(args['batch_size']):
                if labels[idx] == 0:
                    if preds[idx] == 0:
                        sum_0 += 1
                    total_0 += 1
                else:
                    if preds[idx] == 1:
                        sum_1 += 1
                    total_1 += 1

            _, positive_features = model(positive)
            _, negative_features = model(negative)

            triplet_loss = criterion(anchor_features, positive_features, negative_features)
            cross_loss = criterion_cross_entropy(prediction, labels)
            loss = triplet_loss + cross_loss

            running_cross_loss += cross_loss.item()
            running_triplet_loss += triplet_loss.item()
            total_loss += loss.item()

            if (i % args['log_freq']) == 0:
                log.add_scalar('val_cross_entropy_loss', running_cross_loss, epoch * num_iters + i)
                acc = (sum_0 + sum_1) / (total_0 + total_1)
                macro_acc = (sum_0 / total_0 + sum_1 / total_1) / 2
                log.add_scalar('val_micro_acc', acc, epoch * num_iters + i)
                log.add_scalar('val_macro_acc', macro_acc, epoch * num_iters + i)
                log.add_scalar('val_triplet_loss', running_triplet_loss, epoch * num_iters + i)
                print(
                    'Epoch: {:03d} \t Iteration: {:03d}/{:03d} \t Trielet_loss: {:.04f}  \t Cross_entropy_loss: {:.04f} \t Micro_acc: {:.04f} \t Macro_acc: {:.04f}'.format(
                        current_epoch, i + 1,
                        num_iters, running_triplet_loss / (i + 1),
                        running_cross_loss / (
                                i + 1), acc, macro_acc))
                running_triplet_loss = 0
                running_cross_loss = 0

    return total_loss/num_iters, acc


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

    tri_val_dataset = RealMaskedFacesDataset_Triplet(split_file='train.txt', transforms=val_transform)

    current_best_val_acc = -1
    model = ResNet18()
    model = model.to(device)
    criterion = nn.TripletMarginLoss(margin=1.0, p=2)
    criterion = criterion.to(device)
    criterion_cross_entropy = nn.CrossEntropyLoss()
    criterion_cross_entropy = criterion_cross_entropy.to((device))
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['moment'])

    for epoch in range(args['num_epoch']):
        tri_train_dataset = RealMaskedFacesDataset_Triplet(split_file='val.txt', transforms=train_transform)
        train_loss, model = train_epoch(model, epoch, criterion, criterion_cross_entropy, optimizer, tri_train_dataset, args, log)
        val_loss, val_acc = val_epoch(model, epoch, criterion, criterion_cross_entropy, tri_val_dataset, args)

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










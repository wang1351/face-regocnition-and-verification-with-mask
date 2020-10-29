import torchvision.transforms as transforms
import torch.optim as optim
import json
from torch.utils.tensorboard import SummaryWriter
import pdb
from torch.optim import lr_scheduler

import torch.nn as nn
from settings import process_args
from models import ResNet18, Net
from dataset import *
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
torch.backends.cudnn.benchmark = True

PATH_TO_MODEL = '/Users/wzy/Downloads/best_so_far_cpu.pth'

def test_epoch(model, current_epoch ,criterion, val_dataset, args):
    with torch.no_grad():
        running_loss = 0
        total_loss = 0
        sum_0 = 0
        sum_1 = 0
        total_0 = 0
        total_1 = 0
        trainloader = torch.utils.data.DataLoader(val_dataset, batch_size=args['batch_size'], shuffle=True,
                                                  num_workers=0)
        num_iters = len(trainloader)

        for i, data in enumerate(trainloader):
            input, labels = data
            input = input.to(device)
            labels = labels.to(device)
            outputs, features = model(input)

            preds = torch.argmax(outputs, dim=1)
            for idx in range(args['batch_size']):
                if labels[idx] == 0:
                    if preds[idx] == 0:
                        sum_0 += 1
                    total_0 += 1
                else:
                    if preds[idx] == 1:
                        sum_1 += 1
                    total_1 += 1


            loss = criterion(outputs, labels)

            total_loss += loss.item()
            running_loss += loss.item()

            if (i % args['log_freq']) == 0:
                log.add_scalar('val_loss', running_loss, epoch * num_iters + i)
                acc = (sum_0 + sum_1) / (total_0 + total_1)
                macro_acc = (sum_0/total_0 + sum_1/total_1)/2
                log.add_scalar('val_acc', acc, epoch * num_iters + i)
                print(
                    'Epoch: {:03d} \t Iteration: {:03d}/{:03d} \t Loss: {:.04f}  \t Micro_acc: {:.04f} \t Macro_acc: {:.04f}'.format(
                        current_epoch, i + 1,
                        num_iters,
                        running_loss / (
                                i + 1), acc,
                        macro_acc))
                running_loss = 0
    return total_loss/num_iters, macro_acc

if __name__ == '__main__':
    args = process_args()
    log = SummaryWriter(os.path.join(args['log_out_dir'], '{:s}'.format(args['exp_name'])))
    if not os.path.exists(os.path.join(args['check_point_out_dir'], args['exp_name'])):
        os.makedirs(os.path.join(args['check_point_out_dir'], args['exp_name']))
    if not os.path.exists(args['runargs_out_dir']):
        os.makedirs(args['runargs_out_dir'])
    with open(os.path.join(args['runargs_out_dir'], '{:s}.json'.format(args['exp_name'])), 'wt') as fp:
        json.dump(args, fp, indent=2)


    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    r2r_test_dataset = RealMaskedFacesDataset_Classification(split_file='test.txt', transforms=val_transform)
    # pdb.set_trace()

    current_best_val_acc = 0
    model = ResNet18()

    weights = torch.load(PATH_TO_MODEL)['model_state_dict']

    model.load_state_dict(weights)

    # model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['moment'])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args['lr_step1'], gamma=args['lr_gamma'])


    for epoch in range(args['num_epoch']):
        val_loss, val_acc = test_epoch(model, epoch, criterion, r2r_test_dataset, args)

        scheduler.step()
        log.close()

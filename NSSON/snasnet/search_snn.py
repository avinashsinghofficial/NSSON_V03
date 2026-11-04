import os
import time
import torch
import utils
import config
import torchvision
import torch.nn as nn
import numpy as np
import random

from model_snn import SNASNet, find_best_neuroncell
from utils import data_transforms
from spikingjelly.clock_driven.functional import reset_net


def main():
    args = config.get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # define dataset
    train_transform, valid_transform = data_transforms(args)

    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=os.path.join(args.data_dir, 'cifar10'),
            train=True,
            download=True,
            transform=train_transform
        )
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=4
        )

        valset = torchvision.datasets.CIFAR10(
            root=os.path.join(args.data_dir, 'cifar10'),
            train=False,
            download=True,
            transform=valid_transform
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
            num_workers=4
        )

    elif args.dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=os.path.join(args.data_dir, 'cifar100'),
            train=True,
            download=True,
            transform=train_transform
        )
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=(device.type == "cuda"),
            num_workers=4
        )

        valset = torchvision.datasets.CIFAR100(
            root=os.path.join(args.data_dir, 'cifar100'),
            train=False,
            download=True,
            transform=valid_transform
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
            num_workers=4
        )

    elif args.dataset == 'tinyimagenet':
        trainset = torchvision.datasets.ImageFolder(
            os.path.join('/gpfs/loomis/project/panda/shared/tiny-imagenet-200/train'),
            train_transform
        )
        valset = torchvision.datasets.ImageFolder(
            os.path.join('/gpfs/loomis/project/panda/shared/tiny-imagenet-200/val'),
            valid_transform
        )
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
            sampler=None
        )
        val_loader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device.type == "cuda")
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # neuron cell
    if args.cnt_mat is None:
        best_neuroncell = find_best_neuroncell(args, trainset)
    else:
        int_list = []
        for line in args.cnt_mat:
            row_list = []
            for element in line:
                row_list.append(int(element))
            int_list.append(row_list)
        best_neuroncell = torch.tensor(int_list, dtype=torch.float32)

    print('-' * 7, "best_neuroncell", '-' * 7)
    print(best_neuroncell)
    print('-' * 30)

    # Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = SNASNet(args, best_neuroncell).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    # Optional pretrained checkpoint
    if args.savemodel_pth is not None:
        if os.path.isfile(args.savemodel_pth):
            print(f"[INFO] Loading pretrained model from {args.savemodel_pth}")
            checkpoint = torch.load(args.savemodel_pth, map_location=device)
            print(checkpoint.keys())
            model.load_state_dict(checkpoint['state_dict'])
            print('test only...')
            validate(args, 0, val_loader, model, criterion, device)
            return
        else:
            print(f"[WARN] Pretrained checkpoint not found at {args.savemodel_pth}. "
                  f"Continuing without pretrained weights.")

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    # Scheduler
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(args.epochs * 0.5), int(args.epochs * 0.75)],
            gamma=0.1
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(args.epochs),
            eta_min=args.learning_rate * 0.01
        )
    else:
        raise ValueError(f"Unsupported scheduler: {args.scheduler}")

    # Training loop
    start = time.time()
    for epoch in range(args.epochs):
        train(args, epoch, train_loader, model, criterion, optimizer, scheduler, device)
        scheduler.step()
        if (epoch + 1) % args.val_interval == 0:
            validate(args, epoch, val_loader, model, criterion, device)
            utils.save_checkpoint(
                {'state_dict': model.state_dict()},
                epoch + 1,
                tag=args.exp_name + '_super'
            )
    utils.time_record(start)


def train(args, epoch, train_data, model, criterion, optimizer, scheduler, device):
    model.train()
    train_loss = 0.0
    top1 = utils.AvgrageMeter()

    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', current_lr))

    for step, (inputs, targets) in enumerate(train_data):
        inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
        targets = targets.to(device, non_blocking=(device.type == "cuda"))

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
        n = inputs.size(0)
        top1.update(prec1.item(), n)
        train_loss += loss.item()

        # reset SNN states between batches
        reset_net(model)

    print('train_loss: %.6f' % (train_loss / len(train_data)),
          'train_acc: %.6f' % top1.avg)


def validate(args, epoch, val_data, model, criterion, device):
    model.eval()
    val_loss = 0.0
    val_top1 = utils.AvgrageMeter()

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(val_data):
            inputs = inputs.to(device, non_blocking=(device.type == "cuda"))
            targets = targets.to(device, non_blocking=(device.type == "cuda"))

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            prec1, prec5 = utils.accuracy(outputs, targets, topk=(1, 5))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)

            reset_net(model)

    print('[Val_Accuracy epoch:%d] val_acc:%f val_loss:%f'
          % (epoch + 1, val_top1.avg, val_loss / len(val_data)))
    return val_top1.avg


if __name__ == '__main__':
    main()

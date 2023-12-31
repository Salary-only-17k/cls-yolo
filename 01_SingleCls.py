import tqdm
import time
import copy
import os
import datetime as dt
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader as DataLoader

from nets.factory_nets import all_net
from utils.getdata import Dataset_1head
from utils.common import show_lg
from utils.parse_cfg import parse_opt

"""
python 01_SingleCls.py sh --data_path /home/cheng/Videos/update/dist  #!^
"""

def run(opts):
    show_lg("train cfg:", opts)
    mode_Lst = opts.mode_Lst[:2]
    show_lg(mode_Lst)
    dataDataset = {mode: Dataset_1head(mode, opts.data_path) for mode in mode_Lst}
   
    dataloader = {mode: DataLoader(dataDataset[mode],batch_size=opts.batch_size,  \
                                    num_workers=opts.workers,drop_last=True) \
                for mode in mode_Lst}
    show_lg(f'Dataset loaded! length of train set is ', len(dataloader[mode_Lst[0]]))
    data_size = {v: len(dataDataset[v]) for v in mode_Lst}   #!^
    show_lg('data_size',data_size)

    opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Net = all_net['yolov5_62cls_st'][0]
    model = Net(num_cls1=opts.n_cls) # .run()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=opts.n_cuda)
    model.to(device=opts.device)
    show_lg('model using device ', opts.device)
    show_lg('parallel mulit device ', opts.n_cuda)

    if opts.resume_weights:
        model.load_state_dict(torch.load(opts.resume_weights)['model'].state_dict())

    criterion = nn.CrossEntropyLoss()
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    if opts.adam:
        optimizer = optim.Adam(pg0, lr=opts.lr0, betas=(opts.momentum, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=opts.lr0, momentum=opts.momentum, nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': opts.weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # a               dd pg2 (biases)
    show_lg('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)),'')
    del pg0, pg1, pg2

    scaler = amp.GradScaler()

    # log_pth = check_save_pth(opts.log_name)
    os.makedirs(os.path.join("runs",opts.log_pth),exist_ok=True)
    with SummaryWriter(log_dir=os.path.join("runs",opts.log_pth)) as writer:
        train_loop(model, dataloader, criterion, optimizer, scaler, writer, data_size, opts)
        # test_loop(model,dataloader, data_size)
def train_loop(model, dataloader, criterion, optimizer, scaler, writer, data_size, opts):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for nepoch in range(opts.epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:  # val
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for data, label in tqdm.tqdm(dataloader[phase], desc=f'{nepoch + 1}/{opts.epochs}'):
                data = data.to(device=opts.n_cuda)  #!^ 
                label = label.to(device=opts.n_cuda) #!^
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # show_db("hello")
                    with amp.autocast():
                        out = model(data)
                        _, pre = torch.max(out, 1)
                        loss = criterion(out, label.squeeze())
                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                running_loss += loss.item() * data.size(0)
                running_corrects += torch.sum(pre.view(opts.batch_size, -1) == label.data)
            epoch_loss = running_loss / data_size[phase]
            # show_db(type(running_corrects))
            epoch_acc = running_corrects.double() / data_size[phase]
            writer.add_scalar(f"{phase}/Loss", epoch_loss, nepoch)
            writer.add_scalar(f"{phase}/Acc", epoch_acc, nepoch)
            show_lg(f'\n{phase} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}', '\n')
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        since = time.time()
        show_lg('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60), '')
    writer.add_graph(model, data)
    show_lg('Best val Acc: {:4f}'.format(best_acc), '')
    lastpth = f"{os.path.join('runs',opts.log_pth)}/{model.get_name()}_lastacc_{epoch_acc:.3f}_{dt.datetime.now().strftime('%Y_%m_%d-%H%M%S')}.pt"
    torch.save({"model": model, "acc": best_acc},lastpth)
    bestpth  = f"{os.path.join('runs',opts.log_pth)}/{model.get_name()}_bestacc_{best_acc:.3f}_{dt.datetime.now().strftime('%Y_%m_%d-%H%M%S')}.pt"
    model.load_state_dict(best_model_wts)
    torch.save({"model": model, "acc": best_acc},bestpth )
    print(f"best-model save to : {bestpth}")
    print(f"last-model save to : {lastpth}")



if __name__ == "__main__":
    opts = parse_opt()
    opts.adam = True
    run(opts)

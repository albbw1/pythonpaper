#coding=utf8
from __future__ import print_function, division
import utils
import os,time,datetime
import numpy as np
from math import ceil
import datetime
from models.LoadModel import MainModel

import torch
from torch import nn
from torch.autograd import Variable
#from torchvision.utils import make_grid, save_image


from utils import LossRecord, clip_gradient
from utils.focal_loss import FocalLoss
from utils.eval_model import eval_turn
from utils.Asoftmax_loss import AngleLoss

import pdb
from matplotlib import pyplot as plt
def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

loss1=[]
loss2=[]

def train(Config,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          exp_lr_scheduler,
          data_loader,
          save_dir,
          data_size=448,
          savepoint=500,
          checkpoint=1000
          ):
    # savepoint: save without evalution
    # checkpoint: save with evaluatio






    step = 0
    eval_train_flag = False
    rec_loss = []
    checkpoint_list = []

    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)

    if savepoint > train_epoch_step:
        savepoint = 1*train_epoch_step
        checkpoint = savepoint

    date_suffix = dt()
    log_file = open(os.path.join(Config.log_folder, 'formal_log_r50_dcl_%s_%s.log'%(str(data_size), date_suffix)), 'a')

    add_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()
    get_focal_loss = FocalLoss()
    get_angle_loss = AngleLoss()




    for epoch in range(start_epoch,epoch_num-1):
        loss3=[]
        exp_lr_scheduler.step(epoch)
        model.train(True)

        save_grad = []
        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            loss = 0
            model.train(True)
            if Config.use_backbone:
                inputs, labels, img_names = data
                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).cuda())

            if Config.use_dcl:
                inputs, labels, labels_swap, swap_law, img_names = data

                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).cuda())
                labels_swap = Variable(torch.from_numpy(np.array(labels_swap)).cuda())
                swap_law = Variable(torch.from_numpy(np.array(swap_law)).float().cuda())

            optimizer.zero_grad()

            if inputs.size(0) < 2*train_batch_size:
                outputs = model(inputs, inputs[0:-1:2])
            else:
                outputs = model(inputs, None)

            if Config.use_focal_loss:
                ce_loss = get_focal_loss(outputs[0], labels.long())
            else:
                ce_loss = get_ce_loss(outputs[0], labels.long())

            if Config.use_Asoftmax:
                fetch_batch = labels.size(0)
                if batch_cnt % (train_epoch_step // 5) == 0:
                    angle_loss = get_angle_loss(outputs[3], labels[0:fetch_batch:2], decay=0.9)
                else:
                    angle_loss = get_angle_loss(outputs[3], labels[0:fetch_batch:2])
                loss += angle_loss

            loss += ce_loss

            alpha_ = 1
            beta_ = 1
            gamma_ = 0.01 if Config.dataset == 'STCAR' or Config.dataset == 'AIR' else 1
            if Config.use_dcl:
                swap_loss = get_ce_loss(outputs[1], labels_swap.long()) * beta_
                loss += swap_loss
                law_loss = add_loss(outputs[2], swap_law.long()) * gamma_
                loss += law_loss

            loss.backward()
            torch.cuda.synchronize()

            optimizer.step()
            torch.cuda.synchronize()

            if Config.use_dcl:
                print('step: {:-8d} / {:d} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} + {:6.4f} + {:6.4f} '.format(step, train_epoch_step, loss.detach().item(), ce_loss.detach().item(), swap_loss.detach().item(), law_loss.detach().item()), flush=True)
            if Config.use_backbone:
                print('step: {:-8d} / {:d} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} '.format(step, train_epoch_step, loss.detach().item(), ce_loss.detach().item()), flush=True)
            rec_loss.append(loss.detach().item())

            train_loss_recorder.update(loss.detach().item())

            # evaluation & save
            if step % checkpoint == 0:
                rec_loss = []
                print(32*'-', flush=True)
                print('step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'.format(step, train_epoch_step, 1.0*step/train_epoch_step, epoch, train_loss_recorder.get_val()), flush=True)
                print('current lr:%s' % exp_lr_scheduler.get_lr(), flush=True)
                if eval_train_flag:
                    trainval_acc1, trainval_acc2, trainval_acc3 = eval_turn(Config, model, data_loader['trainval'], 'trainval', epoch, log_file)
                    if abs(trainval_acc1 - trainval_acc3) < 0.01:
                        eval_train_flag = False

                val_acc1, val_acc2, val_acc3 = eval_turn(Config, model, data_loader['val'], 'val', epoch, log_file)

                save_path = os.path.join(save_dir, 'weights_%d_%d_%.4f_%.4f.pth'%(epoch, batch_cnt, val_acc1, val_acc3))
                torch.cuda.synchronize()
                torch.save(model.state_dict(), save_path)
                print('saved model to %s' % (save_path), flush=True)
                torch.cuda.empty_cache()

            # save only
            elif step % savepoint == 0:
                train_loss_recorder.update(rec_loss)
                rec_loss = []
                save_path = os.path.join(save_dir, 'savepoint_weights-%d-%s.pth'%(step, dt()))

                checkpoint_list.append(save_path)
                if len(checkpoint_list) == 6:
                    os.remove(checkpoint_list[0])
                    del checkpoint_list[0]
                torch.save(model.state_dict(), save_path)
                torch.cuda.empty_cache()
            loss3.append(ce_loss)
        k=len(loss3)
        loss1.append(sum(loss3)/k)
    #plt.subplot(1, 2, 1)
    plt.plot(utils.eval_model.loss2, 'b', label='val_loss')
    # plt.ylabel('val_loss')
    # plt.xlabel('iter_num')
    #plt.subplot(1, 2, 2)
    plt.plot(loss1, 'r', label='train_loss')
    plt.ylabel('loss')
    plt.xlabel('epoch_num')
    plt.show()
    log_file.close()




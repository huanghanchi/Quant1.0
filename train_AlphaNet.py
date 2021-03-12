import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.autograd import Variable
import argparse
from dataset import *
from AlphaNet import *
from torch.utils.data import DataLoader
import numpy as np
from torch import nn

def main(opts):
    batch_size = opts.batch_size
    num_iters = opts.num_iters
    dataset = opts.dataset
    ckpt_load_path = opts.ckpt_load_path
    ckpt_save_path = opts.ckpt_save_path

    loss_log_path = opts.loss_log_path

    # load data
    dset = SEDataset(dataset, rolling=20, if_train=True)
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=0)
    iter_dloader = dloader

    # load net
    if ckpt_load_path is None:
        alphanet = AlphaNet1D()
    else:
        alphanet = torch.load(ckpt_load_path)
        print('load', ckpt_load_path)
    alphanet = alphanet.cuda()

    print("Total number of parameters is {}  ".format(sum(x.numel() for x in alphanet.parameters())))

    criterion = nn.CrossEntropyLoss(reduction='sum').cuda()
    optimizer = torch.optim.Adam(alphanet.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150, 200, 250, 300], gamma=0.1)

    for num_iter in range(num_iters):
        #try:
            alphanet.train()
            # scheduler.step()

            df, label, code = iter(iter_dloader).next()
            # print(label)
            # print(df)
            # print(label)
            # df = df.unsqueeze(1)
            label = label.squeeze()
            # print(df.shape)
            # continue
            df = Variable(df.cuda())

            label = Variable(label.cuda())
            code = Variable(code.cuda())

            optimizer.zero_grad()
            # df = torch.ones((64, 240)).cuda()
            # label = np.ones(64)
            # label = torch.LongTensor(label).cuda()
            predict_label = alphanet(df, code)
            print()
            print()
            print(predict_label[:3])
            # for i in range(opts.batch_size):
            #     print(i, df[i], code[i])
            # for i in range(opts.batch_size):
            #     print(i, float(label[i]), float(predict_label[i]))

            # print(predict_label.shape)
            # print(label.shape)
            # print(float(predict_label[0]), float(predict_label[1]))
            # exit()
            # print(label)
            loss = criterion(predict_label, label)
            if np.isnan(loss.item()):
                print('Loss value is NaN!')
                break
                # raise
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(alphanet.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            if loss_log_path is None:
                print('Iter:{}    Loss:{}'.format(num_iter, loss.item()))
            else:
                f = open(loss_log_path, 'a')
                print('Iter:{}    Loss:{}'.format(num_iter, loss.item()))
                print('Iter:{}    Loss:{}'.format(num_iter, loss.item()), file=f)

            if num_iter % 100 == 0 and num_iter > 0:
                torch.save(alphanet, ckpt_save_path + str(num_iter) + '.ckpt')
                print('save {}'.format(ckpt_save_path + str(num_iter) + '.ckpt'))

        #except Exception as e:
        #    continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--num_iters', type=int, default=1000000)

    parser.add_argument('--dataset', type=str,
                        default=r'C:\Users\SpiceeYJ\Desktop\量化投资\project_demo\data\stock_preprocessed\\')

    parser.add_argument('--ckpt_load_path', type=str,
                        default=r'C:\Users\SpiceeYJ\Desktop\量化投资\project_demo\weights\34100.ckpt')

    parser.add_argument('--ckpt_save_path', type=str,
                        default=r'C:\Users\SpiceeYJ\Desktop\量化投资\project_demo\weights\\')

    parser.add_argument('--loss_log_path', type=str,
                        default='log.txt')

    opts = parser.parse_args()
    main(opts)
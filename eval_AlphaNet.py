import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.autograd import Variable
from dataset import *
import argparse
from torch.utils.data import DataLoader
from torch import nn

def main(opts):
    batch_size = opts.batch_size
    dataset = opts.dataset
    ckpt_load_path = opts.ckpt_load_path

    # load data
    dset = SEDataset(dataset, rolling=20, if_train=False)

    dloader = DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=0)
    iter_dloader = dloader

    assert ckpt_load_path is not None
    alphanet = torch.load(ckpt_load_path)
    print('load', ckpt_load_path)
    alphanet = alphanet.cuda()
    alphanet.eval()

    correct_test = 0
    num_test_instances = 0
    count = 0

    while True:
        #try:
            with torch.no_grad():
                count += 1
                df, label, code = iter(iter_dloader).next()
                # df = df.unsqueeze(1)
                df = Variable(df.cuda())

                label = Variable(label.cuda())
                code = Variable(code.cuda())
                predict_label = alphanet(df, code).data.max(1)[1]
                # print(predict_label)
                correct_test += predict_label.eq(label.data.long()).sum()
                print('True:{}    Predict:{} '.format(float(label), float(predict_label)))
                if count % 1000 == 0:
                    break
    print(correct_test)

        #except Exception as e:
        #    print('error')
        #    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dataset', type=str,
                        default=r'C:\Users\SpiceeYJ\Desktop\量化投资\project_demo\data\stock_preprocessed\\')
    parser.add_argument('--ckpt_load_path', type=str, default=
                        r'C:\Users\SpiceeYJ\Desktop\量化投资\project_demo\weights\14100.ckpt')
    opts = parser.parse_args()
    main(opts)
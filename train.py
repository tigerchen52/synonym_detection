import torch
import torch.optim as optim
import sys
import os
import argparse
from torch.optim import lr_scheduler
from loss import registry as loss_f
from loader import registry as loader
from model import registry as Producer
from evaluate import overall


#hyper-parameters
parser = argparse.ArgumentParser(description='contrastive learning framework for word vector')
parser.add_argument('-dataset', help='the file of target vectors', type=str, default='data/wiki_100.vec')
parser.add_argument('-batch_size', help='the number of samples in one batch', type=int, default=32)
parser.add_argument('-epochs', help='the number of epochs to train the model', type=int, default=20)
parser.add_argument('-shuffle', help='whether shuffle the samples', type=bool, default=True)
parser.add_argument('-model_type', help='sum, rnn, cnn, attention, pam', type=str, default='pam')
parser.add_argument('-loader_type', help='simple, aug, hard', type=str, default='hard')
parser.add_argument('-loss_type', help='mse, ntx, align_uniform', type=str, default='ntx')


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

def main():

    data_loader = loader[args.loader_type](batch_size=args.batch_size, shuffle=args.shuffle)
    train_iterator = data_loader(data_path=args.dataset)

    model = Producer[args.model_type]()
    print(model)
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_num)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    criterion = loss_f[args.loss_type]()

    max_acc = 0
    for e in range(args.epochs):
        epoch_loss = 0
        batch_num = 0

        for words, oririn_repre, aug_repre_ids, mask in train_iterator:
            model.train()
            optimizer.zero_grad()
            batch_num += 1

            if batch_num % 1000 == 0:
                print('sample = {b}, loss = {a}'.format(a=epoch_loss/batch_num, b=batch_num*args.batch_size))

            # get produced vectors
            oririn_repre = oririn_repre.cuda()
            aug_repre_ids = aug_repre_ids.cuda()
            mask = mask.cuda()
            aug_embeddings = model(aug_repre_ids, mask)

            # calculate loss
            loss = criterion(oririn_repre, aug_embeddings)

            # backward
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        print('[ lr rate] = {a}'.format(a=optimizer.state_dict()['param_groups'][0]['lr']))

        print('----------------------')
        print('this is the {a} epoch, loss = {b}'.format(a=e + 1, b=epoch_loss / len(train_iterator)))

        if (e) % 1 == 0:
            model_path = './output/model_{a}.pt'.format(a=e+1)
            torch.save(model.state_dict(), model_path)
            overall(model_path=model_path, model_type=args.model_type)
    return max_acc


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
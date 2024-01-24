import argparse

from models import *
from train import Trainer
from utils import *
import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='amaphoto', help='dataset name')
parser.add_argument('--type', default='Type1', choices=['Type1', 'Type2'], help="The type of missing")
parser.add_argument('--rate', type=float, default=0.1, help='missing rate')
parser.add_argument('--nhid', type=int, default=64, help='the number of hidden units')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--wd', type=float, default=1e-2, help='weight decay')
#cora 5e-3 citeseer 8e-2 amaphoto and amacomp 1e-2
parser.add_argument('--niter', type=int, default=20, help='running times of each p_m')
parser.add_argument('--patience', type=int, default=100, help='Early stop condition')
parser.add_argument('--epoch', type=int, default=10000, help='the maximum number of iterations ')
parser.add_argument('--seed', default=42)


args = parser.parse_args()


if __name__ == '__main__':
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = load_data(args.dataset)
    mask = generate_mask(data.features, args.rate, args.type)

    data.features[mask] = 0.
    M = (~mask).float()
    Z = torch.spmm(data.adj_nonorm, M).pow(-1)
    Z = torch.where(torch.isinf(Z), torch.full_like(Z, 0.), Z)
    if args.dataset == "arxiv":
        model = PaGCN_ogb(data, M.to(device), Z.to(device), nhid=args.nhid, dropout=args.dropout)
    else:
        model = PaGCN(data, M.to(device), Z.to(device), nhid=args.nhid, dropout=args.dropout)

    #train
    trainer = Trainer(data, model, lr=args.lr, weight_decay=args.wd, niter=args.niter, patience=args.patience, epochs=args.epoch)
    result = trainer.run()
    print("The final result:",result)
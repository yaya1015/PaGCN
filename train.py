import torch
import torch.nn.functional as F

from copy import deepcopy
from numpy import mean, std
from torch.optim import Adam
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.state_dict = None

    def reset(self):
        self.counter = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
        self.state_dict = None

    def check(self, evals, model, epoch):
        val_loss, val_acc = evals[0], evals[1]
        if val_loss <= self.best_val_loss or val_acc >= self.best_val_acc:
            if val_loss <= self.best_val_loss and val_acc >= self.best_val_acc:
                self.state_dict = deepcopy(model.state_dict())
            self.best_val_loss = min(self.best_val_loss, val_loss)
            self.best_val_acc = max(self.best_val_acc, val_acc)
            self.counter = 0
        else:
            self.counter += 1
        stop = False
        if self.counter >= self.patience:
            stop = True
            model.load_state_dict(self.state_dict)
        return stop


class Trainer(object):
    def __init__(self, data, model, lr, weight_decay, epochs, niter, early_stopping=True, patience=100):
        self.data = data
        self.model = model
        self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.niter = niter
        self.early_stopping = early_stopping
        if early_stopping:
            self.stop_checker = EarlyStopping(patience)

        self.data.to(device)

    def train(self):
        data, model, optimizer = self.data, self.model, self.optimizer
        model.train()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output[data.train_mask], data.labels[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def evaluate(self, key):
        data, model = self.data, self.model
        model.eval()

        with torch.no_grad():
            output = model(data)

        if key == 'train':
            mask = data.train_mask
        elif key == 'val':
            mask = data.val_mask
        else:
            mask = data.test_mask
            
        loss = F.nll_loss(output[mask], data.labels[mask]).item()
        pred = output[mask].max(dim=1)[1]
        acc = pred.eq(data.labels[mask]).sum().item() / mask.sum().item()

        return loss, acc

    def reset(self):
        self.model.to(device).reset_parameters()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.early_stopping:
            self.stop_checker.reset()

    def run(self):
        test_acc_list = []

        for _ in tqdm(range(self.niter)):
            self.reset()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            for epoch in range(1, self.epochs + 1):
                loss_train = self.train()
                evals = self.evaluate("val")
                # print("Loss_train:", loss_train, "evals_acc:", evals[1])

                if self.early_stopping:
                    if self.stop_checker.check(evals, self.model, epoch):
                        break

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            _, test_acc = self.evaluate("test")
            test_acc_list.append(test_acc)

        return mean(test_acc_list)


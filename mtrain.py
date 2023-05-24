import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from dataset import ECGDataset
from hrvresnet import resnet34
from utils import cal_f1s, cal_aucs, split_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()


def train(model, dataloader, args, criterion, epoch, scheduler, optimizer, device):
    print('Training epoch %d:' % epoch)
    model.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, hrv, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(data, hrv)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    # scheduler.step()
    print('Loss: %.4f' % running_loss)
    

def evaluate(model, dataloader, args, criterion, device):
    print('Validating...')
    model.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, hrv, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = model(data, hrv)
        loss = criterion(output, labels)
        running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print('Loss: %.4f' % running_loss)
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = cal_f1s(y_trues, y_scores)
    avg_f1 = np.mean(f1s)
    print('F1s:', f1s)
    print('Avg F1: %.4f' % avg_f1)
    if args.phase == 'train' and avg_f1 > args.best_metric:
        args.best_metric = avg_f1
        torch.save(model.state_dict(), args.model_path)
    else:
        aucs = cal_aucs(y_trues, y_scores)
        avg_auc = np.mean(aucs)
        print('AUCs:', aucs)
        print('Avg AUC: %.4f' % avg_auc)


if __name__ == "__main__":
    args = parse_args()
    args.best_metric = 0
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)
#     data_dir = 'ecg-diagnosis/data/CPSC'
    if not args.model_path:
        args.model_path = f'models/ecghrv200N100E_{database}_{args.leads}_{args.seed}.pth'

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    
    if args.leads == 'all':
        leads = 'all'
        nleads = 12
    else:
        leads = args.leads.split(',')
        nleads = len(leads)
    
    label_csv = os.path.join(data_dir, 'labelx.csv')
    hrv_features = 'mean_hrv_features.csv'
    
    train_folds, val_folds, test_folds = split_data(seed=args.seed)
    train_dataset = ECGDataset('train', data_dir, hrv_features, label_csv, train_folds, leads)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = ECGDataset('val', data_dir, hrv_features, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_dataset = ECGDataset('test', data_dir, hrv_features, label_csv, test_folds, leads)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    model = resnet34(input_channels=nleads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
    criterion = nn.BCEWithLogitsLoss()

    if args.phase == 'train':
        if args.resume:
            model.load_state_dict(torch.load(args.model_path, map_location=device))
        for epoch in range(1, args.epochs+1):
            train(model, train_loader, args, criterion, epoch, scheduler, optimizer, device)
            evaluate(model, val_loader, args, criterion, device)
    else:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluate(model, test_loader, args, criterion, device)
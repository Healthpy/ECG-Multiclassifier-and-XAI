import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from hrvresnet import resnet34
from dataset import ECGDataset
from utils import cal_scores, find_optimal_threshold, split_data
from sklearn.metrics import roc_curve, auc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory to data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers to load data')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use gpu')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()


def get_thresholds(val_loader, model, hrv, device, threshold_path):
    print('Finding optimal thresholds...')
    if os.path.exists(threshold_path):
        return pickle.load(open(threshold_path, 'rb'))
    output_list, label_list = [], []
    for _, (data, hrv, label) in enumerate(tqdm(val_loader)):
        data, labels = data.to(device), label.to(device)
        output = model(data, hrv)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    thresholds = []
    for i in range(y_trues.shape[1]):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        threshold = find_optimal_threshold(y_true, y_score)
        thresholds.append(threshold)
    return thresholds


def apply_thresholds(test_loader, model, hrv, device, thresholds):
    output_list, label_list = [], []
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    for _, (data, hrv, label) in enumerate(tqdm(test_loader)):
        data, labels = data.to(device), label.to(device)
        output = model(data, hrv)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    y_preds = []
    scores = []
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(thresholds)):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        y_pred = (y_score >= thresholds[i]).astype(int)
        y_preds.append(y_pred)
        scores.append(cal_scores(y_true, y_pred, y_score))
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
        roc_auc[classes[i]] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'{classes[i]} (AUC = {roc_auc[classes[i]]:0.2})')   
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ECG-HRV ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('results/ROC_ECG-HRV.png')
    plt.show()
    plt.close()
        
    y_preds = np.array(y_preds).transpose()
    scores = np.array(scores)
    print('Precisions:', scores[:, 0])
    print('Recalls:', scores[:, 1])
    print('F1s:', scores[:, 2])
    print('AUCs:', scores[:, 3])
    print('Accs:', scores[:, 4])
    print(np.mean(scores, axis=0))
    plot_cm(y_trues, y_preds)
    comp_confmat(y_trues, y_preds)


def plot_cm(y_trues, y_preds, normalize=True, cmap=plt.cm.Blues):
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    for i, label in enumerate(classes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=[0, 1], yticklabels=[0, 1],
           title=label,
           ylabel='True label',
           xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), ha="center")

        fmt = '.3f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        np.set_printoptions(precision=3)
        fig.tight_layout()
        plt.savefig(f'results/{label}.png')
        plt.close(fig)

def comp_confmat(y_true, y_test_scores, cmap=plt.cm.Blues):

    # extract the different classes
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']

    confusion_matrix = [[0 for _ in range(9)] for _ in range(9)]
    for true_label, predicted_label in zip(y_true, y_test_scores):
        true_index = true_label.argmax()
        predicted_index = predicted_label.argmax()
        if true_index == predicted_index:
            confusion_matrix[true_index][true_index] += 1
        else:
            confusion_matrix[true_index][predicted_index] += 1

    ys = np.array(confusion_matrix)
    fig, axs = plt.subplots()
    im = axs.imshow(ys, cmap=cmap)
    axs.figure.colorbar(im, ax=axs)
    fmt = '.2f'
    xlabels = classes
    ylabels = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    axs.set_xticks(np.arange(len(xlabels)))
    axs.set_yticks(np.arange(len(ylabels)))
    axs.set_xticklabels(xlabels)
    axs.set_yticklabels(ylabels)
    thresh = ys.max() / 2
    for i in range(ys.shape[0]):
        for j in range(ys.shape[1]):
            axs.text(j, i, format(ys[i, j], fmt),
                    ha='center', va='center',
                    color='white' if ys[i, j] > thresh else 'black')
    #np.set_printoptions(precision=2)
    fig.tight_layout()
    plt.show()
    plt.savefig(f'./cmatrix/complete_cm.png')
    plt.clf()
    

if __name__ == "__main__": 
    args = parse_args()
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)
    if not args.model_path:
        args.model_path = f'models/ecghrv_{database}_{args.leads}_{args.seed}.pth'
    args.threshold_path = f'models/{database}-threshold.pkl'
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
    data_dir = args.data_dir
    label_csv = os.path.join(data_dir, 'labelx.csv')
    hrv_features = 'mean_hrv_features.csv'

    model = resnet34(input_channels=nleads).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    train_folds, val_folds, test_folds = split_data(seed=42)
    train_dataset = ECGDataset('train', data_dir, hrv_features, label_csv, train_folds, leads)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = ECGDataset('val', data_dir, hrv_features, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataset = ECGDataset('test', data_dir, hrv_features, label_csv, test_folds, leads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    thresholds = get_thresholds(val_loader, model, hrv_features, device, args.threshold_path)
    print('Thresholds:', thresholds)

    print('Results on validation data:')
    apply_thresholds(val_loader, model, hrv_features, device, thresholds)

    print('Results on test data:')
    apply_thresholds(test_loader, model, hrv_features, device, thresholds)
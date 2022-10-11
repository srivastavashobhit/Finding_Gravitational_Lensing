from cmath import isnan
import math
from xml.etree.ElementTree import C14NWriterTarget
import cv2
import torch
import torch.nn as nn
import unet
import attunet
# import utransformer
import dataset
import numpy as np

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.ops import sigmoid_focal_loss

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
epochs = 300
batch_size = 1
lr = 0.0003
weight_decay = 1e-5
# momentum = 0.9
threshold = 0.95
checkpoints = [80, 100, 150, 200, 300]
fl_alpha = 0.25 # default: 0.25 (range: [0, 1])
fl_gamma = 2 # default: 2
fl_reduction = 'mean' # sum / mean / None

def train():
    # model
    # model = unet.UNet(n_channels = 1, n_classes = 1, use_attention = True)
    model = attunet.AttU_Net(n_channels = 1, n_classes = 2)
    model.to(device = device)

    best_model = None

    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    # dataset
    train_dataset = dataset.FeatureDataset(is_train = True)
    train_indices, valid_indices = train_dataset.randomSplit(0.2) # split dataset for training and validate
    valid_sampler = SubsetRandomSampler(valid_indices)

    
    valid_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, sampler = valid_sampler)

    
    total_valid_batchs = math.ceil(len(valid_indices) / batch_size)
    
    best_model = 'best_model.pt'
    # model = attunet.AttU_Net(n_channels = 1, n_classes = 2)
    model = model = unet.UNet(n_channels = 1, n_classes = 2)
    model.to(device = device)
    model.load_state_dict(torch.load(best_model, map_location = device))
    model.eval()

    n = len(valid_loader)
    liver_count = 0
    kidney_count = 0
    both_count = 0
    print(n)
    tot_iou = 0
    tot_liver = 0
    tot_kidney = 0
    batch = 0
    zeros = 0
    for data, labels in valid_loader:
        batch += 1
        data2 = data[0][0].detach().numpy().astype(np.uint8)
        cv2.imwrite('./orig/' + str(batch) + '.jpg', data2)
        if 0 == batch % 100:
            print(f"[ Valid Batch | {batch:04d}/{total_valid_batchs:04d} ]")

        data = data.to(device = device, dtype = torch.float32)
        labels = labels.to(device = device, dtype = torch.float32)

        pred = model(data)

        # set mask
        pred[threshold <= pred] = 1
        pred[threshold > pred] = 0

        # added myself
        pred_path = "./pred/"
        gt_path = "./gt/"
        pred = pred[0]
        labels = labels[0]
        pixel_thresh = 256 * 256 * 0.000
        iou = float(torch.sum(torch.logical_and(pred, labels)) / torch.sum(torch.logical_or(pred, labels)))
        iou_liver = float(torch.sum(torch.logical_and(pred[0], labels[0])) / torch.sum(torch.logical_or(pred[0], labels[0])))
        iou_kidney = float(torch.sum(torch.logical_and(pred[1], labels[1])) / torch.sum(torch.logical_or(pred[1], labels[1])))
        
        if not math.isnan(iou_kidney) and torch.sum(labels[1]) > pixel_thresh and not math.isnan(iou_liver) and torch.sum(labels[0]) > pixel_thresh:
            both_count += 1
        if (not math.isnan(iou_kidney) and torch.sum(labels[1]) > pixel_thresh) or (not math.isnan(iou_liver) and torch.sum(labels[0]) > pixel_thresh):
            tot_iou += iou
            liv = "".ljust(19, ' ')
            kid = "".ljust(19, ' ')
            if not math.isnan(iou_liver) and torch.sum(labels[0]) > pixel_thresh:
                liver_count += 1
                tot_liver += iou_liver
                liv = str(iou_liver).ljust(19, ' ')
            if not math.isnan(iou_kidney) and torch.sum(labels[1]) > pixel_thresh:
                kidney_count += 1
                tot_kidney += iou_kidney
                kid = str(iou_kidney).ljust(19, ' ')
            print((str(batch) + ":").ljust(4, ' '), liv, kid)
            if iou == 0.0:
                zeros += 1
                print('areas:', torch.sum(pred), torch.sum(labels))
            pred = torch.logical_or(pred[0], pred[1]).numpy().astype(np.uint8)
            labels = torch.logical_or(labels[0], labels[1]).numpy().astype(np.uint8)
            # pred = pred.detach().numpy().astype(np.uint8)
            # labels = labels.detach().numpy().astype(np.uint8)
            cv2.imwrite(pred_path + str(batch) + '_pred' + '.jpg', pred * 255)
            cv2.imwrite(gt_path + str(batch) + '_gt' + '.jpg', labels * 255)
            # cv2.imwrite("./best/residual/" + str(batch) + "_residual.jpg", np.logical_xor(pred, labels) * 255)
                
        else:
            n -= 1
    print("zeros:", zeros)
    print("n:", n)
    print("mean_iou:", tot_iou / n)
    print("liver:", liver_count)
    print("liver_iou:", tot_liver / liver_count)
    print("kidney:", kidney_count)
    print("kidney_iou:", tot_kidney / kidney_count)
    print("both:", both_count)


        

if '__main__' == __name__:
    train()

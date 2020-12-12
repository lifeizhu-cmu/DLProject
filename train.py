import numpy as np
import torch
import torch.nn.functional as F
import cv2
import imageio
import time
import matplotlib.pyplot as plt

from loss import IOULoss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, train_loader, criterion, optimizer, epoch):
    start = time.time()

    avg_loss = 0
    for batch_num, (feats, labels) in enumerate(train_loader):
        feats = feats.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(feats)
        
        optimizer.zero_grad()
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()

        if batch_num % 20 == 19:
            print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/20))
            avg_loss = 0
        
        torch.cuda.empty_cache()
        del feats, labels, loss

    end = time.time()
    print('Train: {:.1f}s'.format(end-start))
    
    return model


def validate(model, val_loader, criterion, class_index):
    model.eval()
    val_loss = []
    accuracy = 0
    total = 0

    IOU = [[] for _ in class_index]
    IOU_loss = IOULoss()

    for feats, labels in val_loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        outputs = model(feats)

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)

        loss = criterion(outputs, labels.long())

        accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels.view(-1))
        val_loss.extend([loss.item()]*feats.size()[0])

        for i, index in enumerate(class_index):
            true = (labels == index).type(torch.uint8)
            pred = (pred_labels == index).type(torch.uint8)
            IOU[i].extend([-IOU_loss(true, pred).item()]*feats.size()[0])

        torch.cuda.empty_cache()
        del feats, labels, loss

    mean_IOU = [np.mean(IOU[i]) for i in range(len(class_index))]

    model.train()
    return np.mean(val_loss), accuracy/total, mean_IOU


def form_colormap(prediction, mapping):
    h,w = prediction.shape
    color_label = np.zeros((h,w,3), dtype=np.uint8)    
    color_label = mapping[prediction]
    color_label = color_label.astype(np.uint8)
    return color_label


def test(model, test_loader, class_map):
    images_true_pred = []
    images_original_pred = []

    model.eval()
    for feats, labels in test_loader:

        feats = feats.to(DEVICE)
        outputs = model(feats)

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)

        for i in range(feats.size()[0]):
            img = np.array(255*feats[i].cpu(), dtype=np.uint8).transpose(1,2,0)

            pred = np.array(pred_labels[i].cpu(), dtype=np.uint8)
            img_pred = form_colormap(pred, np.array(class_map))

            label = np.array(labels[i], dtype=np.uint8)
            img_true = form_colormap(label, np.array(class_map))

            img_true_pred = cv2.vconcat([img_pred, img_true])
            images_true_pred.append(img_true_pred)

            img_original_pred = cv2.vconcat([img_pred, img])
            images_original_pred.append(img_original_pred)

            plt.figure(figsize=(8,3))
            plt.subplot(1,3,1)
            plt.imshow(img)
            plt.title('Original image')
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.imshow(img_pred)
            plt.title('Predicted masks')
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.imshow(img_true)
            plt.title('True masks')
            plt.axis('off')
            plt.show()

    imageio.mimsave('true_pred.gif', images_true_pred)
    imageio.mimsave('original_pred.gif', images_original_pred)

    model.train()

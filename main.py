import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

from model_builder import init_weights, build_model
from loss import IOULoss
from train import train, test, validate
from dataloader import load_data, form_class_map, ImageDataset
#from dataloader_DAG import load_data, form_class_map, ImageDataset
from util import visualize_results, test_inf_speed, calc_n_params

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():

    batch_size = 5
    num_epochs = 100

    df_train, df_val, df_test = load_data()
    class_map, class_index, class_name = form_class_map()

    train_set = ImageDataset(df_train)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_set = ImageDataset(df_val)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)
    test_set = ImageDataset(df_test)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    model = build_model('UNet_ConvTranspose2d')
    model.apply(init_weights)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    n_params = calc_n_params(model)
    print("No. of parameters:", n_params)
    frame_rate = test_inf_speed(model, test_set, test_dataloader)
    print("Frame rate: {:.2f} FPS (frame per second)".format(frame_rate))

    df = pd.DataFrame(columns = class_name+["Train","Val"])

    model.train()
    for epoch in range(num_epochs):
        model = train(model, train_dataloader, criterion, optimizer, epoch)

        start = time.time()
        val_loss, val_acc, val_iou = validate(model, val_dataloader, criterion, class_index)
        train_loss, train_acc, _ = validate(model, train_dataloader, criterion, class_index)
        end = time.time()
        print('Validate: {:.1f}s'.format(end-start))

        print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.
            format(train_loss, train_acc, val_loss, val_acc))

        df.loc[len(df.index)] = val_iou+ [train_acc, val_acc]

        print('Val IOU:')
        for i in range(len(class_index)):
            print('{}: {:.4f}\t'.format(class_name[i], val_iou[i]), end='')
            if i % 6 == 5 or i == len(class_index)-1:
                print()
        print('mIOU: {:.4f}'.format(np.mean(val_iou)))

    df.to_csv("iou.csv", header=True, index=False, columns=class_name)
    df.to_csv("acc.csv", header=True, index=False, columns=["Train","Val"])

    visualize_results(df, class_name)
    test(model, test_dataloader, class_map)

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, "Model_"+str(num_epochs))


if __name__ == '__main__':
    main()

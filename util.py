import numpy as np
import torch
import torch.nn.functional as F
import traceback
import matplotlib.pyplot as plt
import time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def visualize_results(val_accuracies, train_accuracies, IOU_all, class_name):
    '''
    Saving and showing graph
    '''
    try:
        plt.plot([i+1 for i in range(len(train_accuracies))], train_accuracies, label="train")
        plt.plot([i+1 for i in range(len(val_accuracies))], val_accuracies, label="validation")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc="upper left")
        plt.savefig('validation_accuracy.png')
        plt.show()

        fig, axs = plt.subplots(nrows=1, ncols=1)
        for i in range(len(class_name)):
            axs.plot([i+1 for i in range(len(IOU_all[0]))], IOU_all[i])
        axs.set_xlabel('Epochs')
        axs.set_ylabel('IOU')
        lgd = axs.legend(class_name, loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        fig.savefig('IOU.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
        fig.show()

    except Exception as e:
        traceback.print_exc()
        print("Error: Problems generating plot. See if a .png was generated. "
              "If not, check the writeup and Piazza hw1p1 thread.")


def test_inf_speed(model, test_dataset, test_loader):
    model.eval()

    start = time.time()
    for feats, labels in test_loader:

        feats = feats.to(DEVICE)
        outputs = model(feats)

        _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)

        for i in range(feats.size()[0]):
            pred = np.array(pred_labels[i].cpu(), dtype=np.uint8)
    
    end = time.time()
    model.train()

    frame_rate = len(test_dataset)/(end-start)
    #print("Frame rate: {:.2f} FPS (frame per second)".format(frame_rate))
    return frame_rate


def calc_n_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in model_parameters])
    return n_params
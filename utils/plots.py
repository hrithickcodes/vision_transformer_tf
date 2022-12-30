import os
import numpy as np
import matplotlib.pyplot as plt

def plot_accuracy(history, savedir, show = False):
    training_accs, test_accs = history.history["acc"], history.history["val_acc"]
    plt.grid(True)
    plt.plot(np.arange(len(training_accs)),training_accs, color = "blue")
    plt.plot(np.arange(len(test_accs)), test_accs, color = "orange")
    plt.legend(['Training Accuracy', 'Test Accuracy'])
    plt.title('Accuracy vs Epoch')
    if savedir:
        savepath = os.path.join(savedir, "train-test-accuracy.png")
        plt.savefig(savepath)
    
    if show:    plt.show()
    plt.clf()
    print(f"Accuracy results saved at {savepath}")
    
    
    
def plot_loss(history, savedir, show = False):
    training_losses, test_losses = history.history["loss"], history.history["val_loss"]
    
    plt.grid(True)
    plt.plot(np.arange(len(training_losses)),training_losses, color = "blue")
    plt.plot(np.arange(len(test_losses)), test_losses, color = "orange")
    plt.legend(['Training Loss', 'Test Loss'])
    plt.title('Loss vs Epoch')
    if savedir:
        savepath = os.path.join(savedir, "train-test-loss.png")
        plt.savefig(savepath)
    
    if show:    plt.show()
    plt.clf()
    print(f"Loss results saved at {savepath}")
    

    
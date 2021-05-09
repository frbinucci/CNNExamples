# This is a sample Python script.

# Press Maiusc+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy
import numpy as np
import torch.utils.data
import torchvision

from dataset import MonkeyDataset
from solver import Solver
import matplotlib.pyplot as plt

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    solver = Solver(epoch=20)
    training_accuracy_array,test_accuracy_array,training_loss_array,test_loss_array = solver.fit(early_stopping=True)
    numpy.save('./training_accuracy',training_accuracy_array)
    numpy.save('./test_accuracy',test_accuracy_array)
    numpy.save('./training_loss',training_loss_array)
    numpy.save('./test_loss',test_loss_array)

    training = numpy.load('./training_accuracy.npy')
    test = numpy.load('./test_accuracy.npy')

    epoch_axis = np.arange(start=0,stop=20,step=1)

    plt.plot(epoch_axis,training,label="Training")
    plt.plot(epoch_axis,test,label="Test")
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

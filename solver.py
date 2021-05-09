import torch.utils.data
from pyexpat import model

import torchvision
from torch import optim
from torch import nn
from torch.autograd import Variable
from torchvision.transforms import ToTensor, transforms

from dataset import MonkeyDataset
from net import ClassificationNetwork
import os

class Solver():
    def __init__(self,**kwargs):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5), (0.5))])

        self.training_set = MonkeyDataset(True,data_root='./dataset')
        self.test_set = MonkeyDataset(False,data_root='./dataset')

        self.training_loader = torch.utils.data.DataLoader(self.training_set,
                                                           batch_size=4,
                                                           shuffle=True,
                                                           num_workers=2,
                                                           pin_memory=True)

        self.test_loader = torch.utils.data.DataLoader(self.test_set,
                                                       batch_size=4,
                                                       shuffle=True,
                                                       num_workers=2,
                                                       pin_memory=True)


        self.model_output_dir = kwargs.get('model_output_dir','./model')
        if not os.path.exists(self.model_output_dir):
            os.mkdir(self.model_output_dir)

        self.device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')

        print(self.device)
        self.model = ClassificationNetwork().to(self.device)
        self.epoch_number = kwargs.get('epoch',20)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()


    def fit(self,**kwargs):
        early_stopping = kwargs.get('early_stopping',False)
        best_test_loss = float('inf')
        train_acc_array = list()
        test_acc_array = list()
        test_loss_array = list()
        train_loss_array = list()
        for epoch_index in range(self.epoch_number):
            running_loss=0
            self.model.train()
            for data in self.training_loader:
                input,labels = data

                labels = torch.LongTensor(labels)

                input = input.to(self.device)
                labels = labels.to(self.device)

                #Zero gradient
                self.optimizer.zero_grad()
                #Computing outputs
                output = self.model(input)
                #Computing loss
                loss = self.criterion(output,labels)

                #Backpropagation
                loss.backward()
                #Step forward
                self.optimizer.step()

                #Computing mean loss
                running_loss+=loss.item()

            training_loss = self.propagate_test(self.training_loader)
            test_loss = self.propagate_test(self.test_loader)

            print("Epoch n. " + str(epoch_index) + " average loss in training set: " + str(training_loss))
            print("Epoch n. "+str(epoch_index)+" average loss in test set: "+str(test_loss))

            training_accuracy = self.evaluate(self.training_loader)
            test_accuracy = self.evaluate(self.test_loader)

            train_acc_array.append(training_accuracy)
            test_acc_array.append(test_accuracy)
            train_loss_array.append(training_loss)
            test_loss_array.append(test_loss)

            if test_loss<=best_test_loss:
                best_test_loss = test_loss
            else:
                if early_stopping==True:
                    self.save_model()
                    return train_acc_array,test_acc_array,train_loss_array,test_loss_array

        self.save_model()
        return train_acc_array, test_acc_array, train_loss_array, test_loss_array

    def propagate_test(self,model):
        running_loss = 0
        self.model.eval()
        with torch.no_grad():
            for item in model:
                input,labels = item

                labels = labels.to(self.device)
                input = input.to(self.device)
                output = self.model(input)

                loss = self.criterion(output, labels)
                running_loss+=loss.item()
            running_loss=running_loss/len(model)
        return running_loss

    def save_model(self):
        torch.save(self.model.state_dict(),self.model_output_dir)

    #Method used in order to implement early stopping
    def evaluate(self,loader):
        num_correct = 0
        num_total = 0
        self.model.eval()
        with torch.no_grad():
            for item in loader:
                input,labels = item

                labels = labels.to(self.device)
                input = input.to(self.device)
                output = self.model(input)
                _,preds = torch.max(output.detach(),1)
                num_correct+=(preds==labels).sum().item()
                num_total+=labels.size(0)

        return num_correct/num_total

















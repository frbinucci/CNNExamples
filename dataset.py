import os

from PIL import Image
from torch.utils.data import dataset
from torchvision import transforms

#General workflow of Pytorch foresees the creation of:
#1)A dataset class, used in order to load data
#2)A solver class, used in order to train the net
#3)A network class, used in order to define the network architecture
#Here we are defining a dataset class, used in order to load the dataset. The structure of this class depends on how data
#are structured.
class MonkeyDataset(dataset.Dataset):

    #Constructor
    def __init__(self,train,**kwargs):
        self.size = kwargs.get('size',128)
        self.str2label={}
        self.label2str={}
        self.data_root = kwargs.get('data_root','./dataset')
        self.data=list()

        with open(os.path.join(self.data_root,'monkey_labels.txt')) as f:
            lines = f.readlines()
            linecount=0
            for line in lines:
                if linecount!=0:
                    current_line = line.split(",")
                    self.label2str[int(current_line[0])] = current_line[1]
                    self.str2label[current_line[1]] = current_line[0]
                linecount+=1

        if train:
            directory = os.path.join(self.data_root,'training')
        else:
            directory = os.path.join(self.data_root,'validation')

        for current_directory in os.listdir(directory):
            for current_photo in os.listdir(os.path.join(directory,current_directory)):
                 self.data.append([str(os.path.join(directory,current_directory,current_photo)),
                                    str(current_directory)[1]])

        self.transform=transforms.Compose(
            [transforms.ToTensor()]
        )

    #This method must be overrided. It returns information about features and labels.
    def __getitem__(self, item):
        path,label = self.data[item]
        image = Image.open(path)
        image = image.resize((self.size, self.size), Image.BICUBIC)

        return self.transform(image),int(label)

    #This method must be override. It returns information about length of data vector
    def __len__(self):
        return len(self.data)

"""
This file is for GA + BDNN implementation.
"""
# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from preprocessor import Preprocessor
from postprocessor import Postprocessor
from feature_selector import Selector
initial_path='comp1111 data/data.csv'
target_path= 'comp1111 data/preprocessing.csv'
train_data_path='comp1111 data/train_data.csv'
test_data_path='comp1111 data/test_data.csv'
ga_path= 'comp1111 data/ga_data.csv'

Preprocessor(initial_path,target_path)
Selector(target_path,ga_path) # apply GA for feature selection
res=[] # result of the accuracy of each iteration
"""
Step 1: Define hyper parameters and the number of iterations
"""

# Hyper Parameters
for i in range (200):  # 200 iterations in total
    input_size = len(pd.read_csv(ga_path).columns)-1
    hidden_size = 10
    num_classes = 1
    num_epochs = 1000
    batch_size = 14
    learning_rate = 0.01
    # load all data
    data = pd.read_csv(ga_path)


    # randomly split data into training set (80%) and testing set (20%)
    msk = np.random.rand(len(data)) < 0.8
    train_data = data[msk]
    test_data = data[~msk]
    train_data=Postprocessor().process(train_data,train_data_path) # to get 4 classes and their sub-classes
    # test_data=Postprocessor().process(test_data,test_data_path)
    """
    Step 2: Define a neural network
    """

    # Neural Network
    class Net(nn.Module):
        def __init__(self,input_size, hidden_size, num_classes):
            super().__init__()
            self.W1 = nn.Linear(input_size, hidden_size)
            self.W2 = nn.Linear(hidden_size, num_classes)
            self.bias=0

        def forward(self, x):
            h = F.relu(self.W1(x))
            return self.W2(h)

        def reverseforward(self, x):
            h = F.relu(torch.matmul(x, self.W2.weight)+self.W1.bias)
            return torch.matmul(h, self.W1.weight)+self.bias



    net = Net(input_size, hidden_size, num_classes)

    # Loss and Optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    criterion=F.mse_loss
    # get X and Y from training data and convert them to tensor
    X=torch.Tensor((train_data.iloc[:,:-1]).values).float()
    Y=torch.Tensor(pd.DataFrame(list(train_data.iloc[:,-1])).values).float()
    # train the model by batch
    for epoch in range(num_epochs):
        total_loss = 0

        optimizer.zero_grad()  # zero the gradient buffer
        if epoch <= 300 or epoch > 600: # first 300 and last 600 epochs run forward
            outputs = net(X)
            lf = criterion(outputs, Y)
            lf.backward()
            optimizer.step()
            total_loss += lf

        elif epoch > 300 and epoch <= 600: # between 300 and 600 epochs run backward
            outputs = net.reverseforward(Y)
            lb = criterion(outputs, X)
            lb.backward()
            optimizer.step()
            total_loss += lb


        if (epoch % 100 == 0):
            print('Epoch [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs,
                     total_loss))
    """
    Step 3: Test the neural network

    Pass testing data to the built neural network and get its performance
    """
    # get testing data
    test_input = test_data.iloc[:, :input_size]
    test_target = test_data.iloc[:, input_size]

    inputs = torch.Tensor(test_input.values).float()
    targets = torch.Tensor(test_target.values - 1).long()

    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    total = predicted.size(0)
    correct = predicted.data.numpy() == targets.data.numpy()
    res.append(100*sum(correct)/total) # add each iteration in the result list
print(np.max(res))          # pick the best performance result
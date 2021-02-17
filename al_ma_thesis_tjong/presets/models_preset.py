import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Store models used
'''


# custom made CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # (n_feature_input, n_feature_output, kernel_size, stride)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)  # encourage dropout for next MLP/fc
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)  # softmax and log afterwards, since softmax is exp(),
        # log becomes like the inverse of exp() and the probability value becomes more apparent
        return output


# custom made CNN model
class Net_complex(nn.Module):
    def __init__(self):
        super(Net_complex, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # (n_feature_input, n_feature_output, kernel_size, stride)
        self.conv2 = nn.Conv2d(32, 128, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.25)
        self.dropout4 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216*2, 512*2)
        self.fc2 = nn.Linear(512*2, 512)
        self.fc3 = nn.Linear(512, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)  # encourage dropout for next MLP/fc
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.dropout4(x)
        x = self.fc4(x)

        output = F.log_softmax(x, dim=1)  # softmax and log afterwards, since softmax is exp(),
        # log becomes like the inverse of exp() and the probability value becomes more apparent
        return output


# First half of Autoencoder, to reduce data on MNIST
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder_in = nn.Linear(784, 128)
        self.encoder_hidden = nn.Linear(128, 128)

    def forward(self, input):
        data = self.encoder_in(input)
        data = F.relu(data)
        data = self.encoder_hidden(data)
        code = torch.relu(data)

        return code


# Custom made autoencoder, defined using nn.Linear
class Autoencoder(nn.Module):


    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder_in = nn.Linear(784, 128)
        self.encoder_hidden = nn.Linear(128, 128)
        self.decoder_hidden = nn.Linear(128, 128)
        self.decoder_out = nn.Linear(128, 784)

    def forward(self, input):
        data = self.encoder_in(input)
        data = F.relu(data)
        data = self.encoder_hidden(data)
        code = torch.relu(data)
        data = self.decoder_hidden(code)
        data = F.relu(data)
        data = self.decoder_out(data)
        output = F.relu(data)

        return output



def restore_dropout(fcn_model, orig_prob_list):
    """fcn_model_0 = fcn_model.to('cuda:0')
    del fcn_model"""
    for each_module in fcn_model.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.eval()
            each_module.p = orig_prob_list[0]
            orig_prob_list.pop(0)

    """fcn_model_0 = fcn_model_0.cuda()
    fcn_model = nn.DataParallel(fcn_model_0)"""
    return fcn_model


# apply dropout on line 144/145/146, but later is better
# To apply dropout even during .eval()
def enable_dropout(fcn_model, p):
    prob_list = []
    """fcn_model_0 = fcn_model.to('cuda:0')
    del fcn_model"""
    for each_module in fcn_model.modules():
        if each_module.__class__.__name__.startswith('Dropout'):
            each_module.train()
            prob_list.append(each_module.p)
            each_module.p = p
    """fcn_model_0 = fcn_model_0.cuda()
    fcn_model = nn.DataParallel(fcn_model_0)
    # print(fcn_model)"""
    return fcn_model, prob_list
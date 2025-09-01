import torch
import torchvision
import torch.nn as nn
from numpy.ma.extras import average
import torchvision.transforms
from OLD_FILES_DELETE_BEFORE_SUB.jh_data_loader import TrafficFlowDataSet
from torch.utils.data import DataLoader
from OLD_FILES_DELETE_BEFORE_SUB.jh_transforms import ToTensor, ScaleAndShiftX, ScaleY
import matplotlib.pyplot as plt
import torch.nn.functional as F




def gru():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #Hyper Parameters
    num_classes = 1
    num_epochs = 5
    batch_size = 16
    lr = 0.000005

    input_size = 42
    sequence_length = 12
    hidden_size = 1000
    num_layers = 1

    #TFV dataset
    dataset = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx",
                                 sequence_length=sequence_length,
                                 transform=ToTensor(),
                                 keep_date=False)

    # Composing the transforms
    composed = torchvision.transforms.Compose([
        ToTensor(),
        ScaleAndShiftX(feature_index=40, divisor=dataset.max_day),
        ScaleAndShiftX(feature_index=41, divisor=dataset.max_time),
        ScaleY(divisor=dataset.max_tfv)])

    # Putting the transforms into the datasets
    dataset.set_transform(composed)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])


    # Loading the training data
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    class GRU(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, num_classes):
            super(GRU, self).__init__()
            self._num_layers = num_layers
            self._hidden_size = hidden_size

            self._gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

            self._fc = nn.Linear(hidden_size, num_classes)

        def forward(self, x):

            h0 = torch.zeros(self._num_layers, x.size(0), self._hidden_size).to(device)

            out, _ = self._gru(x, h0)

            out = out[:, -1, :]

            out = self._fc(out)

            return out

    model = GRU(input_size, hidden_size, num_layers, num_classes).to(device)

    # Cost function
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    test_losses = []
    losses = []

    test_losses.append(test(test_loader, device, model, dataset.max_tfv))

    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.reshape([-1, 1]).to(device)


            yhat = model(x)


            cost = loss_function(yhat, y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                losses.append(cost.item())
                print("Epoch: " + str(epoch + 1) + " Epoch Percentage: " + str((i + 1) / len(train_loader)) + " Loss: " + str(cost.item()))
        test_losses.append(test(test_loader, device, model, dataset.max_tfv))

    plt.plot(test_losses)
    plt.show()

def test(test_loader, device, model, maxtfv):
    with torch.no_grad():
        test_losses = []
        for i, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.reshape([-1, 1]).to(device)
            yhat = model(x)
            cost = F.l1_loss(yhat, y)
            test_losses.append(cost.item())
    return average(test_losses) * maxtfv


from pyexpat import features

from ml_models.jh_data_loader import TrafficFlowDataSet
from torch.utils.data import DataLoader, Dataset
import numpy as np

if __name__ == "__main__":
    dataset = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx", sequence_length=3)

    train_loader = DataLoader(dataset=dataset,
                              batch_size=4,
                              shuffle=True)
    
    dataiter = iter(train_loader)
    data = next(dataiter)
    features, labels = data
    print(features, labels)

    print("Fin")
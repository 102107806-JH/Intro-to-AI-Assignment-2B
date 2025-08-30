from pyexpat import features

from ml_models.jh_data_loader import TrafficFlowDataSet
from torch.utils.data import DataLoader, Dataset
import numpy as np

if __name__ == "__main__":
    dataset = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx", sequence_length=3, keep_date=False)

    train_loader = DataLoader(dataset=dataset,
                              batch_size=1,
                              shuffle=False)
    


    total_samples = len(dataset)

    for i, (input, labels) in enumerate(train_loader):
        print(input, labels)



    print("Fin")
from pyexpat import features

from ml_models.jh_data_loader import TrafficFlowDataSet
from torch.utils.data import DataLoader, Dataset
import numpy as np

if __name__ == "__main__":
    dataset = TrafficFlowDataSet(data_set_file_name="data/model_data.xlsx", sequence_length=3, keep_date=True)

    train_loader = DataLoader(dataset=dataset,
                              batch_size=1,
                              shuffle=False)
    


    total_samples = len(dataset)

    for i, (input, labels) in enumerate(train_loader):
        first_date = input[0, 0, -1]
        last_date = input[0, -1, -1]
        if first_date != last_date:
            print(first_date, last_date)

        if first_date > last_date:
            print("error")
            break



    print("Fin")
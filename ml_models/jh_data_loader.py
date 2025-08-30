import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import openpyxl
import re
import csv
import pandas

class TrafficFlowDataSet(Dataset):
    def __init__(self, data_set_file_name, sequence_length):
        self._sequence_length = sequence_length

        self._scats_site_to_one_hot = {}
        self._index_to_data_set_index = {}
        self._days_to_values = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }
        self._length = None # Inited in a private member function


        data_list = self._put_data_set_in_list(data_set_file_name)  # Store the data in a list #

        self._index_to_data_point_populator(data_list, sequence_length=sequence_length)

        data_list = self._format_data(data_list)

        self._np_data_array = np.asarray(data_list)

    def __getitem__(self, index):
        ds_index = self._index_to_data_set_index[index]
        data = self._np_data_array[ds_index:ds_index + self._sequence_length, :]
        x = data[:,:-1]  # Remove labels #
        y = data[-1,-1]  # Only Interested in the last label #
        return x, y



    def __len__(self):
        return self._length

    def _put_data_set_in_list(self, data_set_file_name):
        pandas_data = pandas.read_excel(data_set_file_name)

        data_list = []
        scats_site_count = 0
        for i in range(0, pandas_data.shape[0]):
            scats_number = pandas_data['SCATS_Number'].iloc[i]
            day = pandas_data['Weekday'].iloc[i]

            if scats_number not in self._scats_site_to_one_hot:
                self._scats_site_to_one_hot[scats_number] = scats_site_count
                scats_site_count += 1

            time = 0
            for j in range(3, pandas_data.shape[1]):
                tfv = pandas_data.iat[i, j]
                data_list.append([scats_number, day, time, tfv])
                time += 1

        return data_list

    def _index_to_data_point_populator(self, data_list, sequence_length):

        data_set_index = 0

        for i in range(len(data_list) - sequence_length + 1): # + 1 because ranges last number is excluded
            start_datum = data_list[i]
            end_datum = data_list[i + sequence_length - 1]  # -1 because i is included. Eg: say we have a sequence length of 3 [i, i + 1, i + 2]


            start_time = start_datum[2]
            end_time = end_datum[2]
            if end_time > start_time:
                self._index_to_data_set_index[data_set_index] = i;
                data_set_index += 1

        self._length = data_set_index

    def _format_data(self, data_list):
        formated_data_list = []
        for datum in data_list:
            scats_number = datum[0]
            day = datum[1]
            time = datum[2]
            tfv = datum[3]

            formated_datum = [0] * len(self._scats_site_to_one_hot)
            idx = self._scats_site_to_one_hot[scats_number]
            formated_datum[idx] = 1

            formated_datum.append(self._days_to_values[day])

            formated_datum.append(time)

            formated_datum.append(int(tfv))

            formated_data_list.append(formated_datum)

        return formated_data_list



def print_list_ds(list):
    for element in list:
        print(element)


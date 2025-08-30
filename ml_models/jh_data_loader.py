import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import math
import openpyxl
import re
import csv
import pandas
from datetime import datetime

class TrafficFlowDataSet(Dataset):
    def __init__(self, data_set_file_name, sequence_length, keep_date=False):
        self._sequence_length = sequence_length
        self._keep_date = keep_date

        self._scats_site_to_one_hot = {}
        self._data_set_index_to_data_array_index = {}
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

        self._data_set_index_to_data_array_index_populator(data_list, sequence_length=sequence_length)

        data_list = self._format_data(data_list)

        self._np_data_array = np.asarray(data_list)

    def __getitem__(self, data_set_index):
        data_array_index = self._data_set_index_to_data_array_index[data_set_index]
        data = self._np_data_array[data_array_index : data_array_index + self._sequence_length, :]
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
            date = int(pandas_data['Date'].iloc[i].strftime("%d"))

            if scats_number not in self._scats_site_to_one_hot:
                self._scats_site_to_one_hot[scats_number] = scats_site_count
                scats_site_count += 1

            time = 0
            for j in range(3, pandas_data.shape[1]):
                tfv = pandas_data.iat[i, j]
                data_list.append([scats_number, day, time, tfv, date])
                time += 1

        return data_list

    def _data_set_index_to_data_array_index_populator(self, data_list, sequence_length):
        data_set_index = 0

        for data_array_index in range(len(data_list) - sequence_length + 1): # + 1 because ranges last number is excluded
            start_datum = data_list[data_array_index]
            end_datum = data_list[data_array_index + sequence_length - 1]  # -1 because i is included. Eg: say we have a sequence length of 3 [i, i + 1, i + 2]


            start_date = start_datum[4]
            end_date = end_datum[4]
            if end_date >= start_date:
                self._data_set_index_to_data_array_index[data_set_index] = data_array_index
                data_set_index += 1

        self._length = data_set_index

    def _format_data(self, data_list):
        formated_data_list = []
        for datum in data_list:
            scats_number = datum[0]
            day = datum[1]
            time = datum[2]
            tfv = datum[3]
            date = datum[4]

            formated_datum = [0] * len(self._scats_site_to_one_hot)
            idx = self._scats_site_to_one_hot[scats_number]
            formated_datum[idx] = 1

            formated_datum.append(self._days_to_values[day])

            formated_datum.append(time)

            if self._keep_date:
                formated_datum.append(date)

            formated_datum.append(int(tfv))

            formated_data_list.append(formated_datum)

        return formated_data_list


def print_list(list):
    for item in list:
        print(item)
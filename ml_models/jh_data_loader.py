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
    def __init__(self, data_set_file_name, sequence_length, transform=None, keep_date=False):
        self._sequence_length = sequence_length  # The sequence length of the RNN #
        self._transform = transform  # stores the transform to be performed on the data set #
        self._keep_date = keep_date  # The date is only kept for testing purposes #

        self._scats_site_to_one_hot = {}  # Dictionary that maps a scats site to its one hot encoding #
        self._data_set_index_to_data_array_index = {}  # Converts a dataset index to an array index #
        self._days_to_values = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6
        }  # Convert a day to a value #
        self._length = None # Inited in a private member function


        data_list = self._put_data_set_in_list(data_set_file_name)  # Store the data in a list #

        self._data_set_index_to_data_array_index_populator(data_list, sequence_length=sequence_length)  # Populate the "data_set_index_to_data_array_index" dictionary

        data_list = self._format_data(data_list)  # Formated data list #

        self._np_data_array = np.asarray(data_list, dtype=np.float32)  # Store the formated data in a np array #

        self._max_day = max(self._np_data_array[:, -3])
        self._max_time = max(self._np_data_array[:, -2])
        self._max_tfv = max(self._np_data_array[:, -1])

    @property
    def max_day(self):
        return self._max_day

    @property
    def max_time(self):
        return self._max_time

    @property
    def max_tfv(self):
        return self._max_tfv

    def set_transform(self, transform):
        self._transform = transform

    def __getitem__(self, data_set_index):
        data_array_index = self._data_set_index_to_data_array_index[data_set_index]  # Convert the dataset index into a data array index #
        data = self._np_data_array[data_array_index : data_array_index + self._sequence_length, :]  # Extract the sequence
        x_raw = data[:,:-1]  # Remove labels #
        y_raw = np.array(data[-1,-1])  # Only Interested in the last label #
        datum = x_raw, y_raw

        if self._transform:
            datum = self._transform(datum)

        return datum

    def __len__(self):
        return self._length  # Return the length of the dataset

    def _put_data_set_in_list(self, data_set_file_name):
        pandas_data = pandas.read_excel(data_set_file_name)  # Read the data excel file with pandas #

        data_list = []  # list to store the read data #
        scats_site_count = 0  # counts the number of unique scats sites
        for i in range(0, pandas_data.shape[0]):  # Go through every row #
            scats_number = pandas_data['SCATS_Number'].iloc[i]  # Scats number for given row #
            day = pandas_data['Weekday'].iloc[i]  # Day for given row #
            date = int(pandas_data['Date'].iloc[i].strftime("%d"))  # Day of the month formated in day formating #

            if scats_number not in self._scats_site_to_one_hot:  # If the scats number has not been seen #
                self._scats_site_to_one_hot[scats_number] = scats_site_count  # Map Scats number to site count #
                scats_site_count += 1  # Increase the site count #

            time = 0  # The time is just recorded as an integer
            for j in range(3, pandas_data.shape[1]):
                tfv = pandas_data.iat[i, j]  # Get traffic flow volume for a given time #
                data_list.append([scats_number, day, time, date, tfv])   # Create a new data point in the graph #
                time += 1  # Increment the time #

        return data_list  # Return the data list #

    def _data_set_index_to_data_array_index_populator(self, data_list, sequence_length):
        """
        The purpose of this function is to map an index in the data set to the
        underlying data array. This function filters out any indexes where
        there is a crossover between the start and the end of the same month.
        For example you cant feed sequence into the rnn that has its start on
        the 31st and its end on the 1st. If this occurs it means that there has
        been a 'wrap around' or there is a new SCATs site either way these points
        need to be excluded.
        """
        data_set_index = 0  # The index within the dataset #

        for data_array_index in range(len(data_list) - sequence_length + 1): # + 1 because ranges last number is excluded
            start_datum = data_list[data_array_index]  # First datum in the sequence
            end_datum = data_list[data_array_index + sequence_length - 1]  # -1 because i is included. Eg: say we have a sequence length of 3 [i, i + 1, i + 2]


            start_date = start_datum[3]  # The start date
            end_date = end_datum[3]  # The end date
            if end_date >= start_date:  # If the start date is ever greater than the end date we are at a crossover point and they are not valid data points #
                self._data_set_index_to_data_array_index[data_set_index] = data_array_index  # Valid point is recorded in the dictionary #
                data_set_index += 1  # Increment the index #

        self._length = data_set_index  # Record the length of the dataset #

    def _format_data(self, data_list):
        formated_data_list = []
        for datum in data_list:
            scats_number = datum[0]
            day = datum[1]
            time = datum[2]
            date = datum[3]
            tfv = datum[4]

            # Translate the scats site into a one hot encoded vector that appears at the start of the vector #
            formated_datum = [0] * len(self._scats_site_to_one_hot)
            idx = self._scats_site_to_one_hot[scats_number]
            formated_datum[idx] = 1

            # Appending the remaining datapoints (they dont need to be one hot encoded) #
            formated_datum.append(self._days_to_values[day])

            formated_datum.append(time)

            if self._keep_date:
                formated_datum.append(date)

            formated_datum.append(int(tfv))

            formated_data_list.append(formated_datum)

        return formated_data_list

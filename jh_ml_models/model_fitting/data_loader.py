import pandas
import numpy as np
from torch.utils.data import DataLoader, Dataset

class TrafficFlowDataSet(Dataset):
    def __init__(self, data_set_file_name, sequence_length, selected_scats_site):
        self._transform_dict = None

        self._sequence_length = sequence_length  # The sequence length of the RNN #

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

        data_list = self._put_data_set_in_list(data_set_file_name, selected_scats_site)  # Store the data in a list #

        self._data_set_index_to_data_array_index_populator(data_list, sequence_length)

        np_data_array = np.asarray(data_list, dtype=np.float32)

        self._np_data_array = self._transform_data(np_data_array)

    def __len__(self):
        return self._length

    def __getitem__(self, data_set_index):
        data_array_index = self._data_set_index_to_data_array_index[data_set_index]  # Convert the dataset index into a data array index #
        data = self._np_data_array[data_array_index: data_array_index + self._sequence_length,:]  # Extract the sequence
        x_raw = data[:, :-1]  # Remove labels #
        y_raw = np.array(data[-1, -1])  # Only Interested in the last label #
        datum = x_raw, y_raw


        return datum

    def _put_data_set_in_list(self, data_set_file_name, selected_scats_site):
        pandas_data = pandas.read_excel(data_set_file_name)  # Read the data excel file with pandas #

        data_list = []  # list to store the read data #
        scats_site_count = 0  # counts the number of unique scats sites
        for i in range(0, pandas_data.shape[0]):  # Go through every row #
            scats_number = pandas_data['SCATS_Number'].iloc[i]  # Scats number for given row #

            #only load data from selected site
            if  selected_scats_site != "ALL" and selected_scats_site != scats_number:
                continue

            day = pandas_data['Weekday'].iloc[i]  # Day for given row #
            date = int(pandas_data['Date'].iloc[i].strftime("%d"))  # Day of the month formated in day formating #

            time = 0  # The time is just recorded as an integer
            for j in range(3, pandas_data.shape[1]):
                tfv = pandas_data.iat[i, j]  # Get traffic flow volume for a given time #
                data_list.append([int(scats_number), self._days_to_values[day], time, date, int(tfv)])   # Create a new data point in the graph #
                time += 1  # Increment the time #

        # Creating the target for each data point #
        for i in range(len(data_list) - 1):
            next_tfv = data_list[i + 1][-1]
            data_list[i].append(next_tfv)

        data_list.pop()  # Last element doesnt have a index #

        overlap_months_removed_data_list = []
        # Removing datums where the target is in the next month
        for i in range(len(data_list) - 1):
            cur_day = data_list[i][3]
            next_day = data_list[i + 1][3]
            if next_day >= cur_day:
                overlap_months_removed_data_list.append(data_list[i])

        return overlap_months_removed_data_list  # Return the data list #

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

    def _transform_data(self, np_data_array):

        # Scaling the day
        day_feature = np_data_array[:, 1]
        max_day = np.max(day_feature)
        min_day = np.min(day_feature)
        day_feature = (day_feature - min_day) / (max_day - min_day)
        np_data_array[:, 1] = day_feature

        # Scaling the date
        time_feature = np_data_array[:, 2]
        max_time = np.max(time_feature)
        min_time = np.min(time_feature)
        time_feature = (time_feature - min_time) / (max_time - min_time)
        np_data_array[:, 2] = time_feature

        # Scaling the time
        date_feature = np_data_array[:, 3]
        max_date = np.max(date_feature)
        min_date = np.min(date_feature)
        date_feature = (date_feature - min_date) / (max_date - min_date)
        np_data_array[:, 3] = date_feature

        # Scaling TFV values
        tfv_feature = np_data_array[:, 4]
        tfv_label = np_data_array[:, 5]
        max_tfv = np.max(tfv_feature)
        min_tfv = np.min(tfv_feature)
        tfv_feature = (tfv_feature - min_tfv) / (max_tfv - min_tfv)
        tfv_label = (tfv_label - min_tfv) / (max_tfv - min_tfv)
        np_data_array[:, 4] = tfv_feature
        np_data_array[:, 5] = tfv_label

        #remove the scats site number
        np_data_array = np_data_array[:, 1:]

        self._transform_dict = {
            "max_day": max_day,
            "max_time": max_time,
            "max_date": max_date,
            "max_tfv": max_tfv,
            "min_day": min_day,
            "min_time": min_time,
            "min_date": min_date,
            "min_tfv": min_tfv
        }
        return np_data_array

    @property
    def transform_dict(self):
        return self._transform_dict

def print_ls(list):
    for item in list:
        print(item)

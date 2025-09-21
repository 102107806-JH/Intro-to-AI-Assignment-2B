import copy

import pandas
import datetime
import numpy as np
import pandas as pd
from dateutil.rrule import weekday
from numpy.ma.extras import average


class MockDataBaseCleaner():
    """
    The data from the 07/2025 and 08/2025 had some corrupted entries. More specifically
    some entire rows were set to zero and there were -1's for some values.
    """
    def __init__(self, path_to_database):
        self._np_data_arr = pd.read_excel(path_to_database).to_numpy()  # The excel file is now stored in a numpy array #
        self._first_tfv_col_idx = 3  # The index of the first column that contains traffic flow volume values#

    def clean_data(self):
        av_dict = self._get_day_time_averages()  # Returns a dictionary with the averages for each day of the week and timestep #
        self._fill_in_bad_days(av_dict)  # Fills in days that are all zeros with the corresponding average of each day element #
        self._remove_isolated_bad_input()  # Removes the -1's #
        self._write_to_excel()  # Write data to excel #

    def _get_day_time_averages(self):
        """
        :return: A dictionary that has keys that represent a day of the week and a time during that day.
        There is a unique key for each combination of day of the week and time. The value is the average
        corresponding to that day of the week and time.
        """
        av_dict = {}  # Output dictionary #
        for i in range(self._np_data_arr.shape[0]):  # Go through every row of the data array #
            #Row is bad disregard (we dont want to include it in the average)
            if np.sum(self._np_data_arr[i,self._first_tfv_col_idx:]) <= 0:  # We know the entire row is 0's if this is the case some of these rows also had -1's thus the  "<= 0" #
                continue

            weekday = self._np_data_arr[i, 1]  # Get the weekday for the row #
            for time, j in enumerate(range(self._first_tfv_col_idx, self._np_data_arr.shape[1])):  # Loop through all the possible times #
                key = f"{weekday} {time}"  # Construct the dictionary key #

                if self._np_data_arr[i, j] < 0:  # Means that the cell contains a -1 as such we dont want this value to be involved in average calculation #
                    continue # bad input less than 0
                elif key not in av_dict:  # Add a new key to the dictionary #
                    av_dict[key]=[self._np_data_arr[i, j], 1]  # The list elements are temporary. The 0 element is the sum corresponding to the key and the 1 element is the count of the number of those elements #
                else:  # Key has already been encountered #
                    entry = av_dict[key]  # Get the entry for the key #
                    entry[0] += self._np_data_arr[i, j]  # Performing tfv sum #
                    entry[1] += 1  # Counting the number of key occurances #
        # Calculate the average #
        for key in av_dict:
            av_dict[key] = av_dict[key][0] / av_dict[key][1]

        return av_dict # Return the average dictionary

    def _fill_in_bad_days(self, av_dict):

        for i in range(self._np_data_arr.shape[0]):  # Go through entire array #
            #Row is not bad continue
            if np.sum(self._np_data_arr[i,self._first_tfv_col_idx:]) > 0:
                continue

            weekday = self._np_data_arr[i, 1]  # Get the week day
            for time, j in enumerate(range(self._first_tfv_col_idx, self._np_data_arr.shape[1])):  # Go through all tfv value for the given row
                key = f"{weekday} {time}"  # Construct the dictionary key #
                overwrite_val = av_dict[key]  # Get the average from the dictionary #
                self._np_data_arr[i, j] = overwrite_val  # Overwrite the value in the data array #

    def _remove_isolated_bad_input(self):
        for i in range(self._np_data_arr.shape[0]):  # Go through all entries in the data array #
            weekday = self._np_data_arr[i, 1]  # Get the weekday #
            for time, j in enumerate(range(self._first_tfv_col_idx, self._np_data_arr.shape[1])):
                key = f"{weekday} {time}"  # Get the key (turned out to be uncesary) #
                if self._np_data_arr[i, j] < 0:
                    #All isolated -1's removed. Calculate the average of past and future vals if not possible just get the past or the future value
                    if j == self._first_tfv_col_idx:
                        self._np_data_arr[i, j] = self._np_data_arr[i, j+1]
                    elif j == self._np_data_arr.shape[1] - 1:
                        self._np_data_arr[i, j] = self._np_data_arr[i, j-1]
                    else:
                        self._np_data_arr[i, j] = (self._np_data_arr[i, j-1] + self._np_data_arr[i, j+1])/2

                # -1's occur in a row take the previous days value at this time
                if self._np_data_arr[i, j] < 0:
                    self._np_data_arr[i, j] = self._np_data_arr[i-1, j]
        # After this point the file was checked and it was confirmed that there was no more bad input however it is possible this could change with
        # another file and further processing would have to done.





    def _write_to_excel(self):
        pandas.DataFrame(self._np_data_arr).to_excel("data/data_base_test_run.xlsx", index=False)






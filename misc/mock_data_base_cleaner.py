import pandas
import datetime
import numpy as np
import pandas as pd


class MockDataBaseCleaner():
    def __init__(self, path_to_database):
        self._dataframe = pd.read_excel(path_to_database)

    def process(self):
        self._remove_negatives()
        date_av_dictionary = self._get_average_value_for_each_date()
        self._replace_zero_days_with_average(date_av_dictionary)
        self._write_to_new_excel()

    def _remove_negatives(self):
        for i in range(self._dataframe.shape[0]):
            for j in range(self._dataframe.shape[1]):

                # Average replacing isolated -1's
                if type(self._dataframe.iat[i, j]) == np.int64 and self._dataframe.iat[i, j] < 0:
                    if j == 3: # The first timestep for the day cant take the previous value as it is a date
                        next_val = self._dataframe.iat[i, j + 1]
                        self._dataframe.iat[i, j] = next_val
                    elif j == self._dataframe.shape[1] - 1:  # The last timestep for the day cant take the next value as it does not exist
                        previous_val = self._dataframe.iat[i, j - 1]
                        self._dataframe.iat[i, j] = previous_val
                    else:
                        next_val = self._dataframe.iat[i, j + 1]
                        previous_val = self._dataframe.iat[i, j - 1]
                        self._dataframe.iat[i, j] = int((previous_val + next_val) / 2)  # Take the average of the current and the previous val

                #Average replacing for multiple -1's in a row. All isolated -1's have been replaced.
                if type(self._dataframe.iat[i, j]) == np.int64 and self._dataframe.iat[i, j] < 0:
                    if type(self._dataframe.iat[i - 1, j]) == np.int64:
                        self._dataframe.iat[i, j] = self._dataframe.iat[i - 1, j]  # if the previous days data is an int set the current days data to it
                    else:
                        self._dataframe.iat[i, j] = 0  # The previous days data was not an int so zero it

    def _get_average_value_for_each_date(self):
        date_av_dictionary = {}
        for i in range(self._dataframe.shape[0]):
            date = self._dataframe.iat[i, 2]
            if date not in date_av_dictionary:
                date_av_dictionary[date] = []

            row_tfv_list = []
            # Calculate the row average. To determine if it needs to be included in the average
            for j in range(3, self._dataframe.shape[1]):
                row_tfv_list.append(self._dataframe.iat[i, j])

            if np.average(np.asarray(row_tfv_list)) != 0:  # Exclude rows that contain only zeros #
                date_av_dictionary[date].append(row_tfv_list)  # Is storing all valid entries for a given date


        for date in date_av_dictionary:
            date_av_dictionary[date] = np.asarray(date_av_dictionary[date])  # Convert to numpy array for easier processing #
            date_av_dictionary[date] = np.mean(date_av_dictionary[date], axis=0)  # Convert the average for each time step #

        return date_av_dictionary

    def _replace_zero_days_with_average(self, date_av_dictionary):
        for i in range(self._dataframe.shape[0]):
            date = self._dataframe.iat[i, 2]
            row_tfv_list = []

            for j in range(3, self._dataframe.shape[1]):
                row_tfv_list.append(self._dataframe.iat[i, j])

            if np.average(np.asarray(row_tfv_list)) != 0:  # Row is not zero based and does not need replacing we can therfore continue to the next row #
                continue

            date_av_list = list(date_av_dictionary[date])  #Get the average for given date
            for av_list_idx, j in enumerate(range(3, self._dataframe.shape[1])):
                self._dataframe.iat[i, j] = int(date_av_list[av_list_idx])


    def _write_to_new_excel(self):
        self._dataframe.to_excel("data/cleaned_data_base.xlsx", index=False)






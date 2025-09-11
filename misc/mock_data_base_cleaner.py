import copy

import pandas
import datetime
import numpy as np
import pandas as pd
from dateutil.rrule import weekday
from numpy.ma.extras import average


class MockDataBaseCleaner():
    def __init__(self, path_to_database):
        self._np_data_arr = pd.read_excel(path_to_database).to_numpy()
        self._first_tfv_col_idx = 3

    def clean_data(self):
        av_dict = self._get_day_time_averages()
        self._fill_in_bad_days(av_dict)
        self._remove_isolated_bad_input()
        self._write_to_excel()

    def _get_day_time_averages(self):
        av_dict = {}
        for i in range(self._np_data_arr.shape[0]):
            #Row is bad disregard
            if np.sum(self._np_data_arr[i,self._first_tfv_col_idx:]) <= 0:
                continue

            weekday = self._np_data_arr[i, 1]
            for time, j in enumerate(range(self._first_tfv_col_idx, self._np_data_arr.shape[1])):
                key = f"{weekday} {time}"

                if self._np_data_arr[i, j] < 0:
                    continue # bad input less than 0
                elif key not in av_dict:
                    av_dict[key]=[self._np_data_arr[i, j], 1]
                else:
                    entry = av_dict[key]
                    entry[0] += self._np_data_arr[i, j]
                    entry[1] += 1

        for key in av_dict:
            av_dict[key] = av_dict[key][0] / av_dict[key][1]

        return av_dict

    def _fill_in_bad_days(self, av_dict):

        for i in range(self._np_data_arr.shape[0]):
            #Row is not bad continue
            if np.sum(self._np_data_arr[i,self._first_tfv_col_idx:]) > 0:
                continue

            weekday = self._np_data_arr[i, 1]
            for time, j in enumerate(range(self._first_tfv_col_idx, self._np_data_arr.shape[1])):
                key = f"{weekday} {time}"
                overwrite_val = av_dict[key]
                self._np_data_arr[i, j] = overwrite_val

    def _remove_isolated_bad_input(self):
        for i in range(self._np_data_arr.shape[0]):
            weekday = self._np_data_arr[i, 1]
            for time, j in enumerate(range(self._first_tfv_col_idx, self._np_data_arr.shape[1])):
                key = f"{weekday} {time}"
                if self._np_data_arr[i, j] < 0:
                    #All isolated -1's removed
                    if j == self._first_tfv_col_idx:
                        self._np_data_arr[i, j] = self._np_data_arr[i, j+1]
                    elif j == self._np_data_arr.shape[1] - 1:
                        self._np_data_arr[i, j] = self._np_data_arr[i, j-1]
                    else:
                        self._np_data_arr[i, j] = (self._np_data_arr[i, j-1] + self._np_data_arr[i, j+1])/2

                # -1's occur in a row take the previous days value at this time
                if self._np_data_arr[i, j] < 0:
                    self._np_data_arr[i, j] = self._np_data_arr[i-1, j]





    def _write_to_excel(self):
        pandas.DataFrame(self._np_data_arr).to_excel("data/cleaned_data_base.xlsx", index=False)






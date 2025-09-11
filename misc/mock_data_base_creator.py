import sys
import datetime
from venv import create
import xlsxwriter
import pandas


class MockDataBaseCreator():
    """
    This class uses REAL DATA to create a mock database the data is from 07/2025 and 08/2025
    """
    def __init__(self, file_path):
        self._file_path = file_path  # Where the data will be stored #
        self._scats_site_list = []
        self._populate_scats_site_list()
        self._number_to_weekday = {
            0 : "Monday",
            1 : "Tuesday",
            2 : "Wednesday",
            3 : "Thursday",
            4 : "Friday",
            5 : "Saturday",
            6 : "Sunday"
        }
        self._data_dictionary = {}

        for scat_number in self._scats_site_list:  # Put an empty list for each scats site #
            self._data_dictionary[scat_number] = []

    def _populate_scats_site_list(self):
        """
        Populate a list with all the different SCAT values
        :return:None
        """
        path = "data/model_data.xlsx"
        pandas_data = pandas.read_excel(path)
        for i in range(pandas_data.shape[0]):
            scat_number = pandas_data.iat[i, 0]
            if scat_number not in self._scats_site_list:
                self._scats_site_list.append(int(scat_number))

    def write_new_excel_file(self):
        path_prefix = "data/traffic_signal_volume_cur/VSDATA_2025"  # Path prefix where of all the files with the data for 07 and 08

        # Get list of months in days in string format 2 characters long
        months = self._format_list_content(list(range(7,9)))
        days = self._format_list_content(list(range(1, 32)))


        for m_string in months:
            for d_string in days:
                path_suffix = m_string + d_string + ".csv"  # The path suffix to the file that contains the data #
                print("Extracting Data From: " + path_prefix + path_suffix)
                pandas_data = pandas.read_csv(path_prefix + path_suffix)  # Get the data into a pandas data frame #
                self._populate_data_dictionary(pandas_data, m_string, d_string)  # Combine the data from the seperate files into one dictionary #

        file_as_list = []  # List that will store all the data enabling writing to an excel file #
        self._add_columns_headings_to_list(file_as_list)  # Adding column headings to the files in the form of the list #
        # Append all the single days of data to the list from the dictionary
        for scat_key in self._data_dictionary:
            scat_site_data_list = self._data_dictionary[scat_key]
            for single_days_data in scat_site_data_list:
                file_as_list.append(single_days_data)


        file_as_pd_data_frame = pandas.DataFrame(file_as_list)  # Convert the list to a pandas data frame
        # Writing to and Excel file#
        writer = pandas.ExcelWriter(self._file_path , engine='xlsxwriter')
        file_as_pd_data_frame.to_excel(writer, sheet_name="Current_Data", index=False)
        writer._save()

    def _add_columns_headings_to_list(self, list):
        headings = []
        # First three headings #
        headings.append("SCATS_Number")
        headings.append("Weekday")
        headings.append("Date")
        # Adding all the time headings #
        for hour in range(24):
            for minute in range(0, 60, 15):

                hour_str = str(hour)
                if minute < 10:
                    minute_str = "0" + str(minute)
                else:
                    minute_str = str(minute)

                headings.append(hour_str + ":" + minute_str)

        list.append(headings)

    def _format_list_content(self, list):
        new_list = []
        # Format the times correctly
        for item in list:
            new_item = ""
            if item < 10:  # If less then 10 we need to place a zero at the start
                new_item += "0" + str(item)
            else:
                new_item += str(item)

            new_list.append(new_item)

        return new_list

    def _populate_data_dictionary(self, pandas_data, m_string, d_string):

        current_date = datetime.datetime(2025, int(m_string), int(d_string))

        for scat_number in self._scats_site_list:  # Go through all the different scats sites #
            for i in range(pandas_data.shape[0]):  # Got through the entire Excel file #
                cur_entry_list = []  # Stores the entry for the current row #
                cur_entry_list.append(scat_number)
                cur_entry_list.append(self._number_to_weekday[current_date.weekday()])
                cur_entry_list.append(current_date.strftime("%d/%m/%Y"))

                cur_scats = pandas_data.iat[i, 0]
                if cur_scats != scat_number:
                    continue

                for j in range(3, pandas_data.shape[1] - 4):
                    cur_entry_list.append(int(pandas_data.iat[i, j]))  # Append all the tfv values

                self._data_dictionary[scat_number].append(cur_entry_list)
                break # Only take the first appearance of the scat site

    def _print_ls(self, list):
        for item in list:
            print(item)













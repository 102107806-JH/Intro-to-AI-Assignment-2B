from misc.mock_data_base_creator import MockDataBaseCreator
if __name__ == "__main__":
    file_path_to_data_base = 'data/uncleaned_data_base_test_run.xlsx'
    database = MockDataBaseCreator(file_path_to_data_base)
    database.write_new_excel_file()

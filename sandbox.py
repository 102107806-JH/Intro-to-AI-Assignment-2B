from helper_functions.mock_data_base_creator import MockDataBaseCreator
if __name__ == "__main__":
    database = MockDataBaseCreator()
    database.write_new_excel_file()

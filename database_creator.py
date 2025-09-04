from data_base.mock_data_base_creator import MockDataBaseCreator
if __name__ == "__main__":
    database = MockDataBaseCreator()
    database.write_new_excel_file()

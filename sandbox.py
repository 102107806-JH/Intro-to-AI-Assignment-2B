from jh_ml_models.mock_data_base import MockDataBaseCreator
if __name__ == "__main__":
    database = MockDataBaseCreator()
    database._write_new_excel_file()
    print("Hello World")
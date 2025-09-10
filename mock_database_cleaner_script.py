from misc.mock_data_base_cleaner import MockDataBaseCleaner
if __name__ == "__main__":
    database_processor = MockDataBaseCleaner(path_to_database="data/uncleaned_data_base.xlsx")
    database_processor.clean_data()
import logging
import os
import pandas as pd


def load_excel(file_name: str) -> pd.DataFrame:
    """
    Reads an excel file into a DataFrame. Looks for the file in the working directory and in the
    data/raw sub-folder. Throws FileNotFoundError

    :param file_name: Name of the excel file to load
    :return: A data frame containing the data in the excel file
    """
    data_folder = os.path.join('data')
    data_path = os.path.join(data_folder, file_name)
    raw_data_folder = os.path.join('data', 'raw')
    raw_data_path = os.path.join(raw_data_folder, file_name)
    processed_data_folder = os.path.join('data', 'processed')
    processed_data_path = os.path.join(processed_data_folder, file_name)
    project_paths = [file_name, data_path, raw_data_path, processed_data_path]
    notebook_paths = [os.path.join('..', p) for p in project_paths]
    search_paths = project_paths + notebook_paths
    for p in search_paths:
        try:
            excel_raw = pd.read_excel(p)
            logging.info('Found data source file at %s' % p)
            return excel_raw
        except FileNotFoundError:
            logging.info('Could not find data source file at %s' % p)
    msg = 'Could not find source data file ' + file_name
    raise FileNotFoundError(msg)

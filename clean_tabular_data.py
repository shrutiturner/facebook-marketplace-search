from math import prod
from numpy import product
import pandas as pd


def clean_price(price_column: pd.Series) -> pd.Series:
    """Function taking a pandas series containing prices to remove all characters that are not digits or '.' and convert values to float.

    Args:
        price_column (pd.Series): pandas series of price data in string format.

    Returns:
        pd.Series: pandas series of clean price data in float format.
    """
    
    clean_column = price_column.replace(to_replace='[^0-9.]', value='', regex=True)

    float_column = pd.to_numeric(clean_column)

    return float_column


def clean_long_names(column: pd.Series, splitter: str) -> pd.Series:
    """Function takes in the product name column and retrives the test before the first | for clarity.

    Args:
        product_column (pd.Series): pandas series containing the product names retrieved from facebook marketplace.

    Returns:
        pd.Series: pandas series with cleaned product names.
    """

    product_name_series = column.str.split(splitter).str.get(0)

    clean_product_series = product_name_series.str.strip()

    return clean_product_series


def convert_category(category_column: pd.Series) -> pd.Series:
    """Function taking in a pandas series and converts it to type 'category'.

    Args:
        category_column (pd.Series): pandas series of data to be changed to category type.

    Returns:
        pd.Series: pandas series of type category.
    """

    column_cat_type = category_column.astype('category')

    return column_cat_type


def convert_date(date_column: pd.Series) -> pd.Series:
    """Function to convert a column of objects with date information to datetime type.

    Args:
        time_column (pd.Series): pandas series of strings containing date information.

    Returns:
        pd.Series: pandas series containing type date time in format %d/%m/%Y.
    """

    date_column =  pd.to_datetime(date_column)
    formatted_time_column = date_column.dt.strftime('%d/%m/%Y')

    return formatted_time_column


def convert_integer(integer_column: pd.Series) -> pd.Series:
    """Function to convert a column of string values containing integers to integers.

    Args:
        integer_column (pd.Series): pandas series of strings containing integer characters.

    Returns:
        pd.Series: pandas series containing type integer.
    """

    integer_column = integer_column.astype('int64')

    return integer_column


def get_tabular_data(filepath: str, lineterminator: str = ",") -> pd.DataFrame:
    """Function to import data from a csv file and save to a pandas dataframe, dropping all rows with missing data.

    Args:
        filepath (str): string of the path of the file to be imported.
        lineterminator (str, optional): string to state the line terminator used in the csv file. Defaults to ",".

    Returns:
        pd.DataFrame: dataframe of the csv contents.
        
    """
    df = pd.read_csv(filepath, lineterminator=lineterminator)

    df.rename(columns={'create_time\r':'create_time'}, inplace=True)

    return df


def get_and_normalise_data(file_path, lineterminator):
    """Combines defined functions to get the data from the file and reformat it as required.

    Args:
        filepath (str): string of the path of the file to be imported.
        lineterminator (str, optional): string to state the line terminator used in the csv file. Defaults to ",".

    Returns:
        pd.DataFrame: reformatted dataframe of imported data.
    """
    tab_data = get_tabular_data(file_path, lineterminator)

    tab_data['price'] = clean_price(tab_data['price'])

    tab_data['category'] = clean_long_names(tab_data['category'], '/')
    tab_data['category'] = convert_category(tab_data['category'])
    tab_data['location'] = convert_category(tab_data['location'])

    tab_data['product_name'] = clean_long_names(tab_data['product_name'], '|')

    tab_data['page_id'] = convert_integer(tab_data['page_id'])

    tab_data['create_time'] = convert_date(tab_data['create_time'])

    tab_data.rename(columns = {'id':'product_id'}, inplace = True)

    return tab_data


if __name__ == "__main__":
    file_path = "Products.csv"
    lineterminator = "\n"

    get_and_normalise_data(file_path, lineterminator)
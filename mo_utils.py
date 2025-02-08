import numpy as np
import pandas as pd
import configparser
import os
from datetime import datetime
import datetime as dt
from dateutil import parser
import ast
import loguru 
from loguru import logger
import ntplib
from time import ctime
import sys

##########################################################################################################################################
#                                                            Config                                                                      #
##########################################################################################################################################


def get_config_value(config, group, value):
    """
    Get the value of a specific option from a section in the config.

    Parameters:
        config (configparser.ConfigParser): The ConfigParser object containing the config file data.
        section (str): The section name in the config file.
        option (str): The option whose value is required.

    Returns:
        Any: The value of the specified option, parsed as its appropriate Python data type.
    """
    if not config.has_section(group):
        raise ValueError(f"Section '{group}' not found in the config file")

    if not config.has_option(group, value):
        raise ValueError(f"Option '{value}' not found in the section '{group}'")

    # Evaluate the value to its appropriate Python data type
    raw_value = config.get(group, value)
    try:
        raw_value =ast.literal_eval(raw_value) 
        return raw_value
    except (SyntaxError, ValueError) as e:
        logger.info(f"Error evaluating value: {e}")
        # If the value can't be evaluated, return it as a string
        return raw_value
    except Exception as e:
        logger.info(f"Unexpected error: {e}")
        # Return the raw value in case of any other unexpected errors
        return raw_value


def read_config_file(file_path):
    """
    Read and parse the config file from the specified path.

    Parameters:
        file_path (str): The path to the config file.

    Returns:
        configparser.ConfigParser: The ConfigParser object containing the config file data.
    """
    if file_path is None:
        pass
    else:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file not found at '{file_path}'")

    config = configparser.ConfigParser()
    config.read(file_path)
    return config

##########################################################################################################################################
#                                                            Dates                                                                       #
##########################################################################################################################################

def get_days_offset(date, offset, before=True, is_business_day=True, holidays=None):
    if holidays is not None:
        # Format the holidays for use with numpy's busdaycalendar
        holidays = [np.datetime64(holiday) for holiday in holidays]

    if is_business_day:
        # Create a custom calendar that counts weekends as non-business days
        cal = np.busdaycalendar(weekmask='1111100', holidays=holidays)
        # Convert the date to a numpy datetime64 object
        date_np = np.datetime64(date)
        # Calculate the offset date
        offset_np = np.busday_offset(date_np, offset, roll='forward' if before else 'backward', busdaycal=cal)
        result_date = np.datetime_as_string(offset_np, unit='D')
    else:
        # If not using business days, just add/subtract the offset to/from the date
        date_pd = pd.to_datetime(date)
        if before:
            result_date = (date_pd - pd.Timedelta(days=offset)).strftime('%Y-%m-%d')
        else:
            result_date = (date_pd + pd.Timedelta(days=offset)).strftime('%Y-%m-%d')

    return result_date


def get_offset_datetime(start_datetime, offset, unit='D', before=True):
    offset_td = pd.Timedelta(offset, unit)
    start_datetime = pd.to_datetime(start_datetime)
    
    if before:
        result_datetime = start_datetime - offset_td
    else:
        result_datetime = start_datetime + offset_td

    return result_datetime


def generate_date_series(start_date, end_date, **kwargs):
    """
    business_days_only = True or False
    holidays = series of dates 
    
    """
    
    business_days_only = kwargs.get('business_days_only', False)
    holidays = kwargs.get('holidays', None)

    start_date = convert_date_format(start_date, return_datetime_object=True)
    end_date = convert_date_format(end_date, return_datetime_object=True)
    
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)


    if business_days_only:
        dates = pd.date_range(start_date, end_date, freq='B')
    else:
        dates = pd.date_range(start_date, end_date, freq='D')
    
    if holidays is not None:
        # Make sure the dates in the holidays DataFrame are in the right format.
        holidays.iloc[:,0] = pd.to_datetime(holidays.iloc[:,0])
        # Remove the holidays from the date series.
        dates = dates[~dates.isin(holidays.iloc[:,0])]
    
    dates = pd.to_datetime(dates).to_series().dt.strftime('%Y-%m-%d')
    dates = pd.DatetimeIndex(dates)   
    dates = dates.normalize()
    
    return dates


def generate_time_series(start_date, end_date, frequency, holidays=None, format=None):
    freq_dict = {
        '5y': '5AS',
        '2yr': '2AS',
        '1y': 'AS',
        'semi-annual': '6MS',
        'quarterly': '3MS',
        '1M': 'MS',
        'bi-weekly': '2W',
        '1w': 'W',
        '3d': '3D',
        '1d': 'D',
        '12h': '12h',
        '4h': '4h',
        '1h': 'h',
        '30m': '30min',  # '30T' instead of '30M'
        '15m': '15min',  # '15T' instead of '15M'
        '5m': '5min',    # '5T' instead of '5M'
        '1m': 'min'      # 'T' instead of 'M' for minutes
    }

    # Adjust the end_date to the end of the day for intraday frequencies
    if frequency in ['12h', '4h', '1h', '30m', '15m', '5m', '1m']:
        end_date = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)

    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.strftime("%Y-%m-%d %H:%M:%S")  # Include time for start_date
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.strftime("%Y-%m-%d %H:%M:%S")  # Include time for end_date

    if frequency not in freq_dict:
        return "Invalid frequency specified"
    
    try:
        dates = pd.date_range(start_date, end_date, freq=freq_dict[frequency])
    except Exception as e:
        print('Invalid frequency:', e)
        return None

    # Apply floor only for fixed frequencies like H, T (minutes), and D
    fixed_frequencies = ['H', 'T', 'D']
    if freq_dict[frequency] in fixed_frequencies:
        current_time = pd.Timestamp.utcnow().floor(freq_dict[frequency])
        current_time = current_time.tz_localize(None)
        dates = dates[dates <= current_time]
    
    if holidays is not None:
        # Make sure the dates in the holidays DataFrame are in the right format.
        print('Adding holidays')
        holidays['Date'] = pd.to_datetime(holidays['Date'])
        # Remove the holidays from the date series.
        dates = dates[~dates.isin(holidays['Date'])]
    
    if format is not None:
        dates = dates.strftime(format)

    return dates

def split_intervals(start_date, end_date, complexity_score, frequency, holidays=None, format=None):
        
        
        num_intervals = int(max(1, complexity_score // 50000))
        
        if complexity_score % 50000 != 0:
            num_intervals += 1
            
        # Generate the date range with the specified frequency
        date_range = generate_time_series(start_date, end_date, frequency, holidays, format)
        
        if date_range is None:
            return None
        
        # Calculate the indices at which to split the date range
        indices = np.linspace(0, len(date_range) - 1, num_intervals + 1).astype(int)
        
       # Select the start date, split dates, and end date
        interval_dates = [date_range[indices[0]]] + [date_range[i] for i in indices[1:-1]] + [date_range[indices[-1]]]

        # Convert the datetime objects back to strings in the specified format
        if format:
            divided_intervals = [datetime.strptime(dt, format).strftime(format) if isinstance(dt, str) else dt.strftime(format) for dt in interval_dates]
        else:
            divided_intervals = interval_dates
    
        return divided_intervals


def convert_date_format(date, format_in=None, format_out=None, return_datetime_object=True):
    """
    format_in = format of current variable 
    format_out = format of reformatted variable 
    return_datetime_object = if return datetime object or string, default=False 
    """

    if format_in is None:
        # Try to parse the date using dateutil.parser
        try:
            date = parser.parse(date)
        except:
            print('Could not parse the input date. Please provide a valid format.')
            return None
    else:
        try:
            # Convert the string to a datetime object using the provided input format
            date = pd.to_datetime(date, format=format_in)
        except:
            print('Could not convert the input date to a datetime object. Please check the input format.')
            return None

    if return_datetime_object:
        # Return the date as a datetime object
        return date
    else:
        # Convert the datetime object to a string using the provided output format
        if format_out is not None:
            return date.strftime(format_out)
        else:
            print("format_out not provided for conversion to string. Returning datetime object.")
            return date
    

def read_holidays_from_excel(file_path):
    """
    Read holidays from the specified Excel file.

    Parameters:
        file_path (str): The path to the Excel file containing holidays.

    Returns:
        list: A list of holiday dates in the format 'YYYY-MM-DD'.
    """
    try:
        df = pd.read_excel(file_path, header=True, names=['Date'])
        holidays_list = df['Date'].astype(str).tolist()
        return holidays_list
    except FileNotFoundError:
        print(f"Excel file not found at '{file_path}'")
        return []
    except Exception as e:
        print(f"Error occurred while reading the Excel file: {e}")
        return []


import pandas as pd


def get_ntp_time(server="pool.ntp.org", in_ms =False):
    try:
        ntp_client = ntplib.NTPClient()
        response = ntp_client.request(server, version=3)
        if in_ms:
            current_time_in_milliseconds = int(response.tx_time * 1000)
            return current_time_in_milliseconds
        else:
            return ctime(response.tx_time)
    except Exception as e:
        print(f"Failed to get NTP time: {e}")
        return None

def convert_timestamp_to_datetime(timestamp):
        """
        REQ:        timestamp       timestamp in unix ms format

        Returns:    datetime:       datetime object of format "Y-m-d H:M" 
        """
        if isinstance(timestamp, pd.Series):
            # Apply the conversion to each element in the series
            return timestamp.apply(convert_timestamp_to_datetime)
        else:
            timestamp_sec = timestamp / 1000.0
            datetime_object = dt.datetime.fromtimestamp(timestamp_sec)

            # Format datetime object as "%Y-%m-%d %H:%M:%S"
            formatted_datetime = datetime_object.strftime("%Y-%m-%d %H:%M:%S")
        return formatted_datetime
        
def convert_datetime_to_timestamp(datetime_input):
    """
    REQ:        datetime       datetime object of format "Y-m-d H:M" 

    Returns:    timestamp:       timestamp in unix ms format
    """
    if isinstance(datetime_input, pd.Series):
        # Apply the conversion to each element in the series
        return datetime_input.apply(convert_datetime_to_timestamp)
    elif isinstance(datetime_input, str):
        # Convert date string to datetime object
        datetime_object = dt.datetime.strptime(datetime_input, "%Y-%m-%d %H:%M:%S")
    elif isinstance(datetime_input, dt.datetime):
        # Use datetime object directly
        datetime_object = datetime_input
    else:
        raise ValueError("Input must be a string, datetime object, or pandas Series")

    # Convert datetime object to Unix timestamp in milliseconds
    unix_timestamp_ms = int(datetime_object.timestamp() * 1000)
    return unix_timestamp_ms

##########################################################################################################################################
#                                                            Directory (OS) functions                                                                       #
##########################################################################################################################################

def create_folder_in_directory(directory_path, folder_name):
    # Join the directory path and folder name to get the full path
    full_path = os.path.join(directory_path, folder_name)
    
    # Check if the directory already exists
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Folder '{folder_name}' created in '{directory_path}'")
    else:
        print(f"Folder '{folder_name}' already exists in '{directory_path}'")

def find_config_path():
    for path in sys.path:
        if "Config" in path:  # Check if 'Config' is part of the path
            #print(f"Config path found: {path}")
            return path
    print("Config path not found.")
    return None

def find_trading_path():
    for path in sys.path:
        # Split the path to analyze its components
        parts = path.split(os.sep)
        # Check if the last part of the path is 'Trading'
        if parts[-1] == "Trading":
            #print(f"Trading path found: {path}")
            return path
    print("Trading path not found.")
    return None

def find_logging_path():
    for path in sys.path:
        # Split the path to analyze its components
        parts = path.split(os.sep)
        # Check if the last part of the path is 'Trading'
        if parts[-1] == "Logging":
            #print(f"Trading path found: {path}")
            return path
    print("Logging path not found.")
    return None
##########################################################################################################################################
#                                                            Data alteration functions                                                                       #
##########################################################################################################################################

def clean_and_convert(value):
    try:
        # Remove non-numeric characters and convert to float
        cleaned_value = ''.join(filter(str.isdigit, value))
        return float(cleaned_value)
    except ValueError:
        return float('NaN')


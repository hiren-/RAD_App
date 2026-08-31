import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import pandas as pd
import numpy as np
from tvDatafeed import TvDatafeed, Interval
from datetime import datetime
import sqlite3
from sqlite3 import Error
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, List
import pytz
import warnings
from abc import ABC, abstractmethod
import time
from symbols import symbols

warnings.filterwarnings('ignore', message='.*SSL.*')

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    formatter = logging.Formatter(
        '%(asctime)s|%(name)s|%(levelname)s|%(lineno)s|%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = RotatingFileHandler(
        log_file, maxBytes=2*1024*1024, backupCount=10
    )
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class IndicatorProcessor(ABC):
    def __init__(self, db_path: str, table_name: str):
        self.tv = TvDatafeed()
        # self.tv.connect()
        self.db_path = db_path
        self.table_name = table_name
        self.logger = setup_logger(f'{table_name}_processor', f'{table_name}_processor.log')


    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


    @abstractmethod
    def prepare_data(self, df: pd.DataFrame, timeframe: str) -> List[tuple]:
        pass

    def clean_table(self):
        pass

    def get_interval(self, timeframe: str) -> Interval:
        tf_map = {
            '1m': Interval.in_1_minute,
            '3m': Interval.in_3_minute,
            '5m': Interval.in_5_minute,
            '15m': Interval.in_15_minute,
            '30m': Interval.in_30_minute,
            '4h': Interval.in_4_hour,
            '1d': Interval.in_daily,
            '1w': Interval.in_weekly,
            '1M': Interval.in_monthly
        }
        return tf_map.get(timeframe)

    def get_db_connection(self):
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except Error as e:
            self.logger.error(f"Database connection error: {str(e)}")
            return None


    def save_to_db(self, df: pd.DataFrame, timeframe: str, batch_size: int = 1000) -> bool:
        if df is None or df.empty:
            return False

        conn = self.get_db_connection()
        if not conn:
            return False

        try:
            cur = conn.cursor()
            self.create_table(cur)
            data = self.prepare_data(df, timeframe)
            
            # SQLite upsert syntax
            columns = self.get_all_columns()
            update_cols = self.get_update_columns()
            
            insert_sql = f"""
            INSERT INTO {self.table_name} ({','.join(columns)})
            VALUES ({','.join(['?' for _ in columns])})
            ON CONFLICT(symbol, timestamp, timeframe) 
            DO UPDATE SET 
            """ + ','.join([f"{col} = excluded.{col}" for col in update_cols])
            
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                cur.executemany(insert_sql, batch)
                conn.commit()

            return True

        except Exception as e:
            self.logger.error(f"Database error: {str(e)}")
            return False
        finally:
            if conn:
                conn.close()

    def process_symbol(self, symbol: str, timeframe: str):
        if symbol == "FINNIFTY":
            symbol = "CNXFINANCE"
        print (f"Fetching : {symbol}")

        retry = 0
        while retry < 5:
            try:
                if retry > 1:
                    self.tv = TvDatafeed()
                interval = self.get_interval(timeframe)
                data = self.tv.get_hist(
                    symbol=symbol,
                    exchange='NSE',
                    interval=interval,
                    n_bars=500
                )
                if data is None or data.empty:
                    self.logger.error(f"No data received for {symbol}")
                    retry += 1
                    continue
                
                df = data.copy()
                df['symbol'] = symbol
                df = self.calculate_indicators(df)
                
                if self.save_to_db(df, timeframe):
                    self.logger.info(f"Successfully saved {self.table_name} data for {symbol}")
                else:
                    self.logger.error(f"Failed to save {self.table_name} data for {symbol}")
                break
                    
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {str(e)}")
                retry += 1
                if retry < 2:
                    time.sleep(0.2)

    @abstractmethod
    def get_update_columns(self) -> List[str]:
        pass

    def _check_new_period(self, df: pd.DataFrame) -> pd.Series:
        """Determine new periods for VWAP calculation based on anchor"""
        df['date'] = pd.to_datetime(df.index)
        
        if self.anchor == 'Session':
            return df['date'].dt.date != df['date'].dt.date.shift(1)
        elif self.anchor == 'Week':
            return df['date'].dt.isocalendar().week != df['date'].dt.isocalendar().week.shift(1)
        elif self.anchor == 'Month':
            return df['date'].dt.month != df['date'].dt.month.shift(1)
        elif self.anchor == 'Quarter':
            return df['date'].dt.quarter != df['date'].dt.quarter.shift(1)
        elif self.anchor == 'Year':
            return df['date'].dt.year != df['date'].dt.year.shift(1)
        elif self.anchor == 'Decade':
            return (df['date'].dt.year % 10 == 0) & (df['date'].dt.year != df['date'].dt.year.shift(1))
        elif self.anchor == 'Century':
            return (df['date'].dt.year % 100 == 0) & (df['date'].dt.year != df['date'].dt.year.shift(1))
        return pd.Series(False, index=df.index)


class PriceLevelsProcessor(IndicatorProcessor):
    def __init__(self, db_path: str):
        super().__init__(db_path, 'price_levels')
        self.anchor = 'Week'  # Default anchor period

    def create_table(self, cur: sqlite3.Cursor):
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS price_levels (
            symbol VARCHAR(20),
            timestamp TIMESTAMP,
            open FLOAT,
            high FLOAT,
            low FLOAT,
            close FLOAT,
            volume FLOAT,
            upper_1 FLOAT,
            upper_2 FLOAT,
            upper_3 FLOAT,
            lower_1 FLOAT,
            lower_2 FLOAT,
            lower_3 FLOAT,
            is_below_upper_1 BOOLEAN,
            is_below_upper_2 BOOLEAN,
            is_below_upper_3 BOOLEAN,
            is_above_lower_1 BOOLEAN,
            is_above_lower_2 BOOLEAN,
            is_above_lower_3 BOOLEAN,
            weekly_high FLOAT,
            weekly_low FLOAT,
            is_high_below_yesterday_high BOOLEAN,
            is_low_above_yesterday_low BOOLEAN,
            is_low_above_last_week_high BOOLEAN,
            is_high_below_last_week_low BOOLEAN,
            is_open_equal_high BOOLEAN,
            is_open_equal_low BOOLEAN,
            is_blue_line BOOLEAN,
            is_inside_bar BOOLEAN,
            last_2_inside_bar_yday_h FLOAT,
            last_2_inside_bar_yday_l FLOAT,
            last_2_inside_bar_hh FLOAT,
            last_2_inside_bar_ll FLOAT,
            last_2_inside_bar_date TIMESTAMP,
            timeframe VARCHAR(5),
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, timestamp, timeframe)
        );
        """
        cur.execute(create_table_sql)

    def get_all_columns(self) -> List[str]:
        return ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 
                'volume', 'upper_1', 'upper_2', 'upper_3', 'lower_1', 'lower_2', 'lower_3',
                'is_below_upper_1', 'is_below_upper_2', 'is_below_upper_3',
                'is_above_lower_1', 'is_above_lower_2', 'is_above_lower_3', 
                'weekly_high', 'weekly_low',
                'is_high_below_yesterday_high', 'is_low_above_yesterday_low',
                'is_low_above_last_week_high', 'is_high_below_last_week_low',
                'is_open_equal_high', 'is_open_equal_low', 'is_blue_line',
                'is_inside_bar', 'last_2_inside_bar_yday_h', 'last_2_inside_bar_yday_l',
                'last_2_inside_bar_hh', 'last_2_inside_bar_ll', 'last_2_inside_bar_date', 'timeframe']

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Sort dataframe by date ascending (ensure chronological order)
        df = df.sort_index()

        # Initialize columns for upper and lower levels
        df['upper_1'] = np.nan
        df['upper_2'] = np.nan
        df['upper_3'] = np.nan
        df['lower_1'] = np.nan
        df['lower_2'] = np.nan
        df['lower_3'] = np.nan
        
        # Initialize columns for condition flags
        df['is_below_upper_1'] = False
        df['is_below_upper_2'] = False
        df['is_below_upper_3'] = False
        df['is_above_lower_1'] = False
        df['is_above_lower_2'] = False
        df['is_above_lower_3'] = False
        
        # Initialize new columns
        df['weekly_high'] = np.nan
        df['weekly_low'] = np.nan
        df['is_high_below_yesterday_high'] = False
        df['is_low_above_yesterday_low'] = False
        df['is_low_above_last_week_high'] = False
        df['is_high_below_last_week_low'] = False
        df['is_open_equal_high'] = False
        df['is_open_equal_low'] = False
        df['is_blue_line'] = False
        
        # Initialize new inside bar columns
        df['is_inside_bar'] = False
        df['last_2_inside_bar_yday_h'] = np.nan
        df['last_2_inside_bar_yday_l'] = np.nan
        df['last_2_inside_bar_hh'] = np.nan
        df['last_2_inside_bar_ll'] = np.nan
        df['last_2_inside_bar_date'] = pd.NaT  # Initialize the new column with NaT (Not a Time)

        # Convert index to datetime for week calculations
        df['date'] = pd.to_datetime(df.index)
        
        # Add ISO calendar week and year columns
        df['year'] = df['date'].dt.isocalendar().year
        df['week'] = df['date'].dt.isocalendar().week
        
        # Create a unique week identifier (format: YYYY-WW)
        df['week_id'] = df['year'].astype(str) + '-' + df['week'].astype(str).str.zfill(2)
        
        # Find all unique weeks in chronological order
        unique_weeks = df[['week_id', 'year', 'week']].drop_duplicates().sort_values(['year', 'week']).reset_index(drop=True)
        
        # Add previous week identifier
        unique_weeks['prev_week_id'] = unique_weeks['week_id'].shift(1)
        
        # Calculate weekly highs and lows for each week
        weekly_stats = {}
        for week_id in df['week_id'].unique():
            week_data = df[df['week_id'] == week_id]
            weekly_stats[week_id] = {
                'high': week_data['high'].max(),
                'low': week_data['low'].min()
            }
        # Populate weekly high and low columns
        for week_id, stats in weekly_stats.items():
            mask = df['week_id'] == week_id
            df.loc[mask, 'weekly_high'] = stats['high']
            df.loc[mask, 'weekly_low'] = stats['low']
        
        # Populate comparison flags
        # Track first occurrence in the week for last week high/low conditions
        week_first_high_below_last_week_low = {}
        week_first_low_above_last_week_high = {}
        for i in range(len(df)):
            current_row = df.iloc[i]
            
            # Open equals high or low checks
            tolerance = 0.0001  # Tolerance for floating point equality
            df.loc[df.index[i], 'is_open_equal_high'] = abs(current_row['open'] - current_row['high']) < tolerance
            df.loc[df.index[i], 'is_open_equal_low'] = abs(current_row['open'] - current_row['low']) < tolerance
            
            # Previous day comparisons
            if i > 0:
                yesterday = df.iloc[i-1]
                df.loc[df.index[i], 'is_high_below_yesterday_high'] = current_row['high'] < yesterday['high']
                df.loc[df.index[i], 'is_low_above_yesterday_low'] = current_row['low'] > yesterday['low']
                # Set is_blue_line to True if the current row high is below the previous day's high and low is above the previous day's low
                df.loc[df.index[i], 'is_blue_line'] = current_row['high'] < yesterday['high'] and current_row['low'] > yesterday['low']
                
                # Mark as inside bar if both conditions are met
                df.loc[df.index[i], 'is_inside_bar'] = current_row['high'] <= yesterday['high'] and current_row['low'] >= yesterday['low']
            
            # Previous week comparisons
            current_week_id = current_row['week_id']
            current_week_idx = unique_weeks[unique_weeks['week_id'] == current_week_id].index
            
            if len(current_week_idx) > 0 and current_week_idx[0] > 0:
                prev_week_id = unique_weeks.loc[current_week_idx[0] - 1, 'week_id']
                
                if prev_week_id in weekly_stats:
                    prev_week_high = weekly_stats[prev_week_id]['high']
                    prev_week_low = weekly_stats[prev_week_id]['low']
                    
                    # Only set is_low_above_last_week_high for first occurrence in the week
                    if (current_week_id not in week_first_low_above_last_week_high and
                        current_row['low'] > prev_week_high):
                        df.loc[df.index[i], 'is_low_above_last_week_high'] = True
                        week_first_low_above_last_week_high[current_week_id] = True
                    # Only set is_high_below_last_week_low for first occurrence in the week
                    if (current_week_id not in week_first_high_below_last_week_low and
                        current_row['high'] < prev_week_low):
                        df.loc[df.index[i], 'is_high_below_last_week_low'] = True
                        week_first_high_below_last_week_low[current_week_id] = True
        
        # Identify successive inside bars and add requested columns
        for i in range(len(df)):
            # Need at least 6 previous bars to calculate the statistics
            if i < 6:
                continue
            
            # Check if current and previous bars are inside bars
            if df.iloc[i]['is_inside_bar'] and df.iloc[i-1]['is_inside_bar']:
                # First inside bar is at i-1, so the day before it is i-2
                df.loc[df.index[i], 'last_2_inside_bar_yday_h'] = df.iloc[i-2]['high']
                df.loc[df.index[i], 'last_2_inside_bar_yday_l'] = df.iloc[i-2]['low']
                
                # Calculate highest high and lowest low of 6 days (2 inside bars + 4 previous days)
                six_day_window = df.iloc[i-5:i+1]  # Get current and 5 previous days
                df.loc[df.index[i], 'last_2_inside_bar_hh'] = six_day_window['high'].max()
                df.loc[df.index[i], 'last_2_inside_bar_ll'] = six_day_window['low'].min()
                
                # Set the last_2_inside_bar_date to the date of the second inside bar (current row)
                df.loc[df.index[i], 'last_2_inside_bar_date'] = df.index[i]
                
                # Also mark the previous day (first inside bar) with the same values for consistency
                df.loc[df.index[i-1], 'last_2_inside_bar_yday_h'] = df.iloc[i-2]['high']
                df.loc[df.index[i-1], 'last_2_inside_bar_yday_l'] = df.iloc[i-2]['low']
                df.loc[df.index[i-1], 'last_2_inside_bar_hh'] = six_day_window['high'].max()
                df.loc[df.index[i-1], 'last_2_inside_bar_ll'] = six_day_window['low'].min()
                df.loc[df.index[i-1], 'last_2_inside_bar_date'] = df.index[i]  # Use the same date for the first inside bar
                
                # Check if there are more than 2 successive inside bars
                # If so, we'll update values for all of them
                inside_count = 2
                j = i - 2
                while j >= 0 and df.iloc[j]['is_inside_bar'] and inside_count < 10:  # Limit to prevent infinite loops
                    # Since we found more inside bars, we still use the same values
                    # (based on the day before the first inside bar and the same 6-day window)
                    df.loc[df.index[j], 'last_2_inside_bar_yday_h'] = df.iloc[i-2]['high']
                    df.loc[df.index[j], 'last_2_inside_bar_yday_l'] = df.iloc[i-2]['low']
                    df.loc[df.index[j], 'last_2_inside_bar_hh'] = six_day_window['high'].max()
                    df.loc[df.index[j], 'last_2_inside_bar_ll'] = six_day_window['low'].min()
                    df.loc[df.index[j], 'last_2_inside_bar_date'] = df.index[i]  # Use the same date for all inside bars
                    inside_count += 1
                    j -= 1
        
        # Process each row separately for price levels - original code
        for i in range(3, min(len(df), 500)):  # Start from index 3 to have enough historical data
            # Get current dataframe up to this date
            current_df = df.iloc[:i+1].copy()
            
            # Find lower lows and higher highs within this historical window
            current_df['is_lower_low'] = current_df['low'] < current_df['low'].shift(1)
            current_df['is_higher_high'] = current_df['high'] > current_df['high'].shift(1)
            
            # Get positions of lower lows and higher highs
            lower_low_indices = current_df[current_df['is_lower_low'] == True].index.tolist()
            higher_high_indices = current_df[current_df['is_higher_high'] == True].index.tolist()
            
            # Calculate upper levels based on lower lows
            # For upper_1: Find the most recent lower low
            if len(lower_low_indices) >= 1:
                latest_lower_low_date = lower_low_indices[-1]
                latest_lower_low_pos = current_df.index.get_loc(latest_lower_low_date)
                
                if latest_lower_low_pos >= 2:  # Need at least 2 previous bars
                    # Get the date index for current and previous position
                    curr_date = latest_lower_low_date
                    prev_date = current_df.index[latest_lower_low_pos-1]
                    
                    # Set upper_1 just for the current row
                    upper_1_value = max(current_df.loc[curr_date, 'high'], current_df.loc[prev_date, 'high'])
                    df.loc[df.index[i], 'upper_1'] = upper_1_value
                    
                    # Calculate lower_1: min(low on same day as upper_1, low on previous day)
                    lower_1_value = min(current_df.loc[curr_date, 'low'], current_df.loc[prev_date, 'low'])
                    df.loc[df.index[i], 'lower_1'] = lower_1_value
            
            # For upper_2: Find the second most recent lower low
            if len(lower_low_indices) >= 2:
                second_lower_low_date = lower_low_indices[-2]
                second_lower_low_pos = current_df.index.get_loc(second_lower_low_date)
                
                if second_lower_low_pos >= 2:  # Need at least 2 previous bars
                    # Get the date index for current and previous position
                    curr_date = second_lower_low_date
                    prev_date = current_df.index[second_lower_low_pos-1]
                    
                    # Set upper_2 just for the current row
                    upper_2_value = max(current_df.loc[curr_date, 'high'], current_df.loc[prev_date, 'high'])
                    df.loc[df.index[i], 'upper_2'] = upper_2_value
                    
                    # Calculate lower_2: min(low on same day as upper_2, low on previous day)
                    lower_2_value = min(current_df.loc[curr_date, 'low'], current_df.loc[prev_date, 'low'])
                    df.loc[df.index[i], 'lower_2'] = lower_2_value
                    
            # For upper_3: Find the third most recent lower low
            if len(lower_low_indices) >= 3:
                third_lower_low_date = lower_low_indices[-3]
                third_lower_low_pos = current_df.index.get_loc(third_lower_low_date)
                
                if third_lower_low_pos >= 2:  # Need at least 2 previous bars
                    # Get the date index for current and previous position
                    curr_date = third_lower_low_date
                    prev_date = current_df.index[third_lower_low_pos-1]
                    
                    # Set upper_3 just for the current row
                    upper_3_value = max(current_df.loc[curr_date, 'high'], current_df.loc[prev_date, 'high'])
                    df.loc[df.index[i], 'upper_3'] = upper_3_value
                    
                    # Calculate lower_3: min(low on same day as upper_3, low on previous day)
                    lower_3_value = min(current_df.loc[curr_date, 'low'], current_df.loc[prev_date, 'low'])
                    df.loc[df.index[i], 'lower_3'] = lower_3_value
            
            # Calculate condition flags for the current row
            current_row = df.iloc[i]
            
            # For upper levels (high < upper_X)
            if pd.notna(current_row['high']) and pd.notna(current_row['upper_1']):
                df.loc[df.index[i], 'is_below_upper_1'] = current_row['high'] < current_row['upper_1']
                
            if pd.notna(current_row['high']) and pd.notna(current_row['upper_2']):
                df.loc[df.index[i], 'is_below_upper_2'] = current_row['high'] < current_row['upper_2']
                
            if pd.notna(current_row['high']) and pd.notna(current_row['upper_3']):
                df.loc[df.index[i], 'is_below_upper_3'] = current_row['high'] < current_row['upper_3']
            
            # For lower levels (low > lower_X)
            if pd.notna(current_row['low']) and pd.notna(current_row['lower_1']):
                df.loc[df.index[i], 'is_above_lower_1'] = current_row['low'] > current_row['lower_1']
                
            if pd.notna(current_row['low']) and pd.notna(current_row['lower_2']):
                df.loc[df.index[i], 'is_above_lower_2'] = current_row['low'] > current_row['lower_2']
                
            if pd.notna(current_row['low']) and pd.notna(current_row['lower_3']):
                df.loc[df.index[i], 'is_above_lower_3'] = current_row['low'] > current_row['lower_3']
        
        # Forward fill the inside bar related columns
        df['last_2_inside_bar_yday_h'] = df['last_2_inside_bar_yday_h'].ffill()
        df['last_2_inside_bar_yday_l'] = df['last_2_inside_bar_yday_l'].ffill()
        df['last_2_inside_bar_hh'] = df['last_2_inside_bar_hh'].ffill()
        df['last_2_inside_bar_ll'] = df['last_2_inside_bar_ll'].ffill()
        df['last_2_inside_bar_date'] = df['last_2_inside_bar_date'].ffill()  # Forward fill the date column

        # Drop helper columns before returning
        df = df.drop(columns=['date', 'year', 'week', 'week_id'])
        return df
    
    def prepare_data(self, df: pd.DataFrame, timeframe: str) -> List[tuple]:
        return [
            (row.symbol, 
            row.name.strftime('%Y-%m-%d %H:%M:%S'),
            row.open, row.high, row.low, row.close, 
            row.volume, row.upper_1, row.upper_2, row.upper_3, 
            row.lower_1, row.lower_2, row.lower_3,
            1 if row.is_below_upper_1 else 0,
            1 if row.is_below_upper_2 else 0,
            1 if row.is_below_upper_3 else 0,
            1 if row.is_above_lower_1 else 0,
            1 if row.is_above_lower_2 else 0,
            1 if row.is_above_lower_3 else 0,
            row.weekly_high, row.weekly_low,
            1 if row.is_high_below_yesterday_high else 0,
            1 if row.is_low_above_yesterday_low else 0,
            1 if row.is_low_above_last_week_high else 0,
            1 if row.is_high_below_last_week_low else 0,
            1 if row.is_open_equal_high else 0,
            1 if row.is_open_equal_low else 0,
            1 if row.is_blue_line else 0,
            1 if row.is_inside_bar else 0,
            row.last_2_inside_bar_yday_h,
            row.last_2_inside_bar_yday_l,
            row.last_2_inside_bar_hh,
            row.last_2_inside_bar_ll,
            row.last_2_inside_bar_date.strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row.last_2_inside_bar_date) else None,
            timeframe)
            for idx, row in df.iterrows()
        ]

    def get_update_columns(self) -> List[str]:
        return ['upper_1', 'upper_2', 'upper_3', 'lower_1', 'lower_2', 'lower_3',
                'is_below_upper_1', 'is_below_upper_2', 'is_below_upper_3',
                'is_above_lower_1', 'is_above_lower_2', 'is_above_lower_3',
                'weekly_high', 'weekly_low',
                'is_high_below_yesterday_high', 'is_low_above_yesterday_low',
                'is_low_above_last_week_high', 'is_high_below_last_week_low',
                'is_open_equal_high', 'is_open_equal_low', 'is_blue_line',
                'is_inside_bar', 'last_2_inside_bar_yday_h', 'last_2_inside_bar_yday_l',
                'last_2_inside_bar_hh', 'last_2_inside_bar_ll', 'last_2_inside_bar_date']

def process_timeframe(symbols: List[str], timeframe: str, processors: List[IndicatorProcessor]):
    for symbol in symbols:
        for processor in processors:
            processor.process_symbol(symbol+"1!", timeframe)
            processor.process_symbol(symbol, timeframe)

def is_market_hours():
    now = datetime.now()
    
    # Market holidays for 2025
    holidays = [
        datetime(2025, 2, 26),  # Mahashivratri
        datetime(2025, 3, 14),  # Holi
        datetime(2025, 3, 31),  # Id-Ul-Fitr
        datetime(2025, 4, 10),  # Mahavir Jayanti
        datetime(2025, 4, 14),  # Ambedkar Jayanti
        datetime(2025, 4, 18),  # Good Friday
        datetime(2025, 5, 1),   # Maharashtra Day
        datetime(2025, 8, 15),  # Independence Day
        datetime(2025, 8, 27),  # Ganesh Chaturthi
        datetime(2025, 10, 2),  # Gandhi Jayanti/Dussehra
        datetime(2025, 10, 21), # Diwali Laxmi Pujan
        datetime(2025, 10, 22), # Balipratipada
        datetime(2025, 11, 5),  # Gurpurb
        datetime(2025, 12, 25), # Christmas
    ]
    
    # Check if current date is a holiday
    current_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if current_date in holidays:
        return False
    
    # Check if it's a weekday (0 is Monday, 5 is Saturday) and not the Budget day
    if now.weekday() >= 5 and now.date != datetime(2025, 2, 1):
        return False
    
    # Check market hours
    current_time = now.time()
    market_open = datetime.strptime('09:00', '%H:%M').time()
    market_close = datetime.strptime('15:30', '%H:%M').time()
    
    return market_open <= current_time <= market_close

def main():
    db_path = "market_data_monthly.db"
    
    # Initialize processor
    price_levels_processor = PriceLevelsProcessor(db_path)
    
    processors = [price_levels_processor]
    
    current_time = datetime.now().time()
    for processor in processors:
        processor.clean_table()

    timeframes = ['1M']
    for timeframe in timeframes:
        process_timeframe(symbols, timeframe, processors)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping the process...")
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(60)  # Wait for 1 minute before retry after error

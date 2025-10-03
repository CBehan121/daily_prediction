"""
predict_today_trades.py
Script to fetch today's insider trades, process them, and predict which ones are likely to result in price increases.
"""
import traceback
import linecache
import re
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from io import StringIO
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import plot_importance

class TodayInsiderPredictor:
    """
    A service to fetch today's insider trades and predict which ones are likely to result in price increases.
    """
    
    def __init__(self):
        self.base_url = "http://openinsider.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load the trained model
        self.model, self.threshold = self.load_model()
        self.fundamental_data = None
        self.prices_by_ticker = {}
        
    def load_model(self, model_path='insider_trading_model.joblib'):
        """Load the trained model and threshold."""
        try:
            model_data = joblib.load(model_path)
            self.logger.info(f"Model loaded from {model_path} with threshold {model_data['threshold']}")
            return model_data['model'], model_data['threshold']
        except FileNotFoundError:
            self.logger.error(f"Model file {model_path} not found. Please train a model first.")
            raise
    
    def fetch_today_trades_fallback(self):
        """Fallback method to get recent trades and filter manually."""
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Try a simpler URL that gets recent trades
        url = f"http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd=&fdr=&is=1&td=0&fdlyl=&fdlyh=&daysago=7&xp=1"
        
        self.logger.info(f"Fallback: Fetching recent trades from: {url}")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse the HTML table
        tables = pd.read_html(StringIO(response.text))
        df = tables[11]  # 11th table contains the data
        df.columns = df.columns.str.replace('\xa0', ' ').str.strip()
        
        if df.empty:
            self.logger.info("No trades found in fallback method.")
            return pd.DataFrame()
        
        # Filter for today's filings
        if 'Filing Date' in df.columns:
            df['Filing Date'] = pd.to_datetime(df['Filing Date'])
            today_dt = pd.to_datetime(today)
            df_filtered = df[df['Filing Date'].dt.date == today_dt.date()]
            
            self.logger.info(f"Fallback: Found {len(df_filtered)} trades filed on {today} out of {len(df)} total")
            return df_filtered
        else:
            self.logger.warning("No 'Filing Date' column found in fallback data")
            return df

    def fetch_today_trades(self):
        """Fetch today's insider trades from OpenInsider."""
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        # URL for insider buys filed today (more specific filtering)
        url = f"http://openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=&fd={yesterday}&fdr={today}&is=1&td=0&fdlyl=&fdlyh=&daysago=&xp=1"
        self.logger.info(f"Fetching trades filed from {yesterday} to {today} from: {url}")
        response = self.session.get(url, timeout=30)
        response.raise_for_status()
        # Parse the HTML table
        tables = pd.read_html(StringIO(response.text))
        df = tables[11]  # 11th table contains the data
        df.columns = df.columns.str.replace('\xa0', ' ').str.strip()
       
        if df.empty:
            self.logger.info(f"No trades filed on {today} or {yesterday} found.")
            return pd.DataFrame()
        
        # Debug: Show what we got
        self.logger.info(f"Raw data from OpenInsider: {len(df)} rows")
        if 'Filing Date' in df.columns:
            self.logger.info(f"Filing Date range: {df['Filing Date'].min()} to {df['Filing Date'].max()}")
            self.logger.info(f"Sample filing dates: {df['Filing Date'].head().tolist()}")
        
        # Additional filtering to ensure we only get trades filed today
        
        if 'Filing Date' in df.columns:
            df['Filing Date'] = pd.to_datetime(df['Filing Date'])
            today_dt = pd.to_datetime(today)
            yesterday_dt = pd.to_datetime(yesterday)
            # Filter for exact date match
            df_filtered = df[(df['Filing Date'].dt.date == today_dt.date()) | (df['Filing Date'].dt.date == yesterday_dt.date())]
            
            self.logger.info(f"After filtering for {today} / {yesterday}: {len(df_filtered)} trades")
            
            if len(df_filtered) != len(df):
                self.logger.info(f"Filtered out {len(df) - len(df_filtered)} trades that weren't filed on {today} / {yesterday}")
            
            # If we got too many results, try fallback method
#            if len(df_filtered) > 20:  # Suspicious if more than 20 trades filed today
#                self.logger.warning(f"Got {len(df_filtered)} trades - this seems high. Trying fallback method...")
#                return self.fetch_today_trades_fallback()
            
            return df_filtered
        else:
            self.logger.warning("No 'Filing Date' column found in data")
            return df
    
    def load_fundamental_data(self):
        """Load fundamental data for all tickers."""
        try:
            self.fundamental_data = pd.read_csv('up-to-date/stock_fundamentals.csv')
            self.logger.info(f"Loaded fundamental data for {len(self.fundamental_data)} tickers")
        except FileNotFoundError:
            self.logger.warning("Fundamental data not found. Will skip fundamental features.")
            self.fundamental_data = pd.DataFrame()
    
    def get_historical_prices(self, tickers):
        """Get historical prices for the given tickers."""
        self.prices_by_ticker = {}
        valid_tickers = []
        
        for ticker in tickers:
            if pd.isna(ticker) or not isinstance(ticker, str) or ticker.strip() == "":
                continue
                
            try:
                # Get 30 days of data
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
                if (ticker == 'SI') or (ticker == 'FIG'):
                    print(ticker)
                    print(df)
                    print(start_date)
                    print(end_date)
                if (not df.empty) and (len(df) > 10):
                    # Reset index to get Date as a column
                    df.reset_index(inplace=True)
                    
                                        # Ensure we have the right columns
                    required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                    if all(col in df.columns for col in required_cols):
                        # Set Date as index
                        df.set_index('Date', inplace=True)
                        self.prices_by_ticker[ticker] = df
                        valid_tickers.append(ticker)

                    else:
                        self.logger.warning(f"Missing required columns for {ticker}")
                else:
                    self.logger.warning(f"No data found for {ticker}")
            except Exception as e:
                self.logger.warning(f"Error downloading data for {ticker}: {e}")
        
        if self.prices_by_ticker:
            self.logger.info(f"Downloaded historical prices for {len(valid_tickers)} tickers")
        else:
            self.logger.warning("No historical price data could be downloaded")
    
    def resolve_filing_date(self, row):
        """Resolve the actual filing date based on available price data."""
        ticker = row['Ticker']
        date = row['Filing Date']
        
        if ticker not in self.prices_by_ticker:
            return pd.NaT
            
        # Convert date to datetime if it's a string
        if isinstance(date, str):
            date = pd.to_datetime(date)
            
        price_df = self.prices_by_ticker[ticker].sort_index()
        try:
            return price_df.index.asof(date)
        except Exception as e:
            self.logger.error(f"Error in resolve_filing_date for {ticker}: {e}")
            self.logger.error(f"date type: {type(date)}, value: {date}")
            return pd.NaT
    
    def extract_technical_features(self, row):
        """Extract technical and fundamental features for a given trade."""
        ticker = row['Ticker']
        filing_date = row['resolved_filing_date']
        # Ensure filing_date is a datetime
        if isinstance(filing_date, str):
            filing_date = pd.to_datetime(filing_date)
        elif pd.isna(filing_date):
            return pd.Series({
                'Return': np.nan, 'MA5': np.nan, 'MA10': np.nan, 'STD5': np.nan, 'Volume_Change': np.nan,
                'MarketCap': np.nan, 'Beta': np.nan, 'EpsCurrentYear': np.nan, 'FreeCashflow': np.nan,
                'HeldPercentInsiders': np.nan, 'HeldPercentInstitutions': np.nan, 'AverageVolume': np.nan,
                'Volume': np.nan, 'EarningsTimestampStart': np.nan, 'TimeToNextEarnings': np.nan
            })
        
        # Get fundamental data for this ticker
        if not self.fundamental_data.empty:
            fundamental_data_ticker = self.fundamental_data[self.fundamental_data['Ticker'] == ticker].copy()
        else:
            fundamental_data_ticker = pd.DataFrame()
        
        # Initialize empty return
        empty_return = {
            'Return': np.nan, 'MA5': np.nan, 'MA10': np.nan, 'STD5': np.nan, 'Volume_Change': np.nan,
            'MarketCap': np.nan, 'Beta': np.nan, 'EpsCurrentYear': np.nan, 'FreeCashflow': np.nan,
            'HeldPercentInsiders': np.nan, 'HeldPercentInstitutions': np.nan, 'AverageVolume': np.nan,
            'Volume': np.nan, 'EarningsTimestampStart': np.nan, 'TimeToNextEarnings': np.nan
        }
        
        if fundamental_data_ticker.empty:
            time_to_next_earnings = np.nan
        else:
            # Calculate time to next earnings
            earnings_date = fundamental_data_ticker['EarningsTimestampStart'].iloc[0]
            if pd.notna(earnings_date):
                try:
                    earnings_date_dt = pd.to_datetime(earnings_date)
                    time_to_next_earnings = (filing_date - earnings_date_dt).days
                except:
                    time_to_next_earnings = np.nan
            else:
                time_to_next_earnings = np.nan
        
        empty_return['TimeToNextEarnings'] = time_to_next_earnings
        
        if ticker not in self.prices_by_ticker:
            return pd.Series(empty_return)
        
        try:
            df = self.prices_by_ticker[ticker].copy().sort_index()
            
            # Handle MultiIndex if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Define window for feature calculation
            window_before = 15
            window_after = 0
            
            start_date = df.index.asof(filing_date - pd.Timedelta(days=window_before))
            end_date = df.index.asof(filing_date + pd.Timedelta(days=window_after))
            # Restrict to this window
            df = df.loc[start_date:end_date]
            
            # Compute technical features
            df = df.copy()  # Ensure we have a copy

            # Check if 'Close' column exists and is a Series
            if 'Close' not in df.columns:
                self.logger.error(f"'Close' column not found in {ticker}. Available columns: {df.columns}")
                return pd.Series(empty_return)
            
            close_series = df['Close']
            if not isinstance(close_series, pd.Series):
                self.logger.error(f"'Close' is not a Series for {ticker}. Type: {type(close_series)}")
                return pd.Series(empty_return)
            
            df['Return'] = close_series.pct_change(fill_method=None)
            df['MA5'] = df['Close'].rolling(5, min_periods=1).mean()
            df['MA10'] = df['Close'].rolling(10, min_periods=1).mean()
            df['STD5'] = df['Close'].rolling(5, min_periods=1).std()
            df['Volume_Change'] = df['Volume'].pct_change(fill_method=None)
            df['Volume_Change'] = df['Volume_Change'].replace([np.inf, -np.inf], 1000)
            
            if filing_date not in df.index:
                filing_date = df.index.asof(filing_date)

            features = df.loc[filing_date, ['Return', 'MA5', 'MA10', 'STD5', 'Volume_Change']]
            
            # Combine technical and fundamental features
            result = {
                'Return': features['Return'],
                'MA5': features['MA5'],
                'MA10': features['MA10'],
                'STD5': features['STD5'],
                'Volume_Change': features['Volume_Change'],
                'MarketCap': fundamental_data_ticker['MarketCap'].iloc[0] if not fundamental_data_ticker.empty else np.nan,
                'Beta': fundamental_data_ticker['Beta'].iloc[0] if not fundamental_data_ticker.empty else np.nan,
                'EpsCurrentYear': fundamental_data_ticker['EpsCurrentYear'].iloc[0] if not fundamental_data_ticker.empty else np.nan,
                'FreeCashflow': fundamental_data_ticker['FreeCashflow'].iloc[0] if not fundamental_data_ticker.empty else np.nan,
                'HeldPercentInsiders': fundamental_data_ticker['HeldPercentInsiders'].iloc[0] if not fundamental_data_ticker.empty else np.nan,
                'HeldPercentInstitutions': fundamental_data_ticker['HeldPercentInstitutions'].iloc[0] if not fundamental_data_ticker.empty else np.nan,
                'AverageVolume': fundamental_data_ticker['AverageVolume'].iloc[0] if not fundamental_data_ticker.empty else np.nan,
                'Volume': fundamental_data_ticker['Volume'].iloc[0] if not fundamental_data_ticker.empty else np.nan,
                'EarningsTimestampStart': pd.to_datetime(fundamental_data_ticker['EarningsTimestampStart'].iloc[0]).value if not fundamental_data_ticker.empty and pd.notna(fundamental_data_ticker['EarningsTimestampStart'].iloc[0]) else np.nan,
                'TimeToNextEarnings': time_to_next_earnings
            }
            return pd.Series(result)
            
        except Exception as e:
#            self.logger.error("------------------------ TOP ------------------------")
            self.logger.error(f"Error extracting features for {df}")
            self.logger.error(f"Error extracting features for {ticker}: {e}")
#            self.logger.error("Full traceback:")
#            self.logger.error(traceback.format_exc())
#            self.logger.error(f"filing_date type: {type(filing_date)}, value: {filing_date}")
#            self.logger.error("------------------------ Bottom ------------------------")
            return pd.Series(empty_return)
    
    def separate_titles(self, df):
        """Separate and encode titles that are separated by / or ,."""
        df['Title_List'] = df['Title'].str.split(r'[\\,]\s*')
        titles_exploded = df.explode('Title_List')
        title_dummies = pd.get_dummies(titles_exploded['Title_List'], prefix='Role')
        title_encoded = title_dummies.groupby(titles_exploded.index).max()
        return pd.concat([df, title_encoded], axis=1)
    
    def calculate_time_between_filing_and_trading(self, df):
        """Calculate time between filing and trading dates."""
        df['Filing Date'] = pd.to_datetime(df['Filing Date'])
        df['Trade Date'] = pd.to_datetime(df['Trade Date'])
        df['time_between'] = (df['Filing Date'] - df['Trade Date']).dt.days
        return df

    def add_current_price(self, row):
        ticker = row['Ticker']
        if ticker in self.prices_by_ticker:
            # Get the most recent price as a scalar
            current_price = self.prices_by_ticker[ticker]['Close'].iloc[-1].item()
            return current_price
        else:
            return None

    def preprocess_data(self, df):
        """Preprocess the data for model prediction."""
        df = df.copy()
        
        # Convert numeric columns
        df['Price'] = df['Price'].replace(r'[\$,]', '', regex=True).astype(float)
        df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce')
        
        # Handle ownership changes
        df['is_new'] = df['ΔOwn'].eq('New').astype(int)
        df['percentage_increase'] = df['ΔOwn'].where(df['ΔOwn'] != 'New')
        df['percentage_increase'] = df['percentage_increase'].str.replace('[^0-9.-]', '', regex=True).astype(float) / 100
        df['percentage_increase'] = df['percentage_increase'].fillna(0)
        
        # Log transform owned shares
        df['Owned'] = np.log1p(df['Owned'])
        
        # Add time features
        df = self.calculate_time_between_filing_and_trading(df)
        df = self.separate_titles(df)
        
        # Define feature columns (same as training)
        feature_cols = [
            'Price', 'Qty', 'Return', 'MA5', 'MA10', 'STD5', 'Volume_Change', 'Owned',
            'Role_10%', 'Role_CEO', 'Role_CFO', 'Role_CHIEF BANKING OFFICER', 'Role_COB', 'Role_COO', 'Role_CORP SECRET', 'Role_CRBT',
            'Role_Dir', 'Role_EVP', 'Role_Exec COB', 'Role_Former 10% Owner', 'Role_GC', 'Role_Pres', 'Role_QCRH', 'Role_SVP', 'time_between',
            'MarketCap', 'Beta', 'EpsCurrentYear', 'FreeCashflow', 'HeldPercentInsiders', 'HeldPercentInstitutions', 'AverageVolume', 'Volume', 'EarningsTimestampStart', 'TimeToNextEarnings',
            'total_trades', 'total_value', 'avg_trade_size', 'companies_traded', 'winning-trades'
        ]
        
        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Select features
        X = df[feature_cols]
        
        return X, df

    def return_date_range(self, date, beginning, day_size):
            if beginning:
                start_date = (date - pd.Timedelta(days=day_size))
                start_date_str = start_date.strftime('%Y-%m-%d')
                saturday_start_date = (start_date + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
                sunday_start_date = (start_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                return [start_date_str, saturday_start_date, sunday_start_date]
            else:
                full_date = pd.to_datetime(date) + pd.Timedelta(days=day_size)
                date = (full_date).strftime('%Y-%m-%d')
                friday_end_date = (full_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                three_days_before = (full_date - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
                thursday_end_date = (full_date - pd.Timedelta(days=2)).strftime('%Y-%m-%d')
                saturday_end_date = (full_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                sunday_end_date = (full_date + pd.Timedelta(days=2)).strftime('%Y-%m-%d')
                monday_end_date = (full_date + pd.Timedelta(days=3)).strftime('%Y-%m-%d')
                return ([date,thursday_end_date, friday_end_date, saturday_end_date, sunday_end_date, monday_end_date, three_days_before ])

    def count_winning_trades(self, insider_trades, historical_prices):
        result = 0
        for _, entry in insider_trades[['Ticker' , 'Filing Date', 'Price']].iterrows():
            price = float(re.sub(r'[\$,]', '', entry['Price']))
            ticker = entry['Ticker']
            possible_end_dates = self.return_date_range(entry['Filing Date'], False, 5 )
            historical_prices_filtered = (historical_prices.loc[(historical_prices['Date'].isin(possible_end_dates)) &
                                       (historical_prices["Ticker"] == ticker)]).sort_values('High', ascending=True)
            if not historical_prices_filtered.empty:
                if historical_prices_filtered.iloc[-1]['High'] > price:
                    result = result + 1
                else:
                    result = result - 1
                    
        return result


    def get_insider_history(self, insider_name, insiders_trades):
        """Get all trades by a specific insider from your existing data"""
        insider_trades = insiders_trades[insiders_trades['Insider Name'].str.contains(insider_name, case=False, na=False)]
        
        return insider_trades

    def get_insider_stats(self, row): 
        """Calculate stats for a specific insider"""
        insider_info_df = pd.DataFrame({
            'total_trades': pd.Series(dtype='float'),
            'total_value': pd.Series(dtype='float'), 
            'avg_trade_size': pd.Series(dtype='float'), 
            'companies_traded': pd.Series(dtype='float'), 
            'winning-trades': pd.Series(dtype='float')
        })
        insiders_trades = pd.read_csv("up-to-date/insider_trades_3.csv")
        insiders_trades.columns = insiders_trades.columns.str.replace('\xa0', ' ').str.strip()
        insiders_trades = pd.concat([insiders_trades, row.to_frame().T], ignore_index=True)
        
        
        historical_prices = pd.read_csv("up-to-date/historical_prices_3.csv")
        insider_name = row['Insider Name']
        insider_trades = self.get_insider_history(insider_name, insiders_trades)
        result = {
            'total_trades': len(insider_trades),
            'total_value': insider_trades['Value'].str.replace(r'[\$,+]', '', regex=True).astype(float).sum(),
            'avg_trade_size': insider_trades['Value'].str.replace(r'[\$,+]', '', regex=True).astype(float).mean(),
            'companies_traded': insider_trades['Ticker'].nunique(),
            'winning-trades' : self.count_winning_trades(insider_trades, historical_prices ) }
        return pd.Series(result)  

    def predict_trades(self):
        """Main function to fetch today's trades and predict which ones are likely to result in price increases."""
        self.logger.info("Starting prediction for today's insider trades...")
        
        # Fetch today's trades
        today_trades = self.fetch_today_trades()
        if today_trades.empty:
            self.logger.info("No trades found for today.")
            return pd.DataFrame()
        
        # Clean column names
        today_trades.columns = today_trades.columns.str.replace('\u00A0', ' ').str.strip()
        
        # Load fundamental data
        self.load_fundamental_data()
        
        # Get historical prices for all tickers
        tickers = today_trades['Ticker'].unique()
        self.get_historical_prices(tickers)
        
        # Resolve filing dates
        today_trades['resolved_filing_date'] = today_trades.apply(self.resolve_filing_date, axis=1)
        
        # Extract technical features
        self.logger.info("Extracting technical features...")
        tech_features = today_trades.apply(self.extract_technical_features, axis=1)
        
        tech_features.columns = [
            'Return', 'MA5', 'MA10', 'STD5', 'Volume_Change', 'MarketCap', 'Beta', 'EpsCurrentYear',
            'FreeCashflow', 'HeldPercentInsiders', 'HeldPercentInstitutions', 'AverageVolume', 'Volume', 'EarningsTimestampStart', 'TimeToNextEarnings'
        ]
        trader_history = today_trades.apply(self.get_insider_stats, axis=1)
        trader_history.columns = ['total_trades', 'total_value', 'avg_trade_size', 'companies_traded', 'winning-trades'] 
        print(len(tech_features))
        print(len(trader_history))
        print(len(today_trades))
        # Combine features
        today_trades = pd.concat([today_trades.reset_index(drop=True), tech_features, trader_history], axis=1)
        
        #today_trades = pd.concat([today_trades.reset_index(drop=True), trader_history], axis=1)
        # Remove unnecessary columns if present
        for col in ['X', '1d', '1w', '1m', '6m']:
            if col in today_trades.columns:
                today_trades.drop(columns=col, inplace=True)

        # Preprocess for prediction
        X, processed_df = self.preprocess_data(today_trades)
        
        # Make predictions
        self.logger.info("Making predictions...")
        predictions, probabilities = self.predict_with_threshold(X)
        # Add predictions to the dataframe
        processed_df['predicted_class'] = predictions
        processed_df['prediction_probability'] = probabilities
        
        # Filter for trades predicted as class 1 (likely price increase)
        positive_predictions = processed_df[processed_df['predicted_class'] == 1].copy()
        
        if not positive_predictions.empty:
            self.logger.info(f"Found {len(positive_predictions)} trades predicted to result in price increases")
            
            # Sort by prediction probability (highest first)
            positive_predictions = positive_predictions.sort_values('prediction_probability', ascending=False)
            
            # Select relevant columns for output
            output_columns = [
                'Ticker', 'Company', 'Insider Name', 'Title', 'Trade Type', 'Price', 'Qty', 'Value',
                'prediction_probability', 'Filing Date', 'Trade Date', 'Current_Price'
            ]
            
            # Only include columns that exist
            positive_predictions['Current_Price'] = positive_predictions.apply(self.add_current_price, axis=1)
            available_columns = [col for col in output_columns if col in positive_predictions.columns]
            result = positive_predictions[available_columns]
            
            return result
        else:
            self.logger.info("No trades predicted to result in price increases")
            return pd.DataFrame()
    
    def check_importance (self, probs, predictions): 
        # Get feature importance (works for RandomForest, XGBoost, etc.)
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            print(importance)
            
            feature_names = ['Price', 'Qty', 'Return', 'MA5', 'MA10', 'STD5', 'Volume_Change',
       'Owned', 'Role_10%', 'Role_CEO', 'Role_CFO',
       'Role_CHIEF BANKING OFFICER', 'Role_COB', 'Role_COO',
       'Role_CORP SECRET', 'Role_CRBT', 'Role_Dir', 'Role_EVP',
       'Role_Exec COB', 'Role_Former 10% Owner', 'Role_GC', 'Role_Pres',
       'Role_QCRH', 'Role_SVP', 'time_between', 'MarketCap', 'Beta',
       'EpsCurrentYear', 'FreeCashflow', 'HeldPercentInsiders',
       'HeldPercentInstitutions', 'AverageVolume', 'Volume',
       'EarningsTimestampStart', 'TimeToNextEarnings',
       'total_trades', 'total_value', 'avg_trade_size', 'companies_traded', 'winning-trades']
            plot_importance(self.model, importance_type='gain')
            # Create importance DataFrame and sort
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            print("Top 10 Most Important Features:")
            print(importance_df)
            
            return predictions, probs, importance_df
        else:
            print("Model doesn't have built-in feature importance")
            return predictions, probs, None

    def predict_with_threshold(self, X):
        """Make predictions using the custom threshold."""
        probs = self.model.predict_proba(X)[:, 1]
        predictions = (probs >= self.threshold).astype(int)
        self.check_importance(probs, predictions)
        return predictions, probs

def main():
    """Main function to run the prediction pipeline."""
    try:
        predictor = TodayInsiderPredictor()
        results = predictor.predict_trades()
        
        if not results.empty:
            print("\n" + "="*80)
            print("TRADES PREDICTED TO RESULT IN PRICE INCREASES")
            print("="*80)
            print(results.to_string(index=False))
            
            # Save results to CSV
            output_file = f"predicted_trades_{datetime.now().strftime('%Y%m%d')}.csv"
            results.to_csv(output_file, index=False)
            print(f"\nResults saved to: {output_file}")
        else:
            print("\nNo trades predicted to result in price increases today.")
            
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main() 
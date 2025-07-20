# core/data_loader.py - Historical Data Loader for M5 XAUUSD

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle

class HistoricalDataLoader:
    """
    Historical Data Loader for AI Training
    - Downloads M5 XAUUSD data (2 years)
    - Calculates technical indicators
    - Saves/loads processed data
    - Optimized for recovery trading strategy
    """
    
    def __init__(self, mt5_interface=None):
        print("📥 Initializing Historical Data Loader...")
        
        self.mt5_interface = mt5_interface
        self.symbol = "XAUUSD.v"  # ✅ Updated to correct symbol
        self.timeframe = mt5.TIMEFRAME_M5
        self.lookback_years = 2
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.indicators_data = None
        
        # File paths
        self.data_dir = os.path.abspath('data')
        self.raw_data_file = os.path.join(self.data_dir, 'xauusd_v_m5_raw.pkl')  # ✅ Updated filename
        self.processed_data_file = os.path.join(self.data_dir, 'xauusd_v_m5_processed.pkl')  # ✅ Updated filename
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        print("✅ Data Loader initialized")
        print(f"   - Symbol: {self.symbol}")
        print(f"   - Timeframe: M5") 
        print(f"   - Lookback: {self.lookback_years} years")
        print(f"   - Data Dir: {self.data_dir}")
        print(f"   - Using XAUUSD.v (Correct Symbol)")  # ✅ Confirmation message

    def download_historical_data(self, force_download=False):
        """Download M5 XAUUSD data for last 2 years with smart symbol detection"""
        try:
            # Check if data already exists
            if os.path.exists(self.raw_data_file) and not force_download:
                print("📂 Raw data file exists, loading from cache...")
                return self.load_raw_data()
            
            print("🌐 Downloading historical data from MT5...")
            
            # Check MT5 connection
            if not self._check_mt5_connection():
                print("❌ MT5 not connected")
                return False
            
            # ✅ Smart symbol detection
            correct_symbol = self._find_correct_symbol()
            if not correct_symbol:
                print("❌ Cannot find valid Gold symbol")
                return False
            
            print(f"✅ Using symbol: {correct_symbol}")
            
            # ✅ Adjusted date range (avoid future dates)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=min(730, 365))  # Max 2 years or 1 year
            
            print(f"📅 Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # ✅ Try different timeframes if M5 fails
            rates = self._download_with_fallback(correct_symbol, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                print("❌ No data received from MT5")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Basic data info
            print(f"✅ Downloaded {len(df)} candles")
            print(f"   - Period: {df.index[0]} to {df.index[-1]}")
            print(f"   - Data size: {len(df):,} rows")
            print(f"   - Approximate trading days: {len(df) / 288:.0f}")
            
            # Save raw data
            self.raw_data = df
            self.save_raw_data()
            
            return True
            
        except Exception as e:
            print(f"❌ Download error: {e}")
            return False

    def _find_correct_symbol(self):
        """Find correct Gold symbol in MT5"""
        try:
            # Common Gold symbol variations
            symbol_candidates = [
                "XAUUSD",
                "XAUUSD.raw", 
                "XAUUSD.m",
                "XAUUSD.c",
                "GOLD",
                "GOLD.m",
                "XAU/USD",
                "Gold"
            ]
            
            print("🔍 Searching for Gold symbol...")
            
            for symbol in symbol_candidates:
                try:
                    # Test if symbol exists and has data
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is not None:
                        print(f"✅ Found working symbol: {symbol}")
                        return symbol
                except:
                    continue
            
            # If no predefined symbol works, search through all symbols
            print("🔍 Searching through all symbols...")
            symbols = mt5.symbols_get()
            if symbols:
                for symbol in symbols:
                    symbol_name = symbol.name.upper()
                    if 'XAU' in symbol_name or 'GOLD' in symbol_name:
                        try:
                            tick = mt5.symbol_info_tick(symbol.name)
                            if tick is not None:
                                print(f"✅ Found working symbol: {symbol.name}")
                                return symbol.name
                        except:
                            continue
            
            return None
            
        except Exception as e:
            print(f"❌ Symbol search error: {e}")
            return None

    def _download_with_fallback(self, symbol, start_date, end_date):
        """Download data with timeframe fallback"""
        try:
            # Try M5 first
            print("⬇️ Trying M5 data...")
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
            
            if rates is not None and len(rates) > 1000:  # Need reasonable amount of data
                print(f"✅ M5 data successful: {len(rates)} candles")
                return rates
            
            # Fallback to M15
            print("⬇️ M5 failed, trying M15 data...")
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M15, start_date, end_date)
            
            if rates is not None and len(rates) > 500:
                print(f"✅ M15 data successful: {len(rates)} candles")
                return rates
            
            # Fallback to H1
            print("⬇️ M15 failed, trying H1 data...")
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start_date, end_date)
            
            if rates is not None and len(rates) > 100:
                print(f"✅ H1 data successful: {len(rates)} candles")
                return rates
            
            # Try shorter period if still failing
            print("⬇️ Trying shorter period (3 months)...")
            shorter_start = end_date - timedelta(days=90)
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, shorter_start, end_date)
            
            if rates is not None and len(rates) > 100:
                print(f"✅ Short period M5 successful: {len(rates)} candles")
                return rates
            
            return None
            
        except Exception as e:
            print(f"❌ Download with fallback error: {e}")
            return None

    def calculate_indicators(self):
        """Calculate technical indicators for the data"""
        try:
            if self.raw_data is None:
                print("❌ No raw data available")
                return False
            
            print("📊 Calculating technical indicators...")
            
            df = self.raw_data.copy()
            
            # === TREND INDICATORS ===
            print("   - Calculating SMA 20, 50...")
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            
            print("   - Calculating EMA 12, 26...")
            df['EMA_12'] = df['close'].ewm(span=12).mean()
            df['EMA_26'] = df['close'].ewm(span=26).mean()
            
            # === MOMENTUM INDICATORS ===
            print("   - Calculating RSI 14...")
            df['RSI_14'] = self._calculate_rsi(df['close'], 14)
            
            print("   - Calculating MACD...")
            macd_line = df['EMA_12'] - df['EMA_26']
            signal_line = macd_line.ewm(span=9).mean()
            df['MACD'] = macd_line
            df['MACD_Signal'] = signal_line
            df['MACD_Histogram'] = macd_line - signal_line
            
            # === VOLATILITY INDICATORS ===
            print("   - Calculating Bollinger Bands...")
            bb_period = 20
            bb_std = 2
            bb_middle = df['close'].rolling(window=bb_period).mean()
            bb_std_dev = df['close'].rolling(window=bb_period).std()
            df['BB_Upper'] = bb_middle + (bb_std_dev * bb_std)
            df['BB_Lower'] = bb_middle - (bb_std_dev * bb_std)
            df['BB_Middle'] = bb_middle
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            print("   - Calculating ATR 14...")
            df['ATR_14'] = self._calculate_atr(df, 14)
            
            # === ADDITIONAL FEATURES FOR RECOVERY TRADING ===
            print("   - Calculating additional features...")
            
            # Price position relative to Bollinger Bands
            df['BB_Position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Trend strength
            df['Trend_Strength'] = (df['close'] - df['SMA_50']) / df['ATR_14']
            
            # Volatility regime
            df['Volatility_Regime'] = df['ATR_14'] / df['ATR_14'].rolling(window=50).mean()
            
            # RSI momentum
            df['RSI_Momentum'] = df['RSI_14'].diff()
            
            # MACD momentum  
            df['MACD_Momentum'] = df['MACD_Histogram'].diff()
            
            # Remove NaN values (first 50 rows due to indicators)
            df = df.dropna()
            
            print(f"✅ Indicators calculated successfully")
            print(f"   - Final dataset: {len(df):,} rows")
            print(f"   - Features: {len(df.columns)} columns")
            print(f"   - Clean data from: {df.index[0]} to {df.index[-1]}")
            
            self.processed_data = df
            self.save_processed_data()
            
            return True
            
        except Exception as e:
            print(f"❌ Indicator calculation error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=prices.index, dtype=float)

    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr
        except:
            return pd.Series(index=df.index, dtype=float)

    def save_raw_data(self):
        """Save raw data to file"""
        try:
            with open(self.raw_data_file, 'wb') as f:
                pickle.dump(self.raw_data, f)
            print(f"💾 Raw data saved: {self.raw_data_file}")
        except Exception as e:
            print(f"❌ Save raw data error: {e}")

    def load_raw_data(self):
        """Load raw data from file"""
        try:
            if os.path.exists(self.raw_data_file):
                with open(self.raw_data_file, 'rb') as f:
                    self.raw_data = pickle.load(f)
                print(f"📂 Raw data loaded: {len(self.raw_data):,} rows")
                return True
            return False
        except Exception as e:
            print(f"❌ Load raw data error: {e}")
            return False

    def save_processed_data(self):
        """Save processed data to file"""
        try:
            with open(self.processed_data_file, 'wb') as f:
                pickle.dump(self.processed_data, f)
            print(f"💾 Processed data saved: {self.processed_data_file}")
        except Exception as e:
            print(f"❌ Save processed data error: {e}")

    def load_processed_data(self):
        """Load processed data from file"""
        try:
            if os.path.exists(self.processed_data_file):
                with open(self.processed_data_file, 'rb') as f:
                    self.processed_data = pickle.load(f)
                print(f"📂 Processed data loaded: {len(self.processed_data):,} rows")
                return True
            return False
        except Exception as e:
            print(f"❌ Load processed data error: {e}")
            return False

    def _check_mt5_connection(self):
        """Check if MT5 is connected"""
        try:
            if self.mt5_interface and self.mt5_interface.is_connected:
                return True
            
            # Try direct MT5 connection
            account_info = mt5.account_info()
            return account_info is not None
        except:
            return False

    def get_data_summary(self):
        """Get summary of available data"""
        try:
            summary = {
                'raw_data_available': self.raw_data is not None,
                'processed_data_available': self.processed_data is not None,
                'raw_data_file_exists': os.path.exists(self.raw_data_file),
                'processed_data_file_exists': os.path.exists(self.processed_data_file)
            }
            
            if self.processed_data is not None:
                summary.update({
                    'total_rows': len(self.processed_data),
                    'date_range': f"{self.processed_data.index[0]} to {self.processed_data.index[-1]}",
                    'columns': list(self.processed_data.columns),
                    'trading_days': len(self.processed_data) / 288,
                    'data_quality': self._check_data_quality()
                })
            
            return summary
            
        except Exception as e:
            print(f"❌ Get summary error: {e}")
            return {}

    def _check_data_quality(self):
        """Check data quality"""
        try:
            if self.processed_data is None:
                return "No data"
            
            df = self.processed_data
            
            # Check for missing values
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            
            # Check for anomalies
            price_changes = df['close'].pct_change()
            extreme_moves = (abs(price_changes) > 0.05).sum()  # >5% moves
            
            if missing_pct < 1 and extreme_moves < 10:
                return "Excellent"
            elif missing_pct < 5 and extreme_moves < 50:
                return "Good" 
            else:
                return "Fair"
                
        except:
            return "Unknown"

    def get_training_data(self, start_date=None, end_date=None):
        """Get data ready for AI training"""
        try:
            if self.processed_data is None:
                print("❌ No processed data available")
                return None
            
            df = self.processed_data.copy()
            
            # Filter by date range if specified
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            # Select features for training
            feature_columns = [
                'open', 'high', 'low', 'close',
                'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
                'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width', 'BB_Position',
                'ATR_14', 'Trend_Strength', 'Volatility_Regime',
                'RSI_Momentum', 'MACD_Momentum'
            ]
            
            # Make sure all columns exist
            available_columns = [col for col in feature_columns if col in df.columns]
            
            training_data = df[available_columns].copy()
            
            print(f"✅ Training data ready:")
            print(f"   - Rows: {len(training_data):,}")
            print(f"   - Features: {len(available_columns)}")
            print(f"   - Date range: {training_data.index[0]} to {training_data.index[-1]}")
            
            return training_data
            
        except Exception as e:
            print(f"❌ Get training data error: {e}")
            return None

    def smart_load_data(self, force_refresh=False):
        """Smart Cache: Load data efficiently with caching"""
        try:
            print("🧠 Smart Cache: Checking for existing data...")
            
            # Check if processed data exists and is recent
            if not force_refresh and self.is_cache_valid():
                print("💾 Using cached data (Smart Cache hit)")
                success = self.load_processed_data()
                if success:
                    summary = self.get_data_summary()
                    print(f"✅ Cache loaded: {summary.get('total_rows', 0):,} rows")
                    return True
            
            print("🔄 Cache miss or refresh requested - downloading fresh data...")
            return self.run_full_pipeline(force_refresh)
            
        except Exception as e:
            print(f"❌ Smart cache error: {e}")
            return False

    def is_cache_valid(self):
        """Check if cached data is valid and recent"""
        try:
            # Check if files exist
            if not (os.path.exists(self.raw_data_file) and os.path.exists(self.processed_data_file)):
                print("📂 No cache files found")
                return False
            
            # Check file age (refresh if older than 1 day)
            import time
            file_age = time.time() - os.path.getmtime(self.processed_data_file)
            max_age = 24 * 60 * 60  # 24 hours
            
            if file_age > max_age:
                print(f"⏰ Cache expired ({file_age/3600:.1f} hours old)")
                return False
            
            # Try to load and verify data integrity
            if self.load_processed_data():
                if self.processed_data is not None and len(self.processed_data) > 50000:
                    print(f"✅ Cache valid ({file_age/3600:.1f} hours old)")
                    return True
                else:
                    print("❌ Cache corrupted (insufficient data)")
                    return False
            
            return False
            
        except Exception as e:
            print(f"❌ Cache validation error: {e}")
            return False

    def run_full_pipeline(self, force_download=False):
        """Run complete data loading and processing pipeline"""
        try:
            print("🚀 Starting full data pipeline...")
            
            # Step 1: Download historical data
            if not self.download_historical_data(force_download):
                print("❌ Pipeline failed at download step")
                return False
            
            # Step 2: Calculate indicators
            if not self.calculate_indicators():
                print("❌ Pipeline failed at indicators step")
                return False
            
            print("✅ Full pipeline completed successfully!")
            
            # Show summary
            summary = self.get_data_summary()
            print(f"\n📊 Data Summary:")
            print(f"   - Total rows: {summary.get('total_rows', 0):,}")
            print(f"   - Trading days: {summary.get('trading_days', 0):.0f}")
            print(f"   - Data quality: {summary.get('data_quality', 'Unknown')}")
            print(f"   - Features: {len(summary.get('columns', []))}")
            
            return True
            
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            return False

    def get_cache_status(self):
        """Get detailed cache status for GUI display"""
        try:
            status = {
                'cache_exists': False,
                'cache_valid': False,
                'cache_age_hours': 0,
                'data_rows': 0,
                'last_updated': None,
                'file_size_mb': 0
            }
            
            if os.path.exists(self.processed_data_file):
                status['cache_exists'] = True
                
                # File age
                import time
                file_time = os.path.getmtime(self.processed_data_file)
                status['cache_age_hours'] = (time.time() - file_time) / 3600
                status['last_updated'] = datetime.fromtimestamp(file_time)
                
                # File size
                status['file_size_mb'] = os.path.getsize(self.processed_data_file) / (1024 * 1024)
                
                # Validate cache
                status['cache_valid'] = self.is_cache_valid()
                
                # Data info
                if self.load_processed_data():
                    status['data_rows'] = len(self.processed_data) if self.processed_data is not None else 0
            
            return status
            
        except Exception as e:
            print(f"❌ Get cache status error: {e}")
            return status
# environment.py - Custom RL Environment for Trading
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import MetaTrader5 as mt5
from datetime import datetime, timedelta

class TradingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for XAUUSD trading with recovery system
    """
    
    def __init__(self, mt5_interface, recovery_engine, config):
        super(TradingEnvironment, self).__init__()
        
        self.mt5_interface = mt5_interface
        self.recovery_engine = recovery_engine
        self.config = config
        
        # Trading parameters
        self.symbol = config.get('symbol', 'XAUUSD')
        self.initial_lot_size = config.get('initial_lot_size', 0.01)
        self.max_positions = config.get('max_positions', 10)
        
        # Environment parameters
        self.lookback_window = 100  # Number of historical bars to include in state
        self.max_steps = 10000  # Maximum steps per episode
        
        # State space: [market_data, positions, account_info, recovery_info]
        # Market data: OHLCV + technical indicators (50 features)
        # Positions: current positions info (20 features)
        # Account info: balance, equity, margin, etc. (10 features)
        # Recovery info: recovery level, drawdown, etc. (10 features)
        self.observation_space = spaces.Box(
            low=np.array([-10.0] * 92, dtype=np.float32), 
            high=np.array([10.0] * 92, dtype=np.float32), 
            shape=(92,),  
            dtype=np.float32
        )       
        # Action space: [action_type, lot_multiplier, recovery_action]
        # action_type: 0=hold, 1=buy, 2=sell, 3=close_all, 4=hedge
        # lot_multiplier: 0.5 to 3.0 (for position sizing)
        # recovery_action: 0=none, 1=martingale, 2=grid, 3=hedge
        self.action_space = spaces.Box(
            low=np.array([0, 0.5, 0]),
            high=np.array([4, 3.0, 3]),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_start_time = None
        self.episode_trades = []
        self.episode_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Market data cache
        self.market_data_cache = []
        self.last_update_time = None
        
        # Recovery tracking
        self.recovery_active = False
        self.recovery_level = 0
        self.recovery_start_equity = 0.0
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_start_time = datetime.now()
        self.episode_trades = []
        self.episode_pnl = 0.0
        self.max_drawdown = 0.0
        
        # Get initial account info
        account_info = self.mt5_interface.get_account_info()
        if account_info:
            self.peak_equity = account_info.get('equity', 0)
            self.recovery_start_equity = self.peak_equity
        
        # Reset recovery system
        self.recovery_engine.reset()
        self.recovery_active = False
        self.recovery_level = 0
        
        # Get initial market data
        self.update_market_data()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Parse action
        action_type = int(action[0])
        lot_multiplier = float(action[1])
        recovery_action = int(action[2])
        
        # Update market data
        self.update_market_data()
        
        # Execute trading action
        reward = self._execute_action(action_type, lot_multiplier, recovery_action)
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Additional info
        info = self._get_info()
        
        return observation, reward, done, False, info
        
    def _execute_action(self, action_type, lot_multiplier, recovery_action):
        """Execute the trading action and return reward"""
        reward = 0.0
        
        try:
            # Get current market data
            current_price = self._get_current_price()
            if current_price is None:
                return -1.0  # Penalty for invalid market data
            
            # Check if in training mode
            is_training_mode = self.config.get('training_mode', True)
            
            if not is_training_mode:
                # Live trading mode - execute real orders
                # Check for profit taking opportunities FIRST
                profit_signals = self.recovery_engine.check_profit_opportunities(
                    self.mt5_interface, self.symbol
                )
                
                if profit_signals:
                    executed_profits = self.recovery_engine.execute_profit_taking(
                        self.mt5_interface, profit_signals
                    )
                    
                    # Reward for successful profit taking
                    if executed_profits:
                        total_profit_taken = sum(action.get('profit', 0) for action in executed_profits)
                        reward += total_profit_taken / 50.0  # Scale reward
                        
                # Check smart profit strategy
                smart_profit_taken = self.recovery_engine.smart_profit_strategy(
                    self.mt5_interface, self.symbol
                )
                
                if smart_profit_taken:
                    reward += 3.0  # Bonus for smart profit taking
            
            # Calculate position size
            base_lot_size = self.initial_lot_size
            if self.recovery_active:
                base_lot_size = self.recovery_engine.calculate_lot_size(
                    base_lot_size, self.recovery_level
                )
            
            lot_size = base_lot_size * lot_multiplier
            lot_size = max(0.01, min(lot_size, 10.0))  # Limit lot size
            # Round to valid MT5 lot size (0.01 increments)
            lot_size = round(lot_size / 0.01) * 0.01
            lot_size = max(0.01, lot_size)  # Ensure minimum 0.01

            # Execute action based on type
            if action_type == 0:  # Hold
                reward += self._calculate_hold_reward()
                
            elif action_type == 1:  # Buy
                if is_training_mode:
                    # Training mode - simulate
                    success = True
                    print(f"SIMULATED BUY: {lot_size} {self.symbol} at {current_price}")
                else:
                    # Live mode - real order
                    success = self.mt5_interface.place_order(
                        symbol=self.symbol,
                        order_type='buy',
                        volume=lot_size,
                        price=current_price
                    )
                reward += self._calculate_trade_reward(success, 'buy', lot_size)
                
            elif action_type == 2:  # Sell
                if is_training_mode:
                    # Training mode - simulate
                    success = True
                    print(f"SIMULATED SELL: {lot_size} {self.symbol} at {current_price}")
                else:
                    # Live mode - real order
                    success = self.mt5_interface.place_order(
                        symbol=self.symbol,
                        order_type='sell',
                        volume=lot_size,
                        price=current_price
                    )
                reward += self._calculate_trade_reward(success, 'sell', lot_size)
                
            elif action_type == 3:  # Close all positions (PROFIT TAKING)
                if is_training_mode:
                    # Training mode - simulate
                    success = True
                    print(f"SIMULATED CLOSE ALL: {self.symbol}")
                else:
                    # Live mode - real close
                    success = self.mt5_interface.close_all_positions(self.symbol)
                reward += self._calculate_close_reward(success)
                
                # Extra reward if closing with profit (simulated in training)
                if is_training_mode:
                    # Simulate current PnL for training
                    simulated_pnl = reward * 10  # Simple simulation
                    if simulated_pnl > 0:
                        reward += 2.0 + (simulated_pnl / 100.0)
                else:
                    current_pnl = self._get_current_pnl()
                    if current_pnl > 0:
                        reward += 2.0 + (current_pnl / 100.0)  # Bonus for profitable close
                    
            elif action_type == 4:  # Hedge
                if is_training_mode:
                    # Training mode - simulate
                    success = True
                    print(f"SIMULATED HEDGE: {lot_size} {self.symbol}")
                else:
                    # Live mode - real hedge
                    success = self._execute_hedge_action(lot_size)
                reward += self._calculate_hedge_reward(success)
            
            # Execute recovery action if needed (only in live mode)
            if not is_training_mode and recovery_action > 0 and self._should_activate_recovery():
                self._execute_recovery_action(recovery_action)
                
            # Update recovery status
            self._update_recovery_status()
            
        except Exception as e:
            print(f"Error executing action: {e}")
            reward = -5.0  # Heavy penalty for errors
            
        return reward
        
    def _calculate_hold_reward(self):
        """Calculate reward for holding position"""
        # Small penalty for inaction, but reward if profitable
        current_pnl = self._get_current_pnl()
        
        if current_pnl > 0:
            return 0.1  # Small reward for profitable hold
        elif current_pnl < -100:  # Significant loss
            return -0.5  # Penalty for holding losing position
        else:
            return -0.01  # Small penalty for inaction
            
    def _calculate_trade_reward(self, success, trade_type, lot_size):
        """Calculate reward for trade execution"""
        if not success:
            return -2.0  # Penalty for failed trade
            
        # Base reward for successful trade
        reward = 1.0
        
        # Adjust reward based on market conditions
        market_trend = self._get_market_trend()
        if (trade_type == 'buy' and market_trend > 0) or \
           (trade_type == 'sell' and market_trend < 0):
            reward += 0.5  # Bonus for trading with trend
            
        # Penalty for oversized positions
        if lot_size > self.initial_lot_size * 2:
            reward -= 0.3
            
        return reward
        
    def _calculate_close_reward(self, success):
        """Calculate reward for closing positions"""
        if not success:
            return -1.0
            
        # Check if closing was profitable
        final_pnl = self._get_current_pnl()
        if final_pnl > 0:
            return 2.0 + (final_pnl / 100)  # Reward based on profit
        else:
            return -1.0 + (final_pnl / 100)  # Penalty based on loss
            
    def _calculate_hedge_reward(self, success):
        """Calculate reward for hedge action"""
        if not success:
            return -1.0
            
        # Reward for risk management
        return 1.5
        
    def _get_observation(self):
        """Get current environment observation"""
        observation = np.zeros(92, dtype=np.float32)
        
        try:
            # Market data features (51 features แทน 50)
            market_features = self._get_market_features()
            observation[:51] = market_features  # เปลี่ยนจาก 50 เป็น 51

            # Position features (20 features) 
            position_features = self._get_position_features()
            observation[51:71] = position_features  # เปลี่ยน index

            # Account features (10 features)
            account_features = self._get_account_features()  
            observation[71:81] = account_features  # เปลี่ยน index

            # Recovery features (10 features)
            recovery_features = self._get_recovery_features()
            observation[81:91] = recovery_features  # เปลี่ยน index

        except Exception as e:
            print(f"Error getting observation: {e}")
            # Return zeros if error occurs
            
        return observation
        
    def _get_market_features(self):
        """Extract market-related features"""
        features = np.zeros(51)
        
        if len(self.market_data_cache) < 20:  # Reduced minimum requirement
            return features
            
        try:
            # Get recent market data
            recent_data = self.market_data_cache[-min(len(self.market_data_cache), 50):]
            df = pd.DataFrame(recent_data)
            
            if df.empty or len(df) < 5:
                return features
                
            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            # Fill any NaN values
            df = df.fillna(method='ffill').fillna(0)
            
            # Basic OHLCV features (normalized) - SAFE VERSION
            if len(df) > 1 and df['close'].std() > 0:
                close_std = df['close'].std()
                high_std = df['high'].std() 
                low_std = df['low'].std()
                
                features[0] = float((df['close'].iloc[-1] - df['close'].mean()) / close_std)
                features[1] = float((df['high'].iloc[-1] - df['high'].mean()) / high_std) if high_std > 0 else 0.0
                features[2] = float((df['low'].iloc[-1] - df['low'].mean()) / low_std) if low_std > 0 else 0.0
                
                if 'volume' in df.columns and df['volume'].std() > 0:
                    features[3] = float((df['volume'].iloc[-1] - df['volume'].mean()) / df['volume'].std())                    
            # Simple technical indicators (4-13)
            features[4:14] = self._safe_technical_indicators(df)
            
            # Simple price momentum (14-23)
            features[14:24] = self._safe_momentum_features(df)
            
            # Simple volatility (24-33)
            features[24:34] = self._safe_volatility_features(df)
            
            # Simple support/resistance (34-43)
            features[34:44] = self._safe_sr_features(df)
            
            # Time-based features (44-49)
            features[44:51] = self._calculate_time_features()
            
            # Final safety check
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        except Exception as e:
            print(f"Error calculating market features: {e}")
            features = np.zeros(50)
            
        return features
        
    def _safe_technical_indicators(self, df):
        """Safe technical indicators calculation"""
        indicators = np.zeros(10)
        
        try:
            if len(df) < 5:
                return indicators
                
            # Simple moving average
            if len(df) >= 10:
                sma_10 = df['close'].rolling(10, min_periods=1).mean()
                last_close = float(df['close'].iloc[-1])
                last_sma = float(sma_10.iloc[-1])
                
            if last_sma > 0 and not np.isnan(last_sma):
                indicators[0] = (last_close - last_sma) / last_sma                    
            
            # Price change ratios
            if len(df) >= 2:
                price_change = float(df['close'].iloc[-1] - df['close'].iloc[-2])
                base_price = float(df['close'].iloc[-2])
                
                if base_price > 0:
                    indicators[1] = price_change / base_price
            # Safe high/low features
            if len(df) > 1:
                high_std = df['high'].std()
                low_std = df['low'].std()
                
                if high_std > 0 and not np.isnan(high_std):
                    indicators[1] = (df['high'].iloc[-1] - df['high'].mean()) / high_std
                
                if low_std > 0 and not np.isnan(low_std):
                    indicators[2] = (df['low'].iloc[-1] - df['low'].mean()) / low_std        
            # High-Low ratio
            last_high = float(df['high'].iloc[-1])
            last_low = float(df['low'].iloc[-1])
            last_close = float(df['close'].iloc[-1])
            
            if last_high > last_low:
                indicators[2] = (last_close - last_low) / (last_high - last_low)
                
            # Simple RSI approximation
            if len(df) >= 5:
                price_changes = df['close'].diff().dropna()
                if len(price_changes) > 0:
                    gains = price_changes[price_changes > 0].sum()
                    losses = abs(price_changes[price_changes < 0].sum())
                    
                    if losses > 0:
                        rs = gains / losses
                        rsi = 100 - (100 / (1 + rs))
                        indicators[3] = (rsi - 50) / 50  # Normalize
                        
            # Clean indicators
            indicators = np.nan_to_num(indicators, nan=0.0, posinf=1.0, neginf=-1.0)
            
        except Exception as e:
            print(f"Error in safe technical indicators: {e}")
            indicators = np.zeros(10)
            
        return indicators
        
    def _safe_momentum_features(self, df):
        """Safe momentum calculation"""
        features = np.zeros(10)
        
        try:
            if len(df) < 3:
                return features
                
            # Simple momentum over different periods
            periods = [2, 3, 5, 10]
            
            for i, period in enumerate(periods):
                if len(df) > period:
                    current_price = float(df['close'].iloc[-1])
                    past_price = float(df['close'].iloc[-period])
                    
                    if past_price > 0:
                        momentum = (current_price - past_price) / past_price
                        features[i] = momentum
                        
            # Simple velocity (price change rate)
            if len(df) >= 3:
                recent_changes = df['close'].diff().tail(3)
                if len(recent_changes) > 0 and df['close'].iloc[-1] > 0:
                    velocity = float(recent_changes.mean() / df['close'].iloc[-1])
                    features[4] = velocity
                    
            # Clean features
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        except Exception as e:
            print(f"Error in safe momentum features: {e}")
            features = np.zeros(10)
            
        return features
        
    def _safe_volatility_features(self, df):
        """Safe volatility calculation"""
        features = np.zeros(10)
        
        try:
            if len(df) < 3:
                return features
                
            # Simple volatility measures
            returns = df['close'].pct_change().dropna()
            
            if len(returns) > 0:
                # Standard deviation of returns
                vol = float(returns.std())
                features[0] = vol
                
                # High-Low volatility
                if len(df) >= 5:
                    hl_range = (df['high'] - df['low']) / df['close']
                    hl_vol = float(hl_range.tail(5).mean())
                    features[1] = hl_vol
                    
                # Recent vs historical volatility
                if len(returns) >= 10:
                    recent_vol = float(returns.tail(5).std())
                    hist_vol = float(returns.std())
                    
                    if hist_vol > 0:
                        features[2] = recent_vol / hist_vol
                        
            # Clean features
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        except Exception as e:
            print(f"Error in safe volatility features: {e}")
            features = np.zeros(10)
            
        return features
        
    def _safe_sr_features(self, df):
        """Safe support/resistance calculation"""
        features = np.zeros(10)
        
        try:
            if len(df) < 5:
                return features
                
            # Simple support/resistance
            recent_high = float(df['high'].tail(10).max())
            recent_low = float(df['low'].tail(10).min())
            current_price = float(df['close'].iloc[-1])
            
            # Distance to recent high/low
            if recent_high > current_price:
                features[0] = (recent_high - current_price) / current_price
                
            if current_price > recent_low:
                features[1] = (current_price - recent_low) / current_price
                
            # Price position in recent range
            if recent_high > recent_low:
                features[2] = (current_price - recent_low) / (recent_high - recent_low)
                
            # Clean features
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        except Exception as e:
            print(f"Error in safe S/R features: {e}")
            features = np.zeros(10)
            
        return features
        
    def _get_position_features(self):
        """Extract position-related features"""
        features = np.zeros(20)
        
        try:
            positions = self.mt5_interface.get_positions()
            
            if positions:
                # Number of positions
                features[0] = len(positions) / self.max_positions
                
                # Total volume
                total_volume = sum(pos.get('volume', 0) for pos in positions)
                features[1] = total_volume / (self.initial_lot_size * 10)
                
                # Average entry price
                if positions:
                    avg_entry = sum(pos.get('price_open', 0) for pos in positions) / len(positions)
                    current_price = self._get_current_price()
                    if current_price:
                        features[2] = (current_price - avg_entry) / current_price
                
                # Position types distribution
                buy_positions = sum(1 for pos in positions if pos.get('type', 0) == 0)
                sell_positions = sum(1 for pos in positions if pos.get('type', 0) == 1)
                features[3] = buy_positions / max(len(positions), 1)
                features[4] = sell_positions / max(len(positions), 1)
                
                # PnL features
                total_pnl = sum(pos.get('profit', 0) for pos in positions)
                features[5] = total_pnl / 1000  # Normalize PnL
                
                # Position age (average)
                current_time = datetime.now()
                avg_age = 0
                for pos in positions:
                    open_time = pos.get('time', current_time)
                    if isinstance(open_time, (int, float)):
                        open_time = datetime.fromtimestamp(open_time)
                    age = (current_time - open_time).total_seconds() / 3600  # Hours
                    avg_age += age
                avg_age /= max(len(positions), 1)
                features[6] = avg_age / 24  # Normalize to days
                
                # Lot size distribution
                lot_sizes = [pos.get('volume', 0) for pos in positions]
                if lot_sizes:
                    features[7] = np.mean(lot_sizes) / self.initial_lot_size
                    features[8] = np.std(lot_sizes) / self.initial_lot_size
                    features[9] = np.max(lot_sizes) / self.initial_lot_size
                    features[10] = np.min(lot_sizes) / self.initial_lot_size
                
        except Exception as e:
            print(f"Error calculating position features: {e}")
            
        return features
        
    def _get_account_features(self):
        """Extract account-related features"""
        features = np.zeros(10)
        
        try:
            account_info = self.mt5_interface.get_account_info()
            
            if account_info:
                balance = account_info.get('balance', 0)
                equity = account_info.get('equity', 0)
                margin = account_info.get('margin', 0)
                free_margin = account_info.get('margin_free', 0)
                
                # Normalize features
                features[0] = balance / 10000  # Normalize balance
                features[1] = equity / 10000   # Normalize equity
                features[2] = margin / 10000   # Normalize margin
                features[3] = free_margin / 10000  # Normalize free margin
                
                # Ratios
                if balance > 0:
                    features[4] = equity / balance  # Equity ratio
                    features[5] = margin / balance  # Margin ratio
                    
                # Drawdown
                if self.peak_equity > 0:
                    drawdown = (self.peak_equity - equity) / self.peak_equity
                    features[6] = drawdown
                    
                # Margin level
                if margin > 0:
                    features[7] = equity / margin  # Margin level
                    
                # Account growth
                if hasattr(self, 'initial_balance'):
                    features[8] = (balance - self.initial_balance) / self.initial_balance
                    
        except Exception as e:
            print(f"Error calculating account features: {e}")
            
        return features
        
    def _get_recovery_features(self):
        """Extract recovery-related features"""
        features = np.zeros(10)
        
        try:
            recovery_info = self.recovery_engine.get_status()
            
            features[0] = 1.0 if self.recovery_active else 0.0
            features[1] = self.recovery_level / 10  # Normalize level
            
            # Recovery type encoding
            recovery_type = recovery_info.get('type', 'none')
            if recovery_type == 'martingale':
                features[2] = 1.0
            elif recovery_type == 'grid':
                features[3] = 1.0
            elif recovery_type == 'hedge':
                features[4] = 1.0
                
            # Recovery metrics
            features[5] = recovery_info.get('total_recovery_attempts', 0) / 10
            features[6] = recovery_info.get('success_rate', 0)
            features[7] = recovery_info.get('average_recovery_time', 0) / 3600  # Hours
            
            # Drawdown during recovery
            if self.recovery_start_equity > 0:
                current_equity = self.mt5_interface.get_account_info().get('equity', 0)
                recovery_drawdown = (self.recovery_start_equity - current_equity) / self.recovery_start_equity
                features[8] = recovery_drawdown
                
        except Exception as e:
            print(f"Error calculating recovery features: {e}")
            
        return features
        
    def _calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        indicators = np.zeros(10)
        
        try:
            # Moving averages
            sma_20 = df['close'].rolling(20).mean()
            sma_50 = df['close'].rolling(50).mean()
            
            indicators[0] = (df['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
            indicators[1] = (df['close'].iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
            indicators[2] = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators[3] = (rsi.iloc[-1] - 50) / 50  # Normalize RSI
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            indicators[4] = macd.iloc[-1] / df['close'].iloc[-1]
            indicators[5] = (macd.iloc[-1] - signal.iloc[-1]) / df['close'].iloc[-1]
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(20).mean()
            std_20 = df['close'].rolling(20).std()
            upper_band = sma_20 + (std_20 * 2)
            lower_band = sma_20 - (std_20 * 2)
            bb_position = (df['close'].iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            indicators[6] = bb_position - 0.5  # Center around 0
            
            # ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = tr.rolling(14).mean()
            indicators[7] = atr.iloc[-1] / df['close'].iloc[-1]
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            
        return indicators
        
    def _calculate_momentum_features(self, df):
        """Calculate momentum-based features"""
        features = np.zeros(10)
        
        try:
            # Price momentum
            for i, period in enumerate([5, 10, 20, 50]):
                if len(df) > period:
                    momentum = (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period]
                    features[i] = momentum
                    
            # Volume momentum
            if 'volume' in df.columns:
                for i, period in enumerate([5, 10, 20]):
                    if len(df) > period:
                        vol_momentum = (df['volume'].iloc[-1] - df['volume'].iloc[-period]) / df['volume'].iloc[-period]
                        features[i + 4] = vol_momentum
                        
            # Rate of change
            if len(df) > 10:
                roc = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
                features[7] = roc
                
            # Velocity and acceleration
            if len(df) > 3:
                velocity = df['close'].diff().rolling(3).mean().iloc[-1] / df['close'].iloc[-1]
                acceleration = df['close'].diff().diff().rolling(3).mean().iloc[-1] / df['close'].iloc[-1]
                features[8] = velocity
                features[9] = acceleration
                
        except Exception as e:
            print(f"Error calculating momentum features: {e}")
            
        return features
        
    def _calculate_volatility_features(self, df):
        """Calculate volatility-based features"""
        features = np.zeros(10)
        
        try:
            # Historical volatility
            returns = df['close'].pct_change()
            for i, period in enumerate([10, 20, 50]):
                if len(returns) > period:
                    vol = returns.rolling(period).std() * np.sqrt(252)  # Annualized
                    features[i] = vol
                    
            # High-Low volatility
            hl_vol = ((df['high'] - df['low']) / df['close']).rolling(20).mean()
            features[3] = hl_vol.iloc[-1]
            
            # Close-to-close volatility
            cc_vol = returns.rolling(20).std()
            features[4] = cc_vol.iloc[-1]
            
            # Volatility ratio
            short_vol = returns.rolling(10).std()
            long_vol = returns.rolling(50).std()
            if long_vol.iloc[-1] > 0:
                features[5] = short_vol.iloc[-1] / long_vol.iloc[-1]
                
        except Exception as e:
            print(f"Error calculating volatility features: {e}")
            
        return features
        
    def _calculate_sr_features(self, df):
        """Calculate support/resistance features"""
        features = np.zeros(10)
        
        try:
            # Simple support/resistance levels
            high_20 = df['high'].rolling(20).max()
            low_20 = df['low'].rolling(20).min()
            
            current_price = df['close'].iloc[-1]
            
            # Distance to support/resistance
            if high_20.iloc[-1] > current_price:
                features[0] = (high_20.iloc[-1] - current_price) / current_price
            if low_20.iloc[-1] < current_price:
                features[1] = (current_price - low_20.iloc[-1]) / current_price
                
            # Pivot points
            pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
            features[2] = (current_price - pivot) / pivot
            
        except Exception as e:
            print(f"Error calculating S/R features: {e}")
            
        return features
        
    def _calculate_time_features(self):
        """Calculate time-based features"""
        features = np.zeros(7)
        
        try:
            now = datetime.now()
            
            # Hour of day (normalized)
            features[0] = now.hour / 24
            
            # Day of week (normalized)
            features[1] = now.weekday() / 7
            
            # Day of month (normalized)
            features[2] = now.day / 31
            
            # Month of year (normalized)
            features[3] = now.month / 12
            
            # Market session indicators
            hour = now.hour
            # Asian session (23:00 - 08:00 GMT)
            if hour >= 23 or hour < 8:
                features[4] = 1.0
            # European session (08:00 - 16:00 GMT)
            elif 8 <= hour < 16:
                features[5] = 1.0
            # American session (16:00 - 23:00 GMT)
            else:
                features[6] = 1.0
                
        except Exception as e:
            print(f"Error calculating time features: {e}")
            
        return features
        
    def update_market_data(self):
        """Update market data cache"""
        try:
            # Get latest market data from MT5
            rates = self.mt5_interface.get_rates(self.symbol, mt5.TIMEFRAME_M1, 1)
            
            if rates is not None and len(rates) > 0:
                rate = rates[0]
                market_data = {
                    'time': rate[0],
                    'open': rate[1],
                    'high': rate[2],
                    'low': rate[3],
                    'close': rate[4],
                    'volume': rate[5] if len(rate) > 5 else 0
                }
                
                # Add to cache
                self.market_data_cache.append(market_data)
                
                # Keep only recent data
                if len(self.market_data_cache) > self.lookback_window * 2:
                    self.market_data_cache = self.market_data_cache[-self.lookback_window:]
                    
                self.last_update_time = datetime.now()
                
        except Exception as e:
            print(f"Error updating market data: {e}")
            
    def _get_current_price(self):
        """Get current market price"""
        try:
            if self.market_data_cache:
                return self.market_data_cache[-1]['close']
            return None
        except:
            return None
            
    def _get_current_pnl(self):
        """Get current unrealized PnL"""
        try:
            positions = self.mt5_interface.get_positions()
            return sum(pos.get('profit', 0) for pos in positions)
        except:
            return 0.0
            
    def _get_market_trend(self):
        """Get market trend direction"""
        try:
            if len(self.market_data_cache) < 20:
                return 0
                
            recent_prices = [data['close'] for data in self.market_data_cache[-20:]]
            trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
            return trend
        except:
            return 0
            
    def _should_activate_recovery(self):
        """Check if recovery should be activated"""
        try:
            current_pnl = self._get_current_pnl()
            return current_pnl < -100  # Activate recovery if loss > $100
        except:
            return False
            
    def _execute_recovery_action(self, recovery_action):
        """Execute recovery action"""
        try:
            if recovery_action == 1:  # Martingale
                self.recovery_engine.activate_martingale()
            elif recovery_action == 2:  # Grid
                self.recovery_engine.activate_grid()
            elif recovery_action == 3:  # Hedge
                self.recovery_engine.activate_hedge()
                
            self.recovery_active = True
            self.recovery_level += 1
            
        except Exception as e:
            print(f"Error executing recovery action: {e}")
            
    def _execute_hedge_action(self, lot_size):
        """Execute hedge action"""
        try:
            positions = self.mt5_interface.get_positions()
            if not positions:
                return False
                
            # Calculate net position
            net_volume = 0
            for pos in positions:
                if pos.get('type', 0) == 0:  # Buy
                    net_volume += pos.get('volume', 0)
                else:  # Sell
                    net_volume -= pos.get('volume', 0)
                    
            # Place hedge order
            if net_volume > 0:  # Net long, place sell
                return self.mt5_interface.place_order(
                    symbol=self.symbol,
                    order_type='sell',
                    volume=abs(net_volume),
                    price=self._get_current_price()
                )
            elif net_volume < 0:  # Net short, place buy
                return self.mt5_interface.place_order(
                    symbol=self.symbol,
                    order_type='buy',
                    volume=abs(net_volume),
                    price=self._get_current_price()
                )
                
            return True
            
        except Exception as e:
            print(f"Error executing hedge: {e}")
            return False
            
    def _update_recovery_status(self):
        """Update recovery status"""
        try:
            current_pnl = self._get_current_pnl()
            
            # Check if recovery is successful
            if self.recovery_active and current_pnl > 0:
                self.recovery_active = False
                self.recovery_level = 0
                self.recovery_engine.reset()
                
            # Update peak equity
            account_info = self.mt5_interface.get_account_info()
            if account_info:
                equity = account_info.get('equity', 0)
                if equity > self.peak_equity:
                    self.peak_equity = equity
                    
        except Exception as e:
            print(f"Error updating recovery status: {e}")
            
    def _is_episode_done(self):
        """Check if episode should end"""
        try:
            # End if maximum steps reached
            if self.current_step >= self.max_steps:
                return True
                
            # End if account equity is too low
            account_info = self.mt5_interface.get_account_info()
            if account_info:
                equity = account_info.get('equity', 0)
                balance = account_info.get('balance', 0)
                if equity < balance * 0.3:  # 70% drawdown
                    return True
                    
            # End if recovery failed multiple times
            if self.recovery_level > 5:
                return True
                
            return False
            
        except:
            return False
            
    def _get_info(self):
        """Get additional info for the step"""
        try:
            account_info = self.mt5_interface.get_account_info()
            positions = self.mt5_interface.get_positions()
            
            info = {
                'current_step': self.current_step,
                'episode_pnl': self._get_current_pnl(),
                'account_balance': account_info.get('balance', 0) if account_info else 0,
                'account_equity': account_info.get('equity', 0) if account_info else 0,
                'open_positions': len(positions) if positions else 0,
                'recovery_active': self.recovery_active,
                'recovery_level': self.recovery_level,
                'max_drawdown': self.max_drawdown,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades
            }
            
            return info
            
        except:
            return {}
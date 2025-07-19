import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import datetime

class Environment(gym.Env):    
    def __init__(self, mt5_interface, config):
        super(Environment, self).__init__()
        
        print("🏗️ Initializing Simple Trading Environment...")
        
        # Core components
        self.mt5_interface = mt5_interface
        self.config = config
        
        # Trading parameters
        self.symbol = config.get('symbol', 'XAUUSD')
        self.initial_lot_size = config.get('lot_size', 0.01)
        self.max_positions = config.get('max_positions', 5)
        
        # === SIMPLE OBSERVATION SPACE (30 features) ===
        self.observation_space = spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(30,),
            dtype=np.float32
        )
        
        # === SIMPLE ACTION SPACE (3 dimensions) ===
        self.action_space = spaces.Box(
            low=np.array([0, 0.01, 0]),      # [action_type, volume, stop_loss]
            high=np.array([3, 0.10, 1]),     # [hold/buy/sell/close, volume, sl%]
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_start_time = None
        self.episode_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Market data
        self.market_data = []
        self.last_prices = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        
        print("✅ Simple Environment initialized:")
        print(f"   - Symbol: {self.symbol}")
        print(f"   - Observation Space: {self.observation_space.shape[0]} features")
        print(f"   - Action Space: {self.action_space.shape[0]} dimensions")
        print(f"   - Max Positions: {self.max_positions}")

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        print(f"🔄 Resetting Environment...")
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_start_time = datetime.now()
        self.episode_pnl = 0.0
        self.max_drawdown = 0.0
        
        # Get initial account info
        try:
            account_info = self.mt5_interface.get_account_info()
            if account_info:
                self.peak_equity = account_info.get('equity', 1000)
        except:
            self.peak_equity = 1000  # Default
        
        # Clear data
        self.market_data.clear()
        self.last_prices.clear()
        
        # Update market data
        self.update_market_data()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        print("✅ Environment reset complete")
        return observation, info

    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Parse action
        action_type = int(action[0])
        volume = float(action[1])
        stop_loss = float(action[2])
        
        # Update market data
        self.update_market_data()
        
        # Execute action
        reward = self._execute_action(action_type, volume, stop_loss)
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Get info
        info = self._get_info()
        
        return observation, reward, done, False, info

    def _get_observation(self):
        """Get simplified observation (30 features)"""
        try:
            obs = np.zeros(30, dtype=np.float32)
            
            # === MARKET DATA (10 features) ===
            if len(self.last_prices) >= 5:
                # Basic OHLC normalized
                prices = self.last_prices[-5:]
                obs[0] = (prices[-1] - prices[0]) / prices[0]  # Price change
                obs[1] = (max(prices) - min(prices)) / prices[0]  # Range
                obs[2] = np.std(prices) / np.mean(prices)  # Volatility
                obs[3] = (prices[-1] - np.mean(prices)) / np.std(prices)  # Z-score
                obs[4] = 1.0 if prices[-1] > prices[-2] else -1.0  # Direction
                
                # Simple indicators
                if len(self.last_prices) >= 20:
                    sma_20 = np.mean(self.last_prices[-20:])
                    obs[5] = (prices[-1] - sma_20) / sma_20  # SMA deviation
                
                if len(self.last_prices) >= 14:
                    obs[6] = self._calculate_rsi()  # RSI
                
                # MACD approximation
                if len(self.last_prices) >= 26:
                    ema_12 = np.mean(self.last_prices[-12:])
                    ema_26 = np.mean(self.last_prices[-26:])
                    obs[7] = (ema_12 - ema_26) / ema_26  # MACD line
                
                obs[8] = np.random.normal(0, 0.1)  # Market noise
                obs[9] = np.random.normal(0, 0.1)  # Future expansion
            
            # === ACCOUNT INFO (10 features) ===
            try:
                account_info = self.mt5_interface.get_account_info()
                if account_info:
                    balance = account_info.get('balance', 1000)
                    equity = account_info.get('equity', 1000)
                    margin = account_info.get('margin', 0)
                    
                    obs[10] = balance / 10000  # Normalized balance
                    obs[11] = equity / 10000   # Normalized equity
                    obs[12] = margin / balance if balance > 0 else 0  # Margin ratio
                    obs[13] = (equity - balance) / balance if balance > 0 else 0  # P&L ratio
                    obs[14] = self.episode_pnl / balance if balance > 0 else 0  # Episode P&L
            except:
                pass  # Keep zeros
            
            # === POSITION INFO (10 features) ===
            try:
                positions = self.mt5_interface.get_positions()
                obs[15] = len(positions) / self.max_positions  # Position count ratio
                
                if positions:
                    total_volume = sum(pos.get('volume', 0) for pos in positions)
                    total_profit = sum(pos.get('profit', 0) for pos in positions)
                    
                    obs[16] = total_volume / 1.0  # Total volume
                    obs[17] = total_profit / 100  # Total profit normalized
                    
                    # Position distribution
                    buy_count = sum(1 for pos in positions if pos.get('type', 0) == 0)
                    sell_count = len(positions) - buy_count
                    obs[18] = buy_count / max(len(positions), 1)  # Buy ratio
                    obs[19] = sell_count / max(len(positions), 1)  # Sell ratio
            except:
                pass  # Keep zeros
            
            # === TIME & SESSION INFO (5 features) ===
            now = datetime.now()
            obs[20] = now.hour / 24  # Hour of day
            obs[21] = now.weekday() / 7  # Day of week
            obs[22] = self.current_step / 1000  # Episode progress
            obs[23] = np.random.normal(0, 0.1)  # Session indicator
            obs[24] = np.random.normal(0, 0.1)  # Future expansion
            
            # === TRADING STATS (5 features) ===
            obs[25] = self.total_trades / 100  # Total trades normalized
            obs[26] = self.winning_trades / max(self.total_trades, 1)  # Win rate
            obs[27] = self.max_drawdown / 100  # Max drawdown
            obs[28] = np.random.normal(0, 0.1)  # Reserved
            obs[29] = np.random.normal(0, 0.1)  # Reserved
            
            # Clip values to valid range
            obs = np.clip(obs, -10.0, 10.0)
            
            return obs
            
        except Exception as e:
            print(f"Observation error: {e}")
            return np.zeros(30, dtype=np.float32)

    def _execute_action(self, action_type, volume, stop_loss):
        """Execute simple trading action"""
        try:
            reward = 0.0
            
            if action_type == 0:  # Hold
                reward = 0.1  # Small positive reward for patience
                
            elif action_type == 1:  # Buy
                reward = self._execute_buy(volume, stop_loss)
                
            elif action_type == 2:  # Sell
                reward = self._execute_sell(volume, stop_loss)
                
            elif action_type == 3:  # Close all
                reward = self._close_all_positions()
            
            return reward
            
        except Exception as e:
            print(f"Action execution error: {e}")
            return -1.0  # Penalty for errors

    def _execute_buy(self, volume, stop_loss):
        """Execute buy order"""
        try:
            # Check position limits
            positions = self.mt5_interface.get_positions()
            if len(positions) >= self.max_positions:
                return -0.5  # Penalty for hitting limits
            
            # Get current price
            price_info = self.mt5_interface.get_current_price(self.symbol)
            if not price_info:
                return -1.0
            
            price = price_info['ask']
            sl_price = price * (1 - stop_loss) if stop_loss > 0 else None
            
            # Place order
            success = self.mt5_interface.place_order(
                symbol=self.symbol,
                order_type='buy',
                volume=volume,
                price=price,
                sl=sl_price
            )
            
            if success:
                self.total_trades += 1
                return 1.0  # Reward for successful trade
            else:
                return -0.5  # Penalty for failed trade
                
        except Exception as e:
            print(f"Buy execution error: {e}")
            return -1.0

    def _execute_sell(self, volume, stop_loss):
        """Execute sell order"""
        try:
            # Check position limits
            positions = self.mt5_interface.get_positions()
            if len(positions) >= self.max_positions:
                return -0.5
            
            # Get current price
            price_info = self.mt5_interface.get_current_price(self.symbol)
            if not price_info:
                return -1.0
            
            price = price_info['bid']
            sl_price = price * (1 + stop_loss) if stop_loss > 0 else None
            
            # Place order
            success = self.mt5_interface.place_order(
                symbol=self.symbol,
                order_type='sell',
                volume=volume,
                price=price,
                sl=sl_price
            )
            
            if success:
                self.total_trades += 1
                return 1.0
            else:
                return -0.5
                
        except Exception as e:
            print(f"Sell execution error: {e}")
            return -1.0

    def _close_all_positions(self):
        """Close all open positions"""
        try:
            positions = self.mt5_interface.get_positions()
            if not positions:
                return -0.1  # Small penalty for unnecessary action
            
            closed_count = 0
            total_profit = 0
            
            for pos in positions:
                ticket = pos.get('ticket')
                profit = pos.get('profit', 0)
                
                if self.mt5_interface.close_position(ticket):
                    closed_count += 1
                    total_profit += profit
                    if profit > 0:
                        self.winning_trades += 1
            
            # Reward based on profit and closure success
            if closed_count > 0:
                base_reward = 2.0  # Reward for cleaning up
                profit_reward = total_profit / 100  # Normalize profit
                return base_reward + profit_reward
            else:
                return -1.0  # Penalty for failed closures
                
        except Exception as e:
            print(f"Close all error: {e}")
            return -1.0

    def update_market_data(self):
        """Update market data for observations"""
        try:
            # Get current price
            price_info = self.mt5_interface.get_current_price(self.symbol)
            if price_info:
                current_price = (price_info['bid'] + price_info['ask']) / 2
                self.last_prices.append(current_price)
                
                # Keep only last 50 prices
                if len(self.last_prices) > 50:
                    self.last_prices.pop(0)
                    
        except Exception as e:
            print(f"Market data update error: {e}")

    def _calculate_rsi(self, period=14):
        """Calculate simple RSI"""
        try:
            if len(self.last_prices) < period + 1:
                return 0.5  # Neutral RSI
            
            prices = self.last_prices[-period-1:]
            gains = []
            losses = []
            
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                return 1.0
            
            rs = avg_gain / avg_loss
            rsi = 1 - (1 / (1 + rs))
            
            return (rsi - 0.5) * 2  # Normalize to -1 to 1
            
        except:
            return 0.0

    def _is_episode_done(self):
        """Check if episode should end"""
        # End episode after certain steps or large drawdown
        if self.current_step >= 1000:
            return True
            
        if self.max_drawdown > 500:  # $500 max loss
            return True
            
        return False

    def _get_info(self):
        """Get episode info"""
        try:
            account_info = self.mt5_interface.get_account_info()
            positions = self.mt5_interface.get_positions()
            
            return {
                'current_step': self.current_step,
                'episode_pnl': self.episode_pnl,
                'account_balance': account_info.get('balance', 0) if account_info else 0,
                'account_equity': account_info.get('equity', 0) if account_info else 0,
                'open_positions': len(positions),
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'win_rate': self.winning_trades / max(self.total_trades, 1),
                'max_drawdown': self.max_drawdown
            }
        except:
            return {
                'current_step': self.current_step,
                'episode_pnl': 0,
                'account_balance': 0,
                'account_equity': 0,
                'open_positions': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0
            }
# core/environment.py - Recovery Trading Environment (ทับของเดิม)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class Environment(gym.Env):
    """
    Recovery Trading Environment for AI Agent
    - Specialized for "แก้ไม้" (Recovery) strategy
    - Uses M5 XAUUSD historical data  
    - Dynamic position sizing based on P&L
    - Risk management with recovery logic
    - REPLACES the old basic environment
    """
    
    def __init__(self, mt5_interface, config, historical_data=None):
        super(Environment, self).__init__()
        
        print("🔥 Initializing Recovery Environment...")
        
        # Core components
        self.mt5_interface = mt5_interface
        self.config = config
        self.historical_data = historical_data
        
        # Trading parameters
        self.symbol = config.get('symbol', 'XAUUSD')
        self.base_lot_size = config.get('lot_size', 0.01)
        self.max_positions = config.get('max_positions', 5)
        self.max_recovery_levels = config.get('max_recovery_levels', 3)
        
        # Recovery strategy parameters
        self.recovery_multiplier = config.get('recovery_multiplier', 1.5)  # 1.5x lot size
        self.recovery_threshold = config.get('recovery_threshold', -20.0)  # -$20 trigger
        # Removed max_drawdown_limit - no stop trading limit
        
        # === OBSERVATION SPACE (40 features for recovery) ===
        self.observation_space = spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(40,),
            dtype=np.float32
        )
        
        # === ACTION SPACE (4 dimensions for recovery) ===
        # ✅ Fixed: Use proper dtypes and ranges
        self.action_space = spaces.Box(
            low=np.array([0, 0.01, 0, 0], dtype=np.float32),           
            high=np.array([4, 0.50, 100, 2], dtype=np.float32),        
            shape=(4,),
            dtype=np.float32
        )
        
        # State tracking
        self.current_step = 0
        self.data_index = 0
        self.episode_start_time = None
        self.episode_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Recovery tracking
        self.recovery_level = 0
        self.in_recovery_mode = False
        self.consecutive_losses = 0
        self.recovery_start_balance = 0.0
        self.recovery_target = 0.0
        
        # Position tracking
        self.positions = []
        self.position_history = []
        self.last_trade_result = 0.0
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        
        # Mode settings
        self.is_training_mode = config.get('training_mode', True)
        
        print("✅ Recovery Environment initialized:")
        print(f"   - Symbol: {self.symbol}")
        print(f"   - Base Lot: {self.base_lot_size}")
        print(f"   - Recovery Multiplier: {self.recovery_multiplier}x")
        print(f"   - Max Recovery Levels: {self.max_recovery_levels}")
        print(f"   - Recovery Threshold: ${self.recovery_threshold}")
        print(f"   - Training Mode: {self.is_training_mode}")
        print(f"   - No Drawdown Limit: Unlimited recovery attempts")

    def set_historical_data(self, historical_data: pd.DataFrame):
        """Set historical data for training"""
        self.historical_data = historical_data
        print(f"📊 Historical data set: {len(historical_data):,} rows")

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        print(f"🔄 Resetting Recovery Environment...")
        
        # Reset episode tracking
        self.current_step = 0
        self.data_index = 50 if self.historical_data is not None else 0  # Start after indicators warmup
        self.episode_start_time = datetime.now()
        self.episode_pnl = 0.0
        self.max_drawdown = 0.0
        
        # Reset recovery tracking
        self.recovery_level = 0
        self.in_recovery_mode = False
        self.consecutive_losses = 0
        self.recovery_start_balance = 1000.0  # Starting balance
        self.recovery_target = 0.0
        self.peak_equity = 1000.0
        
        # Reset positions
        self.positions.clear()
        self.position_history.clear()
        self.last_trade_result = 0.0
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        print("✅ Recovery Environment reset complete")
        return observation, info

    def step(self, action):
        """Execute one step in the recovery environment"""
        self.current_step += 1
        self.data_index += 1
        
        # ✅ Ensure action is proper numpy array with correct shape
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # ✅ Handle both 1D and 2D action arrays from SB3
        if action.ndim > 1:
            action = action.flatten()
        
        # ✅ Ensure we have exactly 4 dimensions
        if len(action) < 4:
            # Pad with zeros if action is too short
            padded_action = np.zeros(4, dtype=np.float32)
            padded_action[:len(action)] = action
            action = padded_action
        elif len(action) > 4:
            # Truncate if action is too long
            action = action[:4]
        
        # Parse action safely
        action_type = int(np.clip(action[0], 0, 4))
        volume = float(np.clip(action[1], 0.01, 0.50))
        sl_pips = float(np.clip(action[2], 0, 100))
        recovery_mode = int(np.clip(action[3], 0, 2))
        
        # Execute action with recovery logic
        reward = self._execute_recovery_action(action_type, volume, sl_pips, recovery_mode)
        
        # Update recovery state
        self._update_recovery_state()
        
        # Get new observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Get info
        info = self._get_info()
        
        return observation, reward, done, False, info

    def _get_observation(self):
        """Get comprehensive observation for recovery trading (40 features)"""
        try:
            obs = np.zeros(40, dtype=np.float32)
            
            # === MARKET DATA (15 features) ===
            if self.historical_data is not None and self.data_index < len(self.historical_data):
                current_data = self.historical_data.iloc[self.data_index]
                
                # Basic OHLC features (normalized)
                close_price = current_data['close']
                obs[0] = (current_data['high'] - current_data['low']) / close_price  # Range
                obs[1] = (current_data['close'] - current_data['open']) / close_price  # Candle body
                
                # Trend indicators
                obs[2] = (current_data['close'] - current_data['SMA_20']) / current_data['ATR_14']  # SMA20 position
                obs[3] = (current_data['close'] - current_data['SMA_50']) / current_data['ATR_14']  # SMA50 position
                obs[4] = (current_data['EMA_12'] - current_data['EMA_26']) / current_data['ATR_14']  # EMA spread
                
                # Momentum indicators
                obs[5] = (current_data['RSI_14'] - 50) / 50  # RSI normalized (-1 to 1)
                obs[6] = current_data['MACD'] / current_data['ATR_14']  # MACD normalized
                obs[7] = current_data['MACD_Histogram'] / current_data['ATR_14']  # MACD momentum
                
                # Volatility indicators
                obs[8] = current_data['BB_Position']  # Position in Bollinger Bands (0 to 1)
                obs[9] = current_data['BB_Width']  # Bollinger Band width
                obs[10] = current_data['ATR_14'] / close_price  # ATR percentage
                obs[11] = current_data['Volatility_Regime']  # Volatility regime
                
                # Market context
                obs[12] = current_data['Trend_Strength']  # Trend strength
                obs[13] = current_data['RSI_Momentum'] / 10  # RSI momentum
                obs[14] = current_data['MACD_Momentum'] / current_data['ATR_14']  # MACD momentum
            
            # === RECOVERY STATE (10 features) ===
            obs[15] = self.recovery_level / self.max_recovery_levels  # Current recovery level
            obs[16] = 1.0 if self.in_recovery_mode else 0.0  # Recovery mode flag
            obs[17] = self.consecutive_losses / 10  # Consecutive losses (normalized)
            obs[18] = min(self.episode_pnl / 100, 5.0)  # Episode P&L (capped)
            obs[19] = min(self.total_pnl / 1000, 5.0)  # Total P&L (capped)
            obs[20] = self.max_drawdown / 500  # Max drawdown (normalized)
            obs[21] = len(self.positions) / self.max_positions  # Position count ratio
            obs[22] = self.last_trade_result / 50  # Last trade result
            obs[23] = (self.recovery_target - self.total_pnl) / 100 if self.in_recovery_mode else 0  # Recovery distance
            obs[24] = self.recovery_attempts / 10  # Recovery attempts
            
            # === POSITION ANALYSIS (8 features) ===
            if self.positions:
                total_volume = sum(pos['volume'] for pos in self.positions)
                total_profit = sum(pos['profit'] for pos in self.positions)
                avg_entry = np.mean([pos['entry_price'] for pos in self.positions])
                
                obs[25] = total_volume / 1.0  # Total volume
                obs[26] = total_profit / 100  # Total unrealized P&L
                obs[27] = len([p for p in self.positions if p['type'] == 'BUY']) / max(len(self.positions), 1)  # Buy ratio
                obs[28] = len([p for p in self.positions if p['type'] == 'SELL']) / max(len(self.positions), 1)  # Sell ratio
                
                if self.historical_data is not None and self.data_index < len(self.historical_data):
                    current_price = self.historical_data.iloc[self.data_index]['close']
                    obs[29] = (current_price - avg_entry) / current_price  # Average position distance
            
            obs[30] = self.winning_trades / max(self.total_trades, 1)  # Win rate
            obs[31] = self.successful_recoveries / max(self.recovery_attempts, 1)  # Recovery success rate
            obs[32] = 0.0  # Reserved for future use
            
            # === TIME & SESSION (4 features) ===
            if self.historical_data is not None and self.data_index < len(self.historical_data):
                current_time = self.historical_data.index[self.data_index]
                obs[33] = current_time.hour / 24  # Hour of day
                obs[34] = current_time.weekday() / 7  # Day of week
                obs[35] = (current_time.hour >= 8 and current_time.hour <= 17) * 1.0  # London session
                obs[36] = (current_time.hour >= 13 and current_time.hour <= 22) * 1.0  # NY session
            
            # === RISK METRICS (3 features) ===
            # Removed drawdown risk calculation - no limits
            obs[37] = 0.0  # Reserved (was drawdown risk)
            obs[38] = self.current_step / 1000  # Episode progress
            obs[39] = 0.0  # Reserved
            
            # Clip values to valid range
            obs = np.clip(obs, -10.0, 10.0)
            
            return obs
            
        except Exception as e:
            print(f"❌ Observation error: {e}")
            return np.zeros(40, dtype=np.float32)

    def _execute_recovery_action(self, action_type, volume, sl_pips, recovery_mode):
        """Execute trading action with recovery logic"""
        try:
            reward = 0.0
            
            if action_type == 0:  # HOLD
                reward = self._handle_hold()
                
            elif action_type == 1:  # BUY
                reward = self._handle_buy(volume, sl_pips)
                
            elif action_type == 2:  # SELL
                reward = self._handle_sell(volume, sl_pips)
                
            elif action_type == 3:  # CLOSE ALL
                reward = self._handle_close_all()
                
            elif action_type == 4:  # RECOVERY ACTION
                reward = self._handle_recovery_action(recovery_mode, volume)
            
            return reward
            
        except Exception as e:
            print(f"❌ Recovery action execution error: {e}")
            return -1.0

    def _handle_hold(self):
        """Handle HOLD action"""
        # Small positive reward for patience when not in crisis
        if not self.in_recovery_mode:
            return 0.1
        # Penalty for holding during recovery mode
        return -0.2

    def _handle_buy(self, volume, sl_pips):
        """Handle BUY action with recovery sizing"""
        # Adjust volume based on recovery level
        adjusted_volume = self._calculate_recovery_volume(volume, 'BUY')
        
        # Get current price for simulation
        if self.historical_data is not None and self.data_index < len(self.historical_data):
            current_price = self.historical_data.iloc[self.data_index]['close']
            entry_price = current_price + 0.0001  # Simulate spread
        else:
            entry_price = 2000.0  # Default for live trading
        
        # Calculate stop loss
        sl_price = entry_price - (sl_pips * 0.01) if sl_pips > 0 else None
        
        # Create position
        position = {
            'type': 'BUY',
            'volume': adjusted_volume,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'entry_time': self.current_step,
            'profit': 0.0
        }
        
        self.positions.append(position)
        self.total_trades += 1
        
        # Reward calculation
        reward = 1.0  # Base reward for taking action
        
        # Bonus for recovery actions
        if self.in_recovery_mode:
            reward += 0.5
        
        # Volume size reward/penalty
        if adjusted_volume > self.base_lot_size * 2:
            reward -= 0.3  # Penalty for oversizing
        
        return reward

    def _handle_sell(self, volume, sl_pips):
        """Handle SELL action with recovery sizing"""
        # Similar to buy but opposite direction
        adjusted_volume = self._calculate_recovery_volume(volume, 'SELL')
        
        if self.historical_data is not None and self.data_index < len(self.historical_data):
            current_price = self.historical_data.iloc[self.data_index]['close']
            entry_price = current_price - 0.0001  # Simulate spread
        else:
            entry_price = 2000.0
        
        sl_price = entry_price + (sl_pips * 0.01) if sl_pips > 0 else None
        
        position = {
            'type': 'SELL',
            'volume': adjusted_volume,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'entry_time': self.current_step,
            'profit': 0.0
        }
        
        self.positions.append(position)
        self.total_trades += 1
        
        reward = 1.0
        if self.in_recovery_mode:
            reward += 0.5
        if adjusted_volume > self.base_lot_size * 2:
            reward -= 0.3
            
        return reward

    def _handle_close_all(self):
        """Handle CLOSE ALL positions"""
        if not self.positions:
            return -0.1  # Small penalty for unnecessary action
        
        total_profit = 0.0
        closed_count = 0
        
        for position in self.positions:
            profit = self._calculate_position_profit(position)
            total_profit += profit
            closed_count += 1
            
            # Track trade results
            if profit > 0:
                self.winning_trades += 1
                self.consecutive_losses = 0
            else:
                self.losing_trades += 1
                self.consecutive_losses += 1
        
        # Clear positions
        self.positions.clear()
        
        # Update P&L
        self.episode_pnl += total_profit
        self.total_pnl += total_profit
        self.last_trade_result = total_profit
        
        # Check if recovery was successful
        if self.in_recovery_mode and total_profit > 0:
            if self.total_pnl >= self.recovery_target:
                self._complete_recovery(success=True)
                return 5.0  # Big reward for successful recovery
        
        # Update drawdown
        if self.total_pnl < self.max_drawdown:
            self.max_drawdown = self.total_pnl
        
        # Reward calculation
        base_reward = 2.0  # Reward for closing positions
        profit_reward = total_profit / 50  # Normalize profit
        
        return base_reward + profit_reward

    def _handle_recovery_action(self, recovery_mode, volume):
        """Handle specialized recovery actions"""
        if not self.in_recovery_mode:
            # Start recovery mode
            self._start_recovery_mode()
            return 1.0
        
        if recovery_mode == 1:  # Aggressive recovery
            # Double the position size
            aggressive_volume = volume * 2
            return self._handle_buy(aggressive_volume, 0)  # No stop loss
        
        elif recovery_mode == 2:  # Conservative recovery
            # Smaller incremental position
            conservative_volume = volume * 0.5
            return self._handle_buy(conservative_volume, 20)  # With stop loss
        
        return 0.0

    def _calculate_recovery_volume(self, base_volume, direction):
        """Calculate position size based on recovery level"""
        if not self.in_recovery_mode:
            return base_volume
        
        # Recovery multiplier increases with level
        multiplier = self.recovery_multiplier ** self.recovery_level
        recovery_volume = base_volume * multiplier
        
        # Cap maximum volume
        max_volume = self.base_lot_size * 10
        return min(recovery_volume, max_volume)

    def _calculate_position_profit(self, position):
        """Calculate current profit for a position"""
        if self.historical_data is None or self.data_index >= len(self.historical_data):
            return 0.0
        
        current_price = self.historical_data.iloc[self.data_index]['close']
        entry_price = position['entry_price']
        volume = position['volume']
        
        if position['type'] == 'BUY':
            pips = (current_price - entry_price) / 0.01
        else:  # SELL
            pips = (entry_price - current_price) / 0.01
        
        # Gold: $1 per pip per 0.01 lot
        profit = pips * volume * 100
        position['profit'] = profit
        
        return profit

    def _update_recovery_state(self):
        """Update recovery state based on current situation"""
        # Check if should enter recovery mode
        if not self.in_recovery_mode and self.total_pnl <= self.recovery_threshold:
            self._start_recovery_mode()
        
        # Update position profits
        for position in self.positions:
            self._calculate_position_profit(position)
        
        # Check recovery completion
        if self.in_recovery_mode and self.total_pnl >= self.recovery_target:
            self._complete_recovery(success=True)

    def _start_recovery_mode(self):
        """Start recovery mode"""
        self.in_recovery_mode = True
        self.recovery_level = 1
        self.recovery_start_balance = self.total_pnl
        self.recovery_target = self.recovery_start_balance + abs(self.recovery_start_balance) + 10  # Recover + small profit
        self.recovery_attempts += 1
        
        print(f"🔄 Recovery mode started: Level {self.recovery_level}, Target: ${self.recovery_target:.2f}")

    def _complete_recovery(self, success=True):
        """Complete recovery mode"""
        self.in_recovery_mode = False
        self.recovery_level = 0
        self.consecutive_losses = 0
        
        if success:
            self.successful_recoveries += 1
            print(f"✅ Recovery successful! P&L: ${self.total_pnl:.2f}")
        else:
            print(f"❌ Recovery failed. P&L: ${self.total_pnl:.2f}")

    def _is_episode_done(self):
        """Check if episode should end"""
        # Removed max drawdown check - no trading limits
        
        # End if data exhausted
        if self.historical_data is not None and self.data_index >= len(self.historical_data) - 1:
            print(f"📈 Episode ended: Data exhausted")
            return True
        
        # End after certain steps
        if self.current_step >= 2000:
            print(f"⏰ Episode ended: Max steps reached")
            return True
        
        return False

    def _get_info(self):
        """Get episode info"""
        return {
            'current_step': self.current_step,
            'data_index': self.data_index,
            'episode_pnl': self.episode_pnl,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'recovery_level': self.recovery_level,
            'in_recovery_mode': self.in_recovery_mode,
            'consecutive_losses': self.consecutive_losses,
            'open_positions': len(self.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'recovery_attempts': self.recovery_attempts,
            'successful_recoveries': self.successful_recoveries,
            'recovery_success_rate': self.successful_recoveries / max(self.recovery_attempts, 1)
        }

    def set_training_mode(self, is_training: bool):
        """Set training mode"""
        self.is_training_mode = is_training
        if is_training:
            print("🎓 Environment set to TRAINING mode")
        else:
            print("🚀 Environment set to LIVE TRADING mode")

    # ===== BACKWARD COMPATIBILITY =====
    # เก็บ methods เดิมไว้เผื่อโค้ดเดิมยังเรียกใช้
    
    def update_market_data(self):
        """Legacy method - not needed with historical data"""
        pass
    
    def get_current_price(self, symbol):
        """Legacy method - use historical data instead"""
        if self.historical_data is not None and self.data_index < len(self.historical_data):
            return self.historical_data.iloc[self.data_index]['close']
        return 2000.0
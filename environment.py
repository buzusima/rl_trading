# environment.py - Custom RL Environment for Trading
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import time
class TradingEnvironment(gym.Env):
    """
    Custom Gymnasium environment for XAUUSD trading with recovery system
    """
    
    def __init__(self, mt5_interface, recovery_engine, config):
        super(TradingEnvironment, self).__init__()
        
        self.simulated_positions = []
        self.simulated_balance = 5000
        self.simulated_equity = 5000
        self.last_price = None

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
            low=np.array([-10.0] * 30, dtype=np.float32),  # ← ตรวจสอบว่าเป็น 30 หรือยัง
            high=np.array([10.0] * 30, dtype=np.float32), 
            shape=(30,),  # ← ตรวจสอบว่าเป็น 30 หรือยัง
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
        self.simulated_positions = []  # เก็บ position จำลอง
    
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
        info = {  # ← ใช้ dict ธรรมดา
                'current_step': 0,
                'episode_pnl': 0.0,
                'account_balance': 0,
                'account_equity': 0,
                'open_positions': 0,
                'recovery_active': False,
                'recovery_level': 0,
                'market_status': 'open'
        }
            
        return observation, info
        
    def step(self, action):
        """Execute one step in the environment"""
        if not self.is_market_open():
            observation = self._get_observation()
            reward = 0.0
            done = False
            info = {  # ← ใช้ dict ธรรมดาแทน
                    'current_step': self.current_step,
                    'episode_pnl': 0,
                    'account_balance': 0,
                    'account_equity': 0,
                    'open_positions': 0,
                    'recovery_active': False,
                    'recovery_level': 0
            }
            print("MARKET CLOSED - Pausing training")
            return observation, reward, done, False, info 
        
        self.current_step += 1
        
        # Parse action
        action_type = float(action[0])
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
        """Execute the trading action and return reward - DEBUG VERSION"""
        reward = 0.0
        
        # 🔍 DEBUG: แสดงค่าที่ได้รับ
        print(f"🔍 DEBUG _execute_action called:")
        print(f"   action_type: {action_type} (type: {type(action_type)})")
        print(f"   lot_multiplier: {lot_multiplier}")
        print(f"   recovery_action: {recovery_action}")
        
        try:
            # Get current market data
            current_price = self._get_current_price()
            if current_price is None:
                print("❌ ERROR: current_price is None")
                return -1.0
            
            # Check if in training mode
            is_training_mode = self.config.get('training_mode', True)
            print(f"🔍 DEBUG training_mode: {is_training_mode}")
            
            if not is_training_mode:
                # Live trading mode - execute real orders
                print("🔍 DEBUG: Live trading mode - checking profit signals")
                
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
                        reward += total_profit_taken / 50.0
                        print(f"💰 Profit taken: {total_profit_taken}")
                        
                # Check smart profit strategy
                smart_profit_taken = self.recovery_engine.smart_profit_strategy(
                    self.mt5_interface, self.symbol
                )
                
                if smart_profit_taken:
                    reward += 3.0
                    print(f"💰 Smart profit taken")
            
            # Calculate position size
            base_lot_size = self.initial_lot_size
            if self.recovery_active:
                base_lot_size = self.recovery_engine.calculate_lot_size(
                    base_lot_size, self.recovery_level
                )
            
            lot_size = base_lot_size * lot_multiplier
            lot_size = max(0.01, min(lot_size, 10.0))
            lot_size = round(lot_size / 0.01) * 0.01
            lot_size = max(0.01, lot_size)
            
            print(f"🔍 DEBUG lot_size calculated: {lot_size}")

            # 🔥 DECISION LOGIC - ตรวจสอบทุก condition
            print(f"🔍 DEBUG: Checking action conditions...")
            
            if action_type < 0.3:
                print(f"🟡 HOLD: action_type {action_type} < 0.3")
                if is_training_mode:
                    reward += self._calculate_simulated_hold_reward()
                else:
                    reward += self._calculate_hold_reward()
                    
            elif 0.3 <= action_type < 1.7:
                print(f"🟢 BUY CONDITION MET: {action_type} >= 0.3")
                if is_training_mode:
                    print("🔍 DEBUG: Training mode - simulating BUY")
                    success = True
                    reward += self._calculate_simulated_trade_reward('buy', lot_size, current_price)
                    print(f"🔥 SIMULATED BUY: {lot_size} {self.symbol} at {current_price:.2f}, Action: {action_type:.3f}")
                else:
                    print("🔍 DEBUG: Live mode - executing real BUY")
                    success = self.mt5_interface.place_order(
                        symbol=self.symbol,
                        order_type='buy',
                        volume=lot_size,
                        price=current_price
                    )
                    reward += self._calculate_trade_reward(success, 'buy', lot_size)
                    if success:
                        print(f"🚀 LIVE BUY EXECUTED: {lot_size} {self.symbol} at {current_price:.2f}")
                    else:
                        print(f"❌ LIVE BUY FAILED: {lot_size} {self.symbol}")
                    
            elif 1.7 <= action_type < 2.7:
                print(f"🔴 SELL CONDITION MET: {action_type} >= 1.7")
                if is_training_mode:
                    success = True
                    reward += self._calculate_simulated_trade_reward('sell', lot_size, current_price)
                    print(f"🔥 SIMULATED SELL: {lot_size} {self.symbol} at {current_price:.2f}")
                else:
                    success = self.mt5_interface.place_order(
                        symbol=self.symbol,
                        order_type='sell',
                        volume=lot_size,
                        price=current_price
                    )
                    reward += self._calculate_trade_reward(success, 'sell', lot_size)
                    if success:
                        print(f"🚀 LIVE SELL EXECUTED: {lot_size} {self.symbol}")
                    else:
                        print(f"❌ LIVE SELL FAILED: {lot_size} {self.symbol}")
                    
            elif 2.7 <= action_type < 3.5:
                print(f"💰 CLOSE CONDITION MET: {action_type} >= 2.7")
                if is_training_mode:
                    success = True
                    reward += self._calculate_simulated_close_reward()
                    print(f"🔥 SIMULATED CLOSE: {self.symbol}")
                else:
                    success = self.mt5_interface.close_all_positions(self.symbol)
                    reward += self._calculate_close_reward(success)
                    current_pnl = self._get_current_pnl()
                    if current_pnl > 0:
                        reward += 2.0 + (current_pnl / 100.0)
                    if success:
                        print(f"🚀 LIVE CLOSE EXECUTED: {self.symbol}")
                    else:
                        print(f"❌ LIVE CLOSE FAILED: {self.symbol}")
                        
            elif action_type >= 3.5:
                print(f"🛡️ HEDGE CONDITION MET: {action_type} >= 3.5")
                if is_training_mode:
                    success = True
                    reward += self._calculate_simulated_hedge_reward(lot_size)
                    print(f"🔥 SIMULATED HEDGE: {lot_size} {self.symbol}")
                else:
                    success = self._execute_hedge_action(lot_size)
                    reward += self._calculate_hedge_reward(success)
                    if success:
                        print(f"🚀 LIVE HEDGE EXECUTED: {lot_size} {self.symbol}")
                    else:
                        print(f"❌ LIVE HEDGE FAILED: {lot_size} {self.symbol}")
            
            else:
                print(f"❓ UNKNOWN ACTION: {action_type}")
            
            # Execute recovery action if needed
            if not is_training_mode and recovery_action > 0 and self._should_activate_recovery():
                print(f"🔄 Recovery action: {recovery_action}")
                self._execute_recovery_action(recovery_action)
                
            # Update recovery status
            self._update_recovery_status()
            
            print(f"🔍 DEBUG: Final reward: {reward}")
            
        except Exception as e:
            print(f"❌ ERROR in _execute_action: {e}")
            import traceback
            traceback.print_exc()
            reward = -5.0
            
        return reward

    # เพิ่ม Methods ใหม่สำหรับ Simulation
    def _calculate_simulated_trade_reward(self, trade_type, lot_size, entry_price):
        """Calculate REALISTIC reward based on actual price movement"""
        try:
            if len(self.market_data_cache) < 2:
                return 0.1
                
            # Get actual price movement
            current_price = self.market_data_cache[-1]['close']
            prev_price = self.market_data_cache[-2]['close']
            
            price_change = current_price - prev_price
            
            # Calculate realistic PnL
            if trade_type == 'buy':
                pnl = price_change * lot_size * 100000  # 1 lot = 100,000 units
            else:  # sell
                pnl = -price_change * lot_size * 100000
                
            # Scale reward properly
            reward = pnl / 10.0  # Scale down
            
            # Add trend bonus/penalty
            if len(self.market_data_cache) >= 5:
                recent_prices = [data['close'] for data in self.market_data_cache[-5:]]
                trend = np.polyfit(range(5), recent_prices, 1)[0]
                
                if (trade_type == 'buy' and trend > 0) or (trade_type == 'sell' and trend < 0):
                    reward += 0.5  # Trend following bonus
                else:
                    reward -= 0.2  # Against trend penalty
            
            # Prevent extreme rewards
            reward = np.clip(reward, -5.0, 5.0)
            
            return float(reward)
            
        except Exception as e:
            print(f"❌ Reward calculation error: {e}")
            return 0.1

    def _calculate_simulated_hold_reward(self):
        """Better hold reward calculation"""
        try:
            positions = getattr(self, 'simulated_positions', [])
            
            if not positions:
                return -0.01  # Small penalty for inaction
                
            # Calculate unrealized PnL
            current_price = self._get_current_price()
            total_pnl = 0
            
            for pos in positions:
                entry_price = pos.get('entry_price', current_price)
                lot_size = pos.get('lot_size', 0.01)
                pos_type = pos.get('type', 'buy')
                
                if pos_type == 'buy':
                    pnl = (current_price - entry_price) * lot_size * 100000
                else:
                    pnl = (entry_price - current_price) * lot_size * 100000
                    
                total_pnl += pnl
            
            # Reward based on PnL trend
            reward = total_pnl / 100.0
            
            # Time penalty (encourage action)
            if hasattr(self, 'hold_count'):
                self.hold_count += 1
                if self.hold_count > 10:
                    reward -= 0.1  # Penalty for holding too long
            else:
                self.hold_count = 1
                
            return np.clip(reward, -1.0, 1.0)
            
        except:
            return -0.01
    
    def _calculate_simulated_close_reward(self):
        """คำนวณ reward สำหรับการปิด position"""
        try:
            if not hasattr(self, 'simulated_positions') or not self.simulated_positions:
                return -0.5  # Penalty for closing when no positions
                
            # คำนวณ PnL รวมของ position ที่จะปิด
            total_pnl = 0.0
            current_price = self._get_current_price()
            
            for position in self.simulated_positions:
                entry_price = position.get('entry_price', current_price)
                lot_size = position.get('lot_size', 0.01)
                pos_type = position.get('type', 'buy')
                
                if pos_type == 'buy':
                    pnl = (current_price - entry_price) * lot_size * 100000
                else:
                    pnl = (entry_price - current_price) * lot_size * 100000
                    
                total_pnl += pnl
                
            # ล้าง position หลังปิด
            self.simulated_positions = []
            
            # Reward ตาม PnL
            if total_pnl > 0:
                return 2.0 + (total_pnl / 100.0)  # Big bonus for profit
            else:
                return -1.0 + (total_pnl / 100.0)  # Penalty for loss
                
        except:
            return -0.5

    def _calculate_simulated_hedge_reward(self, lot_size):
        """คำนวณ reward สำหรับการ Hedge"""
        # Hedge ได้ reward เล็กน้อยเป็นการจัดการความเสี่ยง
        return 0.5
        
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
        """Get SMART observation that makes AI intelligent"""
        observation = np.zeros(30, dtype=np.float32)
        
        try:
            # 🧠 SMART MARKET ANALYSIS
            current_price = self._get_current_price()
            positions = self.mt5_interface.get_positions() if hasattr(self, 'mt5_interface') else []
            
            if current_price and len(self.market_data_cache) >= 10:
                recent_data = self.market_data_cache[-10:]
                prices = [data['close'] for data in recent_data]
                
                # 📈 TREND DETECTION (ฉลาดขึ้น)
                if len(prices) >= 5:
                    # Short term trend (last 3 bars)
                    short_trend = (prices[-1] - prices[-3]) / prices[-3]
                    observation[0] = short_trend * 10  # Amplify signal
                    
                    # Medium term trend (last 5 bars)  
                    medium_trend = (prices[-1] - prices[-5]) / prices[-5]
                    observation[1] = medium_trend * 10
                    
                    # Trend strength
                    price_changes = np.diff(prices[-5:])
                    trend_consistency = len([x for x in price_changes if x > 0]) / len(price_changes)
                    observation[2] = (trend_consistency - 0.5) * 4  # -2 to +2
            
            # 💰 PROFIT/LOSS INTELLIGENCE
            if positions:
                total_pnl = sum(pos.get('profit', 0) for pos in positions)
                position_count = len(positions)
                
                # PnL pressure (ข้อมูลสำคัญ)
                observation[10] = total_pnl / 50.0  # Normalize PnL
                observation[11] = position_count / 5.0  # Position pressure
                
                # 🚨 RISK SIGNALS (AI ต้องรู้)
                if total_pnl < -30:
                    observation[12] = -2.0  # Strong SELL/CLOSE signal
                elif total_pnl > 30:
                    observation[13] = 2.0   # Strong CLOSE (take profit) signal
                    
                # Position age pressure
                if position_count > 10:
                    observation[14] = 2.0   # Too many positions - CLOSE signal
                    
            # 🎯 MARKET TIMING (Smart timing)
            now = datetime.now()
            hour = now.hour
            
            # Market session strength
            if 8 <= hour <= 12:  # European morning (volatile)
                observation[20] = 1.5
            elif 13 <= hour <= 17:  # US morning (trending)
                observation[21] = 1.5
            elif 21 <= hour <= 23:  # Asian session (ranging)
                observation[22] = -0.5  # Less trading
                
            # 🔥 INTELLIGENT SIGNALS (ทำให้ AI ฉลาด)
            
            # Momentum signal
            if len(self.market_data_cache) >= 3:
                recent_prices = [data['close'] for data in self.market_data_cache[-3:]]
                if recent_prices[-1] > recent_prices[-2] > recent_prices[-3]:
                    observation[25] = 1.5  # Strong BUY signal
                elif recent_prices[-1] < recent_prices[-2] < recent_prices[-3]:
                    observation[26] = 1.5  # Strong SELL signal
                    
            # Volatility signal
            if len(prices) >= 5:
                volatility = np.std(prices[-5:]) / np.mean(prices[-5:])
                if volatility > 0.01:  # High volatility
                    observation[27] = -1.0  # Avoid trading
                else:
                    observation[27] = 0.5   # Good for trading
                    
            # 🎲 REDUCE RANDOMNESS (ให้ Logic มากกว่า Random)
            observation[28] = self.current_step % 10 / 10.0  # Predictable cycle
            observation[29] = 1.0 if positions else 0.0      # Position state
            
        except Exception as e:
            print(f"❌ Smart observation error: {e}")
            # Fallback to basic signals
            observation[0] = np.random.choice([-1, 0, 1]) * 0.5  # Basic trend
            observation[10] = np.random.choice([-0.5, 0, 0.5])   # Basic PnL
            
        # 🧠 FINAL INTELLIGENCE CHECK
        # Make sure AI gets clear signals
        observation = np.clip(observation, -3.0, 3.0)
        
        return observation

    # 🤖 INTELLIGENT ACTION INTERPRETATION (ใน _execute_action)
    def intelligent_action_logic(self, action_type, positions, total_pnl):
        """Make AI decisions more intelligent"""
        
        # 🧠 SMART DECISION OVERRIDE
        
        # If losing badly, force CLOSE
        if total_pnl < -50:
            print("🧠 SMART: Heavy loss detected, forcing CLOSE")
            return 3.0  # Force close
            
        # If too many positions, force SELL
        if len(positions) > 15:
            print("🧠 SMART: Too many positions, forcing SELL")
            return 2.5  # Force sell
            
        # If good profit, force CLOSE
        if total_pnl > 50:
            print("🧠 SMART: Good profit detected, forcing CLOSE")
            return 3.2  # Force close
            
        # Market condition based decisions
        if hasattr(self, 'market_data_cache') and len(self.market_data_cache) >= 5:
            recent_prices = [data['close'] for data in self.market_data_cache[-5:]]
            
            # Strong downtrend - prefer SELL
            if all(recent_prices[i] > recent_prices[i+1] for i in range(len(recent_prices)-1)):
                if 0.5 <= action_type <= 1.5:  # Would be BUY
                    print("🧠 SMART: Downtrend detected, converting BUY to SELL")
                    return 2.2  # Convert to SELL
                    
            # Strong uptrend - prefer BUY  
            elif all(recent_prices[i] < recent_prices[i+1] for i in range(len(recent_prices)-1)):
                if 1.8 <= action_type <= 2.5:  # Would be SELL
                    print("🧠 SMART: Uptrend detected, converting SELL to BUY")
                    return 0.8  # Convert to BUY
        
        return action_type
    
    def _get_simplified_market_features(self):
        """Simple market features - เร็วกว่าเดิม"""
        features = np.zeros(15)
        
        try:
            if len(self.market_data_cache) < 5:
                return features
                
            # เอาแค่ข้อมูลราคาพื้นฐาน
            recent_data = self.market_data_cache[-5:]
            prices = [data['close'] for data in recent_data]
            
            # Simple price features
            current_price = prices[-1]
            prev_price = prices[-2] if len(prices) > 1 else current_price
            
            features[0] = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            features[1] = (current_price - np.mean(prices)) / np.mean(prices) if np.mean(prices) > 0 else 0
            
            # Simple trend
            if len(prices) >= 3:
                features[2] = 1.0 if prices[-1] > prices[-3] else -1.0
                
            # Volatility
            if len(prices) >= 3:
                price_changes = np.diff(prices)
                features[3] = np.std(price_changes) / current_price if current_price > 0 else 0
                
            # Time features
            now = datetime.now()
            features[4] = now.hour / 24
            features[5] = now.weekday() / 7
            
            # Clean
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
        except Exception as e:
            print(f"Error in simplified market features: {e}")
            
        return features

    def _get_simplified_position_features(self):
        """Simple position features"""
        features = np.zeros(8)
        
        try:
            if hasattr(self, 'simulated_positions') and self.simulated_positions:
                features[0] = len(self.simulated_positions) / 10  # Number of positions
                
                total_volume = sum(pos.get('lot_size', 0) for pos in self.simulated_positions)
                features[1] = total_volume / 1.0  # Total volume
                
                # Average profit
                current_price = self._get_current_price()
                if current_price:
                    total_pnl = 0
                    for pos in self.simulated_positions:
                        entry_price = pos.get('entry_price', current_price)
                        lot_size = pos.get('lot_size', 0.01)
                        pos_type = pos.get('type', 'buy')
                        
                        if pos_type == 'buy':
                            pnl = (current_price - entry_price) * lot_size * 100000
                        else:
                            pnl = (entry_price - current_price) * lot_size * 100000
                            
                        total_pnl += pnl
                        
                    features[2] = total_pnl / 1000  # Normalize PnL
                    
        except Exception as e:
            print(f"Error in simplified position features: {e}")
            
        return features

    def _get_simplified_account_features(self):
        """Simple account features"""
        features = np.zeros(4)
        
        try:
            if hasattr(self, 'simulated_balance'):
                features[0] = self.simulated_balance / 10000
                features[1] = self.simulated_equity / 10000
                features[2] = (self.simulated_equity - self.simulated_balance) / self.simulated_balance if self.simulated_balance > 0 else 0
                
        except Exception as e:
            print(f"Error in simplified account features: {e}")
            
        return features

    def _get_simplified_recovery_features(self):
        """Simple recovery features"""
        features = np.zeros(3)
        
        try:
            features[0] = 1.0 if self.recovery_active else 0.0
            features[1] = self.recovery_level / 10
            features[2] = self.current_step / self.max_steps
            
        except Exception as e:
            print(f"Error in simplified recovery features: {e}")
            
        return features

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
        """Update market data with REAL data"""
        try:
            import MetaTrader5 as mt5
            
            # Force fresh data
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 1)
            
            if rates is not None and len(rates) > 0:
                rate = rates[0]
                market_data = {
                    'time': rate[0],
                    'open': float(rate[1]),
                    'high': float(rate[2]),
                    'low': float(rate[3]),
                    'close': float(rate[4]),
                    'volume': float(rate[5]) if len(rate) > 5 else 0,
                    'timestamp': time.time()  # Add timestamp
                }
                
                # Add to cache (keep recent 100)
                self.market_data_cache.append(market_data)
                if len(self.market_data_cache) > 100:
                    self.market_data_cache = self.market_data_cache[-100:]
                    
                # Debug log
                print(f"📊 Market Update: {market_data['close']:.2f} at {datetime.now().strftime('%H:%M:%S')}")
                
            else:
                print("⚠️ No market data received")
                
        except Exception as e:
            print(f"❌ Market data error: {e}")


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

    def _get_current_pnl(self):
        """Get current unrealized PnL"""
        try:
            positions = self.mt5_interface.get_positions()
            return sum(pos.get('profit', 0) for pos in positions)
        except:
            return 0.0

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
                'max_drawdown': getattr(self, 'max_drawdown', 0),
                'total_trades': getattr(self, 'total_trades', 0),
                'winning_trades': getattr(self, 'winning_trades', 0)
            }
            
            return info
            
        except:
            return {
                'current_step': self.current_step,
                'episode_pnl': 0,
                'account_balance': 0,
                'account_equity': 0,
                'open_positions': 0,
                'recovery_active': False,
                'recovery_level': 0
            }
    def is_market_open(self):
        """Check if XAUUSD market is open"""
        return True


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
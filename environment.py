# environment.py - NEW Clean Trading Environment with State Machine
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime, timedelta
from enum import Enum

class TradingState(Enum):
    """Trading State Machine States"""
    ANALYZE = "ANALYZE"        # วิเคราะห์หาโอกาส
    ENTRY = "ENTRY"           # เข้าไม้
    MONITOR = "MONITOR"       # ดูแลไม้ที่เปิดไว้  
    RECOVERY = "RECOVERY"     # แก้ไม้ขาดทุน
    EXIT = "EXIT"             # ปิดไม้

class TradingStatusManager:
    """จัดการ Trading Status สำหรับ GUI"""
    
    def __init__(self, gui_callback=None):
        self.gui_callback = gui_callback
        self.current_phase = "WAITING"
        self.total_positions = 0
        self.total_pnl = 0
        self.last_status_log = ""
    
    def update_status(self, state, positions, pnl):
        """อัพเดท status และส่งไป GUI"""
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            self.total_positions = len(positions) if positions else 0
            self.total_pnl = pnl
            
            # กำหนด status message ตาม state
            if state == TradingState.ANALYZE:
                message = f"🔍 วิเคราะห์ตลาด XAUUSD - หาจังหวะเข้าไม้..."
            elif state == TradingState.ENTRY:
                message = f"🎯 เข้าไม้ - รอการ execute..."
            elif state == TradingState.MONITOR:
                message = f"👀 ดูแลไม้ - {self.total_positions} ไม้ | PnL: ${self.total_pnl:.2f}"
            elif state == TradingState.RECOVERY:
                message = f"🛠️ โหมดแก้ไม้ - ขาดทุน ${abs(self.total_pnl):.2f} | รอแก้ไม้..."
            elif state == TradingState.EXIT:
                message = f"💰 ปิดไม้ - PnL: ${self.total_pnl:.2f}"
            else:
                message = f"📊 สถานะ: {state} | {self.total_positions} ไม้ | ${self.total_pnl:.2f}"
            
            # ส่งไป GUI ถ้าเปลี่ยนแปลง
            if message != self.last_status_log:
                self._send_to_gui(f"[{current_time}] {message}")
                self.last_status_log = message
                
        except Exception as e:
            self._send_to_gui(f"❌ Status update error: {e}")
    
    def _send_to_gui(self, message):
        """ส่ง message ไป GUI Log"""
        try:
            if self.gui_callback:
                self.gui_callback(message, level="INFO")
            print(f"🎯 STATUS: {message}")
        except Exception as e:
            print(f"GUI callback error: {e}")

class TradingEnvironment(gym.Env):
    """
    Clean Trading Environment with State Machine
    Compatible with existing GUI and systems
    """
    
    def __init__(self, mt5_interface, recovery_engine, config):
        super(TradingEnvironment, self).__init__()
        
        print(f"🏗️ Initializing NEW Trading Environment...")
        
        # Core components (from original)
        self.mt5_interface = mt5_interface
        self.recovery_engine = recovery_engine
        self.config = config
        
        # Trading parameters (from original)
        self.symbol = config.get('symbol', 'XAUUSD')
        self.initial_lot_size = config.get('initial_lot_size', 0.01)
        self.max_positions = config.get('max_positions', 10)
        
        # Environment parameters (from original)
        self.lookback_window = 100
        self.max_steps = config.get('max_steps', 200)  # Shorter episodes
        
        # Gym spaces (matching original)
        self.observation_space = spaces.Box(
            low=np.array([-10.0] * 30, dtype=np.float32),
            high=np.array([10.0] * 30, dtype=np.float32), 
            shape=(30,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=np.array([0, 0.5, 0]),
            high=np.array([4, 3.0, 3]),
            dtype=np.float32
        )
        
        # Episode tracking (from original)
        self.current_step = 0
        self.episode_start_time = None
        self.episode_trades = []
        self.episode_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Market data cache (from original)
        self.market_data_cache = []
        self.last_update_time = None
        
        # ✅ NEW: State Machine
        self.trading_state = TradingState.ANALYZE
        self.entry_time = None
        self.entry_price = None
        self.target_profit = config.get('target_profit', 10.0)  # $10
        self.stop_loss_amount = config.get('stop_loss', 5.0)    # $5
        self.max_monitor_time = config.get('max_monitor_time', 300)  # 5 minutes
        self.state_start_time = time.time()
        
        # ✅ NEW: Enhanced tracking
        self.consecutive_holds = 0
        self.recent_actions = []
        self.state_history = []
        
        # Portfolio manager integration (from original)
        from portfolio_manager import AIPortfolioManager
        self.portfolio_manager = AIPortfolioManager(config)
        
        # Training mode
        self.is_training_mode = config.get('training_mode', True)
        
        # GUI integration (from original)
        self.status_manager = TradingStatusManager(gui_callback=self.gui_log_callback)
        self.gui_instance = None
        
        print(f"✅ Environment initialized:")
        print(f"   - Symbol: {self.symbol}")
        print(f"   - Max Steps: {self.max_steps}")
        print(f"   - Training Mode: {self.is_training_mode}")
        print(f"   - Initial State: {self.trading_state.value}")

    def gui_log_callback(self, message, level="INFO"):
        """Callback สำหรับส่ง log ไป GUI (from original)"""
        try:
            if hasattr(self, 'gui_instance') and self.gui_instance:
                self.gui_instance.log_message(message, level)
        except:
            pass

    def reset(self, seed=None, options=None):
        """Reset environment (enhanced from original)"""
        super().reset(seed=seed)
        
        print(f"🔄 Resetting environment...")
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_start_time = datetime.now()
        self.episode_trades = []
        self.episode_pnl = 0.0
        self.max_drawdown = 0.0
        
        # Reset state machine
        self.trading_state = TradingState.ANALYZE
        self.entry_time = None
        self.entry_price = None
        self.state_start_time = time.time()
        self.consecutive_holds = 0
        self.recent_actions = []
        self.state_history = []
        
        # Get initial account info
        try:
            account_info = self.mt5_interface.get_account_info()
            if account_info:
                self.peak_equity = account_info.get('equity', 0)
        except:
            self.peak_equity = 1000  # Default
        
        # Reset recovery system
        if self.recovery_engine:
            self.recovery_engine.reset()
        
        # Initialize portfolio manager
        if self.portfolio_manager:
            self.portfolio_manager.initialize_portfolio(self.mt5_interface)
        
        # Update market data
        self.update_market_data()
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            'current_step': 0,
            'trading_state': self.trading_state.value,
            'episode_pnl': 0.0,
            'account_balance': 0,
            'account_equity': 0,
            'open_positions': 0
        }
        
        print(f"✅ Environment reset complete - State: {self.trading_state.value}")
        return observation, info

    def step(self, action):
        """Execute one step with state machine (enhanced from original)"""
        
        self.current_step += 1
        
        # Update portfolio (only in live mode)
        if not self.is_training_mode and self.portfolio_manager:
            self.portfolio_manager.update_portfolio_status(self.mt5_interface)
        
        # Parse action (from original)
        action_type = float(action[0])
        lot_multiplier = float(action[1])
        recovery_action = int(action[2])
        
        # Update market data
        self.update_market_data()
        
        # ✅ NEW: Execute action with state machine
        reward = self._execute_action_with_state_machine(action_type, lot_multiplier, recovery_action)
        
        # Get new observation
        observation = self._get_observation()
        
        # Update GUI status
        positions = self._get_positions()
        total_pnl = sum(pos.get('profit', 0) for pos in positions) if positions else 0
        self.status_manager.update_status(self.trading_state, positions, total_pnl)
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Additional info (enhanced from original)
        info = self._get_info()
        
        return observation, reward, done, False, info

    def _execute_action_with_state_machine(self, action_type, lot_multiplier, recovery_action):
        """NEW: Execute action based on current state"""
        
        positions = self._get_positions()
        total_pnl = sum(pos.get('profit', 0) for pos in positions) if positions else 0
        current_time = time.time()
        
        print(f"")
        print(f"🎯 === STATE MACHINE EXECUTION ===")
        print(f"   Current State: {self.trading_state.value}")
        print(f"   Action Input: {action_type:.3f}")
        print(f"   Positions: {len(positions)}")
        print(f"   PnL: ${total_pnl:.2f}")
        print(f"   Time in State: {current_time - self.state_start_time:.1f}s")
        
        # Record action
        self.recent_actions.append(action_type)
        if len(self.recent_actions) > 10:
            self.recent_actions = self.recent_actions[-10:]
        
        reward = 0.0
        
        # State machine logic
        if self.trading_state == TradingState.ANALYZE:
            reward = self._handle_analyze_state(action_type, positions, total_pnl)
            
        elif self.trading_state == TradingState.ENTRY:
            reward = self._handle_entry_state(action_type, positions, total_pnl)
            
        elif self.trading_state == TradingState.MONITOR:
            reward = self._handle_monitor_state(action_type, positions, total_pnl, current_time)
            
        elif self.trading_state == TradingState.RECOVERY:
            reward = self._handle_recovery_state(action_type, positions, total_pnl)
            
        elif self.trading_state == TradingState.EXIT:
            reward = self._handle_exit_state(action_type, positions, total_pnl)
        
        print(f"🎁 STATE REWARD: {reward:.3f}")
        print(f"🎯 === END STATE MACHINE ===")
        print(f"")
        
        return reward

    def _handle_analyze_state(self, action_type, positions, total_pnl):
        """Handle ANALYZE state"""
        print(f"📊 ANALYZE: Looking for entry opportunities...")
        
        # In analyze state, only accept entry signals
        if 0.3 <= action_type < 2.7:  # BUY or SELL
            entry_quality = self._analyze_entry_signal(action_type)
            
            if entry_quality > 0.6:  # Good entry signal
                self._transition_to_state(TradingState.ENTRY)
                self.entry_price = self._get_current_price()
                return 2.0 + entry_quality  # Reward good analysis
            else:
                return -0.1  # Penalty for bad analysis
        else:
            self.consecutive_holds += 1
            if self.consecutive_holds > 10:  # Too much analysis
                return -0.5  # Force action
            return -0.05  # Small penalty for not taking action

    def _handle_entry_state(self, action_type, positions, total_pnl):
        """Handle ENTRY state"""
        print(f"🎯 ENTRY: Executing position entry...")
        
        success = self._execute_entry_order(action_type)
        
        if success:
            self._transition_to_state(TradingState.MONITOR)
            return 3.0  # Big reward for successful entry
        else:
            self._transition_to_state(TradingState.ANALYZE)
            return -1.0  # Penalty for failed entry

    def _handle_monitor_state(self, action_type, positions, total_pnl, current_time):
        """Handle MONITOR state"""
        print(f"👀 MONITOR: Watching position performance...")
        
        # Check if we need to exit
        exit_reason = self._check_exit_conditions(total_pnl, current_time)
        
        if exit_reason == "PROFIT_TARGET":
            self._transition_to_state(TradingState.EXIT)
            return 5.0  # Excellent profit target hit
            
        elif exit_reason == "STOP_LOSS":
            self._transition_to_state(TradingState.EXIT)
            return 1.0  # Good stop loss
            
        elif exit_reason == "BIG_LOSS":
            self._transition_to_state(TradingState.RECOVERY)
            return -1.0  # Need recovery
            
        elif exit_reason == "TIMEOUT":
            self._transition_to_state(TradingState.EXIT)
            return 0.5  # Neutral timeout exit
            
        # Manual exit action
        elif 2.7 <= action_type < 3.5:  # CLOSE action
            self._transition_to_state(TradingState.EXIT)
            return 2.0  # Good manual exit
            
        else:
            return 0.1  # Small reward for monitoring

    def _handle_recovery_state(self, action_type, positions, total_pnl):
        """Handle RECOVERY state"""
        print(f"🛡️ RECOVERY: Attempting to recover from loss...")
        
        if action_type >= 3.5:  # HEDGE action
            success = self._execute_recovery_action(action_type)
            if success:
                self._transition_to_state(TradingState.MONITOR)
                return 2.0  # Good recovery attempt
            else:
                return -1.0  # Failed recovery
                
        elif 2.7 <= action_type < 3.5:  # CLOSE (cut loss)
            self._transition_to_state(TradingState.EXIT)
            return 1.5  # Brave decision to cut loss
            
        else:
            return -0.5  # Wrong action in recovery

    def _handle_exit_state(self, action_type, positions, total_pnl):
        """Handle EXIT state"""
        print(f"🚪 EXIT: Closing all positions...")
        
        success = self._execute_exit_orders()
        
        if success:
            self._transition_to_state(TradingState.ANALYZE)
            
            # Reward based on final result
            if total_pnl > 0:
                return 4.0 + (total_pnl / 10.0)  # Profit bonus
            elif total_pnl > -self.stop_loss_amount:
                return 1.0  # Acceptable small loss
            else:
                return -0.5  # Big loss penalty
        else:
            return -2.0  # Failed exit

    def _transition_to_state(self, new_state):
        """Transition to new state"""
        old_state = self.trading_state
        self.trading_state = new_state
        self.state_start_time = time.time()
        self.consecutive_holds = 0  # Reset hold counter
        
        # Record state transition
        self.state_history.append({
            'from_state': old_state.value,
            'to_state': new_state.value,
            'timestamp': datetime.now(),
            'step': self.current_step
        })
        
        print(f"🔄 STATE TRANSITION: {old_state.value} → {new_state.value}")

    def _analyze_entry_signal(self, action_type):
        """Analyze entry signal quality"""
        try:
            if len(self.market_data_cache) < 3:
                return 0.3  # Weak signal
            
            # Get recent prices
            prices = [data['close'] for data in self.market_data_cache[-3:]]
            
            if action_type < 1.7:  # BUY signal
                # Check for uptrend
                if prices[-1] > prices[-2] > prices[-3]:
                    return 0.8  # Strong buy signal
                elif prices[-1] > prices[-3]:
                    return 0.6  # Moderate buy signal
                else:
                    return 0.2  # Weak buy signal
                    
            else:  # SELL signal
                # Check for downtrend
                if prices[-1] < prices[-2] < prices[-3]:
                    return 0.8  # Strong sell signal
                elif prices[-1] < prices[-3]:
                    return 0.6  # Moderate sell signal
                else:
                    return 0.2  # Weak sell signal
                    
        except:
            return 0.3  # Default weak signal

    def _check_exit_conditions(self, total_pnl, current_time):
        """Check if we should exit position"""
        
        # Profit target
        if total_pnl >= self.target_profit:
            return "PROFIT_TARGET"
        
        # Stop loss
        if total_pnl <= -self.stop_loss_amount:
            return "STOP_LOSS"
        
        # Big loss (need recovery)
        if total_pnl <= -(self.stop_loss_amount * 3):
            return "BIG_LOSS"
        
        # Timeout
        if self.entry_time and (current_time - self.entry_time) > self.max_monitor_time:
            return "TIMEOUT"
        
        return "CONTINUE"

    def _execute_entry_order(self, action_type):
        """Execute entry order"""
        if self.is_training_mode:
            print(f"🎓 SIMULATED ENTRY")
            self.entry_time = time.time()
            return True
        else:
            # Real trading
            current_price = self._get_current_price()
            order_type = 'buy' if action_type < 1.7 else 'sell'
            
            success = self.mt5_interface.place_order(
                symbol=self.symbol,
                order_type=order_type,
                volume=self.initial_lot_size,
                price=current_price
            )
            
            if success:
                self.entry_time = time.time()
                
            print(f"🚀 LIVE ENTRY: {order_type.upper()} {'SUCCESS' if success else 'FAILED'}")
            return success

    def _execute_recovery_action(self, action_type):
        """Execute recovery action"""
        if self.is_training_mode:
            print(f"🎓 SIMULATED RECOVERY")
            return True
        else:
            # Real recovery using recovery engine
            if self.recovery_engine:
                try:
                    positions = self._get_positions()
                    total_pnl = sum(pos.get('profit', 0) for pos in positions) if positions else 0
                    
                    account_info = self.mt5_interface.get_account_info()
                    current_equity = account_info.get('equity', 0) if account_info else 0
                    
                    success = self.recovery_engine.activate_recovery(
                        symbol=self.symbol,
                        mt5_interface=self.mt5_interface,
                        current_pnl=total_pnl,
                        current_equity=current_equity,
                        recovery_type='combined'
                    )
                    
                    print(f"🚀 LIVE RECOVERY: {'SUCCESS' if success else 'FAILED'}")
                    return success
                except:
                    return False
            return False

    def _execute_exit_orders(self):
        """Execute exit orders"""
        if self.is_training_mode:
            print(f"🎓 SIMULATED EXIT")
            return True
        else:
            success = self.mt5_interface.close_all_positions(self.symbol)
            print(f"🚀 LIVE EXIT: {'SUCCESS' if success else 'FAILED'}")
            return success

    def _get_positions(self):
        """Get current positions (from original)"""
        try:
            if hasattr(self, 'mt5_interface') and self.mt5_interface:
                return self.mt5_interface.get_positions()
            return []
        except:
            return []

    def _get_current_price(self):
        """Get current market price (from original)"""
        try:
            if self.market_data_cache:
                return self.market_data_cache[-1]['close']
            return 2000.0  # Default XAUUSD price
        except:
            return 2000.0

    def _get_observation(self):
        """Get observation (simplified from original)"""
        observation = np.zeros(30, dtype=np.float32)
        
        try:
            # Market features (0-9)
            if len(self.market_data_cache) >= 3:
                prices = [data['close'] for data in self.market_data_cache[-3:]]
                
                # Simple trend
                if len(prices) >= 2:
                    trend = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
                    observation[0] = float(trend * 10)
                
                # Current price normalized
                current_price = prices[-1]
                observation[1] = float(current_price / 2000.0)  # Normalize around 2000
            
            # Position features (10-14)
            positions = self._get_positions()
            total_pnl = sum(pos.get('profit', 0) for pos in positions) if positions else 0
            
            observation[10] = float(len(positions))
            observation[11] = float(total_pnl / 100.0)  # Normalize PnL
            
            # State features (15-19)
            observation[15] = float(self.trading_state.value == "ANALYZE")
            observation[16] = float(self.trading_state.value == "ENTRY")
            observation[17] = float(self.trading_state.value == "MONITOR")
            observation[18] = float(self.trading_state.value == "RECOVERY")
            observation[19] = float(self.trading_state.value == "EXIT")
            
            # Time features (20-24)
            now = datetime.now()
            observation[20] = float(now.hour / 24.0)
            observation[21] = float(now.minute / 60.0)
            observation[22] = float(self.current_step / self.max_steps)
            
            # Fill remaining with small values
            for i in range(25, 30):
                observation[i] = np.random.normal(0, 0.01)
            
            # Safety checks
            observation = np.clip(observation, -10.0, 10.0)
            observation = np.nan_to_num(observation, nan=0.0)
            
        except Exception as e:
            print(f"Observation error: {e}")
            # Return safe default
            observation = np.random.normal(0, 0.1, size=30)
            observation = np.clip(observation, -1.0, 1.0)
        
        return observation.astype(np.float32)

    def update_market_data(self):
        """Update market data (simplified from original)"""
        try:
            if hasattr(self, 'mt5_interface') and self.mt5_interface:
                # Get basic rate data
                rates = self.mt5_interface.get_rates(self.symbol, 1, 10)  # M1, 10 bars
                
                if rates is not None and len(rates) > 0:
                    # Convert to cache format
                    new_data = []
                    for rate in rates:
                        new_data.append({
                            'time': rate[0],
                            'open': float(rate[1]),
                            'high': float(rate[2]),
                            'low': float(rate[3]),
                            'close': float(rate[4]),
                            'volume': float(rate[5]) if len(rate) > 5 else 0
                        })
                    
                    self.market_data_cache = new_data[-50:]  # Keep last 50 bars
                    
        except Exception as e:
            print(f"Market data update error: {e}")

    def _is_episode_done(self):
        """Check if episode should end (from original)"""
        try:
            # End if maximum steps reached
            if self.current_step >= self.max_steps:
                return True
                
            # End if account equity is too low
            if not self.is_training_mode:
                account_info = self.mt5_interface.get_account_info()
                if account_info:
                    equity = account_info.get('equity', 0)
                    balance = account_info.get('balance', 0)
                    if equity < balance * 0.5:  # 50% drawdown
                        return True
                        
            return False
            
        except:
            return False

    def _get_info(self):
        """Get additional info (enhanced from original)"""
        try:
            account_info = None
            positions = []
            
            if not self.is_training_mode and hasattr(self, 'mt5_interface'):
                account_info = self.mt5_interface.get_account_info()
                positions = self.mt5_interface.get_positions()
            
            info = {
                'current_step': self.current_step,
                'trading_state': self.trading_state.value,
                'state_duration': time.time() - self.state_start_time,
                'episode_pnl': sum(pos.get('profit', 0) for pos in positions) if positions else 0,
                'account_balance': account_info.get('balance', 0) if account_info else 0,
                'account_equity': account_info.get('equity', 0) if account_info else 0,
                'open_positions': len(positions),
                'max_drawdown': self.max_drawdown,
                'total_trades': len(self.episode_trades),
                'state_transitions': len(self.state_history)
            }
            
            return info
            
        except Exception as e:
            print(f"Info error: {e}")
            return {
                'current_step': self.current_step,
                'trading_state': self.trading_state.value,
                'episode_pnl': 0,
                'open_positions': 0
            }
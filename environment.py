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
        from portfolio_manager import AIPortfolioManager
        self.portfolio_manager = AIPortfolioManager(config)
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
        
        # 🎯 Position State Management
        self.position_state = 'READY'  # READY, ACTIVE, RECOVERY, CLOSING
        self.last_entry_time = 0
        self.active_positions_count = 0
        self.entry_cooldown = 10  # 10 seconds between entries
        self.analysis_mode = 'ENTRY'  # ENTRY, MANAGEMENT, RECOVERY

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
        self.portfolio_manager.initialize_portfolio(self.mt5_interface)
        # Get initial observation
        observation = self._get_observation()

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
        # Update portfolio before action
        self.portfolio_manager.update_portfolio_status(self.mt5_interface)
    
        # Parse action
        action_type = float(action[0])
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
        """Execute trading action with POSITION STATE MANAGEMENT and RECOVERY SYSTEM"""
        reward = 0.0
        
        print(f"🔍 DEBUG _execute_action called:")
        print(f"   Original action_type: {action_type}")
        
        try:
            # 🎯 UPDATE POSITION STATE FIRST
            self.update_position_state()
            
            # Get current market data
            current_price = self._get_current_price()
            if current_price is None:
                print("❌ ERROR: current_price is None")
                return -1.0
            
            # Get positions for intelligent decision
            positions = self.mt5_interface.get_positions() if hasattr(self, 'mt5_interface') else []
            total_pnl = sum(pos.get('profit', 0) for pos in positions) if positions else 0
            
            # 🧠 APPLY INTELLIGENT ACTION LOGIC
            try:
                intelligent_action = self.intelligent_action_logic(action_type, positions, total_pnl)
                
                if intelligent_action is not None and intelligent_action != action_type:
                    print(f"🧠 SMART OVERRIDE: {action_type:.3f} → {intelligent_action:.3f}")
                    action_type = intelligent_action
            except Exception as e:
                print(f"❌ ERROR in intelligent_action_logic: {e}")
                print(f"🔄 Using original action: {action_type:.3f}")
                
            # 🎯 APPLY POSITION STATE LOGIC
            adjusted_action = self.get_appropriate_action(action_type)
            if adjusted_action != action_type:
                print(f"🎯 STATE OVERRIDE: {action_type:.3f} → {adjusted_action:.3f}")
                action_type = adjusted_action
            
            # 🚫 CHECK IF NEW ENTRY IS ALLOWED
            if 0.3 <= action_type < 2.7:  # Would open new position
                if not self.should_allow_new_entry(action_type):
                    print("🚫 NEW ENTRY BLOCKED - Converting to HOLD")
                    action_type = 0.1  # Force hold
            
            # Check training mode
            is_training_mode = self.config.get('training_mode', True)
            print(f"🔍 DEBUG training_mode: {is_training_mode}")
            
            # 💼 Position sizing (existing code)
            if 0.3 <= action_type < 2.7:
                if hasattr(self, 'portfolio_manager'):
                    stop_loss = current_price * 0.999 if action_type < 1.7 else current_price * 1.001
                    optimal_size = self.portfolio_manager.calculate_position_size(
                        self.symbol, current_price, stop_loss, self.mt5_interface
                    )
                    lot_size = min(optimal_size * lot_multiplier, 0.01)  # Max 0.01 lot for safety
                    lot_size = max(0.01, lot_size)
                else:
                    lot_size = 0.01  # Safe default
            else:
                lot_size = 0.01

            # 🔥 EXECUTE ACTIONS WITH STATE AWARENESS
            if action_type < 0.3:
                print(f"🟡 HOLD: action_type {action_type:.3f} < 0.3")
                if is_training_mode:
                    reward += self._calculate_simulated_hold_reward()
                    print(f"🔥 SIMULATED HOLD")
                else:
                    reward += 0.1
                    print(f"🚀 LIVE HOLD")
                    
            elif 0.3 <= action_type < 1.7:
                print(f"🟢 BUY: {action_type:.3f} in [0.3, 1.7)")
                if is_training_mode:
                    reward += self._calculate_simulated_trade_reward('buy', lot_size, current_price)
                    print(f"🔥 SIMULATED BUY: {lot_size} {self.symbol} at {current_price:.2f}")
                else:
                    success = self.mt5_interface.place_order(
                        symbol=self.symbol, order_type='buy', volume=lot_size, price=current_price
                    )
                    if success:
                        self.last_entry_time = time.time()
                        self.position_state = 'ACTIVE'
                        self.analysis_mode = 'MANAGEMENT'
                        
                    reward += 2.0 if success else -2.0
                    print(f"🚀 LIVE BUY: {'SUCCESS' if success else 'FAILED'} - {lot_size} lots")
                    
            elif 1.7 <= action_type < 2.7:
                print(f"🔴 SELL: {action_type:.3f} in [1.7, 2.7)")
                if is_training_mode:
                    reward += self._calculate_simulated_trade_reward('sell', lot_size, current_price)
                    print(f"🔥 SIMULATED SELL: {lot_size} {self.symbol} at {current_price:.2f}")
                else:
                    success = self.mt5_interface.place_order(
                        symbol=self.symbol, order_type='sell', volume=lot_size, price=current_price
                    )
                    if success:
                        self.last_entry_time = time.time()
                        self.position_state = 'ACTIVE'
                        self.analysis_mode = 'MANAGEMENT'
                        
                    reward += 2.0 if success else -2.0
                    print(f"🚀 LIVE SELL: {'SUCCESS' if success else 'FAILED'} - {lot_size} lots")
                    
            elif 2.7 <= action_type < 3.5:
                print(f"💰 CLOSE: {action_type:.3f} in [2.7, 3.5)")
                if is_training_mode:
                    reward += self._calculate_simulated_close_reward()
                    print(f"🔥 SIMULATED CLOSE ALL")
                else:
                    positions = self.mt5_interface.get_positions()
                    if positions:
                        success = self._enhanced_close_all_positions(positions)
                        if success:
                            self.position_state = 'READY'
                            self.analysis_mode = 'ENTRY'
                            print("🔄 STATE RESET: Ready for new entries")
                        
                        reward += 3.0 if success and total_pnl > 0 else 1.0
                        print(f"🚀 LIVE CLOSE: {'SUCCESS' if success else 'FAILED'}")
                    else:
                        reward += 0.5
                        print("⚠️ No positions to close")
                    
            elif action_type >= 3.5:
                print(f"🛡️ HEDGE: {action_type:.3f} ≥ 3.5")
                if is_training_mode:
                    reward += 1.0
                    print(f"🔥 SIMULATED HEDGE: {lot_size}")
                else:
                    success = self._execute_hedge_action(lot_size)
                    reward += 1.5 if success else -1.0
                    print(f"🚀 LIVE HEDGE: {'SUCCESS' if success else 'FAILED'}")
            
            # 🔄 ENHANCED RECOVERY SYSTEM
            if not is_training_mode and hasattr(self, 'recovery_engine'):
                # Check if recovery is needed
                if total_pnl < -30 or self.analysis_mode == 'RECOVERY':
                    # Get current account info
                    account_info = self.mt5_interface.get_account_info()
                    current_equity = account_info.get('equity', 0) if account_info else 0
                    
                    print(f"🔄 RECOVERY CHECK: PnL=${total_pnl:.2f}, Mode={self.analysis_mode}")
                    
                    # Check recovery opportunities  
                    profit_opportunities = self.recovery_engine.check_profit_opportunities(
                        self.mt5_interface, self.symbol
                    )
                    
                    if profit_opportunities:
                        print(f"💰 PROFIT OPPORTUNITIES FOUND: {len(profit_opportunities)}")
                        executed = self.recovery_engine.execute_profit_taking(
                            self.mt5_interface, profit_opportunities
                        )
                        if executed:
                            reward += 1.0
                            print("✅ PROFIT TAKEN via recovery engine")
                    
                    # Smart profit strategy
                    profit_taken = self.recovery_engine.smart_profit_strategy(
                        self.mt5_interface, self.symbol
                    )
                    if profit_taken:
                        reward += 2.0
                        print("✅ SMART PROFIT STRATEGY executed")
                        # Reset to ready state after successful profit taking
                        self.position_state = 'READY'
                        self.analysis_mode = 'ENTRY'
                    
                    # If still in loss, activate recovery
                    elif total_pnl < -50:
                        print(f"🚨 ACTIVATING RECOVERY: Loss=${total_pnl:.2f}")
                        recovery_activated = self.recovery_engine.activate_recovery(
                            symbol=self.symbol,
                            mt5_interface=self.mt5_interface,
                            current_pnl=total_pnl,
                            current_equity=current_equity,
                            recovery_type='combined'
                        )
                        
                        if recovery_activated:
                            reward += 0.5
                            print("🔄 RECOVERY SYSTEM activated")
                        else:
                            print("❌ RECOVERY activation failed")
            
            print(f"🎯 FINAL STATE: {self.position_state} | MODE: {self.analysis_mode}")
            print(f"🔍 DEBUG: Final reward: {reward:.3f}")
            
        except Exception as e:
            print(f"❌ ERROR in _execute_action: {e}")
            import traceback
            traceback.print_exc()
            reward = -5.0
            
        return reward
    
    def get_appropriate_action(self, original_action):
        """Get appropriate action based on current position state with RECOVERY ENGINE"""
        try:
            if self.analysis_mode == 'ENTRY':
                # In entry mode - allow original AI decision
                return original_action
                
            elif self.analysis_mode == 'MANAGEMENT':
                # In management mode - focus on profit taking and position management
                positions = self.mt5_interface.get_positions() if hasattr(self, 'mt5_interface') else []
                total_pnl = sum(pos.get('profit', 0) for pos in positions)
                
                # 💰 Smart profit taking with recovery engine settings
                if hasattr(self, 'recovery_engine'):
                    recovery_status = self.recovery_engine.get_status()
                    profit_settings = recovery_status.get('profit_settings', {})
                    min_profit_target = profit_settings.get('min_profit_target', 25)
                    
                    if total_pnl > min_profit_target:
                        print(f"💰 MANAGEMENT: Taking profit at ${total_pnl:.2f} (target: ${min_profit_target:.2f})")
                        return 3.2  # Force close for profit
                else:
                    if total_pnl > 30:  # Default profit target
                        print("💰 MANAGEMENT: Recommending profit taking")
                        return 3.2
                    
                # Check if need recovery
                if total_pnl < -50:  # Significant loss
                    print("🔄 MANAGEMENT: Switching to recovery mode")
                    self.analysis_mode = 'RECOVERY'
                    self.position_state = 'RECOVERY'
                    return self.get_appropriate_action(original_action)  # Re-evaluate in recovery mode
                    
                # Hold and monitor
                print("📊 MANAGEMENT: Hold and monitor")
                return 0.1  # Force hold
                    
            elif self.analysis_mode == 'RECOVERY':
                # 🚨 ACTIVATE RECOVERY ENGINE
                positions = self.mt5_interface.get_positions() if hasattr(self, 'mt5_interface') else []
                total_pnl = sum(pos.get('profit', 0) for pos in positions)
                
                if hasattr(self, 'recovery_engine'):
                    # Get current account info
                    account_info = self.mt5_interface.get_account_info() if hasattr(self, 'mt5_interface') else None
                    current_equity = account_info.get('equity', 0) if account_info else 0
                    
                    print(f"🚨 RECOVERY ENGINE: Activating for loss ${total_pnl:.2f}")
                    
                    # Activate recovery system
                    success = self.recovery_engine.activate_recovery(
                        symbol=self.symbol,
                        mt5_interface=self.mt5_interface,
                        current_pnl=total_pnl,
                        current_equity=current_equity,
                        recovery_type='combined'  # Use smart combined recovery
                    )
                    
                    if success:
                        print("✅ RECOVERY: Recovery system activated")
                        # Let recovery system handle, return hold for now
                        return 0.1
                    else:
                        print("❌ RECOVERY: Failed to activate, trying hedge")
                        return 4.0  # Try hedge as fallback
                else:
                    print("🚨 RECOVERY: No recovery engine, using hedge")
                    return 4.0  # Force hedge if no recovery engine
                    
            return original_action
            
        except Exception as e:
            print(f"Action adjustment error: {e}")
            return original_action
    
    def check_recovery_completion(self):
        """Check if recovery is completed and reset state"""
        try:
            if hasattr(self, 'recovery_engine'):
                completion = self.recovery_engine.check_recovery_completion(self.mt5_interface)
                
                if completion:
                    print("🎉 RECOVERY COMPLETED - Resetting to entry mode")
                    self.position_state = 'READY'
                    self.analysis_mode = 'ENTRY'
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Recovery completion check error: {e}")
            return False

    def get_recovery_status(self):
        """Get current recovery status"""
        try:
            if hasattr(self, 'recovery_engine'):
                return self.recovery_engine.get_status()
            return {'recovery_active': False}
            
        except Exception as e:
            print(f"Recovery status error: {e}")
            return {'recovery_active': False}



    # เพิ่ม helper method ใหม่
    def _get_action_name(self, action_type):
        """Convert action type to string name"""
        if action_type < 0.3:
            return 'hold'
        elif 0.3 <= action_type < 1.7:
            return 'buy'
        elif 1.7 <= action_type < 2.7:
            return 'sell'
        elif 2.7 <= action_type < 3.5:
            return 'close'
        else:
            return 'hedge'
    
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

    def _enhanced_close_all_positions(self, positions):
        """Enhanced position closing with multiple retry methods"""
        try:
            if not positions:
                return True
                
            print(f"🔄 Trying to close {len(positions)} positions...")
            
            # Method 1: Close all at once
            success = self.mt5_interface.close_all_positions(self.symbol)
            if success:
                print("✅ Method 1: Bulk close successful")
                return True
                
            print("⚠️ Method 1 failed, trying individual closes...")
            
            # Method 2: Close individual positions
            closed_count = 0
            for i, position in enumerate(positions):
                ticket = position.get('ticket')
                if ticket:
                    individual_success = self.mt5_interface.close_position(ticket)
                    if individual_success:
                        closed_count += 1
                        print(f"✅ Closed position {ticket}")
                    else:
                        print(f"❌ Failed to close position {ticket}")
                        
                    # Small delay between closes
                    time.sleep(0.1)
                    
            success_rate = closed_count / len(positions) if len(positions) > 0 else 0
            print(f"📊 Individual close result: {closed_count}/{len(positions)} ({success_rate:.1%})")
            
            # Consider success if at least 70% closed
            return success_rate >= 0.7
            
        except Exception as e:
            print(f"❌ Enhanced close error: {e}")
            return False

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
        """Get SMART observation for AI decision making"""
        observation = np.zeros(30, dtype=np.float32)
        
        try:
            # 🧠 SMART MARKET ANALYSIS (features 0-14)
            market_features = self._get_enhanced_market_features()
            observation[0:15] = market_features
            
            # 💰 POSITION INTELLIGENCE (features 15-22)
            position_features = self._get_enhanced_position_features()
            observation[15:23] = position_features
            
            # 💳 ACCOUNT STATUS (features 23-26)
            account_features = self._get_enhanced_account_features()
            observation[23:27] = account_features
            
            # 🔄 RECOVERY STATUS (features 27-29)
            recovery_features = self._get_enhanced_recovery_features()
            observation[27:30] = recovery_features
            
            # 🛡️ SAFETY CHECK
            observation = np.clip(observation, -10.0, 10.0)
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 🔍 DEBUG: Print key signals
            if hasattr(self, 'current_step') and self.current_step % 10 == 0:
                print(f"🧠 KEY SIGNALS: trend={observation[0]:.2f}, pnl={observation[15]:.2f}, positions={observation[16]:.2f}")
            
        except Exception as e:
            print(f"❌ Observation error: {e}")
            # Fallback to basic signals
            observation[0] = np.random.choice([-1, 0, 1]) * 0.5
            observation[15] = np.random.choice([-0.5, 0, 0.5])
            
        return observation

    # 4. เพิ่ม Enhanced Feature Methods
    def _get_enhanced_market_features(self):
        """Get enhanced market features with clear signals"""
        features = np.zeros(15)
        
        try:
            if len(self.market_data_cache) < 5:
                return features
                
            recent_data = self.market_data_cache[-10:]
            prices = [data['close'] for data in recent_data]
            
            # 📈 TREND SIGNALS (stronger signals)
            if len(prices) >= 5:
                # Short-term trend (3 bars)
                if len(prices) >= 3:
                    short_trend = (prices[-1] - prices[-3]) / prices[-3] if prices[-3] > 0 else 0
                    features[0] = short_trend * 20  # Amplify signal
                
                # Medium-term trend (5 bars)
                if len(prices) >= 5:
                    medium_trend = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
                    features[1] = medium_trend * 15
                
                # Trend consistency
                if len(prices) >= 4:
                    changes = np.diff(prices[-4:])
                    if len(changes) > 0:
                        up_moves = sum(1 for x in changes if x > 0)
                        features[2] = (up_moves / len(changes) - 0.5) * 4  # -2 to +2
            
            # 📊 MOMENTUM SIGNALS
            if len(prices) >= 6:
                # Price acceleration
                if prices[-6] > 0:
                    momentum = (prices[-1] - prices[-6]) / prices[-6]
                    features[3] = momentum * 10
                
                # Volatility signal
                recent_vol = np.std(prices[-5:]) / np.mean(prices[-5:]) if np.mean(prices[-5:]) > 0 else 0
                features[4] = recent_vol * 100  # Scale up
            
            # 🕐 TIME SIGNALS
            now = datetime.now()
            hour = now.hour
            
            # Market session strength
            if 8 <= hour <= 12:  # EU morning
                features[10] = 1.5
            elif 13 <= hour <= 17:  # US morning  
                features[11] = 1.2
            elif 21 <= hour <= 23:  # Asian session
                features[12] = 0.8
                
            # Weekend proximity
            if now.weekday() >= 4:  # Friday/Weekend
                features[13] = -0.5  # Reduce activity
                
        except Exception as e:
            print(f"Enhanced market features error: {e}")
            
        return features

    def _get_enhanced_position_features(self):
        """Get enhanced position features with risk signals"""
        features = np.zeros(8)
        
        try:
            positions = self.mt5_interface.get_positions() if hasattr(self, 'mt5_interface') else []
            
            if positions:
                # Position count pressure
                position_count = len(positions)
                features[0] = min(position_count / 5.0, 2.0)  # Cap at 2.0
                
                # Total PnL signal
                total_pnl = sum(pos.get('profit', 0) for pos in positions)
                features[1] = total_pnl / 25.0  # More sensitive
                
                # 🚨 RISK SIGNALS
                if total_pnl < -30:
                    features[2] = -2.0  # STRONG CLOSE signal
                elif total_pnl > 30:
                    features[3] = 2.0   # STRONG TAKE PROFIT signal
                    
                # Position balance
                buy_count = sum(1 for pos in positions if pos.get('type', 0) == 0)
                sell_count = position_count - buy_count
                if position_count > 0:
                    balance = (buy_count - sell_count) / position_count
                    features[4] = balance  # -1 to +1
                    
                # Volume exposure
                total_volume = sum(pos.get('volume', 0) for pos in positions)
                features[5] = min(total_volume / 0.5, 2.0)  # Risk signal
                
            else:
                # No positions - neutral
                features[6] = 0.5  # Slight bias towards action
                
        except Exception as e:
            print(f"Enhanced position features error: {e}")
            
        return features

    def _get_enhanced_account_features(self):
        """Get enhanced account features"""
        features = np.zeros(4)
        
        try:
            if hasattr(self, 'simulated_balance'):
                # Simulated account (training)
                features[0] = self.simulated_equity / 5000.0  # Normalize to starting balance
                features[1] = (self.simulated_equity - self.simulated_balance) / self.simulated_balance if self.simulated_balance > 0 else 0
            else:
                # Real account
                account_info = self.mt5_interface.get_account_info() if hasattr(self, 'mt5_interface') else None
                if account_info:
                    balance = account_info.get('balance', 0)
                    equity = account_info.get('equity', 0)
                    features[0] = equity / max(balance, 1)  # Equity ratio
                    features[1] = (equity - balance) / max(balance, 1)  # PnL ratio
                    
        except Exception as e:
            print(f"Enhanced account features error: {e}")
            
        return features

    def _get_enhanced_recovery_features(self):
        """Get enhanced recovery features"""
        features = np.zeros(3)
        
        try:
            features[0] = 1.0 if self.recovery_active else 0.0
            features[1] = self.recovery_level / 5.0  # Normalize
            features[2] = self.current_step / self.max_steps
            
        except Exception as e:
            print(f"Enhanced recovery features error: {e}")
            
        return features

    def intelligent_action_logic(self, action_type, positions, total_pnl):
        """PROFESSIONAL Multi-Timeframe Analysis"""
        
        try:
            # Safety checks
            if action_type is None:
                return 0.1
            positions = positions or []
            total_pnl = total_pnl or 0.0
            
            # 🔧 Spread analysis
            spread_cost = self._calculate_total_spread_cost(positions)
            adjusted_pnl = total_pnl - spread_cost
            
            print(f"💰 PnL: Raw=${total_pnl:.2f}, Spread=${spread_cost:.2f}, Net=${adjusted_pnl:.2f}")
            
            # 1. Profit/Loss management
            if adjusted_pnl > spread_cost + 20:
                print(f"💰 TAKE PROFIT: ${adjusted_pnl:.2f}")
                return 3.2
            if adjusted_pnl < -(spread_cost + 30):
                print(f"🚨 STOP LOSS: ${adjusted_pnl:.2f}")
                return 3.5
                
            # 2. 🎯 PROFESSIONAL MULTI-TIMEFRAME ANALYSIS
            if 0.3 <= action_type < 2.7:  # Would open new position
                
                # Get professional analysis
                main_trend = self.analyze_main_trend()
                momentum = self.analyze_entry_momentum()
                volatility = self.calculate_multi_tf_volatility()
                sr_levels = self.detect_support_resistance()
                
                # Calculate required profit (same as before)
                required_profit = 45 * 0.01 * 1.5  # $0.675
                
                # Professional potential calculation
                potential_profit = volatility * 0.5  # More conservative but accurate
                
                print(f"📊 PROFESSIONAL ANALYSIS:")
                print(f"   Main Trend: {main_trend}")
                print(f"   Momentum Score: {momentum:.2f}")
                print(f"   Multi-TF Volatility: {volatility:.3f}")
                print(f"   Potential Profit: ${potential_profit:.2f}")
                print(f"   Required: ${required_profit:.2f}")
                
                # 🧠 SMART DECISION MATRIX
                trade_score = 0
                
                # Trend alignment bonus
                if action_type < 1.7:  # BUY action
                    if main_trend in ['UP', 'STRONG_UP']:
                        trade_score += 3
                        print("✅ BUY aligned with uptrend")
                    elif main_trend in ['DOWN', 'STRONG_DOWN']:
                        trade_score -= 2
                        print("⚠️ BUY against downtrend")
                else:  # SELL action
                    if main_trend in ['DOWN', 'STRONG_DOWN']:
                        trade_score += 3
                        print("✅ SELL aligned with downtrend")
                    elif main_trend in ['UP', 'STRONG_UP']:
                        trade_score -= 2
                        print("⚠️ SELL against uptrend")
                        
                # Momentum bonus
                if momentum > 0.1:
                    trade_score += 2
                    print(f"✅ Good momentum: {momentum:.2f}")
                elif momentum < -0.1:
                    trade_score += 1  # Contrarian opportunity
                    print(f"⚡ Contrarian opportunity: {momentum:.2f}")
                    
                # Volatility check
                if potential_profit >= required_profit:
                    trade_score += 2
                    print(f"✅ Sufficient volatility: ${potential_profit:.2f}")
                else:
                    trade_score -= 3
                    print(f"❌ Insufficient volatility: ${potential_profit:.2f}")
                    
                # Final decision
                print(f"🎯 Trade Score: {trade_score}/7")
                
                if trade_score >= 3:
                    print("✅ HIGH-PROBABILITY TRADE - EXECUTE")
                    return action_type  # Allow trade
                else:
                    print("⚠️ LOW-PROBABILITY TRADE - HOLD")
                    return 0.1  # Hold
                    
            # Continue with other checks (spread, position optimization, etc.)
            return self._original_intelligent_logic(action_type, positions, adjusted_pnl)
            
        except Exception as e:
            print(f"❌ Professional analysis error: {e}")
            return action_type

    def update_position_state(self):
        """Update position state based on current market positions"""
        try:
            positions = self.mt5_interface.get_positions() if hasattr(self, 'mt5_interface') else []
            self.active_positions_count = len(positions)
            
            # Determine current state
            if self.active_positions_count == 0:
                self.position_state = 'READY'
                self.analysis_mode = 'ENTRY'
                print("🟢 STATE: READY - No positions, ready for new entry")
                
            elif self.active_positions_count > 0:
                # Check if positions are profitable or need management
                total_pnl = sum(pos.get('profit', 0) for pos in positions)
                
                if total_pnl > 20:  # In profit
                    self.position_state = 'ACTIVE'
                    self.analysis_mode = 'MANAGEMENT'
                    print(f"💰 STATE: ACTIVE - {self.active_positions_count} positions, PnL: ${total_pnl:.2f}")
                    
                elif total_pnl < -50:  # In significant loss
                    self.position_state = 'RECOVERY'
                    self.analysis_mode = 'RECOVERY'
                    print(f"🚨 STATE: RECOVERY - {self.active_positions_count} positions, Loss: ${total_pnl:.2f}")
                    
                else:  # Normal management
                    self.position_state = 'ACTIVE'
                    self.analysis_mode = 'MANAGEMENT'
                    print(f"📊 STATE: ACTIVE - {self.active_positions_count} positions, PnL: ${total_pnl:.2f}")
                    
        except Exception as e:
            print(f"Position state update error: {e}")

    def should_allow_new_entry(self, action_type):
        """Determine if new entry should be allowed"""
        try:
            # Only allow entry actions when in READY state
            if action_type in [0.3, 1.7]:  # BUY or SELL range
                if self.position_state != 'READY':
                    print(f"🚫 NEW ENTRY BLOCKED - Current state: {self.position_state}")
                    print(f"   Active positions: {self.active_positions_count}")
                    print(f"   Analysis mode: {self.analysis_mode}")
                    return False
                    
                # Check cooldown
                current_time = time.time()
                if current_time - self.last_entry_time < self.entry_cooldown:
                    remaining = self.entry_cooldown - (current_time - self.last_entry_time)
                    print(f"⏰ ENTRY COOLDOWN: {remaining:.1f} seconds remaining")
                    return False
                    
                print("✅ NEW ENTRY ALLOWED - State is READY")
                return True
                
            return True  # Allow other actions (HOLD, CLOSE, HEDGE)
            
        except Exception as e:
            print(f"Entry permission error: {e}")
            return False

    def update_position_state(self):
        """Update position state based on current market positions with RECOVERY CHECK"""
        try:
            positions = self.mt5_interface.get_positions() if hasattr(self, 'mt5_interface') else []
            self.active_positions_count = len(positions)
            
            # 🔄 Check recovery completion first
            if self.analysis_mode == 'RECOVERY':
                if self.check_recovery_completion():
                    return  # State already reset by check_recovery_completion
            
            # Determine current state
            if self.active_positions_count == 0:
                self.position_state = 'READY'
                self.analysis_mode = 'ENTRY'
                print("🟢 STATE: READY - No positions, ready for new entry")
                
            elif self.active_positions_count > 0:
                # Check if positions are profitable or need management
                total_pnl = sum(pos.get('profit', 0) for pos in positions)
                
                # Get recovery status
                recovery_status = self.get_recovery_status()
                recovery_active = recovery_status.get('recovery_active', False)
                
                if recovery_active:
                    self.position_state = 'RECOVERY'
                    self.analysis_mode = 'RECOVERY'
                    recovery_level = recovery_status.get('recovery_level', 0)
                    print(f"🔄 STATE: RECOVERY - Level {recovery_level}, PnL: ${total_pnl:.2f}")
                    
                elif total_pnl > 20:  # In profit
                    self.position_state = 'ACTIVE'
                    self.analysis_mode = 'MANAGEMENT'
                    print(f"💰 STATE: ACTIVE - {self.active_positions_count} positions, PnL: ${total_pnl:.2f}")
                    
                elif total_pnl < -50:  # In significant loss, but recovery not active yet
                    self.position_state = 'RECOVERY'
                    self.analysis_mode = 'RECOVERY'
                    print(f"🚨 STATE: RECOVERY - {self.active_positions_count} positions, Loss: ${total_pnl:.2f}")
                    
                else:  # Normal management
                    self.position_state = 'ACTIVE'
                    self.analysis_mode = 'MANAGEMENT'
                    print(f"📊 STATE: ACTIVE - {self.active_positions_count} positions, PnL: ${total_pnl:.2f}")
                    
        except Exception as e:
            print(f"Position state update error: {e}")
                

    def _calculate_total_spread_cost(self, positions):
        """Calculate total spread cost - FIXED FOR YOUR BROKER"""
        try:
            if not positions:
                return 0.0
                
            total_spread_cost = 0.0
            
            # Get current spread (minimum 45 points for your broker)
            spread_info = self._get_current_spread_info()
            spread_usd_per_lot = max(spread_info.get('spread_usd_per_lot', 45), 45)  # Minimum $45/lot
            
            for position in positions:
                volume = position.get('volume', 0)
                spread_cost = volume * spread_usd_per_lot
                total_spread_cost += spread_cost
                
            print(f"💰 TOTAL SPREAD COST: ${total_spread_cost:.2f} for {len(positions)} positions")
            return total_spread_cost
            
        except Exception as e:
            print(f"Spread cost calculation error: {e}")
            # Conservative estimate for your broker
            return len(positions) * 0.45  # $0.45 per 0.01 lot position
    
    def _get_current_spread_info(self):
        """Get real-time spread information - FIXED FOR YOUR BROKER"""
        try:
            if not hasattr(self, 'mt5_interface'):
                return self._get_default_spread_info()
                
            current_price = self.mt5_interface.get_current_price(self.symbol)
            if not current_price:
                return self._get_default_spread_info()
                
            spread_points = current_price.get('spread', 45)  # Default 45 points
            bid = current_price.get('bid', 0)
            ask = current_price.get('ask', 0)
            
            # 🔧 FIXED CALCULATION FOR YOUR BROKER
            # XAUUSD: 45 points minimum spread
            # 1 pip = 10 points
            # 1 pip = $10 per lot
            
            spread_pips = spread_points / 10.0  # Convert points to pips
            spread_usd_per_lot = spread_pips * 10.0  # $10 per pip per lot
            
            # Ensure minimum spread (your broker minimum)
            if spread_pips < 4.5:  # Less than 45 points
                spread_pips = 4.5
                spread_usd_per_lot = 45.0
                
            print(f"📊 SPREAD INFO: {spread_points} points = {spread_pips:.1f} pips = ${spread_usd_per_lot:.1f}/lot")
            
            return {
                'spread_points': spread_points,
                'spread_pips': spread_pips,
                'spread_usd_per_lot': spread_usd_per_lot,
                'bid': bid,
                'ask': ask,
                'broker_min_spread': 45  # Your broker minimum
            }
            
        except Exception as e:
            print(f"Spread info error: {e}")
            return self._get_default_spread_info()
        
    def _get_default_spread_info(self):
        """Default spread info for your broker"""
        return {
            'spread_points': 45,
            'spread_pips': 4.5,
            'spread_usd_per_lot': 45.0,
            'bid': 0,
            'ask': 0,
            'broker_min_spread': 45
        }
    
    def _original_intelligent_logic(self, action_type, positions, adjusted_pnl):
        """Original intelligence logic with adjusted PnL"""
        
        try:
            # Safety checks
            if action_type is None:
                return 0.1
                
            if positions is None:
                positions = []
                
            # Position overload
            position_count = len(positions)
            if position_count > 10:  # Reduced from 20 due to spread cost
                print("🚨 TOO MANY POSITIONS: Forcing consolidation")
                return 2.2                
            
            # Time-based intelligence
            try:
                now = datetime.now()
                hour = now.hour
                
                if hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # Low liquidity = higher spreads
                    if 0.3 <= action_type < 2.7:
                        print(f"⏰ LOW LIQUIDITY: Hour {hour}, avoiding high-spread trades")
                        return 0.1
            except Exception as e:
                print(f"Time logic error: {e}")
                    
            # Trend following
            try:
                if hasattr(self, 'market_data_cache') and len(self.market_data_cache) >= 10:
                    recent_prices = [data['close'] for data in self.market_data_cache[-10:]]
                    
                    if len(recent_prices) >= 5:
                        price_changes = np.diff(recent_prices[-5:])
                        up_moves = sum(1 for x in price_changes if x > 0)
                        trend_strength = up_moves / len(price_changes)
                        
                        if trend_strength >= 0.8 and 1.7 <= action_type < 2.7:
                            print(f"📈 STRONG UPTREND: Converting SELL to BUY")
                            return 1.2
                        elif trend_strength <= 0.2 and 0.3 <= action_type < 1.7:
                            print(f"📉 STRONG DOWNTREND: Converting BUY to SELL")
                            return 2.0
            except Exception as e:
                print(f"Trend logic error: {e}")
                        
            return action_type
            
        except Exception as e:
            print(f"❌ _original_intelligent_logic error: {e}")
            return action_type  # Return original if error
        
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
        """Update multi-timeframe market data with PROFESSIONAL analysis"""
        try:
            import MetaTrader5 as mt5
            
            # 📊 MULTI-TIMEFRAME DATA COLLECTION
            timeframes = {
                'H1': {'tf': mt5.TIMEFRAME_H1, 'count': 24},    # 24 hours (main trend)
                'M15': {'tf': mt5.TIMEFRAME_M15, 'count': 40},  # 10 hours (entry trend)
                'M5': {'tf': mt5.TIMEFRAME_M5, 'count': 60},    # 5 hours (momentum)
                'M1': {'tf': mt5.TIMEFRAME_M1, 'count': 100}    # 100 minutes (timing)
            }
            
            # Initialize storage if not exists
            if not hasattr(self, 'multi_tf_data'):
                self.multi_tf_data = {}
                
            # Collect data from all timeframes
            for tf_name, tf_config in timeframes.items():
                rates = mt5.copy_rates_from_pos(self.symbol, tf_config['tf'], 0, tf_config['count'])
                
                if rates is not None and len(rates) > 0:
                    # Convert to list of dicts for easier access
                    tf_data = []
                    for rate in rates:
                        tf_data.append({
                            'time': rate[0],
                            'open': float(rate[1]),
                            'high': float(rate[2]),
                            'low': float(rate[3]),
                            'close': float(rate[4]),
                            'volume': float(rate[5]) if len(rate) > 5 else 0
                        })
                    
                    self.multi_tf_data[tf_name] = tf_data
                    
            # Keep M1 in original cache for compatibility
            if 'M1' in self.multi_tf_data and self.multi_tf_data['M1']:
                self.market_data_cache = self.multi_tf_data['M1'][-100:]  # Keep last 100
                
            # Debug current price
            if self.market_data_cache:
                current_price = self.market_data_cache[-1]['close']
                print(f"📊 Multi-TF Update: {current_price:.2f} at {datetime.now().strftime('%H:%M:%S')}")
                
        except Exception as e:
            print(f"❌ Multi-TF data error: {e}")

    # 2. เพิ่ม Professional Analysis Functions

    def analyze_main_trend(self):
        """Analyze H1 trend direction"""
        try:
            if 'H1' not in self.multi_tf_data or len(self.multi_tf_data['H1']) < 10:
                return 'NEUTRAL'
                
            h1_data = self.multi_tf_data['H1']
            
            # EMA trend analysis
            recent_closes = [bar['close'] for bar in h1_data[-10:]]
            ema_short = sum(recent_closes[-5:]) / 5  # 5-period EMA
            ema_long = sum(recent_closes) / 10       # 10-period EMA
            
            current_price = recent_closes[-1]
            
            # Trend strength
            if current_price > ema_short > ema_long:
                trend_strength = (current_price - ema_long) / ema_long * 100
                if trend_strength > 0.5:  # Strong uptrend
                    return 'STRONG_UP'
                else:
                    return 'UP'
            elif current_price < ema_short < ema_long:
                trend_strength = (ema_long - current_price) / ema_long * 100
                if trend_strength > 0.5:  # Strong downtrend
                    return 'STRONG_DOWN'
                else:
                    return 'DOWN'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            print(f"Main trend error: {e}")
            return 'NEUTRAL'

    def analyze_entry_momentum(self):
        """Analyze M15 and M5 momentum for entry timing"""
        try:
            momentum_score = 0
            
            # M15 momentum
            if 'M15' in self.multi_tf_data and len(self.multi_tf_data['M15']) >= 5:
                m15_data = self.multi_tf_data['M15']
                m15_closes = [bar['close'] for bar in m15_data[-5:]]
                
                # Price momentum
                price_momentum = (m15_closes[-1] - m15_closes[0]) / m15_closes[0] * 100
                momentum_score += price_momentum * 2  # Weight M15 higher
                
            # M5 momentum
            if 'M5' in self.multi_tf_data and len(self.multi_tf_data['M5']) >= 10:
                m5_data = self.multi_tf_data['M5']
                m5_closes = [bar['close'] for bar in m5_data[-10:]]
                
                # Recent momentum (last 5 bars)
                recent_momentum = (m5_closes[-1] - m5_closes[-5]) / m5_closes[-5] * 100
                momentum_score += recent_momentum
                
            return momentum_score
            
        except Exception as e:
            print(f"Momentum error: {e}")
            return 0

    def calculate_multi_tf_volatility(self):
        """Calculate volatility across timeframes"""
        try:
            total_volatility = 0
            
            # H1 volatility (main market volatility)
            if 'H1' in self.multi_tf_data and len(self.multi_tf_data['H1']) >= 5:
                h1_data = self.multi_tf_data['H1'][-5:]
                h1_ranges = [(bar['high'] - bar['low']) for bar in h1_data]
                h1_avg_range = sum(h1_ranges) / len(h1_ranges)
                total_volatility += h1_avg_range * 10  # Scale H1 volatility
                
            # M15 volatility (entry volatility)
            if 'M15' in self.multi_tf_data and len(self.multi_tf_data['M15']) >= 10:
                m15_data = self.multi_tf_data['M15'][-10:]
                m15_ranges = [(bar['high'] - bar['low']) for bar in m15_data]
                m15_avg_range = sum(m15_ranges) / len(m15_ranges)
                total_volatility += m15_avg_range * 5  # Medium weight
                
            # M5 volatility (immediate volatility)
            if 'M5' in self.multi_tf_data and len(self.multi_tf_data['M5']) >= 20:
                m5_data = self.multi_tf_data['M5'][-20:]
                m5_ranges = [(bar['high'] - bar['low']) for bar in m5_data]
                m5_avg_range = sum(m5_ranges) / len(m5_ranges)
                total_volatility += m5_avg_range * 2  # Lower weight
                
            return total_volatility
            
        except Exception as e:
            print(f"Multi-TF volatility error: {e}")
            return 0

    def detect_support_resistance(self):
        """Detect key support/resistance levels"""
        try:
            levels = {'support': [], 'resistance': []}
            
            # Use H1 and M15 for S/R detection
            for tf_name in ['H1', 'M15']:
                if tf_name not in self.multi_tf_data:
                    continue
                    
                tf_data = self.multi_tf_data[tf_name]
                if len(tf_data) < 20:
                    continue
                    
                # Find swing highs and lows
                highs = [bar['high'] for bar in tf_data[-20:]]
                lows = [bar['low'] for bar in tf_data[-20:]]
                
                # Resistance levels (recent highs)
                max_high = max(highs)
                levels['resistance'].append(max_high)
                
                # Support levels (recent lows)
                min_low = min(lows)
                levels['support'].append(min_low)
                
            return levels
            
        except Exception as e:
            print(f"S/R detection error: {e}")
            return {'support': [], 'resistance': []}
    

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
        """Check if XAUUSD market is open - REALISTIC VERSION"""
        try:
            import pytz
            from datetime import datetime
            
            # Get current time in UTC
            utc_now = datetime.now(pytz.UTC)
            weekday = utc_now.weekday()  # 0=Monday, 6=Sunday
            hour = utc_now.hour
            
            # XAUUSD trades 24/5 (Sunday 22:00 GMT - Friday 21:00 GMT)
            
            # Market closed periods:
            if weekday == 4 and hour >= 21:  # Friday 21:00+ GMT
                return False
            elif weekday == 5:  # Saturday (all day)
                return False  
            elif weekday == 6 and hour < 22:  # Sunday before 22:00 GMT
                return False
                
            # During training, always return True for faster learning
            if self.config.get('training_mode', True):
                return True
                
            # For live trading, also check news hours
            if hour in [12, 13, 14, 20, 21]:  # News hours
                minute = utc_now.minute
                if 25 <= minute <= 35:  # 10-minute news break
                    return False
                    
            return True
            
        except Exception as e:
            print(f"Market hours check error: {e}")
            return True  # Default to open if error

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
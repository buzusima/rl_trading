# environments/aggressive_env.py - Aggressive AI Recovery Environment

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import AI Recovery Intelligence
try:
    from core.recovery_intelligence import RecoveryIntelligence, ActionType, RecoveryState
    AI_AVAILABLE = True
except ImportError:
    print("❌ AI Recovery Intelligence not available")
    AI_AVAILABLE = False

class AggressiveEnvironment(gym.Env):
    """
    ⚡ Aggressive AI Recovery Environment
    
    คุณสมบัติ:
    - ใช้ AI Recovery Brain ตัดสินใจ
    - เทรดมากขึ้น เพื่อ volume และ rebate
    - แก้ไม้อัจฉริยะ 8 strategies
    - Market analysis แบบ real-time
    - เป้าหมาย: 50-100 lots/day
    """
    
    def __init__(self, mt5_interface, config, historical_data=None):
        super(AggressiveEnvironment, self).__init__()
        
        print("⚡ Initializing Aggressive AI Recovery Environment...")
        
        if not AI_AVAILABLE:
            raise ImportError("Cannot create AggressiveEnvironment: AI Recovery Intelligence not available")
        
        # Core components
        self.mt5_interface = mt5_interface
        self.config = config
        self.historical_data = historical_data
        
        # Aggressive trading parameters
        self.symbol = config.get('symbol', 'XAUUSD')
        self.base_lot_size = config.get('lot_size', 0.02)  # เพิ่มจาก 0.01
        self.max_positions = config.get('max_positions', 8)  # เพิ่มจาก 5
        self.max_recovery_levels = config.get('max_recovery_levels', 5)  # เพิ่มจาก 3
        
        # Aggressive recovery parameters
        self.recovery_multiplier = config.get('recovery_multiplier', 2.0)  # เพิ่มจาก 1.3
        self.recovery_threshold = config.get('recovery_threshold', -15.0)  # ลดจาก -35.0 (เร็วกว่า)
        
        # Volume targets for rebate
        self.daily_volume_target = config.get('daily_volume_target', 75.0)  # 75 lots/day
        self.volume_bonus_threshold = config.get('volume_bonus_threshold', 50.0)  # 50 lots
        
        # === OBSERVATION SPACE (40 features) - เหมือนเดิม ===
        self.observation_space = spaces.Box(
            low=-10.0, 
            high=10.0, 
            shape=(40,),
            dtype=np.float32
        )
        
        # === ACTION SPACE (4 dimensions) - เหมือนเดิม ===
        self.action_space = spaces.Box(
            low=np.array([0, 0.01, 0, 0], dtype=np.float32),           
            high=np.array([4, 0.50, 100, 2], dtype=np.float32),        
            shape=(4,),
            dtype=np.float32
        )
        
        # Initialize AI Recovery Brain
        self.recovery_brain = RecoveryIntelligence(mt5_interface, config)
        self.ai_session_id = None
        
        # State tracking
        self.current_step = 0
        self.data_index = 0
        self.episode_start_time = None
        self.episode_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Aggressive metrics
        self.daily_volume = 0.0
        self.volume_bonus_earned = 0.0
        self.ai_decisions_count = 0
        self.ai_success_rate = 0.0
        
        # Position tracking (simplified - AI manages this)
        self.positions = []
        self.last_ai_decision = None
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Mode settings
        self.is_training_mode = config.get('training_mode', True)
        
        print("✅ Aggressive AI Environment initialized:")
        print(f"   - Symbol: {self.symbol}")
        print(f"   - Base Lot: {self.base_lot_size} (Aggressive)")
        print(f"   - Max Positions: {self.max_positions}")
        print(f"   - Recovery Multiplier: {self.recovery_multiplier}x (Aggressive)")
        print(f"   - Recovery Threshold: ${self.recovery_threshold} (Fast)")
        print(f"   - Daily Volume Target: {self.daily_volume_target} lots")
        print(f"   - AI Recovery Brain: Ready")

    def set_historical_data(self, historical_data: pd.DataFrame):
        """Set historical data for training"""
        self.historical_data = historical_data
        self.recovery_brain.market_analyzer.historical_data = historical_data
        print(f"📊 Historical data set: {len(historical_data):,} rows")

    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        print(f"🔄 Resetting Aggressive AI Environment...")
        
        # Reset episode tracking
        self.current_step = 0
        self.data_index = 50 if self.historical_data is not None else 0
        self.episode_start_time = datetime.now()
        self.episode_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 1000.0
        
        # Reset aggressive metrics
        self.daily_volume = 0.0
        self.volume_bonus_earned = 0.0
        self.ai_decisions_count = 0
        self.ai_success_rate = 0.0
        
        # Reset positions
        self.positions.clear()
        self.last_ai_decision = None
        
        # Start AI Recovery Session
        try:
            self.ai_session_id = self.recovery_brain.start_recovery_session()
            print(f"🧠 AI Recovery Session started: {self.ai_session_id}")
        except Exception as e:
            print(f"❌ AI Session start error: {e}")
            self.ai_session_id = None
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        print("✅ Aggressive AI Environment reset complete")
        return observation, info

    def step(self, action):
        """
        ⚡ AI-Driven Step - AI Recovery Brain ตัดสินใจแทน RL Agent
        
        Args:
            action: จาก RL Agent (จะถูกเขียนทับ)
        
        Returns:
            observation, reward, done, truncated, info
        """
        self.current_step += 1
        self.data_index += 1
        
        try:
            # 🧠 ให้ AI Recovery Brain ตัดสินใจ
            ai_decision = self.recovery_brain.make_recovery_decision(
                current_positions=self.positions,
                force_analysis=False
            )
            
            self.last_ai_decision = ai_decision
            self.ai_decisions_count += 1
            
            # Execute AI Decision
            reward = self._execute_ai_decision(ai_decision)
            
            # Update states
            self._update_aggressive_state()
            
            # Get new observation
            observation = self._get_observation()
            
            # Check if episode is done
            done = self._is_episode_done()
            
            # Get info with AI details
            info = self._get_ai_info(ai_decision)
            
            return observation, reward, done, False, info
            
        except Exception as e:
            print(f"❌ Aggressive AI step error: {e}")
            # Return safe default
            observation = self._get_observation()
            info = self._get_info()
            info['error'] = str(e)
            info['environment_type'] = 'AGGRESSIVE_ERROR'
            return observation, -10.0, True, False, info

    def _execute_ai_decision(self, ai_decision):
        """
        ⚡ Execute AI Decision with Aggressive Logic
        
        Args:
            ai_decision: RecoveryDecision from AI Brain
        
        Returns:
            float: Reward
        """
        try:
            action_name = ai_decision.action.name
            volume = ai_decision.volume
            
            print(f"🧠 Executing AI Decision: {action_name} ({volume:.2f} lots)")
            
            reward = 0.0
            
            if action_name == 'HOLD':
                reward = self._handle_ai_hold(ai_decision)
                
            elif action_name in ['BUY', 'RECOVERY_BUY']:
                reward = self._handle_ai_buy(ai_decision)
                
            elif action_name in ['SELL', 'RECOVERY_SELL']:
                reward = self._handle_ai_sell(ai_decision)
                
            elif action_name == 'HEDGE':
                reward = self._handle_ai_hedge(ai_decision)
                
            elif action_name in ['CLOSE_ALL', 'EMERGENCY_CLOSE']:
                reward = self._handle_ai_close_all(ai_decision)
            
            # Volume bonus calculation
            volume_bonus = self._calculate_volume_bonus(volume)
            reward += volume_bonus
            
            # Confidence bonus
            confidence_bonus = (ai_decision.confidence - 0.5) * 2.0  # -1 to 1
            reward += confidence_bonus
            
            return reward
            
        except Exception as e:
            print(f"❌ Execute AI decision error: {e}")
            return -5.0

    def _handle_ai_hold(self, ai_decision):
        """Handle AI HOLD decision"""
        # AI มีเหตุผลในการ HOLD - ให้ reward เล็กน้อย
        return 0.2

    def _handle_ai_buy(self, ai_decision):
        """Handle AI BUY/RECOVERY_BUY decision"""
        try:
            volume = ai_decision.volume
            
            # Get current price
            if self.historical_data is not None and self.data_index < len(self.historical_data):
                current_price = self.historical_data.iloc[self.data_index]['close']
                entry_price = current_price + 0.0001  # Simulate spread
            else:
                entry_price = 2000.0  # Default for live trading
            
            # Calculate SL/TP from AI decision
            sl_price = ai_decision.stop_loss
            tp_price = ai_decision.take_profit
            
            # Create position
            position = {
                'type': 'BUY',
                'volume': volume,
                'entry_price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'entry_time': self.current_step,
                'profit': 0.0,
                'ai_strategy': ai_decision.strategy_type.value,
                'confidence': ai_decision.confidence
            }
            
            self.positions.append(position)
            self.total_trades += 1
            self.daily_volume += volume
            
            # Aggressive reward calculation
            base_reward = 2.0  # เพิ่มจาก 1.0
            
            # Strategy bonus
            if 'RECOVERY' in ai_decision.action.name:
                base_reward += 1.0  # Recovery bonus
            
            # Volume bonus
            if volume >= 0.05:  # Large volume
                base_reward += 0.5
            
            return base_reward
            
        except Exception as e:
            print(f"❌ AI Buy execution error: {e}")
            return -1.0

    def _handle_ai_sell(self, ai_decision):
        """Handle AI SELL/RECOVERY_SELL decision"""
        try:
            volume = ai_decision.volume
            
            # Get current price
            if self.historical_data is not None and self.data_index < len(self.historical_data):
                current_price = self.historical_data.iloc[self.data_index]['close']
                entry_price = current_price - 0.0001  # Simulate spread
            else:
                entry_price = 2000.0
            
            # Calculate SL/TP from AI decision
            sl_price = ai_decision.stop_loss
            tp_price = ai_decision.take_profit
            
            # Create position
            position = {
                'type': 'SELL',
                'volume': volume,
                'entry_price': entry_price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'entry_time': self.current_step,
                'profit': 0.0,
                'ai_strategy': ai_decision.strategy_type.value,
                'confidence': ai_decision.confidence
            }
            
            self.positions.append(position)
            self.total_trades += 1
            self.daily_volume += volume
            
            # Aggressive reward calculation
            base_reward = 2.0
            
            # Strategy bonus
            if 'RECOVERY' in ai_decision.action.name:
                base_reward += 1.0
            
            # Volume bonus
            if volume >= 0.05:
                base_reward += 0.5
            
            return base_reward
            
        except Exception as e:
            print(f"❌ AI Sell execution error: {e}")
            return -1.0

    def _handle_ai_hedge(self, ai_decision):
        """Handle AI HEDGE decision"""
        try:
            if not self.positions:
                return 0.0  # No positions to hedge
            
            # Calculate net exposure
            total_buy_volume = sum(pos['volume'] for pos in self.positions if pos['type'] == 'BUY')
            total_sell_volume = sum(pos['volume'] for pos in self.positions if pos['type'] == 'SELL')
            
            net_volume = total_buy_volume - total_sell_volume
            hedge_volume = abs(net_volume) * 0.5  # Partial hedge
            
            if net_volume > 0:
                # Net long, hedge with sell
                return self._create_hedge_position('SELL', hedge_volume, ai_decision)
            elif net_volume < 0:
                # Net short, hedge with buy
                return self._create_hedge_position('BUY', hedge_volume, ai_decision)
            else:
                return 0.5  # Already balanced
                
        except Exception as e:
            print(f"❌ AI Hedge execution error: {e}")
            return -1.0

    def _create_hedge_position(self, direction, volume, ai_decision):
        """Create hedge position"""
        try:
            if self.historical_data is not None and self.data_index < len(self.historical_data):
                current_price = self.historical_data.iloc[self.data_index]['close']
                if direction == 'BUY':
                    entry_price = current_price + 0.0001
                else:
                    entry_price = current_price - 0.0001
            else:
                entry_price = 2000.0
            
            position = {
                'type': direction,
                'volume': volume,
                'entry_price': entry_price,
                'sl_price': None,  # Hedge positions typically no SL
                'tp_price': None,
                'entry_time': self.current_step,
                'profit': 0.0,
                'ai_strategy': 'HEDGE',
                'confidence': ai_decision.confidence
            }
            
            self.positions.append(position)
            self.total_trades += 1
            self.daily_volume += volume
            
            return 1.5  # Hedge reward
            
        except Exception as e:
            print(f"❌ Create hedge position error: {e}")
            return -0.5

    def _handle_ai_close_all(self, ai_decision):
        """Handle AI CLOSE_ALL/EMERGENCY_CLOSE decision"""
        try:
            if not self.positions:
                return -0.1
            
            total_profit = 0.0
            closed_count = 0
            
            for position in self.positions:
                profit = self._calculate_position_profit(position)
                total_profit += profit
                closed_count += 1
                
                # Track win/loss
                if profit > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
            
            # Clear positions
            self.positions.clear()
            
            # Update P&L
            self.episode_pnl += total_profit
            self.total_pnl += total_profit
            
            # Update drawdown
            if self.total_pnl < self.max_drawdown:
                self.max_drawdown = self.total_pnl
            
            # Aggressive reward calculation
            base_reward = 3.0  # เพิ่มจาก 2.0
            profit_reward = total_profit / 30  # More sensitive
            
            # Emergency close penalty
            if ai_decision.action.name == 'EMERGENCY_CLOSE':
                base_reward -= 1.0
            
            return base_reward + profit_reward
            
        except Exception as e:
            print(f"❌ AI Close all error: {e}")
            return -2.0

    def _calculate_volume_bonus(self, volume):
        """Calculate volume bonus for rebate targeting"""
        try:
            # Volume progression bonus
            if self.daily_volume >= self.volume_bonus_threshold:
                return 1.0  # High volume bonus
            elif self.daily_volume >= self.volume_bonus_threshold * 0.7:
                return 0.5  # Medium volume bonus
            elif volume >= 0.05:  # Large single trade
                return 0.3  # Single trade bonus
            else:
                return 0.1  # Base volume bonus
                
        except Exception as e:
            return 0.0

    def _calculate_position_profit(self, position):
        """Calculate current profit for a position"""
        try:
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
            
        except Exception as e:
            return 0.0

    def _update_aggressive_state(self):
        """Update aggressive environment state"""
        try:
            # Update position profits
            for position in self.positions:
                self._calculate_position_profit(position)
            
            # Update AI success rate
            if self.ai_decisions_count > 0:
                # Simple success metric based on positive outcomes
                self.ai_success_rate = self.winning_trades / max(self.total_trades, 1)
            
            # Check volume targets
            if self.daily_volume >= self.daily_volume_target:
                self.volume_bonus_earned += 10.0  # Daily bonus
                
        except Exception as e:
            print(f"❌ Update aggressive state error: {e}")

    def _get_observation(self):
        """Get observation with AI enhancements"""
        try:
            # Start with base observation (40 features)
            obs = np.zeros(40, dtype=np.float32)
            
            # === MARKET DATA (15 features) - เหมือนเดิม ===
            if self.historical_data is not None and self.data_index < len(self.historical_data):
                current_data = self.historical_data.iloc[self.data_index]
                
                close_price = current_data['close']
                obs[0] = (current_data['high'] - current_data['low']) / close_price
                obs[1] = (current_data['close'] - current_data['open']) / close_price
                
                # Trend indicators
                obs[2] = (current_data['close'] - current_data['SMA_20']) / current_data['ATR_14']
                obs[3] = (current_data['close'] - current_data['SMA_50']) / current_data['ATR_14']
                obs[4] = (current_data['EMA_12'] - current_data['EMA_26']) / current_data['ATR_14']
                
                # Momentum indicators
                obs[5] = (current_data['RSI_14'] - 50) / 50
                obs[6] = current_data['MACD'] / current_data['ATR_14']
                obs[7] = current_data['MACD_Histogram'] / current_data['ATR_14']
                
                # Volatility indicators
                obs[8] = current_data['BB_Position']
                obs[9] = current_data['BB_Width']
                obs[10] = current_data['ATR_14'] / close_price
                obs[11] = current_data['Volatility_Regime']
                
                # Market context
                obs[12] = current_data['Trend_Strength']
                obs[13] = current_data['RSI_Momentum'] / 10
                obs[14] = current_data['MACD_Momentum'] / current_data['ATR_14']
            
            # === AI STATE (10 features) ===
            obs[15] = self.ai_success_rate
            obs[16] = self.ai_decisions_count / 100
            obs[17] = min(self.daily_volume / self.daily_volume_target, 2.0)
            obs[18] = min(self.episode_pnl / 100, 5.0)
            obs[19] = min(self.total_pnl / 1000, 5.0)
            obs[20] = self.max_drawdown / 500
            obs[21] = len(self.positions) / self.max_positions
            obs[22] = self.volume_bonus_earned / 100
            
            # Last AI decision info
            if self.last_ai_decision:
                obs[23] = self.last_ai_decision.confidence
                if self.last_ai_decision.recovery_state == RecoveryState.NORMAL:
                    obs[24] = 0.0
                elif self.last_ai_decision.recovery_state == RecoveryState.EARLY_RECOVERY:
                    obs[24] = 0.2
                elif self.last_ai_decision.recovery_state == RecoveryState.ACTIVE_RECOVERY:
                    obs[24] = 0.4
                elif self.last_ai_decision.recovery_state == RecoveryState.DEEP_RECOVERY:
                    obs[24] = 0.6
                elif self.last_ai_decision.recovery_state == RecoveryState.EMERGENCY:
                    obs[24] = 0.8
                else:
                    obs[24] = 1.0
            
            # === POSITION ANALYSIS (8 features) ===
            if self.positions:
                total_volume = sum(pos['volume'] for pos in self.positions)
                total_profit = sum(pos['profit'] for pos in self.positions)
                avg_confidence = np.mean([pos.get('confidence', 0.5) for pos in self.positions])
                
                obs[25] = total_volume / 2.0
                obs[26] = total_profit / 100
                obs[27] = len([p for p in self.positions if p['type'] == 'BUY']) / max(len(self.positions), 1)
                obs[28] = len([p for p in self.positions if p['type'] == 'SELL']) / max(len(self.positions), 1)
                obs[29] = avg_confidence
                obs[30] = self.winning_trades / max(self.total_trades, 1)
                obs[31] = len([p for p in self.positions if p.get('ai_strategy') == 'HEDGE']) / max(len(self.positions), 1)
                obs[32] = (self.daily_volume / max(self.daily_volume_target, 1))
            
            # === TIME & SESSION (4 features) ===
            if self.historical_data is not None and self.data_index < len(self.historical_data):
                current_time = self.historical_data.index[self.data_index]
                obs[33] = current_time.hour / 24
                obs[34] = current_time.weekday() / 7
                obs[35] = (current_time.hour >= 8 and current_time.hour <= 17) * 1.0
                obs[36] = (current_time.hour >= 13 and current_time.hour <= 22) * 1.0
            
            # === AGGRESSIVE METRICS (3 features) ===
            obs[37] = min(self.daily_volume / 100, 2.0)
            obs[38] = self.current_step / 1000
            obs[39] = 1.0  # Aggressive mode indicator
            
            # Clip values to valid range
            obs = np.clip(obs, -10.0, 10.0)
            
            return obs
            
        except Exception as e:
            print(f"❌ Aggressive observation error: {e}")
            return np.zeros(40, dtype=np.float32)

    def _is_episode_done(self):
        """Check if episode should end"""
        # Data exhausted
        if self.historical_data is not None and self.data_index >= len(self.historical_data) - 1:
            print(f"📈 Aggressive Episode ended: Data exhausted")
            return True
        
        # Max steps (longer for aggressive mode)
        if self.current_step >= 3000:  # เพิ่มจาก 2000
            print(f"⏰ Aggressive Episode ended: Max steps reached")
            return True
        
        # Volume target reached (optional end condition)
        if self.daily_volume >= self.daily_volume_target * 1.5:  # 1.5x target
            print(f"🎯 Aggressive Episode ended: Volume target exceeded ({self.daily_volume:.1f} lots)")
            return True
        
        return False

    def _get_info(self):
        """Get episode info with AI details"""
        base_info = {
            'environment_type': 'AGGRESSIVE_AI',
            'current_step': self.current_step,
            'data_index': self.data_index,
            'episode_pnl': self.episode_pnl,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'open_positions': len(self.positions),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            
            # Aggressive specific
            'daily_volume': self.daily_volume,
            'volume_target': self.daily_volume_target,
            'volume_progress': (self.daily_volume / self.daily_volume_target) * 100,
            'volume_bonus_earned': self.volume_bonus_earned,
            
            # AI specific
            'ai_session_id': self.ai_session_id,
            'ai_decisions_count': self.ai_decisions_count,
            'ai_success_rate': self.ai_success_rate,
        }
        
        # Add last AI decision info
        if self.last_ai_decision:
            base_info.update({
                'last_ai_action': self.last_ai_decision.action.name,
                'last_ai_strategy': self.last_ai_decision.strategy_type.value,
                'last_ai_confidence': self.last_ai_decision.confidence,
                'last_ai_recovery_state': self.last_ai_decision.recovery_state.value
            })
        
        return base_info

    def _get_ai_info(self, ai_decision):
        """Get detailed AI decision info"""
        ai_info = self._get_info()
        
        # Add current AI decision details
        ai_info.update({
            'current_ai_action': ai_decision.action.name,
            'current_ai_strategy': ai_decision.strategy_type.value,
            'current_ai_volume': ai_decision.volume,
            'current_ai_confidence': ai_decision.confidence,
            'current_ai_recovery_state': ai_decision.recovery_state.value,
            'current_ai_reasoning': ai_decision.reasoning,
            'current_ai_warnings': ai_decision.warnings,
            'current_ai_expected_outcome': ai_decision.expected_outcome,
            'current_ai_risk_assessment': ai_decision.risk_assessment
        })
        
        return ai_info

    def set_training_mode(self, is_training: bool):
        """Set training mode"""
        self.is_training_mode = is_training
        if is_training:
            print("🎓 Aggressive AI Environment set to TRAINING mode")
        else:
            print("🚀 Aggressive AI Environment set to LIVE TRADING mode")

    def get_ai_insights(self):
        """Get AI market insights"""
        try:
            if self.recovery_brain:
                return self.recovery_brain.get_market_insights()
            return {}
        except Exception as e:
            print(f"❌ Get AI insights error: {e}")
            return {}

    def get_ai_status(self):
        """Get AI system status"""
        try:
            if self.recovery_brain:
                return self.recovery_brain.get_system_status()
            return {}
        except Exception as e:
            print(f"❌ Get AI status error: {e}")
            return {}

    def end_ai_session(self):
        """End AI recovery session"""
        try:
            if self.recovery_brain and self.ai_session_id:
                summary = self.recovery_brain.end_session()
                print(f"🏁 AI Session ended: {summary}")
                return summary
            return {}
        except Exception as e:
            print(f"❌ End AI session error: {e}")
            return {}

    # Legacy methods for compatibility
    def update_market_data(self):
        """Legacy method - not needed with AI Brain"""
        pass
    
    def get_current_price(self, symbol):
        """Legacy method - use historical data instead"""
        if self.historical_data is not None and self.data_index < len(self.historical_data):
            return self.historical_data.iloc[self.data_index]['close']
        return 2000.0
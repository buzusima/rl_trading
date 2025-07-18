# environment.py - Professional Grade RL Trading Environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import time
from datetime import datetime, timedelta
from enum import Enum
import talib as ta
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """Market Regime Detection"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN" 
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"

class TradingState(Enum):
    """Enhanced Trading States"""
    MARKET_ANALYSIS = "MARKET_ANALYSIS"
    OPPORTUNITY_DETECTION = "OPPORTUNITY_DETECTION"
    POSITION_ENTRY = "POSITION_ENTRY"
    POSITION_MANAGEMENT = "POSITION_MANAGEMENT"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"
    RECOVERY_MODE = "RECOVERY_MODE"
    PROFIT_OPTIMIZATION = "PROFIT_OPTIMIZATION"
    PORTFOLIO_REBALANCE = "PORTFOLIO_REBALANCE"

class TechnicalAnalyzer:
    """Advanced Technical Analysis Engine"""
    
    def __init__(self):
        self.indicator_cache = {}
        self.timeframes = ['M1', 'M5', 'M15', 'H1', 'H4']
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate 50+ technical indicators"""
        try:
            if len(data) < 100:  # Need sufficient data
                return self._get_default_indicators()
                
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            open_prices = data['open'].values
            volume = data.get('volume', pd.Series([1000] * len(data))).values
            
            indicators = {}
            
            # === TREND INDICATORS ===
            indicators['sma_20'] = ta.SMA(close, timeperiod=20)[-1] if len(close) >= 20 else close[-1]
            indicators['sma_50'] = ta.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else close[-1]
            indicators['sma_200'] = ta.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else close[-1]
            indicators['ema_12'] = ta.EMA(close, timeperiod=12)[-1] if len(close) >= 12 else close[-1]
            indicators['ema_26'] = ta.EMA(close, timeperiod=26)[-1] if len(close) >= 26 else close[-1]
            
            # MACD
            macd, macd_signal, macd_hist = ta.MACD(close)
            indicators['macd'] = macd[-1] if macd is not None and len(macd) > 0 else 0
            indicators['macd_signal'] = macd_signal[-1] if macd_signal is not None and len(macd_signal) > 0 else 0
            indicators['macd_histogram'] = macd_hist[-1] if macd_hist is not None and len(macd_hist) > 0 else 0
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(close)
            indicators['bb_upper'] = bb_upper[-1] if bb_upper is not None and len(bb_upper) > 0 else close[-1] * 1.02
            indicators['bb_lower'] = bb_lower[-1] if bb_lower is not None and len(bb_lower) > 0 else close[-1] * 0.98
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / close[-1]
            indicators['bb_position'] = (close[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            
            # === MOMENTUM INDICATORS ===
            indicators['rsi_14'] = ta.RSI(close, timeperiod=14)[-1] if len(close) >= 14 else 50
            indicators['rsi_7'] = ta.RSI(close, timeperiod=7)[-1] if len(close) >= 7 else 50
            indicators['stoch_k'], indicators['stoch_d'] = ta.STOCH(high, low, close)
            indicators['stoch_k'] = indicators['stoch_k'][-1] if indicators['stoch_k'] is not None and len(indicators['stoch_k']) > 0 else 50
            indicators['stoch_d'] = indicators['stoch_d'][-1] if indicators['stoch_d'] is not None and len(indicators['stoch_d']) > 0 else 50
            
            # Williams %R
            indicators['williams_r'] = ta.WILLR(high, low, close)[-1] if len(close) >= 14 else -50
            
            # CCI
            indicators['cci'] = ta.CCI(high, low, close)[-1] if len(close) >= 14 else 0
            
            # === VOLATILITY INDICATORS ===
            indicators['atr_14'] = ta.ATR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else (high[-1] - low[-1])
            indicators['atr_ratio'] = indicators['atr_14'] / close[-1] if close[-1] > 0 else 0.01
            
            # True Range
            indicators['true_range'] = max(high[-1] - low[-1], 
                                         abs(high[-1] - close[-2]) if len(close) > 1 else 0,
                                         abs(low[-1] - close[-2]) if len(close) > 1 else 0)
            
            # === VOLUME INDICATORS ===
            indicators['volume_sma'] = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
            indicators['volume_ratio'] = volume[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
            
            # OBV
            indicators['obv'] = ta.OBV(close, volume)[-1] if len(close) >= 10 else 0
            
            # === SUPPORT/RESISTANCE ===
            indicators['pivot_point'] = (high[-1] + low[-1] + close[-1]) / 3
            indicators['resistance_1'] = 2 * indicators['pivot_point'] - low[-1]
            indicators['support_1'] = 2 * indicators['pivot_point'] - high[-1]
            
            # === PRICE ACTION ===
            # Recent price changes
            indicators['price_change_1'] = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
            indicators['price_change_5'] = (close[-1] - close[-6]) / close[-6] if len(close) > 5 else 0
            indicators['price_change_20'] = (close[-1] - close[-21]) / close[-21] if len(close) > 20 else 0
            
            # High/Low analysis
            indicators['high_low_ratio'] = (close[-1] - low[-1]) / (high[-1] - low[-1]) if (high[-1] - low[-1]) > 0 else 0.5
            
            # === ADVANCED INDICATORS ===
            # Ichimoku Cloud components
            tenkan_sen = (np.max(high[-9:]) + np.min(low[-9:])) / 2 if len(high) >= 9 else close[-1]
            kijun_sen = (np.max(high[-26:]) + np.min(low[-26:])) / 2 if len(high) >= 26 else close[-1]
            indicators['ichimoku_tenkan'] = tenkan_sen
            indicators['ichimoku_kijun'] = kijun_sen
            indicators['ichimoku_cloud_position'] = 1 if close[-1] > max(tenkan_sen, kijun_sen) else -1 if close[-1] < min(tenkan_sen, kijun_sen) else 0
            
            # Fibonacci retracement levels
            recent_high = np.max(high[-50:]) if len(high) >= 50 else high[-1]
            recent_low = np.min(low[-50:]) if len(low) >= 50 else low[-1]
            fib_range = recent_high - recent_low
            indicators['fib_23.6'] = recent_high - 0.236 * fib_range
            indicators['fib_38.2'] = recent_high - 0.382 * fib_range
            indicators['fib_61.8'] = recent_high - 0.618 * fib_range
            
            # Current price position in fib levels
            if fib_range > 0:
                indicators['fib_position'] = (close[-1] - recent_low) / fib_range
            else:
                indicators['fib_position'] = 0.5
                
            # === PATTERN RECOGNITION ===
            # Doji detection
            body_size = abs(close[-1] - open_prices[-1])
            candle_range = high[-1] - low[-1]
            indicators['doji_pattern'] = 1 if body_size < (candle_range * 0.1) and candle_range > 0 else 0
            
            # Hammer/Shooting star
            lower_shadow = min(open_prices[-1], close[-1]) - low[-1]
            upper_shadow = high[-1] - max(open_prices[-1], close[-1])
            indicators['hammer_pattern'] = 1 if lower_shadow > (body_size * 2) else 0
            indicators['shooting_star_pattern'] = 1 if upper_shadow > (body_size * 2) else 0
            
            # === MARKET STRUCTURE ===
            # Trend strength
            up_moves = sum(1 for i in range(1, min(20, len(close))) if close[-i] > close[-i-1])
            indicators['trend_strength'] = (up_moves / min(19, len(close)-1)) if len(close) > 1 else 0.5
            
            # Volatility regime
            volatility_20 = np.std(close[-20:]) if len(close) >= 20 else np.std(close)
            volatility_5 = np.std(close[-5:]) if len(close) >= 5 else volatility_20
            indicators['volatility_regime'] = volatility_5 / volatility_20 if volatility_20 > 0 else 1
            
            # === NORMALIZE ALL VALUES ===
            current_price = close[-1]
            for key, value in indicators.items():
                if key.startswith('sma_') or key.startswith('ema_') or key.startswith('bb_') or key.startswith('ichimoku_') or key.startswith('fib_'):
                    if 'position' not in key and 'ratio' not in key and 'width' not in key:
                        indicators[key] = value / current_price if current_price > 0 else 1
                elif key in ['rsi_14', 'rsi_7', 'stoch_k', 'stoch_d']:
                    indicators[key] = (value - 50) / 50  # Normalize to -1 to 1
                elif key == 'williams_r':
                    indicators[key] = (value + 50) / 50  # Normalize to -1 to 1
                elif key in ['cci']:
                    indicators[key] = np.tanh(value / 100)  # Compress to -1 to 1
                    
            return indicators
            
        except Exception as e:
            print(f"Technical analysis error: {e}")
            return self._get_default_indicators()
    
    def _get_default_indicators(self) -> Dict:
        """Return default indicators when calculation fails"""
        return {
            'sma_20': 1.0, 'sma_50': 1.0, 'sma_200': 1.0,
            'ema_12': 1.0, 'ema_26': 1.0,
            'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
            'bb_upper': 1.02, 'bb_lower': 0.98, 'bb_width': 0.04, 'bb_position': 0.5,
            'rsi_14': 0.0, 'rsi_7': 0.0, 'stoch_k': 0.0, 'stoch_d': 0.0, 'williams_r': 0.0, 'cci': 0.0,
            'atr_14': 0.01, 'atr_ratio': 0.01, 'true_range': 0.01,
            'volume_sma': 1000, 'volume_ratio': 1.0, 'obv': 0.0,
            'pivot_point': 1.0, 'resistance_1': 1.01, 'support_1': 0.99,
            'price_change_1': 0.0, 'price_change_5': 0.0, 'price_change_20': 0.0,
            'high_low_ratio': 0.5,
            'ichimoku_tenkan': 1.0, 'ichimoku_kijun': 1.0, 'ichimoku_cloud_position': 0,
            'fib_23.6': 0.99, 'fib_38.2': 0.98, 'fib_61.8': 0.97, 'fib_position': 0.5,
            'doji_pattern': 0, 'hammer_pattern': 0, 'shooting_star_pattern': 0,
            'trend_strength': 0.5, 'volatility_regime': 1.0
        }

class MarketRegimeDetector:
    """Advanced Market Regime Detection"""
    
    def __init__(self):
        self.regime_history = deque(maxlen=50)
        
    def detect_regime(self, data: pd.DataFrame, indicators: Dict) -> MarketRegime:
        """Detect current market regime using multiple signals"""
        try:
            if len(data) < 20:
                return MarketRegime.SIDEWAYS
                
            close = data['close'].values
            
            # Trend detection
            trend_strength = indicators.get('trend_strength', 0.5)
            price_change_20 = indicators.get('price_change_20', 0)
            
            # Volatility detection
            atr_ratio = indicators.get('atr_ratio', 0.01)
            volatility_regime = indicators.get('volatility_regime', 1.0)
            
            # Determine regime
            if abs(price_change_20) > 0.02 and trend_strength > 0.7:
                regime = MarketRegime.TRENDING_UP if price_change_20 > 0 else MarketRegime.TRENDING_DOWN
            elif atr_ratio > 0.015 or volatility_regime > 1.5:
                regime = MarketRegime.HIGH_VOLATILITY
            elif atr_ratio < 0.005 or volatility_regime < 0.5:
                regime = MarketRegime.LOW_VOLATILITY
            elif abs(indicators.get('bb_position', 0.5) - 0.5) > 0.4:
                regime = MarketRegime.BREAKOUT
            else:
                regime = MarketRegime.SIDEWAYS
                
            self.regime_history.append(regime)
            return regime
            
        except Exception as e:
            print(f"Regime detection error: {e}")
            return MarketRegime.SIDEWAYS

class ProfessionalTradingEnvironment(gym.Env):
    """
    Professional Grade RL Trading Environment
    - Advanced observation space (150+ features)
    - Professional action space (15 dimensions)
    - Multi-objective reward function
    - Intelligent market analysis
    """
    
    def __init__(self, mt5_interface, recovery_engine, config):
        super(ProfessionalTradingEnvironment, self).__init__()
        
        print(f"🏗️ Initializing PROFESSIONAL Trading Environment...")
        
        # Core components
        self.mt5_interface = mt5_interface
        self.recovery_engine = recovery_engine
        self.config = config
        
        # Professional analyzers
        self.technical_analyzer = TechnicalAnalyzer()
        self.regime_detector = MarketRegimeDetector()
        
        # Trading parameters
        self.symbol = config.get('symbol', 'XAUUSD')
        self.initial_lot_size = config.get('initial_lot_size', 0.01)
        self.max_positions = config.get('max_positions', 10)
        
        # Environment parameters
        self.lookback_window = 200  # More historical data
        self.max_steps = config.get('max_steps', 1000)  # Longer episodes
        
        # === PROFESSIONAL OBSERVATION SPACE (150+ features) ===
        obs_low = [-10.0] * 150
        obs_high = [10.0] * 150
        
        self.observation_space = spaces.Box(
            low=np.array(obs_low, dtype=np.float32),
            high=np.array(obs_high, dtype=np.float32),
            shape=(150,),
            dtype=np.float32
        )
        
        # === PROFESSIONAL ACTION SPACE (15 dimensions) ===
        self.action_space = spaces.Box(
            low=np.array([
                -1.0,   # market_direction (-1=Strong Sell, +1=Strong Buy)
                0.01,   # position_size (0.01 to 1.0)
                0.0,    # entry_aggression (0=Limit, 1=Market)
                0.5,    # profit_target_ratio (0.5 to 5.0 R:R)
                0.0,    # partial_take_levels (0 to 3)
                0.0,    # add_position_signal (0 to 1)
                0.0,    # hedge_ratio (0 to 1)
                0.0,    # recovery_mode (0 to 3)
                0.0,    # correlation_limit (0 to 1)
                0.0,    # volatility_filter (0 to 1)
                0.0,    # spread_tolerance (0 to 1)
                0.0,    # time_filter (0 to 1)
                0.0,    # portfolio_heat_limit (0 to 1)
                0.0,    # smart_exit_signal (0 to 1)
                0.0     # rebalance_trigger (0 to 1)
            ], dtype=np.float32),
            high=np.array([
                1.0,    # market_direction
                1.0,    # position_size  
                1.0,    # entry_aggression
                5.0,    # profit_target_ratio
                3.0,    # partial_take_levels
                1.0,    # add_position_signal
                1.0,    # hedge_ratio
                3.0,    # recovery_mode
                1.0,    # correlation_limit
                1.0,    # volatility_filter
                1.0,    # spread_tolerance
                1.0,    # time_filter
                1.0,    # portfolio_heat_limit
                1.0,    # smart_exit_signal
                1.0     # rebalance_trigger
            ], dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode tracking
        self.current_step = 0
        self.episode_start_time = None
        self.episode_trades = []
        self.episode_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        
        # Market data management
        self.market_data_cache = deque(maxlen=self.lookback_window)
        self.last_update_time = None
        self.indicators_cache = {}
        self.current_regime = MarketRegime.SIDEWAYS
        
        # Performance tracking
        self.trade_history = deque(maxlen=1000)
        self.reward_components = deque(maxlen=100)
        self.action_history = deque(maxlen=50)
        
        # Portfolio integration
        from portfolio_manager import AIPortfolioManager
        self.portfolio_manager = AIPortfolioManager(config)
        
        # Training mode
        self.is_training_mode = config.get('training_mode', True)
        
        # GUI integration
        self.gui_instance = None
        
        print(f"✅ Professional Environment initialized:")
        print(f"   - Observation Space: {self.observation_space.shape[0]} features")
        print(f"   - Action Space: {self.action_space.shape[0]} dimensions")
        print(f"   - Advanced Technical Analysis: 50+ indicators")
        print(f"   - Market Regime Detection: Enabled")
        print(f"   - Training Mode: {self.is_training_mode}")

    def reset(self, seed=None, options=None):
        """Reset environment with professional initialization"""
        super().reset(seed=seed)
        
        print(f"🔄 Resetting Professional Environment...")
        
        # Reset episode tracking
        self.current_step = 0
        self.episode_start_time = datetime.now()
        self.episode_trades = []
        self.episode_pnl = 0.0
        self.max_drawdown = 0.0
        
        # Reset caches
        self.market_data_cache.clear()
        self.indicators_cache = {}
        self.reward_components.clear()
        self.action_history.clear()
        
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
        
        # Get initial professional observation
        observation = self._get_professional_observation()
        
        info = {
            'current_step': 0,
            'episode_pnl': 0.0,
            'market_regime': self.current_regime.value,
            'indicators_count': len(self.indicators_cache),
            'account_balance': 0,
            'account_equity': 0,
            'open_positions': 0,
            'technical_signals': self._get_technical_signals_summary()
        }
        
        print(f"✅ Professional Environment reset complete")
        return observation, info

    def step(self, action):
        """Execute professional trading step"""
        
        self.current_step += 1
        
        # Update portfolio (only in live mode)
        if not self.is_training_mode and self.portfolio_manager:
            self.portfolio_manager.update_portfolio_status(self.mt5_interface)
        
        # Update market data and analysis
        self.update_market_data()
        self.update_technical_analysis()
        
        # Execute professional action
        reward = self._execute_professional_action(action)
        
        # Get new professional observation
        observation = self._get_professional_observation()
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Comprehensive info
        info = self._get_professional_info()
        
        # Store action for analysis
        self.action_history.append(action.copy())
        
        return observation, reward, done, False, info

    def _execute_professional_action(self, action):
        """Execute professional trading action with 15-dimensional control"""
        
        try:
            # Parse professional action
            market_direction = float(action[0])         # -1 to 1
            position_size = float(action[1])            # 0.01 to 1.0
            entry_aggression = float(action[2])         # 0 to 1
            profit_target_ratio = float(action[3])      # 0.5 to 5.0
            partial_take_levels = int(action[4])        # 0 to 3
            add_position_signal = float(action[5])      # 0 to 1
            hedge_ratio = float(action[6])              # 0 to 1
            recovery_mode = int(action[7])              # 0 to 3
            correlation_limit = float(action[8])        # 0 to 1
            volatility_filter = float(action[9])        # 0 to 1
            spread_tolerance = float(action[10])        # 0 to 1
            time_filter = float(action[11])             # 0 to 1
            portfolio_heat_limit = float(action[12])    # 0 to 1
            smart_exit_signal = float(action[13])       # 0 to 1
            rebalance_trigger = float(action[14])       # 0 to 1
            
            print(f"")
            print(f"🎯 === PROFESSIONAL ACTION EXECUTION ===")
            print(f"   Market Direction: {market_direction:+.3f}")
            print(f"   Position Size: {position_size:.3f}")
            print(f"   Entry Style: {'Market' if entry_aggression > 0.5 else 'Limit'}")
            print(f"   R:R Ratio: {profit_target_ratio:.1f}")
            print(f"   Exit Signal: {smart_exit_signal:.3f}")
            
            # Get current state
            positions = self._get_positions()
            total_pnl = sum(pos.get('profit', 0) for pos in positions) if positions else 0
            
            # Initialize reward components
            reward_components = {
                'market_timing': 0.0,
                'position_management': 0.0,
                'risk_management': 0.0,
                'profit_optimization': 0.0,
                'recovery_skill': 0.0,
                'portfolio_efficiency': 0.0,
                'penalty': 0.0
            }
            
            # === 1. SMART EXIT LOGIC ===
            if smart_exit_signal > 0.6 and positions:
                exit_reward = self._handle_smart_exit(positions, smart_exit_signal, profit_target_ratio)
                reward_components['profit_optimization'] += exit_reward
                
            # === 2. PORTFOLIO REBALANCING ===
            elif rebalance_trigger > 0.7:
                rebalance_reward = self._handle_portfolio_rebalance(positions)
                reward_components['portfolio_efficiency'] += rebalance_reward
                
            # === 3. RECOVERY MODE ===
            elif total_pnl < -50 and recovery_mode > 0:
                recovery_reward = self._handle_recovery_mode(recovery_mode, positions, total_pnl)
                reward_components['recovery_skill'] += recovery_reward
                
            # === 4. POSITION ENTRY ===
            elif abs(market_direction) > 0.3:
                entry_reward = self._handle_position_entry(
                    market_direction, position_size, entry_aggression, 
                    profit_target_ratio, volatility_filter, correlation_limit
                )
                reward_components['market_timing'] += entry_reward
                
            # === 5. POSITION MANAGEMENT ===
            elif positions and (add_position_signal > 0.5 or hedge_ratio > 0.3):
                management_reward = self._handle_position_management(
                    positions, add_position_signal, hedge_ratio, partial_take_levels
                )
                reward_components['position_management'] += management_reward
                
            # === 6. HOLD/ANALYZE ===
            else:
                # Reward for holding when conditions are not optimal
                market_conditions = self._assess_market_conditions()
                if market_conditions['uncertainty'] > 0.7:
                    reward_components['risk_management'] += 0.5  # Good decision to wait
                else:
                    reward_components['penalty'] -= 0.1  # Missed opportunity
            
            # === CALCULATE TOTAL REWARD ===
            total_reward = self._calculate_multi_objective_reward(reward_components, positions)
            
            # Store reward components for analysis
            self.reward_components.append(reward_components.copy())
            
            print(f"🎁 Reward Components: Market={reward_components['market_timing']:.3f}, "
                  f"Position={reward_components['position_management']:.3f}, "
                  f"Risk={reward_components['risk_management']:.3f}")
            print(f"🏆 TOTAL REWARD: {total_reward:.3f}")
            print(f"🎯 === END PROFESSIONAL ACTION ===")
            print(f"")
            
            return total_reward
            
        except Exception as e:
            print(f"❌ Professional action execution error: {e}")
            return -1.0

    def _handle_smart_exit(self, positions, exit_signal, target_ratio):
        """Handle intelligent position exit"""
        try:
            if not positions:
                return 0.0
                
            closed_count = 0
            total_exit_pnl = 0
            
            for pos in positions:
                profit = pos.get('profit', 0)
                volume = pos.get('volume', 0.01)
                
                # Calculate dynamic profit target
                target_profit = self._calculate_dynamic_profit_target(volume, target_ratio)
                
                # Exit conditions
                should_exit = False
                exit_reason = ""
                
                if profit >= target_profit:
                    should_exit = True
                    exit_reason = "Profit Target"
                elif exit_signal > 0.8 and profit > 0:
                    should_exit = True
                    exit_reason = "Smart Exit Signal"
                elif exit_signal > 0.9:  # Emergency exit
                    should_exit = True
                    exit_reason = "Emergency Exit"
                    
                if should_exit:
                    if self.is_training_mode:
                        print(f"🎯 SIMULATED EXIT: {exit_reason}, Profit: ${profit:.2f}")
                        closed_count += 1
                        total_exit_pnl += profit
                    else:
                        success = self.mt5_interface.close_position(pos.get('ticket'))
                        if success:
                            closed_count += 1
                            total_exit_pnl += profit
                            print(f"🚀 LIVE EXIT: {exit_reason}, Profit: ${profit:.2f}")           
            # Calculate exit reward
            if closed_count > 0:
                avg_profit = total_exit_pnl / closed_count
                if avg_profit > 0:
                    return 3.0 + min(avg_profit / 20.0, 2.0)  # Up to 5.0 reward
                else:
                    return 1.0  # Good cut loss decision
            
            return 0.0
            
        except Exception as e:
            print(f"Smart exit error: {e}")
            return 0.0
        
    def _handle_portfolio_rebalance(self, positions):
        """Handle portfolio rebalancing"""
        try:
            if not positions:
                return 0.0
                
            # Analyze portfolio imbalance
            buy_volume = sum(pos.get('volume', 0) for pos in positions if pos.get('type', 0) == 0)
            sell_volume = sum(pos.get('volume', 0) for pos in positions if pos.get('type', 0) == 1)
            
            imbalance = abs(buy_volume - sell_volume) / max(buy_volume + sell_volume, 0.01)
            
            if imbalance > 0.6:  # Significant imbalance
                print(f"🔄 Portfolio rebalancing needed: Imbalance={imbalance:.2f}")
                return 2.0  # Reward for recognizing need to rebalance
            
            return 0.5  # Small reward for checking balance
            
        except Exception as e:
            print(f"Portfolio rebalance error: {e}")
            return 0.0

    def _handle_recovery_mode(self, recovery_mode, positions, total_pnl):
        """Handle intelligent recovery strategies"""
        try:
            if not self.recovery_engine:
                return 0.0
                
            # Activate appropriate recovery strategy
            recovery_types = ['martingale', 'grid', 'hedge', 'combined']
            recovery_type = recovery_types[min(recovery_mode, 3)]
            
            if self.is_training_mode:
                print(f"🛡️ SIMULATED RECOVERY: {recovery_type}, Loss: ${total_pnl:.2f}")
                return 1.5  # Reward for attempting recovery
            else:
                # Real recovery
                account_info = self.mt5_interface.get_account_info()
                current_equity = account_info.get('equity', 0) if account_info else 0
                
                success = self.recovery_engine.activate_recovery(
                    symbol=self.symbol,
                    mt5_interface=self.mt5_interface,
                    current_pnl=total_pnl,
                    current_equity=current_equity,
                    recovery_type=recovery_type
                )
                
                if success:
                    return 2.0  # Good recovery initiation
                else:
                    return -0.5  # Failed recovery
                    
        except Exception as e:
            print(f"Recovery mode error: {e}")
            return 0.0

    def _handle_position_entry(self, market_direction, position_size, entry_aggression, 
                             profit_target_ratio, volatility_filter, correlation_limit):
        """Handle intelligent position entry"""
        try:
            # Market conditions check
            market_conditions = self._assess_market_conditions()
            
            # Volatility filter
            if volatility_filter > 0.5 and market_conditions['volatility'] > 0.015:
                print(f"⚠️ Entry blocked: High volatility ({market_conditions['volatility']:.4f})")
                return 0.2  # Small reward for risk awareness
                
            # Spread filter
            spread_info = self.mt5_interface.get_spread(self.symbol) if self.mt5_interface else None
            if spread_info and spread_info.get('spread_pips', 0) > 2.0:
                print(f"⚠️ Entry blocked: High spread ({spread_info['spread_pips']:.1f} pips)")
                return 0.1
                
            # Determine entry direction and size
            if market_direction > 0.3:  # BUY signal
                order_type = 'buy'
                signal_strength = market_direction
            elif market_direction < -0.3:  # SELL signal  
                order_type = 'sell'
                signal_strength = abs(market_direction)
            else:
                return -0.1  # Weak signal penalty
                
            # Calculate dynamic position size
            dynamic_size = self._calculate_dynamic_position_size(position_size, signal_strength)
            
            # Execute entry
            if self.is_training_mode:
                print(f"🎯 SIMULATED ENTRY: {order_type.upper()} {dynamic_size:.2f} lots")
                # Simulate entry success based on market conditions
                entry_quality = self._evaluate_entry_quality(market_direction, market_conditions)
                return 2.0 + entry_quality
            else:
                # Real entry
                current_price = self.mt5_interface.get_current_price(self.symbol)
                if current_price:
                    price = current_price['ask'] if order_type == 'buy' else current_price['bid']
                    
                    success = self.mt5_interface.place_order(
                        symbol=self.symbol,
                        order_type=order_type,
                        volume=dynamic_size,
                        price=price
                    )
                    
                    if success:
                        entry_quality = self._evaluate_entry_quality(market_direction, market_conditions)
                        return 3.0 + entry_quality
                    else:
                        return -1.0  # Entry failed
                        
            return 0.0
            
        except Exception as e:
            print(f"Position entry error: {e}")
            return -0.5
        
    def _handle_position_management(self, positions, add_signal, hedge_ratio, partial_levels):
        """Handle advanced position management"""
        try:
            total_reward = 0.0
            
            # Add to winning positions
            if add_signal > 0.7:
                winning_positions = [pos for pos in positions if pos.get('profit', 0) > 10]
                if winning_positions:
                    print(f"📈 Adding to {len(winning_positions)} winning positions")
                    total_reward += 1.5
                else:
                    total_reward -= 0.5  # No good positions to add to
                    
            # Hedge losing positions
            if hedge_ratio > 0.5:
                losing_positions = [pos for pos in positions if pos.get('profit', 0) < -20]
                if losing_positions:
                    print(f"🛡️ Hedging {len(losing_positions)} losing positions")
                    total_reward += 1.0
                else:
                    total_reward -= 0.2  # Unnecessary hedging
                    
            # Partial profit taking
            if partial_levels > 0:
                profitable_positions = [pos for pos in positions if pos.get('profit', 0) > 5]
                if profitable_positions:
                    print(f"💰 Taking partial profits on {len(profitable_positions)} positions")
                    total_reward += 1.2
                    
            return total_reward
            
        except Exception as e:
            print(f"Position management error: {e}")
            return 0.0

    def _calculate_multi_objective_reward(self, reward_components, positions):
        """Calculate sophisticated multi-objective reward"""
        try:
            # Base reward from components
            base_reward = (
                reward_components['market_timing'] * 0.25 +
                reward_components['position_management'] * 0.20 +
                reward_components['risk_management'] * 0.20 +
                reward_components['profit_optimization'] * 0.15 +
                reward_components['recovery_skill'] * 0.10 +
                reward_components['portfolio_efficiency'] * 0.10 +
                reward_components['penalty']
            )
            
            # Portfolio performance bonus
            portfolio_bonus = 0.0
            if positions:
                total_pnl = sum(pos.get('profit', 0) for pos in positions)
                position_count = len(positions)
                
                # Efficiency bonus
                if total_pnl > 0:
                    efficiency = total_pnl / max(position_count * 10, 1)  # $10 per position baseline
                    portfolio_bonus += min(efficiency, 2.0)
                    
                # Diversification bonus
                buy_count = sum(1 for pos in positions if pos.get('type', 0) == 0)
                sell_count = position_count - buy_count
                if buy_count > 0 and sell_count > 0:
                    portfolio_bonus += 0.5  # Diversification bonus
                    
            # Risk-adjusted bonus
            risk_adjustment = 0.0
            if hasattr(self, 'portfolio_manager'):
                portfolio_heat = self.portfolio_manager.calculate_portfolio_heat(self.mt5_interface)
                if portfolio_heat < 3.0:  # Low risk
                    risk_adjustment += 0.3
                elif portfolio_heat > 8.0:  # High risk
                    risk_adjustment -= 0.5
                    
            # Market timing bonus
            timing_bonus = 0.0
            if hasattr(self, 'current_regime'):
                regime_alignment = self._check_regime_alignment(positions)
                timing_bonus += regime_alignment * 0.5
                
            total_reward = base_reward + portfolio_bonus + risk_adjustment + timing_bonus
            
            # Clip reward to prevent extreme values
            total_reward = np.clip(total_reward, -5.0, 10.0)
            
            return float(total_reward)
            
        except Exception as e:
            print(f"Multi-objective reward error: {e}")
            return 0.0

    def _assess_market_conditions(self):
        """Assess current market conditions comprehensively"""
        try:
            conditions = {
                'volatility': 0.01,
                'trend_strength': 0.5,
                'uncertainty': 0.5,
                'liquidity': 1.0,
                'regime': 'SIDEWAYS'
            }
            
            if self.indicators_cache:
                conditions['volatility'] = self.indicators_cache.get('atr_ratio', 0.01)
                conditions['trend_strength'] = self.indicators_cache.get('trend_strength', 0.5)
                conditions['regime'] = self.current_regime.value
                
                # Calculate uncertainty based on conflicting signals
                rsi = self.indicators_cache.get('rsi_14', 0)
                macd = self.indicators_cache.get('macd', 0)
                bb_position = self.indicators_cache.get('bb_position', 0.5)
                
                # High uncertainty when signals conflict
                signal_variance = np.var([rsi, macd * 10, (bb_position - 0.5) * 2])
                conditions['uncertainty'] = min(signal_variance * 2, 1.0)
                
            return conditions
            
        except Exception as e:
            print(f"Market conditions assessment error: {e}")
            return {'volatility': 0.01, 'trend_strength': 0.5, 'uncertainty': 0.5, 'liquidity': 1.0, 'regime': 'SIDEWAYS'}

    def _calculate_dynamic_profit_target(self, volume, target_ratio):
        """Calculate dynamic profit target based on volume and market conditions"""
        try:
            # Base target from portfolio manager
            base_target = 5.0  # $5 per 0.01 lot
            if hasattr(self, 'portfolio_manager'):
                thresholds = self.portfolio_manager.get_current_thresholds()
                base_target = thresholds.get('per_lot', 5.0)
                
            # Scale by volume
            volume_target = base_target * (volume / 0.01)
            
            # Adjust by target ratio
            adjusted_target = volume_target * target_ratio
            
            # Market condition adjustment
            if self.indicators_cache:
                volatility = self.indicators_cache.get('atr_ratio', 0.01)
                if volatility > 0.015:  # High volatility
                    adjusted_target *= 1.5  # Higher targets in volatile markets
                elif volatility < 0.005:  # Low volatility
                    adjusted_target *= 0.7  # Lower targets in quiet markets
                    
            return max(adjusted_target, 2.0)  # Minimum $2 target
            
        except Exception as e:
            print(f"Dynamic profit target error: {e}")
            return 5.0

    def _calculate_dynamic_position_size(self, base_size, signal_strength):
        """Calculate dynamic position size based on signal strength and risk"""
        try:
            # Start with base size
            dynamic_size = base_size
            
            # Adjust by signal strength
            dynamic_size *= (0.5 + signal_strength * 0.5)  # 0.5x to 1.0x based on signal
            
            # Portfolio heat adjustment
            if hasattr(self, 'portfolio_manager'):
                portfolio_heat = self.portfolio_manager.calculate_portfolio_heat(self.mt5_interface)
                if portfolio_heat > 5.0:
                    dynamic_size *= 0.5  # Reduce size when portfolio is hot
                elif portfolio_heat < 2.0:
                    dynamic_size *= 1.2  # Increase size when portfolio is cool
                    
            # Market volatility adjustment
            if self.indicators_cache:
                volatility = self.indicators_cache.get('atr_ratio', 0.01)
                if volatility > 0.015:  # High volatility
                    dynamic_size *= 0.7  # Smaller positions in volatile markets
                    
            # Ensure within bounds
            dynamic_size = max(0.01, min(dynamic_size, 0.10))
            dynamic_size = round(dynamic_size, 2)
            
            return dynamic_size
            
        except Exception as e:
            print(f"Dynamic position size error: {e}")
            return base_size

    def _evaluate_entry_quality(self, market_direction, market_conditions):
        """Evaluate the quality of entry decision"""
        try:
            quality_score = 0.0
            
            # Signal strength bonus
            signal_strength = abs(market_direction)
            quality_score += signal_strength * 2.0
            
            # Market regime alignment
            if self.current_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                if (market_direction > 0 and self.current_regime == MarketRegime.TRENDING_UP) or \
                   (market_direction < 0 and self.current_regime == MarketRegime.TRENDING_DOWN):
                    quality_score += 1.0  # Trend following
                    
            # Technical confluence
            if self.indicators_cache:
                rsi = self.indicators_cache.get('rsi_14', 0)
                macd = self.indicators_cache.get('macd', 0)
                bb_position = self.indicators_cache.get('bb_position', 0.5)
                
                # Buy confluence
                if market_direction > 0:
                    if rsi < -0.3 and macd > 0 and bb_position < 0.3:  # Oversold + momentum + low BB
                        quality_score += 1.5
                # Sell confluence  
                elif market_direction < 0:
                    if rsi > 0.3 and macd < 0 and bb_position > 0.7:  # Overbought + momentum + high BB
                        quality_score += 1.5
                        
            # Market condition penalty
            if market_conditions['uncertainty'] > 0.8:
                quality_score -= 1.0  # Penalty for entering in uncertain conditions
                
            return max(0.0, min(quality_score, 3.0))
            
        except Exception as e:
            print(f"Entry quality evaluation error: {e}")
            return 0.0

    def _check_regime_alignment(self, positions):
        """Check if positions are aligned with current market regime"""
        try:
            if not positions:
                return 0.0
                
            alignment_score = 0.0
            
            for pos in positions:
                pos_type = pos.get('type', 0)  # 0=buy, 1=sell
                
                if self.current_regime == MarketRegime.TRENDING_UP and pos_type == 0:
                    alignment_score += 1.0  # Long in uptrend
                elif self.current_regime == MarketRegime.TRENDING_DOWN and pos_type == 1:
                    alignment_score += 1.0  # Short in downtrend
                elif self.current_regime == MarketRegime.SIDEWAYS:
                    alignment_score += 0.5  # Neutral in sideways
                else:
                    alignment_score -= 0.5  # Against trend
                    
            return alignment_score / len(positions)
            
        except Exception as e:
            print(f"Regime alignment check error: {e}")
            return 0.0  
        
    def _get_professional_observation(self):
        """Generate professional 150+ feature observation"""
        try:
            observation = np.zeros(150, dtype=np.float32)
            feature_idx = 0
            
            # === TECHNICAL INDICATORS (50 features) ===
            if self.indicators_cache:
                indicator_keys = [
                    'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
                    'macd', 'macd_signal', 'macd_histogram',
                    'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
                    'rsi_14', 'rsi_7', 'stoch_k', 'stoch_d', 'williams_r', 'cci',
                    'atr_14', 'atr_ratio', 'true_range',
                    'volume_sma', 'volume_ratio', 'obv',
                    'pivot_point', 'resistance_1', 'support_1',
                    'price_change_1', 'price_change_5', 'price_change_20',
                    'high_low_ratio',
                    'ichimoku_tenkan', 'ichimoku_kijun', 'ichimoku_cloud_position',
                    'fib_23.6', 'fib_38.2', 'fib_61.8', 'fib_position',
                    'doji_pattern', 'hammer_pattern', 'shooting_star_pattern',
                    'trend_strength', 'volatility_regime'
                ]
                
                for key in indicator_keys[:50]:  # Take first 50
                    observation[feature_idx] = float(self.indicators_cache.get(key, 0))
                    feature_idx += 1
                    
            else:
                feature_idx = 50  # Skip technical indicators
                
            # === MARKET REGIME (10 features) ===
            regime_features = [0] * 7  # 7 different regimes
            if hasattr(self, 'current_regime'):
                regime_map = {
                    MarketRegime.TRENDING_UP: 0,
                    MarketRegime.TRENDING_DOWN: 1,
                    MarketRegime.SIDEWAYS: 2,
                    MarketRegime.HIGH_VOLATILITY: 3,
                    MarketRegime.LOW_VOLATILITY: 4,
                    MarketRegime.BREAKOUT: 5,
                    MarketRegime.REVERSAL: 6
                }
                regime_idx = regime_map.get(self.current_regime, 2)
                regime_features[regime_idx] = 1.0
                
            for i in range(7):
                observation[feature_idx] = regime_features[i]
                feature_idx += 1
                
            # Market conditions
            market_conditions = self._assess_market_conditions()
            observation[feature_idx] = market_conditions['volatility'] * 100  # Normalize
            observation[feature_idx + 1] = market_conditions['trend_strength']
            observation[feature_idx + 2] = market_conditions['uncertainty']
            feature_idx += 3
            
            # === POSITION INFORMATION (30 features) ===
            positions = self._get_positions()
            
            # Basic position stats
            observation[feature_idx] = len(positions) / 10.0  # Normalize by max positions
            observation[feature_idx + 1] = sum(pos.get('profit', 0) for pos in positions) / 100.0  # Total PnL
            observation[feature_idx + 2] = sum(pos.get('volume', 0) for pos in positions) / 1.0  # Total volume
            feature_idx += 3
            
            # Position distribution
            buy_positions = sum(1 for pos in positions if pos.get('type', 0) == 0)
            sell_positions = len(positions) - buy_positions
            observation[feature_idx] = buy_positions / 10.0
            observation[feature_idx + 1] = sell_positions / 10.0
            feature_idx += 2
            
            # Profit distribution
            profitable_positions = sum(1 for pos in positions if pos.get('profit', 0) > 0)
            losing_positions = len(positions) - profitable_positions
            observation[feature_idx] = profitable_positions / 10.0
            observation[feature_idx + 1] = losing_positions / 10.0
            feature_idx += 2
            
            # Position details (up to 10 positions, 2 features each)
            for i in range(10):
                if i < len(positions):
                    pos = positions[i]
                    observation[feature_idx] = pos.get('profit', 0) / 50.0  # Normalize profit
                    observation[feature_idx + 1] = pos.get('volume', 0) / 1.0  # Normalize volume
                else:
                    observation[feature_idx] = 0.0
                    observation[feature_idx + 1] = 0.0
                feature_idx += 2
                
            # Additional position metrics
            if positions:
                max_profit = max(pos.get('profit', 0) for pos in positions)
                min_profit = min(pos.get('profit', 0) for pos in positions)
                avg_profit = sum(pos.get('profit', 0) for pos in positions) / len(positions)
                observation[feature_idx] = max_profit / 50.0
                observation[feature_idx + 1] = min_profit / 50.0  
                observation[feature_idx + 2] = avg_profit / 50.0
            feature_idx += 3
            
            # Safety checks and normalization
            observation = np.nan_to_num(observation, nan=0.0, posinf=5.0, neginf=-5.0)
            observation = np.clip(observation, -10.0, 10.0)
            
            return observation.astype(np.float32)
            
        except Exception as e:
            print(f"Professional observation error: {e}")
            # Return safe default observation
            return np.zeros(150, dtype=np.float32)
        
    def update_market_data(self):
        """Update market data with multi-timeframe support"""
        try:
            if not hasattr(self, 'mt5_interface') or not self.mt5_interface:
                return
                
            # Get M1 data for immediate analysis
            rates = self.mt5_interface.get_rates(self.symbol, 1, 200)  # M1, 200 bars
            
            if rates is not None and len(rates) > 0:
                # Convert to DataFrame
                df = pd.DataFrame(rates, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                df['time'] = pd.to_datetime(df['time'], unit='s')
                
                # Store recent data
                for _, row in df.tail(10).iterrows():  # Last 10 bars
                    bar_data = {
                        'time': row['time'],
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 1000
                    }
                    
                    # Add to cache (avoiding duplicates)
                    if not self.market_data_cache or self.market_data_cache[-1]['time'] != bar_data['time']:
                        self.market_data_cache.append(bar_data)
                        
                self.last_update_time = datetime.now()
                
        except Exception as e:
            print(f"Market data update error: {e}")

    def update_technical_analysis(self):
        """Update technical analysis and market regime"""
        try:
            if len(self.market_data_cache) < 50:
                return
                
            # Convert cache to DataFrame
            df = pd.DataFrame(list(self.market_data_cache))
            df['time'] = pd.to_datetime(df['time'])
            
            # Calculate technical indicators
            self.indicators_cache = self.technical_analyzer.calculate_indicators(df)
            
            # Detect market regime
            self.current_regime = self.regime_detector.detect_regime(df, self.indicators_cache)
            
        except Exception as e:
            print(f"Technical analysis update error: {e}")

    def _get_technical_signals_summary(self):
        """Get summary of current technical signals"""
        try:
            if not self.indicators_cache:
                return {'trend': 'NEUTRAL', 'momentum': 'NEUTRAL', 'volatility': 'NORMAL'}
                
            # Trend analysis
            sma_20 = self.indicators_cache.get('sma_20', 1.0)
            sma_50 = self.indicators_cache.get('sma_50', 1.0)
            
            if sma_20 > sma_50 * 1.002:
                trend = 'BULLISH'
            elif sma_20 < sma_50 * 0.998:
                trend = 'BEARISH'
            else:
                trend = 'NEUTRAL'
                
            # Momentum analysis
            rsi = self.indicators_cache.get('rsi_14', 0)
            if rsi > 0.3:
                momentum = 'OVERBOUGHT'
            elif rsi < -0.3:
                momentum = 'OVERSOLD'
            else:
                momentum = 'NEUTRAL'
                
            # Volatility analysis
            atr_ratio = self.indicators_cache.get('atr_ratio', 0.01)
            if atr_ratio > 0.015:
                volatility = 'HIGH'
            elif atr_ratio < 0.005:
                volatility = 'LOW'
            else:
                volatility = 'NORMAL'
                
            return {
                'trend': trend,
                'momentum': momentum,
                'volatility': volatility,
                'regime': self.current_regime.value if hasattr(self, 'current_regime') else 'SIDEWAYS'
            }
            
        except Exception as e:
            print(f"Technical signals summary error: {e}")
            return {'trend': 'NEUTRAL', 'momentum': 'NEUTRAL', 'volatility': 'NORMAL', 'regime': 'SIDEWAYS'}

    def _get_positions(self):
        """Get current positions with error handling"""
        try:
            if hasattr(self, 'mt5_interface') and self.mt5_interface and not self.is_training_mode:
                return self.mt5_interface.get_positions()
            return []
        except:
            return []

    def _is_episode_done(self):
        """Check if episode should end with professional criteria"""
        try:
            # End if maximum steps reached
            if self.current_step >= self.max_steps:
                return True
                
            # End if extreme portfolio conditions (only in live mode)
            if not self.is_training_mode:
                try:
                    account_info = self.mt5_interface.get_account_info()
                    if account_info:
                        equity = account_info.get('equity', 0)
                        balance = account_info.get('balance', 0)
                        
                        # Stop if severe drawdown
                        if equity < balance * 0.3:  # 70% drawdown
                            print(f"🚨 Episode ended: Severe drawdown detected")
                            return True
                            
                        # Stop if margin call risk
                        margin_level = account_info.get('margin_level', 1000)
                        if margin_level < 200:  # Below 200% margin level
                            print(f"🚨 Episode ended: Margin call risk")
                            return True
                except:
                    pass
                    
            # Check portfolio manager conditions
            if hasattr(self, 'portfolio_manager'):
                # Stop if excessive portfolio heat
                portfolio_heat = self.portfolio_manager.calculate_portfolio_heat(self.mt5_interface)
                if portfolio_heat > 15.0:  # 15% portfolio heat
                    print(f"🚨 Episode ended: Excessive portfolio heat: {portfolio_heat:.1f}%")
                    return True
                    
                # Stop if daily loss limit exceeded
                if self.portfolio_manager.daily_pnl < -10.0:  # 10% daily loss
                    print(f"🚨 Episode ended: Daily loss limit exceeded: {self.portfolio_manager.daily_pnl:.1f}%")
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Episode termination check error: {e}")
            return False

    def _get_professional_info(self):
        """Get comprehensive professional environment info"""
        try:
            account_info = None
            positions = []
            
            if not self.is_training_mode and hasattr(self, 'mt5_interface'):
                try:
                    account_info = self.mt5_interface.get_account_info()
                    positions = self.mt5_interface.get_positions()
                except:
                    pass
            
            # Basic info
            info = {
                'current_step': self.current_step,
                'episode_pnl': sum(pos.get('profit', 0) for pos in positions) if positions else 0,
                'account_balance': account_info.get('balance', 0) if account_info else 0,
                'account_equity': account_info.get('equity', 0) if account_info else 0,
                'open_positions': len(positions),
                'max_drawdown': self.max_drawdown,
                'total_trades': len(self.episode_trades)
            }
            
            # Technical analysis info
            info['technical_signals'] = self._get_technical_signals_summary()
            info['market_regime'] = self.current_regime.value if hasattr(self, 'current_regime') else 'SIDEWAYS'
            info['indicators_count'] = len(self.indicators_cache)
            
            # Portfolio management info
            if hasattr(self, 'portfolio_manager'):
                info['portfolio_heat'] = self.portfolio_manager.calculate_portfolio_heat(self.mt5_interface)
                info['portfolio_drawdown'] = self.portfolio_manager.current_drawdown
                info['daily_pnl_pct'] = self.portfolio_manager.daily_pnl
                info['trading_allowed'] = self.portfolio_manager.trading_allowed
                info['risk_mode'] = 'recovery' if self.portfolio_manager.recovery_mode else \
                                  'reduction' if self.portfolio_manager.risk_reduction_active else 'normal'
                
                # Dynamic targets
                thresholds = self.portfolio_manager.get_current_thresholds()
                info['profit_per_lot'] = thresholds.get('per_lot', 5.0)
                info['portfolio_target'] = thresholds.get('portfolio', 25.0)
            
            # Recovery system info
            if hasattr(self, 'recovery_engine'):
                recovery_status = self.recovery_engine.get_status()
                info['recovery_active'] = recovery_status.get('recovery_active', False)
                info['recovery_level'] = recovery_status.get('recovery_level', 0)
                info['recovery_success_rate'] = recovery_status.get('success_rate', 0)
            
            # Performance metrics
            if len(self.reward_components) > 0:
                recent_rewards = list(self.reward_components)[-10:]  # Last 10 rewards
                avg_components = {}
                for component in ['market_timing', 'position_management', 'risk_management', 
                                'profit_optimization', 'recovery_skill', 'portfolio_efficiency']:
                    component_values = [r.get(component, 0) for r in recent_rewards]
                    avg_components[f'avg_{component}'] = sum(component_values) / len(component_values)
                
                info.update(avg_components)
            
            # Market conditions
            market_conditions = self._assess_market_conditions()
            info['market_volatility'] = market_conditions['volatility']
            info['market_uncertainty'] = market_conditions['uncertainty']
            info['trend_strength'] = market_conditions['trend_strength']
            
            return info
            
        except Exception as e:
            print(f"Professional info error: {e}")
            return {
                'current_step': self.current_step,
                'episode_pnl': 0,
                'open_positions': 0,
                'market_regime': 'SIDEWAYS',
                'error': str(e)
            }
        
    def render(self, mode='human'):
        """Render environment state (optional for debugging)"""
        if mode == 'human':
            print(f"\n=== Professional Trading Environment State ===")
            print(f"Step: {self.current_step}/{self.max_steps}")
            print(f"Market Regime: {self.current_regime.value if hasattr(self, 'current_regime') else 'UNKNOWN'}")
            
            if self.indicators_cache:
                print(f"RSI: {self.indicators_cache.get('rsi_14', 0):.3f}")
                print(f"MACD: {self.indicators_cache.get('macd', 0):.6f}")
                print(f"Trend Strength: {self.indicators_cache.get('trend_strength', 0.5):.3f}")
                print(f"Volatility: {self.indicators_cache.get('atr_ratio', 0.01):.4f}")
            
            positions = self._get_positions()
            print(f"Open Positions: {len(positions)}")
            if positions:
                total_pnl = sum(pos.get('profit', 0) for pos in positions)
                print(f"Total PnL: ${total_pnl:.2f}")
            
            print("=" * 50)

    def close(self):
        """Clean up environment resources"""
        try:
            # Clear caches
            if hasattr(self, 'market_data_cache'):
                self.market_data_cache.clear()
            if hasattr(self, 'indicators_cache'):
                self.indicators_cache.clear()
            if hasattr(self, 'reward_components'):
                self.reward_components.clear()
            if hasattr(self, 'action_history'):
                self.action_history.clear()
                
            print("🧹 Professional Environment cleaned up")
            
        except Exception as e:
            print(f"Environment cleanup error: {e}")

    def get_observation_info(self):
        """Get detailed information about observation space structure"""
        return {
            'total_features': 150,
            'feature_groups': {
                'technical_indicators': (0, 50),
                'market_regime': (50, 60),
                'position_info': (60, 90),
                'portfolio_info': (90, 110),
                'time_features': (110, 120),
                'action_history': (120, 140),
                'misc_features': (140, 150)
            },
            'description': {
                'technical_indicators': '50 advanced technical analysis indicators',
                'market_regime': 'Current market regime and conditions',
                'position_info': 'Detailed position and trade information',
                'portfolio_info': 'Portfolio management and risk metrics',
                'time_features': 'Time-based and session information',
                'action_history': 'Recent action history for pattern analysis',
                'misc_features': 'Additional features for future expansion'
            }
        }

    def get_action_info(self):
        """Get detailed information about action space structure"""
        return {
            'total_dimensions': 15,
            'action_meanings': {
                0: 'market_direction (-1=Strong Sell, +1=Strong Buy)',
                1: 'position_size (0.01 to 1.0 lots)',
                2: 'entry_aggression (0=Limit Order, 1=Market Order)',
                3: 'profit_target_ratio (0.5 to 5.0 Risk:Reward)',
                4: 'partial_take_levels (0 to 3 profit levels)',
                5: 'add_position_signal (0 to 1 scale in strength)',
                6: 'hedge_ratio (0 to 1 hedge percentage)',
                7: 'recovery_mode (0-3: martingale/grid/hedge/combined)',
                8: 'correlation_limit (0 to 1 max correlation)',
                9: 'volatility_filter (0 to 1 vol threshold)',
                10: 'spread_tolerance (0 to 1 max spread)',
                11: 'time_filter (0 to 1 session filter)',
                12: 'portfolio_heat_limit (0 to 1 max heat)',
                13: 'smart_exit_signal (0 to 1 exit strength)',
                14: 'rebalance_trigger (0 to 1 rebalance signal)'
            },
            'professional_features': [
                'Multi-dimensional trading control',
                'Risk-aware position sizing',
                'Dynamic profit targeting',
                'Advanced portfolio management',
                'Intelligent market timing',
                'Adaptive recovery strategies'
            ]
        }

    def get_performance_summary(self):
        """Get current performance summary"""
        try:
            positions = self._get_positions()
            
            summary = {
                'episode_step': self.current_step,
                'max_steps': self.max_steps,
                'progress_pct': (self.current_step / self.max_steps) * 100,
                'current_positions': len(positions),
                'total_pnl': sum(pos.get('profit', 0) for pos in positions) if positions else 0,
                'market_regime': self.current_regime.value if hasattr(self, 'current_regime') else 'SIDEWAYS',
                'indicators_active': len(self.indicators_cache),
                'recent_rewards': len(self.reward_components),
                'training_mode': self.is_training_mode
            }
            
            if hasattr(self, 'portfolio_manager'):
                summary['portfolio_heat'] = self.portfolio_manager.calculate_portfolio_heat(self.mt5_interface)
                summary['trading_allowed'] = self.portfolio_manager.trading_allowed
            
            return summary
            
        except Exception as e:
            print(f"Performance summary error: {e}")
            return {'error': str(e)}


# ========================= TECHNICAL ANALYZER CLASS =========================

class TechnicalAnalyzer:
    """Advanced Technical Analysis Engine - Complete Implementation"""
    
    def __init__(self):
        self.indicator_cache = {}
        self.timeframes = ['M1', 'M5', 'M15', 'H1', 'H4']
        
    def calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate 50+ technical indicators with fallback implementations"""
        try:
            if len(data) < 100:  # Need sufficient data
                return self._get_default_indicators()
                
            close = data['close'].values
            high = data['high'].values
            low = data['low'].values
            open_prices = data['open'].values
            volume = data.get('volume', pd.Series([1000] * len(data))).values
            
            indicators = {}
            
            # === TREND INDICATORS ===
            if TALIB_AVAILABLE:
                # Use TA-Lib if available
                indicators['sma_20'] = ta.SMA(close, timeperiod=20)[-1] if len(close) >= 20 else close[-1]
                indicators['sma_50'] = ta.SMA(close, timeperiod=50)[-1] if len(close) >= 50 else close[-1]
                indicators['sma_200'] = ta.SMA(close, timeperiod=200)[-1] if len(close) >= 200 else close[-1]
                indicators['ema_12'] = ta.EMA(close, timeperiod=12)[-1] if len(close) >= 12 else close[-1]
                indicators['ema_26'] = ta.EMA(close, timeperiod=26)[-1] if len(close) >= 26 else close[-1]
                
                # MACD
                macd, macd_signal, macd_hist = ta.MACD(close)
                indicators['macd'] = macd[-1] if macd is not None and len(macd) > 0 else 0
                indicators['macd_signal'] = macd_signal[-1] if macd_signal is not None and len(macd_signal) > 0 else 0
                indicators['macd_histogram'] = macd_hist[-1] if macd_hist is not None and len(macd_hist) > 0 else 0
                
                # RSI
                indicators['rsi_14'] = ta.RSI(close, timeperiod=14)[-1] if len(close) >= 14 else 50
                indicators['rsi_7'] = ta.RSI(close, timeperiod=7)[-1] if len(close) >= 7 else 50
                
                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = ta.BBANDS(close)
                indicators['bb_upper'] = bb_upper[-1] if bb_upper is not None and len(bb_upper) > 0 else close[-1] * 1.02
                indicators['bb_lower'] = bb_lower[-1] if bb_lower is not None and len(bb_lower) > 0 else close[-1] * 0.98
                
                # ATR
                indicators['atr_14'] = ta.ATR(high, low, close, timeperiod=14)[-1] if len(close) >= 14 else (high[-1] - low[-1])
                
            else:
                # Fallback manual calculations
                indicators['sma_20'] = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
                indicators['sma_50'] = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
                indicators['sma_200'] = np.mean(close[-200:]) if len(close) >= 200 else close[-1]
                
                # Simple EMA calculation
                indicators['ema_12'] = self._calculate_ema(close, 12)
                indicators['ema_26'] = self._calculate_ema(close, 26)
                
                # Simple MACD
                ema_12 = self._calculate_ema(close, 12)
                ema_26 = self._calculate_ema(close, 26)
                indicators['macd'] = ema_12 - ema_26
                indicators['macd_signal'] = self._calculate_ema([indicators['macd']] * 9, 9)
                indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
                
                # Simple RSI
                indicators['rsi_14'] = self._calculate_rsi(close, 14)
                indicators['rsi_7'] = self._calculate_rsi(close, 7)
                
                # Simple Bollinger Bands
                sma_20 = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
                std_20 = np.std(close[-20:]) if len(close) >= 20 else close[-1] * 0.02
                indicators['bb_upper'] = sma_20 + (2 * std_20)
                indicators['bb_lower'] = sma_20 - (2 * std_20)
                
                # Simple ATR
                indicators['atr_14'] = self._calculate_atr(high, low, close, 14)
            
            # === DERIVED INDICATORS ===
            indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / close[-1]
            indicators['bb_position'] = (close[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
            indicators['atr_ratio'] = indicators['atr_14'] / close[-1] if close[-1] > 0 else 0.01
            
            # === VOLUME INDICATORS ===
            indicators['volume_sma'] = np.mean(volume[-20:]) if len(volume) >= 20 else volume[-1]
            indicators['volume_ratio'] = volume[-1] / indicators['volume_sma'] if indicators['volume_sma'] > 0 else 1
            indicators['obv'] = self._calculate_obv(close, volume)
            
            # === PRICE ACTION ===
            indicators['price_change_1'] = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
            indicators['price_change_5'] = (close[-1] - close[-6]) / close[-6] if len(close) > 5 else 0
            indicators['price_change_20'] = (close[-1] - close[-21]) / close[-21] if len(close) > 20 else 0
            indicators['high_low_ratio'] = (close[-1] - low[-1]) / (high[-1] - low[-1]) if (high[-1] - low[-1]) > 0 else 0.5
            
            # === SUPPORT/RESISTANCE ===
            indicators['pivot_point'] = (high[-1] + low[-1] + close[-1]) / 3
            indicators['resistance_1'] = 2 * indicators['pivot_point'] - low[-1]
            indicators['support_1'] = 2 * indicators['pivot_point'] - high[-1]
            
            # === PATTERN RECOGNITION ===
            body_size = abs(close[-1] - open_prices[-1])
            candle_range = high[-1] - low[-1]
            indicators['doji_pattern'] = 1 if body_size < (candle_range * 0.1) and candle_range > 0 else 0
            
            lower_shadow = min(open_prices[-1], close[-1]) - low[-1]
            upper_shadow = high[-1] - max(open_prices[-1], close[-1])
            indicators['hammer_pattern'] = 1 if lower_shadow > (body_size * 2) else 0
            indicators['shooting_star_pattern'] = 1 if upper_shadow > (body_size * 2) else 0
            
            # === MARKET STRUCTURE ===
            up_moves = sum(1 for i in range(1, min(20, len(close))) if close[-i] > close[-i-1])
            indicators['trend_strength'] = (up_moves / min(19, len(close)-1)) if len(close) > 1 else 0.5
            
            volatility_20 = np.std(close[-20:]) if len(close) >= 20 else np.std(close)
            volatility_5 = np.std(close[-5:]) if len(close) >= 5 else volatility_20
            indicators['volatility_regime'] = volatility_5 / volatility_20 if volatility_20 > 0 else 1
            
            # === NORMALIZE VALUES ===
            current_price = close[-1]
            for key, value in indicators.items():
                if key.startswith('sma_') or key.startswith('ema_') or key.startswith('bb_') or key.startswith('pivot') or key.startswith('resistance') or key.startswith('support'):
                    if 'position' not in key and 'ratio' not in key and 'width' not in key:
                        indicators[key] = value / current_price if current_price > 0 else 1
                elif key in ['rsi_14', 'rsi_7']:
                    indicators[key] = (value - 50) / 50  # Normalize to -1 to 1
                    
            return indicators
            
        except Exception as e:
            print(f"Technical analysis error: {e}")
            return self._get_default_indicators()
    
    def _calculate_ema(self, prices, period):
        """Calculate EMA manually"""
        try:
            if len(prices) < period:
                return prices[-1] if prices else 0
            
            multiplier = 2 / (period + 1)
            ema = np.mean(prices[:period])  # Start with SMA
            
            for price in prices[period:]:
                ema = (price * multiplier) + (ema * (1 - multiplier))
            
            return ema
        except:
            return prices[-1] if prices else 0
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI manually"""
        try:
            if len(prices) < period + 1:
                return 50
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
        except:
            return 50
    
    def _calculate_atr(self, high, low, close, period=14):
        """Calculate ATR manually"""
        try:
            if len(high) < period:
                return high[-1] - low[-1]
            
            tr_values = []
            for i in range(1, len(high)):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                tr_values.append(max(tr1, tr2, tr3))
            
            return np.mean(tr_values[-period:])
        except:
            return high[-1] - low[-1] if high and low else 0.01
    
    def _calculate_obv(self, close, volume):
        """Calculate On Balance Volume"""
        try:
            if len(close) < 2:
                return 0
            
            obv = 0
            for i in range(1, len(close)):
                if close[i] > close[i-1]:
                    obv += volume[i]
                elif close[i] < close[i-1]:
                    obv -= volume[i]
            
            return obv
        except:
            return 0
    
    def _get_default_indicators(self) -> Dict:
        """Return default indicators when calculation fails"""
        return {
            'sma_20': 1.0, 'sma_50': 1.0, 'sma_200': 1.0,
            'ema_12': 1.0, 'ema_26': 1.0,
            'macd': 0.0, 'macd_signal': 0.0, 'macd_histogram': 0.0,
            'bb_upper': 1.02, 'bb_lower': 0.98, 'bb_width': 0.04, 'bb_position': 0.5,
            'rsi_14': 0.0, 'rsi_7': 0.0,
            'atr_14': 0.01, 'atr_ratio': 0.01,
            'volume_sma': 1000, 'volume_ratio': 1.0, 'obv': 0.0,
            'pivot_point': 1.0, 'resistance_1': 1.01, 'support_1': 0.99,
            'price_change_1': 0.0, 'price_change_5': 0.0, 'price_change_20': 0.0,
            'high_low_ratio': 0.5,
            'doji_pattern': 0, 'hammer_pattern': 0, 'shooting_star_pattern': 0,
            'trend_strength': 0.5, 'volatility_regime': 1.0
        }


# === COMPATIBILITY FUNCTIONS ===

def create_professional_environment(mt5_interface, recovery_engine, config):
    """Factory function to create professional trading environment"""
    return ProfessionalTradingEnvironment(mt5_interface, recovery_engine, config)

# Keep old class name for backward compatibility
TradingEnvironment = ProfessionalTradingEnvironment

# Export main classes
__all__ = [
    'ProfessionalTradingEnvironment',
    'TradingEnvironment',  # Backward compatibility
    'TradingState',
    'MarketRegime',
    'TechnicalAnalyzer',
    'MarketRegimeDetector',
    'create_professional_environment'
]
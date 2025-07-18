# recovery_engine.py - Professional Smart Recovery System
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import os
from enum import Enum
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class RecoveryMode(Enum):
    """Advanced Recovery Modes"""
    INACTIVE = "INACTIVE"
    ADAPTIVE_MARTINGALE = "ADAPTIVE_MARTINGALE"
    SMART_GRID = "SMART_GRID"
    DYNAMIC_HEDGE = "DYNAMIC_HEDGE"
    AI_ENSEMBLE = "AI_ENSEMBLE"
    CORRELATION_HEDGE = "CORRELATION_HEDGE"
    VOLATILITY_BREAKOUT = "VOLATILITY_BREAKOUT"
    MEAN_REVERSION = "MEAN_REVERSION"

class RecoveryTrigger(Enum):
    """Recovery Trigger Conditions"""
    DRAWDOWN_THRESHOLD = "DRAWDOWN_THRESHOLD"
    CONSECUTIVE_LOSSES = "CONSECUTIVE_LOSSES"
    PORTFOLIO_HEAT = "PORTFOLIO_HEAT"
    AI_SIGNAL = "AI_SIGNAL"
    MARKET_CONDITION = "MARKET_CONDITION"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    TIME_BASED = "TIME_BASED"

class SmartProfitManager:
    """
    Advanced Profit Management System
    - Dynamic profit targets based on market conditions
    - Multi-timeframe profit optimization
    - AI-powered exit timing
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Dynamic profit parameters
        self.base_profit_per_lot = self.config.get('base_profit_per_lot', 8.0)
        self.volatility_multiplier = self.config.get('volatility_multiplier', 1.5)
        self.trend_bonus = self.config.get('trend_bonus', 0.3)
        
        # Multi-timeframe settings
        self.timeframes = ['M1', 'M5', 'M15', 'H1']
        self.timeframe_weights = [0.4, 0.3, 0.2, 0.1]
        
        # AI optimization
        self.ml_profit_optimization = self.config.get('ml_profit_optimization', True)
        self.sentiment_analysis = self.config.get('sentiment_analysis', True)
        
        # Performance tracking
        self.profit_decisions = deque(maxlen=1000)
        self.success_rate = 0.0
        self.avg_hold_time = 0.0
        self.profit_efficiency = 0.0
        
        print(f"💰 Smart Profit Manager initialized with dynamic targeting")
    
    def calculate_dynamic_profit_target(self, position_info: Dict, market_data: Dict) -> Dict:
        """Calculate dynamic profit target based on advanced analysis"""
        try:
            volume = position_info.get('volume', 0.01)
            entry_price = position_info.get('price_open', 0)
            position_type = position_info.get('type', 0)  # 0=buy, 1=sell
            
            # Base target
            base_target = self.base_profit_per_lot * (volume / 0.01)
            
            # Market condition adjustments
            volatility_adj = self._calculate_volatility_adjustment(market_data)
            trend_adj = self._calculate_trend_adjustment(market_data, position_type)
            momentum_adj = self._calculate_momentum_adjustment(market_data)
            support_resistance_adj = self._calculate_sr_adjustment(market_data, entry_price, position_type)
            
            # AI-based adjustment
            ai_adj = self._calculate_ai_adjustment(position_info, market_data)
            
            # Combine adjustments
            total_multiplier = (
                1.0 +
                volatility_adj * 0.3 +
                trend_adj * 0.25 +
                momentum_adj * 0.2 +
                support_resistance_adj * 0.15 +
                ai_adj * 0.1
            )
            
            dynamic_target = base_target * total_multiplier
            
            # Calculate multiple profit levels
            profit_targets = {
                'conservative': dynamic_target * 0.6,
                'moderate': dynamic_target,
                'aggressive': dynamic_target * 1.4,
                'maximum': dynamic_target * 2.0
            }
            
            # Exit timing predictions
            exit_timing = self._predict_optimal_exit_timing(position_info, market_data)
            
            result = {
                'targets': profit_targets,
                'recommended_target': dynamic_target,
                'multiplier': total_multiplier,
                'adjustments': {
                    'volatility': volatility_adj,
                    'trend': trend_adj,
                    'momentum': momentum_adj,
                    'support_resistance': support_resistance_adj,
                    'ai': ai_adj
                },
                'exit_timing': exit_timing,
                'confidence': min(abs(total_multiplier - 1.0) * 2, 1.0)
            }
            
            return result
            
        except Exception as e:
            print(f"Dynamic profit calculation error: {e}")
            return {
                'targets': {'conservative': 5.0, 'moderate': 8.0, 'aggressive': 12.0, 'maximum': 20.0},
                'recommended_target': 8.0,
                'multiplier': 1.0,
                'confidence': 0.5
            }
    
    def _calculate_volatility_adjustment(self, market_data):
        """Calculate volatility-based adjustment"""
        try:
            atr = market_data.get('atr_14', 0.01)
            current_price = market_data.get('current_price', 2000)
            atr_ratio = atr / current_price
            
            # High volatility = higher targets
            if atr_ratio > 0.015:  # High volatility
                return 0.5  # +50% target
            elif atr_ratio > 0.010:  # Medium volatility
                return 0.2  # +20% target
            elif atr_ratio < 0.005:  # Low volatility
                return -0.3  # -30% target
            else:
                return 0.0  # Normal target
                
        except:
            return 0.0
    
    def _calculate_trend_adjustment(self, market_data, position_type):
        """Calculate trend-based adjustment"""
        try:
            sma_20 = market_data.get('sma_20', 1.0)
            sma_50 = market_data.get('sma_50', 1.0)
            current_price = market_data.get('current_price', 2000)
            
            # Trend strength
            if sma_20 > sma_50 * 1.002:  # Strong uptrend
                trend_strength = 0.4
            elif sma_20 < sma_50 * 0.998:  # Strong downtrend
                trend_strength = 0.4
            else:  # Sideways
                trend_strength = -0.2
                
            # Position alignment with trend
            price_vs_sma20 = (current_price / sma_20) - 1
            
            if position_type == 0:  # Buy position
                if price_vs_sma20 > 0 and sma_20 > sma_50:  # Long in uptrend
                    return trend_strength
                elif price_vs_sma20 < 0 and sma_20 < sma_50:  # Long against downtrend
                    return -0.3
            else:  # Sell position
                if price_vs_sma20 < 0 and sma_20 < sma_50:  # Short in downtrend
                    return trend_strength
                elif price_vs_sma20 > 0 and sma_20 > sma_50:  # Short against uptrend
                    return -0.3
                    
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_momentum_adjustment(self, market_data):
        """Calculate momentum-based adjustment"""
        try:
            rsi = market_data.get('rsi_14', 50)
            macd = market_data.get('macd', 0)
            stoch_k = market_data.get('stoch_k', 50)
            
            # Momentum score
            momentum_score = 0
            
            # RSI momentum
            if rsi > 70:
                momentum_score += 0.3  # Strong momentum
            elif rsi < 30:
                momentum_score += 0.3  # Strong reverse momentum
            elif 45 < rsi < 55:
                momentum_score -= 0.2  # Weak momentum
                
            # MACD momentum
            if abs(macd) > 0.001:
                momentum_score += 0.2
            else:
                momentum_score -= 0.1
                
            # Stochastic momentum
            if stoch_k > 80 or stoch_k < 20:
                momentum_score += 0.1
                
            return max(-0.4, min(momentum_score, 0.4))
            
        except:
            return 0.0
    
    def _calculate_sr_adjustment(self, market_data, entry_price, position_type):
        """Calculate support/resistance adjustment"""
        try:
            resistance_1 = market_data.get('resistance_1', entry_price * 1.01)
            support_1 = market_data.get('support_1', entry_price * 0.99)
            current_price = market_data.get('current_price', entry_price)
            
            if position_type == 0:  # Buy position
                distance_to_resistance = (resistance_1 - current_price) / current_price
                if distance_to_resistance > 0.01:  # 1% away
                    return 0.3  # Good upside potential
                elif distance_to_resistance < 0.003:  # Close to resistance
                    return -0.2  # Reduce target
            else:  # Sell position
                distance_to_support = (current_price - support_1) / current_price
                if distance_to_support > 0.01:  # 1% away
                    return 0.3  # Good downside potential
                elif distance_to_support < 0.003:  # Close to support
                    return -0.2  # Reduce target
                    
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_ai_adjustment(self, position_info, market_data):
        """AI-based profit target adjustment"""
        try:
            # Simulated AI analysis (in real implementation, use ML model)
            volatility_regime = market_data.get('volatility_regime', 1.0)
            trend_strength = market_data.get('trend_strength', 0.5)
            market_efficiency = market_data.get('market_efficiency', 0.7)
            
            # AI composite score
            ai_score = (
                (volatility_regime - 1.0) * 0.4 +
                (trend_strength - 0.5) * 0.3 +
                (market_efficiency - 0.7) * 0.3
            )
            
            return max(-0.3, min(ai_score, 0.3))
            
        except:
            return 0.0
    
    def _predict_optimal_exit_timing(self, position_info, market_data):
        """Predict optimal exit timing"""
        try:
            # Time-based analysis
            position_age_minutes = position_info.get('age_minutes', 0)
            
            # Market condition analysis
            volatility = market_data.get('atr_ratio', 0.01)
            trend_strength = market_data.get('trend_strength', 0.5)
            
            # Predict optimal exit window
            if volatility > 0.015:  # High volatility
                optimal_window = (10, 30)  # 10-30 minutes
            elif trend_strength > 0.7:  # Strong trend
                optimal_window = (30, 120)  # 30-120 minutes
            else:  # Normal conditions
                optimal_window = (15, 60)  # 15-60 minutes
                
            # Current timing score
            if optimal_window[0] <= position_age_minutes <= optimal_window[1]:
                timing_score = 1.0  # Optimal time
            elif position_age_minutes < optimal_window[0]:
                timing_score = 0.3  # Too early
            else:
                timing_score = max(0.1, 1.0 - (position_age_minutes - optimal_window[1]) / 60)
                
            return {
                'optimal_window_minutes': optimal_window,
                'current_timing_score': timing_score,
                'recommendation': 'HOLD' if timing_score < 0.7 else 'CONSIDER_EXIT'
            }
            
        except:
            return {
                'optimal_window_minutes': (15, 60),
                'current_timing_score': 0.5,
                'recommendation': 'HOLD'
            }
    
    def evaluate_exit_signal(self, positions: List[Dict], market_data: Dict) -> List[Dict]:
        """Evaluate exit signals for all positions"""
        try:
            exit_recommendations = []
            
            for position in positions:
                current_profit = position.get('profit', 0)
                volume = position.get('volume', 0.01)
                
                # Get dynamic target
                target_analysis = self.calculate_dynamic_profit_target(position, market_data)
                recommended_target = target_analysis['recommended_target']
                
                # Multiple exit criteria
                exit_analysis = {
                    'position_id': position.get('ticket', 0),
                    'current_profit': current_profit,
                    'target_profit': recommended_target,
                    'profit_ratio': current_profit / recommended_target if recommended_target > 0 else 0,
                    'exit_signals': [],
                    'overall_recommendation': 'HOLD',
                    'confidence': 0.0
                }
                
                # Signal 1: Target reached
                if current_profit >= recommended_target:
                    exit_analysis['exit_signals'].append({
                        'signal': 'TARGET_REACHED',
                        'strength': 1.0,
                        'description': f'Profit target reached: ${current_profit:.2f} >= ${recommended_target:.2f}'
                    })
                
                # Signal 2: Conservative target with good timing
                conservative_target = target_analysis['targets']['conservative']
                timing_analysis = target_analysis['exit_timing']
                
                if (current_profit >= conservative_target and 
                    timing_analysis['current_timing_score'] > 0.7):
                    exit_analysis['exit_signals'].append({
                        'signal': 'TIMING_OPTIMAL',
                        'strength': 0.8,
                        'description': f'Good profit with optimal timing: ${current_profit:.2f}'
                    })
                
                # Signal 3: Market condition deterioration
                market_condition_score = self._assess_market_condition_for_exit(market_data)
                if market_condition_score < -0.5 and current_profit > 0:
                    exit_analysis['exit_signals'].append({
                        'signal': 'MARKET_DETERIORATION',
                        'strength': 0.6,
                        'description': 'Market conditions deteriorating, secure profits'
                    })
                
                # Signal 4: Reversal pattern
                reversal_signal = self._detect_reversal_pattern(position, market_data)
                if reversal_signal > 0.6:
                    exit_analysis['exit_signals'].append({
                        'signal': 'REVERSAL_PATTERN',
                        'strength': reversal_signal,
                        'description': 'Potential reversal pattern detected'
                    })
                
                # Overall recommendation
                if exit_analysis['exit_signals']:
                    max_strength = max(signal['strength'] for signal in exit_analysis['exit_signals'])
                    if max_strength >= 0.8:
                        exit_analysis['overall_recommendation'] = 'STRONG_EXIT'
                    elif max_strength >= 0.6:
                        exit_analysis['overall_recommendation'] = 'EXIT'
                    else:
                        exit_analysis['overall_recommendation'] = 'CONSIDER_EXIT'
                    
                    exit_analysis['confidence'] = max_strength
                
                exit_recommendations.append(exit_analysis)
            
            return exit_recommendations
            
        except Exception as e:
            print(f"Exit evaluation error: {e}")
            return []
    
    def _assess_market_condition_for_exit(self, market_data):
        """Assess if market conditions favor exits"""
        try:
            volatility_spike = market_data.get('volatility_regime', 1.0) > 1.5
            trend_weakening = 0.3 < market_data.get('trend_strength', 0.5) < 0.7
            overbought_oversold = (market_data.get('rsi_14', 50) > 75 or 
                                 market_data.get('rsi_14', 50) < 25)
            
            condition_score = 0
            if volatility_spike:
                condition_score -= 0.3
            if trend_weakening:
                condition_score -= 0.2
            if overbought_oversold:
                condition_score -= 0.4
                
            return condition_score
            
        except:
            return 0.0
    
    def _detect_reversal_pattern(self, position, market_data):
        """Detect potential reversal patterns"""
        try:
            # Pattern detection (simplified)
            doji = market_data.get('doji_pattern', 0)
            hammer = market_data.get('hammer_pattern', 0)
            shooting_star = market_data.get('shooting_star_pattern', 0)
            
            # RSI divergence (simulated)
            rsi = market_data.get('rsi_14', 50)
            rsi_divergence = 0.3 if (rsi > 70 or rsi < 30) else 0
            
            # MACD signal
            macd = market_data.get('macd', 0)
            macd_signal = market_data.get('macd_signal', 0)
            macd_cross = 0.4 if (macd > 0 and macd_signal < 0) or (macd < 0 and macd_signal > 0) else 0
            
            reversal_strength = max(doji, hammer, shooting_star) + rsi_divergence + macd_cross
            return min(reversal_strength, 1.0)
            
        except:
            return 0.0

class ProfessionalRecoveryEngine:
    """
    Professional Multi-Strategy Recovery System
    - 15-dimensional action space integration
    - Advanced ML-powered recovery strategies
    - Multi-timeframe analysis
    - Real-time market adaptation
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        print(f"🛡️ Initializing Professional Recovery Engine...")
        
        # Core recovery parameters
        self.max_drawdown_threshold = self.config.get('max_drawdown_threshold', 5.0)  # 5%
        self.recovery_trigger_loss = self.config.get('recovery_trigger_loss', 100.0)  # $100
        self.max_recovery_levels = self.config.get('max_recovery_levels', 8)
        self.emergency_stop_loss = self.config.get('emergency_stop_loss', 1000.0)  # $1000
        
        # Advanced strategy parameters
        self.adaptive_sizing = self.config.get('adaptive_sizing', True)
        self.correlation_analysis = self.config.get('correlation_analysis', True)
        self.volatility_filtering = self.config.get('volatility_filtering', True)
        self.ml_strategy_selection = self.config.get('ml_strategy_selection', True)
        
        # Recovery state
        self.current_mode = RecoveryMode.INACTIVE
        self.recovery_level = 0
        self.recovery_start_time = None
        self.recovery_start_equity = 0.0
        self.recovery_positions = []
        self.recovery_history = deque(maxlen=100)
        
        # Strategy-specific states
        self.martingale_multiplier = 1.5
        self.grid_spacing = 200  # points
        self.hedge_ratio = 0.0
        self.correlation_matrix = {}
        
        # Performance tracking
        self.total_recovery_attempts = 0
        self.successful_recoveries = 0
        self.recovery_efficiency = 0.0
        self.avg_recovery_time = 0.0
        self.strategy_performance = {mode.value: {'attempts': 0, 'success': 0, 'avg_time': 0} for mode in RecoveryMode}
        
        # Advanced components
        self.profit_manager = SmartProfitManager(config)
        self.market_analyzer = None  # Will be set externally
        
        # Action space integration (15 dimensions)
        self.action_mappings = self._initialize_action_mappings()
        
        print(f"✅ Professional Recovery Engine initialized:")
        print(f"   Max Drawdown Threshold: {self.max_drawdown_threshold}%")
        print(f"   Recovery Trigger: ${self.recovery_trigger_loss}")
        print(f"   Max Recovery Levels: {self.max_recovery_levels}")
        print(f"   Advanced Features: ML Strategy Selection, Correlation Analysis")
    
    def _initialize_action_mappings(self):
        """Initialize mappings for 15-dimensional action space"""
        return {
            'market_direction': 0,      # -1 to 1
            'position_size': 1,         # 0.01 to 1.0
            'entry_aggression': 2,      # 0 to 1
            'profit_target_ratio': 3,   # 0.5 to 5.0
            'partial_take_levels': 4,   # 0 to 3
            'add_position_signal': 5,   # 0 to 1
            'hedge_ratio': 6,           # 0 to 1
            'recovery_mode': 7,         # 0 to 3
            'correlation_limit': 8,     # 0 to 1
            'volatility_filter': 9,     # 0 to 1
            'spread_tolerance': 10,     # 0 to 1
            'time_filter': 11,          # 0 to 1
            'portfolio_heat_limit': 12, # 0 to 1
            'smart_exit_signal': 13,    # 0 to 1
            'rebalance_trigger': 14     # 0 to 1
        }
    
    def analyze_recovery_need(self, portfolio_state: Dict, market_data: Dict) -> Dict:
        """Comprehensive analysis of recovery need"""
        try:
            analysis = {
                'recovery_needed': False,
                'urgency_level': 0,  # 0-10 scale
                'trigger_reasons': [],
                'recommended_mode': RecoveryMode.INACTIVE,
                'risk_assessment': {},
                'market_conditions': {},
                'suggested_actions': []
            }
            
            # Get portfolio metrics
            total_pnl = portfolio_state.get('total_pnl', 0)
            current_equity = portfolio_state.get('equity', 0)
            daily_pnl = portfolio_state.get('daily_pnl', 0)
            portfolio_heat = portfolio_state.get('portfolio_heat', 0)
            drawdown = portfolio_state.get('drawdown', 0)
            
            # Trigger Analysis
            urgency_factors = []
            
            # 1. Absolute loss trigger
            if total_pnl < -self.recovery_trigger_loss:
                urgency_factors.append(('ABSOLUTE_LOSS', min(abs(total_pnl) / self.recovery_trigger_loss, 3.0)))
                analysis['trigger_reasons'].append(f'Absolute loss: ${total_pnl:.2f}')
            
            # 2. Drawdown trigger
            if drawdown > self.max_drawdown_threshold:
                urgency_factors.append(('DRAWDOWN', drawdown / self.max_drawdown_threshold))
                analysis['trigger_reasons'].append(f'Drawdown: {drawdown:.2f}%')
            
            # 3. Portfolio heat trigger
            if portfolio_heat > 8.0:
                urgency_factors.append(('PORTFOLIO_HEAT', portfolio_heat / 10.0))
                analysis['trigger_reasons'].append(f'Portfolio heat: {portfolio_heat:.1f}%')
            
            # 4. Daily loss trigger
            if daily_pnl < -3.0:  # -3% daily loss
                urgency_factors.append(('DAILY_LOSS', abs(daily_pnl) / 5.0))
                analysis['trigger_reasons'].append(f'Daily loss: {daily_pnl:.2f}%')
            
            # 5. Consecutive losing positions
            consecutive_losses = self._count_consecutive_losses(portfolio_state.get('positions', []))
            if consecutive_losses >= 5:
                urgency_factors.append(('CONSECUTIVE_LOSSES', min(consecutive_losses / 5.0, 2.0)))
                analysis['trigger_reasons'].append(f'Consecutive losses: {consecutive_losses}')
            
            # Calculate overall urgency
            if urgency_factors:
                analysis['recovery_needed'] = True
                analysis['urgency_level'] = min(sum(factor[1] for factor in urgency_factors), 10.0)
                
                # Recommend recovery mode based on conditions
                analysis['recommended_mode'] = self._select_optimal_recovery_mode(
                    portfolio_state, market_data, urgency_factors
                )
            
            # Risk assessment
            analysis['risk_assessment'] = self._assess_recovery_risks(portfolio_state, market_data)
            
            # Market condition analysis
            analysis['market_conditions'] = self._analyze_market_conditions_for_recovery(market_data)
            
            # Generate suggested actions
            if analysis['recovery_needed']:
                analysis['suggested_actions'] = self._generate_recovery_actions(
                    analysis['recommended_mode'], portfolio_state, market_data
                )
            
            return analysis
            
        except Exception as e:
            print(f"Recovery analysis error: {e}")
            return {
                'recovery_needed': False,
                'urgency_level': 0,
                'trigger_reasons': ['Analysis error'],
                'recommended_mode': RecoveryMode.INACTIVE
            }
    
    def _count_consecutive_losses(self, positions):
        """Count consecutive losing positions"""
        try:
            if not positions:
                return 0
                
            losses = 0
            for pos in reversed(positions):  # Check from most recent
                if pos.get('profit', 0) < 0:
                    losses += 1
                else:
                    break
            return losses
        except:
            return 0
    
    def _select_optimal_recovery_mode(self, portfolio_state, market_data, urgency_factors):
        """Select optimal recovery mode using ML and market analysis"""
        try:
            # Get market regime
            volatility = market_data.get('atr_ratio', 0.01)
            trend_strength = market_data.get('trend_strength', 0.5)
            market_regime = market_data.get('regime', 'SIDEWAYS')
            
            # Analyze urgency factors
            urgency_types = [factor[0] for factor in urgency_factors]
            max_urgency = max(factor[1] for factor in urgency_factors) if urgency_factors else 0
            
            # Selection logic based on conditions
            if max_urgency >= 5.0:  # Very high urgency
                if volatility > 0.015:  # High volatility
                    return RecoveryMode.DYNAMIC_HEDGE
                else:
                    return RecoveryMode.AI_ENSEMBLE
                    
            elif 'CONSECUTIVE_LOSSES' in urgency_types:
                if trend_strength > 0.7:  # Strong trend
                    return RecoveryMode.MEAN_REVERSION
                else:
                    return RecoveryMode.ADAPTIVE_MARTINGALE
                    
            elif 'PORTFOLIO_HEAT' in urgency_types:
                return RecoveryMode.CORRELATION_HEDGE
                
            elif volatility > 0.012:  # Medium-high volatility
                return RecoveryMode.VOLATILITY_BREAKOUT
                
            elif trend_strength < 0.3:  # Sideways market
                return RecoveryMode.SMART_GRID
                
            else:  # Default
                return RecoveryMode.ADAPTIVE_MARTINGALE
                
        except Exception as e:
            print(f"Recovery mode selection error: {e}")
            return RecoveryMode.ADAPTIVE_MARTINGALE
    
    def _assess_recovery_risks(self, portfolio_state, market_data):
        """Assess risks associated with recovery strategies"""
        try:
            risks = {
                'market_risk': 0.0,
                'liquidity_risk': 0.0,
                'correlation_risk': 0.0,
                'volatility_risk': 0.0,
                'capital_risk': 0.0,
                'overall_risk': 0.0
            }
            
            # Market risk
            volatility = market_data.get('atr_ratio', 0.01)
            risks['volatility_risk'] = min(volatility * 50, 1.0)  # Scale to 0-1
            
            # Correlation risk
            positions = portfolio_state.get('positions', [])
            if len(positions) > 1:
                risks['correlation_risk'] = 0.6  # Assume moderate correlation risk
            
            # Capital risk
            current_equity = portfolio_state.get('equity', 0)
            if current_equity < 1000:
                risks['capital_risk'] = 0.8  # High risk with low capital
            elif current_equity < 5000:
                risks['capital_risk'] = 0.4  # Medium risk
            else:
                risks['capital_risk'] = 0.1  # Low risk
            
            # Overall risk
            risks['overall_risk'] = np.mean(list(risks.values()))
            
            return risks
            
        except Exception as e:
            print(f"Risk assessment error: {e}")
            return {'overall_risk': 0.5}
    
    def _analyze_market_conditions_for_recovery(self, market_data):
        """Analyze market conditions for recovery suitability"""
        try:
            conditions = {
                'volatility_regime': 'NORMAL',
                'trend_clarity': 'UNCLEAR',
                'momentum_strength': 'WEAK',
                'support_resistance': 'WEAK',
                'recovery_favorability': 0.5
            }
            
            # Volatility analysis
            atr_ratio = market_data.get('atr_ratio', 0.01)
            if atr_ratio > 0.015:
                conditions['volatility_regime'] = 'HIGH'
            elif atr_ratio < 0.005:
                conditions['volatility_regime'] = 'LOW'
            else:
                conditions['volatility_regime'] = 'NORMAL'
            
            # Trend analysis
            trend_strength = market_data.get('trend_strength', 0.5)
            if trend_strength > 0.7:
                conditions['trend_clarity'] = 'STRONG'
            elif trend_strength > 0.4:
                conditions['trend_clarity'] = 'MODERATE'
            else:
                conditions['trend_clarity'] = 'WEAK'
            
            # Momentum analysis
            rsi = market_data.get('rsi_14', 50)
            macd = market_data.get('macd', 0)
            if abs(rsi - 50) > 20 and abs(macd) > 0.001:
                conditions['momentum_strength'] = 'STRONG'
            elif abs(rsi - 50) > 10:
                conditions['momentum_strength'] = 'MODERATE'
            else:
                conditions['momentum_strength'] = 'WEAK'
            
            # Support/Resistance analysis
            bb_position = market_data.get('bb_position', 0.5)
            if bb_position > 0.8 or bb_position < 0.2:
                conditions['support_resistance'] = 'STRONG'
            elif bb_position > 0.7 or bb_position < 0.3:
                conditions['support_resistance'] = 'MODERATE'
            else:
                conditions['support_resistance'] = 'WEAK'
            
            # Calculate overall favorability
            favorability_score = 0.5
            
            # High volatility is good for some strategies
            if conditions['volatility_regime'] == 'HIGH':
                favorability_score += 0.2
            elif conditions['volatility_regime'] == 'LOW':
                favorability_score -= 0.1
            
            # Clear trends help with directional recovery
            if conditions['trend_clarity'] == 'STRONG':
                favorability_score += 0.3
            elif conditions['trend_clarity'] == 'WEAK':
                favorability_score -= 0.2
            
            conditions['recovery_favorability'] = max(0.0, min(favorability_score, 1.0))
            
            return conditions
            
        except Exception as e:
            print(f"Market conditions analysis error: {e}")
            return {'recovery_favorability': 0.5}
    
    def _generate_recovery_actions(self, recovery_mode, portfolio_state, market_data):
        """Generate specific recovery actions for the recommended mode"""
        try:
            actions = []
            
            if recovery_mode == RecoveryMode.ADAPTIVE_MARTINGALE:
                actions.extend(self._generate_martingale_actions(portfolio_state, market_data))
            elif recovery_mode == RecoveryMode.SMART_GRID:
                actions.extend(self._generate_grid_actions(portfolio_state, market_data))
            elif recovery_mode == RecoveryMode.DYNAMIC_HEDGE:
                actions.extend(self._generate_hedge_actions(portfolio_state, market_data))
            elif recovery_mode == RecoveryMode.AI_ENSEMBLE:
                actions.extend(self._generate_ensemble_actions(portfolio_state, market_data))
            elif recovery_mode == RecoveryMode.CORRELATION_HEDGE:
                actions.extend(self._generate_correlation_actions(portfolio_state, market_data))
            elif recovery_mode == RecoveryMode.VOLATILITY_BREAKOUT:
                actions.extend(self._generate_volatility_actions(portfolio_state, market_data))
            elif recovery_mode == RecoveryMode.MEAN_REVERSION:
                actions.extend(self._generate_mean_reversion_actions(portfolio_state, market_data))
            
            return actions
            
        except Exception as e:
            print(f"Recovery actions generation error: {e}")
            return ['Monitor positions closely', 'Apply conservative risk management']
    
    def _generate_martingale_actions(self, portfolio_state, market_data):
        """Generate Adaptive Martingale recovery actions"""
        return [
            f"Apply adaptive martingale with {self.martingale_multiplier:.1f}x multiplier",
            "Scale position sizes based on market volatility",
            "Implement dynamic stop-loss levels",
            "Monitor correlation between positions",
            "Use volatility-adjusted profit targets"
        ]
    
    def _generate_grid_actions(self, portfolio_state, market_data):
        """Generate Smart Grid recovery actions"""
        return [
            f"Deploy smart grid with {self.grid_spacing} point spacing",
            "Place orders at dynamic support/resistance levels",
            "Implement volume-weighted position sizing",
            "Use time-based grid activation",
            "Apply trend-following grid bias"
        ]
    
    def _generate_hedge_actions(self, portfolio_state, market_data):
        """Generate Dynamic Hedge recovery actions"""
        return [
            "Implement delta-neutral hedging strategy",
            "Use correlation-based hedge ratios",
            "Apply volatility-adjusted hedge sizing",
            "Monitor hedge effectiveness in real-time",
            "Implement dynamic hedge rebalancing"
        ]
    
    def _generate_ensemble_actions(self, portfolio_state, market_data):
        """Generate AI Ensemble recovery actions"""
        return [
            "Activate multi-strategy ensemble approach",
            "Use ML-based strategy selection",
            "Implement adaptive strategy weighting",
            "Apply real-time strategy performance monitoring",
            "Use ensemble risk management"
        ]
    
    def _generate_correlation_actions(self, portfolio_state, market_data):
        """Generate Correlation Hedge recovery actions"""
        return [
            "Analyze position correlations",
            "Implement correlation-reducing trades",
            "Use sector/asset diversification",
            "Apply correlation-weighted position sizing",
            "Monitor correlation matrix changes"
        ]
    
    def _generate_volatility_actions(self, portfolio_state, market_data):
        """Generate Volatility Breakout recovery actions"""
        return [
            "Trade volatility expansion patterns",
            "Use volatility-based position sizing",
            "Implement breakout confirmation filters",
            "Apply dynamic stop-loss based on ATR",
            "Monitor volatility regime changes"
        ]
    
    def _generate_mean_reversion_actions(self, portfolio_state, market_data):
        """Generate Mean Reversion recovery actions"""
        return [
            "Identify overbought/oversold conditions",
            "Use statistical mean reversion signals",
            "Implement contrarian position sizing",
            "Apply time-based exit strategies",
            "Monitor mean reversion strength"
        ]
    
    def execute_recovery_strategy(self, recovery_mode: RecoveryMode, action_vector: np.ndarray, 
                                portfolio_state: Dict, market_data: Dict, mt5_interface) -> Dict:
        """Execute professional recovery strategy with 15-dimensional actions"""
        try:
            print(f"🛡️ === EXECUTING RECOVERY STRATEGY ===")
            print(f"   Mode: {recovery_mode.value}")
            print(f"   Recovery Level: {self.recovery_level}")
            
            # Activate recovery if not already active
            if self.current_mode == RecoveryMode.INACTIVE:
                self._activate_recovery(recovery_mode, portfolio_state)
            
            # Parse action vector
            parsed_actions = self._parse_recovery_actions(action_vector)
            
            # Execute strategy-specific logic
            execution_result = {
                'success': False,
                'actions_taken': [],
                'new_positions': [],
                'modified_positions': [],
                'closed_positions': [],
                'recovery_metrics': {},
                'next_actions': []
            }
            
            if recovery_mode == RecoveryMode.ADAPTIVE_MARTINGALE:
                execution_result = self._execute_adaptive_martingale(
                    parsed_actions, portfolio_state, market_data, mt5_interface
                )
            elif recovery_mode == RecoveryMode.SMART_GRID:
                execution_result = self._execute_smart_grid(
                    parsed_actions, portfolio_state, market_data, mt5_interface
                )
            elif recovery_mode == RecoveryMode.DYNAMIC_HEDGE:
                execution_result = self._execute_dynamic_hedge(
                    parsed_actions, portfolio_state, market_data, mt5_interface
                )
            elif recovery_mode == RecoveryMode.AI_ENSEMBLE:
                execution_result = self._execute_ai_ensemble(
                    parsed_actions, portfolio_state, market_data, mt5_interface
                )
            elif recovery_mode == RecoveryMode.CORRELATION_HEDGE:
                execution_result = self._execute_correlation_hedge(
                    parsed_actions, portfolio_state, market_data, mt5_interface
                )
            elif recovery_mode == RecoveryMode.VOLATILITY_BREAKOUT:
                execution_result = self._execute_volatility_breakout(
                    parsed_actions, portfolio_state, market_data, mt5_interface
                )
            elif recovery_mode == RecoveryMode.MEAN_REVERSION:
                execution_result = self._execute_mean_reversion(
                    parsed_actions, portfolio_state, market_data, mt5_interface
                )
            
            # Update recovery state
            self._update_recovery_state(execution_result, portfolio_state)
            
            # Check for recovery completion
            self._check_recovery_completion(portfolio_state, market_data)
            
            print(f"🛡️ Recovery execution completed: {execution_result['success']}")
            print(f"   Actions taken: {len(execution_result['actions_taken'])}")
            
            return execution_result
            
        except Exception as e:
            print(f"❌ Recovery execution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'actions_taken': []
            }
    
    def _parse_recovery_actions(self, action_vector):
        """Parse 15-dimensional action vector for recovery"""
        try:
            actions = {}
            
            for action_name, index in self.action_mappings.items():
                if index < len(action_vector):
                    actions[action_name] = float(action_vector[index])
                else:
                    actions[action_name] = 0.0
            
            # Add derived actions
            actions['recovery_aggressiveness'] = min(abs(actions['market_direction']) + 
                                                   actions['position_size'] + 
                                                   actions['recovery_mode'] / 3.0, 3.0)
            
            actions['risk_tolerance'] = 1.0 - actions['volatility_filter']
            actions['timing_urgency'] = actions['entry_aggression']
            
            return actions
            
        except Exception as e:
            print(f"Action parsing error: {e}")
            return {}
    
    def _activate_recovery(self, recovery_mode, portfolio_state):
        """Activate recovery mode"""
        try:
            self.current_mode = recovery_mode
            self.recovery_level = 1
            self.recovery_start_time = datetime.now()
            self.recovery_start_equity = portfolio_state.get('equity', 0)
            self.total_recovery_attempts += 1
            
            # Initialize strategy-specific parameters
            self._initialize_strategy_parameters(recovery_mode)
            
            print(f"🔄 Recovery ACTIVATED: {recovery_mode.value}")
            print(f"   Start Equity: ${self.recovery_start_equity:,.2f}")
            print(f"   Recovery Level: {self.recovery_level}")
            
        except Exception as e:
            print(f"Recovery activation error: {e}")
    
    def _initialize_strategy_parameters(self, recovery_mode):
        """Initialize parameters for specific recovery strategy"""
        try:
            if recovery_mode == RecoveryMode.ADAPTIVE_MARTINGALE:
                self.martingale_multiplier = 1.5
                self.max_martingale_level = 6
                
            elif recovery_mode == RecoveryMode.SMART_GRID:
                self.grid_spacing = 200
                self.grid_levels = []
                self.grid_base_price = 0.0
                
            elif recovery_mode == RecoveryMode.DYNAMIC_HEDGE:
                self.hedge_ratio = 0.0
                self.hedge_positions = []
                self.target_delta = 0.0
                
            elif recovery_mode == RecoveryMode.CORRELATION_HEDGE:
                self.correlation_matrix = {}
                self.correlation_threshold = 0.7
                
        except Exception as e:
            print(f"Strategy parameter initialization error: {e}")
    
    def _execute_adaptive_martingale(self, actions, portfolio_state, market_data, mt5_interface):
        """Execute Adaptive Martingale recovery strategy"""
        try:
            result = {
                'success': False,
                'actions_taken': [],
                'new_positions': [],
                'recovery_metrics': {}
            }
            
            positions = portfolio_state.get('positions', [])
            losing_positions = [pos for pos in positions if pos.get('profit', 0) < 0]
            
            if not losing_positions:
                result['success'] = True
                result['actions_taken'].append("No losing positions found")
                return result
            
            # Calculate adaptive sizing
            market_direction = actions.get('market_direction', 0)
            position_size = actions.get('position_size', 0.01)
            volatility_filter = actions.get('volatility_filter', 0.5)
            
            # Volatility adjustment
            atr_ratio = market_data.get('atr_ratio', 0.01)
            volatility_multiplier = 1.0
            if atr_ratio > volatility_filter:
                volatility_multiplier = 0.7  # Reduce size in high volatility
            
            # Calculate recovery volume
            total_losing_volume = sum(pos.get('volume', 0) for pos in losing_positions)
            recovery_volume = total_losing_volume * self.martingale_multiplier * volatility_multiplier
            recovery_volume = max(0.01, min(recovery_volume, 0.10))  # Bounds check
            
            # Determine direction (follow dominant losing direction)
            buy_losses = sum(pos.get('profit', 0) for pos in losing_positions if pos.get('type', 0) == 0)
            sell_losses = sum(pos.get('profit', 0) for pos in losing_positions if pos.get('type', 0) == 1)
            
            if abs(buy_losses) > abs(sell_losses):
                recovery_direction = 'buy'  # Double down on buy positions
            else:
                recovery_direction = 'sell'  # Double down on sell positions
            
            # Apply market direction override
            if abs(market_direction) > 0.5:
                recovery_direction = 'buy' if market_direction > 0 else 'sell'
            
            # Execute recovery trade
            current_price = market_data.get('current_price', 2000)
            success = mt5_interface.place_order(
                symbol='XAUUSD',
                order_type=recovery_direction,
                volume=recovery_volume,
                price=current_price
            )
            
            if success:
                result['success'] = True
                result['actions_taken'].append(f"Adaptive Martingale: {recovery_direction.upper()} {recovery_volume:.2f} lots")
                result['new_positions'].append({
                    'type': recovery_direction,
                    'volume': recovery_volume,
                    'strategy': 'adaptive_martingale',
                    'recovery_level': self.recovery_level
                })
                
                # Update martingale parameters
                self.martingale_multiplier = min(self.martingale_multiplier * 1.1, 3.0)
                self.recovery_level += 1
                
            result['recovery_metrics'] = {
                'martingale_multiplier': self.martingale_multiplier,
                'recovery_volume': recovery_volume,
                'volatility_adjustment': volatility_multiplier,
                'total_losing_volume': total_losing_volume
            }
            
            return result
            
        except Exception as e:
            print(f"Adaptive Martingale execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_smart_grid(self, actions, portfolio_state, market_data, mt5_interface):
        """Execute Smart Grid recovery strategy"""
        try:
            result = {
                'success': False,
                'actions_taken': [],
                'new_positions': [],
                'recovery_metrics': {}
            }
            
            current_price = market_data.get('current_price', 2000)
            
            # Initialize grid if first time
            if not hasattr(self, 'grid_levels') or not self.grid_levels:
                self.grid_base_price = current_price
                self._setup_smart_grid(actions, market_data)
            
            # Check for grid level triggers
            triggered_levels = []
            for level in self.grid_levels:
                if not level['triggered']:
                    if level['type'] == 'buy' and current_price <= level['price']:
                        triggered_levels.append(level)
                    elif level['type'] == 'sell' and current_price >= level['price']:
                        triggered_levels.append(level)
            
            # Execute triggered levels
            for level in triggered_levels:
                success = mt5_interface.place_order(
                    symbol='XAUUSD',
                    order_type=level['type'],
                    volume=level['volume'],
                    price=level['price']
                )
                
                if success:
                    level['triggered'] = True
                    result['actions_taken'].append(f"Grid {level['type'].upper()}: {level['volume']:.2f} lots at {level['price']:.2f}")
                    result['new_positions'].append({
                        'type': level['type'],
                        'volume': level['volume'],
                        'price': level['price'],
                        'strategy': 'smart_grid'
                    })
            
            result['success'] = len(triggered_levels) > 0
            result['recovery_metrics'] = {
                'grid_base_price': self.grid_base_price,
                'active_levels': len([l for l in self.grid_levels if not l['triggered']]),
                'triggered_count': len(triggered_levels)
            }
            
            return result
            
        except Exception as e:
            print(f"Smart Grid execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _setup_smart_grid(self, actions, market_data):
        """Setup smart grid levels"""
        try:
            self.grid_levels = []
            
            # Dynamic grid spacing based on volatility
            atr = market_data.get('atr_14', 10)
            dynamic_spacing = max(atr * 0.5, 100)  # Minimum 100 points
            
            position_size = actions.get('position_size', 0.01)
            
            # Create grid levels
            for i in range(1, 6):  # 5 levels each side
                # Buy levels (below current price)
                buy_price = self.grid_base_price - (dynamic_spacing * i)
                buy_volume = position_size * (1 + i * 0.2)  # Increasing volume
                
                self.grid_levels.append({
                    'price': buy_price,
                    'type': 'buy',
                    'volume': min(buy_volume, 0.05),
                    'triggered': False,
                    'level': i
                })
                
                # Sell levels (above current price)
                sell_price = self.grid_base_price + (dynamic_spacing * i)
                sell_volume = position_size * (1 + i * 0.2)
                
                self.grid_levels.append({
                    'price': sell_price,
                    'type': 'sell',
                    'volume': min(sell_volume, 0.05),
                    'triggered': False,
                    'level': i
                })
                
        except Exception as e:
            print(f"Grid setup error: {e}")
    
    def _execute_dynamic_hedge(self, actions, portfolio_state, market_data, mt5_interface):
        """Execute Dynamic Hedge recovery strategy"""
        try:
            result = {
                'success': False,
                'actions_taken': [],
                'new_positions': [],
                'recovery_metrics': {}
            }
            
            positions = portfolio_state.get('positions', [])
            if not positions:
                return result
            
            # Calculate net exposure
            net_volume_buy = sum(pos.get('volume', 0) for pos in positions if pos.get('type', 0) == 0)
            net_volume_sell = sum(pos.get('volume', 0) for pos in positions if pos.get('type', 0) == 1)
            net_exposure = net_volume_buy - net_volume_sell
            
            # Get hedge parameters from actions
            hedge_ratio = actions.get('hedge_ratio', 0.5)
            target_hedge_volume = abs(net_exposure) * hedge_ratio
            
            if target_hedge_volume >= 0.01:  # Minimum hedge size
                # Determine hedge direction
                hedge_type = 'sell' if net_exposure > 0 else 'buy'
                
                # Execute hedge
                current_price = market_data.get('current_price', 2000)
                success = mt5_interface.place_order(
                    symbol='XAUUSD',
                    order_type=hedge_type,
                    volume=target_hedge_volume,
                    price=current_price
                )
                
                if success:
                    result['success'] = True
                    result['actions_taken'].append(f"Dynamic Hedge: {hedge_type.upper()} {target_hedge_volume:.2f} lots")
                    result['new_positions'].append({
                        'type': hedge_type,
                        'volume': target_hedge_volume,
                        'strategy': 'dynamic_hedge',
                        'hedge_ratio': hedge_ratio
                    })
                    
                    self.hedge_ratio = hedge_ratio
            
            result['recovery_metrics'] = {
                'net_exposure': net_exposure,
                'hedge_ratio': hedge_ratio,
                'target_hedge_volume': target_hedge_volume,
                'net_volume_buy': net_volume_buy,
                'net_volume_sell': net_volume_sell
            }
            
            return result
            
        except Exception as e:
            print(f"Dynamic Hedge execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_ai_ensemble(self, actions, portfolio_state, market_data, mt5_interface):
        """Execute AI Ensemble recovery strategy"""
        try:
            result = {
                'success': False,
                'actions_taken': [],
                'new_positions': [],
                'recovery_metrics': {}
            }
            
            # Combine multiple strategies based on market conditions
            strategies_to_use = []
            
            # Analyze market conditions to select strategies
            volatility = market_data.get('atr_ratio', 0.01)
            trend_strength = market_data.get('trend_strength', 0.5)
            
            if volatility > 0.012:
                strategies_to_use.append('volatility_breakout')
            
            if trend_strength > 0.6:
                strategies_to_use.append('trend_following')
            elif trend_strength < 0.4:
                strategies_to_use.append('mean_reversion')
            
            if len(portfolio_state.get('positions', [])) > 3:
                strategies_to_use.append('correlation_hedge')
            
            # Default to adaptive martingale if no specific strategy selected
            if not strategies_to_use:
                strategies_to_use.append('adaptive_martingale')
            
            # Execute ensemble of strategies
            ensemble_success = False
            for strategy in strategies_to_use:
                if strategy == 'adaptive_martingale':
                    sub_result = self._execute_adaptive_martingale(actions, portfolio_state, market_data, mt5_interface)
                elif strategy == 'volatility_breakout':
                    sub_result = self._execute_volatility_breakout(actions, portfolio_state, market_data, mt5_interface)
                elif strategy == 'mean_reversion':
                    sub_result = self._execute_mean_reversion(actions, portfolio_state, market_data, mt5_interface)
                elif strategy == 'correlation_hedge':
                    sub_result = self._execute_correlation_hedge(actions, portfolio_state, market_data, mt5_interface)
                else:
                    continue
                
                if sub_result.get('success', False):
                    ensemble_success = True
                    result['actions_taken'].extend(sub_result.get('actions_taken', []))
                    result['new_positions'].extend(sub_result.get('new_positions', []))
            
            result['success'] = ensemble_success
            result['recovery_metrics'] = {
                'strategies_used': strategies_to_use,
                'ensemble_size': len(strategies_to_use)
            }
            
            return result
            
        except Exception as e:
            print(f"AI Ensemble execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_correlation_hedge(self, actions, portfolio_state, market_data, mt5_interface):
        """Execute Correlation Hedge recovery strategy"""
        try:
            result = {
                'success': False,
                'actions_taken': [],
                'new_positions': [],
                'recovery_metrics': {}
            }
            
            positions = portfolio_state.get('positions', [])
            if len(positions) < 2:
                result['actions_taken'].append("Insufficient positions for correlation hedging")
                return result
            
            # Analyze position correlation (simplified)
            correlation_risk = self._calculate_position_correlation(positions)
            
            if correlation_risk > 0.7:  # High correlation detected
                # Implement diversification trade
                dominant_direction = self._get_dominant_position_direction(positions)
                hedge_volume = sum(pos.get('volume', 0) for pos in positions) * 0.3
                
                # Place opposite direction trade
                hedge_type = 'sell' if dominant_direction == 'buy' else 'buy'
                
                current_price = market_data.get('current_price', 2000)
                success = mt5_interface.place_order(
                    symbol='XAUUSD',
                    order_type=hedge_type,
                    volume=min(hedge_volume, 0.05),
                    price=current_price
                )
                
                if success:
                    result['success'] = True
                    result['actions_taken'].append(f"Correlation Hedge: {hedge_type.upper()} {hedge_volume:.2f} lots")
                    result['new_positions'].append({
                        'type': hedge_type,
                        'volume': hedge_volume,
                        'strategy': 'correlation_hedge'
                    })
            
            result['recovery_metrics'] = {
                'correlation_risk': correlation_risk,
                'position_count': len(positions)
            }
            
            return result
            
        except Exception as e:
            print(f"Correlation Hedge execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_volatility_breakout(self, actions, portfolio_state, market_data, mt5_interface):
        """Execute Volatility Breakout recovery strategy"""
        try:
            result = {
                'success': False,
                'actions_taken': [],
                'new_positions': [],
                'recovery_metrics': {}
            }
            
            # Check for volatility breakout conditions
            atr_ratio = market_data.get('atr_ratio', 0.01)
            volatility_regime = market_data.get('volatility_regime', 1.0)
            bb_position = market_data.get('bb_position', 0.5)
            
            # Volatility breakout signal
            breakout_signal = False
            breakout_direction = None
            
            if atr_ratio > 0.015 and volatility_regime > 1.3:  # High volatility expansion
                if bb_position > 0.8:  # Breaking upper Bollinger Band
                    breakout_signal = True
                    breakout_direction = 'buy'
                elif bb_position < 0.2:  # Breaking lower Bollinger Band
                    breakout_signal = True
                    breakout_direction = 'sell'
            
            if breakout_signal:
                # Calculate breakout position size
                volatility_filter = actions.get('volatility_filter', 0.5)
                base_size = actions.get('position_size', 0.01)
                
                # Increase size for strong breakouts
                breakout_multiplier = min(atr_ratio * 50, 2.0)  # Up to 2x multiplier
                breakout_volume = base_size * breakout_multiplier
                breakout_volume = max(0.01, min(breakout_volume, 0.08))
                
                # Execute breakout trade
                current_price = market_data.get('current_price', 2000)
                success = mt5_interface.place_order(
                    symbol='XAUUSD',
                    order_type=breakout_direction,
                    volume=breakout_volume,
                    price=current_price
                )
                
                if success:
                    result['success'] = True
                    result['actions_taken'].append(f"Volatility Breakout: {breakout_direction.upper()} {breakout_volume:.2f} lots")
                    result['new_positions'].append({
                        'type': breakout_direction,
                        'volume': breakout_volume,
                        'strategy': 'volatility_breakout'
                    })
            
            result['recovery_metrics'] = {
                'atr_ratio': atr_ratio,
                'volatility_regime': volatility_regime,
                'bb_position': bb_position,
                'breakout_signal': breakout_signal,
                'breakout_direction': breakout_direction
            }
            
            return result
            
        except Exception as e:
            print(f"Volatility Breakout execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _execute_mean_reversion(self, actions, portfolio_state, market_data, mt5_interface):
        """Execute Mean Reversion recovery strategy"""
        try:
            result = {
                'success': False,
                'actions_taken': [],
                'new_positions': [],
                'recovery_metrics': {}
            }
            
            # Check for mean reversion conditions
            rsi = market_data.get('rsi_14', 50)
            bb_position = market_data.get('bb_position', 0.5)
            price_change_20 = market_data.get('price_change_20', 0)
            
            # Mean reversion signals
            reversion_signal = False
            reversion_direction = None
            signal_strength = 0.0
            
            # Oversold conditions
            if rsi < 30 and bb_position < 0.2 and price_change_20 < -0.02:
                reversion_signal = True
                reversion_direction = 'buy'
                signal_strength = (30 - rsi) / 30 + (0.2 - bb_position) / 0.2
                
            # Overbought conditions
            elif rsi > 70 and bb_position > 0.8 and price_change_20 > 0.02:
                reversion_signal = True
                reversion_direction = 'sell'
                signal_strength = (rsi - 70) / 30 + (bb_position - 0.8) / 0.2
            
            if reversion_signal:
                # Calculate reversion position size
                base_size = actions.get('position_size', 0.01)
                reversion_volume = base_size * min(signal_strength, 2.0)
                reversion_volume = max(0.01, min(reversion_volume, 0.06))
                
                # Execute reversion trade
                current_price = market_data.get('current_price', 2000)
                success = mt5_interface.place_order(
                    symbol='XAUUSD',
                    order_type=reversion_direction,
                    volume=reversion_volume,
                    price=current_price
                )
                
                if success:
                    result['success'] = True
                    result['actions_taken'].append(f"Mean Reversion: {reversion_direction.upper()} {reversion_volume:.2f} lots")
                    result['new_positions'].append({
                        'type': reversion_direction,
                        'volume': reversion_volume,
                        'strategy': 'mean_reversion'
                    })
            
            result['recovery_metrics'] = {
                'rsi': rsi,
                'bb_position': bb_position,
                'price_change_20': price_change_20,
                'signal_strength': signal_strength,
                'reversion_signal': reversion_signal
            }
            
            return result
            
        except Exception as e:
            print(f"Mean Reversion execution error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_position_correlation(self, positions):
        """Calculate correlation risk among positions"""
        try:
            if len(positions) < 2:
                return 0.0
                
            # Simplified correlation calculation
            # In real implementation, use actual price correlation
            
            same_direction_count = 0
            total_pairs = 0
            
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    pos1_type = positions[i].get('type', 0)
                    pos2_type = positions[j].get('type', 0)
                    
                    if pos1_type == pos2_type:
                        same_direction_count += 1
                    total_pairs += 1
            
            correlation_ratio = same_direction_count / total_pairs if total_pairs > 0 else 0
            return correlation_ratio
            
        except:
            return 0.5
    
    def _get_dominant_position_direction(self, positions):
        """Get the dominant direction of positions"""
        try:
            buy_volume = sum(pos.get('volume', 0) for pos in positions if pos.get('type', 0) == 0)
            sell_volume = sum(pos.get('volume', 0) for pos in positions if pos.get('type', 0) == 1)
            
            return 'buy' if buy_volume > sell_volume else 'sell'
            
        except:
            return 'buy'
    
    def _update_recovery_state(self, execution_result, portfolio_state):
        """Update recovery state based on execution results"""
        try:
            if execution_result.get('success', False):
                # Record successful recovery action
                recovery_record = {
                    'timestamp': datetime.now().isoformat(),
                    'recovery_mode': self.current_mode.value,
                    'recovery_level': self.recovery_level,
                    'actions_taken': execution_result.get('actions_taken', []),
                    'new_positions': len(execution_result.get('new_positions', [])),
                    'portfolio_pnl': portfolio_state.get('total_pnl', 0),
                    'execution_metrics': execution_result.get('recovery_metrics', {})
                }
                
                self.recovery_history.append(recovery_record)
                
                # Update strategy performance
                strategy_key = self.current_mode.value
                if strategy_key in self.strategy_performance:
                    self.strategy_performance[strategy_key]['attempts'] += 1
                    
        except Exception as e:
            print(f"Recovery state update error: {e}")
    
    def _check_recovery_completion(self, portfolio_state, market_data):
        """Check if recovery is completed successfully"""
        try:
            if self.current_mode == RecoveryMode.INACTIVE:
                return False
                
            total_pnl = portfolio_state.get('total_pnl', 0)
            current_equity = portfolio_state.get('equity', 0)
            
            # Recovery success conditions
            recovery_success = False
            completion_reason = ""
            
            # Condition 1: Return to profitability
            if total_pnl >= 0:
                recovery_success = True
                completion_reason = "Returned to profitability"
                
            # Condition 2: Significant improvement from start
            elif (self.recovery_start_equity > 0 and 
                  current_equity > self.recovery_start_equity * 1.02):  # 2% improvement
                recovery_success = True
                completion_reason = "Significant equity improvement"
                
            # Condition 3: Time-based completion (prevent infinite recovery)
            elif (self.recovery_start_time and 
                  (datetime.now() - self.recovery_start_time).total_seconds() > 3600):  # 1 hour
                recovery_success = True
                completion_reason = "Time limit reached"
                
            # Emergency stop conditions
            emergency_stop = False
            stop_reason = ""
            
            # Emergency condition 1: Excessive loss
            if total_pnl < -self.emergency_stop_loss:
                emergency_stop = True
                stop_reason = f"Emergency stop: Loss exceeds ${self.emergency_stop_loss}"
                
            # Emergency condition 2: Maximum recovery levels reached
            elif self.recovery_level >= self.max_recovery_levels:
                emergency_stop = True
                stop_reason = f"Maximum recovery levels ({self.max_recovery_levels}) reached"
                
            # Emergency condition 3: Equity too low
            elif current_equity < self.recovery_start_equity * 0.5:  # 50% equity loss
                emergency_stop = True
                stop_reason = "Excessive equity drawdown"
            
            # Complete recovery
            if recovery_success or emergency_stop:
                self._complete_recovery(
                    success=recovery_success,
                    reason=completion_reason if recovery_success else stop_reason,
                    final_pnl=total_pnl,
                    final_equity=current_equity
                )
                return True
                
            return False
            
        except Exception as e:
            print(f"Recovery completion check error: {e}")
            return False
    
    def _complete_recovery(self, success, reason, final_pnl, final_equity):
        """Complete the recovery process"""
        try:
            recovery_duration = (datetime.now() - self.recovery_start_time).total_seconds() if self.recovery_start_time else 0
            
            # Update performance statistics
            if success:
                self.successful_recoveries += 1
                
            # Calculate efficiency
            equity_change = final_equity - self.recovery_start_equity if self.recovery_start_equity > 0 else 0
            efficiency_score = max(0, min(equity_change / max(abs(final_pnl), 1), 10))
            
            # Update strategy performance
            strategy_key = self.current_mode.value
            if strategy_key in self.strategy_performance:
                perf = self.strategy_performance[strategy_key]
                if success:
                    perf['success'] += 1
                
                # Update average time
                total_time = perf['avg_time'] * perf['attempts'] + recovery_duration
                perf['attempts'] += 1
                perf['avg_time'] = total_time / perf['attempts']
            
            # Create completion record
            completion_record = {
                'completion_time': datetime.now().isoformat(),
                'recovery_mode': self.current_mode.value,
                'success': success,
                'reason': reason,
                'duration_seconds': recovery_duration,
                'recovery_levels_used': self.recovery_level,
                'start_equity': self.recovery_start_equity,
                'final_equity': final_equity,
                'equity_change': equity_change,
                'final_pnl': final_pnl,
                'efficiency_score': efficiency_score,
                'actions_history': list(self.recovery_history)
            }
            
            # Save completion record
            self._save_recovery_completion(completion_record)
            
            # Reset recovery state
            self._reset_recovery_state()
            
            # Log completion
            status = "✅ SUCCESSFUL" if success else "❌ TERMINATED"
            print(f"🛡️ === RECOVERY COMPLETED: {status} ===")
            print(f"   Mode: {completion_record['recovery_mode']}")
            print(f"   Duration: {recovery_duration:.0f} seconds")
            print(f"   Levels Used: {self.recovery_level}")
            print(f"   Equity Change: ${equity_change:+.2f}")
            print(f"   Final PnL: ${final_pnl:+.2f}")
            print(f"   Reason: {reason}")
            print(f"   Efficiency: {efficiency_score:.2f}")
            
        except Exception as e:
            print(f"Recovery completion error: {e}")
    
    def _save_recovery_completion(self, completion_record):
        """Save recovery completion record"""
        try:
            os.makedirs('data/recovery_logs', exist_ok=True)
            
            # Save individual completion record
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/recovery_logs/recovery_completion_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(completion_record, f, indent=4, default=str)
            
            # Update summary statistics
            self._update_recovery_summary()
            
        except Exception as e:
            print(f"Recovery completion save error: {e}")
    
    def _update_recovery_summary(self):
        """Update recovery summary statistics"""
        try:
            summary = {
                'last_updated': datetime.now().isoformat(),
                'total_attempts': self.total_recovery_attempts,
                'successful_recoveries': self.successful_recoveries,
                'success_rate': self.successful_recoveries / max(self.total_recovery_attempts, 1),
                'strategy_performance': self.strategy_performance,
                'current_mode': self.current_mode.value,
                'recovery_efficiency': self.recovery_efficiency
            }
            
            summary_file = 'data/recovery_logs/recovery_summary.json'
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=4, default=str)
                
        except Exception as e:
            print(f"Recovery summary update error: {e}")
    
    def _reset_recovery_state(self):
        """Reset recovery system to inactive state"""
        try:
            self.current_mode = RecoveryMode.INACTIVE
            self.recovery_level = 0
            self.recovery_start_time = None
            self.recovery_start_equity = 0.0
            self.recovery_positions = []
            
            # Reset strategy-specific states
            self.martingale_multiplier = 1.5
            self.grid_levels = []
            self.hedge_ratio = 0.0
            self.hedge_positions = []
            
            print(f"🔄 Recovery system reset to INACTIVE")
            
        except Exception as e:
            print(f"Recovery reset error: {e}")
    
    def check_profit_opportunities(self, portfolio_state: Dict, market_data: Dict) -> List[Dict]:
        """Check for intelligent profit-taking opportunities"""
        try:
            positions = portfolio_state.get('positions', [])
            if not positions:
                return []
                
            return self.profit_manager.evaluate_exit_signal(positions, market_data)
            
        except Exception as e:
            print(f"Profit opportunities check error: {e}")
            return []
    
    def execute_smart_exits(self, exit_recommendations: List[Dict], mt5_interface) -> Dict:
        """Execute smart exit strategies"""
        try:
            execution_results = {
                'exits_executed': 0,
                'total_profit_taken': 0.0,
                'positions_closed': [],
                'errors': []
            }
            
            for recommendation in exit_recommendations:
                if recommendation['overall_recommendation'] in ['STRONG_EXIT', 'EXIT']:
                    position_id = recommendation['position_id']
                    
                    success = mt5_interface.close_position(position_id)
                    if success:
                        execution_results['exits_executed'] += 1
                        execution_results['total_profit_taken'] += recommendation['current_profit']
                        execution_results['positions_closed'].append({
                            'position_id': position_id,
                            'profit': recommendation['current_profit'],
                            'reason': recommendation['exit_signals'][0]['signal'] if recommendation['exit_signals'] else 'Unknown'
                        })
                        
                        print(f"💰 Smart Exit: Position {position_id}, Profit: ${recommendation['current_profit']:.2f}")
                    else:
                        execution_results['errors'].append(f"Failed to close position {position_id}")
            
            return execution_results
            
        except Exception as e:
            print(f"Smart exits execution error: {e}")
            return {'exits_executed': 0, 'errors': [str(e)]}
    
    def get_recovery_status(self) -> Dict:
        """Get comprehensive recovery system status"""
        try:
            status = {
                'recovery_active': self.current_mode != RecoveryMode.INACTIVE,
                'current_mode': self.current_mode.value,
                'recovery_level': self.recovery_level,
                'performance_stats': {
                    'total_attempts': self.total_recovery_attempts,
                    'successful_recoveries': self.successful_recoveries,
                    'success_rate': self.successful_recoveries / max(self.total_recovery_attempts, 1),
                    'recovery_efficiency': self.recovery_efficiency
                },
                'strategy_performance': self.strategy_performance,
                'current_session': {}
            }
            
            # Add current session info if recovery is active
            if self.current_mode != RecoveryMode.INACTIVE:
                current_duration = (datetime.now() - self.recovery_start_time).total_seconds() if self.recovery_start_time else 0
                status['current_session'] = {
                    'start_time': self.recovery_start_time.isoformat() if self.recovery_start_time else None,
                    'duration_seconds': current_duration,
                    'start_equity': self.recovery_start_equity,
                    'recovery_level': self.recovery_level,
                    'actions_taken': len(self.recovery_history)
                }
            
            # Add profit manager status
            status['profit_management'] = {
                'base_profit_per_lot': self.profit_manager.base_profit_per_lot,
                'dynamic_targeting': True,
                'ml_optimization': self.profit_manager.ml_profit_optimization,
                'success_rate': self.profit_manager.success_rate
            }
            
            return status
            
        except Exception as e:
            print(f"Recovery status error: {e}")
            return {
                'recovery_active': False,
                'current_mode': 'ERROR',
                'error': str(e)
            }
    
    def get_strategy_recommendations(self, portfolio_state: Dict, market_data: Dict) -> Dict:
        """Get AI-powered strategy recommendations"""
        try:
            recommendations = {
                'primary_strategy': RecoveryMode.INACTIVE,
                'confidence': 0.0,
                'reasoning': [],
                'alternative_strategies': [],
                'risk_assessment': 'MODERATE',
                'market_suitability': 0.5
            }
            
            # Analyze current situation
            analysis = self.analyze_recovery_need(portfolio_state, market_data)
            
            if analysis['recovery_needed']:
                recommendations['primary_strategy'] = analysis['recommended_mode']
                recommendations['confidence'] = min(analysis['urgency_level'] / 10.0, 1.0)
                recommendations['reasoning'] = analysis['trigger_reasons']
                
                # Risk assessment
                overall_risk = analysis.get('risk_assessment', {}).get('overall_risk', 0.5)
                if overall_risk > 0.7:
                    recommendations['risk_assessment'] = 'HIGH'
                elif overall_risk > 0.4:
                    recommendations['risk_assessment'] = 'MODERATE'
                else:
                    recommendations['risk_assessment'] = 'LOW'
                
                # Market suitability
                market_conditions = analysis.get('market_conditions', {})
                recommendations['market_suitability'] = market_conditions.get('recovery_favorability', 0.5)
                
                # Alternative strategies
                all_strategies = list(RecoveryMode)
                all_strategies.remove(RecoveryMode.INACTIVE)
                all_strategies.remove(recommendations['primary_strategy'])
                
                # Score alternatives
                alternative_scores = []
                for strategy in all_strategies[:3]:  # Top 3 alternatives
                    score = self._score_strategy_suitability(strategy, portfolio_state, market_data)
                    alternative_scores.append((strategy, score))
                
                alternative_scores.sort(key=lambda x: x[1], reverse=True)
                recommendations['alternative_strategies'] = [
                    {'strategy': strategy.value, 'score': score} 
                    for strategy, score in alternative_scores
                ]
            
            return recommendations
            
        except Exception as e:
            print(f"Strategy recommendations error: {e}")
            return {
                'primary_strategy': RecoveryMode.INACTIVE,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _score_strategy_suitability(self, strategy: RecoveryMode, portfolio_state: Dict, market_data: Dict) -> float:
        """Score how suitable a strategy is for current conditions"""
        try:
            score = 0.5  # Base score
            
            volatility = market_data.get('atr_ratio', 0.01)
            trend_strength = market_data.get('trend_strength', 0.5)
            total_pnl = portfolio_state.get('total_pnl', 0)
            position_count = len(portfolio_state.get('positions', []))
            
            if strategy == RecoveryMode.ADAPTIVE_MARTINGALE:
                # Good for moderate losses with clear trend
                if -200 < total_pnl < -50 and trend_strength > 0.6:
                    score += 0.3
                if volatility < 0.012:  # Prefer lower volatility
                    score += 0.2
                    
            elif strategy == RecoveryMode.SMART_GRID:
                # Good for sideways markets
                if trend_strength < 0.4:
                    score += 0.4
                if volatility > 0.008:  # Need some movement
                    score += 0.2
                    
            elif strategy == RecoveryMode.DYNAMIC_HEDGE:
                # Good for high volatility and multiple positions
                if volatility > 0.012:
                    score += 0.3
                if position_count > 3:
                    score += 0.2
                    
            elif strategy == RecoveryMode.VOLATILITY_BREAKOUT:
                # Good for high volatility environments
                if volatility > 0.015:
                    score += 0.4
                    
            elif strategy == RecoveryMode.MEAN_REVERSION:
                # Good for overbought/oversold conditions
                rsi = market_data.get('rsi_14', 50)
                if rsi > 70 or rsi < 30:
                    score += 0.3
                    
            # Adjust based on historical performance
            if strategy.value in self.strategy_performance:
                perf = self.strategy_performance[strategy.value]
                if perf['attempts'] > 0:
                    success_rate = perf['success'] / perf['attempts']
                    score += (success_rate - 0.5) * 0.2  # Adjust by historical success
            
            return max(0.0, min(score, 1.0))
            
        except Exception as e:
            print(f"Strategy scoring error: {e}")
            return 0.5
    
    def optimize_recovery_parameters(self, historical_data: List[Dict]) -> Dict:
        """Optimize recovery parameters based on historical performance"""
        try:
            optimization_results = {
                'parameter_updates': {},
                'expected_improvement': 0.0,
                'confidence': 0.0,
                'recommendations': []
            }
            
            if not historical_data or len(historical_data) < 10:
                optimization_results['recommendations'].append("Insufficient historical data for optimization")
                return optimization_results
            
            # Analyze successful vs failed recoveries
            successful = [r for r in historical_data if r.get('success', False)]
            failed = [r for r in historical_data if not r.get('success', False)]
            
            if len(successful) < 3:
                optimization_results['recommendations'].append("Need more successful recoveries for optimization")
                return optimization_results
            
            # Optimize martingale multiplier
            successful_multipliers = [r.get('martingale_multiplier', 1.5) for r in successful if 'martingale_multiplier' in r]
            if successful_multipliers:
                optimal_multiplier = np.mean(successful_multipliers)
                if abs(optimal_multiplier - self.martingale_multiplier) > 0.1:
                    optimization_results['parameter_updates']['martingale_multiplier'] = optimal_multiplier
                    self.martingale_multiplier = optimal_multiplier
            
            # Optimize trigger thresholds
            successful_triggers = [abs(r.get('trigger_loss', 100)) for r in successful if 'trigger_loss' in r]
            if successful_triggers:
                optimal_trigger = np.mean(successful_triggers)
                if abs(optimal_trigger - self.recovery_trigger_loss) > 20:
                    optimization_results['parameter_updates']['recovery_trigger_loss'] = optimal_trigger
                    self.recovery_trigger_loss = optimal_trigger
            
            # Calculate expected improvement
            if len(optimization_results['parameter_updates']) > 0:
                current_success_rate = len(successful) / len(historical_data)
                optimization_results['expected_improvement'] = min(0.15, len(optimization_results['parameter_updates']) * 0.05)
                optimization_results['confidence'] = min(len(historical_data) / 50.0, 1.0)
                
                optimization_results['recommendations'].append(f"Parameters optimized based on {len(historical_data)} historical records")
                optimization_results['recommendations'].append(f"Expected improvement: {optimization_results['expected_improvement']*100:.1f}%")
            
            return optimization_results
            
        except Exception as e:
            print(f"Parameter optimization error: {e}")
            return {'error': str(e)}
    
    def generate_recovery_report(self) -> str:
        """Generate comprehensive recovery system report"""
        try:
            report = "\n" + "="*80 + "\n"
            report += "🛡️ PROFESSIONAL RECOVERY ENGINE REPORT\n"
            report += "="*80 + "\n"
            
            # System Status
            report += f"📊 SYSTEM STATUS:\n"
            report += f"   Current Mode: {self.current_mode.value}\n"
            report += f"   Recovery Level: {self.recovery_level}\n"
            report += f"   Active Since: {self.recovery_start_time.strftime('%Y-%m-%d %H:%M:%S') if self.recovery_start_time else 'N/A'}\n"
            
            # Performance Statistics
            report += f"\n📈 PERFORMANCE STATISTICS:\n"
            report += f"   Total Attempts: {self.total_recovery_attempts}\n"
            report += f"   Successful Recoveries: {self.successful_recoveries}\n"
            success_rate = self.successful_recoveries / max(self.total_recovery_attempts, 1) * 100
            report += f"   Success Rate: {success_rate:.1f}%\n"
            report += f"   Recovery Efficiency: {self.recovery_efficiency:.3f}\n"
            
            # Strategy Performance
            report += f"\n🎯 STRATEGY PERFORMANCE:\n"
            for strategy, perf in self.strategy_performance.items():
                if perf['attempts'] > 0:
                    strategy_success_rate = perf['success'] / perf['attempts'] * 100
                    avg_time_min = perf['avg_time'] / 60
                    report += f"   {strategy}:\n"
                    report += f"      Attempts: {perf['attempts']}\n"
                    report += f"      Success Rate: {strategy_success_rate:.1f}%\n"
                    report += f"      Avg Time: {avg_time_min:.1f} minutes\n"
            
            # Current Session (if active)
            if self.current_mode != RecoveryMode.INACTIVE and self.recovery_start_time:
                current_duration = (datetime.now() - self.recovery_start_time).total_seconds()
                report += f"\n🔄 CURRENT SESSION:\n"
                report += f"   Duration: {current_duration/60:.1f} minutes\n"
                report += f"   Start Equity: ${self.recovery_start_equity:,.2f}\n"
                report += f"   Actions Taken: {len(self.recovery_history)}\n"
            
            # Recent Recovery History
            if self.recovery_history:
                report += f"\n📚 RECENT ACTIONS ({len(self.recovery_history)} records):\n"
                for i, record in enumerate(list(self.recovery_history)[-5:]):  # Last 5 actions
                    timestamp = record.get('timestamp', 'Unknown')
                    actions = record.get('actions_taken', [])
                    report += f"   {i+1}. {timestamp}: {', '.join(actions[:2])}\n"
            
            # Profit Management Status
            report += f"\n💰 PROFIT MANAGEMENT:\n"
            report += f"   Base Profit/Lot: ${self.profit_manager.base_profit_per_lot:.2f}\n"
            report += f"   Dynamic Targeting: {'✅' if self.profit_manager.ml_profit_optimization else '❌'}\n"
            report += f"   Success Rate: {self.profit_manager.success_rate:.1%}\n"
            
            # System Configuration
            report += f"\n⚙️ CONFIGURATION:\n"
            report += f"   Max Drawdown Threshold: {self.max_drawdown_threshold}%\n"
            report += f"   Recovery Trigger Loss: ${self.recovery_trigger_loss}\n"
            report += f"   Max Recovery Levels: {self.max_recovery_levels}\n"
            report += f"   Emergency Stop Loss: ${self.emergency_stop_loss}\n"
            report += f"   Martingale Multiplier: {self.martingale_multiplier:.2f}\n"
            
            report += "="*80
            
            return report
            
        except Exception as e:
            return f"Report generation error: {e}"


# ========================= FACTORY FUNCTIONS & EXPORTS =========================

def create_professional_recovery_engine(config=None):
    """Factory function to create professional recovery engine"""
    return ProfessionalRecoveryEngine(config)

# Keep old class name for backward compatibility  
RecoveryEngine = ProfessionalRecoveryEngine

# Export main classes
__all__ = [
    'ProfessionalRecoveryEngine',
    'RecoveryEngine',  # Backward compatibility
    'SmartProfitManager', 
    'RecoveryMode',
    'RecoveryTrigger',
    'create_professional_recovery_engine'
]
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

class RiskLevel(Enum):
    """Portfolio Risk Levels"""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    SPECULATIVE = "SPECULATIVE"
    EMERGENCY = "EMERGENCY"

class PortfolioMode(Enum):
    """Portfolio Management Modes"""
    ACCUMULATION = "ACCUMULATION"
    GROWTH = "GROWTH"
    PRESERVATION = "PRESERVATION"
    RECOVERY = "RECOVERY"
    DEFENSIVE = "DEFENSIVE"
    OPPORTUNISTIC = "OPPORTUNISTIC"

class MarketRegime(Enum):
    """Market Regime Classifications"""
    BULL_TREND = "BULL_TREND"
    BEAR_TREND = "BEAR_TREND"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    CRISIS = "CRISIS"

class AdvancedRiskEngine:
    """
    Advanced Risk Management Engine
    - Real-time risk monitoring
    - Dynamic position sizing
    - Correlation analysis
    - VaR calculations
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Risk parameters
        self.max_portfolio_var = self.config.get('max_portfolio_var', 5.0)  # 5% VaR
        self.correlation_threshold = self.config.get('correlation_threshold', 0.7)
        self.volatility_lookback = self.config.get('volatility_lookback', 20)
        self.confidence_level = self.config.get('confidence_level', 0.95)
        
        # Risk tracking
        self.var_history = deque(maxlen=100)
        self.correlation_matrix = {}
        self.volatility_estimates = {}
        self.position_risks = {}
        
        # Performance metrics
        self.sharpe_ratio = 0.0
        self.sortino_ratio = 0.0
        self.max_drawdown = 0.0
        self.calmar_ratio = 0.0
        
        print(f"🛡️ Advanced Risk Engine initialized")
    
    def calculate_portfolio_var(self, positions: List[Dict], market_data: Dict) -> Dict:
        """Calculate Portfolio Value at Risk using multiple methods"""
        try:
            if not positions:
                return {'var_95': 0.0, 'var_99': 0.0, 'expected_shortfall': 0.0}
            
            # Get portfolio value and weights
            total_value = sum(pos.get('volume', 0) * market_data.get('current_price', 2000) for pos in positions)
            if total_value == 0:
                return {'var_95': 0.0, 'var_99': 0.0, 'expected_shortfall': 0.0}
            
            # Method 1: Historical Simulation VaR
            historical_var = self._calculate_historical_var(positions, market_data)
            
            # Method 2: Parametric VaR
            parametric_var = self._calculate_parametric_var(positions, market_data)
            
            # Method 3: Monte Carlo VaR
            monte_carlo_var = self._calculate_monte_carlo_var(positions, market_data)
            
            # Combine methods (weighted average)
            var_95 = (historical_var['var_95'] * 0.4 + 
                     parametric_var['var_95'] * 0.3 + 
                     monte_carlo_var['var_95'] * 0.3)
            
            var_99 = (historical_var['var_99'] * 0.4 + 
                     parametric_var['var_99'] * 0.3 + 
                     monte_carlo_var['var_99'] * 0.3)
            
            # Expected Shortfall (Conditional VaR)
            expected_shortfall = var_99 * 1.3  # Estimate ES as 1.3 * VaR_99
            
            # Store in history
            var_result = {
                'timestamp': datetime.now().isoformat(),
                'var_95': var_95,
                'var_99': var_99,
                'expected_shortfall': expected_shortfall,
                'portfolio_value': total_value,
                'position_count': len(positions)
            }
            
            self.var_history.append(var_result)
            
            return var_result
            
        except Exception as e:
            print(f"Portfolio VaR calculation error: {e}")
            return {'var_95': 0.0, 'var_99': 0.0, 'expected_shortfall': 0.0}
    
    def _calculate_historical_var(self, positions, market_data):
        """Calculate VaR using historical simulation"""
        try:
            # Simplified historical VaR calculation
            # In real implementation, use actual historical returns
            
            current_volatility = market_data.get('atr_ratio', 0.01)
            portfolio_value = sum(pos.get('volume', 0) * market_data.get('current_price', 2000) for pos in positions)
            
            # Estimate daily volatility
            daily_vol = current_volatility * np.sqrt(1440)  # Scale to daily
            
            # Historical simulation (using normal distribution as approximation)
            np.random.seed(42)  # For reproducibility
            simulated_returns = np.random.normal(0, daily_vol, 1000)
            
            # Calculate VaR
            var_95 = np.percentile(simulated_returns, 5) * portfolio_value
            var_99 = np.percentile(simulated_returns, 1) * portfolio_value
            
            return {
                'var_95': abs(var_95),
                'var_99': abs(var_99),
                'method': 'historical_simulation'
            }
            
        except Exception as e:
            print(f"Historical VaR error: {e}")
            return {'var_95': 0.0, 'var_99': 0.0}
    
    def _calculate_parametric_var(self, positions, market_data):
        """Calculate VaR using parametric method"""
        try:
            current_volatility = market_data.get('atr_ratio', 0.01)
            portfolio_value = sum(pos.get('volume', 0) * market_data.get('current_price', 2000) for pos in positions)
            
            # Assume normal distribution
            daily_vol = current_volatility * np.sqrt(1440)
            
            # VaR = z-score * volatility * portfolio value
            z_95 = 1.645  # 95% confidence
            z_99 = 2.326  # 99% confidence
            
            var_95 = z_95 * daily_vol * portfolio_value
            var_99 = z_99 * daily_vol * portfolio_value
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'method': 'parametric'
            }
            
        except Exception as e:
            print(f"Parametric VaR error: {e}")
            return {'var_95': 0.0, 'var_99': 0.0}
    
    def _calculate_monte_carlo_var(self, positions, market_data):
        """Calculate VaR using Monte Carlo simulation"""
        try:
            current_volatility = market_data.get('atr_ratio', 0.01)
            portfolio_value = sum(pos.get('volume', 0) * market_data.get('current_price', 2000) for pos in positions)
            
            # Monte Carlo parameters
            num_simulations = 10000
            time_horizon = 1  # 1 day
            
            # Generate random scenarios
            np.random.seed(123)  # For reproducibility
            random_returns = np.random.normal(
                0,  # Assume zero mean return
                current_volatility * np.sqrt(time_horizon * 1440),  # Daily volatility
                num_simulations
            )
            
            # Calculate portfolio value changes
            value_changes = random_returns * portfolio_value
            
            # Calculate VaR
            var_95 = abs(np.percentile(value_changes, 5))
            var_99 = abs(np.percentile(value_changes, 1))
            
            return {
                'var_95': var_95,
                'var_99': var_99,
                'method': 'monte_carlo'
            }
            
        except Exception as e:
            print(f"Monte Carlo VaR error: {e}")
            return {'var_95': 0.0, 'var_99': 0.0}
    
    def calculate_position_correlation(self, positions: List[Dict]) -> Dict:
        """Calculate correlation matrix for positions"""
        try:
            if len(positions) < 2:
                return {'correlation_matrix': {}, 'max_correlation': 0.0, 'diversification_ratio': 1.0}
            
            # For XAUUSD positions, correlation is mainly based on direction
            correlations = {}
            
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions):
                    if i != j:
                        # Same direction = high correlation, opposite = negative correlation
                        pos1_type = pos1.get('type', 0)
                        pos2_type = pos2.get('type', 0)
                        
                        if pos1_type == pos2_type:
                            correlation = 0.85  # High positive correlation
                        else:
                            correlation = -0.75  # High negative correlation
                        
                        correlations[f"{i}_{j}"] = correlation
            
            # Calculate maximum correlation
            max_correlation = max(abs(corr) for corr in correlations.values()) if correlations else 0.0
            
            # Calculate diversification ratio
            total_positions = len(positions)
            same_direction_pairs = sum(1 for corr in correlations.values() if corr > 0.5)
            diversification_ratio = 1.0 - (same_direction_pairs / max(total_positions * (total_positions - 1) / 2, 1))
            
            return {
                'correlation_matrix': correlations,
                'max_correlation': max_correlation,
                'diversification_ratio': diversification_ratio,
                'position_count': total_positions
            }
            
        except Exception as e:
            print(f"Correlation calculation error: {e}")
            return {'correlation_matrix': {}, 'max_correlation': 0.0, 'diversification_ratio': 1.0}
    
    def assess_position_risk(self, position: Dict, market_data: Dict) -> Dict:
        """Assess individual position risk"""
        try:
            volume = position.get('volume', 0.01)
            entry_price = position.get('price_open', market_data.get('current_price', 2000))
            current_price = market_data.get('current_price', 2000)
            position_type = position.get('type', 0)
            
            # Calculate position value
            position_value = volume * current_price * 100  # Contract size
            
            # Price risk (distance from entry)
            price_change = abs(current_price - entry_price) / entry_price
            
            # Volatility risk
            atr = market_data.get('atr_14', 10)
            volatility_risk = (atr / current_price) * 100  # As percentage
            
            # Time risk (how long position has been open)
            position_age = position.get('age_minutes', 0)
            time_risk = min(position_age / 1440, 1.0)  # Normalize to 1 day
            
            # Market risk (based on market conditions)
            market_risk = self._assess_market_risk(market_data)
            
            # Combined risk score
            risk_components = {
                'price_risk': price_change * 100,
                'volatility_risk': volatility_risk,
                'time_risk': time_risk * 100,
                'market_risk': market_risk * 100
            }
            
            total_risk = np.mean(list(risk_components.values()))
            
            # Risk classification
            if total_risk > 80:
                risk_level = RiskLevel.EMERGENCY
            elif total_risk > 60:
                risk_level = RiskLevel.SPECULATIVE
            elif total_risk > 40:
                risk_level = RiskLevel.AGGRESSIVE
            elif total_risk > 20:
                risk_level = RiskLevel.MODERATE
            else:
                risk_level = RiskLevel.CONSERVATIVE
            
            return {
                'position_value': position_value,
                'total_risk_score': total_risk,
                'risk_level': risk_level.value,
                'risk_components': risk_components,
                'recommendations': self._get_risk_recommendations(risk_level, risk_components)
            }
            
        except Exception as e:
            print(f"Position risk assessment error: {e}")
            return {'total_risk_score': 50.0, 'risk_level': RiskLevel.MODERATE.value}
    
    def _assess_market_risk(self, market_data):
        """Assess overall market risk"""
        try:
            volatility = market_data.get('atr_ratio', 0.01)
            trend_strength = market_data.get('trend_strength', 0.5)
            rsi = market_data.get('rsi_14', 50)
            
            # High volatility = high risk
            vol_risk = min(volatility * 50, 1.0)
            
            # Extreme RSI = high risk
            rsi_risk = max(0, (abs(rsi - 50) - 20) / 30) if abs(rsi - 50) > 20 else 0
            
            # Weak trend = higher uncertainty
            trend_risk = 1.0 - trend_strength
            
            market_risk = (vol_risk * 0.5 + rsi_risk * 0.3 + trend_risk * 0.2)
            return min(market_risk, 1.0)
            
        except:
            return 0.5
    
    def _get_risk_recommendations(self, risk_level, risk_components):
        """Get risk management recommendations"""
        recommendations = []
        
        if risk_level == RiskLevel.EMERGENCY:
            recommendations.extend([
                "Consider immediate position closure",
                "Implement emergency stop loss",
                "Reduce position size significantly"
            ])
        elif risk_level == RiskLevel.SPECULATIVE:
            recommendations.extend([
                "Monitor position closely",
                "Consider partial profit taking",
                "Tighten stop loss levels"
            ])
        elif risk_level == RiskLevel.AGGRESSIVE:
            recommendations.extend([
                "Regular monitoring required",
                "Consider risk reduction strategies",
                "Implement trailing stops"
            ])
        
        # Specific component recommendations
        if risk_components.get('volatility_risk', 0) > 60:
            recommendations.append("High volatility detected - consider hedging")
        
        if risk_components.get('time_risk', 0) > 70:
            recommendations.append("Long-held position - review exit strategy")
        
        return recommendations

class IntelligentPositionSizer:
    """
    AI-Powered Position Sizing Engine
    - Kelly Criterion optimization
    - Risk parity approaches
    - Market regime adaptation
    - Machine learning predictions
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Sizing parameters
        self.base_risk_per_trade = self.config.get('base_risk_per_trade', 1.0)
        self.max_position_size = self.config.get('max_position_size', 0.10)
        self.min_position_size = self.config.get('min_position_size', 0.01)
        
        # Kelly Criterion parameters
        self.kelly_fraction = self.config.get('kelly_fraction', 0.25)  # Conservative Kelly
        self.win_rate_estimate = 0.55  # Initial estimate
        self.avg_win_loss_ratio = 1.5  # Initial estimate
        
        # Adaptive parameters
        self.volatility_adjustment = True
        self.correlation_adjustment = True
        self.regime_adjustment = True
        
        # Performance tracking
        self.sizing_decisions = deque(maxlen=500)
        self.performance_feedback = deque(maxlen=100)
        
        print(f"🎯 Intelligent Position Sizer initialized")
    
    def calculate_optimal_size(self, signal_strength: float, market_data: Dict, 
                             portfolio_state: Dict, action_vector: np.ndarray = None) -> Dict:
        """Calculate optimal position size using multiple methodologies"""
        try:
            # Base size from action vector if provided
            base_size = action_vector[1] if action_vector is not None and len(action_vector) > 1 else 0.01
            
            # Method 1: Kelly Criterion sizing
            kelly_size = self._calculate_kelly_size(signal_strength, market_data)
            
            # Method 2: Risk parity sizing
            risk_parity_size = self._calculate_risk_parity_size(portfolio_state, market_data)
            
            # Method 3: Volatility-adjusted sizing
            volatility_size = self._calculate_volatility_adjusted_size(base_size, market_data)
            
            # Method 4: Correlation-adjusted sizing
            correlation_size = self._calculate_correlation_adjusted_size(base_size, portfolio_state)
            
            # Method 5: Regime-adjusted sizing
            regime_size = self._calculate_regime_adjusted_size(base_size, market_data)
            
            # Combine methodologies using weighted average
            sizing_methods = {
                'kelly': {'size': kelly_size, 'weight': 0.25},
                'risk_parity': {'size': risk_parity_size, 'weight': 0.20},
                'volatility': {'size': volatility_size, 'weight': 0.20},
                'correlation': {'size': correlation_size, 'weight': 0.20},
                'regime': {'size': regime_size, 'weight': 0.15}
            }
            
            # Calculate weighted optimal size
            total_weight = sum(method['weight'] for method in sizing_methods.values())
            optimal_size = sum(method['size'] * method['weight'] for method in sizing_methods.values()) / total_weight
            
            # Apply constraints
            optimal_size = max(self.min_position_size, min(optimal_size, self.max_position_size))
            
            # Round to valid lot size
            optimal_size = round(optimal_size, 2)
            
            # Create sizing decision record
            sizing_decision = {
                'timestamp': datetime.now().isoformat(),
                'signal_strength': signal_strength,
                'base_size': base_size,
                'optimal_size': optimal_size,
                'methods': sizing_methods,
                'constraints_applied': optimal_size != sum(method['size'] * method['weight'] for method in sizing_methods.values()) / total_weight,
                'market_volatility': market_data.get('atr_ratio', 0.01),
                'portfolio_heat': portfolio_state.get('portfolio_heat', 0)
            }
            
            self.sizing_decisions.append(sizing_decision)
            
            return {
                'optimal_size': optimal_size,
                'sizing_methods': sizing_methods,
                'confidence': min(abs(signal_strength), 1.0),
                'risk_adjusted': True,
                'sizing_rationale': self._generate_sizing_rationale(sizing_methods, optimal_size)
            }
            
        except Exception as e:
            print(f"Position sizing error: {e}")
            return {
                'optimal_size': 0.01,
                'sizing_methods': {},
                'confidence': 0.5,
                'error': str(e)
            }
    
    def _calculate_kelly_size(self, signal_strength, market_data):
        """Calculate position size using Kelly Criterion"""
        try:
            # Update estimates based on recent performance
            self._update_kelly_estimates()
            
            # Kelly formula: f = (bp - q) / b
            # where: f = fraction to bet, b = odds, p = win probability, q = loss probability
            
            # Estimate win probability based on signal strength
            base_win_prob = self.win_rate_estimate
            signal_adjustment = signal_strength * 0.1  # Signal can adjust win rate by ±10%
            win_prob = max(0.1, min(0.9, base_win_prob + signal_adjustment))
            
            # Estimate odds (average win/loss ratio)
            odds = self.avg_win_loss_ratio
            
            # Kelly fraction
            kelly_fraction = (odds * win_prob - (1 - win_prob)) / odds
            
            # Conservative Kelly (use fraction of Kelly)
            conservative_kelly = kelly_fraction * self.kelly_fraction
            
            # Convert to position size (scale by base risk)
            kelly_size = max(0.01, conservative_kelly * self.base_risk_per_trade * 0.01)
            
            return min(kelly_size, 0.08)  # Cap at 0.08 lots
            
        except Exception as e:
            print(f"Kelly sizing error: {e}")
            return 0.01
    
    def _calculate_risk_parity_size(self, portfolio_state, market_data):
        """Calculate size based on risk parity principles"""
        try:
            positions = portfolio_state.get('positions', [])
            target_risk_per_position = 1.0  # 1% risk per position
            
            current_volatility = market_data.get('atr_ratio', 0.01)
            current_price = market_data.get('current_price', 2000)
            
            # Calculate position risk budget
            total_capital = portfolio_state.get('equity', 10000)
            position_risk_budget = total_capital * (target_risk_per_position / 100)
            
            # Size based on volatility
            # Risk = Position Size * Price * Volatility
            # Position Size = Risk Budget / (Price * Volatility)
            
            if current_volatility > 0 and current_price > 0:
                risk_parity_size = position_risk_budget / (current_price * current_volatility * 100)
                return max(0.01, min(risk_parity_size, 0.06))
            else:
                return 0.01
                
        except Exception as e:
            print(f"Risk parity sizing error: {e}")
            return 0.01
    
    def _calculate_volatility_adjusted_size(self, base_size, market_data):
        """Adjust size based on current market volatility"""
        try:
            current_volatility = market_data.get('atr_ratio', 0.01)
            avg_volatility = 0.01  # Baseline volatility
            
            # Volatility adjustment factor
            vol_adjustment = avg_volatility / current_volatility if current_volatility > 0 else 1.0
            
            # Limit adjustment to reasonable range
            vol_adjustment = max(0.5, min(vol_adjustment, 2.0))
            
            adjusted_size = base_size * vol_adjustment
            return max(0.01, min(adjusted_size, 0.08))
            
        except Exception as e:
            print(f"Volatility adjustment error: {e}")
            return base_size
    
    def _calculate_correlation_adjusted_size(self, base_size, portfolio_state):
        """Adjust size based on portfolio correlation"""
        try:
            positions = portfolio_state.get('positions', [])
            
            if len(positions) < 2:
                return base_size  # No correlation adjustment needed
            
            # Calculate portfolio concentration
            total_volume = sum(pos.get('volume', 0) for pos in positions)
            buy_volume = sum(pos.get('volume', 0) for pos in positions if pos.get('type', 0) == 0)
            sell_volume = total_volume - buy_volume
            
            # Concentration ratio
            concentration = abs(buy_volume - sell_volume) / max(total_volume, 0.01)
            
            # Reduce size if high concentration (high correlation)
            correlation_adjustment = 1.0 - (concentration * 0.3)  # Up to 30% reduction
            
            adjusted_size = base_size * correlation_adjustment
            return max(0.01, min(adjusted_size, 0.08))
            
        except Exception as e:
            print(f"Correlation adjustment error: {e}")
            return base_size
    
    def _calculate_regime_adjusted_size(self, base_size, market_data):
        """Adjust size based on market regime"""
        try:
            # Identify market regime
            volatility = market_data.get('atr_ratio', 0.01)
            trend_strength = market_data.get('trend_strength', 0.5)
            
            regime_adjustment = 1.0
            
            # High volatility regime - reduce size
            if volatility > 0.015:
                regime_adjustment *= 0.7
            
            # Strong trend regime - can increase size slightly
            elif trend_strength > 0.7:
                regime_adjustment *= 1.2
            
            # Sideways market - standard size
            elif trend_strength < 0.3:
                regime_adjustment *= 0.9
            
            adjusted_size = base_size * regime_adjustment
            return max(0.01, min(adjusted_size, 0.08))
            
        except Exception as e:
            print(f"Regime adjustment error: {e}")
            return base_size
    
    def _update_kelly_estimates(self):
        """Update Kelly Criterion estimates based on recent performance"""
        try:
            if len(self.performance_feedback) < 10:
                return  # Need more data
            
            recent_trades = list(self.performance_feedback)[-20:]  # Last 20 trades
            
            wins = [trade for trade in recent_trades if trade.get('pnl', 0) > 0]
            losses = [trade for trade in recent_trades if trade.get('pnl', 0) < 0]
            
            if len(wins) > 0 and len(losses) > 0:
                # Update win rate
                self.win_rate_estimate = len(wins) / len(recent_trades)
                
                # Update win/loss ratio
                avg_win = np.mean([trade['pnl'] for trade in wins])
                avg_loss = abs(np.mean([trade['pnl'] for trade in losses]))
                
                if avg_loss > 0:
                    self.avg_win_loss_ratio = avg_win / avg_loss
                    
        except Exception as e:
            print(f"Kelly estimates update error: {e}")
    
    def _generate_sizing_rationale(self, sizing_methods, optimal_size):
        """Generate rationale for sizing decision"""
        try:
            rationale = []
            
            # Find dominant method
            dominant_method = max(sizing_methods.items(), key=lambda x: x[1]['weight'])
            rationale.append(f"Primary method: {dominant_method[0]} (weight: {dominant_method[1]['weight']:.1%})")
            
            # Size category
            if optimal_size >= 0.05:
                rationale.append("Large position size - high confidence signal")
            elif optimal_size >= 0.03:
                rationale.append("Medium position size - moderate confidence")
            else:
                rationale.append("Small position size - conservative approach")
            
            # Method-specific notes
            kelly_size = sizing_methods.get('kelly', {}).get('size', 0)
            if kelly_size > optimal_size * 1.5:
                rationale.append("Kelly method suggests larger size - constrained by risk limits")
            
            return rationale
            
        except Exception as e:
            print(f"Rationale generation error: {e}")
            return ["Standard sizing applied"]
    
    def provide_feedback(self, trade_result: Dict):
        """Provide feedback on trade result for learning"""
        try:
            feedback = {
                'timestamp': datetime.now().isoformat(),
                'position_size': trade_result.get('volume', 0.01),
                'pnl': trade_result.get('pnl', 0),
                'duration': trade_result.get('duration_minutes', 0),
                'market_conditions': trade_result.get('market_conditions', {})
            }
            
            self.performance_feedback.append(feedback)
            
            # Trigger learning update
            if len(self.performance_feedback) % 10 == 0:
                self._update_kelly_estimates()
                
        except Exception as e:
            print(f"Feedback processing error: {e}")

class ProfessionalPortfolioManager:
    """
    Professional AI-Powered Portfolio Management System
    - Dynamic capital allocation
    - Advanced risk management
    - Market regime adaptation
    - ML-powered optimization
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        print(f"💼 Initializing Professional Portfolio Manager...")
        
        # Core configuration
        self.initial_capital = 0.0
        self.current_capital = 0.0
        self.target_capital = self.config.get('target_capital', 100000)
        
        # Dynamic allocation parameters
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 8.0)  # 8% max risk
        self.target_return = self.config.get('target_return', 15.0)  # 15% annual target
        self.rebalance_threshold = self.config.get('rebalance_threshold', 5.0)  # 5% drift
        
        # Portfolio modes and states
        self.current_mode = PortfolioMode.GROWTH
        self.risk_level = RiskLevel.MODERATE
        self.market_regime = MarketRegime.SIDEWAYS
        
        # Advanced components
        self.risk_engine = AdvancedRiskEngine(config)
        self.position_sizer = IntelligentPositionSizer(config)
        
        # Dynamic thresholds (auto-adjusting)
        self.dynamic_thresholds = {
            'profit_per_lot': 10.0,
            'portfolio_target': 50.0,
            'max_loss_limit': 200.0,
            'stop_loss_threshold': 500.0
        }
        
        # Portfolio state tracking
        self.portfolio_history = deque(maxlen=1000)
        self.performance_metrics = {
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0
        }
        
        # Real-time monitoring
        self.portfolio_heat = 0.0
        self.current_drawdown = 0.0
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.daily_start_balance = 0.0
        
        # Risk controls
        self.trading_allowed = True
        self.risk_reduction_active = False
        self.emergency_mode = False
        
        # Machine learning components
        self.ml_predictions = {}
        self.regime_classifier = None
        self.optimization_engine = None
        
        # Event tracking
        self.rebalance_events = deque(maxlen=50)
        self.risk_events = deque(maxlen=100)
        self.performance_events = deque(maxlen=200)
        
        print(f"✅ Professional Portfolio Manager initialized:")
        print(f"   Target Return: {self.target_return}%")
        print(f"   Max Portfolio Risk: {self.max_portfolio_risk}%")
        print(f"   Advanced Risk Engine: Enabled")
        print(f"   Intelligent Position Sizer: Enabled")
    
    def initialize_portfolio(self, mt5_interface) -> bool:
        """Initialize portfolio with current account state"""
        try:
            account_info = mt5_interface.get_account_info()
            if not account_info:
                print("❌ Failed to get account info for portfolio initialization")
                return False
            
            self.initial_capital = account_info.get('balance', 0)
            self.current_capital = account_info.get('equity', 0)
            self.peak_equity = self.current_capital
            self.daily_start_balance = self.current_capital
            
            # Calculate initial dynamic thresholds
            self._recalculate_dynamic_thresholds()
            
            # Set initial portfolio mode based on capital
            self._determine_initial_mode()
            
            # Initialize performance tracking
            self._initialize_performance_tracking()
            
            print(f"💼 Portfolio Initialized Successfully:")
            print(f"   Initial Capital: ${self.initial_capital:,.2f}")
            print(f"   Current Equity: ${self.current_capital:,.2f}")
            print(f"   Portfolio Mode: {self.current_mode.value}")
            print(f"   Risk Level: {self.risk_level.value}")
            print(f"   Profit/Lot Target: ${self.dynamic_thresholds['profit_per_lot']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Portfolio initialization error: {e}")
            return False
    
    def _recalculate_dynamic_thresholds(self):
        """Recalculate dynamic thresholds based on current capital"""
        try:
            capital = max(self.current_capital, 1000)  # Minimum for calculations
            
            # Capital scaling factors
            capital_tier = self._get_capital_tier(capital)
            scaling_factor = capital_tier['scaling_factor']
            risk_factor = capital_tier['risk_factor']
            
            # Dynamic profit targets
            base_profit_per_lot = 5.0
            self.dynamic_thresholds['profit_per_lot'] = base_profit_per_lot * scaling_factor
            
            # Portfolio targets
            portfolio_target_pct = min(max(0.3, capital / 100000 * 2), 3.0)  # 0.3% to 3%
            self.dynamic_thresholds['portfolio_target'] = capital * (portfolio_target_pct / 100)
            
            # Loss limits
            max_loss_pct = min(max(1.0, capital / 50000), 4.0)  # 1% to 4%
            self.dynamic_thresholds['max_loss_limit'] = capital * (max_loss_pct / 100)
            
            # Stop loss threshold
            stop_loss_pct = min(max(2.0, capital / 25000), 6.0)  # 2% to 6%
            self.dynamic_thresholds['stop_loss_threshold'] = capital * (stop_loss_pct / 100)
            
            # Update risk engine parameters
            self.risk_engine.max_portfolio_var = max_loss_pct
            
            print(f"🎯 Dynamic thresholds updated for ${capital:,.0f} capital")
            
        except Exception as e:
            print(f"Threshold calculation error: {e}")
    
    def _get_capital_tier(self, capital):
        """Get capital tier classification"""
        if capital >= 100000:
            return {'tier': 'INSTITUTIONAL', 'scaling_factor': 3.0, 'risk_factor': 1.5}
        elif capital >= 50000:
            return {'tier': 'ADVANCED', 'scaling_factor': 2.5, 'risk_factor': 1.3}
        elif capital >= 25000:
            return {'tier': 'INTERMEDIATE', 'scaling_factor': 2.0, 'risk_factor': 1.1}
        elif capital >= 10000:
            return {'tier': 'STANDARD', 'scaling_factor': 1.5, 'risk_factor': 1.0}
        else:
            return {'tier': 'BEGINNER', 'scaling_factor': 1.0, 'risk_factor': 0.8}
    
    def _determine_initial_mode(self):
        """Determine initial portfolio mode based on capital and goals"""
        try:
            capital_ratio = self.current_capital / max(self.target_capital, 100000)
            
            if capital_ratio < 0.1:  # Less than 10% of target
                self.current_mode = PortfolioMode.ACCUMULATION
                self.risk_level = RiskLevel.AGGRESSIVE
            elif capital_ratio < 0.5:  # 10-50% of target
                self.current_mode = PortfolioMode.GROWTH
                self.risk_level = RiskLevel.MODERATE
            elif capital_ratio < 0.8:  # 50-80% of target
                self.current_mode = PortfolioMode.GROWTH
                self.risk_level = RiskLevel.MODERATE
            else:  # Above 80% of target
                self.current_mode = PortfolioMode.PRESERVATION
                self.risk_level = RiskLevel.CONSERVATIVE
                
        except Exception as e:
            print(f"Mode determination error: {e}")
            self.current_mode = PortfolioMode.GROWTH
            self.risk_level = RiskLevel.MODERATE
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking systems"""
        try:
            initial_snapshot = {
                'timestamp': datetime.now().isoformat(),
                'capital': self.current_capital,
                'mode': self.current_mode.value,
                'risk_level': self.risk_level.value,
                'thresholds': self.dynamic_thresholds.copy(),
                'initialization': True
            }
            
            self.portfolio_history.append(initial_snapshot)
            
        except Exception as e:
            print(f"Performance tracking initialization error: {e}")
    
    def update_portfolio_status(self, mt5_interface, market_data: Dict = None):
        """Comprehensive portfolio status update"""
        try:
            # Get current account state
            account_info = mt5_interface.get_account_info()
            if not account_info:
                return
            
            positions = mt5_interface.get_positions()
            
            # Update basic metrics
            old_capital = self.current_capital
            self.current_capital = account_info.get('equity', 0)
            current_balance = account_info.get('balance', 0)
            
            # Update peak equity and drawdown
            if self.current_capital > self.peak_equity:
                self.peak_equity = self.current_capital
            
            self.current_drawdown = ((self.peak_equity - self.current_capital) / self.peak_equity) * 100 if self.peak_equity > 0 else 0
            
            # Update daily PnL
            if self.daily_start_balance > 0:
                self.daily_pnl = ((self.current_capital - self.daily_start_balance) / self.daily_start_balance) * 100
            
            # Calculate portfolio heat
            self.portfolio_heat = self._calculate_portfolio_heat(positions, market_data)
            
            # Update risk assessments
            self._update_risk_assessments(positions, market_data)
            
            # Update market regime
            if market_data:
                self._update_market_regime(market_data)
            
            # Check for mode changes
            self._check_mode_changes()
            
            # Update dynamic thresholds if significant capital change
            if abs(self.current_capital - old_capital) / max(old_capital, 1) > 0.1:  # 10% change
                self._recalculate_dynamic_thresholds()
            
            # Record portfolio snapshot
            self._record_portfolio_snapshot(positions, market_data)
            
            # Update performance metrics
            self._update_performance_metrics(positions)
            
            # Check risk triggers
            self._check_risk_triggers()
            
        except Exception as e:
            print(f"Portfolio status update error: {e}")
    
    def calculate_optimal_position_size(self, signal_strength: float, market_data: Dict, 
                                      action_vector: np.ndarray = None) -> Dict:
        """Calculate optimal position size using AI engine"""
        try:
            # Get current portfolio state
            portfolio_state = {
                'equity': self.current_capital,
                'positions': [],  # Would be populated with actual positions
                'portfolio_heat': self.portfolio_heat,
                'drawdown': self.current_drawdown,
                'mode': self.current_mode.value
            }
            
            # Use intelligent position sizer
            sizing_result = self.position_sizer.calculate_optimal_size(
                signal_strength, market_data, portfolio_state, action_vector
            )
            
            # Apply portfolio-level adjustments
            optimal_size = sizing_result['optimal_size']
            
            # Mode-based adjustments
            if self.current_mode == PortfolioMode.RECOVERY:
                optimal_size *= 0.5  # Conservative in recovery
            elif self.current_mode == PortfolioMode.DEFENSIVE:
                optimal_size *= 0.7  # Defensive mode
            elif self.current_mode == PortfolioMode.ACCUMULATION:
                optimal_size *= 1.2  # Slightly more aggressive
            elif self.current_mode == PortfolioMode.OPPORTUNISTIC:
                optimal_size *= 1.5  # More aggressive
            
            # Risk level adjustments
            if self.risk_level == RiskLevel.CONSERVATIVE:
                optimal_size *= 0.6
            elif self.risk_level == RiskLevel.AGGRESSIVE:
                optimal_size *= 1.3
            elif self.risk_level == RiskLevel.EMERGENCY:
                optimal_size *= 0.3
            
            # Market regime adjustments
            if self.market_regime == MarketRegime.CRISIS:
                optimal_size *= 0.4
            elif self.market_regime == MarketRegime.HIGH_VOLATILITY:
                optimal_size *= 0.7
            elif self.market_regime == MarketRegime.LOW_VOLATILITY:
                optimal_size *= 1.1
            
            # Final bounds check
            optimal_size = max(0.01, min(optimal_size, 0.10))
            optimal_size = round(optimal_size, 2)
            
            return {
                'optimal_size': optimal_size,
                'base_calculation': sizing_result,
                'adjustments_applied': {
                    'mode_factor': self._get_mode_factor(),
                    'risk_factor': self._get_risk_factor(),
                    'regime_factor': self._get_regime_factor()
                },
                'confidence': sizing_result.get('confidence', 0.5),
                'rationale': self._generate_size_rationale(optimal_size, sizing_result)
            }
            
        except Exception as e:
            print(f"Position size calculation error: {e}")
            return {
                'optimal_size': 0.01,
                'error': str(e),
                'confidence': 0.0
            }
    def _get_mode_factor(self):
        """Get position size factor based on current mode"""
        mode_factors = {
            PortfolioMode.ACCUMULATION: 1.2,
            PortfolioMode.GROWTH: 1.0,
            PortfolioMode.PRESERVATION: 0.8,
            PortfolioMode.RECOVERY: 0.5,
            PortfolioMode.DEFENSIVE: 0.7,
            PortfolioMode.OPPORTUNISTIC: 1.5
        }
        return mode_factors.get(self.current_mode, 1.0)
    
    def _get_risk_factor(self):
        """Get position size factor based on risk level"""
        risk_factors = {
            RiskLevel.CONSERVATIVE: 0.6,
            RiskLevel.MODERATE: 1.0,
            RiskLevel.AGGRESSIVE: 1.3,
            RiskLevel.SPECULATIVE: 1.6,
            RiskLevel.EMERGENCY: 0.3
        }
        return risk_factors.get(self.risk_level, 1.0)
    
    def _get_regime_factor(self):
        """Get position size factor based on market regime"""
        regime_factors = {
            MarketRegime.BULL_TREND: 1.1,
            MarketRegime.BEAR_TREND: 0.9,
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.HIGH_VOLATILITY: 0.7,
            MarketRegime.LOW_VOLATILITY: 1.1,
            MarketRegime.CRISIS: 0.4
        }
        return regime_factors.get(self.market_regime, 1.0)
    
    def _calculate_portfolio_heat(self, positions, market_data):
        """Calculate current portfolio heat percentage"""
        try:
            if not positions:
                return 0.0
            
            total_exposure = 0.0
            current_price = market_data.get('current_price', 2000) if market_data else 2000
            
            for position in positions:
                volume = position.get('volume', 0)
                # Calculate exposure as volume * current price * contract size
                position_exposure = volume * current_price * 100  # XAUUSD contract size
                total_exposure += position_exposure
            
            # Portfolio heat as percentage of equity
            heat_percentage = (total_exposure / max(self.current_capital, 1)) * 100
            return min(heat_percentage, 100.0)  # Cap at 100%
            
        except Exception as e:
            print(f"Portfolio heat calculation error: {e}")
            return 0.0
    
    def _update_risk_assessments(self, positions, market_data):
        """Update comprehensive risk assessments"""
        try:
            if not positions or not market_data:
                return
            
            # Calculate VaR
            var_results = self.risk_engine.calculate_portfolio_var(positions, market_data)
            
            # Calculate correlation risk
            correlation_results = self.risk_engine.calculate_position_correlation(positions)
            
            # Assess individual position risks
            position_risks = []
            for position in positions:
                risk_assessment = self.risk_engine.assess_position_risk(position, market_data)
                position_risks.append(risk_assessment)
            
            # Update risk metrics
            self.risk_metrics = {
                'var_95': var_results.get('var_95', 0),
                'var_99': var_results.get('var_99', 0),
                'expected_shortfall': var_results.get('expected_shortfall', 0),
                'max_correlation': correlation_results.get('max_correlation', 0),
                'diversification_ratio': correlation_results.get('diversification_ratio', 1),
                'avg_position_risk': np.mean([r.get('total_risk_score', 0) for r in position_risks]) if position_risks else 0,
                'high_risk_positions': len([r for r in position_risks if r.get('total_risk_score', 0) > 70])
            }
            
        except Exception as e:
            print(f"Risk assessment update error: {e}")
    
    def _update_market_regime(self, market_data):
        """Update current market regime classification"""
        try:
            volatility = market_data.get('atr_ratio', 0.01)
            trend_strength = market_data.get('trend_strength', 0.5)
            price_change_20 = market_data.get('price_change_20', 0)
            
            # Classify market regime
            if volatility > 0.02:  # Very high volatility
                self.market_regime = MarketRegime.CRISIS
            elif volatility > 0.015:  # High volatility
                self.market_regime = MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.005:  # Very low volatility
                self.market_regime = MarketRegime.LOW_VOLATILITY
            elif trend_strength > 0.7 and price_change_20 > 0.02:  # Strong uptrend
                self.market_regime = MarketRegime.BULL_TREND
            elif trend_strength > 0.7 and price_change_20 < -0.02:  # Strong downtrend
                self.market_regime = MarketRegime.BEAR_TREND
            else:  # Default to sideways
                self.market_regime = MarketRegime.SIDEWAYS
                
        except Exception as e:
            print(f"Market regime update error: {e}")
    
    def should_allow_trade(self, action_type: str, signal_strength: float = 0.0) -> Dict:
        """Determine if trade should be allowed based on portfolio state"""
        try:
            decision = {
                'allowed': False,
                'reason': '',
                'confidence': 0.0,
                'restrictions': [],
                'recommendations': []
            }
            
            # Emergency mode check
            if self.emergency_mode:
                decision['reason'] = 'Trading blocked - Emergency mode active'
                decision['restrictions'].append('Emergency stop due to excessive losses')
                return decision
            
            # Basic trading allowed check
            if not self.trading_allowed:
                decision['reason'] = 'Trading blocked - Risk management restrictions'
                decision['restrictions'].append('Automatic trading restrictions in place')
                return decision
            
            # Check specific action types
            if action_type in ['buy', 'sell', 'entry']:
                # New position checks
                if self.portfolio_heat > 12.0:
                    decision['reason'] = f'New positions blocked - Portfolio heat too high: {self.portfolio_heat:.1f}%'
                    decision['restrictions'].append('Portfolio heat exceeds safe limits')
                    return decision
                
                if self.current_drawdown > 10.0:
                    decision['reason'] = f'New positions blocked - High drawdown: {self.current_drawdown:.1f}%'
                    decision['restrictions'].append('Drawdown exceeds conservative limits')
                    return decision
                
                if self.daily_pnl < -6.0:
                    decision['reason'] = f'New positions blocked - Daily loss limit approached: {self.daily_pnl:.1f}%'
                    decision['restrictions'].append('Daily loss limits approaching')
                    return decision
                
                # Signal strength requirements
                min_signal_strength = self._get_min_signal_strength()
                if abs(signal_strength) < min_signal_strength:
                    decision['reason'] = f'Signal too weak: {signal_strength:.2f} < {min_signal_strength:.2f}'
                    decision['restrictions'].append('Signal strength below threshold for current conditions')
                    return decision
            
            # Market regime restrictions
            if self.market_regime == MarketRegime.CRISIS and action_type in ['buy', 'sell']:
                decision['reason'] = 'New positions restricted during crisis conditions'
                decision['restrictions'].append('Crisis market regime detected')
                return decision
            
            # Allow trade with confidence assessment
            decision['allowed'] = True
            decision['reason'] = 'Trade approved by portfolio manager'
            decision['confidence'] = self._calculate_trade_confidence(action_type, signal_strength)
            
            # Add recommendations
            decision['recommendations'] = self._get_trade_recommendations(action_type)
            
            return decision
            
        except Exception as e:
            print(f"Trade authorization error: {e}")
            return {
                'allowed': False,
                'reason': f'Authorization error: {str(e)}',
                'confidence': 0.0
            }
    
    def _get_min_signal_strength(self):
        """Get minimum signal strength required based on current conditions"""
        base_threshold = 0.3
        
        # Adjust based on risk level
        if self.risk_level == RiskLevel.CONSERVATIVE:
            base_threshold = 0.6
        elif self.risk_level == RiskLevel.AGGRESSIVE:
            base_threshold = 0.2
        elif self.risk_level == RiskLevel.EMERGENCY:
            base_threshold = 0.8
        
        # Adjust based on portfolio state
        if self.current_drawdown > 5.0:
            base_threshold += 0.2
        
        if self.portfolio_heat > 8.0:
            base_threshold += 0.1
        
        # Adjust based on market regime
        if self.market_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRISIS]:
            base_threshold += 0.2
        
        return min(base_threshold, 0.9)  # Cap at 0.9
    
    def _calculate_trade_confidence(self, action_type, signal_strength):
        """Calculate confidence level for trade approval"""
        try:
            base_confidence = abs(signal_strength)
            
            # Adjust based on portfolio health
            health_factor = 1.0
            
            if self.current_drawdown < 2.0:
                health_factor = 1.2  # High confidence when performing well
            elif self.current_drawdown > 8.0:
                health_factor = 0.6  # Lower confidence during drawdown
            
            # Adjust based on portfolio heat
            if self.portfolio_heat < 5.0:
                heat_factor = 1.1
            elif self.portfolio_heat > 10.0:
                heat_factor = 0.7
            else:
                heat_factor = 1.0
            
            # Market regime factor
            regime_confidence = {
                MarketRegime.BULL_TREND: 1.1,
                MarketRegime.BEAR_TREND: 1.0,
                MarketRegime.SIDEWAYS: 0.8,
                MarketRegime.HIGH_VOLATILITY: 0.7,
                MarketRegime.LOW_VOLATILITY: 1.0,
                MarketRegime.CRISIS: 0.4
            }
            
            regime_factor = regime_confidence.get(self.market_regime, 0.8)
            
            # Calculate final confidence
            confidence = base_confidence * health_factor * heat_factor * regime_factor
            return max(0.0, min(confidence, 1.0))
            
        except Exception as e:
            print(f"Confidence calculation error: {e}")
            return 0.5
    
    def _get_trade_recommendations(self, action_type):
        """Get trade recommendations based on current portfolio state"""
        recommendations = []
        
        try:
            # General recommendations based on portfolio state
            if self.current_mode == PortfolioMode.RECOVERY:
                recommendations.extend([
                    "Use smaller position sizes during recovery",
                    "Focus on high-probability setups",
                    "Consider quicker profit-taking"
                ])
            
            elif self.current_mode == PortfolioMode.DEFENSIVE:
                recommendations.extend([
                    "Maintain defensive posture",
                    "Implement tighter stop losses",
                    "Avoid aggressive position sizing"
                ])
            
            elif self.current_mode == PortfolioMode.OPPORTUNISTIC:
                recommendations.extend([
                    "Consider larger position sizes for strong signals",
                    "Take advantage of favorable market conditions",
                    "Monitor for trend continuation opportunities"
                ])
            
            # Risk-specific recommendations
            if self.portfolio_heat > 8.0:
                recommendations.append("Monitor portfolio heat levels closely")
            
            if self.current_drawdown > 3.0:
                recommendations.append("Consider risk reduction strategies")
            
            # Market regime recommendations
            if self.market_regime == MarketRegime.HIGH_VOLATILITY:
                recommendations.extend([
                    "Use volatility-adjusted position sizing",
                    "Consider shorter holding periods",
                    "Implement dynamic stop losses"
                ])
            
            return recommendations
            
        except Exception as e:
            print(f"Recommendations generation error: {e}")
            return ["Monitor positions closely"]
    
    def should_take_profit(self, positions: List[Dict], market_data: Dict = None) -> List[Dict]:
        """Determine profit-taking opportunities with advanced logic"""
        try:
            profit_decisions = []
            
            if not positions:
                return profit_decisions
            
            total_pnl = sum(pos.get('profit', 0) for pos in positions)
            
            # Portfolio-level profit taking
            portfolio_decision = self._evaluate_portfolio_profit_taking(positions, total_pnl)
            if portfolio_decision['action'] != 'HOLD':
                profit_decisions.append(portfolio_decision)
            
            # Individual position analysis
            for position in positions:
                individual_decision = self._evaluate_individual_profit_taking(position, market_data)
                if individual_decision['action'] != 'HOLD':
                    profit_decisions.append(individual_decision)
            
            return profit_decisions
            
        except Exception as e:
            print(f"Profit evaluation error: {e}")
            return []
    
    def _evaluate_portfolio_profit_taking(self, positions, total_pnl):
        """Evaluate portfolio-level profit taking"""
        try:
            decision = {
                'type': 'PORTFOLIO',
                'action': 'HOLD',
                'reason': '',
                'confidence': 0.0,
                'target_pnl': total_pnl,
                'positions_affected': len(positions)
            }
            
            # Get current thresholds
            portfolio_target = self.dynamic_thresholds['portfolio_target']
            
            # Strong profit target reached
            if total_pnl >= portfolio_target:
                decision['action'] = 'CLOSE_ALL'
                decision['reason'] = f'Portfolio target reached: ${total_pnl:.2f} >= ${portfolio_target:.2f}'
                decision['confidence'] = 0.9
                
            # Conservative target with favorable conditions
            elif total_pnl >= portfolio_target * 0.7:
                # Check if conditions favor taking profit
                if (self.current_drawdown < 2.0 and 
                    self.market_regime not in [MarketRegime.BULL_TREND] and
                    self.portfolio_heat > 6.0):
                    decision['action'] = 'CLOSE_PROFITABLE'
                    decision['reason'] = f'Conservative profit taking: ${total_pnl:.2f} with favorable conditions'
                    decision['confidence'] = 0.7
            
            # Risk-based profit taking
            elif total_pnl > 0 and self.portfolio_heat > 12.0:
                decision['action'] = 'CLOSE_PROFITABLE'
                decision['reason'] = f'Risk-based profit taking: High portfolio heat {self.portfolio_heat:.1f}%'
                decision['confidence'] = 0.6
            
            # Mode-specific profit taking
            elif self.current_mode == PortfolioMode.DEFENSIVE and total_pnl > portfolio_target * 0.3:
                decision['action'] = 'CLOSE_PROFITABLE'
                decision['reason'] = 'Defensive mode - securing available profits'
                decision['confidence'] = 0.8
            
            return decision
            
        except Exception as e:
            print(f"Portfolio profit evaluation error: {e}")
            return {'type': 'PORTFOLIO', 'action': 'HOLD', 'reason': 'Evaluation error'}
    
    def execute_portfolio_management(self, action_vector: np.ndarray, 
                                   market_data: Dict, mt5_interface) -> Dict:
        """Execute comprehensive portfolio management with 15-dimensional actions"""
        try:
            print(f"📊 === EXECUTING PORTFOLIO MANAGEMENT ===")
            
            # Parse 15-dimensional action vector
            parsed_actions = self._parse_portfolio_actions(action_vector)
            
            # Update portfolio status
            self.update_portfolio_status(mt5_interface)
            
            # Get current positions
            positions = mt5_interface.get_positions()
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(positions, market_data)
            
            # Execute portfolio decisions
            execution_result = {
                'success': False,
                'portfolio_actions': [],
                'position_actions': [],
                'risk_actions': [],
                'new_positions': [],
                'modified_positions': [],
                'closed_positions': [],
                'portfolio_metrics': portfolio_metrics,
                'execution_summary': {}
            }
            
            # 1. Portfolio-level profit taking
            profit_decision = self.evaluate_portfolio_profit_taking(positions, portfolio_metrics.get('total_pnl', 0))
            if profit_decision['action'] != 'HOLD':
                profit_result = self._execute_profit_decision(profit_decision, positions, mt5_interface)
                execution_result['portfolio_actions'].append(profit_result)
            
            # 2. Risk management actions
            risk_decision = self.evaluate_portfolio_risk(positions, portfolio_metrics)
            if risk_decision['action'] != 'HOLD':
                risk_result = self._execute_risk_decision(risk_decision, positions, mt5_interface)
                execution_result['risk_actions'].append(risk_result)
            
            # 3. Position-level management
            for position in positions:
                pos_decision = self.evaluate_position_management(position, market_data)
                if pos_decision['action'] != 'HOLD':
                    pos_result = self._execute_position_decision(pos_decision, position, mt5_interface)
                    execution_result['position_actions'].append(pos_result)
            
            # 4. New position opportunities (if conditions allow)
            if self._can_open_new_positions(parsed_actions, portfolio_metrics):
                new_pos_result = self._evaluate_new_positions(parsed_actions, market_data, mt5_interface)
                if new_pos_result['recommended']:
                    execution_result['new_positions'].extend(new_pos_result['positions'])
            
            # 5. Portfolio rebalancing
            if parsed_actions.get('rebalance_trigger', 0) > 0.7:
                rebalance_result = self._execute_portfolio_rebalancing(positions, market_data, mt5_interface)
                execution_result['portfolio_actions'].append(rebalance_result)
            
            # Update execution summary
            execution_result['execution_summary'] = {
                'total_actions': len(execution_result['portfolio_actions']) + len(execution_result['position_actions']),
                'portfolio_heat': portfolio_metrics.get('portfolio_heat', 0),
                'risk_level': self._calculate_risk_level(portfolio_metrics),
                'efficiency_score': self._calculate_efficiency_score(execution_result),
                'timestamp': datetime.now().isoformat()
            }
            
            execution_result['success'] = True
            
            print(f"✅ Portfolio management executed successfully")
            print(f"   Total Actions: {execution_result['execution_summary']['total_actions']}")
            print(f"   Portfolio Heat: {portfolio_metrics.get('portfolio_heat', 0):.1f}%")
            
            return execution_result
            
        except Exception as e:
            print(f"❌ Portfolio management execution error: {e}")
            return {
                'success': False,
                'error': str(e),
                'portfolio_actions': [],
                'position_actions': []
            }
    
    def _parse_portfolio_actions(self, action_vector):
        """Parse 15-dimensional action vector for portfolio management"""
        try:
            actions = {}
            
            # Map action vector to portfolio parameters
            actions['market_direction'] = np.clip(action_vector[0], -1, 1)
            actions['position_size'] = np.clip(action_vector[1], 0.01, 1.0)
            actions['entry_aggression'] = np.clip(action_vector[2], 0, 1)
            actions['profit_target_ratio'] = np.clip(action_vector[3], 0.5, 5.0)
            actions['partial_take_levels'] = int(np.clip(action_vector[4], 0, 3))
            actions['add_position_signal'] = np.clip(action_vector[5], 0, 1)
            actions['hedge_ratio'] = np.clip(action_vector[6], 0, 1)
            actions['recovery_mode'] = int(np.clip(action_vector[7], 0, 3))
            actions['correlation_limit'] = np.clip(action_vector[8], 0, 1)
            actions['volatility_filter'] = np.clip(action_vector[9], 0, 1)
            actions['spread_tolerance'] = np.clip(action_vector[10], 0, 1)
            actions['time_filter'] = np.clip(action_vector[11], 0, 1)
            actions['portfolio_heat_limit'] = np.clip(action_vector[12], 0, 1)
            actions['smart_exit_signal'] = np.clip(action_vector[13], 0, 1)
            actions['rebalance_trigger'] = np.clip(action_vector[14], 0, 1)
            
            return actions
            
        except Exception as e:
            print(f"Action parsing error: {e}")
            return {}
    
    def _execute_profit_decision(self, profit_decision, positions, mt5_interface):
        """Execute profit-taking decision"""
        try:
            result = {
                'action': profit_decision['action'],
                'positions_affected': 0,
                'total_profit_realized': 0.0,
                'execution_details': []
            }
            
            if profit_decision['action'] == 'CLOSE_ALL':
                for position in positions:
                    close_result = mt5_interface.close_position(position['ticket'])
                    if close_result['success']:
                        result['positions_affected'] += 1
                        result['total_profit_realized'] += position.get('profit', 0)
                        result['execution_details'].append({
                            'ticket': position['ticket'],
                            'profit': position.get('profit', 0),
                            'action': 'CLOSED'
                        })
            
            elif profit_decision['action'] == 'CLOSE_PROFITABLE':
                for position in positions:
                    if position.get('profit', 0) > 0:
                        close_result = mt5_interface.close_position(position['ticket'])
                        if close_result['success']:
                            result['positions_affected'] += 1
                            result['total_profit_realized'] += position.get('profit', 0)
                            result['execution_details'].append({
                                'ticket': position['ticket'],
                                'profit': position.get('profit', 0),
                                'action': 'CLOSED'
                            })
            
            return result
            
        except Exception as e:
            print(f"Profit decision execution error: {e}")
            return {'action': 'ERROR', 'error': str(e)}
    
    def _execute_risk_decision(self, risk_decision, positions, mt5_interface):
        """Execute risk management decision"""
        try:
            result = {
                'action': risk_decision['action'],
                'positions_affected': 0,
                'risk_reduction': 0.0,
                'execution_details': []
            }
            
            if risk_decision['action'] == 'EMERGENCY_CLOSE':
                # Close all positions immediately
                for position in positions:
                    close_result = mt5_interface.close_position(position['ticket'])
                    if close_result['success']:
                        result['positions_affected'] += 1
                        result['execution_details'].append({
                            'ticket': position['ticket'],
                            'action': 'EMERGENCY_CLOSED'
                        })
            
            elif risk_decision['action'] == 'REDUCE_EXPOSURE':
                # Close largest losing positions
                losing_positions = [pos for pos in positions if pos.get('profit', 0) < 0]
                losing_positions.sort(key=lambda x: x.get('profit', 0))  # Sort by largest loss
                
                positions_to_close = losing_positions[:len(losing_positions)//2]  # Close half
                
                for position in positions_to_close:
                    close_result = mt5_interface.close_position(position['ticket'])
                    if close_result['success']:
                        result['positions_affected'] += 1
                        result['risk_reduction'] += abs(position.get('profit', 0))
                        result['execution_details'].append({
                            'ticket': position['ticket'],
                            'profit': position.get('profit', 0),
                            'action': 'RISK_REDUCED'
                        })
            
            return result
            
        except Exception as e:
            print(f"Risk decision execution error: {e}")
            return {'action': 'ERROR', 'error': str(e)}
    
    def _execute_position_decision(self, pos_decision, position, mt5_interface):
        """Execute position-level decision"""
        try:
            result = {
                'ticket': position['ticket'],
                'action': pos_decision['action'],
                'success': False,
                'details': {}
            }
            
            if pos_decision['action'] == 'CLOSE':
                close_result = mt5_interface.close_position(position['ticket'])
                result['success'] = close_result['success']
                result['details'] = close_result
            
            elif pos_decision['action'] == 'MODIFY_SL':
                new_sl = pos_decision.get('new_sl', 0)
                modify_result = mt5_interface.modify_position(position['ticket'], sl=new_sl)
                result['success'] = modify_result['success']
                result['details'] = modify_result
            
            elif pos_decision['action'] == 'MODIFY_TP':
                new_tp = pos_decision.get('new_tp', 0)
                modify_result = mt5_interface.modify_position(position['ticket'], tp=new_tp)
                result['success'] = modify_result['success']
                result['details'] = modify_result
            
            elif pos_decision['action'] == 'PARTIAL_CLOSE':
                partial_volume = pos_decision.get('partial_volume', position['volume'] * 0.5)
                partial_result = mt5_interface.partial_close_position(position['ticket'], partial_volume)
                result['success'] = partial_result['success']
                result['details'] = partial_result
            
            return result
            
        except Exception as e:
            print(f"Position decision execution error: {e}")
            return {'action': 'ERROR', 'error': str(e)}
    
    def _can_open_new_positions(self, parsed_actions, portfolio_metrics):
        """Check if new positions can be opened"""
        try:
            # Check portfolio heat
            if portfolio_metrics.get('portfolio_heat', 0) > 15.0:
                return False
            
            # Check drawdown
            if portfolio_metrics.get('current_drawdown', 0) > 8.0:
                return False
            
            # Check position count
            if portfolio_metrics.get('position_count', 0) >= 10:
                return False
            
            # Check action signals
            if parsed_actions.get('add_position_signal', 0) < 0.5:
                return False
            
            return True
            
        except:
            return False
    
    def _evaluate_new_positions(self, parsed_actions, market_data, mt5_interface):
        """Evaluate new position opportunities"""
        try:
            result = {
                'recommended': False,
                'positions': [],
                'analysis': {}
            }
            
            # Get current market conditions
            current_price = market_data.get('current_price', 0)
            spread = market_data.get('spread', 0)
            volatility = market_data.get('atr', 0)
            
            # Check spread tolerance
            if spread > parsed_actions.get('spread_tolerance', 0.5) * 50:  # Max 50 points spread
                return result
            
            # Check volatility filter
            if volatility < parsed_actions.get('volatility_filter', 0.3) * 10:  # Min volatility
                return result
            
            # Calculate position size
            base_lot_size = 0.01
            size_multiplier = parsed_actions.get('position_size', 0.1)
            lot_size = base_lot_size * size_multiplier * 10  # Max 0.1 lot
            
            # Determine direction
            market_direction = parsed_actions.get('market_direction', 0)
            
            if market_direction > 0.2:  # Buy signal
                result['positions'].append({
                    'type': 'BUY',
                    'volume': lot_size,
                    'price': current_price,
                    'sl': current_price - volatility * 2,
                    'tp': current_price + volatility * parsed_actions.get('profit_target_ratio', 2.0),
                    'comment': 'Portfolio_Manager_Buy'
                })
                result['recommended'] = True
            
            elif market_direction < -0.2:  # Sell signal
                result['positions'].append({
                    'type': 'SELL',
                    'volume': lot_size,
                    'price': current_price,
                    'sl': current_price + volatility * 2,
                    'tp': current_price - volatility * parsed_actions.get('profit_target_ratio', 2.0),
                    'comment': 'Portfolio_Manager_Sell'
                })
                result['recommended'] = True
            
            return result
            
        except Exception as e:
            print(f"New position evaluation error: {e}")
            return {'recommended': False, 'positions': []}
    
    def _execute_portfolio_rebalancing(self, positions, market_data, mt5_interface):
        """Execute portfolio rebalancing"""
        try:
            result = {
                'action': 'REBALANCE',
                'positions_modified': 0,
                'rebalance_details': []
            }
            
            if not positions:
                return result
            
            # Calculate total exposure
            total_buy_volume = sum(pos.get('volume', 0) for pos in positions if pos.get('type', 0) == 0)
            total_sell_volume = sum(pos.get('volume', 0) for pos in positions if pos.get('type', 0) == 1)
            
            # Check if rebalancing is needed
            volume_imbalance = abs(total_buy_volume - total_sell_volume)
            total_volume = total_buy_volume + total_sell_volume
            
            if total_volume > 0:
                imbalance_ratio = volume_imbalance / total_volume
                
                if imbalance_ratio > 0.3:  # 30% imbalance threshold
                    # Rebalance by closing some positions from the dominant side
                    if total_buy_volume > total_sell_volume:
                        # Close some buy positions
                        buy_positions = [pos for pos in positions if pos.get('type', 0) == 0]
                        buy_positions.sort(key=lambda x: x.get('profit', 0), reverse=True)  # Close most profitable first
                        
                        positions_to_close = buy_positions[:len(buy_positions)//3]  # Close 1/3
                        
                        for position in positions_to_close:
                            close_result = mt5_interface.close_position(position['ticket'])
                            if close_result['success']:
                                result['positions_modified'] += 1
                                result['rebalance_details'].append({
                                    'ticket': position['ticket'],
                                    'action': 'CLOSED_FOR_REBALANCE',
                                    'type': 'BUY'
                                })
                    else:
                        # Close some sell positions
                        sell_positions = [pos for pos in positions if pos.get('type', 0) == 1]
                        sell_positions.sort(key=lambda x: x.get('profit', 0), reverse=True)  # Close most profitable first
                        
                        positions_to_close = sell_positions[:len(sell_positions)//3]  # Close 1/3
                        
                        for position in positions_to_close:
                            close_result = mt5_interface.close_position(position['ticket'])
                            if close_result['success']:
                                result['positions_modified'] += 1
                                result['rebalance_details'].append({
                                    'ticket': position['ticket'],
                                    'action': 'CLOSED_FOR_REBALANCE',
                                    'type': 'SELL'
                                })
            
            return result
            
        except Exception as e:
            print(f"Portfolio rebalancing error: {e}")
            return {'action': 'ERROR', 'error': str(e)}
    
    def _calculate_risk_level(self, portfolio_metrics):
        """Calculate overall portfolio risk level"""
        try:
            risk_factors = []
            
            # Portfolio heat risk
            portfolio_heat = portfolio_metrics.get('portfolio_heat', 0)
            heat_risk = min(portfolio_heat / 20.0, 1.0)  # Max at 20%
            risk_factors.append(heat_risk)
            
            # Drawdown risk
            drawdown = portfolio_metrics.get('current_drawdown', 0)
            drawdown_risk = min(drawdown / 10.0, 1.0)  # Max at 10%
            risk_factors.append(drawdown_risk)
            
            # Position count risk
            position_count = portfolio_metrics.get('position_count', 0)
            count_risk = min(position_count / 15.0, 1.0)  # Max at 15 positions
            risk_factors.append(count_risk)
            
            # Correlation risk
            correlation = portfolio_metrics.get('position_correlation', 0)
            correlation_risk = correlation  # Already 0-1
            risk_factors.append(correlation_risk)
            
            # Calculate weighted average
            weights = [0.3, 0.3, 0.2, 0.2]  # Portfolio heat and drawdown most important
            overall_risk = sum(risk * weight for risk, weight in zip(risk_factors, weights))
            
            return min(overall_risk, 1.0)
            
        except:
            return 0.5  # Default moderate risk
    
    def _calculate_efficiency_score(self, execution_result):
        """Calculate portfolio management efficiency score"""
        try:
            total_actions = execution_result['execution_summary']['total_actions']
            successful_actions = sum(1 for action in execution_result['portfolio_actions'] 
                                   if action.get('success', False))
            
            if total_actions == 0:
                return 1.0
            
            efficiency = successful_actions / total_actions
            return efficiency
            
        except:
            return 0.5
    
    def generate_portfolio_report(self) -> str:
        """Generate comprehensive portfolio management report"""
        try:
            report = "="*80 + "\n"
            report += "📊 PROFESSIONAL PORTFOLIO MANAGEMENT REPORT\n"
            report += "="*80 + "\n\n"
            
            # Portfolio Status
            report += f"📈 PORTFOLIO STATUS:\n"
            report += f"   Current Mode: {self.current_mode.value}\n"
            report += f"   Portfolio Heat: {self.portfolio_heat:.1f}%\n"
            report += f"   Current Drawdown: {self.current_drawdown:.1f}%\n"
            report += f"   Peak Equity: ${self.peak_equity:.2f}\n"
            report += f"   Position Count: {self.position_count}\n"
            report += f"   Total Volume: {self.total_volume:.2f} lots\n"
            report += f"   Position Correlation: {self.position_correlation:.2f}\n"
            
            # Dynamic Thresholds
            report += f"\n🎯 DYNAMIC THRESHOLDS:\n"
            for key, value in self.dynamic_thresholds.items():
                report += f"   {key.replace('_', ' ').title()}: ${value:.2f}\n"
            
            # Market Regime Analysis
            report += f"\n🌍 MARKET ANALYSIS:\n"
            report += f"   Current Regime: {self.market_regime.value}\n"
            report += f"   Trend Strength: {self.trend_strength:.2f}\n"
            report += f"   Volatility Level: {self.volatility_regime.value}\n"
            
            # Performance Metrics
            report += f"\n📊 PERFORMANCE METRICS:\n"
            report += f"   Win Rate: {self.win_rate:.1%}\n"
            report += f"   Avg Trade Duration: {self.avg_trade_duration:.1f} hours\n"
            report += f"   Risk-Adjusted Return: {self.risk_adjusted_return:.2f}\n"
            report += f"   Sharpe Ratio: {self.sharpe_ratio:.2f}\n"
            report += f"   Max Favorable Excursion: {self.max_favorable_excursion:.1f}%\n"
            
            # Mode Performance
            report += f"\n🎛️ MODE PERFORMANCE:\n"
            for mode, performance in self.mode_performance.items():
                report += f"   {mode}: {performance.get('win_rate', 0):.1%} win rate, "
                report += f"Avg PnL: ${performance.get('avg_pnl', 0):.2f}\n"
            
            # Recent Actions
            report += f"\n⚡ RECENT ACTIONS:\n"
            recent_actions = list(self.action_history)[-5:]  # Last 5 actions
            for i, action in enumerate(recent_actions, 1):
                timestamp = action.get('timestamp', 'Unknown')
                action_type = action.get('action', 'Unknown')
                report += f"   {i}. [{timestamp}]: {action_type}\n"
            
            # Risk Management
            report += f"\n🛡️ RISK MANAGEMENT:\n"
            report += f"   Portfolio Heat Limit: {self.portfolio_heat_limit:.1f}%\n"
            report += f"   Max Drawdown Limit: {self.max_drawdown_limit:.1f}%\n"
            report += f"   Position Size Limit: {self.position_size_limit:.2f} lots\n"
            report += f"   Correlation Limit: {self.correlation_limit:.2f}\n"
            
            # System Configuration
            report += f"\n⚙️ SYSTEM CONFIGURATION:\n"
            report += f"   Auto Position Management: {'✅' if self.auto_position_management else '❌'}\n"
            report += f"   Dynamic Sizing: {'✅' if self.dynamic_sizing else '❌'}\n"
            report += f"   Advanced Analytics: {'✅' if self.advanced_analytics else '❌'}\n"
            report += f"   ML Profit Optimization: {'✅' if self.ml_profit_optimization else '❌'}\n"
            report += f"   Multi-Timeframe Analysis: {'✅' if self.multi_timeframe_analysis else '❌'}\n"
            
            report += "\n" + "="*80
            
            return report
            
        except Exception as e:
            return f"Portfolio report generation error: {e}"
    
    def save_portfolio_state(self, filepath: str = None):
        """Save current portfolio state to file"""
        try:
            if filepath is None:
                filepath = f"portfolio_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            portfolio_state = {
                'timestamp': datetime.now().isoformat(),
                'current_mode': self.current_mode.value,
                'portfolio_heat': self.portfolio_heat,
                'current_drawdown': self.current_drawdown,
                'peak_equity': self.peak_equity,
                'position_count': self.position_count,
                'total_volume': self.total_volume,
                'position_correlation': self.position_correlation,
                'dynamic_thresholds': self.dynamic_thresholds,
                'market_regime': self.market_regime.value,
                'trend_strength': self.trend_strength,
                'volatility_regime': self.volatility_regime.value,
                'performance_metrics': {
                    'win_rate': self.win_rate,
                    'avg_trade_duration': self.avg_trade_duration,
                    'risk_adjusted_return': self.risk_adjusted_return,
                    'sharpe_ratio': self.sharpe_ratio,
                    'max_favorable_excursion': self.max_favorable_excursion
                },
                'mode_performance': self.mode_performance,
                'action_history': list(self.action_history)
            }
            
            with open(filepath, 'w') as f:
                json.dump(portfolio_state, f, indent=2)
            
            print(f"✅ Portfolio state saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving portfolio state: {e}")
            return False
    
    def load_portfolio_state(self, filepath: str):
        """Load portfolio state from file"""
        try:
            with open(filepath, 'r') as f:
                portfolio_state = json.load(f)
            
            # Restore state
            self.current_mode = PortfolioMode(portfolio_state.get('current_mode', 'BALANCED'))
            self.portfolio_heat = portfolio_state.get('portfolio_heat', 0.0)
            self.current_drawdown = portfolio_state.get('current_drawdown', 0.0)
            self.peak_equity = portfolio_state.get('peak_equity', 0.0)
            self.position_count = portfolio_state.get('position_count', 0)
            self.total_volume = portfolio_state.get('total_volume', 0.0)
            self.position_correlation = portfolio_state.get('position_correlation', 0.0)
            self.dynamic_thresholds = portfolio_state.get('dynamic_thresholds', {})
            self.market_regime = MarketRegime(portfolio_state.get('market_regime', 'SIDEWAYS'))
            self.trend_strength = portfolio_state.get('trend_strength', 0.0)
            self.volatility_regime = VolatilityRegime(portfolio_state.get('volatility_regime', 'NORMAL'))
            
            # Restore performance metrics
            performance_metrics = portfolio_state.get('performance_metrics', {})
            self.win_rate = performance_metrics.get('win_rate', 0.0)
            self.avg_trade_duration = performance_metrics.get('avg_trade_duration', 0.0)
            self.risk_adjusted_return = performance_metrics.get('risk_adjusted_return', 0.0)
            self.sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
            self.max_favorable_excursion = performance_metrics.get('max_favorable_excursion', 0.0)
            
            # Restore mode performance
            self.mode_performance = portfolio_state.get('mode_performance', {})
            
            # Restore action history
            action_history = portfolio_state.get('action_history', [])
            self.action_history = deque(action_history, maxlen=1000)
            
            print(f"✅ Portfolio state loaded from: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading portfolio state: {e}")
            return False


# ========================= FACTORY FUNCTIONS & EXPORTS =========================

def create_ai_portfolio_manager(config=None):
    """Factory function to create AI portfolio manager"""
    return AIPortfolioManager(config)

# Export main classes
__all__ = [
    'AIPortfolioManager',
    'PortfolioMode',
    'MarketRegime', 
    'VolatilityRegime',
    'create_ai_portfolio_manager'
]
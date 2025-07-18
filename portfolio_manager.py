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
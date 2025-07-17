import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os

class AIPortfolioManager:
    """
    AI-Powered Portfolio Management System
    - Dynamic position sizing based on account equity
    - Drawdown protection and recovery
    - Risk correlation analysis
    - Portfolio heat monitoring
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Portfolio parameters
        self.max_risk_per_trade = self.config.get('max_risk_per_trade', 2.0)  # 2% per trade
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 10.0)  # 10% total
        self.max_daily_loss = self.config.get('max_daily_loss', 5.0)  # 5% daily loss limit
        self.max_drawdown = self.config.get('max_drawdown', 15.0)  # 15% max drawdown
        
        # Portfolio state
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.peak_balance = 0.0
        self.daily_start_balance = 0.0
        self.current_drawdown = 0.0
        self.portfolio_heat = 0.0
        
        # Risk tracking
        self.active_positions = {}
        self.daily_pnl = 0.0
        self.risk_per_position = {}
        self.correlation_matrix = {}
        
        # AI decision states
        self.trading_allowed = True
        self.risk_reduction_active = False
        self.recovery_mode = False
        
        # Performance tracking
        self.portfolio_history = []
        self.risk_events = []
        
    def initialize_portfolio(self, mt5_interface):
        """Initialize portfolio with current account state"""
        try:
            account_info = mt5_interface.get_account_info()
            if not account_info:
                return False
                
            self.initial_balance = account_info.get('balance', 0)
            self.current_balance = account_info.get('equity', 0)
            self.peak_balance = self.current_balance
            self.daily_start_balance = self.current_balance
            
            print(f"💼 Portfolio Initialized:")
            print(f"   Initial Balance: ${self.initial_balance:,.2f}")
            print(f"   Current Equity: ${self.current_balance:,.2f}")
            
            return True
            
        except Exception as e:
            print(f"Portfolio initialization error: {e}")
            return False
            
    def calculate_position_size(self, symbol, entry_price, stop_loss_price, mt5_interface):
        """Calculate optimal position size based on portfolio risk"""
        try:
            # Get current account info
            account_info = mt5_interface.get_account_info()
            if not account_info:
                return 0.01
                
            current_equity = account_info.get('equity', 0)
            
            # Calculate risk amount per trade
            risk_amount = current_equity * (self.max_risk_per_trade / 100)
            
            # Calculate stop loss distance in points
            stop_distance = abs(entry_price - stop_loss_price)
            
            if stop_distance <= 0:
                return 0.01
                
            # Calculate lot size based on risk
            # For XAUUSD: 1 lot = $1 per point
            lot_size = risk_amount / (stop_distance * 100000)
            
            # Apply portfolio heat limits
            current_heat = self.calculate_portfolio_heat(mt5_interface)
            if current_heat > self.max_portfolio_risk * 0.8:  # 80% of max
                lot_size *= 0.5  # Reduce size when approaching limits
                print(f"⚠️ High portfolio heat ({current_heat:.1f}%), reducing position size")
                
            # Apply drawdown adjustments
            if self.current_drawdown > 5.0:  # If in drawdown > 5%
                reduction_factor = 1 - (self.current_drawdown / 20.0)  # Reduce up to 75%
                lot_size *= max(reduction_factor, 0.25)
                print(f"📉 Drawdown adjustment: -{self.current_drawdown:.1f}%, size reduced")
                
            # Validate lot size
            lot_size = max(0.01, min(lot_size, 1.0))  # Between 0.01 and 1.0
            lot_size = round(lot_size, 2)
            
            print(f"📊 Position Size Calculation:")
            print(f"   Risk Amount: ${risk_amount:.2f}")
            print(f"   Stop Distance: {stop_distance:.3f}")
            print(f"   Calculated Size: {lot_size:.2f} lots")
            
            return lot_size
            
        except Exception as e:
            print(f"Position size calculation error: {e}")
            return 0.01
            
    def calculate_portfolio_heat(self, mt5_interface):
        """Calculate current portfolio risk exposure"""
        try:
            positions = mt5_interface.get_positions()
            if not positions:
                self.portfolio_heat = 0.0
                return 0.0
                
            account_info = mt5_interface.get_account_info()
            if not account_info:
                return 0.0
                
            current_equity = account_info.get('equity', 0)
            total_risk = 0.0
            
            for position in positions:
                # Estimate risk per position (simplified)
                volume = position.get('volume', 0)
                current_price = position.get('price_current', 0)
                
                # Assume 100 points stop loss for risk calculation
                estimated_risk = volume * 100  # $100 risk per 0.01 lot
                total_risk += estimated_risk
                
            self.portfolio_heat = (total_risk / current_equity) * 100 if current_equity > 0 else 0
            return self.portfolio_heat
            
        except Exception as e:
            print(f"Portfolio heat calculation error: {e}")
            return 0.0
            
    def update_portfolio_status(self, mt5_interface):
        """Update portfolio status and make risk decisions"""
        try:
            account_info = mt5_interface.get_account_info()
            if not account_info:
                return
                
            # Update balances
            self.current_balance = account_info.get('equity', 0)
            current_balance_actual = account_info.get('balance', 0)
            
            # Update peak balance
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
                
            # Calculate current drawdown
            if self.peak_balance > 0:
                self.current_drawdown = ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
            else:
                self.current_drawdown = 0.0
                
            # Calculate daily PnL
            self.daily_pnl = ((self.current_balance - self.daily_start_balance) / self.daily_start_balance) * 100 if self.daily_start_balance > 0 else 0
            
            # Update portfolio heat
            self.portfolio_heat = self.calculate_portfolio_heat(mt5_interface)
            
            # Make risk management decisions
            self.make_risk_decisions()
            
            # Log portfolio status
            self.log_portfolio_status()
            
            # Save portfolio history
            self.save_portfolio_snapshot()
            
        except Exception as e:
            print(f"Portfolio update error: {e}")
            
    def make_risk_decisions(self):
        """AI-powered risk management decisions"""
        try:
            # Check daily loss limit
            if self.daily_pnl < -self.max_daily_loss:
                self.trading_allowed = False
                self.risk_reduction_active = True
                self.log_risk_event("DAILY_LOSS_LIMIT", f"Daily loss: {self.daily_pnl:.2f}%")
                
            # Check drawdown limit
            elif self.current_drawdown > self.max_drawdown:
                self.trading_allowed = False
                self.recovery_mode = True
                self.log_risk_event("MAX_DRAWDOWN", f"Drawdown: {self.current_drawdown:.2f}%")
                
            # Check portfolio heat
            elif self.portfolio_heat > self.max_portfolio_risk:
                self.trading_allowed = False
                self.risk_reduction_active = True
                self.log_risk_event("PORTFOLIO_HEAT", f"Heat: {self.portfolio_heat:.2f}%")
                
            # Recovery conditions
            elif self.daily_pnl > -2.0 and self.current_drawdown < 5.0 and self.portfolio_heat < 5.0:
                self.trading_allowed = True
                self.risk_reduction_active = False
                self.recovery_mode = False
                
        except Exception as e:
            print(f"Risk decision error: {e}")
            
    def should_allow_trade(self, action_type, symbol):
        """Determine if trade should be allowed based on portfolio state"""
        try:
            # Basic trading allowed check
            if not self.trading_allowed:
                print(f"🚫 Trading blocked - Risk management active")
                return False
                
            # Check if opening new position
            if action_type in ['buy', 'sell']:
                # Check portfolio heat before new position
                if self.portfolio_heat > self.max_portfolio_risk * 0.9:
                    print(f"🚫 New position blocked - Portfolio heat too high: {self.portfolio_heat:.1f}%")
                    return False
                    
                # Check daily loss
                if self.daily_pnl < -self.max_daily_loss * 0.8:
                    print(f"🚫 New position blocked - Daily loss approaching limit: {self.daily_pnl:.2f}%")
                    return False
                    
            # Always allow closing positions
            elif action_type in ['close', 'close_all']:
                return True
                
            return True
            
        except Exception as e:
            print(f"Trade authorization error: {e}")
            return False
            
    def get_recovery_strategy(self):
        """Get AI-recommended recovery strategy"""
        try:
            strategy = {
                'action': 'hold',
                'max_position_size': 0.01,
                'risk_per_trade': 1.0,
                'recommended_actions': []
            }
            
            if self.recovery_mode:
                strategy['action'] = 'recovery'
                strategy['max_position_size'] = 0.01
                strategy['risk_per_trade'] = 0.5  # Reduce risk
                strategy['recommended_actions'] = [
                    'Close losing positions gradually',
                    'Wait for high-probability setups only',
                    'Use smaller position sizes',
                    'Focus on trend-following trades'
                ]
                
            elif self.risk_reduction_active:
                strategy['action'] = 'reduce_risk'
                strategy['max_position_size'] = 0.02
                strategy['risk_per_trade'] = 1.0
                strategy['recommended_actions'] = [
                    'Reduce position sizes',
                    'Take partial profits aggressively',
                    'Avoid counter-trend trades'
                ]
                
            elif self.current_drawdown < 2.0 and self.daily_pnl > 0:
                strategy['action'] = 'aggressive'
                strategy['max_position_size'] = 0.05
                strategy['risk_per_trade'] = 2.5
                strategy['recommended_actions'] = [
                    'Take advantage of winning streak',
                    'Increase position sizes moderately',
                    'Look for high-probability setups'
                ]
                
            return strategy
            
        except Exception as e:
            print(f"Recovery strategy error: {e}")
            return {'action': 'hold', 'max_position_size': 0.01}
            
    def log_portfolio_status(self):
        """Log current portfolio status"""
        try:
            print(f"\n💼 PORTFOLIO STATUS:")
            print(f"   Balance: ${self.current_balance:,.2f}")
            print(f"   Peak: ${self.peak_balance:,.2f}")
            print(f"   Drawdown: {self.current_drawdown:.2f}%")
            print(f"   Daily PnL: {self.daily_pnl:.2f}%")
            print(f"   Portfolio Heat: {self.portfolio_heat:.2f}%")
            print(f"   Trading Allowed: {'✅' if self.trading_allowed else '🚫'}")
            
            if self.recovery_mode:
                print(f"   🔄 Recovery Mode Active")
            elif self.risk_reduction_active:
                print(f"   ⚠️ Risk Reduction Active")
                
        except Exception as e:
            print(f"Portfolio logging error: {e}")
            
    def log_risk_event(self, event_type, description):
        """Log risk management events"""
        try:
            risk_event = {
                'timestamp': datetime.now().isoformat(),
                'type': event_type,
                'description': description,
                'portfolio_state': {
                    'balance': self.current_balance,
                    'drawdown': self.current_drawdown,
                    'daily_pnl': self.daily_pnl,
                    'portfolio_heat': self.portfolio_heat
                }
            }
            
            self.risk_events.append(risk_event)
            print(f"🚨 RISK EVENT: {event_type} - {description}")
            
        except Exception as e:
            print(f"Risk event logging error: {e}")
            
    def save_portfolio_snapshot(self):
        """Save portfolio snapshot for analysis"""
        try:
            snapshot = {
                'timestamp': datetime.now().isoformat(),
                'balance': self.current_balance,
                'peak_balance': self.peak_balance,
                'drawdown': self.current_drawdown,
                'daily_pnl': self.daily_pnl,
                'portfolio_heat': self.portfolio_heat,
                'trading_allowed': self.trading_allowed,
                'risk_reduction_active': self.risk_reduction_active,
                'recovery_mode': self.recovery_mode
            }
            
            self.portfolio_history.append(snapshot)
            
            # Keep only last 1000 snapshots
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
                
        except Exception as e:
            print(f"Portfolio snapshot error: {e}")
            
    def get_portfolio_performance(self):
        """Get comprehensive portfolio performance metrics"""
        try:
            if not self.portfolio_history:
                return {}
                
            total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100 if self.initial_balance > 0 else 0
            max_dd = max([snapshot['drawdown'] for snapshot in self.portfolio_history], default=0)
            
            daily_returns = []
            for i in range(1, len(self.portfolio_history)):
                prev_balance = self.portfolio_history[i-1]['balance']
                curr_balance = self.portfolio_history[i]['balance']
                daily_return = ((curr_balance - prev_balance) / prev_balance) * 100 if prev_balance > 0 else 0
                daily_returns.append(daily_return)
                
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) if len(daily_returns) > 1 and np.std(daily_returns) > 0 else 0
            
            performance = {
                'total_return': total_return,
                'max_drawdown': max_dd,
                'current_drawdown': self.current_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'daily_pnl': self.daily_pnl,
                'portfolio_heat': self.portfolio_heat,
                'total_risk_events': len(self.risk_events),
                'trading_allowed': self.trading_allowed
            }
            
            return performance
            
        except Exception as e:
            print(f"Performance calculation error: {e}")
            return {}
            
    def reset_daily_tracking(self):
        """Reset daily tracking (call at start of new trading day)"""
        try:
            self.daily_start_balance = self.current_balance
            self.daily_pnl = 0.0
            
            # Reset daily limits if not in major drawdown
            if self.current_drawdown < 10.0:
                self.trading_allowed = True
                self.risk_reduction_active = False
                
            print(f"📅 Daily tracking reset - Starting balance: ${self.daily_start_balance:,.2f}")
            
        except Exception as e:
            print(f"Daily reset error: {e}")
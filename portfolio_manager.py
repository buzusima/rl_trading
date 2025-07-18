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
    - Smart profit targets based on lot size and capital
    - Risk management adapted to account size
    - Capital efficiency optimization
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Dynamic Portfolio Parameters
        self.base_risk_per_trade = self.config.get('base_risk_per_trade', 1.0)  # 1% per trade
        self.max_portfolio_risk = self.config.get('max_portfolio_risk', 5.0)   # 5% total
        self.max_daily_loss = self.config.get('max_daily_loss', 3.0)           # 3% daily loss limit
        self.max_drawdown = self.config.get('max_drawdown', 10.0)              # 10% max drawdown
        
        # Capital Efficiency Parameters
        self.profit_per_lot_target = self.config.get('profit_per_lot_target', 5.0)    # $5 per 0.01 lot
        self.min_profit_ratio = self.config.get('min_profit_ratio', 0.5)              # 0.5% of capital
        self.lot_scaling_factor = self.config.get('lot_scaling_factor', 1000)         # $1000 = 1 lot scaling
        
        # Portfolio state
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.peak_balance = 0.0
        self.daily_start_balance = 0.0
        self.current_drawdown = 0.0
        self.portfolio_heat = 0.0
        
        # Dynamic thresholds
        self.dynamic_profit_targets = {}
        self.position_risk_limits = {}
        
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
            
            # Calculate dynamic thresholds based on capital
            self._calculate_dynamic_thresholds()
            
            print(f"💼 Portfolio Initialized:")
            print(f"   Capital: ${self.current_balance:,.2f}")
            print(f"   Min Profit per 0.01 lot: ${self.dynamic_profit_targets['per_lot']:.2f}")
            print(f"   Portfolio Profit Target: ${self.dynamic_profit_targets['portfolio']:.2f}")
            print(f"   Max Loss Limit: ${self.dynamic_profit_targets['max_loss']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"Portfolio initialization error: {e}")
            return False
    
    def _calculate_dynamic_thresholds(self):
        """คำนวณ threshold แบบ dynamic ตามทุน"""
        try:
            # Base calculations
            capital = max(self.current_balance, 1000)  # Minimum $1000 for calculations
            
            # 🎯 DYNAMIC PROFIT TARGETS
            # Profit per lot = ($5 base * capital scaling)
            capital_scaling = min(capital / 1000, 5.0)  # Scale up to 5x for large accounts
            profit_per_lot = self.profit_per_lot_target * capital_scaling
            
            # Portfolio profit target = 0.5-2% of capital
            portfolio_profit_pct = max(0.5, min(2.0, capital / 50000))  # 0.5% for small, 2% for large
            portfolio_profit = capital * (portfolio_profit_pct / 100)
            
            # Max loss = 1-3% of capital  
            max_loss_pct = max(1.0, min(3.0, capital / 100000))  # 1% for small, 3% for large
            max_loss = capital * (max_loss_pct / 100)
            
            self.dynamic_profit_targets = {
                'per_lot': profit_per_lot,
                'portfolio': portfolio_profit,
                'max_loss': max_loss,
                'capital': capital,
                'profit_per_lot_pct': (profit_per_lot / capital) * 100,
                'portfolio_profit_pct': portfolio_profit_pct,
                'max_loss_pct': max_loss_pct
            }
            
            print(f"🎯 Dynamic Thresholds Updated:")
            print(f"   Capital: ${capital:,.2f}")
            print(f"   Profit/lot: ${profit_per_lot:.2f} ({(profit_per_lot/capital)*100:.3f}%)")
            print(f"   Portfolio target: ${portfolio_profit:.2f} ({portfolio_profit_pct:.2f}%)")
            print(f"   Max loss: ${max_loss:.2f} ({max_loss_pct:.2f}%)")
            
        except Exception as e:
            print(f"Error calculating dynamic thresholds: {e}")
            # Fallback values
            self.dynamic_profit_targets = {
                'per_lot': 5.0,
                'portfolio': 25.0,
                'max_loss': 30.0,
                'capital': 1000.0
            }
            
    def calculate_position_size(self, symbol, entry_price, stop_loss_price, mt5_interface):
        """Calculate optimal position size based on portfolio risk and capital efficiency"""
        try:
            # Get current account info
            account_info = mt5_interface.get_account_info()
            if not account_info:
                return 0.01
                
            current_equity = account_info.get('equity', 0)
            
            # Update dynamic thresholds
            self.current_balance = current_equity
            self._calculate_dynamic_thresholds()
            
            # 🎯 CAPITAL EFFICIENT LOT SIZING
            
            # Method 1: Based on capital scaling
            capital_based_lot = self._calculate_capital_based_lot_size(current_equity)
            
            # Method 2: Based on risk percentage
            risk_based_lot = self._calculate_risk_based_lot_size(current_equity, entry_price, stop_loss_price)
            
            # Method 3: Based on profit efficiency
            efficiency_based_lot = self._calculate_efficiency_based_lot_size(current_equity)
            
            # Choose the most conservative approach
            recommended_lot = min(capital_based_lot, risk_based_lot, efficiency_based_lot)
            
            # Apply portfolio heat limits
            current_heat = self.calculate_portfolio_heat(mt5_interface)
            if current_heat > self.max_portfolio_risk * 0.8:  # 80% of max
                recommended_lot *= 0.5  # Reduce size when approaching limits
                print(f"⚠️ High portfolio heat ({current_heat:.1f}%), reducing position size")
                
            # Apply drawdown adjustments
            if self.current_drawdown > 3.0:  # If in drawdown > 3%
                reduction_factor = 1 - (self.current_drawdown / 15.0)  # Reduce up to 80%
                recommended_lot *= max(reduction_factor, 0.2)
                print(f"📉 Drawdown adjustment: -{self.current_drawdown:.1f}%, size reduced")
            
            # Final validation
            final_lot = max(0.01, min(recommended_lot, 0.10))  # Between 0.01 and 0.10
            final_lot = round(final_lot, 2)
            
            print(f"📊 Position Size Calculation:")
            print(f"   Capital: ${current_equity:,.2f}")
            print(f"   Capital-based: {capital_based_lot:.3f}")
            print(f"   Risk-based: {risk_based_lot:.3f}")  
            print(f"   Efficiency-based: {efficiency_based_lot:.3f}")
            print(f"   Final size: {final_lot:.2f} lots")
            print(f"   Expected profit target: ${self.dynamic_profit_targets['per_lot'] * (final_lot/0.01):.2f}")
            
            return final_lot
            
        except Exception as e:
            print(f"Position size calculation error: {e}")
            return 0.01
    
    def _calculate_capital_based_lot_size(self, capital):
        """คำนวณขนาด lot ตามทุน"""
        try:
            # Base: $1000 = 0.01 lot, scale proportionally
            base_capital = 1000
            base_lot = 0.01
            
            # Scale lot size with capital (but not linearly)
            capital_ratio = capital / base_capital
            lot_multiplier = min(np.sqrt(capital_ratio), 5.0)  # Square root scaling, max 5x
            
            calculated_lot = base_lot * lot_multiplier
            return min(calculated_lot, 0.10)  # Cap at 0.10 lot
            
        except:
            return 0.01
    
    def _calculate_risk_based_lot_size(self, capital, entry_price, stop_loss_price):
        """คำนวณขนาด lot ตาม risk management"""
        try:
            if not stop_loss_price or stop_loss_price <= 0:
                stop_loss_price = entry_price * 0.995  # Default 0.5% stop loss
                
            # Risk amount (1% of capital)
            risk_amount = capital * (self.base_risk_per_trade / 100)
            
            # Stop loss distance in points
            stop_distance = abs(entry_price - stop_loss_price)
            
            if stop_distance <= 0:
                return 0.01
                
            # For XAUUSD: 1 lot = $1 per point movement
            # 0.01 lot = $0.01 per point
            risk_per_point_per_lot = 100  # $100 per point per 1 lot
            risk_per_point_001_lot = 1    # $1 per point per 0.01 lot
            
            # Calculate lot size based on risk
            lot_size = risk_amount / (stop_distance * risk_per_point_per_lot)
            
            return min(lot_size, 0.05)  # Cap at 0.05 lot for risk management
            
        except Exception as e:
            print(f"Risk-based lot calculation error: {e}")
            return 0.01
    
    def _calculate_efficiency_based_lot_size(self, capital):
        """คำนวณขนาด lot ตามประสิทธิภาพการใช้ทุน"""
        try:
            # Target: Make profit target achievable with reasonable price movement
            target_profit = self.dynamic_profit_targets['per_lot']
            
            # XAUUSD typically moves 10-50 points per trade
            expected_movement = 20  # Conservative 20 points
            
            # Calculate lot size needed to achieve target profit with expected movement
            # $5 profit with 20 points movement = need $0.25 per point
            # 0.01 lot = $1 per point, so need 0.0025 lot
            required_lot = target_profit / expected_movement
            
            # Scale based on capital availability
            max_affordable_lot = (capital * 0.02) / 1000  # 2% of capital max exposure
            
            efficient_lot = min(required_lot, max_affordable_lot)
            return max(0.01, efficient_lot)  # Minimum 0.01 lot
            
        except Exception as e:
            print(f"Efficiency-based lot calculation error: {e}")
            return 0.01
    
    def should_take_individual_profit(self, position, current_price=None):
        """ตัดสินใจเก็บกำไรรายไม้ตาม lot size"""
        try:
            profit = position.get('profit', 0)
            volume = position.get('volume', 0.01)
            
            # คำนวณกำไรเป้าหมายสำหรับไม้นี้
            target_profit = self.dynamic_profit_targets['per_lot'] * (volume / 0.01)
            
            # ปรับตาม market conditions
            profit_buffer = target_profit * 0.1  # 10% buffer
            
            decision = {
                'should_close': False,
                'reason': '',
                'target_profit': target_profit,
                'current_profit': profit,
                'profit_ratio': (profit / target_profit) if target_profit > 0 else 0
            }
            
            if profit >= target_profit:
                decision['should_close'] = True
                decision['reason'] = f'Target reached: ${profit:.2f} >= ${target_profit:.2f}'
                
            elif profit >= (target_profit - profit_buffer) and profit > 0:
                decision['should_close'] = True
                decision['reason'] = f'Near target with profit: ${profit:.2f}'
                
            elif profit < -(target_profit * 2):  # Loss > 2x target profit
                decision['should_close'] = False  # Never cut loss, try to recover
                decision['reason'] = f'Large loss: ${profit:.2f} - Hold for recovery'
                
            else:
                decision['should_close'] = False
                decision['reason'] = f'Hold: ${profit:.2f} / ${target_profit:.2f} target'
            
            return decision
            
        except Exception as e:
            print(f"Individual profit decision error: {e}")
            return {'should_close': False, 'reason': 'Error in calculation'}
    
    def should_take_portfolio_profit(self, positions):
        """ตัดสินใจเก็บกำไร portfolio โดยรวม"""
        try:
            if not positions:
                return {'should_close_all': False, 'reason': 'No positions'}
                
            total_pnl = sum(pos.get('profit', 0) for pos in positions)
            total_volume = sum(pos.get('volume', 0) for pos in positions)
            
            # Portfolio target
            portfolio_target = self.dynamic_profit_targets['portfolio']
            
            # Efficiency check
            efficiency_target = self.dynamic_profit_targets['per_lot'] * (total_volume / 0.01) * 0.8  # 80% of individual targets
            
            decision = {
                'should_close_all': False,
                'should_close_profitable': False,
                'reason': '',
                'total_pnl': total_pnl,
                'portfolio_target': portfolio_target,
                'efficiency_target': efficiency_target
            }
            
            if total_pnl >= portfolio_target:
                decision['should_close_all'] = True
                decision['reason'] = f'Portfolio target reached: ${total_pnl:.2f} >= ${portfolio_target:.2f}'
                
            elif total_pnl >= efficiency_target and total_pnl > 0:
                decision['should_close_profitable'] = True
                decision['reason'] = f'Efficiency target reached: ${total_pnl:.2f} >= ${efficiency_target:.2f}'
                
            elif total_pnl < -self.dynamic_profit_targets['max_loss']:
                decision['should_close_all'] = False  # No cut loss, activate recovery
                decision['reason'] = f'Max loss reached: ${total_pnl:.2f} - Activate recovery'
                
            else:
                decision['reason'] = f'Hold portfolio: ${total_pnl:.2f} / ${portfolio_target:.2f} target'
            
            return decision
            
        except Exception as e:
            print(f"Portfolio profit decision error: {e}")
            return {'should_close_all': False, 'reason': 'Error in calculation'}
    
    def get_current_thresholds(self):
        """ได้ threshold ปัจจุบัน"""
        return self.dynamic_profit_targets.copy()
    
    def update_capital(self, new_balance):
        """อัพเดททุนและคำนวณ threshold ใหม่"""
        try:
            old_balance = self.current_balance
            self.current_balance = new_balance
            
            # Recalculate if significant change (>10%)
            if abs(new_balance - old_balance) / max(old_balance, 1) > 0.1:
                self._calculate_dynamic_thresholds()
                print(f"💰 Capital updated: ${old_balance:,.2f} → ${new_balance:,.2f}")
                print(f"   New targets: Profit/lot=${self.dynamic_profit_targets['per_lot']:.2f}, Portfolio=${self.dynamic_profit_targets['portfolio']:.2f}")
                
        except Exception as e:
            print(f"Capital update error: {e}")
            
    def calculate_portfolio_heat(self, mt5_interface):
        """Calculate current portfolio risk exposure with dynamic targets"""
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
                volume = position.get('volume', 0)
                # Use dynamic target as risk estimate
                estimated_risk = volume * (self.dynamic_profit_targets['per_lot'] / 0.01)
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
            
            # Update dynamic thresholds
            self._calculate_dynamic_thresholds()
            
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
        """AI-powered risk management decisions with dynamic thresholds"""
        try:
            # Check daily loss limit (dynamic)
            daily_loss_limit = self.dynamic_profit_targets.get('max_loss_pct', 3.0)
            if self.daily_pnl < -daily_loss_limit:
                self.trading_allowed = False
                self.risk_reduction_active = True
                self.log_risk_event("DAILY_LOSS_LIMIT", f"Daily loss: {self.daily_pnl:.2f}% > {daily_loss_limit:.1f}%")
                
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
                
            # Recovery conditions (dynamic)
            elif self.daily_pnl > -(daily_loss_limit * 0.5) and self.current_drawdown < 5.0 and self.portfolio_heat < 3.0:
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
                if self.portfolio_heat > self.max_portfolio_risk * 0.8:
                    print(f"🚫 New position blocked - Portfolio heat too high: {self.portfolio_heat:.1f}%")
                    return False
                    
                # Check daily loss (dynamic)
                daily_loss_limit = self.dynamic_profit_targets.get('max_loss_pct', 3.0)
                if self.daily_pnl < -(daily_loss_limit * 0.8):
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
        """Get AI-recommended recovery strategy with dynamic parameters"""
        try:
            # Get current thresholds
            thresholds = self.get_current_thresholds()
            
            strategy = {
                'action': 'hold',
                'max_position_size': 0.01,
                'risk_per_trade': 1.0,
                'recommended_actions': [],
                'profit_target': thresholds.get('per_lot', 5.0),
                'portfolio_target': thresholds.get('portfolio', 25.0)
            }
            
            if self.recovery_mode:
                strategy['action'] = 'recovery'
                strategy['max_position_size'] = 0.01
                strategy['risk_per_trade'] = 0.5  # Reduce risk
                strategy['recommended_actions'] = [
                    f'Target individual profit: ${thresholds.get("per_lot", 5.0):.2f} per 0.01 lot',
                    f'Portfolio target: ${thresholds.get("portfolio", 25.0):.2f}',
                    'Use smaller position sizes',
                    'Focus on high-probability setups'
                ]
                
            elif self.risk_reduction_active:
                strategy['action'] = 'reduce_risk'
                strategy['max_position_size'] = 0.02
                strategy['risk_per_trade'] = 1.0
                strategy['recommended_actions'] = [
                    'Reduce position sizes',
                    'Take profits at lower targets',
                    'Avoid risky trades'
                ]
                
            elif self.current_drawdown < 2.0 and self.daily_pnl > 0:
                strategy['action'] = 'aggressive'
                strategy['max_position_size'] = 0.05
                strategy['risk_per_trade'] = 2.0
                strategy['recommended_actions'] = [
                    'Take advantage of winning streak',
                    'Use optimal position sizes',
                    'Target higher profits'
                ]
                
            return strategy
            
        except Exception as e:
            print(f"Recovery strategy error: {e}")
            return {'action': 'hold', 'max_position_size': 0.01}
            
    def get_position_efficiency_report(self, positions):
        """รายงานประสิทธิภาพการใช้ทุน"""
        try:
            if not positions:
                return "No positions to analyze"
                
            report = "\n" + "="*50 + "\n"
            report += "💰 CAPITAL EFFICIENCY REPORT\n"
            report += "="*50 + "\n"
            
            total_pnl = 0
            total_volume = 0
            
            for i, pos in enumerate(positions):
                profit = pos.get('profit', 0)
                volume = pos.get('volume', 0.01)
                ticket = pos.get('ticket', i)
                pos_type = 'BUY' if pos.get('type', 0) == 0 else 'SELL'
                
                target_profit = self.dynamic_profit_targets['per_lot'] * (volume / 0.01)
                efficiency = (profit / target_profit * 100) if target_profit > 0 else 0
                
                total_pnl += profit
                total_volume += volume
                
                status = "🟢" if profit >= target_profit else "🟡" if profit > 0 else "🔴"
                
                report += f"{status} Pos {ticket}: {pos_type} {volume:.2f} | "
                report += f"PnL: ${profit:+6.2f} | Target: ${target_profit:5.2f} | "
                report += f"Efficiency: {efficiency:+6.1f}%\n"
            
            # Portfolio summary
            portfolio_target = self.dynamic_profit_targets['portfolio']
            portfolio_efficiency = (total_pnl / portfolio_target * 100) if portfolio_target > 0 else 0
            
            report += "-"*50 + "\n"
            report += f"📊 PORTFOLIO: {len(positions)} positions, {total_volume:.2f} lots\n"
            report += f"💰 Total PnL: ${total_pnl:+8.2f} / ${portfolio_target:6.2f} target\n"
            report += f"⚡ Efficiency: {portfolio_efficiency:+6.1f}%\n"
            report += f"💡 Capital: ${self.current_balance:,.2f}\n"
            report += "="*50
            
            return report
            
        except Exception as e:
            return f"Report error: {e}"
            
    def log_portfolio_status(self):
        """Log current portfolio status with dynamic info"""
        try:
            thresholds = self.get_current_thresholds()
            
            print(f"\n💼 PORTFOLIO STATUS:")
            print(f"   Capital: ${self.current_balance:,.2f}")
            print(f"   Peak: ${self.peak_balance:,.2f}")
            print(f"   Drawdown: {self.current_drawdown:.2f}%")
            print(f"   Daily PnL: {self.daily_pnl:.2f}%")
            print(f"   Portfolio Heat: {self.portfolio_heat:.2f}%")
            print(f"   Trading Allowed: {'✅' if self.trading_allowed else '🚫'}")
            print(f"   Profit/lot target: ${thresholds.get('per_lot', 5.0):.2f}")
            print(f"   Portfolio target: ${thresholds.get('portfolio', 25.0):.2f}")
            
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
                    'portfolio_heat': self.portfolio_heat,
                    'dynamic_targets': self.dynamic_profit_targets
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
                'recovery_mode': self.recovery_mode,
                'dynamic_targets': self.dynamic_profit_targets.copy()
            }
            
            self.portfolio_history.append(snapshot)
            
            # Keep only last 1000 snapshots
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
                
        except Exception as e:
            print(f"Portfolio snapshot error: {e}")
            
    def get_portfolio_performance(self):
        """Get comprehensive portfolio performance metrics with dynamic analysis"""
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
            
            # Calculate capital efficiency
            current_targets = self.get_current_thresholds()
            capital_efficiency = (total_return / (current_targets.get('portfolio_profit_pct', 1.0))) if current_targets.get('portfolio_profit_pct', 1.0) > 0 else 0
            
            performance = {
                'total_return': total_return,
                'max_drawdown': max_dd,
                'current_drawdown': self.current_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'daily_pnl': self.daily_pnl,
                'portfolio_heat': self.portfolio_heat,
                'total_risk_events': len(self.risk_events),
                'trading_allowed': self.trading_allowed,
                'capital_efficiency': capital_efficiency,
                'current_targets': current_targets
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
            
            # Recalculate dynamic thresholds for new day
            self._calculate_dynamic_thresholds()
            
            # Reset daily limits if not in major drawdown
            if self.current_drawdown < 8.0:  # Changed to 8% from 10%
                self.trading_allowed = True
                self.risk_reduction_active = False
                
            print(f"📅 Daily tracking reset - Starting balance: ${self.daily_start_balance:,.2f}")
            print(f"🎯 Today's targets: Profit/lot=${self.dynamic_profit_targets['per_lot']:.2f}, Portfolio=${self.dynamic_profit_targets['portfolio']:.2f}")
            
        except Exception as e:
            print(f"Daily reset error: {e}")
            
    def get_capital_summary(self):
        """Get quick capital summary for AI decision making"""
        try:
            thresholds = self.get_current_thresholds()
            
            summary = {
                'capital': thresholds.get('capital', 1000),
                'profit_per_lot': thresholds.get('per_lot', 5.0),
                'portfolio_target': thresholds.get('portfolio', 25.0),
                'max_loss_limit': thresholds.get('max_loss', 30.0),
                'current_heat': self.portfolio_heat,
                'current_drawdown': self.current_drawdown,
                'daily_pnl': self.daily_pnl,
                'trading_allowed': self.trading_allowed,
                'risk_mode': 'recovery' if self.recovery_mode else 'reduction' if self.risk_reduction_active else 'normal'
            }
            
            return summary
            
        except Exception as e:
            print(f"Capital summary error: {e}")
            return {
                'capital': 1000,
                'profit_per_lot': 5.0,
                'portfolio_target': 25.0,
                'max_loss_limit': 30.0
            }
            
    def optimize_portfolio_allocation(self, positions):
        """Optimize portfolio allocation based on performance"""
        try:
            if not positions:
                return {'action': 'no_change', 'reason': 'No positions to optimize'}
                
            total_pnl = sum(pos.get('profit', 0) for pos in positions)
            thresholds = self.get_current_thresholds()
            
            optimization = {
                'action': 'hold',
                'reason': '',
                'recommendations': [],
                'close_positions': [],
                'modify_positions': []
            }
            
            # Analyze each position for optimization
            for i, pos in enumerate(positions):
                profit = pos.get('profit', 0)
                volume = pos.get('volume', 0.01)
                target_profit = thresholds['per_lot'] * (volume / 0.01)
                
                # Individual position optimization
                if profit >= target_profit * 1.2:  # 120% of target
                    optimization['close_positions'].append({
                        'ticket': pos.get('ticket'),
                        'reason': f'Exceeds target by 20%: ${profit:.2f} vs ${target_profit:.2f}'
                    })
                    
                elif profit <= -(target_profit * 1.5):  # Loss > 150% of target
                    optimization['modify_positions'].append({
                        'ticket': pos.get('ticket'),
                        'action': 'consider_hedge',
                        'reason': f'Large loss: ${profit:.2f}'
                    })
            
            # Portfolio-level optimization
            if total_pnl >= thresholds['portfolio'] * 0.9:  # 90% of portfolio target
                optimization['action'] = 'take_profits'
                optimization['reason'] = f'Near portfolio target: ${total_pnl:.2f} / ${thresholds["portfolio"]:.2f}'
                
            elif total_pnl <= -thresholds['max_loss'] * 0.8:  # 80% of max loss
                optimization['action'] = 'activate_recovery'
                optimization['reason'] = f'Approaching loss limit: ${total_pnl:.2f} / ${-thresholds["max_loss"]:.2f}'
                
            return optimization
            
        except Exception as e:
            print(f"Portfolio optimization error: {e}")
            return {'action': 'no_change', 'reason': f'Error: {e}'}
            
    def calculate_required_recovery(self, current_loss):
        """Calculate required recovery based on current loss and capital"""
        try:
            thresholds = self.get_current_thresholds()
            capital = thresholds['capital']
            
            # Calculate recovery requirements
            loss_percentage = (abs(current_loss) / capital) * 100
            
            recovery_info = {
                'current_loss': current_loss,
                'loss_percentage': loss_percentage,
                'capital': capital,
                'recovery_target': abs(current_loss),
                'recommended_lot_size': 0.01,
                'estimated_recovery_time': 0,
                'risk_level': 'low'
            }
            
            # Determine recovery strategy based on loss percentage
            if loss_percentage < 1.0:  # Less than 1% loss
                recovery_info['recommended_lot_size'] = 0.01
                recovery_info['risk_level'] = 'low'
                recovery_info['estimated_recovery_time'] = 2  # 2 trades
                
            elif loss_percentage < 2.0:  # 1-2% loss
                recovery_info['recommended_lot_size'] = 0.02
                recovery_info['risk_level'] = 'medium'
                recovery_info['estimated_recovery_time'] = 3
                
            elif loss_percentage < 5.0:  # 2-5% loss
                recovery_info['recommended_lot_size'] = 0.03
                recovery_info['risk_level'] = 'high'
                recovery_info['estimated_recovery_time'] = 5
                
            else:  # > 5% loss
                recovery_info['recommended_lot_size'] = 0.01  # Conservative
                recovery_info['risk_level'] = 'very_high'
                recovery_info['estimated_recovery_time'] = 10
                
            return recovery_info
            
        except Exception as e:
            print(f"Recovery calculation error: {e}")
            return {
                'current_loss': current_loss,
                'recovery_target': abs(current_loss),
                'recommended_lot_size': 0.01,
                'risk_level': 'unknown'
            }
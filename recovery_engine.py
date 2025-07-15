# recovery_engine.py - Recovery System Engine
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import os

class ProfitManager:
    """
    Smart profit taking system
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.profit_targets = []
        self.trailing_stop_active = False
        self.highest_profit = 0.0
        
        # Profit taking parameters
        self.min_profit_target = self.config.get('min_profit_target', 50)  # $50
        self.profit_ratio = self.config.get('profit_ratio', 2.0)  # Risk:Reward 1:2
        self.trailing_stop_distance = self.config.get('trailing_stop_distance', 30)  # $30
        self.partial_profit_levels = self.config.get('partial_profit_levels', [25, 50, 100])
        
    def calculate_profit_targets(self, positions, symbol_info):
        """
        Calculate dynamic profit targets based on positions
        """
        try:
            if not positions:
                return []
                
            targets = []
            
            for position in positions:
                entry_price = position.get('price_open', 0)
                volume = position.get('volume', 0)
                pos_type = position.get('type', 0)  # 0=buy, 1=sell
                
                if entry_price <= 0 or volume <= 0:
                    continue
                    
                # Calculate ATR-based target
                atr_target = self._calculate_atr_target(entry_price, symbol_info, pos_type)
                
                # Calculate support/resistance target
                sr_target = self._calculate_sr_target(entry_price, pos_type)
                
                # Use the more conservative target
                profit_target = min(atr_target, sr_target) if sr_target > 0 else atr_target
                
                targets.append({
                    'position_id': position.get('ticket', 0),
                    'entry_price': entry_price,
                    'target_price': profit_target,
                    'volume': volume,
                    'type': pos_type,
                    'profit_target': abs(profit_target - entry_price) * volume
                })
                
            return targets
            
        except Exception as e:
            print(f"Error calculating profit targets: {e}")
            return []
            
    def _calculate_atr_target(self, entry_price, symbol_info, pos_type):
        """
        Calculate profit target based on ATR
        """
        try:
            # Get ATR (simplified - in real implementation get from market data)
            estimated_atr = entry_price * 0.01  # 1% of price as ATR estimate
            
            # Calculate target based on risk-reward ratio
            if pos_type == 0:  # Buy position
                target = entry_price + (estimated_atr * self.profit_ratio)
            else:  # Sell position
                target = entry_price - (estimated_atr * self.profit_ratio)
                
            return target
            
        except Exception as e:
            print(f"Error calculating ATR target: {e}")
            return entry_price
            
    def _calculate_sr_target(self, entry_price, pos_type):
        """
        Calculate profit target based on support/resistance
        """
        try:
            # Simplified S/R calculation (in real implementation use technical analysis)
            price_range = entry_price * 0.02  # 2% range
            
            if pos_type == 0:  # Buy position - target at resistance
                target = entry_price + price_range
            else:  # Sell position - target at support
                target = entry_price - price_range
                
            return target
            
        except Exception as e:
            print(f"Error calculating S/R target: {e}")
            return 0
            
    def should_take_profit(self, positions, current_prices):
        """
        Determine if should take profit
        """
        try:
            profit_signals = []
            
            for position in positions:
                current_profit = position.get('profit', 0)
                position_id = position.get('ticket', 0)
                
                # Check minimum profit threshold
                if current_profit >= self.min_profit_target:
                    profit_signals.append({
                        'position_id': position_id,
                        'reason': 'minimum_target_reached',
                        'profit': current_profit,
                        'action': 'close_position'
                    })
                    
                # Check trailing stop
                elif self._check_trailing_stop(position, current_prices):
                    profit_signals.append({
                        'position_id': position_id,
                        'reason': 'trailing_stop_triggered',
                        'profit': current_profit,
                        'action': 'close_position'
                    })
                    
                # Check partial profit taking
                elif self._check_partial_profit(position):
                    profit_signals.append({
                        'position_id': position_id,
                        'reason': 'partial_profit',
                        'profit': current_profit,
                        'action': 'reduce_position'
                    })
                    
            return profit_signals
            
        except Exception as e:
            print(f"Error checking profit conditions: {e}")
            return []
            
    def _check_trailing_stop(self, position, current_prices):
        """
        Check trailing stop condition
        """
        try:
            current_profit = position.get('profit', 0)
            
            # Update highest profit
            if current_profit > self.highest_profit:
                self.highest_profit = current_profit
                self.trailing_stop_active = True
                return False
                
            # Check if profit dropped from peak
            if (self.trailing_stop_active and 
                current_profit < (self.highest_profit - self.trailing_stop_distance)):
                return True
                
            return False
            
        except Exception as e:
            print(f"Error checking trailing stop: {e}")
            return False
            
    def _check_partial_profit(self, position):
        """
        Check if should take partial profit
        """
        try:
            current_profit = position.get('profit', 0)
            
            # Check against partial profit levels
            for level in self.partial_profit_levels:
                if current_profit >= level and not self._already_taken_partial(position, level):
                    return True
                    
            return False
            
        except Exception as e:
            print(f"Error checking partial profit: {e}")
            return False
            
    def _already_taken_partial(self, position, level):
        """
        Check if partial profit already taken at this level
        """
        # This would track which levels have been taken
        # Simplified implementation
        return False
        
    def get_profit_summary(self, positions):
        """
        Get profit summary for all positions
        """
        try:
            summary = {
                'total_positions': len(positions),
                'total_unrealized_pnl': 0,
                'profitable_positions': 0,
                'losing_positions': 0,
                'largest_profit': 0,
                'largest_loss': 0,
                'profit_targets': []
            }
            
            for position in positions:
                profit = position.get('profit', 0)
                summary['total_unrealized_pnl'] += profit
                
                if profit > 0:
                    summary['profitable_positions'] += 1
                    summary['largest_profit'] = max(summary['largest_profit'], profit)
                else:
                    summary['losing_positions'] += 1
                    summary['largest_loss'] = min(summary['largest_loss'], profit)
                    
            return summary
            
        except Exception as e:
            print(f"Error getting profit summary: {e}")
            return {}

class RecoveryEngine:
    """Clean recovery system without syntax errors"""
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Recovery parameters
        self.martingale_multiplier = self.config.get('martingale_multiplier', 2.0)
        self.grid_spacing = self.config.get('grid_spacing', 100)
        self.max_recovery_levels = self.config.get('max_recovery_levels', 10)
        self.recovery_type = self.config.get('recovery_type', 'martingale')
        
        # Recovery state
        self.is_active = False
        self.current_level = 0
        self.recovery_start_time = None
        self.recovery_start_equity = 0.0
        self.recovery_positions = []
        self.total_recovery_volume = 0.0
        
        # Add profit manager
        self.profit_manager = ProfitManager(config)
        
        # Recovery history
        self.recovery_history = []
        self.recovery_attempts = 0
        self.successful_recoveries = 0
        
        # Grid strategy specific
        self.grid_levels = []
        self.grid_base_price = 0.0
        self.grid_direction = 0
        
        # Hedge strategy specific
        self.hedge_positions = []
        self.net_exposure = 0.0
        
        # Performance tracking
        self.total_recovery_time = 0.0
        self.max_drawdown_during_recovery = 0.0
        self.recovery_efficiency = 0.0
        
    def check_profit_opportunities(self, mt5_interface, symbol):
        """Check for profit taking opportunities"""
        try:
            positions = mt5_interface.get_positions()
            if not positions:
                return []
                
            current_prices = mt5_interface.get_current_price(symbol)
            profit_signals = self.profit_manager.should_take_profit(positions, current_prices)
            
            return profit_signals
            
        except Exception as e:
            print(f"Error checking profit opportunities: {e}")
            return []            
    def execute_profit_taking(self, mt5_interface, profit_signals):
        """
        Execute profit taking based on signals
        """
        try:
            executed_actions = []
            
            for signal in profit_signals:
                position_id = signal['position_id']
                action = signal['action']
                reason = signal['reason']
                profit = signal['profit']
                
                if action == 'close_position':
                    success = mt5_interface.close_position(position_id)
                    if success:
                        executed_actions.append({
                            'position_id': position_id,
                            'action': 'closed',
                            'profit': profit,
                            'reason': reason
                        })
                        self.log_recovery_event(f"Profit taken: ${profit:.2f} - {reason}")
                        
                elif action == 'reduce_position':
                    # Implement partial position closing
                    success = self._partial_close_position(mt5_interface, position_id, 0.5)
                    if success:
                        executed_actions.append({
                            'position_id': position_id,
                            'action': 'partial_close',
                            'profit': profit,
                            'reason': reason
                        })
                        self.log_recovery_event(f"Partial profit taken: ${profit:.2f} - {reason}")
                        
            return executed_actions
            
        except Exception as e:
            print(f"Error executing profit taking: {e}")
            return []
            
    def _partial_close_position(self, mt5_interface, position_id, close_ratio=0.5):
        """
        Close partial position
        """
        try:
            positions = mt5_interface.get_positions()
            target_position = None
            
            for pos in positions:
                if pos.get('ticket') == position_id:
                    target_position = pos
                    break
                    
            if not target_position:
                return False
                
            original_volume = target_position.get('volume', 0)
            close_volume = original_volume * close_ratio
            
            # Round to valid lot size
            close_volume = round(close_volume, 2)
            if close_volume < 0.01:
                close_volume = 0.01
                
            # Close partial volume by opening opposite position
            symbol = target_position.get('symbol', '')
            pos_type = target_position.get('type', 0)
            
            if pos_type == 0:  # Buy position, open sell to close
                success = mt5_interface.place_order(symbol, 'sell', close_volume)
            else:  # Sell position, open buy to close
                success = mt5_interface.place_order(symbol, 'buy', close_volume)
                
            return success
            
        except Exception as e:
            print(f"Error partial closing position: {e}")
            return False
            
    def smart_profit_strategy(self, mt5_interface, symbol):
        """
        Intelligent profit taking strategy
        """
        try:
            positions = mt5_interface.get_positions()
            if not positions:
                return False
                
            total_pnl = sum(pos.get('profit', 0) for pos in positions)
            
            # Strategy 1: Take profit if total PnL is good
            if total_pnl >= 100:  # $100 profit threshold
                self.log_recovery_event(f"Taking total profit: ${total_pnl:.2f}")
                return mt5_interface.close_all_positions(symbol)
                
            # Strategy 2: Take profit on individual good positions
            for pos in positions:
                pos_profit = pos.get('profit', 0)
                pos_volume = pos.get('volume', 0)
                
                # Take profit if position profit > $50 per 0.01 lot
                profit_per_lot = pos_profit / max(pos_volume, 0.01)
                
                if profit_per_lot >= 50:
                    success = mt5_interface.close_position(pos.get('ticket'))
                    if success:
                        self.log_recovery_event(f"Individual profit taken: ${pos_profit:.2f}")
                        
            # Strategy 3: Progressive profit taking during recovery
            if self.is_active and total_pnl > 0:
                # During recovery, be more aggressive in taking profits
                profit_threshold = 25 * self.current_level  # Increase threshold with recovery level
                
                if total_pnl >= profit_threshold:
                    self.log_recovery_event(f"Recovery profit taken: ${total_pnl:.2f} at level {self.current_level}")
                    success = mt5_interface.close_all_positions(symbol)
                    if success:
                        self.reset()  # Reset recovery state
                    return success
                    
            return False
            
        except Exception as e:
            print(f"Error in smart profit strategy: {e}")
            return False
        
        # Grid strategy specific
        self.grid_levels = []
        self.grid_base_price = 0.0
        self.grid_direction = 0  # 1 for up, -1 for down
        
        # Hedge strategy specific
        self.hedge_positions = []
        self.net_exposure = 0.0
        
        # Performance tracking
        self.total_recovery_time = 0.0
        self.max_drawdown_during_recovery = 0.0
        self.recovery_efficiency = 0.0
        
    def activate_recovery(self, symbol: str, mt5_interface, current_pnl: float, 
                         current_equity: float, recovery_type: str = None):
        """
        Activate recovery system when losses are detected
        """
        if self.is_active:
            return self.escalate_recovery(symbol, mt5_interface, current_pnl)
            
        self.is_active = True
        self.current_level = 1
        self.recovery_start_time = datetime.now()
        self.recovery_start_equity = current_equity
        self.recovery_attempts += 1
        
        recovery_type = recovery_type or self.recovery_type
        
        self.log_recovery_event(f"Recovery activated - Type: {recovery_type}, PnL: {current_pnl:.2f}")
        
        if recovery_type.lower() == 'martingale':
            return self.execute_martingale_recovery(symbol, mt5_interface, current_pnl)
        elif recovery_type.lower() == 'grid':
            return self.execute_grid_recovery(symbol, mt5_interface, current_pnl)
        elif recovery_type.lower() == 'hedge':
            return self.execute_hedge_recovery(symbol, mt5_interface, current_pnl)
        elif recovery_type.lower() == 'combined':
            return self.execute_combined_recovery(symbol, mt5_interface, current_pnl)
        else:
            return False
            
    def execute_martingale_recovery(self, symbol: str, mt5_interface, current_pnl: float):
        """
        Execute Martingale recovery strategy
        Double the lot size to recover losses quickly
        """
        try:
            positions = mt5_interface.get_positions()
            if not positions:
                return False
                
            # Calculate total losing volume
            losing_volume = 0.0
            losing_side = None
            
            for pos in positions:
                if pos.get('profit', 0) < 0:
                    losing_volume += pos.get('volume', 0)
                    losing_side = 'buy' if pos.get('type', 0) == 0 else 'sell'
                    
            if losing_volume == 0:
                return False
                
            # Calculate recovery volume using martingale multiplier
            recovery_volume = losing_volume * self.martingale_multiplier
            recovery_volume = self.validate_lot_size(recovery_volume)
            
            # Place recovery order in same direction
            current_price = mt5_interface.get_current_price(symbol)
            if not current_price:
                return False
                
            success = mt5_interface.place_order(
                symbol=symbol,
                order_type=losing_side,
                volume=recovery_volume,
                price=current_price.get('bid' if losing_side == 'sell' else 'ask')
            )
            
            if success:
                self.total_recovery_volume += recovery_volume
                self.log_recovery_event(
                    f"Martingale recovery executed - Level: {self.current_level}, "
                    f"Volume: {recovery_volume:.2f}, Side: {losing_side}"
                )
                return True
                
            return False
            
        except Exception as e:
            self.log_recovery_event(f"Martingale recovery error: {str(e)}")
            return False
            
    def execute_grid_recovery(self, symbol: str, mt5_interface, current_pnl: float):
        """
        Execute Grid recovery strategy
        Place multiple orders at different price levels
        """
        try:
            current_price = mt5_interface.get_current_price(symbol)
            if not current_price:
                return False
                
            mid_price = (current_price['bid'] + current_price['ask']) / 2
            
            # Initialize grid if first time
            if not self.grid_levels:
                self.grid_base_price = mid_price
                self.setup_grid_levels(symbol, mt5_interface)
                
            # Determine market direction
            positions = mt5_interface.get_positions()
            net_pnl = sum(pos.get('profit', 0) for pos in positions)
            
            if net_pnl < 0:
                # Place grid orders
                return self.place_grid_orders(symbol, mt5_interface, mid_price)
                
            return False
            
        except Exception as e:
            self.log_recovery_event(f"Grid recovery error: {str(e)}")
            return False
            
    def setup_grid_levels(self, symbol: str, mt5_interface):
        """
        Setup grid levels for recovery
        """
        try:
            # Get symbol info for pip calculation
            symbol_info = mt5_interface.get_symbol_info(symbol)
            if not symbol_info:
                return
                
            point = symbol_info.point
            pip_size = point * 10 if symbol.find('JPY') != -1 else point
            
            # Calculate grid spacing in price units
            grid_spacing_price = self.grid_spacing * pip_size
            
            # Create grid levels above and below current price
            for i in range(1, 6):  # 5 levels each side
                # Buy levels (below current price)
                buy_level = self.grid_base_price - (grid_spacing_price * i)
                self.grid_levels.append({
                    'price': buy_level,
                    'type': 'buy',
                    'volume': self.calculate_grid_volume(i),
                    'placed': False
                })
                
                # Sell levels (above current price)
                sell_level = self.grid_base_price + (grid_spacing_price * i)
                self.grid_levels.append({
                    'price': sell_level,
                    'type': 'sell',
                    'volume': self.calculate_grid_volume(i),
                    'placed': False
                })
                
        except Exception as e:
            self.log_recovery_event(f"Grid setup error: {str(e)}")
            
    def calculate_grid_volume(self, level: int):
        """
        Calculate volume for grid level
        Progressive increase for faster recovery
        """
        base_volume = self.config.get('initial_lot_size', 0.01)
        multiplier = 1 + (level * 0.5)  # Increase by 50% each level
        return self.validate_lot_size(base_volume * multiplier)
        
    def place_grid_orders(self, symbol: str, mt5_interface, current_price: float):
        """
        Place pending grid orders
        """
        orders_placed = 0
        
        try:
            for level in self.grid_levels:
                if level['placed']:
                    continue
                    
                # Check if price level is appropriate
                if level['type'] == 'buy' and current_price > level['price']:
                    # Place buy stop order
                    success = mt5_interface.place_pending_order(
                        symbol=symbol,
                        order_type='buy_stop',
                        volume=level['volume'],
                        price=level['price']
                    )
                    if success:
                        level['placed'] = True
                        orders_placed += 1
                        
                elif level['type'] == 'sell' and current_price < level['price']:
                    # Place sell stop order
                    success = mt5_interface.place_pending_order(
                        symbol=symbol,
                        order_type='sell_stop',
                        volume=level['volume'],
                        price=level['price']
                    )
                    if success:
                        level['placed'] = True
                        orders_placed += 1
                        
            if orders_placed > 0:
                self.log_recovery_event(f"Grid orders placed: {orders_placed}")
                return True
                
            return False
            
        except Exception as e:
            self.log_recovery_event(f"Grid order placement error: {str(e)}")
            return False
            
    def execute_hedge_recovery(self, symbol: str, mt5_interface, current_pnl: float):
        """
        Execute Hedge recovery strategy
        Open opposite positions to neutralize risk while waiting for recovery
        """
        try:
            positions = mt5_interface.get_positions()
            if not positions:
                return False
                
            # Calculate net position
            net_volume_buy = 0.0
            net_volume_sell = 0.0
            
            for pos in positions:
                if pos.get('type', 0) == 0:  # Buy position
                    net_volume_buy += pos.get('volume', 0)
                else:  # Sell position
                    net_volume_sell += pos.get('volume', 0)
                    
            net_exposure = net_volume_buy - net_volume_sell
            
            # Place hedge order
            current_price = mt5_interface.get_current_price(symbol)
            if not current_price:
                return False
                
            hedge_volume = abs(net_exposure)
            if hedge_volume < 0.01:
                return False
                
            if net_exposure > 0:  # Net long, hedge with sell
                hedge_type = 'sell'
                hedge_price = current_price['bid']
            else:  # Net short, hedge with buy
                hedge_type = 'buy'
                hedge_price = current_price['ask']
                
            success = mt5_interface.place_order(
                symbol=symbol,
                order_type=hedge_type,
                volume=hedge_volume,
                price=hedge_price
            )
            
            if success:
                self.hedge_positions.append({
                    'type': hedge_type,
                    'volume': hedge_volume,
                    'price': hedge_price,
                    'time': datetime.now()
                })
                
                self.log_recovery_event(
                    f"Hedge recovery executed - Volume: {hedge_volume:.2f}, "
                    f"Type: {hedge_type}, Price: {hedge_price:.5f}"
                )
                return True
                
            return False
            
        except Exception as e:
            self.log_recovery_event(f"Hedge recovery error: {str(e)}")
            return False
            
    def execute_combined_recovery(self, symbol: str, mt5_interface, current_pnl: float):
        """
        Execute combined recovery strategy
        Use multiple recovery methods based on market conditions
        """
        try:
            # Assess market conditions
            market_volatility = self.assess_market_volatility(symbol, mt5_interface)
            trend_strength = self.assess_trend_strength(symbol, mt5_interface)
            
            recovery_success = False
            
            # Use different strategies based on conditions
            if market_volatility > 0.7:  # High volatility - use hedge
                recovery_success = self.execute_hedge_recovery(symbol, mt5_interface, current_pnl)
                
            elif trend_strength > 0.6:  # Strong trend - use martingale
                recovery_success = self.execute_martingale_recovery(symbol, mt5_interface, current_pnl)
                
            else:  # Sideways market - use grid
                recovery_success = self.execute_grid_recovery(symbol, mt5_interface, current_pnl)
                
            # If primary strategy fails, try hedge as backup
            if not recovery_success and market_volatility < 0.7:
                recovery_success = self.execute_hedge_recovery(symbol, mt5_interface, current_pnl)
                
            return recovery_success
            
        except Exception as e:
            self.log_recovery_event(f"Combined recovery error: {str(e)}")
            return False
            
    def escalate_recovery(self, symbol: str, mt5_interface, current_pnl: float):
        """
        Escalate recovery when current level is not working
        """
        if self.current_level >= self.max_recovery_levels:
            self.log_recovery_event("Maximum recovery levels reached")
            return False
            
        self.current_level += 1
        
        # Increase recovery aggressiveness
        old_multiplier = self.martingale_multiplier
        self.martingale_multiplier = min(self.martingale_multiplier * 1.5, 5.0)
        
        self.log_recovery_event(
            f"Recovery escalated to level {self.current_level}, "
            f"Multiplier: {old_multiplier:.1f} -> {self.martingale_multiplier:.1f}"
        )
        
        # Try more aggressive recovery
        if self.recovery_type.lower() == 'martingale':
            return self.execute_martingale_recovery(symbol, mt5_interface, current_pnl)
        elif self.recovery_type.lower() == 'grid':
            return self.execute_grid_recovery(symbol, mt5_interface, current_pnl)
        else:
            return self.execute_combined_recovery(symbol, mt5_interface, current_pnl)
            
    def check_recovery_completion(self, mt5_interface):
        """
        Check if recovery is completed successfully
        """
        try:
            if not self.is_active:
                return False
                
            # Get current PnL
            positions = mt5_interface.get_positions()
            total_pnl = sum(pos.get('profit', 0) for pos in positions)
            
            # Recovery successful if total PnL is positive
            if total_pnl > 0:
                self.complete_recovery(mt5_interface, total_pnl)
                return True
                
            # Check for partial recovery (break-even)
            if abs(total_pnl) < 10:  # Close to break-even
                self.complete_recovery(mt5_interface, total_pnl)
                return True
                
            return False
            
        except Exception as e:
            self.log_recovery_event(f"Recovery check error: {str(e)}")
            return False
            
    def complete_recovery(self, mt5_interface, final_pnl: float):
        """
        Complete the recovery process
        """
        try:
            recovery_duration = (datetime.now() - self.recovery_start_time).total_seconds()
            
            # Record recovery statistics
            recovery_record = {
                'start_time': self.recovery_start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': recovery_duration,
                'recovery_level': self.current_level,
                'final_pnl': final_pnl,
                'recovery_type': self.recovery_type,
                'total_volume': self.total_recovery_volume,
                'success': final_pnl >= 0
            }
            
            self.recovery_history.append(recovery_record)
            
            if final_pnl >= 0:
                self.successful_recoveries += 1
                
            # Calculate performance metrics
            self.total_recovery_time += recovery_duration
            self.recovery_efficiency = self.successful_recoveries / max(self.recovery_attempts, 1)
            
            # Reset recovery state
            self.reset()
            
            self.log_recovery_event(
                f"Recovery completed - Duration: {recovery_duration:.0f}s, "
                f"Final PnL: {final_pnl:.2f}, Success: {final_pnl >= 0}"
            )
            
            # Save recovery history
            self.save_recovery_history()
            
        except Exception as e:
            self.log_recovery_event(f"Recovery completion error: {str(e)}")
            
    def assess_market_volatility(self, symbol: str, mt5_interface):
        """
        Assess current market volatility
        """
        try:
            # Get recent price data
            rates = mt5_interface.get_rates(symbol, mt5_interface.TIMEFRAME_M5, 20)
            if rates is None or len(rates) < 10:
                return 0.5  # Default moderate volatility
                
            # Calculate ATR
            highs = [rate[2] for rate in rates]
            lows = [rate[3] for rate in rates]
            closes = [rate[4] for rate in rates]
            
            tr_values = []
            for i in range(1, len(rates)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                tr_values.append(max(tr1, tr2, tr3))
                
            atr = np.mean(tr_values)
            current_price = closes[-1]
            
            # Normalize volatility (0-1 scale)
            volatility_ratio = atr / current_price
            normalized_volatility = min(volatility_ratio * 1000, 1.0)  # Scale for forex
            
            return normalized_volatility
            
        except Exception as e:
            self.log_recovery_event(f"Volatility assessment error: {str(e)}")
            return 0.5
            
    def assess_trend_strength(self, symbol: str, mt5_interface):
        """
        Assess current trend strength
        """
        try:
            # Get recent price data
            rates = mt5_interface.get_rates(symbol, mt5_interface.TIMEFRAME_M15, 50)
            if rates is None or len(rates) < 20:
                return 0.5  # Default moderate trend
                
            closes = [rate[4] for rate in rates]
            
            # Calculate moving averages
            sma_20 = np.mean(closes[-20:])
            sma_50 = np.mean(closes)
            
            # Calculate trend strength
            price_vs_sma20 = abs(closes[-1] - sma_20) / sma_20
            sma_divergence = abs(sma_20 - sma_50) / sma_50
            
            # Linear regression slope
            x = np.arange(len(closes))
            slope = np.polyfit(x, closes, 1)[0]
            slope_strength = abs(slope) / closes[-1]
            
            # Combine indicators
            trend_strength = (price_vs_sma20 + sma_divergence + slope_strength) / 3
            return min(trend_strength * 100, 1.0)  # Scale and cap at 1.0
            
        except Exception as e:
            self.log_recovery_event(f"Trend assessment error: {str(e)}")
            return 0.5
            
    def calculate_lot_size(self, base_lot: float, recovery_level: int = None):
        """
        Calculate lot size for recovery
        """
        level = recovery_level or self.current_level
        
        if self.recovery_type.lower() == 'martingale':
            return self.validate_lot_size(base_lot * (self.martingale_multiplier ** level))
        elif self.recovery_type.lower() == 'grid':
            return self.validate_lot_size(base_lot * (1 + level * 0.5))
        else:
            return self.validate_lot_size(base_lot * (1.5 ** level))
            
    def validate_lot_size(self, lot_size: float):
        """
        Validate and adjust lot size within acceptable limits
        """
        min_lot = 0.01
        max_lot = 10.0
        
        # Round to 2 decimal places
        lot_size = round(lot_size, 2)
        
        # Apply limits
        return max(min_lot, min(lot_size, max_lot))
        
    def get_status(self):
        """
        Get current recovery status
        """
        status = {
            'active': self.is_active,
            'level': self.current_level,
            'type': self.recovery_type if self.is_active else 'none',
            'total_recovery_attempts': self.recovery_attempts,
            'successful_recoveries': self.successful_recoveries,
            'success_rate': self.recovery_efficiency,
            'average_recovery_time': self.total_recovery_time / max(self.recovery_attempts, 1),
            'total_recovery_volume': self.total_recovery_volume
        }
        
        if self.is_active and self.recovery_start_time:
            current_duration = (datetime.now() - self.recovery_start_time).total_seconds()
            status['current_duration'] = current_duration
            
        return status
        
    def reset(self):
        """
        Reset recovery system to initial state
        """
        self.is_active = False
        self.current_level = 0
        self.recovery_start_time = None
        self.recovery_start_equity = 0.0
        self.recovery_positions = []
        self.total_recovery_volume = 0.0
        
        # Reset strategy-specific data
        self.grid_levels = []
        self.grid_base_price = 0.0
        self.grid_direction = 0
        self.hedge_positions = []
        self.net_exposure = 0.0
        
        # Reset multiplier to original
        self.martingale_multiplier = self.config.get('martingale_multiplier', 2.0)
        
    def save_recovery_history(self):
        """
        Save recovery history to file
        """
        try:
            os.makedirs('data', exist_ok=True)
            
            history_data = {
                'recovery_history': self.recovery_history,
                'total_attempts': self.recovery_attempts,
                'successful_recoveries': self.successful_recoveries,
                'recovery_efficiency': self.recovery_efficiency,
                'total_recovery_time': self.total_recovery_time
            }
            
            with open('data/recovery_history.json', 'w') as f:
                json.dump(history_data, f, indent=4, default=str)
                
        except Exception as e:
            self.log_recovery_event(f"Error saving recovery history: {str(e)}")
            
    def load_recovery_history(self):
        """
        Load recovery history from file
        """
        try:
            if os.path.exists('data/recovery_history.json'):
                with open('data/recovery_history.json', 'r') as f:
                    history_data = json.load(f)
                    
                self.recovery_history = history_data.get('recovery_history', [])
                self.recovery_attempts = history_data.get('total_attempts', 0)
                self.successful_recoveries = history_data.get('successful_recoveries', 0)
                self.recovery_efficiency = history_data.get('recovery_efficiency', 0.0)
                self.total_recovery_time = history_data.get('total_recovery_time', 0.0)
                
                self.log_recovery_event("Recovery history loaded successfully")
                
        except Exception as e:
            self.log_recovery_event(f"Error loading recovery history: {str(e)}")
            
    def get_recovery_statistics(self):
        """
        Get comprehensive recovery statistics
        """
        stats = {
            'total_attempts': self.recovery_attempts,
            'successful_recoveries': self.successful_recoveries,
            'success_rate': self.recovery_efficiency,
            'average_recovery_time': self.total_recovery_time / max(self.recovery_attempts, 1),
            'total_recovery_volume': self.total_recovery_volume,
            'current_status': self.get_status()
        }
        
        if self.recovery_history:
            # Calculate additional statistics
            recovery_times = [r['duration_seconds'] for r in self.recovery_history]
            final_pnls = [r['final_pnl'] for r in self.recovery_history]
            recovery_levels = [r['recovery_level'] for r in self.recovery_history]
            
            stats.update({
                'min_recovery_time': min(recovery_times),
                'max_recovery_time': max(recovery_times),
                'avg_final_pnl': np.mean(final_pnls),
                'avg_recovery_level': np.mean(recovery_levels),
                'max_recovery_level': max(recovery_levels)
            })
            
        return stats
        
    def log_recovery_event(self, message: str):
        """
        Log recovery events with timestamp
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] RECOVERY: {message}"
        print(log_message)
        
        # Also save to log file
        try:
            os.makedirs('data', exist_ok=True)
            with open('data/recovery_log.txt', 'a') as f:
                f.write(log_message + '\n')
        except:
            pass
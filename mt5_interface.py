# mt5_interface.py - MetaTrader 5 Interface
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import time
import json
import os

class MT5Interface:
    """
    Interface for MetaTrader 5 trading operations
    Handles connection, orders, positions, and market data
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.is_connected = False
        self.account_info = None
        self.symbol_info_cache = {}
        self.last_error = None
        
        # Trading parameters
        self.magic_number = self.config.get('magic_number', 12345)
        self.slippage = self.config.get('slippage', 3)
        self.default_timeout = self.config.get('order_timeout', 10000)  # milliseconds
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # seconds between requests
    
    def set_training_mode(self, is_training=True):
        """Set training mode to bypass unnecessary checks"""
        self.training_mode = is_training
        if is_training:
            self.min_request_interval = 0.01  # เร็วขึ้น
            print("MT5 Interface: Training mode enabled")
        else:
            self.min_request_interval = 0.1   # ปกติ
            print("MT5 Interface: Live mode enabled")    
    def connect(self, login: int = None, password: str = None, server: str = None):
        """
        Connect to MetaTrader 5
        """
        try:
            # Initialize MT5
            if not mt5.initialize():
                self.last_error = "Failed to initialize MT5"
                return False
                
            # Login if credentials provided
            if login and password and server:
                if not mt5.login(login, password, server):
                    self.last_error = f"Failed to login: {mt5.last_error()}"
                    return False
                    
            # Check connection
            account_info = mt5.account_info()
            if account_info is None:
                self.last_error = "Failed to get account info"
                return False
                
            self.account_info = account_info._asdict()
            self.is_connected = True
            
            print(f"Connected to MT5 - Account: {self.account_info.get('login', 'Unknown')}")
            print(f"Server: {self.account_info.get('server', 'Unknown')}")
            print(f"Balance: ${self.account_info.get('balance', 0):.2f}")
            
            return True
            
        except Exception as e:
            self.last_error = f"Connection error: {str(e)}"
            return False
            
    def disconnect(self):
        """
        Disconnect from MetaTrader 5
        """
        try:
            mt5.shutdown()
            self.is_connected = False
            print("Disconnected from MT5")
        except Exception as e:
            print(f"Disconnect error: {str(e)}")
            
    def _rate_limit(self):
        """
        Implement rate limiting for API requests
        """
        if hasattr(self, 'training_mode') and self.training_mode:
            return
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
            
        self.last_request_time = time.time()
        
    def get_account_info(self):
        """
        Get current account information
        """
        try:
            self._rate_limit()
            
            account_info = mt5.account_info()
            if account_info is None:
                return None
                
            self.account_info = account_info._asdict()
            return self.account_info
            
        except Exception as e:
            self.last_error = f"Error getting account info: {str(e)}"
            return None
            
    def get_symbol_info(self, symbol: str):
        """
        Get symbol information with caching
        """
        try:
            # Check cache first
            if symbol in self.symbol_info_cache:
                cache_time = self.symbol_info_cache[symbol].get('cache_time', 0)
                if time.time() - cache_time < 3600:  # Cache for 1 hour
                    return self.symbol_info_cache[symbol]['info']
                    
            self._rate_limit()
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                return None
                
            info_dict = symbol_info._asdict()
            
            # Cache the result
            self.symbol_info_cache[symbol] = {
                'info': info_dict,
                'cache_time': time.time()
            }
            
            return info_dict
            
        except Exception as e:
            self.last_error = f"Error getting symbol info: {str(e)}"
            return None
            
    def get_current_price(self, symbol: str):
        """
        Get current bid/ask prices for symbol
        """
        try:
            self._rate_limit()
            
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
                
            return {
                'bid': tick.bid,
                'ask': tick.ask,
                'time': tick.time,
                'spread': tick.ask - tick.bid
            }
            
        except Exception as e:
            self.last_error = f"Error getting current price: {str(e)}"
            return None
            
    def get_rates(self, symbol: str, timeframe: int, count: int):
        """
        Get historical rate data
        """
        try:
            self._rate_limit()
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                return None
                
            return rates
            
        except Exception as e:
            self.last_error = f"Error getting rates: {str(e)}"
            return None
            
    def get_positions(self, symbol: str = None):
        """
        Get current positions
        """
        try:
            self._rate_limit()
            
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
                
            if positions is None:
                return []
                
            return [pos._asdict() for pos in positions]
            
        except Exception as e:
            self.last_error = f"Error getting positions: {str(e)}"
            return []
            
    def get_orders(self, symbol: str = None):
        """
        Get pending orders
        """
        try:
            self._rate_limit()
            
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()
                
            if orders is None:
                return []
                
            return [order._asdict() for order in orders]
            
        except Exception as e:
            self.last_error = f"Error getting orders: {str(e)}"
            return []
            
    def get_symbol_filling_mode(self, symbol: str):
        """
        Get supported filling modes for symbol
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return mt5.ORDER_FILLING_IOC
                
            filling_mode = symbol_info.get('filling_mode', 0)
            
            # Check which filling modes are supported
            if filling_mode & 1:  # SYMBOL_FILLING_FOK
                return mt5.ORDER_FILLING_FOK
            elif filling_mode & 2:  # SYMBOL_FILLING_IOC  
                return mt5.ORDER_FILLING_IOC
            else:  # SYMBOL_FILLING_RETURN
                return mt5.ORDER_FILLING_RETURN
                
        except Exception as e:
            print(f"Error getting filling mode: {e}")
            return mt5.ORDER_FILLING_IOC
            
    def test_order_filling_modes(self, symbol: str):
        """
        Test which filling modes work with the broker
        """
        filling_modes = {
            'FOK': mt5.ORDER_FILLING_FOK,
            'IOC': mt5.ORDER_FILLING_IOC, 
            'RETURN': mt5.ORDER_FILLING_RETURN
        }
        
        supported_modes = []
        
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return ['IOC']  # Default fallback
                
            filling_mode = symbol_info.get('filling_mode', 0)
            
            if filling_mode & 1:  # FOK supported
                supported_modes.append('FOK')
            if filling_mode & 2:  # IOC supported  
                supported_modes.append('IOC')
            if filling_mode & 4:  # RETURN supported
                supported_modes.append('RETURN')
                
            print(f"Supported filling modes for {symbol}: {supported_modes}")
            
            return supported_modes if supported_modes else ['IOC']
            
        except Exception as e:
            print(f"Error testing filling modes: {e}")
            return ['IOC']
            
    def place_order_with_fallback(self, symbol: str, order_type: str, volume: float, 
                                 price: float = None, sl: float = None, tp: float = None,
                                 comment: str = "RL Trading", magic: int = None):
        """
        Place order with automatic filling mode fallback
        """
        # Get supported filling modes
        supported_modes = self.test_order_filling_modes(symbol)
        
        # Priority order for filling modes
        filling_priority = ['FOK', 'IOC', 'RETURN']
        
        # Try each supported mode in priority order
        for mode_name in filling_priority:
            if mode_name in supported_modes:
                if mode_name == 'FOK':
                    filling_mode = mt5.ORDER_FILLING_FOK
                elif mode_name == 'IOC':
                    filling_mode = mt5.ORDER_FILLING_IOC
                else:
                    filling_mode = mt5.ORDER_FILLING_RETURN
                    
                success = self._execute_order(symbol, order_type, volume, price, 
                                            sl, tp, comment, magic, filling_mode)
                
                if success:
                    print(f"Order placed successfully with {mode_name} filling mode")
                    return True
                else:
                    print(f"Order failed with {mode_name}, trying next mode...")
                    
        print("All filling modes failed")
        return False
    def _execute_order(self, symbol: str, order_type: str, volume: float, 
                      price: float = None, sl: float = None, tp: float = None,
                      comment: str = "RL Trading", magic: int = None, 
                      filling_mode: int = None):
        """
        Execute order with specified filling mode
        """
        try:
            self._rate_limit()
            
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return False
                
            # Validate volume
            min_volume = symbol_info.get('volume_min', 0.01)
            max_volume = symbol_info.get('volume_max', 100.0)
            volume_step = symbol_info.get('volume_step', 0.01)
            
            volume = max(min_volume, min(volume, max_volume))
            volume = round(volume / volume_step) * volume_step
            
            # Get current price if not provided
            if price is None:
                current_price = self.get_current_price(symbol)
                if not current_price:
                    return False
                price = current_price['ask'] if order_type.lower() == 'buy' else current_price['bid']
                
            # Determine order type
            if order_type.lower() == 'buy':
                mt5_order_type = mt5.ORDER_TYPE_BUY
            elif order_type.lower() == 'sell':
                mt5_order_type = mt5.ORDER_TYPE_SELL
            elif order_type.lower() == 'buy_limit':
                mt5_order_type = mt5.ORDER_TYPE_BUY_LIMIT
            elif order_type.lower() == 'sell_limit':
                mt5_order_type = mt5.ORDER_TYPE_SELL_LIMIT
            elif order_type.lower() == 'buy_stop':
                mt5_order_type = mt5.ORDER_TYPE_BUY_STOP
            elif order_type.lower() == 'sell_stop':
                mt5_order_type = mt5.ORDER_TYPE_SELL_STOP
            else:
                self.last_error = f"Invalid order type: {order_type}"
                return False
                
            # Use provided filling mode or get best one
            if filling_mode is None:
                filling_mode = self.get_symbol_filling_mode(symbol)
                
            # Prepare request
            request = {
                "action": mt5.TRADE_ACTION_DEAL if order_type.lower() in ['buy', 'sell'] else mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": price,
                "deviation": self.slippage,
                "magic": magic or self.magic_number,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            # Add SL/TP if provided
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
                
            # Send order
            result = mt5.order_send(request)
            
            if result is None:
                self.last_error = "Order send failed - no result"
                return False
                
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.last_error = f"Order failed: {result.retcode} - {result.comment}"
                
                # Log the filling mode that failed
                filling_name = "UNKNOWN"
                if filling_mode == mt5.ORDER_FILLING_FOK:
                    filling_name = "FOK"
                elif filling_mode == mt5.ORDER_FILLING_IOC:
                    filling_name = "IOC"
                elif filling_mode == mt5.ORDER_FILLING_RETURN:
                    filling_name = "RETURN"
                    
                print(f"Order failed with {filling_name} filling mode: {result.comment}")
                return False
                
            print(f"Order placed successfully - {order_type} {volume} {symbol} at {price}")
            return True
            
        except Exception as e:
            self.last_error = f"Error placing order: {str(e)}"
            return False
            
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: float = None, sl: float = None, tp: float = None,
                   comment: str = "RL Trading", magic: int = None):
        """
        Place order with automatic filling mode detection and fallback
        """
        return self.place_order_with_fallback(symbol, order_type, volume, 
                                            price, sl, tp, comment, magic)
            
    def place_pending_order(self, symbol: str, order_type: str, volume: float, 
                           price: float, sl: float = None, tp: float = None,
                           comment: str = "RL Trading Pending"):
        """
        Place pending order (limit/stop)
        """
        return self.place_order(symbol, order_type, volume, price, sl, tp, comment)
        
    def close_position(self, position_ticket: int):
        """
        Close specific position by ticket
        """
        try:
            self._rate_limit()
            
            # Get position info
            position = mt5.positions_get(ticket=position_ticket)
            if not position:
                return False
                
            pos = position[0]
            
            # Determine close order type
            if pos.type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                price = mt5.symbol_info_tick(pos.symbol).bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                price = mt5.symbol_info_tick(pos.symbol).ask
                
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": position_ticket,
                "price": price,
                "deviation": self.slippage,
                "magic": self.magic_number,
                "comment": "Close by RL Trading",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Position {position_ticket} closed successfully")
                return True
            else:
                self.last_error = f"Close failed: {result.retcode if result else 'No result'}"
                return False
                
        except Exception as e:
            self.last_error = f"Error closing position: {str(e)}"
            return False
            
    def close_all_positions(self, symbol: str = None):
        """
        Close all positions for symbol or all symbols
        """
        try:
            positions = self.get_positions(symbol)
            
            if not positions:
                return True
                
            success_count = 0
            total_count = len(positions)
            
            for position in positions:
                if self.close_position(position['ticket']):
                    success_count += 1
                    time.sleep(0.1)  # Small delay between closes
                    
            print(f"Closed {success_count}/{total_count} positions")
            return success_count == total_count
            
        except Exception as e:
            self.last_error = f"Error closing all positions: {str(e)}"
            return False
            
    def cancel_order(self, order_ticket: int):
        """
        Cancel pending order
        """
        try:
            self._rate_limit()
            
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order_ticket,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Order {order_ticket} cancelled successfully")
                return True
            else:
                self.last_error = f"Cancel failed: {result.retcode if result else 'No result'}"
                return False
                
        except Exception as e:
            self.last_error = f"Error cancelling order: {str(e)}"
            return False
            
    def cancel_all_orders(self, symbol: str = None):
        """
        Cancel all pending orders
        """
        try:
            orders = self.get_orders(symbol)
            
            if not orders:
                return True
                
            success_count = 0
            total_count = len(orders)
            
            for order in orders:
                if self.cancel_order(order['ticket']):
                    success_count += 1
                    time.sleep(0.1)
                    
            print(f"Cancelled {success_count}/{total_count} orders")
            return success_count == total_count
            
        except Exception as e:
            self.last_error = f"Error cancelling all orders: {str(e)}"
            return False
            
    def modify_position(self, position_ticket: int, sl: float = None, tp: float = None):
        """
        Modify position SL/TP
        """
        try:
            self._rate_limit()
            
            # Get position info
            position = mt5.positions_get(ticket=position_ticket)
            if not position:
                return False
                
            pos = position[0]
            
            # Prepare modification request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": position_ticket,
                "magic": self.magic_number,
            }
            
            if sl is not None:
                request["sl"] = sl
            if tp is not None:
                request["tp"] = tp
                
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"Position {position_ticket} modified successfully")
                return True
            else:
                self.last_error = f"Modify failed: {result.retcode if result else 'No result'}"
                return False
                
        except Exception as e:
            self.last_error = f"Error modifying position: {str(e)}"
            return False
            
    def get_market_data(self, symbol: str = "XAUUSD", timeframe: int = mt5.TIMEFRAME_M1, 
                       count: int = 100):
        """
        Get comprehensive market data for RL environment
        """
        try:
            # Get OHLCV data
            rates = self.get_rates(symbol, timeframe, count)
            if rates is None:
                return None
                
            # Get current tick data
            current_price = self.get_current_price(symbol)
            if not current_price:
                return None
                
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            
            # Get account info
            account_info = self.get_account_info()
            
            # Get current positions
            positions = self.get_positions(symbol)
            
            # Get pending orders
            orders = self.get_orders(symbol)
            
            return {
                'rates': rates,
                'current_price': current_price,
                'symbol_info': symbol_info,
                'account_info': account_info,
                'positions': positions,
                'orders': orders,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.last_error = f"Error getting market data: {str(e)}"
            return None
            
    def calculate_lot_size(self, symbol: str, risk_percent: float, 
                          stop_loss_pips: float, account_balance: float = None):
        """
        Calculate position size based on risk management
        """
        try:
            if account_balance is None:
                account_info = self.get_account_info()
                if not account_info:
                    return 0.01
                account_balance = account_info.get('balance', 10000)
                
            # Get symbol info
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return 0.01
                
            # Calculate risk amount
            risk_amount = account_balance * (risk_percent / 100)
            
            # Calculate pip value
            if 'JPY' in symbol:
                pip_value = symbol_info['trade_tick_size'] * 100
            else:
                pip_value = symbol_info['trade_tick_size'] * 10
                
            # Calculate lot size
            lot_size = risk_amount / (stop_loss_pips * pip_value)
            
            # Validate lot size
            min_volume = symbol_info.get('volume_min', 0.01)
            max_volume = symbol_info.get('volume_max', 100.0)
            volume_step = symbol_info.get('volume_step', 0.01)
            
            lot_size = max(min_volume, min(lot_size, max_volume))
            lot_size = round(lot_size / volume_step) * volume_step
            
            return lot_size
            
        except Exception as e:
            self.last_error = f"Error calculating lot size: {str(e)}"
            return 0.01
            
    def get_trading_hours(self, symbol: str):
        """
        Get trading hours for symbol
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None
                
            return {
                'trade_mode': symbol_info.get('trade_mode'),
                'start_time': symbol_info.get('trade_time_from'),
                'end_time': symbol_info.get('trade_time_to'),
                'session_deals': symbol_info.get('session_deals'),
                'session_buy_orders': symbol_info.get('session_buy_orders'),
                'session_sell_orders': symbol_info.get('session_sell_orders')
            }
            
        except Exception as e:
            self.last_error = f"Error getting trading hours: {str(e)}"
            return None
            
    def is_market_open(self, symbol: str):
        """
        Check if market is open for symbol
        """
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return False
                
            # Check if trading is allowed
            trade_mode = symbol_info.get('trade_mode', 0)
            
            # SYMBOL_TRADE_MODE_DISABLED = 0
            # SYMBOL_TRADE_MODE_LONGONLY = 1 
            # SYMBOL_TRADE_MODE_SHORTONLY = 2
            # SYMBOL_TRADE_MODE_CLOSEONLY = 3
            # SYMBOL_TRADE_MODE_FULL = 4
            
            return trade_mode in [1, 2, 4]  # Allow long only, short only, or full trading
            
        except Exception as e:
            self.last_error = f"Error checking market status: {str(e)}"
            return False
            
    def get_spread(self, symbol: str):
        """
        Get current spread for symbol
        """
        try:
            current_price = self.get_current_price(symbol)
            if not current_price:
                return None
                
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return None
                
            # Calculate spread in pips
            spread_points = current_price['spread']
            point = symbol_info.get('point', 0.00001)
            
            if 'JPY' in symbol:
                spread_pips = spread_points / (point * 100)
            else:
                spread_pips = spread_points / (point * 10)
                
            return {
                'spread_points': spread_points,
                'spread_pips': spread_pips,
                'bid': current_price['bid'],
                'ask': current_price['ask']
            }
            
        except Exception as e:
            self.last_error = f"Error getting spread: {str(e)}"
            return None
            
    def get_historical_data_df(self, symbol: str, timeframe: int, count: int):
        """
        Get historical data as pandas DataFrame
        """
        try:
            rates = self.get_rates(symbol, timeframe, count)
            if rates is None:
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            return df
            
        except Exception as e:
            self.last_error = f"Error getting historical data: {str(e)}"
            return None
            
    def test_connection(self):
        """
        Test MT5 connection
        """
        try:
            if not self.is_connected:
                return False
                
            # Try to get account info
            account_info = self.get_account_info()
            if not account_info:
                return False
                
            # Try to get a price quote
            current_price = self.get_current_price("XAUUSD")
            if not current_price:
                return False
                
            return True
            
        except Exception as e:
            self.last_error = f"Connection test failed: {str(e)}"
            return False
            
    def get_last_error(self):
        """
        Get last error message
        """
        return self.last_error
        
    def save_trading_history(self, filename: str = None):
        """
        Save trading history to file
        """
        try:
            if filename is None:
                filename = f"data/trading_history_{datetime.now().strftime('%Y%m%d')}.json"
                
            os.makedirs('data', exist_ok=True)
            
            # Get recent deals/history
            from_date = datetime.now() - timedelta(days=30)  # Last 30 days
            to_date = datetime.now()
            
            deals = mt5.history_deals_get(from_date, to_date)
            orders = mt5.history_orders_get(from_date, to_date)
            
            history_data = {
                'account_info': self.account_info,
                'deals': [deal._asdict() for deal in deals] if deals else [],
                'orders': [order._asdict() for order in orders] if orders else [],
                'export_time': datetime.now().isoformat()
            }
            
            with open(filename, 'w') as f:
                json.dump(history_data, f, indent=4, default=str)
                
            print(f"Trading history saved to {filename}")
            return True
            
        except Exception as e:
            self.last_error = f"Error saving trading history: {str(e)}"
            return False
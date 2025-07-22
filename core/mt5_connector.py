# core/mt5_connector.py - Complete MT5 Interface
import MetaTrader5 as mt5
import time
from typing import Dict, List, Optional
from datetime import datetime

class MT5Connector:
    """
    Complete MT5 Interface for Trading
    - Connection management
    - Trading operations
    - Market data
    - Order management
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.is_connected = False
        self.account_info = None
        self.last_error = None
        
        # Trading parameters
        self.magic_number = self.config.get('magic_number', 12345)
        self.slippage = self.config.get('slippage', 3)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # seconds between requests
        
        print("🔗 MT5Connector initialized")

    def connect(self, login: int = None, password: str = None, server: str = None):
        """Connect to MetaTrader 5"""
        try:
            # Initialize MT5
            if not mt5.initialize():
                self.last_error = "Failed to initialize MT5"
                print("❌ MT5 initialization failed")
                return False
                
            # Login if credentials provided
            if login and password and server:
                if not mt5.login(login, password, server):
                    self.last_error = f"Failed to login: {mt5.last_error()}"
                    print(f"❌ MT5 login failed: {self.last_error}")
                    return False
                    
            # Check connection
            account_info = mt5.account_info()
            if account_info is None:
                self.last_error = "Failed to get account info"
                print("❌ Cannot get account info")
                return False
                
            self.account_info = account_info._asdict()
            self.is_connected = True
            
            print(f"✅ Connected to MT5")
            print(f"   Account: {self.account_info.get('login', 'Unknown')}")
            print(f"   Server: {self.account_info.get('server', 'Unknown')}")
            print(f"   Balance: ${self.account_info.get('balance', 0):.2f}")
            
            return True
            
        except Exception as e:
            self.last_error = f"Connection error: {str(e)}"
            print(f"❌ MT5 connection error: {e}")
            return False

    def disconnect(self):
        """Disconnect from MetaTrader 5"""
        try:
            mt5.shutdown()
            self.is_connected = False
            print("✅ Disconnected from MT5")
        except Exception as e:
            print(f"❌ Disconnect error: {e}")

    def get_account_info(self):
        """Get current account information"""
        try:
            self._rate_limit()
            
            account_info = mt5.account_info()
            if account_info is None:
                self.last_error = "Failed to get account info"
                return None
                
            self.account_info = account_info._asdict()
            return self.account_info
            
        except Exception as e:
            self.last_error = f"Error getting account info: {str(e)}"
            print(f"❌ Account info error: {e}")
            return None

    def get_current_price(self, symbol: str):
        """
        ✅ Fixed: Get current bid/ask prices for symbol
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD" will auto-correct to "XAUUSD.v")
            
        Returns:
            dict: {'bid': float, 'ask': float, 'time': int, 'spread': float} or None
        """
        try:
            self._rate_limit()
            
            # 🔥 FIX 1: Auto-correct symbol name (same as get_rates)
            if symbol == "XAUUSD":
                symbol = "XAUUSD.v"
                print(f"🔄 Auto-corrected symbol to: {symbol}")
            
            print(f"💰 Getting current price for: {symbol}")
            
            # 🔥 FIX 2: Ensure symbol is selected first
            if not mt5.symbol_select(symbol, True):
                print(f"⚠️ Could not select {symbol} in Market Watch")
            
            # 🔥 FIX 3: Get tick with proper error handling
            tick = mt5.symbol_info_tick(symbol)
            
            # 🔥 FIX 4: Check tick properly (MT5 returns None or tick object)
            if tick is None:
                # Try alternative methods
                print(f"⚠️ symbol_info_tick failed, trying alternatives...")
                
                # Method 1: Try symbol_info first
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    self.last_error = f"Symbol {symbol} not found"
                    print(f"❌ Symbol {symbol} not found")
                    return None
                
                # Method 2: Try different tick method
                tick = mt5.symbol_info_tick(symbol)
                if tick is None:
                    # Method 3: Get from recent rates
                    print(f"🔄 Getting price from recent rates...")
                    recent_rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 1)
                    
                    if recent_rates is not None and len(recent_rates) > 0:
                        last_candle = recent_rates[-1]
                        # Simulate bid/ask from close price
                        close_price = last_candle[4]  # close is index 4
                        spread_estimate = close_price * 0.0001  # 0.01% spread estimate
                        
                        return {
                            'bid': close_price - spread_estimate,
                            'ask': close_price + spread_estimate,
                            'time': int(last_candle[0]),  # timestamp
                            'spread': spread_estimate * 2
                        }
                    else:
                        self.last_error = f"Cannot get any price data for {symbol}"
                        print(f"❌ Cannot get any price data for {symbol}")
                        return None
            
            # 🔥 FIX 5: Extract data from tick object properly
            try:
                price_data = {
                    'bid': float(tick.bid),
                    'ask': float(tick.ask),
                    'time': int(tick.time),
                    'spread': float(tick.ask - tick.bid)
                }
                
                print(f"✅ Current price: ${price_data['bid']:.2f} / ${price_data['ask']:.2f}")
                print(f"📊 Spread: ${price_data['spread']:.4f}")
                
                return price_data
                
            except Exception as e:
                print(f"❌ Error extracting tick data: {e}")
                print(f"🔍 Tick object type: {type(tick)}")
                print(f"🔍 Tick object dir: {dir(tick) if tick else 'None'}")
                
                # Fallback: try to access attributes differently
                try:
                    return {
                        'bid': getattr(tick, 'bid', 0.0),
                        'ask': getattr(tick, 'ask', 0.0),
                        'time': getattr(tick, 'time', 0),
                        'spread': getattr(tick, 'ask', 0.0) - getattr(tick, 'bid', 0.0)
                    }
                except:
                    self.last_error = f"Failed to extract tick data for {symbol}"
                    return None
            
        except Exception as e:
            self.last_error = f"Error getting current price: {str(e)}"
            print(f"❌ get_current_price() error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_positions(self, symbol: str = None):
        """Get current positions"""
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
            print(f"❌ Positions error: {e}")
            return []

    def get_symbol_info(self, symbol: str):
        """Get symbol information"""
        try:
            self._rate_limit()
            
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                self.last_error = f"Failed to get symbol info for {symbol}"
                return None
                
            return symbol_info._asdict()
            
        except Exception as e:
            self.last_error = f"Error getting symbol info: {str(e)}"
            print(f"❌ Symbol info error for {symbol}: {e}")
            return None

    def test_order_filling_modes(self, symbol: str):
        """Test supported order filling modes for symbol"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return ['IOC']  # Default fallback
                
            filling_mode = symbol_info.get('filling_mode', 0)
            supported_modes = []
            
            # Check which filling modes are supported
            if filling_mode & 1:  # SYMBOL_FILLING_FOK
                supported_modes.append('FOK')
            if filling_mode & 2:  # SYMBOL_FILLING_IOC
                supported_modes.append('IOC')
            if filling_mode & 4:  # SYMBOL_FILLING_RETURN
                supported_modes.append('RETURN')
                
            return supported_modes if supported_modes else ['IOC']
            
        except Exception as e:
            print(f"Error testing filling modes: {e}")
            return ['IOC']  # Safe fallback

    def _get_filling_mode(self, symbol: str):
        """Get best filling mode for symbol"""
        try:
            symbol_info = self.get_symbol_info(symbol)
            if not symbol_info:
                return mt5.ORDER_FILLING_IOC
                
            filling_mode = symbol_info.get('filling_mode', 0)
            
            # Priority: FOK -> IOC -> RETURN
            if filling_mode & 1:  # SYMBOL_FILLING_FOK
                return mt5.ORDER_FILLING_FOK
            elif filling_mode & 2:  # SYMBOL_FILLING_IOC
                return mt5.ORDER_FILLING_IOC
            else:  # SYMBOL_FILLING_RETURN
                return mt5.ORDER_FILLING_RETURN
                
        except Exception as e:
            print(f"Error getting filling mode: {e}")
            return mt5.ORDER_FILLING_IOC

    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: float = None, sl: float = None, tp: float = None,
                   comment: str = "AI Trading"):
        """Place trading order"""
        try:
            self._rate_limit()
            
            # Get current price if not provided
            if price is None:
                current_price = self.get_current_price(symbol)
                if not current_price:
                    return False
                price = current_price['ask'] if order_type.lower() == 'buy' else current_price['bid']
            
            # Determine MT5 order type
            if order_type.lower() == 'buy':
                mt5_order_type = mt5.ORDER_TYPE_BUY
            elif order_type.lower() == 'sell':
                mt5_order_type = mt5.ORDER_TYPE_SELL
            else:
                self.last_error = f"Invalid order type: {order_type}"
                return False
            
            # Get filling mode
            filling_mode = self._get_filling_mode(symbol)
            
            # Prepare order request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5_order_type,
                "price": price,
                "deviation": self.slippage,
                "magic": self.magic_number,
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
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ {order_type.upper()} order placed: {symbol} {volume} lots")
                return True
            else:
                error_code = result.retcode if result else "No result"
                error_comment = result.comment if result else "Unknown error"
                self.last_error = f"Order failed: {error_code} - {error_comment}"
                print(f"❌ Order failed: {self.last_error}")
                return False
                
        except Exception as e:
            self.last_error = f"Error placing order: {str(e)}"
            print(f"❌ Order error: {e}")
            return False

    def close_position(self, position_ticket: int):
        """Close specific position"""
        try:
            self._rate_limit()
            
            # Get position info with retry
            position = None
            for attempt in range(3):
                positions = mt5.positions_get(ticket=position_ticket)
                if positions and len(positions) > 0:
                    position = positions[0]
                    break
                print(f"⚠️ Retry {attempt + 1}: Getting position {position_ticket}...")
                time.sleep(0.3)
                
            if not position:
                print(f"⚠️ Position {position_ticket} not found (might be already closed)")
                return True  # Consider success if position doesn't exist
                
            pos = position
            symbol = pos.symbol
            volume = pos.volume
            pos_type = pos.type
            
            print(f"🔍 Closing: {symbol} {volume} lots, type: {pos_type}")
            
            # Determine close order type and price
            if pos_type == mt5.POSITION_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                current_price = self.get_current_price(symbol)
                if not current_price:
                    print(f"❌ Cannot get tick for {symbol}")
                    return False
                price = current_price['bid']
            else:
                order_type = mt5.ORDER_TYPE_BUY
                current_price = self.get_current_price(symbol)
                if not current_price:
                    print(f"❌ Cannot get tick for {symbol}")
                    return False
                price = current_price['ask']
            
            # Get filling mode
            filling_mode = self._get_filling_mode(symbol)
            
            # Prepare close request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "position": position_ticket,
                "price": price,
                "deviation": self.slippage,
                "magic": self.magic_number,
                "comment": "AI Close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            
            print(f"🔄 Close request: {request}")
            
            # Send close order with retry
            for attempt in range(3):
                result = mt5.order_send(request)
                
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    print(f"✅ Position {position_ticket} closed successfully")
                    return True
                elif result:
                    print(f"⚠️ Close attempt {attempt + 1} failed: {result.retcode} - {result.comment}")
                    
                    # Try different filling mode on retry
                    if attempt < 2:
                        if filling_mode == mt5.ORDER_FILLING_FOK:
                            filling_mode = mt5.ORDER_FILLING_IOC
                        elif filling_mode == mt5.ORDER_FILLING_IOC:
                            filling_mode = mt5.ORDER_FILLING_RETURN
                        else:
                            filling_mode = mt5.ORDER_FILLING_IOC
                            
                        request["type_filling"] = filling_mode
                        print(f"🔄 Retrying with filling mode: {filling_mode}")
                        time.sleep(0.5)
                else:
                    print(f"❌ Close attempt {attempt + 1}: No result returned")
                    time.sleep(0.5)
                    
            self.last_error = f"Close failed after 3 attempts: {result.retcode if result else 'No result'}"
            print(f"❌ Failed to close position {position_ticket} after 3 attempts")
            return False
            
        except Exception as e:
            self.last_error = f"Error closing position: {str(e)}"
            print(f"❌ close_position exception: {e}")
            return False

    def get_rates(self, symbol: str, timeframe: int, count: int):
        """
        ✅ Production-Ready: Get historical rate data with numpy array handling
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD" will auto-correct to "XAUUSD.v")
            timeframe: Timeframe in minutes (5 for M5)
            count: Number of candles to retrieve
            
        Returns:
            numpy.ndarray: Array of rate data or None if failed
            Structure: (timestamp, open, high, low, close, tick_volume, spread, real_volume)
        """
        try:
            self._rate_limit()
            
            # Auto-correct symbol name for this broker
            if symbol == "XAUUSD":
                symbol = "XAUUSD.v"
                print(f"🔄 Auto-corrected symbol to: {symbol}")
            
            # Convert timeframe to MT5 constant
            timeframe_map = {
                1: mt5.TIMEFRAME_M1,
                5: mt5.TIMEFRAME_M5,
                15: mt5.TIMEFRAME_M15,
                30: mt5.TIMEFRAME_M30,
                60: mt5.TIMEFRAME_H1,
                240: mt5.TIMEFRAME_H4,
                1440: mt5.TIMEFRAME_D1
            }
            
            mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
            
            # Ensure symbol is selected in Market Watch
            if not mt5.symbol_select(symbol, True):
                print(f"⚠️ Could not select {symbol} in Market Watch")
            
            # Get historical rates
            rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, count)
            
            # Handle numpy array properly
            if rates is not None and len(rates) > 0:
                print(f"✅ Retrieved {len(rates)} candles for {symbol}")
                return rates
            else:
                self.last_error = f"No rates data for {symbol}. MT5 Error: {mt5.last_error()}"
                print(f"❌ {self.last_error}")
                return None
                
        except Exception as e:
            self.last_error = f"Error getting rates: {str(e)}"
            print(f"❌ get_rates() error: {e}")
            return None
        
    def get_orders(self, symbol: str = None):
        """Get pending orders"""
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
            print(f"❌ Orders error: {e}")
            return []

    def cancel_order(self, order_ticket: int):
        """Cancel pending order"""
        try:
            self._rate_limit()
            
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": order_ticket,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                print(f"✅ Order {order_ticket} cancelled successfully")
                return True
            else:
                error_code = result.retcode if result else "No result"
                self.last_error = f"Cancel failed: {error_code}"
                print(f"❌ Cancel order failed: {self.last_error}")
                return False
                
        except Exception as e:
            self.last_error = f"Error cancelling order: {str(e)}"
            print(f"❌ Cancel order error: {e}")
            return False

    def _rate_limit(self):
        """Implement rate limiting for API requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
            
        self.last_request_time = time.time()

    def get_last_error(self):
        """Get last error message"""
        return self.last_error

    def test_connection(self):
        """Test MT5 connection"""
        try:
            if not self.is_connected:
                return False
                
            # Try to get account info
            account_info = self.get_account_info()
            if not account_info:
                return False
                
            # Try to get a price quote
            current_price = self.get_current_price("XAUUSD.v")
            if not current_price:
                return False
                
            return True
            
        except Exception as e:
            self.last_error = f"Connection test failed: {str(e)}"
            print(f"❌ Connection test error: {e}")
            return False

# Compatibility alias
MT5Interface = MT5Connector
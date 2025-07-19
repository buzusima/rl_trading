import MetaTrader5 as mt5
from typing import Dict
import time

class MT5Connector:
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

    def connect(self, login: int = None, password: str = None, server: str = None):
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
        
    def get_positions(self, symbol: str = None):
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
        
    def _place_order_with_fallback(self, symbol: str, order_type: str, volume: float, 
                                 price: float = None, sl: float = None, tp: float = None,
                                 comment: str = "RL Trading", magic: int = None):
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
   
    def place_order(self, symbol: str, order_type: str, volume: float, 
                   price: float = None, sl: float = None, tp: float = None,
                   comment: str = "RL Trading", magic: int = None):
        return self._place_order_with_fallback(symbol, order_type, volume, 
                                            price, sl, tp, comment, magic)
    
    def close_position(self, position_ticket: int):
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
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    print(f"❌ Cannot get tick for {symbol}")
                    return False
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUY
                tick = mt5.symbol_info_tick(symbol)
                if not tick:
                    print(f"❌ Cannot get tick for {symbol}")
                    return False
                price = tick.ask
                
            # Get best filling mode for this symbol
            filling_mode = self.get_symbol_filling_mode(symbol)
            
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
        try:
            self._rate_limit()
            
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            if rates is None or len(rates) == 0:
                return None
                
            return rates
            
        except Exception as e:
            self.last_error = f"Error getting rates: {str(e)}"
            return None
        
    def _rate_limit(self):
        if hasattr(self, 'training_mode') and self.training_mode:
            return
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
            
        self.last_request_time = time.time()

    def get_last_error(self):
        return self.last_error

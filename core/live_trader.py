# core/live_trader.py - Live Trading Executor for Recovery System

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

# Import existing Recovery components
try:
    from .recovery_intelligence import RecoveryDecision, ActionType, RecoveryState
    from .mt5_connector import MT5Connector
except ImportError:
    from recovery_intelligence import RecoveryDecision, ActionType, RecoveryState
    from mt5_connector import MT5Connector

class LiveExecutionResult(Enum):
    """📊 ผลการเทรดจริง - สำหรับ Recovery System"""
    SUCCESS = "success"           # สำเร็จ
    FAILED = "failed"             # ล้มเหลว
    REJECTED = "rejected"         # ถูกปฏิเสธ (risk limits)
    DISABLED = "disabled"         # Live mode ปิดอยู่
    ERROR = "error"               # เกิดข้อผิดพลาด

@dataclass
class LiveTradeResult:
    """📈 ผลการเทรดจริงครบชุด - เก็บข้อมูลสำหรับ Recovery Brain"""
    result: LiveExecutionResult
    action_executed: ActionType
    order_ticket: Optional[int] = None
    execution_price: Optional[float] = None
    volume_executed: Optional[float] = None
    execution_time: datetime = None
    mt5_error: Optional[str] = None
    profit_loss: Optional[float] = None
    total_positions: int = 0
    message: str = ""

class LiveTrader:
    """
    🔴 Live Trading Executor - Pure Executor สำหรับ Recovery System
    
    หน้าที่:
    1. รับ RecoveryDecision จาก AI Recovery Brain
    2. Execute ตาม AI decision โดยไม่เปลี่ยนแปลงอะไร
    3. ส่งคำสั่งไปยัง MT5 จริง (ไม่มี TP/SL)
    4. รายงานผลกลับให้ AI Brain
    5. จัดการ safety และ error handling เท่านั้น
    
    ⚠️ สำคัญ: ไม่ตัดสินใจอะไรเอง ทำตาม AI เท่านั้น!
    """
    
    def __init__(self, mt5_interface: MT5Connector, config: Dict = None):
        """🏗️ เริ่มต้น Live Trader - Pure Executor"""
        print("🔴 Initializing Live Trading Executor...")
        
        self.mt5_interface = mt5_interface
        self.config = config or {}
        
        # ใช้ค่าจาก config เดิม (ไม่เปลี่ยน logic)
        self.symbol = self.config.get('symbol', 'XAUUSD.v')
        self.max_lot_size = self.config.get('max_lot_size', 0.1)
        self.min_lot_size = 0.01
        self.daily_loss_limit = self.config.get('max_daily_risk', 10.0) * 100  # 10% = $1000
        
        # Live Trading State
        self.is_live_enabled = False  # ⚠️ Default: OFF
        self.session_start_time = None
        self.session_start_balance = 0.0
        self.live_trades_today = 0
        self.live_volume_today = 0.0
        
        # Execution tracking
        self.execution_log = []
        
        print("✅ Live Trading Executor ready (LIVE MODE: DISABLED)")
        print(f"   Symbol: {self.symbol}")
        print(f"   Max Daily Risk: ${self.daily_loss_limit:.0f}")
        print(f"   Max Lot Size: {self.max_lot_size}")

    def enable_live_trading(self, confirmation: bool = False) -> bool:
        """🔓 เปิด Live Trading - ต้องยืนยัน"""
        if not confirmation:
            print("❌ Live trading requires explicit confirmation")
            return False
            
        try:
            # ตรวจสอบ MT5 connection
            if not self.mt5_interface or not self.mt5_interface.is_connected:
                print("❌ MT5 not connected")
                return False
            
            # ดึง account info
            account_info = self.mt5_interface.get_account_info()
            if not account_info:
                print("❌ Cannot get account info")
                return False
            
            self.session_start_balance = account_info.get('balance', 0.0)
            self.session_start_time = datetime.now()
            self.is_live_enabled = True
            
            print("🔴 LIVE TRADING ENABLED!")
            print(f"   Start Balance: ${self.session_start_balance:.2f}")
            print(f"   Session Start: {self.session_start_time.strftime('%H:%M:%S')}")
            
            return True
            
        except Exception as e:
            print(f"❌ Enable live trading error: {e}")
            return False

    def disable_live_trading(self) -> bool:
        """🔒 ปิด Live Trading"""
        self.is_live_enabled = False
        print("🔒 LIVE TRADING DISABLED")
        self._log_session_summary()
        return True

    def execute_ai_decision(self, ai_decision: RecoveryDecision) -> LiveTradeResult:
        """
        ⚡ Execute AI Decision - หัวใจหลักของ Live Trader
        
        Args:
            ai_decision: คำตัดสินใจจาก AI Recovery Brain
            
        Returns:
            LiveTradeResult: ผลการ execute
        """
        try:
            # ตรวจสอบ Live Mode
            if not self.is_live_enabled:
                return LiveTradeResult(
                    result=LiveExecutionResult.DISABLED,
                    action_executed=ai_decision.action,
                    execution_time=datetime.now(),
                    message="Live trading disabled"
                )
            
            # ตรวจสอบ Daily Risk Limit
            if not self._check_daily_risk():
                return LiveTradeResult(
                    result=LiveExecutionResult.REJECTED,
                    action_executed=ai_decision.action,
                    execution_time=datetime.now(),
                    message="Daily risk limit exceeded"
                )
            
            # Execute ตาม AI Action (ไม่เปลี่ยนแปลงอะไร)
            if ai_decision.action == ActionType.HOLD:
                return self._execute_hold(ai_decision)
                
            elif ai_decision.action == ActionType.BUY:
                return self._execute_buy(ai_decision, is_recovery=False)
                
            elif ai_decision.action == ActionType.SELL:
                return self._execute_sell(ai_decision, is_recovery=False)
                
            elif ai_decision.action == ActionType.RECOVERY_BUY:
                return self._execute_buy(ai_decision, is_recovery=True)
                
            elif ai_decision.action == ActionType.RECOVERY_SELL:
                return self._execute_sell(ai_decision, is_recovery=True)
                
            elif ai_decision.action == ActionType.CLOSE_ALL:
                return self._execute_close_all(ai_decision, is_emergency=False)
                
            elif ai_decision.action == ActionType.EMERGENCY_CLOSE:
                return self._execute_close_all(ai_decision, is_emergency=True)
                
            elif ai_decision.action == ActionType.HEDGE:
                return self._execute_hedge(ai_decision)
            
            else:
                return LiveTradeResult(
                    result=LiveExecutionResult.ERROR,
                    action_executed=ai_decision.action,
                    execution_time=datetime.now(),
                    message=f"Unknown action: {ai_decision.action}"
                )
                
        except Exception as e:
            print(f"❌ Execute AI decision error: {e}")
            return LiveTradeResult(
                result=LiveExecutionResult.ERROR,
                action_executed=ai_decision.action,
                execution_time=datetime.now(),
                message=str(e)
            )

    def _execute_hold(self, ai_decision: RecoveryDecision) -> LiveTradeResult:
        """⏸️ Execute HOLD - ไม่ทำอะไร"""
        return LiveTradeResult(
            result=LiveExecutionResult.SUCCESS,
            action_executed=ActionType.HOLD,
            execution_time=datetime.now(),
            message="HOLD - No action taken"
        )

    def _execute_buy(self, ai_decision: RecoveryDecision, is_recovery: bool = False) -> LiveTradeResult:
        """🟢 Execute BUY/RECOVERY_BUY - ใช้ volume ที่ AI กำหนด"""
        try:
            # ใช้ volume ที่ AI คำนวณมาแล้ว (ไม่แก้ไข)
            volume = ai_decision.volume
            
            # จำกัดแค่ safety limits
            volume = max(self.min_lot_size, min(volume, self.max_lot_size))
            
            action_type = "RECOVERY_BUY" if is_recovery else "BUY"
            print(f"🟢 Executing {action_type}: {volume:.2f} lots")
            
            # ดึงราคาปัจจุบัน
            current_price = self.mt5_interface.get_current_price(self.symbol)
            if not current_price:
                return LiveTradeResult(
                    result=LiveExecutionResult.ERROR,
                    action_executed=ai_decision.action,
                    execution_time=datetime.now(),
                    message="Cannot get current price"
                )
            
            entry_price = current_price['ask']
            
            # ส่งคำสั่ง MT5 (ไม่มี TP/SL - ตาม Recovery Logic)
            success = self.mt5_interface.place_order(
                symbol=self.symbol,
                order_type='buy',
                volume=volume,
                price=entry_price,
                sl=None,  # ✅ ไม่ใส่ SL ตาม Recovery System
                tp=None,  # ✅ ไม่ใส่ TP ตาม Recovery System
                comment=f"AI:{ai_decision.strategy_type.value}:{action_type}"
            )
            
            if success:
                self.live_trades_today += 1
                self.live_volume_today += volume
                
                print(f"✅ {action_type} executed: {volume:.2f} lots at ${entry_price:.5f}")
                
                return LiveTradeResult(
                    result=LiveExecutionResult.SUCCESS,
                    action_executed=ai_decision.action,
                    execution_price=entry_price,
                    volume_executed=volume,
                    execution_time=datetime.now(),
                    total_positions=len(self.mt5_interface.get_positions(self.symbol)),
                    message=f"{action_type} executed successfully"
                )
            else:
                error_msg = self.mt5_interface.get_last_error()
                print(f"❌ {action_type} failed: {error_msg}")
                
                return LiveTradeResult(
                    result=LiveExecutionResult.FAILED,
                    action_executed=ai_decision.action,
                    execution_time=datetime.now(),
                    mt5_error=error_msg,
                    message=f"{action_type} order failed"
                )
                
        except Exception as e:
            print(f"❌ Execute buy error: {e}")
            return LiveTradeResult(
                result=LiveExecutionResult.ERROR,
                action_executed=ai_decision.action,
                execution_time=datetime.now(),
                message=str(e)
            )

    def _execute_sell(self, ai_decision: RecoveryDecision, is_recovery: bool = False) -> LiveTradeResult:
        """🔴 Execute SELL/RECOVERY_SELL - ใช้ volume ที่ AI กำหนด"""
        try:
            # ใช้ volume ที่ AI คำนวณมาแล้ว (ไม่แก้ไข)
            volume = ai_decision.volume
            volume = max(self.min_lot_size, min(volume, self.max_lot_size))
            
            action_type = "RECOVERY_SELL" if is_recovery else "SELL"
            print(f"🔴 Executing {action_type}: {volume:.2f} lots")
            
            # ดึงราคาปัจจุบัน
            current_price = self.mt5_interface.get_current_price(self.symbol)
            if not current_price:
                return LiveTradeResult(
                    result=LiveExecutionResult.ERROR,
                    action_executed=ai_decision.action,
                    execution_time=datetime.now(),
                    message="Cannot get current price"
                )
            
            entry_price = current_price['bid']
            
            # ส่งคำสั่ง MT5 (ไม่มี TP/SL - ตาม Recovery Logic)
            success = self.mt5_interface.place_order(
                symbol=self.symbol,
                order_type='sell',
                volume=volume,
                price=entry_price,
                sl=None,  # ✅ ไม่ใส่ SL ตาม Recovery System
                tp=None,  # ✅ ไม่ใส่ TP ตาม Recovery System
                comment=f"AI:{ai_decision.strategy_type.value}:{action_type}"
            )
            
            if success:
                self.live_trades_today += 1
                self.live_volume_today += volume
                
                print(f"✅ {action_type} executed: {volume:.2f} lots at ${entry_price:.5f}")
                
                return LiveTradeResult(
                    result=LiveExecutionResult.SUCCESS,
                    action_executed=ai_decision.action,
                    execution_price=entry_price,
                    volume_executed=volume,
                    execution_time=datetime.now(),
                    total_positions=len(self.mt5_interface.get_positions(self.symbol)),
                    message=f"{action_type} executed successfully"
                )
            else:
                error_msg = self.mt5_interface.get_last_error()
                print(f"❌ {action_type} failed: {error_msg}")
                
                return LiveTradeResult(
                    result=LiveExecutionResult.FAILED,
                    action_executed=ai_decision.action,
                    execution_time=datetime.now(),
                    mt5_error=error_msg,
                    message=f"{action_type} order failed"
                )
                
        except Exception as e:
            print(f"❌ Execute sell error: {e}")
            return LiveTradeResult(
                result=LiveExecutionResult.ERROR,
                action_executed=ai_decision.action,
                execution_time=datetime.now(),
                message=str(e)
            )

    def _execute_close_all(self, ai_decision: RecoveryDecision, is_emergency: bool = False) -> LiveTradeResult:
        """🚪 Execute CLOSE_ALL/EMERGENCY_CLOSE - ปิดทั้งหมดตาม AI สั่ง"""
        try:
            action_type = "EMERGENCY_CLOSE" if is_emergency else "CLOSE_ALL"
            print(f"🚪 Executing {action_type}...")
            
            # ดึงรายการ positions ทั้งหมด
            positions = self.mt5_interface.get_positions(self.symbol)
            
            if not positions:
                return LiveTradeResult(
                    result=LiveExecutionResult.SUCCESS,
                    action_executed=ai_decision.action,
                    execution_time=datetime.now(),
                    total_positions=0,
                    profit_loss=0.0,
                    message="No positions to close"
                )
            
            total_profit = 0.0
            closed_count = 0
            failed_count = 0
            
            print(f"🔍 Found {len(positions)} positions to close...")
            
            for position in positions:
                ticket = position.get('ticket')
                current_profit = position.get('profit', 0.0)
                volume = position.get('volume', 0.0)
                pos_type = position.get('type', 0)
                
                print(f"   Closing #{ticket}: {volume:.2f} lots, P&L: ${current_profit:.2f}")
                
                success = self.mt5_interface.close_position(ticket)
                
                if success:
                    closed_count += 1
                    total_profit += current_profit
                    print(f"   ✅ Closed #{ticket}")
                else:
                    failed_count += 1
                    error_msg = self.mt5_interface.get_last_error()
                    print(f"   ❌ Failed to close #{ticket}: {error_msg}")
            
            # สรุปผล
            if closed_count > 0:
                result_status = LiveExecutionResult.SUCCESS if failed_count == 0 else LiveExecutionResult.FAILED
                message = f"Closed {closed_count}/{len(positions)} positions, Total P&L: ${total_profit:.2f}"
                
                if failed_count > 0:
                    message += f" ({failed_count} failed)"
                
                print(f"✅ {action_type} completed: {message}")
                
                return LiveTradeResult(
                    result=result_status,
                    action_executed=ai_decision.action,
                    execution_time=datetime.now(),
                    total_positions=len(positions) - closed_count,
                    profit_loss=total_profit,
                    message=message
                )
            else:
                return LiveTradeResult(
                    result=LiveExecutionResult.FAILED,
                    action_executed=ai_decision.action,
                    execution_time=datetime.now(),
                    total_positions=len(positions),
                    message="Failed to close any positions"
                )
                
        except Exception as e:
            print(f"❌ Execute close all error: {e}")
            return LiveTradeResult(
                result=LiveExecutionResult.ERROR,
                action_executed=ai_decision.action,
                execution_time=datetime.now(),
                message=str(e)
            )

    def _execute_hedge(self, ai_decision: RecoveryDecision) -> LiveTradeResult:
        """🔄 Execute HEDGE - เปิดตำแหน่งตรงข้าม"""
        try:
            print("🔄 Executing HEDGE strategy...")
            
            # ดึง positions ปัจจุบัน
            positions = self.mt5_interface.get_positions(self.symbol)
            
            if not positions:
                return LiveTradeResult(
                    result=LiveExecutionResult.ERROR,
                    action_executed=ActionType.HEDGE,
                    execution_time=datetime.now(),
                    message="No positions to hedge"
                )
            
            # คำนวณ net exposure
            total_buy_volume = sum(pos['volume'] for pos in positions if pos.get('type') == 0)  # BUY
            total_sell_volume = sum(pos['volume'] for pos in positions if pos.get('type') == 1)  # SELL
            
            net_volume = total_buy_volume - total_sell_volume
            
            if abs(net_volume) < 0.01:
                return LiveTradeResult(
                    result=LiveExecutionResult.SUCCESS,
                    action_executed=ActionType.HEDGE,
                    execution_time=datetime.now(),
                    message="Positions already balanced"
                )
            
            # เปิดตำแหน่งตรงข้าม
            hedge_volume = min(abs(net_volume), ai_decision.volume)
            
            if net_volume > 0:
                # Net long -> Hedge with SELL
                return self._execute_sell(ai_decision, is_recovery=False)
            else:
                # Net short -> Hedge with BUY  
                return self._execute_buy(ai_decision, is_recovery=False)
                
        except Exception as e:
            print(f"❌ Execute hedge error: {e}")
            return LiveTradeResult(
                result=LiveExecutionResult.ERROR,
                action_executed=ActionType.HEDGE,
                execution_time=datetime.now(),
                message=str(e)
            )

    def _check_daily_risk(self) -> bool:
        """⚠️ ตรวจสอบ Daily Risk Limit"""
        try:
            if not self.session_start_balance:
                return True  # ไม่สามารถตรวจสอบได้
            
            account_info = self.mt5_interface.get_account_info()
            if not account_info:
                return True  # ไม่สามารถตรวจสอบได้
            
            current_balance = account_info.get('balance', 0.0)
            daily_loss = self.session_start_balance - current_balance
            
            if daily_loss >= self.daily_loss_limit:
                print(f"🛑 Daily risk limit exceeded: ${daily_loss:.2f} >= ${self.daily_loss_limit:.2f}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Risk check error: {e}")
            return True  # ถ้าเช็คไม่ได้ ให้ผ่านไป (safer)

    def _log_session_summary(self):
        """📊 สรุปผลการเทรด Live Session"""
        try:
            if not self.session_start_time:
                return
                
            duration = datetime.now() - self.session_start_time
            
            # ดึง balance ปัจจุบัน
            account_info = self.mt5_interface.get_account_info()
            current_balance = account_info.get('balance', 0.0) if account_info else 0.0
            
            daily_pnl = current_balance - self.session_start_balance
            
            print("📊 Live Trading Session Summary:")
            print(f"   Duration: {duration}")
            print(f"   Start Balance: ${self.session_start_balance:.2f}")
            print(f"   End Balance: ${current_balance:.2f}")
            print(f"   Daily P&L: ${daily_pnl:.2f}")
            print(f"   Total Trades: {self.live_trades_today}")
            print(f"   Total Volume: {self.live_volume_today:.2f} lots")
            
            if self.live_trades_today > 0:
                avg_volume = self.live_volume_today / self.live_trades_today
                print(f"   Avg Volume/Trade: {avg_volume:.2f} lots")
                
        except Exception as e:
            print(f"❌ Session summary error: {e}")

    def get_live_status(self) -> Dict:
        """📊 ดูสถานะ Live Trading"""
        try:
            # ดึงข้อมูล positions ปัจจุบัน
            positions = self.mt5_interface.get_positions(self.symbol) if self.mt5_interface else []
            total_positions = len(positions)
            unrealized_pnl = sum(pos.get('profit', 0.0) for pos in positions)
            
            # ดึง account info
            account_info = self.mt5_interface.get_account_info() if self.mt5_interface else {}
            current_balance = account_info.get('balance', 0.0)
            
            # คำนวณ daily P&L
            daily_pnl = current_balance - self.session_start_balance if self.session_start_balance else 0.0
            
            return {
                'live_enabled': self.is_live_enabled,
                'session_start': self.session_start_time.strftime('%H:%M:%S') if self.session_start_time else None,
                'start_balance': self.session_start_balance,
                'current_balance': current_balance,
                'daily_pnl': daily_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_positions': total_positions,
                'live_trades_today': self.live_trades_today,
                'live_volume_today': self.live_volume_today,
                'daily_risk_used': (abs(daily_pnl) / self.daily_loss_limit * 100) if self.daily_loss_limit > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Get live status error: {e}")
            return {'live_enabled': False, 'error': str(e)}

    def is_live_trading_enabled(self) -> bool:
        """🔍 ตรวจสอบสถานะ Live Trading"""
        return self.is_live_enabled
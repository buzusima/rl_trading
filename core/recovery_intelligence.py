# core/recovery_intelligence.py - AI Recovery Intelligence Brain

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
import threading
import time

# Import จากไฟล์อื่นๆ
try:
    from .market_analyzer import MarketAnalyzer, MarketContext, MarketRegime
    from .strategy_selector import StrategySelector, StrategySelection, StrategyType
except ImportError:
    from market_analyzer import MarketAnalyzer, MarketContext, MarketRegime
    from strategy_selector import StrategySelector, StrategySelection, StrategyType

class RecoveryState(Enum):
    """
    🔄 สถานะการแก้ไม้ - ระบุว่าระบบอยู่ในสถานะใด
    """
    NORMAL = "normal"                    # ปกติ ไม่ได้แก้ไม้
    EARLY_RECOVERY = "early_recovery"    # เริ่มแก้ไม้ระยะแรก
    ACTIVE_RECOVERY = "active_recovery"  # กำลังแก้ไม้อย่างจริงจัง
    DEEP_RECOVERY = "deep_recovery"      # แก้ไม้ลึก (สถานการณ์หนัก)
    EMERGENCY = "emergency"              # ฉุกเฉิน (ใช้กลยุทธ์สุดท้าย)
    SUCCESS = "success"                  # แก้ไม้สำเร็จ
    FAILURE = "failure"                  # แก้ไม้ล้มเหลว

class ActionType(Enum):
    """
    ⚡ ประเภทการกระทำ - AI สามารถสั่งได้
    """
    HOLD = 0            # รอดู ไม่ทำอะไร
    BUY = 1            # ซื้อ
    SELL = 2           # ขาย  
    CLOSE_ALL = 3      # ปิดทั้งหมด
    RECOVERY_BUY = 4   # ซื้อแก้ไม้
    RECOVERY_SELL = 5  # ขายแก้ไม้
    HEDGE = 6          # เปิด hedge
    EMERGENCY_CLOSE = 7 # ปิดฉุกเฉิน

@dataclass
class RecoveryDecision:
    """
    🎯 คำตัดสินใจของ AI Recovery - ข้อมูลการตัดสินใจครบชุด
    """
    action: ActionType                    # การกระทำหลัก
    strategy_type: StrategyType          # กลยุทธ์ที่ใช้
    volume: float                        # ขนาด lot
    entry_price: Optional[float]         # ราคาเข้า (ถ้ามี)
    stop_loss: Optional[float]           # Stop loss (ถ้ามี)
    take_profit: Optional[float]         # Take profit (ถ้ามี) 
    recovery_state: RecoveryState        # สถานะการแก้ไม้
    confidence: float                    # ความเชื่อมั่น (0-1)
    reasoning: List[str]                 # เหตุผลการตัดสินใจ
    warnings: List[str]                  # คำเตือน
    market_context: MarketContext        # บริบทตลาดที่ใช้ตัดสินใจ
    expected_outcome: Dict[str, float]   # ผลที่คาดหวัง
    risk_assessment: Dict[str, Any]      # การประเมินความเสี่ยง
    timestamp: datetime                  # เวลาตัดสินใจ

@dataclass
class RecoverySessionState:
    """
    📊 สถานะ Recovery Session - เก็บข้อมูลการแก้ไม้ทั้ง session
    """
    session_id: str                      # รหัส session
    start_time: datetime                 # เวลาเริ่ม
    initial_balance: float               # ยอดเงินเริ่มต้น
    current_balance: float               # ยอดเงินปัจจุบัน
    total_pnl: float                     # กำไร/ขาดทุนรวม
    max_drawdown: float                  # Drawdown สูงสุด
    recovery_state: RecoveryState        # สถานะปัจจุบัน
    current_strategy: Optional[StrategyType] # กลยุทธ์ปัจจุบัน
    recovery_attempts: int               # จำนวนครั้งที่พยายามแก้ไม้
    successful_recoveries: int           # จำนวนครั้งที่แก้ไม้สำเร็จ
    failed_recoveries: int               # จำนวนครั้งที่แก้ไม้ล้มเหลว
    total_trades: int                    # จำนวนเทรดรวม
    winning_trades: int                  # จำนวนเทรดกำไร
    losing_trades: int                   # จำนวนเทรดขาดทุน
    active_positions: int                # ตำแหน่งที่เปิดอยู่
    last_update: datetime                # อัปเดตล่าสุด

class RecoveryIntelligence:
    """
    🧠 สมองกลาง AI Recovery System - รวมทุกอย่างเป็นระบบเดียว
    
    หน้าที่หลัก:
    1. รวม MarketAnalyzer + StrategySelector เป็นระบบเดียว
    2. ตัดสินใจการเทรดและการแก้ไม้อัตโนมัติ
    3. จัดการสถานะและประวัติการแก้ไม้
    4. ปรับเปลี่ยนกลยุทธ์แบบไดนามิก
    5. ให้ข้อมูลสำหรับ AI Agent และ GUI
    6. ควบคุมความเสี่ยงและการจัดการเงิน
    """
    
    def __init__(self, mt5_interface, config: Dict = None):
        """
        🏗️ เริ่มต้น Recovery Intelligence Brain
        
        หน้าที่:
        - สร้าง MarketAnalyzer และ StrategySelector
        - ตั้งค่าพารามิเตอร์การแก้ไม้
        - เริ่มระบบ monitoring
        - เตรียม session state
        """
        print("🧠 Initializing Recovery Intelligence Brain...")
        
        self.mt5_interface = mt5_interface
        self.config = config or {}
        
        # สร้างคอมโพเนนต์หลัก
        self.market_analyzer = MarketAnalyzer(mt5_interface, config)
        self.strategy_selector = StrategySelector(config)
        
        # พารามิเตอร์การแก้ไม้
        self.recovery_thresholds = {
            'early_recovery': self.config.get('early_recovery_threshold', -20.0),    # -$20
            'active_recovery': self.config.get('active_recovery_threshold', -50.0),  # -$50
            'deep_recovery': self.config.get('deep_recovery_threshold', -100.0),     # -$100
            'emergency': self.config.get('emergency_threshold', -200.0)              # -$200
        }
        
        # การจัดการความเสี่ยง
        self.max_positions = self.config.get('max_positions', 5)
        self.max_daily_loss = self.config.get('max_daily_loss', 500.0)  # $500
        self.max_drawdown_percent = self.config.get('max_drawdown_percent', 20.0)  # 20%
        
        # สถานะปัจจุบัน
        self.current_session = None
        self.recovery_history = []
        self.decision_cache = {}
        self.last_decision = None
        
        # ระบบ monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.update_interval = 5  # วินาที
        
        # Performance tracking
        self.performance_metrics = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'failed_decisions': 0,
            'avg_decision_time': 0.0,
            'strategy_changes': 0
        }
        
        print("✅ Recovery Intelligence Brain initialized")
        self._log_system_status()

    def start_recovery_session(self, initial_balance: float = None) -> str:
        """
        🚀 เริ่ม Recovery Session ใหม่
        
        หน้าที่:
        1. สร้าง session ใหม่
        2. รีเซ็ตสถานะทั้งหมด  
        3. เริ่มระบบ monitoring
        4. คืน session ID
        """
        try:
            # ดึง balance จาก MT5 ถ้าไม่ระบุ
            if initial_balance is None:
                account_info = self.mt5_interface.get_account_info()
                initial_balance = account_info.get('balance', 0.0) if account_info else 0.0
            
            # สร้าง session ใหม่
            session_id = f"recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.current_session = RecoverySessionState(
                session_id=session_id,
                start_time=datetime.now(),
                initial_balance=initial_balance,
                current_balance=initial_balance,
                total_pnl=0.0,
                max_drawdown=0.0,
                recovery_state=RecoveryState.NORMAL,
                current_strategy=None,
                recovery_attempts=0,
                successful_recoveries=0,
                failed_recoveries=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                active_positions=0,
                last_update=datetime.now()
            )
            
            # เริ่ม monitoring
            self.start_monitoring()
            
            print(f"🚀 Recovery session started: {session_id}")
            print(f"   Initial balance: ${initial_balance:.2f}")
            
            return session_id
            
        except Exception as e:
            print(f"❌ Start session error: {e}")
            return None

    def make_recovery_decision(self, current_positions: List = None, 
                              force_analysis: bool = False) -> RecoveryDecision:
        """
        🎯 ตัดสินใจหลัก - ฟังก์ชันหลักสำหรับตัดสินใจการเทรดและแก้ไม้
        
        หน้าที่:
        1. วิเคราะห์สถานการณ์ปัจจุบัน
        2. ประเมินสถานะการแก้ไม้
        3. เลือกกลยุทธ์ที่เหมาะสม
        4. ตัดสินใจการกระทำ
        5. คำนวณพารามิเตอร์การเทรด
        
        Args:
            current_positions: ตำแหน่งที่เปิดอยู่
            force_analysis: บังคับวิเคราะห์ใหม่
        """
        try:
            decision_start_time = time.time()
            print("🎯 Making recovery decision...")
            
            # 1. อัปเดตสถานะ session
            self._update_session_state(current_positions)
            
            # 2. ตรวจสอบเงื่อนไขหยุดการเทรด
            if self._should_stop_trading():
                return self._create_stop_trading_decision()
            
            # 3. วิเคราะห์ตลาด
            market_context = self._get_market_analysis(force_analysis)
            if not market_context:
                return self._create_error_decision("Market analysis failed")
            
            # 4. ประเมินสถานะการแก้ไม้
            recovery_state = self._assess_recovery_state()
            
            # 5. เลือกกลยุทธ์
            strategy_selection = self._select_recovery_strategy(market_context, recovery_state)
            
            # 6. ตัดสินใจการกระทำ
            action_decision = self._decide_action(strategy_selection, current_positions, recovery_state)
            
            # 7. คำนวณพารามิเตอร์
            trading_params = self._calculate_trading_parameters(action_decision, strategy_selection, market_context)
            
            # 8. ประเมินความเสี่ยงและผลที่คาดหวัง
            risk_assessment = self._assess_risks(action_decision, trading_params, market_context)
            expected_outcome = self._calculate_expected_outcome(action_decision, trading_params)
            
            # 9. สร้างคำตัดสินใจสุดท้าย
            decision = RecoveryDecision(
                action=action_decision['action'],
                strategy_type=strategy_selection.primary_strategy.strategy_type,
                volume=trading_params['volume'],
                entry_price=trading_params.get('entry_price'),
                stop_loss=trading_params.get('stop_loss'),
                take_profit=trading_params.get('take_profit'),
                recovery_state=recovery_state,
                confidence=action_decision['confidence'],
                reasoning=action_decision['reasoning'],
                warnings=action_decision['warnings'],
                market_context=market_context,
                expected_outcome=expected_outcome,
                risk_assessment=risk_assessment,
                timestamp=datetime.now()
            )
            
            # 10. บันทึกและอัปเดต
            self.last_decision = decision
            self._update_performance_metrics(decision, time.time() - decision_start_time)
            
            print(f"✅ Decision made: {decision.action.name} with {decision.strategy_type.value}")
            print(f"   Confidence: {decision.confidence:.2f}, Volume: {decision.volume}")
            
            return decision
            
        except Exception as e:
            print(f"❌ Decision making error: {e}")
            return self._create_error_decision(str(e))

    def _update_session_state(self, current_positions: List = None):
        """
        📊 อัปเดตสถานะ Session
        
        หน้าที่:
        1. อัปเดต balance และ P&L
        2. คำนวณ drawdown
        3. นับตำแหน่งที่เปิด
        4. อัปเดตเวลา
        """
        try:
            if not self.current_session:
                return
                
            # อัปเดต balance
            account_info = self.mt5_interface.get_account_info()
            if account_info:
                self.current_session.current_balance = account_info.get('balance', 0.0)
                equity = account_info.get('equity', 0.0)
                
                # คำนวณ P&L
                self.current_session.total_pnl = self.current_session.current_balance - self.current_session.initial_balance
                
                # คำนวณ drawdown
                current_drawdown = min(0, self.current_session.total_pnl)
                if current_drawdown < self.current_session.max_drawdown:
                    self.current_session.max_drawdown = current_drawdown
            
            # อัปเดตจำนวนตำแหน่ง
            if current_positions is not None:
                self.current_session.active_positions = len(current_positions)
            else:
                positions = self.mt5_interface.get_positions() if self.mt5_interface else []
                self.current_session.active_positions = len(positions)
            
            self.current_session.last_update = datetime.now()
            
        except Exception as e:
            print(f"❌ Session state update error: {e}")

    def _should_stop_trading(self) -> bool:
        """
        🛑 ตรวจสอบว่าควรหยุดเทรดมั้ย
        
        หน้าที่:
        1. ตรวจสอบ daily loss limit
        2. ตรวจสอบ max drawdown
        3. ตรวจสอบเวลาเทรด
        4. ตรวจสอบสถานะฉุกเฉิน
        """
        if not self.current_session:
            return False
        
        # ตรวจสอบ daily loss
        if abs(self.current_session.total_pnl) >= self.max_daily_loss:
            print(f"🛑 Daily loss limit reached: ${self.current_session.total_pnl:.2f}")
            return True
        
        # ตรวจสอบ max drawdown percentage
        if self.current_session.initial_balance > 0:
            drawdown_percent = abs(self.current_session.max_drawdown) / self.current_session.initial_balance * 100
            if drawdown_percent >= self.max_drawdown_percent:
                print(f"🛑 Max drawdown reached: {drawdown_percent:.1f}%")
                return True
        
        # ตรวจสอบสถานะฉุกเฉิน
        if self.current_session.recovery_state == RecoveryState.FAILURE:
            print("🛑 Recovery failure state - stopping trading")
            return True
        
        return False

    def _get_market_analysis(self, force_analysis: bool = False) -> Optional[MarketContext]:
        """
        📊 ดึงการวิเคราะห์ตลาด
        
        หน้าที่:
        1. ใช้แคชหากไม่บังคับวิเคราะห์ใหม่
        2. วิเคราะห์ตลาดผ่าน MarketAnalyzer
        3. เก็บแคชสำหรับใช้ครั้งถัดไป
        """
        try:
            cache_key = f"market_analysis_{int(time.time() // 60)}"  # แคช 1 นาที
            
            if not force_analysis and cache_key in self.decision_cache:
                return self.decision_cache[cache_key]
            
            market_context = self.market_analyzer.analyze_market()
            self.decision_cache[cache_key] = market_context
            
            return market_context
            
        except Exception as e:
            print(f"❌ Market analysis error: {e}")
            return None

    def _assess_recovery_state(self) -> RecoveryState:
        """
        🔍 ประเมินสถานะการแก้ไม้
        
        หน้าที่:
        1. ตรวจสอบ P&L ปัจจุบัน
        2. เปรียบเทียบกับเกณฑ์ต่างๆ
        3. กำหนดสถานะการแก้ไม้
        """
        if not self.current_session:
            return RecoveryState.NORMAL
        
        current_pnl = self.current_session.total_pnl
        
        # ตรวจสอบตามลำดับความรุนแรง
        if current_pnl <= self.recovery_thresholds['emergency']:
            self.current_session.recovery_state = RecoveryState.EMERGENCY
        elif current_pnl <= self.recovery_thresholds['deep_recovery']:
            self.current_session.recovery_state = RecoveryState.DEEP_RECOVERY
        elif current_pnl <= self.recovery_thresholds['active_recovery']:
            self.current_session.recovery_state = RecoveryState.ACTIVE_RECOVERY
        elif current_pnl <= self.recovery_thresholds['early_recovery']:
            self.current_session.recovery_state = RecoveryState.EARLY_RECOVERY
        else:
            # ถ้า P&L เป็นบวกหรือขาดทุนน้อย
            if current_pnl >= 0:
                self.current_session.recovery_state = RecoveryState.SUCCESS
            else:
                self.current_session.recovery_state = RecoveryState.NORMAL
        
        return self.current_session.recovery_state

    def _select_recovery_strategy(self, market_context: MarketContext, 
                                 recovery_state: RecoveryState) -> StrategySelection:
        """
        🎯 เลือกกลยุทธ์แก้ไม้
        
        หน้าที่:
        1. ส่ง market context ให้ StrategySelector
        2. ปรับการเลือกตาม recovery state
        3. บังคับใช้กลยุทธ์เฉพาะในสถานการณ์ฉุกเฉิน
        """
        current_drawdown = abs(self.current_session.total_pnl) if self.current_session else 0.0
        
        # บังคับกลยุทธ์ในสถานการณ์พิเศษ
        forced_strategy = None
        
        if recovery_state == RecoveryState.EMERGENCY:
            forced_strategy = StrategyType.EMERGENCY_RECOVERY
        elif recovery_state == RecoveryState.DEEP_RECOVERY:
            # เลือกระหว่างกลยุทธ์ที่ปลอดภัย
            if market_context.volatility_score > 80:
                forced_strategy = StrategyType.HEDGING_RECOVERY
            else:
                forced_strategy = StrategyType.CONSERVATIVE_MARTINGALE
        
        return self.strategy_selector.select_strategy(
            market_context=market_context,
            current_drawdown=current_drawdown,
            force_strategy=forced_strategy
        )

    def _decide_action(self, strategy_selection: StrategySelection, 
                      current_positions: List, recovery_state: RecoveryState) -> Dict:
        """
        ⚡ ตัดสินใจการกระทำ
        
        หน้าที่:
        1. ใช้กลยุทธ์ที่เลือกในการตัดสินใจ
        2. พิจารณาตำแหน่งปัจจุบัน
        3. ปรับการกระทำตาม recovery state
        4. คำนวณ confidence level
        """
        strategy = strategy_selection.primary_strategy.strategy_type
        market_context = strategy_selection.market_context
        
        reasoning = []
        warnings = []
        base_confidence = strategy_selection.confidence_score
        
        # จำนวนตำแหน่งปัจจุบัน
        num_positions = len(current_positions) if current_positions else 0
        
        # ตัดสินใจตาม recovery state
        if recovery_state == RecoveryState.EMERGENCY:
            # สถานการณ์ฉุกเฉิน - ปิดทั้งหมดหรือรอ
            if num_positions > 0:
                action = ActionType.EMERGENCY_CLOSE
                confidence = 0.9
                reasoning.append("Emergency state - closing all positions")
            else:
                action = ActionType.HOLD
                confidence = 0.3
                reasoning.append("Emergency state - holding position")
                warnings.append("Emergency recovery state active")
        
        elif recovery_state in [RecoveryState.DEEP_RECOVERY, RecoveryState.ACTIVE_RECOVERY]:
            # การแก้ไม้จริงจัง
            action = self._choose_recovery_action(strategy, market_context, num_positions)
            confidence = base_confidence * 0.8  # ลดความเชื่อมั่นเล็กน้อย
            reasoning.append(f"Active recovery using {strategy.value}")
            
        elif recovery_state == RecoveryState.EARLY_RECOVERY:
            # เริ่มแก้ไม้
            if num_positions < self.max_positions // 2:  # ยังไม่เกินครึ่ง
                action = self._choose_recovery_action(strategy, market_context, num_positions)
                confidence = base_confidence
                reasoning.append("Early recovery phase")
            else:
                action = ActionType.HOLD
                confidence = base_confidence * 0.6
                reasoning.append("Too many positions - holding")
                
        elif recovery_state == RecoveryState.SUCCESS:
            # แก้ไม้สำเร็จ - พิจารณาปิดหรือเทรดต่อ
            if num_positions > 0 and market_context.confidence_score < 0.6:
                action = ActionType.CLOSE_ALL
                confidence = 0.8
                reasoning.append("Recovery successful - taking profits")
            else:
                action = self._choose_normal_action(strategy, market_context)
                confidence = base_confidence
                reasoning.append("Recovery successful - normal trading")
        
        else:  # NORMAL
            # เทรดปกติ
            action = self._choose_normal_action(strategy, market_context)
            confidence = base_confidence
            reasoning.append("Normal trading mode")
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'warnings': warnings
        }

    def _choose_recovery_action(self, strategy: StrategyType, 
                               market_context: MarketContext, 
                               num_positions: int) -> ActionType:
        """
        🔄 เลือกการกระทำแก้ไม้
        
        หน้าที่:
        - เลือกการกระทำที่เหมาะสมสำหรับแก้ไม้
        - ใช้ strategy และ market context ในการตัดสินใจ
        """
        # ถ้ามีตำแหน่งเยอะแล้ว อาจต้องปิดบางส่วน
        if num_positions >= self.max_positions:
            return ActionType.CLOSE_ALL
        
        # เลือกตาม strategy type
        if strategy == StrategyType.HEDGING_RECOVERY:
            # สำหรับ hedge - เปิดตรงข้าม
            return ActionType.HEDGE
        
        elif strategy == StrategyType.AGGRESSIVE_GRID:
            # Grid strategy - เปิดตาม trend
            if market_context.trend_strength > 0:
                return ActionType.RECOVERY_BUY
            else:
                return ActionType.RECOVERY_SELL
        
        elif strategy == StrategyType.MEAN_REVERSION:
            # Mean reversion - เปิดตรงข้ามกับ trend
            if market_context.trend_strength > 0:
                return ActionType.RECOVERY_SELL
            else:
                return ActionType.RECOVERY_BUY
        
        elif strategy == StrategyType.BREAKOUT_RECOVERY:
            # Breakout - ตาม momentum
            if market_context.volatility_score > 60:
                if market_context.trend_strength > 30:
                    return ActionType.RECOVERY_BUY
                elif market_context.trend_strength < -30:
                    return ActionType.RECOVERY_SELL
        
        elif strategy == StrategyType.MOMENTUM_RECOVERY:
            # Momentum - ขยายทิศทางเดิม
            if market_context.trend_strength > 40:
                return ActionType.RECOVERY_BUY
            elif market_context.trend_strength < -40:
                return ActionType.RECOVERY_SELL
        
        # Default - conservative approach
        return ActionType.HOLD

    def _choose_normal_action(self, strategy: StrategyType, 
                             market_context: MarketContext) -> ActionType:
        """
        📈 เลือกการกระทำปกติ (ไม่ใช่แก้ไม้)
        
        หน้าที่:
        - เทรดปกติตามสภาวะตลาด
        - ไม่ใช่การแก้ไม้
        """
        # ตัดสินใจตาม market regime
        if market_context.regime.name == 'TRENDING_UP' and market_context.confidence_score > 0.7:
            return ActionType.BUY
        elif market_context.regime.name == 'TRENDING_DOWN' and market_context.confidence_score > 0.7:
            return ActionType.SELL
        elif market_context.confidence_score < 0.4:
            return ActionType.HOLD
        elif market_context.volatility_score > 80:
            # ตลาดผันผวนสูง - รอดู
            return ActionType.HOLD
        else:
            # ตัดสินใจตาม trend strength
            if market_context.trend_strength > 60:
                return ActionType.BUY
            elif market_context.trend_strength < -60:
                return ActionType.SELL
            else:
                return ActionType.HOLD

    def _calculate_trading_parameters(self, action_decision: Dict, 
                                     strategy_selection: StrategySelection,
                                     market_context: MarketContext) -> Dict:
        """
        📊 คำนวณพารามิเตอร์การเทรด
        
        หน้าที่:
        1. คำนวณขนาด lot
        2. คำนวณ stop loss และ take profit
        3. คำนวณราคาเข้า
        4. ปรับตาม recovery state
        """
        try:
            params = {}
            action = action_decision['action']
            strategy = strategy_selection.primary_strategy
            
            # คำนวณขนาด lot
            base_volume = strategy_selection.position_sizing['base_lot']
            
            if action in [ActionType.RECOVERY_BUY, ActionType.RECOVERY_SELL]:
                # ใช้ recovery volume
                params['volume'] = strategy_selection.position_sizing['recovery_lot']
            elif action == ActionType.HEDGE:
                # ขนาดพิเศษสำหรับ hedge
                params['volume'] = strategy_selection.position_sizing['base_lot'] * 1.5
            else:
                # ขนาดปกติ
                params['volume'] = base_volume
            
            # จำกัดขนาดสูงสุด
            max_volume = self.config.get('max_lot_size', 0.10)
            params['volume'] = min(params['volume'], max_volume)
            
            # คำนวณ stop loss และ take profit
            if strategy.stop_loss_pips:
                params['stop_loss_pips'] = strategy.stop_loss_pips
            
            if strategy.take_profit_pips:
                params['take_profit_pips'] = strategy.take_profit_pips
            
            # ปรับ SL/TP ตาม volatility
            volatility_multiplier = 1.0
            if market_context.volatility_score > 80:
                volatility_multiplier = 1.5  # เพิ่ม SL/TP เมื่อผันผวนสูง
            elif market_context.volatility_score < 30:
                volatility_multiplier = 0.7  # ลด SL/TP เมื่อผันผวนต่ำ
            
            if 'stop_loss_pips' in params:
                params['stop_loss_pips'] = int(params['stop_loss_pips'] * volatility_multiplier)
            if 'take_profit_pips' in params:
                params['take_profit_pips'] = int(params['take_profit_pips'] * volatility_multiplier)
            
            return params
            
        except Exception as e:
            print(f"❌ Trading parameters calculation error: {e}")
            return {'volume': 0.01, 'stop_loss_pips': 30, 'take_profit_pips': 15}

    def _assess_risks(self, action_decision: Dict, trading_params: Dict, 
                     market_context: MarketContext) -> Dict[str, Any]:
        """
        ⚠️ ประเมินความเสี่ยง
        
        หน้าที่:
        1. คำนวณความเสี่ยงของการตัดสินใจ
        2. ประเมิน worst case scenario
        3. คำนวณ risk/reward ratio
        """
        try:
            risk_assessment = {}
            
            # ความเสี่ยงพื้นฐาน
            volume = trading_params.get('volume', 0.01)
            sl_pips = trading_params.get('stop_loss_pips', 30)
            
            # คำนวณความเสี่ยงเป็นเงิน (สำหรับ Gold)
            risk_amount = volume * sl_pips * 100  # $1 per pip per 0.01 lot
            
            risk_assessment['position_risk'] = risk_amount
            risk_assessment['risk_percentage'] = (risk_amount / self.current_session.current_balance * 100) if self.current_session else 0
            
            # ความเสี่ยงตลาด
            market_risk = 'LOW'
            if market_context.volatility_score > 80:
                market_risk = 'HIGH'
            elif market_context.volatility_score > 60:
                market_risk = 'MEDIUM'
            
            risk_assessment['market_risk'] = market_risk
            risk_assessment['volatility_score'] = market_context.volatility_score
            risk_assessment['confidence_score'] = market_context.confidence_score
            
            # ความเสี่ยงจากข่าว
            news_risk = 'LOW'
            if market_context.news_impact_level >= 3:
                news_risk = 'HIGH'
            elif market_context.news_impact_level >= 2:
                news_risk = 'MEDIUM'
            
            risk_assessment['news_risk'] = news_risk
            
            # Risk/Reward ratio
            tp_pips = trading_params.get('take_profit_pips', 15)
            if sl_pips > 0:
                risk_assessment['risk_reward_ratio'] = tp_pips / sl_pips
            else:
                risk_assessment['risk_reward_ratio'] = 0
            
            # Overall risk level
            risk_factors = 0
            if market_risk == 'HIGH': risk_factors += 2
            elif market_risk == 'MEDIUM': risk_factors += 1
            
            if news_risk == 'HIGH': risk_factors += 2
            elif news_risk == 'MEDIUM': risk_factors += 1
            
            if market_context.confidence_score < 0.5: risk_factors += 1
            if risk_assessment['risk_percentage'] > 5: risk_factors += 1
            
            if risk_factors >= 4:
                overall_risk = 'VERY_HIGH'
            elif risk_factors >= 3:
                overall_risk = 'HIGH'
            elif risk_factors >= 2:
                overall_risk = 'MEDIUM'
            else:
                overall_risk = 'LOW'
            
            risk_assessment['overall_risk'] = overall_risk
            risk_assessment['risk_factors_count'] = risk_factors
            
            return risk_assessment
            
        except Exception as e:
            print(f"❌ Risk assessment error: {e}")
            return {'overall_risk': 'UNKNOWN', 'position_risk': 0}

    def _calculate_expected_outcome(self, action_decision: Dict, 
                                   trading_params: Dict) -> Dict[str, float]:
        """
        🎯 คำนวณผลที่คาดหวัง
        
        หน้าที่:
        1. คำนวณกำไร/ขาดทุนที่คาดหวัง
        2. คำนวณความน่าจะเป็นของผล
        3. คำนวณ expected value
        """
        try:
            expected = {}
            
            volume = trading_params.get('volume', 0.01)
            sl_pips = trading_params.get('stop_loss_pips', 30)
            tp_pips = trading_params.get('take_profit_pips', 15)
            
            # คำนวณกำไร/ขาดทุนเป็นเงิน
            max_loss = -(volume * sl_pips * 100) if sl_pips else 0
            max_profit = volume * tp_pips * 100 if tp_pips else 0
            
            expected['max_profit'] = max_profit
            expected['max_loss'] = max_loss
            
            # ประเมินความน่าจะเป็น (แบบง่าย)
            confidence = action_decision.get('confidence', 0.5)
            
            # ความน่าจะเป็นชนะ = confidence ปรับด้วยปัจจัยต่างๆ
            win_probability = confidence * 0.6  # Base 60% ของ confidence
            
            expected['win_probability'] = win_probability
            expected['loss_probability'] = 1 - win_probability
            
            # Expected value
            expected_value = (max_profit * win_probability) + (max_loss * (1 - win_probability))
            expected['expected_value'] = expected_value
            
            return expected
            
        except Exception as e:
            print(f"❌ Expected outcome calculation error: {e}")
            return {'expected_value': 0, 'max_profit': 0, 'max_loss': 0}

    def _create_stop_trading_decision(self) -> RecoveryDecision:
        """
        🛑 สร้างคำตัดสินใจหยุดเทรด
        """
        return RecoveryDecision(
            action=ActionType.HOLD,
            strategy_type=StrategyType.EMERGENCY_RECOVERY,
            volume=0.0,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            recovery_state=RecoveryState.FAILURE,
            confidence=0.1,
            reasoning=["Trading stopped due to risk limits"],
            warnings=["Risk limits exceeded - trading halted"],
            market_context=None,
            expected_outcome={},
            risk_assessment={'overall_risk': 'EXTREME'},
            timestamp=datetime.now()
        )

    def _create_error_decision(self, error_message: str) -> RecoveryDecision:
        """
        ❌ สร้างคำตัดสินใจเมื่อเกิดข้อผิดพลาด
        """
        return RecoveryDecision(
            action=ActionType.HOLD,
            strategy_type=StrategyType.EMERGENCY_RECOVERY,
            volume=0.0,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            recovery_state=RecoveryState.NORMAL,
            confidence=0.0,
            reasoning=[f"Error occurred: {error_message}"],
            warnings=["System error - holding position"],
            market_context=None,
            expected_outcome={},
            risk_assessment={'overall_risk': 'UNKNOWN'},
            timestamp=datetime.now()
        )

    def start_monitoring(self):
        """
        📡 เริ่มระบบ monitoring
        
        หน้าที่:
        1. เริ่ม background thread
        2. อัปเดตสถานะเป็นระยะ
        3. ตรวจสอบเงื่อนไขฉุกเฉิน
        """
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        
        def monitoring_loop():
            while self.monitoring_active:
                try:
                    # อัปเดตสถานะ
                    if self.current_session:
                        self._update_session_state()
                        
                        # ตรวจสอบเงื่อนไขฉุกเฉิน
                        if self._should_stop_trading():
                            print("🚨 Emergency condition detected!")
                            # สามารถเพิ่ม callback ที่นี่
                    
                    time.sleep(self.update_interval)
                    
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    time.sleep(10)  # รอนานกว่าเมื่อเกิดข้อผิดพลาด
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        print("📡 Recovery monitoring started")

    def stop_monitoring(self):
        """
        ⏹️ หยุดระบบ monitoring
        """
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("📡 Recovery monitoring stopped")

    def _update_performance_metrics(self, decision: RecoveryDecision, decision_time: float):
        """
        📊 อัปเดต performance metrics
        
        หน้าที่:
        - นับจำนวนการตัดสินใจ
        - คำนวณเวลาเฉลี่ย
        - ติดตามผลงาน
        """
        self.performance_metrics['total_decisions'] += 1
        
        # อัปเดตเวลาเฉลี่ยในการตัดสินใจ
        current_avg = self.performance_metrics['avg_decision_time']
        total_decisions = self.performance_metrics['total_decisions']
        
        new_avg = ((current_avg * (total_decisions - 1)) + decision_time) / total_decisions
        self.performance_metrics['avg_decision_time'] = new_avg

    def get_system_status(self) -> Dict:
        """
        📊 ดูสถานะระบบทั้งหมด
        
        หน้าที่:
        - รวบรวมข้อมูลสถานะทั้งหมด
        - ใช้สำหรับ GUI และ monitoring
        """
        try:
            status = {
                'timestamp': datetime.now(),
                'monitoring_active': self.monitoring_active,
                'session': None,
                'last_decision': None,
                'performance': self.performance_metrics
            }
            
            # Session status
            if self.current_session:
                status['session'] = {
                    'id': self.current_session.session_id,
                    'start_time': self.current_session.start_time,
                    'duration_minutes': (datetime.now() - self.current_session.start_time).total_seconds() / 60,
                    'initial_balance': self.current_session.initial_balance,
                    'current_balance': self.current_session.current_balance,
                    'total_pnl': self.current_session.total_pnl,
                    'max_drawdown': self.current_session.max_drawdown,
                    'recovery_state': self.current_session.recovery_state.value,
                    'current_strategy': self.current_session.current_strategy.value if self.current_session.current_strategy else None,
                    'total_trades': self.current_session.total_trades,
                    'win_rate': (self.current_session.winning_trades / max(1, self.current_session.total_trades)) * 100,
                    'active_positions': self.current_session.active_positions
                }
            
            # Last decision status
            if self.last_decision:
                status['last_decision'] = {
                    'action': self.last_decision.action.name,
                    'strategy': self.last_decision.strategy_type.value,
                    'confidence': self.last_decision.confidence,
                    'recovery_state': self.last_decision.recovery_state.value,
                    'timestamp': self.last_decision.timestamp,
                    'warnings': self.last_decision.warnings
                }
            
            return status
            
        except Exception as e:
            print(f"❌ Get system status error: {e}")
            return {'error': str(e), 'timestamp': datetime.now()}

    def get_recovery_summary(self) -> str:
        """
        📋 สรุปสถานะแบบย่อ
        
        หน้าที่:
        - สรุปข้อมูลสำคัญเป็นข้อความสั้นๆ
        - เหมาะสำหรับ log
        """
        try:
            if not self.current_session:
                return "🔄 No active recovery session"
            
            session = self.current_session
            
            summary = (f"🧠 Recovery: {session.recovery_state.value.upper()} | "
                      f"P&L: ${session.total_pnl:.2f} | "
                      f"DD: ${session.max_drawdown:.2f} | "
                      f"Trades: {session.total_trades} | "
                      f"Positions: {session.active_positions}")
            
            if self.last_decision:
                summary += f" | Last: {self.last_decision.action.name}"
            
            return summary
            
        except Exception as e:
            return f"❌ Summary error: {e}"

    def _log_system_status(self):
        """
        📋 แสดงสถานะระบบ
        """
        print("📋 Recovery Intelligence Brain Status:")
        print(f"   - Market Analyzer: Ready")
        print(f"   - Strategy Selector: Ready (8 strategies)")
        print(f"   - Recovery Thresholds:")
        print(f"     * Early: ${self.recovery_thresholds['early_recovery']}")
        print(f"     * Active: ${self.recovery_thresholds['active_recovery']}")
        print(f"     * Deep: ${self.recovery_thresholds['deep_recovery']}")
        print(f"     * Emergency: ${self.recovery_thresholds['emergency']}")
        print(f"   - Risk Limits:")
        print(f"     * Max Daily Loss: ${self.max_daily_loss}")
        print(f"     * Max Drawdown: {self.max_drawdown_percent}%")
        print(f"     * Max Positions: {self.max_positions}")

    def end_session(self) -> Dict:
        """
        🏁 จบ Recovery Session
        
        หน้าที่:
        1. หยุด monitoring
        2. บันทึกประวัติ
        3. สรุปผลงาน
        4. รีเซ็ตสถานะ
        """
        try:
            if not self.current_session:
                return {'error': 'No active session'}
            
            # หยุด monitoring
            self.stop_monitoring()
            
            # สร้างสรุปผลงาน
            session_summary = {
                'session_id': self.current_session.session_id,
                'start_time': self.current_session.start_time,
                'end_time': datetime.now(),
                'duration_hours': (datetime.now() - self.current_session.start_time).total_seconds() / 3600,
                'initial_balance': self.current_session.initial_balance,
                'final_balance': self.current_session.current_balance,
                'total_pnl': self.current_session.total_pnl,
                'max_drawdown': self.current_session.max_drawdown,
                'total_trades': self.current_session.total_trades,
                'win_rate': (self.current_session.winning_trades / max(1, self.current_session.total_trades)) * 100,
                'recovery_attempts': self.current_session.recovery_attempts,
                'successful_recoveries': self.current_session.successful_recoveries,
                'final_state': self.current_session.recovery_state.value
            }
            
            # เก็บประวัติ
            self.recovery_history.append(session_summary)
            
            # รีเซ็ตสถานะ
            self.current_session = None
            self.last_decision = None
            
            print(f"🏁 Recovery session ended")
            print(f"   Final P&L: ${session_summary['total_pnl']:.2f}")
            print(f"   Win Rate: {session_summary['win_rate']:.1f}%")
            
            return session_summary
            
        except Exception as e:
            print(f"❌ End session error: {e}")
            return {'error': str(e)}

    def force_strategy_change(self, new_strategy: StrategyType, reason: str = "Manual override"):
        """
        🔄 บังคับเปลี่ยนกลยุทธ์
        
        หน้าที่:
        - เปลี่ยนกลยุทธ์โดยไม่ขึ้นกับ market analysis
        - ใช้สำหรับการทดสอบหรือกรณีพิเศษ
        """
        if self.current_session:
            old_strategy = self.current_session.current_strategy
            self.current_session.current_strategy = new_strategy
            self.performance_metrics['strategy_changes'] += 1
            
            print(f"🔄 Strategy changed: {old_strategy} → {new_strategy.value}")
            print(f"   Reason: {reason}")

    def get_strategy_performance_report(self) -> Dict:
        """
        📊 รายงานประสิทธิภาพกลยุทธ์
        
        หน้าที่:
        - สรุปประสิทธิภาพของแต่ละกลยุทธ์
        - ใช้สำหรับการปรับปรุงระบบ
        """
        try:
            report = {
                'total_sessions': len(self.recovery_history),
                'current_session': None,
                'strategy_usage': {},
                'success_rates': {},
                'avg_performance': {}
            }
            
            # ข้อมูล current session
            if self.current_session:
                report['current_session'] = {
                    'state': self.current_session.recovery_state.value,
                    'pnl': self.current_session.total_pnl,
                    'trades': self.current_session.total_trades,
                    'strategy': self.current_session.current_strategy.value if self.current_session.current_strategy else None
                }
            
            # วิเคราะห์ประวัติ
            strategy_stats = {}
            for session in self.recovery_history:
                final_state = session.get('final_state', 'unknown')
                # Note: ในเวอร์ชันนี้ยังไม่ได้เก็บ strategy ใน history
                # ควรเพิ่มในอนาคต
            
            report['performance_metrics'] = self.performance_metrics
            
            return report
            
        except Exception as e:
            print(f"❌ Performance report error: {e}")
            return {'error': str(e)}

    def update_strategy_feedback(self, strategy_type: StrategyType, 
                                performance_data: Dict):
        """
        📈 อัปเดตผลตอบรับของกลยุทธ์
        
        หน้าที่:
        - รับ feedback จากผลการเทรด
        - ส่งต่อไปยัง StrategySelector
        - ปรับปรุงการเลือกกลยุทธ์ในอนาคต
        """
        try:
            self.strategy_selector.update_strategy_performance(
                strategy_type, performance_data
            )
            
            print(f"📈 Strategy feedback updated for {strategy_type.value}")
            print(f"   Performance data: {performance_data}")
            
        except Exception as e:
            print(f"❌ Strategy feedback error: {e}")

    def get_market_insights(self) -> Dict:
        """
        💡 ข้อมูลเชิงลึกเกี่ยวกับตลาด
        
        หน้าที่:
        - ดึงข้อมูล market analysis ล่าสุด
        - สรุปเป็นรูปแบบที่เข้าใจง่าย
        - ใช้สำหรับ GUI display
        """
        try:
            market_context = self.market_analyzer.analyze_market()
            
            insights = {
                'market_regime': market_context.regime.value,
                'trading_session': market_context.session.value,
                'volatility_level': self._classify_volatility(market_context.volatility_score),
                'trend_direction': self._classify_trend(market_context.trend_strength),
                'confidence_level': self._classify_confidence(market_context.confidence_score),
                'recommended_strategies': market_context.recommended_strategies[:3],
                'warnings': [],
                'opportunities': []
            }
            
            # เพิ่มคำเตือนและโอกาส
            if market_context.volatility_score > 80:
                insights['warnings'].append("High volatility - increased risk")
            
            if market_context.confidence_score > 0.8:
                insights['opportunities'].append("High confidence signals")
            
            if market_context.news_impact_level >= 3:
                insights['warnings'].append("High impact news expected")
            
            return insights
            
        except Exception as e:
            print(f"❌ Market insights error: {e}")
            return {'error': str(e)}

    def _classify_volatility(self, score: float) -> str:
        """จำแนกระดับความผันผวน"""
        if score > 80: return "Very High"
        elif score > 60: return "High"
        elif score > 40: return "Medium"
        elif score > 20: return "Low"
        else: return "Very Low"

    def _classify_trend(self, strength: float) -> str:
        """จำแนกทิศทางเทรนด์"""
        if strength > 60: return "Strong Bullish"
        elif strength > 20: return "Weak Bullish"
        elif strength > -20: return "Sideways"
        elif strength > -60: return "Weak Bearish"
        else: return "Strong Bearish"

    def _classify_confidence(self, confidence: float) -> str:
        """จำแนกระดับความเชื่อมั่น"""
        if confidence > 0.8: return "Very High"
        elif confidence > 0.6: return "High"
        elif confidence > 0.4: return "Medium"
        elif confidence > 0.2: return "Low"
        else: return "Very Low"

# Helper functions สำหรับใช้งานง่าย
def create_recovery_brain(mt5_interface, config: Dict = None) -> RecoveryIntelligence:
    """
    🚀 ฟังก์ชันช่วยสำหรับสร้าง Recovery Brain
    
    Usage:
        brain = create_recovery_brain(mt5_interface, config)
        session_id = brain.start_recovery_session()
        decision = brain.make_recovery_decision(current_positions)
    """
    return RecoveryIntelligence(mt5_interface, config)

def quick_recovery_decision(mt5_interface, config: Dict = None, 
                          current_positions: List = None) -> RecoveryDecision:
    """
    ⚡ ฟังก์ชันช่วยสำหรับตัดสินใจแบบเร็ว
    
    Usage:
        decision = quick_recovery_decision(mt5_interface, config, positions)
        print(f"Action: {decision.action.name}, Strategy: {decision.strategy_type.value}")
    """
    brain = RecoveryIntelligence(mt5_interface, config)
    brain.start_recovery_session()
    return brain.make_recovery_decision(current_positions)

# Example usage and testing functions
if __name__ == "__main__":
    # ตัวอย่างการใช้งาน (สำหรับทดสอบ)
    print("🧠 Recovery Intelligence Brain - Example Usage")
    print("This module requires MT5 interface to run properly")
    print("Import this module and use create_recovery_brain() function")
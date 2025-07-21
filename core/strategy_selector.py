# core/strategy_selector.py - AI Strategy Selection System

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json

# Import จาก market_analyzer
try:
    from .market_analyzer import MarketContext, MarketRegime, TradingSession
except ImportError:
    from market_analyzer import MarketContext, MarketRegime, TradingSession

class StrategyType(Enum):
    """
    🔄 ประเภทกลยุทธ์แก้ไม้ทั้งหมด 8 แบบ
    """
    CONSERVATIVE_MARTINGALE = "conservative_martingale"    # แก้ไม้แบบระมัดระวัง
    AGGRESSIVE_GRID = "aggressive_grid"                    # แก้ไม้แบบกริดก้าวร้าว
    HEDGING_RECOVERY = "hedging_recovery"                  # แก้ไม้แบบ hedge
    BREAKOUT_RECOVERY = "breakout_recovery"                # แก้ไม้แบบ breakout
    MEAN_REVERSION = "mean_reversion"                      # แก้ไม้แบบกลับค่าเฉลี่ย
    MOMENTUM_RECOVERY = "momentum_recovery"                # แก้ไม้แบบ momentum
    NEWS_BASED = "news_based"                             # แก้ไม้ตามข่าว
    EMERGENCY_RECOVERY = "emergency_recovery"              # แก้ไม้ฉุกเฉิน

class RiskLevel(Enum):
    """
    ⚠️ ระดับความเสี่ยงของกลยุทธ์
    """
    VERY_LOW = 1      # ความเสี่ยงต่ำมาก
    LOW = 2           # ความเสี่ยงต่ำ
    MEDIUM = 3        # ความเสี่ยงปานกลาง
    HIGH = 4          # ความเสี่ยงสูง
    VERY_HIGH = 5     # ความเสี่ยงสูงมาก

@dataclass
class StrategyConfig:
    """
    ⚙️ การกำหนดค่ากลยุทธ์ - เก็บพารามิเตอร์ของแต่ละกลยุทธ์
    """
    strategy_type: StrategyType              # ประเภทกลยุทธ์
    risk_level: RiskLevel                    # ระดับความเสี่ยง
    base_lot_multiplier: float               # ตัวคูณขนาด lot พื้นฐาน
    max_positions: int                       # จำนวนตำแหน่งสูงสุด
    recovery_multiplier: float               # ตัวคูณการแก้ไม้
    stop_loss_pips: Optional[int]            # Stop loss (pips)
    take_profit_pips: Optional[int]          # Take profit (pips)
    max_drawdown_threshold: float            # เกณฑ์ drawdown สูงสุด (%)
    suitable_regimes: List[MarketRegime]     # สภาวะตลาดที่เหมาะสม
    suitable_sessions: List[TradingSession]  # ช่วงเวลาที่เหมาะสม
    min_confidence: float                    # Confidence ต่ำสุดที่ใช้ได้
    description: str                         # คำอธิบายกลยุทธ์

@dataclass  
class StrategySelection:
    """
    🎯 ผลการเลือกกลยุทธ์ - ข้อมูลกลยุทธ์ที่ AI เลือก
    """
    primary_strategy: StrategyConfig         # กลยุทธ์หลัก
    backup_strategy: StrategyConfig          # กลยุทธ์สำรอง
    confidence_score: float                  # ความเชื่อมั่นในการเลือก (0-1)
    risk_adjustment: float                   # การปรับความเสี่ยง (0.5-2.0)
    position_sizing: Dict[str, float]        # การกำหนดขนาดตำแหน่ง
    market_context: MarketContext            # บริบทตลาดที่ใช้ตัดสินใจ
    selection_reasons: List[str]             # เหตุผลการเลือก
    warnings: List[str]                      # คำเตือนหรือข้อควรระวัง
    timestamp: datetime                      # เวลาที่เลือก

class StrategySelector:
    """
    🧠 ตัวเลือกกลยุทธ์อัจฉริยะ - ระบบเลือกกลยุทธ์แก้ไม้ที่เหมาะสมที่สุด
    
    หน้าที่หลัก:
    1. วิเคราะห์ MarketContext จาก MarketAnalyzer
    2. เปรียบเทียบกลยุทธ์ทั้งหมดกับสถานการณ์ปัจจุบัน
    3. คำนวณคะแนนความเหมาะสมของแต่ละกลยุทธ์
    4. เลือกกลยุทธ์หลักและสำรองที่ดีที่สุด
    5. ปรับพารามิเตอร์ตามสภาวะตลาด
    6. จัดการการเปลี่ยนกลยุทธ์แบบไดนามิก
    """
    
    def __init__(self, config: Dict = None):
        """
        🏗️ เริ่มต้นระบบเลือกกลยุทธ์
        
        หน้าที่:
        - โหลดการกำหนดค่ากลยุทธ์ทั้งหมด
        - ตั้งค่าเกณฑ์การตัดสินใจ
        - เตรียมระบบ scoring
        """
        print("🎯 Initializing Strategy Selector...")
        
        self.config = config or {}
        
        # การกำหนดค่าพื้นฐาน
        self.base_lot_size = self.config.get('lot_size', 0.01)
        self.max_daily_risk = self.config.get('max_daily_risk', 10.0)
        self.account_balance = self.config.get('balance', 10000.0)
        
        # สร้างกลยุทธ์ทั้งหมด
        self.strategies = self._create_all_strategies()
        
        # ประวัติการเลือกกลยุทธ์
        self.selection_history = []
        self.current_strategy = None
        self.strategy_performance = {}
        
        # ระบบ scoring weights
        self.scoring_weights = {
            'regime_match': 0.25,      # ความเข้ากับ market regime
            'session_match': 0.20,     # ความเข้ากับ trading session  
            'risk_appetite': 0.20,     # ความเหมาะสมของความเสี่ยง
            'confidence_level': 0.15,  # ความเชื่อมั่นของ market analysis
            'performance_history': 0.20 # ประสิทธิภาพในอดีต
        }
        
        print("✅ Strategy Selector initialized with 8 strategies")
        self._log_available_strategies()

    def _create_all_strategies(self) -> Dict[StrategyType, StrategyConfig]:
        """
        🏭 สร้างกลยุทธ์ทั้งหมด - กำหนดพารามิเตอร์ทุกกลยุทธ์
        
        หน้าที่:
        1. สร้าง StrategyConfig สำหรับทุกกลยุทธ์
        2. กำหนดพารามิเตอร์ที่เหมาะสมแต่ละแบบ
        3. ระบุ market regime และ session ที่เหมาะสม
        4. ตั้งค่าการจัดการความเสี่ยง
        """
        strategies = {}
        
        # 1. Conservative Martingale - แก้ไม้แบบระมัดระวัง
        strategies[StrategyType.CONSERVATIVE_MARTINGALE] = StrategyConfig(
            strategy_type=StrategyType.CONSERVATIVE_MARTINGALE,
            risk_level=RiskLevel.LOW,
            base_lot_multiplier=1.0,
            max_positions=3,
            recovery_multiplier=1.3,          # เพิ่ม 30% แต่ละครั้ง
            stop_loss_pips=30,
            take_profit_pips=15,
            max_drawdown_threshold=5.0,       # 5% max drawdown
            suitable_regimes=[MarketRegime.RANGING, MarketRegime.LOW_VOLATILITY],
            suitable_sessions=[TradingSession.ASIAN, TradingSession.QUIET],
            min_confidence=0.6,
            description="แก้ไม้แบบระมัดระวัง เพิ่มขนาดน้อยๆ เหมาะกับตลาดเงียบ"
        )
        
        # 2. Aggressive Grid - แก้ไม้แบบกริดก้าวร้าว
        strategies[StrategyType.AGGRESSIVE_GRID] = StrategyConfig(
            strategy_type=StrategyType.AGGRESSIVE_GRID,
            risk_level=RiskLevel.HIGH,
            base_lot_multiplier=1.2,
            max_positions=6,
            recovery_multiplier=1.8,          # เพิ่ม 80% แต่ละครั้ง
            stop_loss_pips=None,              # ไม่ใช้ stop loss
            take_profit_pips=10,
            max_drawdown_threshold=15.0,      # 15% max drawdown
            suitable_regimes=[MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY],
            suitable_sessions=[TradingSession.OVERLAP, TradingSession.LONDON],
            min_confidence=0.7,
            description="แก้ไม้แบบกริดก้าวร้าว เพิ่มขนาดเยอะ เหมาะกับตลาดผันผวน"
        )
        
        # 3. Hedging Recovery - แก้ไม้แบบ Hedge
        strategies[StrategyType.HEDGING_RECOVERY] = StrategyConfig(
            strategy_type=StrategyType.HEDGING_RECOVERY,
            risk_level=RiskLevel.MEDIUM,
            base_lot_multiplier=1.1,
            max_positions=4,
            recovery_multiplier=1.5,
            stop_loss_pips=40,
            take_profit_pips=20,
            max_drawdown_threshold=8.0,       # 8% max drawdown
            suitable_regimes=[MarketRegime.HIGH_VOLATILITY, MarketRegime.NEWS_IMPACT],
            suitable_sessions=[TradingSession.LONDON, TradingSession.NEW_YORK],
            min_confidence=0.5,
            description="แก้ไม้แบบ hedge เปิดทิศทางตรงข้าม เหมาะกับข่าวสำคัญ"
        )
        
        # 4. Breakout Recovery - แก้ไม้แบบ Breakout
        strategies[StrategyType.BREAKOUT_RECOVERY] = StrategyConfig(
            strategy_type=StrategyType.BREAKOUT_RECOVERY,
            risk_level=RiskLevel.MEDIUM,
            base_lot_multiplier=1.3,
            max_positions=3,
            recovery_multiplier=1.6,
            stop_loss_pips=25,
            take_profit_pips=50,
            max_drawdown_threshold=10.0,      # 10% max drawdown
            suitable_regimes=[MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
            suitable_sessions=[TradingSession.LONDON, TradingSession.OVERLAP],
            min_confidence=0.7,
            description="แก้ไม้แบบ breakout ตามทิศทางใหม่ เหมาะกับตลาดเทรนด์"
        )
        
        # 5. Mean Reversion - แก้ไม้แบบกลับค่าเฉลี่ย
        strategies[StrategyType.MEAN_REVERSION] = StrategyConfig(
            strategy_type=StrategyType.MEAN_REVERSION,
            risk_level=RiskLevel.MEDIUM,
            base_lot_multiplier=1.0,
            max_positions=4,
            recovery_multiplier=1.4,
            stop_loss_pips=35,
            take_profit_pips=25,
            max_drawdown_threshold=7.0,       # 7% max drawdown
            suitable_regimes=[MarketRegime.RANGING, MarketRegime.HIGH_VOLATILITY],
            suitable_sessions=[TradingSession.ASIAN, TradingSession.NEW_YORK],
            min_confidence=0.6,
            description="แก้ไม้แบบกลับค่าเฉลี่ย รอราคากลับช่วง เหมาะกับตลาดไซด์เวย์"
        )
        
        # 6. Momentum Recovery - แก้ไม้แบบ Momentum
        strategies[StrategyType.MOMENTUM_RECOVERY] = StrategyConfig(
            strategy_type=StrategyType.MOMENTUM_RECOVERY,
            risk_level=RiskLevel.HIGH,
            base_lot_multiplier=1.4,
            max_positions=3,
            recovery_multiplier=2.0,          # เพิ่ม 100% แต่ละครั้ง
            stop_loss_pips=20,
            take_profit_pips=40,
            max_drawdown_threshold=12.0,      # 12% max drawdown
            suitable_regimes=[MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN],
            suitable_sessions=[TradingSession.NEW_YORK, TradingSession.OVERLAP],
            min_confidence=0.8,
            description="แก้ไม้แบบ momentum ขยายทิศทางเดิม เหมาะกับเทรนด์แข็ง"
        )
        
        # 7. News Based - แก้ไม้ตามข่าว
        strategies[StrategyType.NEWS_BASED] = StrategyConfig(
            strategy_type=StrategyType.NEWS_BASED,
            risk_level=RiskLevel.VERY_HIGH,
            base_lot_multiplier=0.8,          # ขนาดเล็กกว่าปกติ
            max_positions=2,
            recovery_multiplier=2.5,          # แก้ไม้เร็วมาก
            stop_loss_pips=15,
            take_profit_pips=30,
            max_drawdown_threshold=20.0,      # 20% max drawdown
            suitable_regimes=[MarketRegime.NEWS_IMPACT, MarketRegime.HIGH_VOLATILITY],
            suitable_sessions=[TradingSession.LONDON, TradingSession.NEW_YORK],
            min_confidence=0.4,               # รับข่าวที่ไม่แน่ใจได้
            description="แก้ไม้ตามข่าวสำคัญ รวดเร็วและเสี่ยงสูง"
        )
        
        # 8. Emergency Recovery - แก้ไม้ฉุกเฉิน
        strategies[StrategyType.EMERGENCY_RECOVERY] = StrategyConfig(
            strategy_type=StrategyType.EMERGENCY_RECOVERY,
            risk_level=RiskLevel.VERY_LOW,
            base_lot_multiplier=0.5,          # ขนาดเล็กมาก
            max_positions=2,
            recovery_multiplier=1.2,          # แก้ไม้น้อยๆ
            stop_loss_pips=50,                # Stop loss กว้าง
            take_profit_pips=10,              # Take profit เล็ก
            max_drawdown_threshold=3.0,       # 3% max drawdown
            suitable_regimes=[MarketRegime.HIGH_VOLATILITY, MarketRegime.NEWS_IMPACT],
            suitable_sessions=list(TradingSession),  # ใช้ได้ทุก session
            min_confidence=0.2,               # รับสถานการณ์ไม่แน่ใจได้
            description="แก้ไม้ฉุกเฉิน ความปลอดภัยสูงสุด ใช้เมื่อสถานการณ์วิกฤต"
        )
        
        return strategies

    def select_strategy(self, market_context: MarketContext, 
                       current_drawdown: float = 0.0,
                       force_strategy: Optional[StrategyType] = None) -> StrategySelection:
        """
        🎯 เลือกกลยุทธ์หลัก - ฟังก์ชันหลักสำหรับเลือกกลยุทธ์ที่เหมาะสมที่สุด
        
        หน้าที่:
        1. คำนวณคะแนนความเหมาะสมของทุกกลยุทธ์
        2. เลือกกลยุทธ์หลักและสำรอง
        3. ปรับพารามิเตอร์ตามสภาวะตลาด
        4. สร้าง StrategySelection สมบูรณ์
        
        Args:
            market_context: ผลวิเคราะห์จาก MarketAnalyzer
            current_drawdown: Drawdown ปัจจุบัน (%)
            force_strategy: บังคับใช้กลยุทธ์เฉพาะ (สำหรับทดสอบ)
        """
        try:
            print(f"🎯 Selecting strategy for {market_context.regime.value} market...")
            
            # ถ้าบังคับใช้กลยุทธ์เฉพาะ
            if force_strategy and force_strategy in self.strategies:
                primary_strategy = self.strategies[force_strategy]
                backup_strategy = self._find_backup_strategy(primary_strategy, market_context)
                
                selection = self._create_selection_result(
                    primary_strategy, backup_strategy, market_context,
                    confidence_score=1.0, reasons=["Forced strategy selection"]
                )
                self._update_selection_history(selection)
                return selection
            
            # คำนวณคะแนนทุกกลยุทธ์
            strategy_scores = self._calculate_all_scores(market_context, current_drawdown)
            
            # เรียงลำดับตามคะแนน
            sorted_strategies = sorted(strategy_scores.items(), 
                                     key=lambda x: x[1]['total_score'], reverse=True)
            
            # เลือกกลยุทธ์หลัก (คะแนนสูงสุด)
            primary_type, primary_score = sorted_strategies[0]
            primary_strategy = self.strategies[primary_type]
            
            # เลือกกลยุทธ์สำรอง (คะแนนสูงที่สุดที่ไม่ใช่หลัก)
            backup_strategy = self._find_backup_strategy(primary_strategy, market_context)
            
            # สร้างผลลัพธ์
            confidence = min(1.0, primary_score['total_score'] / 100)
            reasons = self._generate_selection_reasons(primary_strategy, market_context, primary_score)
            warnings = self._generate_warnings(primary_strategy, market_context, current_drawdown)
            
            selection = self._create_selection_result(
                primary_strategy, backup_strategy, market_context,
                confidence, reasons, warnings
            )
            
            # บันทึกประวัติ
            self._update_selection_history(selection)
            self.current_strategy = selection
            
            print(f"✅ Selected: {primary_strategy.strategy_type.value} (confidence: {confidence:.2f})")
            return selection
            
        except Exception as e:
            print(f"❌ Strategy selection error: {e}")
            # คืนค่ากลยุทธ์ปลอดภัยในกรณีเกิดข้อผิดพลาด
            return self._get_safe_fallback_strategy(market_context)

    def _calculate_all_scores(self, market_context: MarketContext, 
                             current_drawdown: float) -> Dict[StrategyType, Dict]:
        """
        📊 คำนวณคะแนนทุกกลยุทธ์ - ประเมินความเหมาะสมของแต่ละกลยุทธ์
        
        หน้าที่:
        1. คำนวณคะแนนในแต่ละมิติ
        2. รวมคะแนนตาม weights
        3. พิจารณาข้อจำกัดต่างๆ
        4. ส่งคืนคะแนนรวมและรายละเอียด
        """
        scores = {}
        
        for strategy_type, strategy in self.strategies.items():
            try:
                # คำนวณคะแนนแต่ละมิติ
                regime_score = self._score_regime_match(strategy, market_context)
                session_score = self._score_session_match(strategy, market_context)
                risk_score = self._score_risk_appetite(strategy, market_context, current_drawdown)
                confidence_score = self._score_confidence_level(strategy, market_context)
                performance_score = self._score_performance_history(strategy)
                
                # คำนวณคะแนนรวม
                total_score = (
                    regime_score * self.scoring_weights['regime_match'] +
                    session_score * self.scoring_weights['session_match'] +
                    risk_score * self.scoring_weights['risk_appetite'] +
                    confidence_score * self.scoring_weights['confidence_level'] +
                    performance_score * self.scoring_weights['performance_history']
                )
                
                # ปรับคะแนนตามข้อจำกัด
                total_score = self._apply_constraints(total_score, strategy, market_context, current_drawdown)
                
                scores[strategy_type] = {
                    'total_score': total_score,
                    'regime_score': regime_score,
                    'session_score': session_score,
                    'risk_score': risk_score,
                    'confidence_score': confidence_score,
                    'performance_score': performance_score
                }
                
            except Exception as e:
                print(f"❌ Score calculation error for {strategy_type.value}: {e}")
                scores[strategy_type] = {'total_score': 0, 'error': str(e)}
        
        return scores

    def _score_regime_match(self, strategy: StrategyConfig, 
                           market_context: MarketContext) -> float:
        """
        🎯 คะแนนความเข้ากับ Market Regime
        
        หน้าที่:
        - ตรวจสอบว่า market regime ปัจจุบันเหมาะกับกลยุทธ์มั้ย
        - ให้คะแนน 0-100 ตามความเหมาะสม
        """
        if market_context.regime in strategy.suitable_regimes:
            return 100.0
        
        # คะแนนบางส่วนสำหรับ regime ที่ใกล้เคียง
        compatibility_map = {
            MarketRegime.TRENDING_UP: {
                MarketRegime.TRENDING_DOWN: 30,
                MarketRegime.HIGH_VOLATILITY: 50,
                MarketRegime.RANGING: 20
            },
            MarketRegime.TRENDING_DOWN: {
                MarketRegime.TRENDING_UP: 30,
                MarketRegime.HIGH_VOLATILITY: 50,
                MarketRegime.RANGING: 20
            },
            MarketRegime.RANGING: {
                MarketRegime.LOW_VOLATILITY: 80,
                MarketRegime.HIGH_VOLATILITY: 40
            },
            MarketRegime.HIGH_VOLATILITY: {
                MarketRegime.NEWS_IMPACT: 70,
                MarketRegime.TRENDING_UP: 50,
                MarketRegime.TRENDING_DOWN: 50
            },
            MarketRegime.LOW_VOLATILITY: {
                MarketRegime.RANGING: 80,
            },
            MarketRegime.NEWS_IMPACT: {
                MarketRegime.HIGH_VOLATILITY: 70
            }
        }
        
        current_regime = market_context.regime
        if current_regime in compatibility_map:
            for suitable_regime in strategy.suitable_regimes:
                if suitable_regime in compatibility_map[current_regime]:
                    return compatibility_map[current_regime][suitable_regime]
        
        return 10.0  # คะแนนขั้นต่ำ

    def _score_session_match(self, strategy: StrategyConfig, 
                            market_context: MarketContext) -> float:
        """
        🌏 คะแนนความเข้ากับ Trading Session
        
        หน้าที่:
        - ตรวจสอบว่า session ปัจจุบันเหมาะกับกลยุทธ์มั้ย
        - ให้คะแนน 0-100 ตามความเหมาะสม
        """
        if market_context.session in strategy.suitable_sessions:
            return 100.0
        
        # คะแนนบางส่วนสำหรับ session ที่ใกล้เคียง
        session_compatibility = {
            TradingSession.LONDON: {
                TradingSession.OVERLAP: 80,
                TradingSession.NEW_YORK: 60
            },
            TradingSession.NEW_YORK: {
                TradingSession.OVERLAP: 80,
                TradingSession.LONDON: 60
            },
            TradingSession.OVERLAP: {
                TradingSession.LONDON: 90,
                TradingSession.NEW_YORK: 90
            },
            TradingSession.ASIAN: {
                TradingSession.QUIET: 70
            },
            TradingSession.QUIET: {
                TradingSession.ASIAN: 70
            }
        }
        
        current_session = market_context.session
        if current_session in session_compatibility:
            for suitable_session in strategy.suitable_sessions:
                if suitable_session in session_compatibility[current_session]:
                    return session_compatibility[current_session][suitable_session]
        
        return 20.0  # คะแนนขั้นต่ำ

    def _score_risk_appetite(self, strategy: StrategyConfig, 
                            market_context: MarketContext, 
                            current_drawdown: float) -> float:
        """
        ⚠️ คะแนนความเหมาะสมของความเสี่ยง
        
        หน้าที่:
        - ประเมินความเสี่ยงของกลยุทธ์เทียบกับสถานการณ์
        - ปรับคะแนนตาม drawdown ปัจจุบัน
        - คำนวณความเหมาะสมของระดับเสี่ยง
        """
        base_score = 50.0
        
        # ปรับคะแนนตาม drawdown ปัจจุบัน
        if abs(current_drawdown) > strategy.max_drawdown_threshold:
            # ถ้า drawdown เกินเกณฑ์ ลดคะแนน
            return 10.0
        
        # ปรับคะแนนตามความผันผวนของตลาด
        volatility_score = market_context.volatility_score
        
        if strategy.risk_level == RiskLevel.VERY_LOW:
            # กลยุทธ์ความเสี่ยงต่ำมาก เหมาะกับทุกสถานการณ์
            base_score = 80.0
            if volatility_score > 80:  # ตลาดผันผวนมาก
                base_score = 95.0  # ยิ่งเหมาะสม
                
        elif strategy.risk_level == RiskLevel.LOW:
            base_score = 70.0
            if volatility_score < 40:  # ตลาดเงียบ
                base_score = 85.0
                
        elif strategy.risk_level == RiskLevel.MEDIUM:
            base_score = 60.0
            if 30 < volatility_score < 70:  # ตลาดปกติ
                base_score = 80.0
                
        elif strategy.risk_level == RiskLevel.HIGH:
            base_score = 40.0
            if volatility_score > 60:  # ตลาดผันผวน
                base_score = 75.0
            if abs(current_drawdown) > 5.0:  # มี drawdown อยู่แล้ว
                base_score = 20.0  # ลดความเสี่ยง
                
        elif strategy.risk_level == RiskLevel.VERY_HIGH:
            base_score = 30.0
            if volatility_score > 80 and market_context.confidence_score > 0.8:
                base_score = 70.0  # เหมาะเมื่อมั่นใจและผันผวนสูง
            if abs(current_drawdown) > 3.0:  # มี drawdown เล็กน้อยก็ลดคะแนนแล้ว
                base_score = 10.0
        
        return min(100.0, max(0.0, base_score))

    def _score_confidence_level(self, strategy: StrategyConfig, 
                               market_context: MarketContext) -> float:
        """
        🎯 คะแนนระดับความเชื่อมั่น
        
        หน้าที่:
        - ตรวจสอบว่า confidence ของ market analysis เพียงพอมั้ย
        - ปรับคะแนนตาม min_confidence ของกลยุทธ์
        """
        market_confidence = market_context.confidence_score
        required_confidence = strategy.min_confidence
        
        if market_confidence >= required_confidence:
            # Confidence เพียงพอ ให้คะแนนตามระดับ confidence
            return min(100.0, (market_confidence / required_confidence) * 80.0)
        else:
            # Confidence ไม่เพียงพอ
            confidence_ratio = market_confidence / required_confidence
            return max(10.0, confidence_ratio * 50.0)

    def _score_performance_history(self, strategy: StrategyConfig) -> float:
        """
        📈 คะแนนประสิทธิภาพในอดีต
        
        หน้าที่:
        - ดูประสิทธิภาพการทำงานของกลยุทธ์ในอดีต
        - ให้คะแนนตามผลงาน win rate, profit factor
        """
        strategy_type = strategy.strategy_type
        
        if strategy_type not in self.strategy_performance:
            return 50.0  # คะแนนกลางสำหรับกลยุทธ์ใหม่
        
        performance = self.strategy_performance[strategy_type]
        
        # คำนวณคะแนนจากหลายปัจจัย
        win_rate = performance.get('win_rate', 0.5)
        profit_factor = performance.get('profit_factor', 1.0)
        avg_profit = performance.get('avg_profit', 0.0)
        max_drawdown = performance.get('max_drawdown', 0.0)
        
        # คะแนนย่อย
        win_rate_score = min(100, win_rate * 100)
        profit_factor_score = min(100, (profit_factor - 0.5) * 100)
        profit_score = max(0, min(100, (avg_profit + 50)))
        drawdown_score = max(0, 100 - abs(max_drawdown) * 2)
        
        # คะแนนรวม
        total_score = (win_rate_score * 0.3 + 
                      profit_factor_score * 0.3 + 
                      profit_score * 0.2 + 
                      drawdown_score * 0.2)
        
        return max(10.0, min(100.0, total_score))

    def _apply_constraints(self, base_score: float, strategy: StrategyConfig, 
                          market_context: MarketContext, current_drawdown: float) -> float:
        """
        🚫 ใช้ข้อจำกัดกับคะแนน - ปรับคะแนนตามข้อจำกัดต่างๆ
        
        หน้าที่:
        - ตรวจสอบข้อจำกัดที่แข็งตัว (hard constraints)
        - ลดคะแนนหรือตัดออกกลยุทธ์ที่ไม่เหมาะสม
        """
        adjusted_score = base_score
        
        # ข้อจำกัดด้าน drawdown
        if abs(current_drawdown) > strategy.max_drawdown_threshold:
            adjusted_score *= 0.1  # ลดคะแนนมาก
        
        # ข้อจำกัดด้าน confidence
        if market_context.confidence_score < strategy.min_confidence:
            confidence_penalty = market_context.confidence_score / strategy.min_confidence
            adjusted_score *= confidence_penalty
        
        # ข้อจำกัดด้านข่าว
        if market_context.news_impact_level >= 3:  # ข่าวผลกระทบสูง
            if strategy.risk_level.value >= 4:  # กลยุทธ์เสี่ยงสูง
                adjusted_score *= 0.3  # ลดคะแนนมาก
        
        # ข้อจำกัดด้านเวลา (หลีกเลี่ยงกลยุทธ์เสี่ยงสูงในเวลาเงียบ)
        if market_context.session == TradingSession.QUIET:
            if strategy.risk_level.value >= 4:
                adjusted_score *= 0.2
        
        return max(0.0, adjusted_score)

    def _find_backup_strategy(self, primary_strategy: StrategyConfig, 
                             market_context: MarketContext) -> StrategyConfig:
        """
        🔄 หากลยุทธ์สำรอง - เลือกกลยุทธ์สำรองที่เหมาะสม
        
        หน้าที่:
        - หากลยุทธ์ที่ต่างจากหลักแต่เหมาะสม
        - ควรเป็นกลยุทธ์ที่ความเสี่ยงต่างกัน
        """
        # ไม่เลือกกลยุทธ์เดียวกับหลัก
        backup_candidates = [s for s in self.strategies.values() 
                           if s.strategy_type != primary_strategy.strategy_type]
        
        # หากลยุทธ์ที่ risk level ต่างกัน
        preferred_risk_levels = []
        
        if primary_strategy.risk_level.value <= 2:  # หลักเป็นความเสี่ยงต่ำ
            preferred_risk_levels = [RiskLevel.MEDIUM, RiskLevel.HIGH]
        elif primary_strategy.risk_level.value >= 4:  # หลักเป็นความเสี่ยงสูง  
            preferred_risk_levels = [RiskLevel.LOW, RiskLevel.VERY_LOW]
        else:  # หลักเป็นความเสี่ยงปานกลาง
            preferred_risk_levels = [RiskLevel.LOW, RiskLevel.HIGH]
        
        # หาตัวเลือกที่ดีที่สุด
        best_backup = None
        best_score = 0
        
        for candidate in backup_candidates:
            score = 0
            
            # คะแนนพิเศษหาก risk level ตามที่ต้องการ
            if candidate.risk_level in preferred_risk_levels:
                score += 30
            
            # คะแนนความเหมาะสมกับ market regime
            if market_context.regime in candidate.suitable_regimes:
                score += 50
            
            # คะแนนความเหมาะสมกับ session
            if market_context.session in candidate.suitable_sessions:
                score += 20
            
            if score > best_score:
                best_score = score
                best_backup = candidate
        
        # ถ้าไม่เจอที่เหมาะสม ใช้ emergency recovery
        return best_backup or self.strategies[StrategyType.EMERGENCY_RECOVERY]

    def _create_selection_result(self, primary_strategy: StrategyConfig, 
                               backup_strategy: StrategyConfig,
                               market_context: MarketContext,
                               confidence_score: float,
                               reasons: List[str] = None,
                               warnings: List[str] = None) -> StrategySelection:
        """
        📋 สร้างผลลัพธ์การเลือกกลยุทธ์
        
        หน้าที่:
        - รวบรวมข้อมูลทั้งหมดเป็น StrategySelection
        - คำนวณ risk adjustment และ position sizing
        - เตรียมข้อมูลสำหรับ AI Agent
        """
        # คำนวณการปรับความเสี่ยง
        risk_adjustment = self._calculate_risk_adjustment(primary_strategy, market_context)
        
        # คำนวณการกำหนดขนาดตำแหน่ง
        position_sizing = self._calculate_position_sizing(primary_strategy, market_context, risk_adjustment)
        
        return StrategySelection(
            primary_strategy=primary_strategy,
            backup_strategy=backup_strategy,
            confidence_score=confidence_score,
            risk_adjustment=risk_adjustment,
            position_sizing=position_sizing,
            market_context=market_context,
            selection_reasons=reasons or [],
            warnings=warnings or [],
            timestamp=datetime.now()
        )

    def _calculate_risk_adjustment(self, strategy: StrategyConfig, 
                                  market_context: MarketContext) -> float:
        """
        ⚖️ คำนวณการปรับความเสี่ยง
        
        หน้าที่:
        - ปรับความเสี่ยงตามสภาวะตลาด
        - คืนค่า multiplier (0.5-2.0)
        """
        base_adjustment = 1.0
        
        # ปรับตาม volatility
        if market_context.volatility_score > 80:
            base_adjustment *= 0.7  # ลดความเสี่ยงเมื่อผันผวนสูง
        elif market_context.volatility_score < 30:
            base_adjustment *= 1.3  # เพิ่มความเสี่ยงเมื่อผันผวนต่ำ
        
        # ปรับตาม confidence
        confidence_multiplier = 0.5 + (market_context.confidence_score * 1.0)
        base_adjustment *= confidence_multiplier
        
        # ปรับตามข่าว
        if market_context.news_impact_level >= 3:
            base_adjustment *= 0.6  # ระมัดระวังข่าวสำคัญ
        
        # ปรับตาม session
        session_multipliers = {
            TradingSession.OVERLAP: 1.2,    # ปริมาณสูง เพิ่มความเสี่ยงได้
            TradingSession.LONDON: 1.1,
            TradingSession.NEW_YORK: 1.0,
            TradingSession.ASIAN: 0.9,
            TradingSession.QUIET: 0.7       # เวลาเงียบ ลดความเสี่ยง
        }
        
        base_adjustment *= session_multipliers.get(market_context.session, 1.0)
        
        # จำกัดช่วง 0.5-2.0
        return max(0.5, min(2.0, base_adjustment))

    def _calculate_position_sizing(self, strategy: StrategyConfig, 
                                  market_context: MarketContext,
                                  risk_adjustment: float) -> Dict[str, float]:
        """
        📏 คำนวณขนาดตำแหน่ง
        
        หน้าที่:
        - คำนวณขนาด lot แต่ละประเภท
        - ปรับตาม risk adjustment
        - คืนค่า dictionary ของขนาดต่างๆ
        """
        # ขนาดพื้นฐานจาก config
        base_lot = self.base_lot_size * strategy.base_lot_multiplier
        
        # ปรับด้วย risk adjustment
        adjusted_base_lot = base_lot * risk_adjustment
        
        # คำนวณขนาดต่างๆ
        return {
            'base_lot': max(0.01, round(adjusted_base_lot, 2)),
            'recovery_lot': max(0.01, round(adjusted_base_lot * strategy.recovery_multiplier, 2)),
            'max_total_lot': max(0.01, round(adjusted_base_lot * strategy.max_positions, 2)),
            'emergency_lot': max(0.01, round(adjusted_base_lot * 0.5, 2))
        }

    def _generate_selection_reasons(self, strategy: StrategyConfig, 
                                   market_context: MarketContext,
                                   score_details: Dict) -> List[str]:
        """
        📝 สร้างเหตุผลการเลือกกลยุทธ์
        
        หน้าที่:
        - อธิบายว่าทำไมเลือกกลยุทธ์นี้
        - ให้ข้อมูลสำหรับ log และ debug
        """
        reasons = []
        
        # เหตุผลหลัก
        reasons.append(f"Market regime: {market_context.regime.value}")
        reasons.append(f"Trading session: {market_context.session.value}")
        reasons.append(f"Strategy risk level: {strategy.risk_level.name}")
        
        # เหตุผลจากคะแนน
        if score_details.get('regime_score', 0) > 80:
            reasons.append("High compatibility with current market regime")
        
        if score_details.get('session_score', 0) > 80:
            reasons.append("Optimal trading session for this strategy")
        
        if score_details.get('confidence_score', 0) > 70:
            reasons.append("High market analysis confidence")
        
        if market_context.volatility_score > 70 and strategy.risk_level.value <= 2:
            reasons.append("Conservative approach for high volatility market")
        
        return reasons

    def _generate_warnings(self, strategy: StrategyConfig, 
                          market_context: MarketContext,
                          current_drawdown: float) -> List[str]:
        """
        ⚠️ สร้างคำเตือน
        
        หน้าที่:
        - ระบุความเสี่ยงที่ควรระวัง
        - แนะนำการปรับแต่งพารามิเตอร์
        """
        warnings = []
        
        # เตือนเรื่อง drawdown
        if abs(current_drawdown) > strategy.max_drawdown_threshold * 0.8:
            warnings.append("Approaching maximum drawdown threshold")
        
        # เตือนเรื่อง confidence
        if market_context.confidence_score < strategy.min_confidence * 1.2:
            warnings.append("Market analysis confidence is low")
        
        # เตือนเรื่องข่าว
        if market_context.news_impact_level >= 3:
            warnings.append("High impact news expected - increased caution advised")
        
        # เตือนเรื่อง volatility mismatch
        if (market_context.volatility_score > 80 and 
            strategy.risk_level.value >= 4):
            warnings.append("High risk strategy in volatile market - monitor closely")
        
        # เตือนเรื่อง session
        if (market_context.session == TradingSession.QUIET and 
            strategy.risk_level.value >= 3):
            warnings.append("Medium/high risk strategy during quiet session")
        
        return warnings

    def _get_safe_fallback_strategy(self, market_context: MarketContext) -> StrategySelection:
        """
        🛡️ กลยุทธ์สำรอง - ใช้เมื่อเกิดข้อผิดพลาด
        
        หน้าที่:
        - คืนค่ากลยุทธ์ที่ปลอดภัยที่สุด
        - ใช้ในกรณีฉุกเฉิน
        """
        safe_strategy = self.strategies[StrategyType.EMERGENCY_RECOVERY]
        backup_strategy = self.strategies[StrategyType.CONSERVATIVE_MARTINGALE]
        
        return self._create_selection_result(
            safe_strategy, backup_strategy, market_context,
            confidence_score=0.3,
            reasons=["Fallback strategy due to error"],
            warnings=["Using emergency fallback strategy"]
        )

    def _update_selection_history(self, selection: StrategySelection):
        """
        📚 อัปเดตประวัติการเลือกกลยุทธ์
        
        หน้าที่:
        - บันทึกการเลือกกลยุทธ์
        - เก็บประวัติสำหรับการวิเคราะห์
        """
        self.selection_history.append(selection)
        
        # เก็บประวัติไว้ 100 ครั้งล่าสุด
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-100:]

    def _log_available_strategies(self):
        """
        📋 แสดงกลยุทธ์ที่มี
        
        หน้าที่:
        - แสดงรายการกลยุทธ์ทั้งหมด
        - ใช้สำหรับ debug
        """
        print("📋 Available strategies:")
        for strategy_type, strategy in self.strategies.items():
            print(f"   - {strategy_type.value}: {strategy.description}")

    def update_strategy_performance(self, strategy_type: StrategyType, 
                                   performance_data: Dict):
        """
        📊 อัปเดตประสิทธิภาพกลยุทธ์
        
        หน้าที่:
        - บันทึกผลการทำงานของแต่ละกลยุทธ์
        - ใช้สำหรับปรับปรุงการเลือกในอนาคต
        """
        self.strategy_performance[strategy_type] = performance_data
        print(f"📊 Updated performance for {strategy_type.value}")

    def get_strategy_info(self, strategy_type: StrategyType) -> Dict:
        """
        📖 ดูข้อมูลกลยุทธ์
        
        หน้าที่:
        - คืนข้อมูลโดยละเอียดของกลยุทธ์
        - ใช้สำหรับ GUI หรือ debug
        """
        if strategy_type not in self.strategies:
            return {"error": "Strategy not found"}
        
        strategy = self.strategies[strategy_type]
        return {
            "type": strategy.strategy_type.value,
            "risk_level": strategy.risk_level.name,
            "description": strategy.description,
            "base_lot_multiplier": strategy.base_lot_multiplier,
            "max_positions": strategy.max_positions,
            "recovery_multiplier": strategy.recovery_multiplier,
            "suitable_regimes": [r.value for r in strategy.suitable_regimes],
            "suitable_sessions": [s.value for s in strategy.suitable_sessions]
        }

    def get_current_strategy_status(self) -> Dict:
        """
        📊 ดูสถานะกลยุทธ์ปัจจุบัน
        
        หน้าที่:
        - คืนข้อมูลกลยุทธ์ที่กำลังใช้
        - ใช้สำหรับ monitoring
        """
        if not self.current_strategy:
            return {"status": "No strategy selected"}
        
        strategy = self.current_strategy
        return {
            "primary_strategy": strategy.primary_strategy.strategy_type.value,
            "backup_strategy": strategy.backup_strategy.strategy_type.value,
            "confidence": strategy.confidence_score,
            "risk_adjustment": strategy.risk_adjustment,
            "position_sizing": strategy.position_sizing,
            "selected_at": strategy.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "warnings": strategy.warnings
        }

# Helper function สำหรับใช้งานง่าย
def quick_strategy_selection(market_analyzer, mt5_interface, 
                           config: Dict = None) -> StrategySelection:
    """
    🚀 ฟังก์ชันช่วยสำหรับเลือกกลยุทธ์แบบเร็ว
    
    Usage:
        selection = quick_strategy_selection(market_analyzer, mt5_interface)
        print(f"Selected: {selection.primary_strategy.strategy_type.value}")
    """
    # วิเคราะห์ตลาด
    market_context = market_analyzer.analyze_market()
    
    # เลือกกลยุทธ์
    selector = StrategySelector(config)
    return selector.select_strategy(market_context)
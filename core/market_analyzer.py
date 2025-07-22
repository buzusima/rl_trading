# core/market_analyzer.py - ระบบวิเคราะห์ตลาดสำหรับ AI Recovery Intelligence

import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class MarketRegime(Enum):
    """
    📊 ประเภทสภาวะตลาด - ใช้สำหรับจำแนกสถานการณ์ตลาด
    """
    TRENDING_UP = "trending_up"      # ตลาดเทรนด์ขึ้น
    TRENDING_DOWN = "trending_down"  # ตลาดเทรนด์ลง
    RANGING = "ranging"              # ตลาดเคลื่อนไหวในช่วง
    HIGH_VOLATILITY = "high_vol"     # ความผันผวนสูง
    LOW_VOLATILITY = "low_vol"       # ความผันผวนต่ำ
    NEWS_IMPACT = "news"             # มีผลกระทบข่าว

class TradingSession(Enum):
    """
    🌏 ช่วงเวลาตลาด - แต่ละช่วงมีลักษณะการเทรดต่างกัน
    """
    ASIAN = "asian"          # 02:00-10:00 GMT (เงียบ)
    LONDON = "london"        # 08:00-17:00 GMT (ผันผวน)
    NEW_YORK = "new_york"    # 13:00-22:00 GMT (ปริมาณสูง)
    OVERLAP = "overlap"      # 13:00-17:00 GMT (ปริมาณสูงสุด)
    QUIET = "quiet"          # เวลาเงียบ

@dataclass
class MarketContext:
    """
    📈 ข้อมูลบริบทตลาดรวม - เก็บข้อมูลการวิเคราะห์ทั้งหมด
    """
    regime: MarketRegime                    # สภาวะตลาดหลัก
    session: TradingSession                 # ช่วงเวลาตลาด
    volatility_score: float                 # คะแนนความผันผวน (0-100)
    trend_strength: float                   # ความแข็งแกร่งเทรนด์ (-100 ถึง 100)
    volume_profile: str                     # ระดับปริมาณ (low/medium/high)
    news_impact_level: int                  # ระดับผลกระทบข่าว (0-3)
    recommended_strategies: List[str]       # กลยุทธ์ที่แนะนำ
    confidence_score: float                 # ความเชื่อมั่นในการวิเคราะห์ (0-1)
    timestamp: datetime                     # เวลาที่วิเคราะห์

class MarketAnalyzer:
    """
    🧠 ตัวววิเคราะห์ตลาดหลัก - วิเคราะห์สภาวะตลาดเพื่อให้ AI เลือกกลยุทธ์
    
    หน้าที่หลัก:
    1. วิเคราะห์ความผันผวนตลาด
    2. ตรวจจับแนวโน้มและความแข็งแกร่ง
    3. จำแนกช่วงเวลาตลาด
    4. ประเมินผลกระทบข่าว
    5. แนะนำกลยุทธ์แก้ไม้ที่เหมาะสม
    """
    
    def __init__(self, mt5_interface=None, config: Dict = None):
        """
        🏗️ เริ่มต้นระบบวิเคราะห์ตลาด
        
        หน้าที่:
        - กำหนดพารามิเตอร์การวิเคราะห์
        - เตรียมข้อมูลพื้นฐาน
        - ตั้งค่าเกณฑ์ต่างๆ
        """
        print("🧠 Initializing Market Analyzer...")
        
        self.mt5_interface = mt5_interface
        self.config = config or {}
        
        # พารามิเตอร์การวิเคราะห์ความผันผวน
        self.volatility_periods = {
            'short': 14,    # ระยะสั้น (14 เทียน)
            'medium': 50,   # ระยะกลาง (50 เทียน)  
            'long': 200     # ระยะยาว (200 เทียน)
        }
        
        # เกณฑ์ความผันผวน (ATR เป็นเปอร์เซ็นต์ของราคา)
        self.volatility_thresholds = {
            'low': 0.5,     # < 0.5% = ความผันผวนต่ำ
            'medium': 1.0,  # 0.5-1.0% = ความผันผวนปกติ
            'high': 1.5     # > 1.5% = ความผันผวนสูง
        }
        
        # เกณฑ์ความแข็งแกร่งเทรนด์
        self.trend_thresholds = {
            'weak': 0.3,    # < 0.3 = เทรนด์อย่อย
            'medium': 0.6,  # 0.3-0.6 = เทรนด์ปานกลาง
            'strong': 0.8   # > 0.8 = เทรนด์แข็งแกร่ง
        }
        
        # กำหนดช่วงเวลาตลาด (GMT)
        self.session_times = {
            TradingSession.ASIAN: (time(2, 0), time(10, 0)),
            TradingSession.LONDON: (time(8, 0), time(17, 0)),
            TradingSession.NEW_YORK: (time(13, 0), time(22, 0)),
            TradingSession.OVERLAP: (time(13, 0), time(17, 0))
        }
        
        # Cache สำหรับเก็บข้อมูลวิเคราะห์
        self.analysis_cache = {}
        self.cache_duration = 300  # เก็บแคช 5 นาที
        
        print("✅ Market Analyzer initialized successfully")

    def analyze_market(self, symbol: str = "XAUUSD") -> MarketContext:
        """
        📊 วิเคราะห์ตลาดรวม - ฟังก์ชันหลักสำหรับวิเคราะห์สภาวะตลาด
        
        หน้าที่:
        1. รวบรวมข้อมูลตลาดปัจจุบัน
        2. วิเคราะห์ในมิติต่างๆ
        3. สรุปเป็นบริบทตลาดที่ชัดเจน
        4. แนะนำกลยุทธ์ที่เหมาะสม
        
        Returns:
            MarketContext: ข้อมูลการวิเคราะห์ครบชุด
        """
        try:
            print(f"📊 Analyzing market for {symbol}...")
            
            # ตรวจสอบแคช
            cache_key = f"{symbol}_{int(datetime.now().timestamp() // self.cache_duration)}"
            if cache_key in self.analysis_cache:
                print("💾 Using cached analysis")
                return self.analysis_cache[cache_key]
            
            # รวบรวมข้อมูลพื้นฐาน
            market_data = self._get_market_data(symbol)
            if not market_data:
                return self._create_default_context()
            
            # วิเคราะห์แต่ละมิติ
            volatility_analysis = self._analyze_volatility(market_data)
            trend_analysis = self._analyze_trend(market_data)
            session_analysis = self._analyze_session()
            volume_analysis = self._analyze_volume(market_data)
            news_analysis = self._analyze_news_impact()
            
            # สร้างบริบทตลาดรวม
            market_context = self._synthesize_context(
                volatility_analysis,
                trend_analysis,
                session_analysis,
                volume_analysis,
                news_analysis
            )
            
            # แนะนำกลยุทธ์
            market_context.recommended_strategies = self._recommend_strategies(market_context)
            
            # เก็บแคช
            self.analysis_cache[cache_key] = market_context
            
            print(f"✅ Market analysis complete: {market_context.regime.value}")
            return market_context
            
        except Exception as e:
            print(f"❌ Market analysis error: {e}")
            return self._create_default_context()


    def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """
        📈 ดึงข้อมูลตลาด - Real-time Version (แก้ไขแล้ว)
        """
        try:
            if not self.mt5_interface:
                print("⚠️ No MT5 interface available")
                return None
            
            # 🔥 ลดความต้องการข้อมูล จาก 200 → 50 เทียน
            required_candles = 50  # แทนที่ 200
            min_acceptable = 15    # แทนที่ 50
            
            print(f"📊 Requesting {required_candles} candles for {symbol}...")
            
            # ดึงข้อมูลเทียน M5
            rates = self.mt5_interface.get_rates(symbol, 5, required_candles)
            
            # 🔥 ผ่อนปรนเงื่อนไข
            if not rates:
                print("❌ No rates data received from MT5")
                return None
                
            actual_candles = len(rates)
            print(f"📊 Received {actual_candles} candles")
            
            if actual_candles < min_acceptable:
                print(f"❌ Insufficient data: {actual_candles} candles (need {min_acceptable}+ minimum)")
                return None
            
            # แสดงสถานะข้อมูล
            if actual_candles < required_candles:
                print(f"⚠️ Limited data: {actual_candles}/{required_candles} candles (acceptable)")
            else:
                print(f"✅ Sufficient data: {actual_candles} candles")
            
            # ดึงราคาปัจจุบัน
            current_price = self.mt5_interface.get_current_price(symbol)
            if not current_price:
                print("❌ Cannot get current price")
                return None
            
            print(f"💰 Current price: ${current_price['bid']:.2f}")
            
            # แปลงเป็น DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            # 🔥 คำนวณ indicators แบบ flexible
            df = self._calculate_basic_indicators(df)
            
            return {
                'rates': rates,
                'dataframe': df,
                'current_price': current_price,
                'symbol': symbol,
                'data_quality': len(df)
            }
            
        except Exception as e:
            print(f"❌ Get market data error: {e}")
            return None
    
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        📊 คำนวณ Technical Indicators พื้นฐาน
        
        หน้าที่:
        1. ATR (Average True Range) - วัดความผันผวน
        2. SMA/EMA - วัดเทรนด์
        3. RSI - วัด momentum
        4. Bollinger Bands - วัดช่วงราคา
        """
        try:
            data_length = len(df)
            print(f"📊 Calculating indicators with {data_length} candles")
            
            if data_length < 5:
                print("⚠️ Very limited data - using basic calculations")
                return df
            
            # 🔥 ปรับ periods ตามข้อมูลที่มี
            atr_period = min(14, max(3, data_length - 2))
            sma20_period = min(20, max(5, data_length - 2))
            sma50_period = min(50, max(10, data_length - 2))
            rsi_period = min(14, max(5, data_length - 2))
            bb_period = min(20, max(5, data_length - 2))
            
            print(f"📊 Using periods - ATR:{atr_period}, SMA20:{sma20_period}, SMA50:{sma50_period}")
            
            # ATR (Average True Range)
            try:
                df['high_low'] = df['high'] - df['low']
                df['high_close'] = abs(df['high'] - df['close'].shift())
                df['low_close'] = abs(df['low'] - df['close'].shift())
                df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
                df['atr_14'] = df['true_range'].rolling(atr_period).mean()
                print(f"✅ ATR calculated (period: {atr_period})")
            except Exception as e:
                print(f"⚠️ ATR calculation error: {e}")
                # Simple fallback
                df['atr_14'] = df['high'] - df['low']
            
            # Moving Averages
            try:
                df['sma_20'] = df['close'].rolling(sma20_period).mean()
                print(f"✅ SMA20 calculated (period: {sma20_period})")
            except Exception as e:
                print(f"⚠️ SMA20 calculation error: {e}")
                
            try:
                df['sma_50'] = df['close'].rolling(sma50_period).mean()
                print(f"✅ SMA50 calculated (period: {sma50_period})")
            except Exception as e:
                print(f"⚠️ SMA50 calculation error: {e}")
            
            # EMA
            try:
                ema12_span = min(12, max(3, data_length // 2))
                ema26_span = min(26, max(5, data_length - 1))
                df['ema_12'] = df['close'].ewm(span=ema12_span).mean()
                df['ema_26'] = df['close'].ewm(span=ema26_span).mean()
                print(f"✅ EMA calculated (spans: {ema12_span}, {ema26_span})")
            except Exception as e:
                print(f"⚠️ EMA calculation error: {e}")
            
            # RSI
            try:
                if data_length >= 10:
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
                    loss = -delta.where(delta < 0, 0).rolling(rsi_period).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    print(f"✅ RSI calculated (period: {rsi_period})")
            except Exception as e:
                print(f"⚠️ RSI calculation error: {e}")
            
            # Bollinger Bands
            try:
                df['bb_middle'] = df['close'].rolling(bb_period).mean()
                bb_std = df['close'].rolling(bb_period).std()
                df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
                df['bb_lower'] = df['bb_middle'] - (2 * bb_std)
                print(f"✅ Bollinger Bands calculated (period: {bb_period})")
            except Exception as e:
                print(f"⚠️ Bollinger Bands calculation error: {e}")
            
            print(f"✅ Indicators calculation completed")
            return df
            
        except Exception as e:
            print(f"❌ Calculate indicators error: {e}")
            return df

    def _analyze_volatility(self, market_data: Dict) -> Dict:
        """
        📈 วิเคราะห์ความผันผวน - ประเมินระดับความผันผวนของตลาด
        
        หน้าที่:
        1. คำนวณ ATR เปอร์เซ็นต์
        2. เปรียบเทียบกับค่าเฉลี่ย
        3. จำแนกระดับความผันผวน
        4. ประเมินแนวโน้มความผันผวน
        """
        try:
            df = market_data['dataframe']
            current_price = market_data['current_price']['bid']
            
            # ATR เปอร์เซ็นต์ปัจจุบัน
            current_atr = df['atr_14'].iloc[-1]
            atr_percentage = (current_atr / current_price) * 100
            
            # ATR เฉลี่ย 50 เทียน
            atr_mean_50 = df['atr_14'].tail(50).mean()
            atr_mean_percentage = (atr_mean_50 / current_price) * 100
            
            # เปรียบเทียบกับค่าปกติ
            volatility_ratio = atr_percentage / max(atr_mean_percentage, 0.1)
            
            # จำแนกระดับ
            if atr_percentage < self.volatility_thresholds['low']:
                level = "LOW"
                score = 25
            elif atr_percentage < self.volatility_thresholds['medium']:
                level = "MEDIUM"
                score = 50
            elif atr_percentage < self.volatility_thresholds['high']:
                level = "HIGH"
                score = 75
            else:
                level = "EXTREME"
                score = 100
            
            # แนวโน้ม (ATR เพิ่มขึ้นหรือลดลง)
            atr_trend = "INCREASING" if current_atr > atr_mean_50 else "DECREASING"
            
            return {
                'level': level,
                'score': score,
                'atr_percentage': atr_percentage,
                'volatility_ratio': volatility_ratio,
                'trend': atr_trend,
                'current_atr': current_atr
            }
            
        except Exception as e:
            print(f"❌ Volatility analysis error: {e}")
            return {'level': 'UNKNOWN', 'score': 50, 'atr_percentage': 1.0, 'volatility_ratio': 1.0, 'trend': 'STABLE'}

    def _analyze_trend(self, market_data: Dict) -> Dict:
        """
        📈 วิเคราะห์เทรนด์ - ประเมินทิศทางและความแข็งแกร่งของเทรนด์
        
        หน้าที่:
        1. เปรียบเทียบราคากับ SMA/EMA
        2. คำนวณ slope ของ moving average
        3. ประเมินความแข็งแกร่งเทรนด์
        4. ระบุทิศทางหลัก
        """
        try:
            df = market_data['dataframe']
            current_close = df['close'].iloc[-1]
            
            # เปรียบเทียบกับ Moving Averages
            sma_20 = df['sma_20'].iloc[-1]
            sma_50 = df['sma_50'].iloc[-1]
            ema_12 = df['ema_12'].iloc[-1]
            ema_26 = df['ema_26'].iloc[-1]
            
            # คำนวณระยะห่างจาก MA (เปอร์เซ็นต์)
            distance_sma20 = ((current_close - sma_20) / sma_20) * 100
            distance_sma50 = ((current_close - sma_50) / sma_50) * 100
            
            # Slope ของ SMA (ความชันในช่วง 10 เทียน)
            sma20_slope = (df['sma_20'].iloc[-1] - df['sma_20'].iloc[-10]) / df['sma_20'].iloc[-10] * 100
            sma50_slope = (df['sma_50'].iloc[-1] - df['sma_50'].iloc[-10]) / df['sma_50'].iloc[-10] * 100
            
            # MACD สำหรับ confirm trend
            macd_line = ema_12 - ema_26
            macd_signal = df['ema_12'].ewm(span=9).mean().iloc[-1] - df['ema_26'].ewm(span=9).mean().iloc[-1]
            macd_histogram = macd_line - macd_signal
            
            # ประเมินทิศทาง
            bullish_signals = 0
            bearish_signals = 0
            
            if current_close > sma_20: bullish_signals += 1
            else: bearish_signals += 1
            
            if current_close > sma_50: bullish_signals += 1
            else: bearish_signals += 1
            
            if sma20_slope > 0: bullish_signals += 1
            else: bearish_signals += 1
            
            if sma50_slope > 0: bullish_signals += 1
            else: bearish_signals += 1
            
            if macd_histogram > 0: bullish_signals += 1
            else: bearish_signals += 1
            
            # ประเมินความแข็งแกร่ง
            total_signals = bullish_signals + bearish_signals
            trend_strength = abs(bullish_signals - bearish_signals) / total_signals
            
            # ทิศทางหลัก
            if bullish_signals > bearish_signals:
                direction = "BULLISH"
                strength_score = (bullish_signals / total_signals) * 100
            elif bearish_signals > bullish_signals:
                direction = "BEARISH"
                strength_score = (bearish_signals / total_signals) * 100
            else:
                direction = "SIDEWAYS"
                strength_score = 50
            
            # จำแนกความแข็งแกร่ง
            if trend_strength < self.trend_thresholds['weak']:
                strength = "WEAK"
            elif trend_strength < self.trend_thresholds['medium']:
                strength = "MEDIUM"  
            else:
                strength = "STRONG"
            
            return {
                'direction': direction,
                'strength': strength,
                'strength_score': strength_score,
                'trend_strength': trend_strength,
                'distance_sma20': distance_sma20,
                'distance_sma50': distance_sma50,
                'sma20_slope': sma20_slope,
                'sma50_slope': sma50_slope,
                'macd_histogram': macd_histogram
            }
            
        except Exception as e:
            print(f"❌ Trend analysis error: {e}")
            return {'direction': 'SIDEWAYS', 'strength': 'WEAK', 'strength_score': 50, 'trend_strength': 0}

    def _analyze_session(self) -> Dict:
        """
        🌏 วิเคราะห์ช่วงเวลาตลาด - ระบุ session และลักษณะการเทรด
        
        หน้าที่:
        1. ระบุช่วงเวลาตลาดปัจจุบัน
        2. ประเมินลักษณะการเทรดของแต่ละ session
        3. คำนวณเวลาจนถึง session สำคัญ
        4. แนะนำกลยุทธ์ตาม session
        """
        try:
            now_gmt = datetime.utcnow().time()
            current_session = TradingSession.QUIET  # default
            
            # ระบุ session ปัจจุบัน
            for session, (start_time, end_time) in self.session_times.items():
                if start_time <= now_gmt <= end_time:
                    current_session = session
                    break
            
            # ลักษณะของแต่ละ session
            session_characteristics = {
                TradingSession.ASIAN: {
                    'volume': 'LOW',
                    'volatility': 'LOW',
                    'preferred_strategies': ['conservative', 'ranging'],
                    'risk_level': 'LOW'
                },
                TradingSession.LONDON: {
                    'volume': 'HIGH',
                    'volatility': 'HIGH',
                    'preferred_strategies': ['breakout', 'trend_following'],
                    'risk_level': 'MEDIUM'
                },
                TradingSession.NEW_YORK: {
                    'volume': 'HIGH',
                    'volatility': 'MEDIUM',
                    'preferred_strategies': ['momentum', 'news_trading'],
                    'risk_level': 'MEDIUM'
                },
                TradingSession.OVERLAP: {
                    'volume': 'HIGHEST',
                    'volatility': 'HIGHEST',
                    'preferred_strategies': ['aggressive', 'scalping'],
                    'risk_level': 'HIGH'
                },
                TradingSession.QUIET: {
                    'volume': 'LOWEST',
                    'volatility': 'LOWEST',
                    'preferred_strategies': ['conservative', 'hold'],
                    'risk_level': 'LOWEST'
                }
            }
            
            current_char = session_characteristics[current_session]
            
            # คำนวณเวลาถึง session ถัดไป
            next_session_info = self._calculate_next_major_session()
            
            return {
                'current_session': current_session.value,
                'volume_level': current_char['volume'],
                'volatility_expectation': current_char['volatility'],
                'preferred_strategies': current_char['preferred_strategies'],
                'risk_level': current_char['risk_level'],
                'next_major_session': next_session_info,
                'gmt_time': now_gmt.strftime('%H:%M')
            }
            
        except Exception as e:
            print(f"❌ Session analysis error: {e}")
            return {
                'current_session': 'unknown',
                'volume_level': 'MEDIUM',
                'volatility_expectation': 'MEDIUM',
                'preferred_strategies': ['conservative'],
                'risk_level': 'MEDIUM'
            }

    def _analyze_volume(self, market_data: Dict) -> Dict:
        """
        📊 วิเคราะห์ปริมาณการเทรด - ประเมินปริมาณและแนวโน้ม
        
        หน้าที่:
        1. เปรียบเทียบปริมาณปัจจุบันกับค่าเฉลี่ย
        2. ระบุแนวโน้มปริมาณ
        3. ประเมินความสอดคล้องกับการเคลื่อนไหวราคา
        4. จำแนกระดับปริมาณ
        """
        try:
            df = market_data['dataframe']
            
            # ปริมาณเฉลี่ย
            recent_volume = df['tick_volume'].tail(10).mean()  # 10 เทียนล่าสุด
            average_volume = df['tick_volume'].tail(50).mean()  # 50 เทียนเฉลี่ย
            
            # อัตราส่วนปริมาณ
            volume_ratio = recent_volume / max(average_volume, 1)
            
            # แนวโน้มปริมาณ
            volume_trend_short = df['tick_volume'].tail(5).mean() / df['tick_volume'].tail(10).mean()
            
            # จำแนกระดับ
            if volume_ratio < 0.7:
                level = "LOW"
                score = 25
            elif volume_ratio < 1.3:
                level = "NORMAL"
                score = 50
            elif volume_ratio < 2.0:
                level = "HIGH"
                score = 75
            else:
                level = "VERY_HIGH"
                score = 100
            
            # แนวโน้ม
            if volume_trend_short > 1.1:
                trend = "INCREASING"
            elif volume_trend_short < 0.9:
                trend = "DECREASING"
            else:
                trend = "STABLE"
            
            return {
                'level': level,
                'score': score,
                'volume_ratio': volume_ratio,
                'trend': trend,
                'recent_vs_average': volume_ratio
            }
            
        except Exception as e:
            print(f"❌ Volume analysis error: {e}")
            return {'level': 'NORMAL', 'score': 50, 'volume_ratio': 1.0, 'trend': 'STABLE'}

    def _analyze_news_impact(self) -> Dict:
        """
        📰 วิเคราะห์ผลกระทบข่าว - ประเมินผลกระทบข่าวเศรษฐกิจ
        
        หน้าที่:
        1. ตรวจสอบเวลาข่าวสำคัญ
        2. ประเมินระดับผลกระทบ
        3. แนะนำการปรับกลยุทธ์
        4. คำนวณเวลาถึงข่าวถัดไป
        
        Note: ในเวอร์ชันนี้ใช้การประเมินแบบง่าย
        """
        try:
            current_hour_gmt = datetime.utcnow().hour
            
            # เวลาข่าวสำคัญ (GMT)
            high_impact_hours = [8, 9, 13, 14, 15]  # London open, US data, US open
            medium_impact_hours = [1, 2, 10, 11, 16, 17]  # Asian data, other sessions
            
            # ประเมินระดับผลกระทบ
            if current_hour_gmt in high_impact_hours:
                impact_level = 3  # สูง
                risk_adjustment = "INCREASE_CAUTION"
                recommended_action = "WAIT_OR_CONSERVATIVE"
            elif current_hour_gmt in medium_impact_hours:
                impact_level = 2  # ปานกลาง
                risk_adjustment = "SLIGHT_CAUTION"
                recommended_action = "NORMAL_WITH_STOPS"
            else:
                impact_level = 1  # ต่ำ
                risk_adjustment = "NORMAL"
                recommended_action = "NORMAL_TRADING"
            
            # เวลาถึงข่าวสำคัญถัดไป
            next_high_impact = self._find_next_impact_hour(high_impact_hours)
            
            return {
                'impact_level': impact_level,
                'risk_adjustment': risk_adjustment,
                'recommended_action': recommended_action,
                'next_high_impact_hour': next_high_impact,
                'current_hour_gmt': current_hour_gmt
            }
            
        except Exception as e:
            print(f"❌ News analysis error: {e}")
            return {
                'impact_level': 1,
                'risk_adjustment': 'NORMAL',
                'recommended_action': 'NORMAL_TRADING'
            }

    def _synthesize_context(self, volatility: Dict, trend: Dict, session: Dict, 
                           volume: Dict, news: Dict) -> MarketContext:
        """
        🧩 รวมบริบทตลาด - สังเคราะห์การวิเคราะห์ทั้งหมดเป็นบริบทเดียว
        
        หน้าที่:
        1. รวมผลการวิเคราะห์จากทุกมิติ
        2. ประเมินสภาวะตลาดหลัก
        3. คำนวณคะแนนความเชื่อมั่น
        4. สร้าง MarketContext สมบูรณ์
        """
        try:
            # ระบุสภาวะตลาดหลัก
            regime = self._determine_market_regime(volatility, trend, volume)
            
            # ระบุช่วงเวลาตลาด
            session_enum = getattr(TradingSession, session['current_session'].upper(), TradingSession.QUIET)
            
            # คำนวณคะแนนรวม
            volatility_score = volatility['score']
            trend_score = trend['strength_score']
            volume_score = volume['score']
            
            # ปรับคะแนนตามข่าว
            news_multiplier = {1: 1.0, 2: 0.8, 3: 0.6}[news['impact_level']]
            adjusted_confidence = (volatility_score + trend_score + volume_score) / 3 * news_multiplier / 100
            
            # ระบุปริมาณการเทรด
            volume_profile = volume['level'].lower()
            
            context = MarketContext(
                regime=regime,
                session=session_enum,
                volatility_score=volatility_score,
                trend_strength=trend['strength_score'],
                volume_profile=volume_profile,
                news_impact_level=news['impact_level'],
                recommended_strategies=[],  # จะเติมใน _recommend_strategies
                confidence_score=max(0.1, min(1.0, adjusted_confidence)),
                timestamp=datetime.now()
            )
            
            return context
            
        except Exception as e:
            print(f"❌ Context synthesis error: {e}")
            return self._create_default_context()

    def _determine_market_regime(self, volatility: Dict, trend: Dict, volume: Dict) -> MarketRegime:
        """
        🎯 ระบุสภาวะตลาดหลัก - จำแนกตลาดเป็นประเภทต่างๆ
        
        หน้าที่:
        1. พิจารณาความผันผวน เทรนด์ และปริมาณ
        2. จำแนกตลาดตามลักษณะหลัก
        3. ใช้ logic rules สำหรับการตัดสินใจ
        """
        try:
            vol_level = volatility['level']
            trend_direction = trend['direction']
            trend_strength = trend['strength']
            volume_level = volume['level']
            
            # High/Low Volatility Regimes
            if vol_level in ['EXTREME', 'HIGH']:
                if volume_level in ['HIGH', 'VERY_HIGH']:
                    return MarketRegime.HIGH_VOLATILITY
                else:
                    return MarketRegime.NEWS_IMPACT
            
            # Low Volatility
            elif vol_level == 'LOW':
                if trend_strength == 'WEAK':
                    return MarketRegime.LOW_VOLATILITY
                else:
                    return MarketRegime.RANGING
            
            # Trending Markets
            elif trend_direction == 'BULLISH' and trend_strength in ['MEDIUM', 'STRONG']:
                return MarketRegime.TRENDING_UP
            elif trend_direction == 'BEARISH' and trend_strength in ['MEDIUM', 'STRONG']:
                return MarketRegime.TRENDING_DOWN
            
            # Default to ranging
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            print(f"❌ Regime determination error: {e}")
            return MarketRegime.RANGING

    def _recommend_strategies(self, context: MarketContext) -> List[str]:
        """
        💡 แนะนำกลยุทธ์แก้ไม้ - เลือกกลยุทธ์ที่เหมาะสมกับสภาวะตลาด
        
        หน้าที่:
        1. วิเคราะห์ context ทั้งหมด
        2. จับคู่กับกลยุทธ์ที่เหมาะสม
        3. เรียงลำดับตามความเหมาะสม
        4. คืนค่ารายชื่อกลยุทธ์
        """
        try:
            recommendations = []
            
            # กลยุทธ์ตาม Market Regime
            regime_strategies = {
                MarketRegime.TRENDING_UP: ['momentum_recovery', 'breakout_recovery', 'conservative_martingale'],
                MarketRegime.TRENDING_DOWN: ['mean_reversion', 'hedging_recovery', 'conservative_martingale'],
                MarketRegime.RANGING: ['aggressive_grid', 'mean_reversion', 'hedging_recovery'],
                MarketRegime.HIGH_VOLATILITY: ['emergency_recovery', 'hedging_recovery', 'news_based'],
                MarketRegime.LOW_VOLATILITY: ['aggressive_grid', 'conservative_martingale', 'momentum_recovery'],
                MarketRegime.NEWS_IMPACT: ['news_based', 'emergency_recovery', 'hedging_recovery']
            }
            
            # เพิ่มกลยุทธ์ตาม regime
            if context.regime in regime_strategies:
                recommendations.extend(regime_strategies[context.regime])
            
            # ปรับตาม session
            session_preferences = {
                TradingSession.ASIAN: ['conservative_martingale', 'aggressive_grid'],
                TradingSession.LONDON: ['breakout_recovery', 'momentum_recovery'],
                TradingSession.NEW_YORK: ['news_based', 'momentum_recovery'],
                TradingSession.OVERLAP: ['aggressive_grid', 'momentum_recovery'],
                TradingSession.QUIET: ['conservative_martingale', 'hedging_recovery']
            }
            
            if context.session in session_preferences:
                session_recs = session_preferences[context.session]
                # เพิ่มกลยุทธ์ session โดยไม่ซ้ำ
                for strategy in session_recs:
                    if strategy not in recommendations:
                        recommendations.append(strategy)
            
            # ปรับตามระดับข่าว
            if context.news_impact_level >= 3:
                # ข่าวผลกระทบสูง - ใช้กลยุทธ์ระมัดระวัง
                safe_strategies = ['emergency_recovery', 'hedging_recovery', 'conservative_martingale']
                recommendations = [s for s in recommendations if s in safe_strategies] + safe_strategies
            
            # ปรับตามความผันผวน
            if context.volatility_score >= 80:
                # ความผันผวนสูงมาก - หลีกเลี่ยงกลยุทธ์ก้าวร้าว
                risky_strategies = ['aggressive_grid', 'momentum_recovery']
                recommendations = [s for s in recommendations if s not in risky_strategies]
            
            # ปรับตาม confidence
            if context.confidence_score < 0.5:
                # ความเชื่อมั่นต่ำ - ใช้กลยุทธ์ปลอดภัย
                recommendations = ['conservative_martingale', 'hedging_recovery', 'emergency_recovery']
            
            # ลบ duplicate และจำกัดจำนวน
            unique_recommendations = []
            for strategy in recommendations:
                if strategy not in unique_recommendations:
                    unique_recommendations.append(strategy)
            
            # คืนค่า top 3 strategies
            return unique_recommendations[:3] if unique_recommendations else ['conservative_martingale']
            
        except Exception as e:
            print(f"❌ Strategy recommendation error: {e}")
            return ['conservative_martingale']

    def _calculate_next_major_session(self) -> Dict:
        """
        ⏰ คำนวณ session ถัดไป - หาเวลาถึง session สำคัญถัดไป
        
        หน้าที่:
        1. หา session ที่มีผลกระทบสูงถัดไป
        2. คำนวณเวลาที่เหลือ
        3. แนะนำการเตรียมตัว
        """
        try:
            now_gmt = datetime.utcnow()
            current_time = now_gmt.time()
            
            # Major sessions ตามลำดับเวลา
            major_sessions = [
                ('LONDON_OPEN', time(8, 0)),
                ('LONDON_CLOSE', time(17, 0)),
                ('NY_OPEN', time(13, 0)),
                ('NY_CLOSE', time(22, 0))
            ]
            
            # หา session ถัดไป
            for session_name, session_time in major_sessions:
                if current_time < session_time:
                    # คำนวณเวลาที่เหลือ
                    today = now_gmt.date()
                    session_datetime = datetime.combine(today, session_time)
                    time_to_session = session_datetime - now_gmt
                    
                    return {
                        'session': session_name,
                        'time': session_time.strftime('%H:%M GMT'),
                        'minutes_remaining': int(time_to_session.total_seconds() / 60)
                    }
            
            # ถ้าผ่านทุก session แล้ว ให้ดู session แรกของวันถัดไป
            tomorrow = now_gmt.date() + pd.Timedelta(days=1)
            london_open = datetime.combine(tomorrow, time(8, 0))
            time_to_session = london_open - now_gmt
            
            return {
                'session': 'LONDON_OPEN_TOMORROW',
                'time': '08:00 GMT (Tomorrow)',
                'minutes_remaining': int(time_to_session.total_seconds() / 60)
            }
            
        except Exception as e:
            print(f"❌ Next session calculation error: {e}")
            return {'session': 'UNKNOWN', 'time': 'UNKNOWN', 'minutes_remaining': 0}

    def _find_next_impact_hour(self, impact_hours: List[int]) -> int:
        """
        📅 หาเวลาข่าวผลกระทบสูงถัดไป
        
        หน้าที่:
        - หาชั่วโมงถัดไปที่มีข่าวผลกระทบสูง
        - คืนค่าชั่วโมง GMT
        """
        try:
            current_hour = datetime.utcnow().hour
            
            # หาชั่วโมงถัดไปในวันนี้
            for hour in sorted(impact_hours):
                if hour > current_hour:
                    return hour
            
            # ถ้าไม่มี ให้ใช้ชั่วโมงแรกของวันถัดไป
            return min(impact_hours)
            
        except Exception as e:
            print(f"❌ Next impact hour error: {e}")
            return 8  # Default London open

    def _create_default_context(self) -> MarketContext:
        """
        🔧 สร้าง Context เริ่มต้น - ใช้เมื่อไม่สามารถวิเคราะห์ได้
        
        หน้าที่:
        - สร้าง MarketContext พื้นฐานที่ปลอดภัย
        - ใช้ค่าเริ่มต้นที่ระมัดระวัง
        """
        return MarketContext(
            regime=MarketRegime.RANGING,
            session=TradingSession.QUIET,
            volatility_score=50.0,
            trend_strength=50.0,
            volume_profile="normal",
            news_impact_level=1,
            recommended_strategies=['conservative_martingale'],
            confidence_score=0.3,
            timestamp=datetime.now()
        )

    def get_quick_analysis(self, symbol: str = "XAUUSD") -> str:
        """
        ⚡ วิเคราะห์แบบย่อ - ให้ข้อมูลสำคัญแบบสั้นๆ
        
        หน้าที่:
        1. วิเคราะห์ตลาดแบบเร็ว
        2. สรุปเป็นข้อความสั้นๆ
        3. เหมาะสำหรับ log หรือ display
        """
        try:
            context = self.analyze_market(symbol)
            
            summary = (f"📊 Market: {context.regime.value.upper()} | "
                      f"Session: {context.session.value.upper()} | "
                      f"Vol: {context.volatility_score:.0f} | "
                      f"Trend: {context.trend_strength:.0f} | "
                      f"Strategy: {context.recommended_strategies[0] if context.recommended_strategies else 'conservative'}")
            
            return summary
            
        except Exception as e:
            return f"❌ Quick analysis error: {e}"

    def clear_cache(self):
        """
        🗑️ ล้างแคช - ล้างข้อมูลแคชทั้งหมด
        
        หน้าที่:
        - ล้างแคชการวิเคราะห์
        - บังคับให้วิเคราะห์ใหม่ครั้งถัดไป
        """
        self.analysis_cache.clear()
        print("🗑️ Market analysis cache cleared")

# Helper function สำหรับใช้งานง่าย
def quick_market_check(mt5_interface, symbol="XAUUSD") -> MarketContext:
    """
    🚀 ฟังก์ชันช่วยสำหรับวิเคราะห์ตลาดแบบเร็ว
    
    Usage:
        context = quick_market_check(mt5_interface)
        print(f"Recommended strategy: {context.recommended_strategies[0]}")
    """
    analyzer = MarketAnalyzer(mt5_interface)
    return analyzer.analyze_market(symbol)
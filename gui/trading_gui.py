import tkinter as tk
from tkinter import ttk, messagebox
import threading
import json
import os
from datetime import datetime
import numpy as np  # ⭐ เพิ่มบรรทัดนี้
from environments import ConservativeEnvironment, AggressiveEnvironment, create_environment

class TradingGUI:
    
    def __init__(self):
        print("🎨 Initializing Recovery Trading GUI...")
        
        # Main window
        self.root = tk.Tk()
        self.root.title("🤖 AI Recovery Trading System")
        self.root.geometry("900x700")
        self.root.configure(bg='#2b2b2b')
        
        # Configure Thai fonts
        self.setup_fonts()
        
        # ⭐ เพิ่มตัวแปร Trading State ที่ขาดหายไป
        self.is_connected = False
        self.is_training = False
        self.is_trading = False
        
        # ⭐ เพิ่มตัวแปร Trading Session ที่ขาดหายไป
        self.current_step = 0
        self.trading_session_start = None
        self.session_start_balance = 0.0
        self.daily_pnl = 0.0
        
        # ⭐ เพิ่มตัวแปร Recovery State ที่ขาดหายไป
        self.in_recovery_mode = False
        self.recovery_level = 0
        self.recovery_start_pnl = 0.0
        self.recovery_target = 0.0
        
        # ⭐ เพิ่มตัวแปร Performance Tracking ที่ขาดหายไป
        self.total_trades_today = 0
        self.winning_trades_today = 0
        self.losing_trades_today = 0
        self.max_drawdown_today = 0.0
        
        # Core components
        self.mt5_interface = None
        self.environment = None
        self.agent = None
        self.data_loader = None
        
        # Configuration
        self.config = self.load_config()
        
        # Initialize GUI variables
        self.initialize_variables()
        
        # Threading
        self.training_thread = None
        self.trading_thread = None
        self.update_thread = None
        self.running = True
        
        # Setup GUI
        self.setup_gui()
        
        # Start update loop
        self.start_update_loop()
        
        print("✅ Recovery Trading GUI initialized successfully")

    def setup_fonts(self):
        """Setup Thai-friendly fonts"""
        import tkinter.font as tkFont
        
        # ✅ Thai-friendly fonts in order of preference
        thai_fonts = [
            ("Segoe UI", 10),           # Windows default - good Thai support
            ("Tahoma", 10),             # Excellent Thai support
            ("Microsoft Sans Serif", 10), # Windows fallback
            ("DejaVu Sans", 10),        # Linux
            ("SF Pro Display", 10),     # macOS
            ("Arial Unicode MS", 10),   # Universal fallback
            ("TkDefaultFont", 10)       # System default
        ]
        
        # Find the best available font
        self.default_font = None
        for font_name, size in thai_fonts:
            try:
                test_font = tkFont.Font(family=font_name, size=size)
                # Test if font exists by getting metrics
                test_font.metrics()
                self.default_font = (font_name, size)
                print(f"✅ Using font: {font_name}")
                break
            except:
                continue
        
        if not self.default_font:
            self.default_font = ("TkDefaultFont", 10)
            print("⚠️ Using system default font")
        
        # ✅ Create font objects for different UI elements (fixed for Python 3.8)
        font_family, base_size = self.default_font
        
        # Console font with fallback handling for older Python
        console_fonts = ['Consolas', 'Courier New', 'monospace', 'TkFixedFont']
        console_font_family = 'TkFixedFont'  # Default fallback
        
        for font_name in console_fonts:
            try:
                test_font = tkFont.Font(family=font_name, size=9)
                test_font.metrics()
                console_font_family = font_name
                break
            except:
                continue
        
        self.fonts = {
            'default': tkFont.Font(family=font_family, size=base_size),
            'header': tkFont.Font(family=font_family, size=base_size+2, weight='bold'),
            'small': tkFont.Font(family=font_family, size=base_size-1),
            'console': tkFont.Font(family=console_font_family, size=9),  # Fixed: removed fallback parameter
            'thai_label': tkFont.Font(family=font_family, size=base_size+1),  # Slightly larger for Thai
            'thai_comment': tkFont.Font(family=font_family, size=base_size-1)  # For comments
        }
        
        # ✅ Configure ttk styles for better Thai rendering
        style = ttk.Style()
        
        # Configure label styles
        style.configure('Thai.TLabel', font=self.fonts['thai_label'])
        style.configure('Comment.TLabel', font=self.fonts['thai_comment'], foreground='gray')
        style.configure('Header.TLabel', font=self.fonts['header'])
        
        # Configure button styles  
        style.configure('Thai.TButton', font=self.fonts['default'])
        
        # Configure entry styles
        style.configure('Thai.TEntry', font=self.fonts['default'])

    def initialize_variables(self):
        """Initialize all GUI variables รวม Risk Management"""
        # Basic config variables (เดิม)
        self.symbol_var = tk.StringVar(value=self.config.get('symbol', 'XAUUSD'))
        self.lot_size_var = tk.DoubleVar(value=self.config.get('lot_size', 0.01))
        self.max_positions_var = tk.IntVar(value=self.config.get('max_positions', 5))
        self.training_steps_var = tk.IntVar(value=self.config.get('training_steps', 100000))
        self.learning_rate_var = tk.DoubleVar(value=self.config.get('learning_rate', 0.0003))
        
        # Recovery config variables (เดิม)
        self.recovery_multiplier_var = tk.DoubleVar(value=self.config.get('recovery_multiplier', 1.5))
        self.recovery_threshold_var = tk.DoubleVar(value=self.config.get('recovery_threshold', -20.0))
        self.max_recovery_levels_var = tk.IntVar(value=self.config.get('max_recovery_levels', 3))
        
        # ⭐ เพิ่ม Risk Management Variables
        self.risk_per_trade_var = tk.DoubleVar(value=self.config.get('risk_per_trade', 2.0))  # 2% per trade
        self.max_daily_risk_var = tk.DoubleVar(value=self.config.get('max_daily_risk', 10.0))  # 10% per day
        self.max_lot_size_var = tk.DoubleVar(value=self.config.get('max_lot_size', 0.10))  # Max 0.10 lots
        self.stop_loss_pips_var = tk.IntVar(value=self.config.get('stop_loss_pips', 20))  # 20 pips SL
        
        # Auto-scroll variable (เดิม)
        self.auto_scroll_var = tk.BooleanVar(value=True)
    
        # ⭐ เพิ่ม Trading Mode Variables
        self.trading_mode_var = tk.StringVar(value="conservative")  # Default: Conservative
        self.current_environment_type = "conservative"
        self.ai_insights_window = None

    def load_config(self):
        """Load configuration รวม Risk Management settings"""
        default_config = {
            'symbol': 'XAUUSD.v',
            'lot_size': 0.02,
            'max_positions': 3,
            'training_steps': 100000,
            'learning_rate': 0.0003,
            # Recovery settings (เดิม)
            'recovery_multiplier': 1.4,
            'recovery_threshold': -35.0,
            'max_recovery_levels': 3,
            # ⭐ Risk Management settings ใหม่
            'risk_per_trade': 2.0,      # 2% risk per trade
            'max_daily_risk': 10.0,     # 10% max daily risk
            'max_lot_size': 0.10,       # Max 0.10 lots
            'stop_loss_pips': 20        # 20 pips stop loss
        }
        
        try:
            config_file = 'config/config.json'
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    print("✅ Configuration loaded (including risk management)")
        except Exception as e:
            print(f"⚠️ Config load error: {e}, using defaults")
        
        return default_config

    def setup_gui(self):
        """Setup GUI interface"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Main tab
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="🏠 Main")
        
        # Training tab
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="🎓 Training")
        
        # Logs tab
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="📋 Logs")
        
        # Setup individual tabs
        self.setup_main_tab()
        self.setup_training_tab()
        self.setup_logs_tab()

    def setup_main_tab(self):
        # === TRADING MODE SECTION === (เพิ่มใหม่)
        mode_frame = ttk.LabelFrame(self.main_frame, text="🎯 Trading Mode Selection")
        mode_frame.pack(fill='x', padx=10, pady=5)
        
        mode_controls = ttk.Frame(mode_frame)
        mode_controls.pack(fill='x', padx=10, pady=10)
        
        # Mode Radio Buttons
        conservative_radio = ttk.Radiobutton(
            mode_controls, 
            text="🛡️ Conservative Mode", 
            variable=self.trading_mode_var, 
            value="conservative",
            command=self.on_mode_change,
            style='Thai.TButton'
        )
        conservative_radio.pack(side='left', padx=5)
        
        aggressive_radio = ttk.Radiobutton(
            mode_controls, 
            text="⚡ Aggressive AI Mode", 
            variable=self.trading_mode_var, 
            value="aggressive",
            command=self.on_mode_change,
            style='Thai.TButton'
        )
        aggressive_radio.pack(side='left', padx=5)
        
        # Mode Description
        mode_desc_frame = ttk.Frame(mode_frame)
        mode_desc_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.mode_description_label = ttk.Label(
            mode_desc_frame, 
            text="🛡️ Conservative: ใช้ RL Agent เดิม (HOLD บ่อย, ปลอดภัย)",
            style='Comment.TLabel'
        )
        self.mode_description_label.pack(side='left')
        
        # AI Insights Button (เฉพาะ Aggressive Mode)
        self.ai_insights_btn = ttk.Button(
            mode_controls, 
            text="🧠 AI Insights", 
            command=self.show_ai_insights,
            state='disabled',  # เปิดเฉพาะ Aggressive Mode
            style='Thai.TButton'
        )
        self.ai_insights_btn.pack(side='right', padx=5)
        
        # Current Environment Status
        env_status_frame = ttk.Frame(mode_frame)
        env_status_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        ttk.Label(env_status_frame, text="Current Environment:", style='Thai.TLabel').pack(side='left')
        self.current_env_label = ttk.Label(env_status_frame, text="Conservative", foreground='blue', style='Thai.TLabel')
        self.current_env_label.pack(side='left', padx=(10, 0))
            # === CONNECTION SECTION ===
        conn_frame = ttk.LabelFrame(self.main_frame, text="🔗 MT5 Connection")
        conn_frame.pack(fill='x', padx=10, pady=5)
        
        conn_controls = ttk.Frame(conn_frame)
        conn_controls.pack(fill='x', padx=10, pady=10)
        
        # Connection status
        self.conn_status_label = ttk.Label(conn_controls, text="❌ Disconnected", foreground='red')
        self.conn_status_label.pack(side='left')
        
        # Connect button
        self.connect_btn = ttk.Button(conn_controls, text="🔌 Connect MT5", command=self.connect_mt5)
        self.connect_btn.pack(side='right', padx=5)
        
        # === ACCOUNT INFO SECTION ===
        account_frame = ttk.LabelFrame(self.main_frame, text="💰 Account Info")
        account_frame.pack(fill='x', padx=10, pady=5)
        
        account_grid = ttk.Frame(account_frame)
        account_grid.pack(fill='x', padx=10, pady=10)
        
        # Account details
        ttk.Label(account_grid, text="Balance:").grid(row=0, column=0, sticky='w')
        self.balance_label = ttk.Label(account_grid, text="$0.00")
        self.balance_label.grid(row=0, column=1, sticky='w', padx=(10, 0))
        
        ttk.Label(account_grid, text="Equity:").grid(row=1, column=0, sticky='w')
        self.equity_label = ttk.Label(account_grid, text="$0.00")
        self.equity_label.grid(row=1, column=1, sticky='w', padx=(10, 0))
        
        ttk.Label(account_grid, text="Positions:").grid(row=2, column=0, sticky='w')
        self.positions_label = ttk.Label(account_grid, text="0")
        self.positions_label.grid(row=2, column=1, sticky='w', padx=(10, 0))
        
        # === MODEL STATUS SECTION ===
        model_frame = ttk.LabelFrame(self.main_frame, text="🤖 AI Model Status")
        model_frame.pack(fill='x', padx=10, pady=5)
        
        model_controls = ttk.Frame(model_frame)
        model_controls.pack(fill='x', padx=10, pady=10)
        
        # Model status
        self.model_status_label = ttk.Label(model_controls, text="❌ No Model Loaded", foreground='red')
        self.model_status_label.pack(side='left')
        
        # Load model button
        self.load_model_btn = ttk.Button(model_controls, text="📥 Load Model", command=self.manual_load_model)
        self.load_model_btn.pack(side='right', padx=5)
        
        # === RECOVERY STATUS SECTION ===
        recovery_frame = ttk.LabelFrame(self.main_frame, text="🔄 Recovery Status")
        recovery_frame.pack(fill='x', padx=10, pady=5)
        
        recovery_grid = ttk.Frame(recovery_frame)
        recovery_grid.pack(fill='x', padx=10, pady=10)
        
        # Recovery mode status
        ttk.Label(recovery_grid, text="Recovery Mode:").grid(row=0, column=0, sticky='w')
        self.recovery_mode_label = ttk.Label(recovery_grid, text="🟢 Normal", foreground='green')
        self.recovery_mode_label.grid(row=0, column=1, sticky='w', padx=(10, 0))
        
        # Recovery level
        ttk.Label(recovery_grid, text="Recovery Level:").grid(row=1, column=0, sticky='w')
        self.recovery_level_label = ttk.Label(recovery_grid, text="0/3")
        self.recovery_level_label.grid(row=1, column=1, sticky='w', padx=(10, 0))
        
        # Total P&L
        ttk.Label(recovery_grid, text="Total P&L:").grid(row=2, column=0, sticky='w')
        self.total_pnl_label = ttk.Label(recovery_grid, text="$0.00")
        self.total_pnl_label.grid(row=2, column=1, sticky='w', padx=(10, 0))
        
        # Max Drawdown
        ttk.Label(recovery_grid, text="Max Drawdown:").grid(row=0, column=2, sticky='w', padx=(20, 0))
        self.max_drawdown_label = ttk.Label(recovery_grid, text="$0.00")
        self.max_drawdown_label.grid(row=0, column=3, sticky='w', padx=(10, 0))
        
        # Recovery Success Rate
        ttk.Label(recovery_grid, text="Recovery Success:").grid(row=1, column=2, sticky='w', padx=(20, 0))
        self.recovery_success_label = ttk.Label(recovery_grid, text="0/0 (0%)")
        self.recovery_success_label.grid(row=1, column=3, sticky='w', padx=(10, 0))
        
        # Win Rate
        ttk.Label(recovery_grid, text="Win Rate:").grid(row=2, column=2, sticky='w', padx=(20, 0))
        self.win_rate_label = ttk.Label(recovery_grid, text="0/0 (0%)")
        self.win_rate_label.grid(row=2, column=3, sticky='w', padx=(10, 0))
        
        # === TRADING CONTROLS SECTION ===
        trading_frame = ttk.LabelFrame(self.main_frame, text="⚡ Recovery Trading Controls")
        trading_frame.pack(fill='x', padx=10, pady=5)
        
        trading_controls = ttk.Frame(trading_frame)
        trading_controls.pack(fill='x', padx=10, pady=10)
        
        # Start/Stop trading
        self.start_trading_btn = ttk.Button(trading_controls, text="🚀 Start Recovery Trading", 
                                           command=self.start_trading, state='disabled')
        self.start_trading_btn.pack(side='left', padx=5)
        
        self.stop_trading_btn = ttk.Button(trading_controls, text="⏹️ Stop Trading", 
                                          command=self.stop_trading, state='disabled')
        self.stop_trading_btn.pack(side='left', padx=5)
        
        # Emergency stop
        self.emergency_stop_btn = ttk.Button(trading_controls, text="🛑 EMERGENCY STOP", 
                                            command=self.emergency_stop)
        self.emergency_stop_btn.pack(side='right', padx=5)

    def on_mode_change(self):
        """เมื่อเปลี่ยน Trading Mode"""
        try:
            new_mode = self.trading_mode_var.get()
            
            if new_mode == self.current_environment_type:
                return  # ไม่เปลี่ยน
            
            self.log_message(f"🔄 Switching to {new_mode.upper()} mode...", "INFO")
            
            # อัปเดต UI
            self._update_mode_ui(new_mode)
            
            # หยุดการเทรดก่อน (ถ้ากำลังเทรด)
            if self.is_trading:
                self.log_message("⏹️ Stopping current trading session...", "WARNING")
                self.stop_trading()
            
            # เปลี่ยน Environment (จะทำตอน start trading)
            self.current_environment_type = new_mode
            self.environment = None  # รีเซ็ต environment
            
            self.log_message(f"✅ Mode changed to {new_mode.upper()}", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"❌ Mode change error: {e}", "ERROR")

    def _update_mode_ui(self, mode):
        """อัปเดต UI ตาม mode"""
        try:
            if mode == "conservative":
                # Conservative Mode UI
                self.mode_description_label.config(
                    text="🛡️ Conservative: ใช้ RL Agent เดิม (HOLD บ่อย, ปลอดภัย)"
                )
                self.current_env_label.config(text="Conservative", foreground='blue')
                self.ai_insights_btn.config(state='disabled')
                
            elif mode == "aggressive":
                # Aggressive Mode UI
                self.mode_description_label.config(
                    text="⚡ Aggressive: ใช้ AI Recovery Brain (เทรดมาก, แก้ไม้อัจฉริยะ)"
                )
                self.current_env_label.config(text="Aggressive AI", foreground='red')
                self.ai_insights_btn.config(state='normal')
                
        except Exception as e:
            self.log_message(f"❌ Update mode UI error: {e}", "ERROR")


    def setup_training_tab(self):
        """Setup training controls tab with recovery settings"""
        # === CONFIGURATION SECTION === (เดิม)
        config_frame = ttk.LabelFrame(self.training_frame, text="⚙️ Configuration")
        config_frame.pack(fill='x', padx=10, pady=5)
        
        config_grid = ttk.Frame(config_frame)
        config_grid.pack(fill='x', padx=10, pady=10)
        
        # Symbol
        ttk.Label(config_grid, text="Symbol:").grid(row=0, column=0, sticky='w')
        symbol_entry = ttk.Entry(config_grid, textvariable=self.symbol_var, width=10)
        symbol_entry.grid(row=0, column=1, sticky='w', padx=(10, 0))
        
        # Lot size
        ttk.Label(config_grid, text="Base Lot Size:").grid(row=1, column=0, sticky='w')
        lot_size_entry = ttk.Entry(config_grid, textvariable=self.lot_size_var, width=10)
        lot_size_entry.grid(row=1, column=1, sticky='w', padx=(10, 0))
        
        # Max positions
        ttk.Label(config_grid, text="Max Positions:").grid(row=2, column=0, sticky='w')
        max_pos_entry = ttk.Entry(config_grid, textvariable=self.max_positions_var, width=10)
        max_pos_entry.grid(row=2, column=1, sticky='w', padx=(10, 0))
        
        # Training steps
        ttk.Label(config_grid, text="Training Steps:", style='Thai.TLabel').grid(row=3, column=0, sticky='w')
        training_steps_entry = ttk.Entry(config_grid, textvariable=self.training_steps_var, width=10, style='Thai.TEntry')
        training_steps_entry.grid(row=3, column=1, sticky='w', padx=(10, 0))
        ttk.Label(config_grid, text="(100,000 = comprehensive training)", style='Comment.TLabel').grid(row=3, column=2, sticky='w', padx=(5, 0))
        
        # Learning rate
        ttk.Label(config_grid, text="Learning Rate:").grid(row=4, column=0, sticky='w')
        lr_entry = ttk.Entry(config_grid, textvariable=self.learning_rate_var, width=10)
        lr_entry.grid(row=4, column=1, sticky='w', padx=(10, 0))

        # === RECOVERY STRATEGY SECTION === (เดิม)
        recovery_frame = ttk.LabelFrame(self.training_frame, text="🔄 Recovery Strategy")
        recovery_frame.pack(fill='x', padx=10, pady=5)
        
        recovery_grid = ttk.Frame(recovery_frame)
        recovery_grid.pack(fill='x', padx=10, pady=10)
        
        # Recovery multiplier
        ttk.Label(recovery_grid, text="Recovery Multiplier:", style='Thai.TLabel').grid(row=0, column=0, sticky='w')
        recovery_mult_entry = ttk.Entry(recovery_grid, textvariable=self.recovery_multiplier_var, width=10, style='Thai.TEntry')
        recovery_mult_entry.grid(row=0, column=1, sticky='w', padx=(10, 0))
        
        # Recovery threshold
        ttk.Label(recovery_grid, text="Recovery Threshold:", style='Thai.TLabel').grid(row=1, column=0, sticky='w')
        recovery_thresh_entry = ttk.Entry(recovery_grid, textvariable=self.recovery_threshold_var, width=10, style='Thai.TEntry')
        recovery_thresh_entry.grid(row=1, column=1, sticky='w', padx=(10, 0))
        
        # Max recovery levels
        ttk.Label(recovery_grid, text="Max Recovery Levels:", style='Thai.TLabel').grid(row=2, column=0, sticky='w')
        max_recovery_entry = ttk.Entry(recovery_grid, textvariable=self.max_recovery_levels_var, width=10, style='Thai.TEntry')
        max_recovery_entry.grid(row=2, column=1, sticky='w', padx=(10, 0))

        # ⭐⭐⭐ เพิ่มส่วนนี้ - RISK MANAGEMENT SECTION ⭐⭐⭐
        risk_frame = ttk.LabelFrame(self.training_frame, text="⚠️ Risk Management")
        risk_frame.pack(fill='x', padx=10, pady=5)
        
        risk_grid = ttk.Frame(risk_frame)
        risk_grid.pack(fill='x', padx=10, pady=10)
        
        # Risk per trade
        ttk.Label(risk_grid, text="Risk per Trade (%):", style='Thai.TLabel').grid(row=0, column=0, sticky='w')
        risk_trade_entry = ttk.Entry(risk_grid, textvariable=self.risk_per_trade_var, width=10, style='Thai.TEntry')
        risk_trade_entry.grid(row=0, column=1, sticky='w', padx=(10, 0))
        ttk.Label(risk_grid, text="(2% = เสี่ยง 2% ของทุนต่อการเทรด)", style='Comment.TLabel').grid(row=0, column=2, sticky='w', padx=(5, 0))
        
        # Max daily risk
        ttk.Label(risk_grid, text="Max Daily Risk (%):", style='Thai.TLabel').grid(row=1, column=0, sticky='w')
        daily_risk_entry = ttk.Entry(risk_grid, textvariable=self.max_daily_risk_var, width=10, style='Thai.TEntry')
        daily_risk_entry.grid(row=1, column=1, sticky='w', padx=(10, 0))
        ttk.Label(risk_grid, text="(10% = หยุดเทรดเมื่อขาดทุน 10%/วัน)", style='Comment.TLabel').grid(row=1, column=2, sticky='w', padx=(5, 0))
        
        # Max lot size
        ttk.Label(risk_grid, text="Max Lot Size:", style='Thai.TLabel').grid(row=2, column=0, sticky='w')
        max_lot_entry = ttk.Entry(risk_grid, textvariable=self.max_lot_size_var, width=10, style='Thai.TEntry')
        max_lot_entry.grid(row=2, column=1, sticky='w', padx=(10, 0))
        ttk.Label(risk_grid, text="(0.10 = ไม่เกิน 0.10 lots ต่อคำสั่ง)", style='Comment.TLabel').grid(row=2, column=2, sticky='w', padx=(5, 0))
        
        # Stop loss pips
        ttk.Label(risk_grid, text="Stop Loss (pips):", style='Thai.TLabel').grid(row=3, column=0, sticky='w')
        sl_pips_entry = ttk.Entry(risk_grid, textvariable=self.stop_loss_pips_var, width=10, style='Thai.TEntry')
        sl_pips_entry.grid(row=3, column=1, sticky='w', padx=(10, 0))
        ttk.Label(risk_grid, text="(20 = Stop loss 20 pips)", style='Comment.TLabel').grid(row=3, column=2, sticky='w', padx=(5, 0))
        # ⭐⭐⭐ จบส่วนที่เพิ่ม ⭐⭐⭐

        # === DATA SETTINGS SECTION === (เดิม - เหมือนเดิม)
        data_frame = ttk.LabelFrame(self.training_frame, text="📊 Historical Data Settings")
        data_frame.pack(fill='x', padx=10, pady=5)
        
        data_grid = ttk.Frame(data_frame)
        data_grid.pack(fill='x', padx=10, pady=10)
        
        # Timeframe (readonly - M5 fixed)
        ttk.Label(data_grid, text="Timeframe:").grid(row=0, column=0, sticky='w')
        ttk.Label(data_grid, text="M5 (5 นาที)", foreground='blue').grid(row=0, column=1, sticky='w', padx=(10, 0))
        
        # Lookback period
        ttk.Label(data_grid, text="Lookback Period:", style='Thai.TLabel').grid(row=1, column=0, sticky='w')
        ttk.Label(data_grid, text="2 Years (Full)", foreground='blue', style='Thai.TLabel').grid(row=1, column=1, sticky='w', padx=(10, 0))
        ttk.Label(data_grid, text="(730 days for comprehensive data)", style='Comment.TLabel').grid(row=1, column=2, sticky='w', padx=(5, 0))
        
        # Indicators (readonly)
        ttk.Label(data_grid, text="Indicators:", style='Thai.TLabel').grid(row=2, column=0, sticky='w')
        indicators_text = "SMA20, SMA50, EMA12, EMA26, RSI14, MACD, BB, ATR14"
        ttk.Label(data_grid, text=indicators_text, foreground='blue', style='Comment.TLabel').grid(row=2, column=1, sticky='w', padx=(10, 0))
        
        # Data cache status
        ttk.Label(data_grid, text="Cache Status:", style='Thai.TLabel').grid(row=3, column=0, sticky='w')
        self.cache_status_label = ttk.Label(data_grid, text="Not checked", foreground='orange', style='Comment.TLabel')
        self.cache_status_label.grid(row=3, column=1, sticky='w', padx=(10, 0))
        
        # Cache refresh button with Thai-friendly style
        self.refresh_cache_btn = ttk.Button(data_grid, text="🔄 Refresh Data Cache", 
                                        command=self.refresh_data_cache, style='Thai.TButton')
        self.refresh_cache_btn.grid(row=3, column=2, sticky='w', padx=(10, 0))
        
        # Save config button with Thai-friendly style
        save_config_btn = ttk.Button(config_grid, text="💾 Save Config", command=self.save_config, style='Thai.TButton')
        save_config_btn.grid(row=5, column=0, columnspan=3, pady=10)

        # === TRAINING CONTROLS SECTION === (เดิม - เหมือนเดิม)
        training_ctrl_frame = ttk.LabelFrame(self.training_frame, text="🎓 Training Controls")
        training_ctrl_frame.pack(fill='x', padx=10, pady=5)
        
        training_controls = ttk.Frame(training_ctrl_frame)
        training_controls.pack(fill='x', padx=10, pady=10)
        
        # Training buttons with Thai-friendly style
        self.start_training_btn = ttk.Button(training_controls, text="🚀 Start Recovery Training", 
                                            command=self.start_training, style='Thai.TButton')
        self.start_training_btn.pack(side='left', padx=5)
        
        self.stop_training_btn = ttk.Button(training_controls, text="⏹️ Stop Training", 
                                        command=self.stop_training, state='disabled', style='Thai.TButton')
        self.stop_training_btn.pack(side='left', padx=5)
        
        # Load model button with Thai-friendly style
        load_model_training_btn = ttk.Button(training_controls, text="📥 Load Model", 
                                        command=self.manual_load_model, style='Thai.TButton')
        load_model_training_btn.pack(side='left', padx=5)
        
        # Training status with Thai-friendly font
        self.training_status_label = ttk.Label(training_controls, text="Ready for Recovery Training", style='Thai.TLabel')
        self.training_status_label.pack(side='right')

    def setup_logs_tab(self):
        """Setup logging tab"""
        # Log display
        log_frame = ttk.Frame(self.logs_frame)
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Log text with scrollbar and Thai-friendly font
        self.log_text = tk.Text(log_frame, bg='#1e1e1e', fg='white', font=self.fonts['console'])
        log_scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')
        
        # Log controls
        log_controls = ttk.Frame(self.logs_frame)
        log_controls.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(log_controls, text="🗑️ Clear Logs", command=self.clear_logs).pack(side='left', padx=5)
        
        # Auto-scroll checkbox
        ttk.Checkbutton(log_controls, text="Auto-scroll", variable=self.auto_scroll_var).pack(side='right')

    def save_config(self):
        """Save current configuration รวม Risk Management"""
        try:
            os.makedirs('config', exist_ok=True)
            config_file = 'config/config.json'
            
            # Update config with current GUI values
            self.config.update({
                'symbol': self.symbol_var.get(),
                'lot_size': self.lot_size_var.get(),
                'max_positions': self.max_positions_var.get(),
                'training_steps': self.training_steps_var.get(),
                'learning_rate': self.learning_rate_var.get(),
                # Recovery settings
                'recovery_multiplier': self.recovery_multiplier_var.get(),
                'recovery_threshold': self.recovery_threshold_var.get(),
                'max_recovery_levels': self.max_recovery_levels_var.get(),
                # ⭐ Risk Management settings
                'risk_per_trade': self.risk_per_trade_var.get(),
                'max_daily_risk': self.max_daily_risk_var.get(),
                'max_lot_size': self.max_lot_size_var.get(),
                'stop_loss_pips': self.stop_loss_pips_var.get()
            })
            
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.log_message("✅ Configuration saved (including risk management)", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"❌ Config save error: {e}", "ERROR")


    def connect_mt5(self):
        """Connect to MT5"""
        try:
            self.log_message("🔌 Connecting to MT5...", "INFO")
            
            # Import MT5 interface
            from core.mt5_connector import MT5Connector
            self.mt5_interface = MT5Connector(self.config)
            
            # Attempt connection
            if self.mt5_interface.connect():
                self.is_connected = True
                self.conn_status_label.config(text="✅ Connected", foreground='green')
                self.log_message("✅ MT5 connected successfully", "SUCCESS")
                
                # Update account info
                self.update_account_info()
                
                # Initialize trading system
                self.initialize_trading_system()
                
            else:
                error_msg = self.mt5_interface.get_last_error()
                self.log_message(f"❌ MT5 connection failed: {error_msg}", "ERROR")
                messagebox.showerror("Connection Error", f"Failed to connect to MT5: {error_msg}")
                
        except ImportError as e:
            self.log_message(f"❌ MT5 module error: {e}", "ERROR")
            messagebox.showerror("Import Error", "Cannot import MT5 interface. Check installation.")
        except Exception as e:
            self.log_message(f"❌ Connection error: {e}", "ERROR")
            messagebox.showerror("Error", f"Connection error: {e}")

    def initialize_trading_system(self):
        """Initialize environment and try to load existing model"""
        try:
            self.log_message("🏗️ Initializing trading system...", "INFO")
            
            # Initialize data loader with Smart Cache
            from core.data_loader import HistoricalDataLoader
            self.data_loader = HistoricalDataLoader(self.mt5_interface)
            
            # Check cache status
            cache_status = self.data_loader.get_cache_status()
            if cache_status['cache_exists'] and cache_status['cache_valid']:
                self.log_message(f"💾 Historical data cache found ({cache_status['data_rows']:,} rows)", "SUCCESS")
            else:
                self.log_message("⚠️ No valid data cache - will download on training", "WARNING")
            
            # Update cache status display
            self.update_cache_status()
            
            # ⭐ Create Environment ตาม Mode ที่เลือก
            mode = self.current_environment_type
            self.log_message(f"🎯 Creating {mode.upper()} environment...", "INFO")
            
            self.environment = create_environment(mode, self.mt5_interface, self.config)
            
            if self.environment:
                self.log_message(f"✅ {mode.upper()} environment created", "SUCCESS")
                
                # แสดงข้อมูล environment
                env_type = getattr(self.environment, '__class__.__name__', 'Unknown')
                self.log_message(f"   - Environment Type: {env_type}", "INFO")
                
                if mode == "aggressive":
                    # แสดงข้อมูล AI Brain
                    try:
                        ai_status = self.environment.get_ai_status()
                        if ai_status:
                            self.log_message("   - AI Recovery Brain: Active", "SUCCESS")
                            session_info = ai_status.get('session', {})
                            if session_info:
                                self.log_message(f"   - AI Session ID: {session_info.get('id', 'Unknown')}", "INFO")
                    except:
                        pass
            else:
                self.log_message(f"❌ Failed to create {mode} environment", "ERROR")
                return
            
            # Create agent (ใช้เดิม)
            from core.rl_agent import RLAgent
            self.agent = RLAgent(self.environment, self.config)
            self.log_message("✅ RL agent created", "SUCCESS")
            
            # Try to auto-load existing model
            self.auto_load_model()
            
        except Exception as e:
            self.log_message(f"❌ System initialization error: {e}", "ERROR")
                        

    def auto_load_model(self):
        """Automatically try to load the latest saved model"""
        try:
            if self.agent is None:
                return
                
            self.log_message("🔍 Looking for saved models...", "INFO")
            
            # List available models
            models = self.agent.list_saved_models()
            
            if models:
                # Try to load the latest model
                latest_model = models[0]['filename']
                self.log_message(f"📥 Attempting to load: {latest_model}", "INFO")
                
                success = self.agent.load_model(latest_model)
                
                if success:
                    self.model_status_label.config(text="✅ Model Loaded", foreground='green')
                    self.start_trading_btn.config(state='normal')
                    self.log_message(f"✅ Model loaded successfully: {latest_model}", "SUCCESS")
                    
                    # Show model info
                    model_info = self.agent.get_model_info()
                    self.log_message(f"📊 Model info: {model_info['algorithm']}, Trained: {model_info['is_trained']}", "INFO")
                else:
                    self.log_message("❌ Failed to load model", "ERROR")
                    self.model_status_label.config(text="❌ Load Failed", foreground='red')
            else:
                self.log_message("⚠️ No saved models found. Please train first.", "WARNING")
                self.model_status_label.config(text="❌ No Models Found", foreground='orange')
                
        except Exception as e:
            self.log_message(f"❌ Auto-load error: {e}", "ERROR")

    def manual_load_model(self):
        """Manually load model"""
        try:
            if self.agent is None:
                if self.is_connected:
                    self.initialize_trading_system()
                else:
                    messagebox.showwarning("Warning", "Please connect to MT5 first")
                    return
            
            self.log_message("📥 Manual model loading...", "INFO")
            
            success = self.agent.load_model()
            
            if success:
                self.model_status_label.config(text="✅ Model Loaded", foreground='green')
                self.start_trading_btn.config(state='normal')
                self.log_message("✅ Model loaded successfully", "SUCCESS")
            else:
                self.model_status_label.config(text="❌ Load Failed", foreground='red')
                self.log_message("❌ Failed to load model", "ERROR")
                messagebox.showerror("Error", "Failed to load model. Check if trained models exist.")
                
        except Exception as e:
            self.log_message(f"❌ Manual load error: {e}", "ERROR")

    def start_training(self):
        """Start training in background thread"""
        if self.is_training:
            self.log_message("⚠️ Training already in progress", "WARNING")
            return
        
        try:
            # Initialize system if not already done
            if not self.is_connected:
                messagebox.showwarning("Warning", "Please connect to MT5 first")
                return
                
            if self.environment is None:
                self.initialize_trading_system()
            
            # ตรวจสอบ mode
            mode_name = self.current_environment_type.upper()
            self.log_message(f"🎓 Starting {mode_name} training...", "INFO")
            
            # แสดงข้อมูล environment
            if hasattr(self.environment, '__class__'):
                env_class = self.environment.__class__.__name__
                self.log_message(f"   Using: {env_class}", "INFO")
            
            self.is_training = True
            self.start_training_btn.config(state='disabled')
            self.stop_training_btn.config(state='normal')
            self.training_status_label.config(text=f"Training {mode_name}...")
            
            # Start training in separate thread
            self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
            self.training_thread.start()
            
        except Exception as e:
            self.log_message(f"❌ Training start error: {e}", "ERROR")
            self.reset_training_controls()


    def _training_worker(self):
        """Training worker thread with Smart Cache data loading"""
        try:
            # Smart Cache: Auto-load historical data
            self.log_message("🧠 Smart Cache: Loading historical data...", "INFO")
            
            force_refresh = False  # Could add GUI option for this
            success = self.data_loader.smart_load_data(force_refresh)
            
            if not success:
                self.log_message("❌ Failed to load historical data", "ERROR")
                return
            
            # Get training data
            training_data = self.data_loader.get_training_data()
            if training_data is None:
                self.log_message("❌ No training data available", "ERROR")
                return
            
            self.log_message(f"✅ Training data ready: {len(training_data):,} rows", "SUCCESS")
            
            # Update environment with historical data
            self.environment.set_historical_data(training_data)
            self.log_message("✅ Environment updated with historical data", "SUCCESS")
            
            # Start training
            training_steps = self.training_steps_var.get()
            self.log_message(f"🚀 Training for {training_steps} steps...", "INFO")
            
            success = self.agent.train(total_timesteps=training_steps)
            
            if success:
                # Save model
                model_path = self.agent.save_model()
                if model_path:
                    self.log_message(f"💾 Model saved: {model_path}", "SUCCESS")
                    
                    # Auto-load the trained model
                    self.root.after(0, self._post_training_load)
            
        except Exception as e:
            self.log_message(f"❌ Training error: {e}", "ERROR")
        finally:
            self.root.after(0, self.reset_training_controls)

    def _post_training_load(self):
        """Load model after training completes"""
        try:
            success = self.agent.load_model()
            if success:
                self.model_status_label.config(text="✅ Model Loaded", foreground='green')
                self.start_trading_btn.config(state='normal')
                self.log_message("✅ Trained model loaded successfully", "SUCCESS")
        except Exception as e:
            self.log_message(f"❌ Post-training load error: {e}", "ERROR")

    def start_trading(self):
        """Start live trading with current mode"""
        try:
            if not self.is_connected:
                messagebox.showwarning("Warning", "Please connect to MT5 first")
                return
            
            # สร้าง environment ใหม่ตาม mode ปัจจุบัน (ถ้ายังไม่มี)
            if self.environment is None:
                self.initialize_trading_system()
            
            if self.environment is None:
                messagebox.showerror("Error", "Failed to initialize trading environment")
                return
            
            # ตรวจสอบ model (เฉพาะ Conservative mode)
            if self.current_environment_type == "conservative":
                if self.agent is None or not self.agent.is_trained:
                    self.log_message("⚠️ No trained model detected, attempting to load...", "WARNING")
                    success = self.agent.load_model() if self.agent else False
                    if not success:
                        messagebox.showerror("Error", "No trained model found. Please train or load a model first.")
                        return
                    else:
                        self.model_status_label.config(text="✅ Model Loaded", foreground='green')
            
            # เริ่มการเทรด
            mode_name = self.current_environment_type.upper()
            self.is_trading = True
            self.start_trading_btn.config(state='disabled')
            self.stop_trading_btn.config(state='normal')
            
            self.log_message(f"🚀 {mode_name} trading started", "SUCCESS")
            
            # แสดงข้อมูล environment
            if hasattr(self.environment, '__class__'):
                env_class = self.environment.__class__.__name__
                self.log_message(f"   Using: {env_class}", "INFO")
            
            # เพิ่มข้อมูล AI (ถ้าเป็น Aggressive mode)
            if self.current_environment_type == "aggressive":
                try:
                    ai_status = self.environment.get_ai_status()
                    if ai_status and 'session' in ai_status:
                        session_id = ai_status['session'].get('id', 'Unknown')
                        self.log_message(f"   AI Session: {session_id}", "INFO")
                except:
                    pass
            
            # Start trading loop
            self.trading_thread = threading.Thread(target=self._trading_worker, daemon=True)
            self.trading_thread.start()
            
        except Exception as e:
            self.log_message(f"❌ Trading start error: {e}", "ERROR")


    def _initialize_trading_session(self):
        """
        🚀 เริ่มต้น Trading Session ใหม่
        
        เชื่อมกับ:
        - start_trading() method
        - MT5 account info
        
        หน้าที่:
        1. รีเซ็ตตัวแปร session
        2. บันทึก balance เริ่มต้น
        3. รีเซ็ต recovery state
        4. เริ่มนับ performance metrics
        """
        try:
            from datetime import datetime
            
            # รีเซ็ต session variables
            self.current_step = 0
            self.trading_session_start = datetime.now()
            
            # ดึง balance เริ่มต้น
            if self.mt5_interface:
                account_info = self.mt5_interface.get_account_info()
                if account_info:
                    self.session_start_balance = account_info.get('balance', 0.0)
                else:
                    self.session_start_balance = 0.0
            else:
                self.session_start_balance = 0.0
            
            # รีเซ็ต recovery state
            self.in_recovery_mode = False
            self.recovery_level = 0
            self.recovery_start_pnl = 0.0
            self.recovery_target = 0.0
            
            # รีเซ็ต performance tracking
            self.total_trades_today = 0
            self.winning_trades_today = 0
            self.losing_trades_today = 0
            self.max_drawdown_today = 0.0
            self.daily_pnl = 0.0
            
            self.log_message(f"🚀 Trading session initialized - Start balance: ${self.session_start_balance:.2f}", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"❌ Session initialization error: {e}", "ERROR")


    def _trading_worker(self):
        """
        🤖 Real AI Trading Worker - หัวใจหลักของระบบเทรด
        
        ⭐ แก้ไข: เพิ่ม Mode Switching แต่รักษาโค้ดเดิมทั้งหมด
        """
        try:
            # ⭐ เช็ค Trading Mode
            mode = getattr(self, 'current_environment_type', 'conservative')
            
            if mode == 'aggressive' and hasattr(self, 'environment') and hasattr(self.environment, 'recovery_brain'):
                # 🧠 Aggressive Mode - ใช้ AI Brain
                self.log_message("⚡ AGGRESSIVE AI MODE - Using AI Recovery Brain", "SUCCESS")
                self._aggressive_ai_trading_worker()
            else:
                # 🛡️ Conservative Mode - ใช้โค้ดเดิม
                self.log_message("🛡️ CONSERVATIVE MODE - Using Original RL System", "SUCCESS")
                self._conservative_original_trading_worker()
                
        except Exception as e:
            self.log_message(f"❌ Trading worker initialization error: {e}", "ERROR")
            self.emergency_stop()

    def _conservative_original_trading_worker(self):
        """
        🛡️ Conservative Trading Worker - โค้ดเดิมของคุณทั้งหมด (ไม่แก้อะไร)
        """
        try:
            import time
            import numpy as np
            from datetime import datetime
            
            self.log_message("🚀 AI Trading Worker Started - LIVE MODE", "SUCCESS")
            self.log_message(f"🎯 Risk per Trade: {self.risk_per_trade_var.get():.2f}%", "INFO")
            self.log_message(f"💰 Max Daily Risk: {self.max_daily_risk_var.get():.2f}%", "INFO")
            
            # Initialize trading session
            session_start_balance = self._get_current_balance()
            daily_risk_used = 0.0
            
            while self.is_trading and self.running:
                try:
                    # ===== 1. GET REAL-TIME MARKET DATA =====
                    # เชื่อมกับ MT5 เพื่อดึงข้อมูลล่าสุด
                    current_observation = self._get_live_observation()
                    if current_observation is None:
                        self.log_message("⚠️ Cannot get market observation", "WARNING")
                        time.sleep(5)
                        continue
                    
                    # ===== 2. DAILY RISK CHECK =====
                    # ตรวจสอบว่าใช้ risk เกินที่กำหนดไหม
                    current_balance = self._get_current_balance()
                    daily_loss = session_start_balance - current_balance
                    daily_risk_pct = (daily_loss / session_start_balance) * 100
                    
                    if daily_risk_pct >= self.max_daily_risk_var.get():
                        self.log_message(f"🛑 Daily risk limit reached: {daily_risk_pct:.2f}%", "ERROR")
                        self.log_message("⏹️ Stopping trading for today", "WARNING")
                        self.stop_trading()
                        break
                    
                    # ===== 3. AI MODEL PREDICTION =====
                    # ใช้ trained model ทำนายการเทรด
                    if not self.agent or not self.agent.is_trained:
                        self.log_message("❌ No trained model available", "ERROR")
                        break
                    
                    # ทำนายด้วย AI
                    predicted_action = self.agent.predict(current_observation)
                    action_type = int(predicted_action[0])
                    volume_ratio = float(predicted_action[1]) if len(predicted_action) > 1 else 0.01
                    
                    # แปลง action เป็นคำสั่งที่เข้าใจได้
                    action_name = self._get_action_name(action_type)
                    self.log_message(f"🧠 AI Decision: {action_name} (confidence: {volume_ratio:.3f})", "INFO")
                    
                    # ===== 4. RISK CALCULATION =====
                    # คำนวณขนาด position ตาม risk management
                    if action_type in [1, 2]:  # BUY or SELL
                        position_size = self._calculate_risk_based_position_size()
                        
                        if position_size == 0:
                            self.log_message("⚠️ Position size too small - skipping trade", "WARNING")
                            time.sleep(10)
                            continue
                            
                        self.log_message(f"📊 Calculated position size: {position_size:.2f} lots", "INFO")
                        
                        # ===== 5. EXECUTE TRADE =====
                        # ส่งคำสั่งไปยัง MT5
                        success = self._execute_ai_trade(action_type, position_size, predicted_action)
                        
                        if success:
                            # อัพเดท daily risk usage
                            trade_risk = (position_size * 100000 * 0.01) / current_balance * 100  # Rough estimate
                            daily_risk_used += trade_risk
                            self.log_message(f"✅ Trade executed - Daily risk used: {daily_risk_used:.2f}%", "SUCCESS")
                        
                    elif action_type == 3:  # CLOSE ALL
                        self.log_message("🔄 AI Signal: Close all positions", "INFO")
                        closed_positions = self._close_all_positions()
                        if closed_positions > 0:
                            self.log_message(f"✅ Closed {closed_positions} positions", "SUCCESS")
                    
                    elif action_type == 0:  # HOLD
                        self.log_message("⏸️ AI Signal: Hold (no action)", "INFO")
                    
                    # ===== 6. RECOVERY MODE CHECK =====
                    # ตรวจสอบว่าควรเข้า recovery mode หรือไม่
                    self._check_recovery_mode()
                    
                    # ===== 7. POSITION MONITORING =====
                    # ตรวจสอบ positions ที่เปิดอยู่
                    self._monitor_open_positions()
                    
                    # แสดงสถานะปัจจุบัน
                    if self.current_step % 12 == 0:  # ทุก 1 นาที (5s * 12)
                        self._log_trading_status(current_balance, daily_risk_pct)
                    
                    self.current_step += 1
                    
                except Exception as trade_error:
                    self.log_message(f"❌ Trading cycle error: {trade_error}", "ERROR")
                    
                # รอ 5 วินาที ก่อนรอบถัดไป
                time.sleep(5)
                
        except Exception as e:
            self.log_message(f"❌ Trading worker critical error: {e}", "ERROR")
            self.emergency_stop()
        
        finally:
            self.log_message("🏁 AI Trading Worker Stopped", "INFO")

    def _aggressive_ai_trading_worker(self):
        """
        ⚡ Aggressive AI Trading Worker - ใช้ AI Recovery Brain
        """
        try:
            import time
            import numpy as np
            from datetime import datetime
            
            self.log_message("⚡ AI Recovery Brain Trading Started", "SUCCESS")
            self.log_message("🎯 Target: High volume for rebate optimization", "INFO")
            
            # Initialize AI trading session
            self._initialize_trading_session()
            
            while self.is_trading and self.running:
                try:
                    # AI Environment จัดการทุกอย่างเอง
                    # เราแค่ step ด้วย dummy action เพื่อให้ AI ตัดสินใจ
                    dummy_action = [0, 0.01, 30, 0]  # HOLD action
                    
                    obs, reward, done, truncated, info = self.environment.step(dummy_action)
                    
                    # แสดงข้อมูล AI Decision
                    if info and 'current_ai_action' in info:
                        ai_action = info['current_ai_action']
                        ai_strategy = info.get('current_ai_strategy', 'Unknown')
                        ai_confidence = info.get('current_ai_confidence', 0)
                        daily_volume = info.get('daily_volume', 0)
                        volume_target = info.get('volume_target', 75)
                        total_pnl = info.get('total_pnl', 0)
                        
                        if ai_action != 'HOLD':  # แสดงเฉพาะเมื่อมีการเทรด
                            self.log_message(
                                f"🧠 AI: {ai_action} | Strategy: {ai_strategy} | "
                                f"Confidence: {ai_confidence:.2f} | Reward: {reward:.2f}",
                                "SUCCESS" if reward > 0 else "INFO"
                            )
                            
                            # แสดง AI reasoning
                            if 'current_ai_reasoning' in info:
                                reasons = info['current_ai_reasoning']
                                if reasons:
                                    for reason in reasons[:2]:  # แสดงแค่ 2 เหตุผลแรก
                                        self.log_message(f"   💡 {reason}", "INFO")
                            
                            # แสดง warnings
                            if 'current_ai_warnings' in info:
                                warnings = info['current_ai_warnings']
                                if warnings:
                                    for warning in warnings[:1]:  # แสดงแค่ warning แรก
                                        self.log_message(f"   ⚠️ {warning}", "WARNING")
                    
                    # แสดงสรุปทุก 24 cycles (2 นาที)
                    if self.current_step % 24 == 0:
                        if info:
                            daily_volume = info.get('daily_volume', 0)
                            volume_target = info.get('volume_target', 75)
                            volume_progress = info.get('volume_progress', 0)
                            total_pnl = info.get('total_pnl', 0)
                            ai_success_rate = info.get('ai_success_rate', 0)
                            recovery_state = info.get('current_ai_recovery_state', 'NORMAL')
                            
                            status_msg = (
                                f"⚡ AI Status: {recovery_state} | "
                                f"Volume {daily_volume:.1f}/{volume_target} ({volume_progress:.1f}%) | "
                                f"P&L: ${total_pnl:.2f} | Success: {ai_success_rate:.1%}"
                            )
                            
                            self.log_message(status_msg, "INFO")
                    
                    # Update GUI recovery display
                    self._update_recovery_display_from_ai_brain()
                    
                    # Check if episode done
                    if done:
                        self.log_message("📈 AI Episode completed - resetting", "INFO")
                        self.environment.reset()
                    
                    self.current_step += 1
                    
                except Exception as cycle_error:
                    self.log_message(f"❌ AI Trading cycle error: {cycle_error}", "ERROR")
                    time.sleep(10)
                    
                # รอ 5 วินาที ก่อนรอบถัดไป
                time.sleep(5)
                
        except Exception as e:
            self.log_message(f"❌ AI Trading worker critical error: {e}", "ERROR")
            self.emergency_stop()
        
        finally:
            self.log_message("🏁 AI Trading Worker Stopped", "INFO")
            
            # End AI session
            if hasattr(self.environment, 'end_ai_session'):
                try:
                    session_summary = self.environment.end_ai_session()
                    if session_summary:
                        self.log_message(f"📊 AI Session Summary: {session_summary.get('total_pnl', 0):.2f} P&L", "INFO")
                except Exception as e:
                    self.log_message(f"❌ End AI session error: {e}", "ERROR")

    def _update_recovery_display_from_ai_brain(self):
        """
        อัปเดต Recovery Display จาก AI Brain
        """
        try:
            if not hasattr(self, 'environment') or not hasattr(self.environment, 'get_ai_status'):
                return
                
            ai_status = self.environment.get_ai_status()
            
            if not ai_status or 'session' not in ai_status:
                return
                
            session = ai_status['session']
            
            # Recovery State
            recovery_state = session.get('recovery_state', 'normal').upper()
            if recovery_state == 'NORMAL':
                self.recovery_mode_label.config(text="🟢 Normal", foreground='green')
            elif recovery_state in ['EARLY_RECOVERY', 'ACTIVE_RECOVERY']:
                self.recovery_mode_label.config(text=f"🔄 {recovery_state}", foreground='orange')
            elif recovery_state in ['DEEP_RECOVERY', 'EMERGENCY']:
                self.recovery_mode_label.config(text=f"🔴 {recovery_state}", foreground='red')
            elif recovery_state == 'SUCCESS':
                self.recovery_mode_label.config(text="✅ SUCCESS", foreground='green')
            
            # P&L
            pnl = session.get('total_pnl', 0)
            pnl_color = 'green' if pnl >= 0 else 'red'
            self.total_pnl_label.config(text=f"${pnl:.2f}", foreground=pnl_color)
            
            # Max Drawdown
            dd = session.get('max_drawdown', 0)
            dd_color = 'red' if dd < -50 else 'orange' if dd < 0 else 'green'
            self.max_drawdown_label.config(text=f"${dd:.2f}", foreground=dd_color)
            
            # Win Rate
            win_rate = session.get('win_rate', 0)
            win_color = 'green' if win_rate >= 60 else 'orange' if win_rate >= 40 else 'red'
            total_trades = session.get('total_trades', 0)
            winning_trades = int(total_trades * win_rate / 100) if total_trades > 0 else 0
            self.win_rate_label.config(text=f"{winning_trades}/{total_trades} ({win_rate:.0f}%)", foreground=win_color)
            
        except Exception as e:
            # Silent fail for GUI updates
            pass

    def _get_current_balance(self):
        """
        💰 ดึง account balance ปัจจุบัน
        
        เชื่อมกับ:
        - self.mt5_interface - ดึงข้อมูล account
        
        หน้าที่:
        - Return balance ปัจจุบัน
        - Handle error กรณีไม่สามารถดึงข้อมูลได้
        """
        try:
            if not self.mt5_interface:
                return 0.0
                
            account_info = self.mt5_interface.get_account_info()
            if account_info:
                return account_info.get('balance', 0.0)
            return 0.0
        except Exception as e:
            self.log_message(f"❌ Balance error: {e}", "ERROR")
            return 0.0

    def _get_action_name(self, action_type):
        """
        📝 แปลง action number เป็นชื่อที่เข้าใจง่าย
        
        เชื่อมกับ:
        - AI Model predictions
        
        หน้าที่:
        - แปลง 0,1,2,3,4 เป็น HOLD, BUY, SELL, CLOSE, RECOVERY
        """
        action_names = {
            0: "HOLD",
            1: "BUY", 
            2: "SELL",
            3: "CLOSE_ALL",
            4: "RECOVERY"
        }
        return action_names.get(action_type, "UNKNOWN")

    def _log_trading_status(self, balance, daily_risk_pct):
        """
        📊 แสดงสถานะการเทรดปัจจุบัน
        
        เชื่อมกับ:
        - GUI Log display
        - Account information
        
        หน้าที่:
        - แสดงข้อมูลสถานะครบถ้วน
        - ช่วยในการ monitor ระบบ
        """
        positions = self.mt5_interface.get_positions() if self.mt5_interface else []
        total_profit = sum(pos.get('profit', 0) for pos in positions)
        
        status_msg = (f"📊 Status: Balance=${balance:.2f} | "
                    f"Positions={len(positions)} | "
                    f"Unrealized P&L=${total_profit:.2f} | "
                    f"Daily Risk={daily_risk_pct:.1f}%")
        
        self.log_message(status_msg, "INFO")


    def _get_live_observation(self):
        
        try:
            symbol = self.config.get('symbol', 'XAUUSD')
            
            # ดึงราคาปัจจุบัน
            current_price = self.mt5_interface.get_current_price(symbol)
            if not current_price:
                return None
                
            # ดึงข้อมูล historical สำหรับคำนวณ indicators
            rates = self.mt5_interface.get_rates(symbol, 5, 100)  # M5, 100 candles
            if rates is None or len(rates) < 50:
                return None
                
            # คำนวณ basic indicators (simplified for live trading)
            closes = [rate[4] for rate in rates]  # Close prices
            current_close = current_price['bid']
            
            # สร้าง observation array (40 มิติ)
            obs = np.zeros(40, dtype=np.float32)
            
            # Market data (15 features)
            obs[0] = (current_price['ask'] - current_price['bid']) / current_close  # Spread
            obs[1] = current_close / closes[-2] - 1  # Price change
            
            # Simple moving averages
            if len(closes) >= 20:
                sma20 = np.mean(closes[-20:])
                obs[2] = (current_close - sma20) / current_close
            
            if len(closes) >= 50:
                sma50 = np.mean(closes[-50:])
                obs[3] = (current_close - sma50) / current_close
                
            # Position และ account info
            positions = self.mt5_interface.get_positions(symbol)
            obs[20] = len(positions) / 5  # Position count ratio
            
            if positions:
                total_profit = sum(pos.get('profit', 0) for pos in positions)
                obs[21] = total_profit / 100  # Unrealized P&L
                
            # Account info
            account_info = self.mt5_interface.get_account_info()
            if account_info:
                obs[22] = account_info.get('equity', 0) / account_info.get('balance', 1)
                
            # เติมข้อมูลอื่นๆ ตามต้องการ...
            
            # Clip ค่าให้อยู่ในช่วงที่กำหนด
            obs = np.clip(obs, -10.0, 10.0)
            
            return obs
            
        except Exception as e:
            self.log_message(f"❌ Live observation error: {e}", "ERROR")
            return None


    def _calculate_risk_based_position_size(self):
        """
        💰 คำนวณขนาด position ตาม risk management
        
        เชื่อมกับ:
        - self.risk_per_trade_var (GUI) - % risk ต่อการเทรด
        - self.mt5_interface - ข้อมูล account balance
        - self.config - ขนาด lot พื้นฐาน
        
        หน้าที่:
        1. ดึง account balance ปัจจุบัน
        2. คำนวณเงินที่เสี่ยงได้ต่อ trade
        3. แปลงเป็นขนาด lot ที่เหมาะสม
        4. ตรวจสอบขีดจำกัดต่างๆ
        """
        try:
            # ดึงข้อมูล account
            account_info = self.mt5_interface.get_account_info()
            if not account_info:
                return 0.0
                
            balance = account_info.get('balance', 0)
            if balance <= 0:
                return 0.0
                
            # คำนวณเงินที่เสี่ยงได้
            risk_per_trade_pct = self.risk_per_trade_var.get()
            risk_amount = balance * (risk_per_trade_pct / 100)
            
            self.log_message(f"💰 Balance: ${balance:.2f}, Risk amount: ${risk_amount:.2f}", "INFO")
            
            # แปลงเป็น lot size
            # สำหรับ Gold: 1 lot = 100 oz, $1 per pip per 0.01 lot
            # Stop loss assumption: 20 pips
            assumed_sl_pips = 20
            lot_size = risk_amount / (assumed_sl_pips * 100)  # 100 = $1 per pip per 0.01 lot
            
            # ปรับให้เป็น 0.01 increments
            lot_size = round(lot_size / 0.01) * 0.01
            
            # จำกัดขนาด lot
            min_lot = 0.01
            max_lot = self.config.get('max_lot_size', 0.10)
            
            lot_size = max(min_lot, min(lot_size, max_lot))
            
            self.log_message(f"📊 Calculated lot size: {lot_size:.2f}", "INFO")
            
            return lot_size
            
        except Exception as e:
            self.log_message(f"❌ Position size calculation error: {e}", "ERROR")
            return 0.0
    
    def stop_trading(self):
        """Stop live trading"""
        try:
            self.is_trading = False
            self.start_trading_btn.config(state='normal')
            self.stop_trading_btn.config(state='disabled')
            
            # ⭐ รีเซ็ต trading session
            self._reset_trading_session()
            
            self.log_message("⏹️ Recovery trading stopped", "INFO")
            
        except Exception as e:
            self.log_message(f"❌ Stop trading error: {e}", "ERROR")

    def stop_training(self):
        """Stop training"""
        self.is_training = False
        self.log_message("⏹️ Training stop requested", "WARNING")

    def emergency_stop(self):
        """Emergency stop all operations"""
        try:
            self.is_training = False
            self.is_trading = False
            
            # Close all positions if connected
            if self.is_connected and self.mt5_interface:
                positions = self.mt5_interface.get_positions()
                for pos in positions:
                    self.mt5_interface.close_position(pos.get('ticket'))
            
            # ⭐ รีเซ็ต trading session
            self._reset_trading_session()
            
            self.reset_training_controls()
            self.start_trading_btn.config(state='normal')
            self.stop_trading_btn.config(state='disabled')
            
            self.log_message("🛑 EMERGENCY STOP ACTIVATED", "ERROR")
            messagebox.showwarning("Emergency Stop", "All operations stopped!")
            
        except Exception as e:
            self.log_message(f"❌ Emergency stop error: {e}", "ERROR")
    
    def _reset_trading_session(self):
        """
        🔄 รีเซ็ต Trading Session
        
        เชื่อมกับ:
        - stop_trading() method
        - emergency_stop() method
        
        หน้าที่:
        1. รีเซ็ตตัวแปรทั้งหมด
        2. บันทึกผลการเทรดสุดท้าย
        3. รีเซ็ต recovery state
        """
        try:
            # บันทึกผลสุดท้าย
            performance = self._calculate_session_performance()
            
            self.log_message(f"📊 Session Summary:", "INFO")
            self.log_message(f"   Daily P&L: ${performance['daily_pnl']:.2f} ({performance['daily_pnl_pct']:.2f}%)", "INFO")
            self.log_message(f"   Max Drawdown: ${performance['drawdown']:.2f} ({performance['drawdown_pct']:.2f}%)", "INFO")
            self.log_message(f"   Total Trades: {self.total_trades_today}", "INFO")
            
            if self.total_trades_today > 0:
                win_rate = (self.winning_trades_today / self.total_trades_today) * 100
                self.log_message(f"   Win Rate: {self.winning_trades_today}/{self.total_trades_today} ({win_rate:.1f}%)", "INFO")
            
            # รีเซ็ตตัวแปร
            self.current_step = 0
            self.trading_session_start = None
            self.in_recovery_mode = False
            self.recovery_level = 0
            
        except Exception as e:
            self.log_message(f"❌ Session reset error: {e}", "ERROR")
    
    def _calculate_session_performance(self):
        """
        📊 คำนวณผลการเทรดในวันนี้
        
        เชื่อมกับ:
        - MT5 account info
        - Session start balance
        
        หน้าที่:
        1. คำนวณ daily P&L
        2. คำนวณ drawdown
        3. อัพเดท performance metrics
        4. Return performance summary
        """
        try:
            if not self.mt5_interface or self.session_start_balance == 0:
                return {
                    'daily_pnl': 0.0,
                    'daily_pnl_pct': 0.0,
                    'drawdown': 0.0,
                    'drawdown_pct': 0.0
                }
            
            # ดึง balance ปัจจุบัน
            account_info = self.mt5_interface.get_account_info()
            if not account_info:
                return {'daily_pnl': 0.0, 'daily_pnl_pct': 0.0, 'drawdown': 0.0, 'drawdown_pct': 0.0}
            
            current_balance = account_info.get('balance', 0.0)
            current_equity = account_info.get('equity', 0.0)
            
            # คำนวณ daily P&L
            self.daily_pnl = current_balance - self.session_start_balance
            daily_pnl_pct = (self.daily_pnl / self.session_start_balance) * 100 if self.session_start_balance > 0 else 0
            
            # คำนวณ drawdown
            drawdown = min(0, self.daily_pnl)
            drawdown_pct = (drawdown / self.session_start_balance) * 100 if self.session_start_balance > 0 else 0
            
            # อัพเดท max drawdown
            if drawdown < self.max_drawdown_today:
                self.max_drawdown_today = drawdown
            
            return {
                'daily_pnl': self.daily_pnl,
                'daily_pnl_pct': daily_pnl_pct,
                'drawdown': self.max_drawdown_today,
                'drawdown_pct': (self.max_drawdown_today / self.session_start_balance) * 100 if self.session_start_balance > 0 else 0,
                'current_balance': current_balance,
                'current_equity': current_equity
            }
            
        except Exception as e:
            self.log_message(f"❌ Performance calculation error: {e}", "ERROR")
            return {'daily_pnl': 0.0, 'daily_pnl_pct': 0.0, 'drawdown': 0.0, 'drawdown_pct': 0.0}


    def reset_training_controls(self):
        """Reset training control states"""
        self.is_training = False
        self.start_training_btn.config(state='normal')
        self.stop_training_btn.config(state='disabled')
        self.training_status_label.config(text="Ready for Recovery Training")

    def refresh_data_cache(self):
        """Refresh historical data cache"""
        try:
            if not self.is_connected:
                messagebox.showwarning("Warning", "Please connect to MT5 first")
                return
            
            if self.data_loader is None:
                from core.data_loader import HistoricalDataLoader
                self.data_loader = HistoricalDataLoader(self.mt5_interface)
            
            self.log_message("🔄 Refreshing data cache...", "INFO")
            self.refresh_cache_btn.config(state='disabled', text="🔄 Refreshing...")
            
            # Run refresh in separate thread
            def refresh_worker():
                try:
                    success = self.data_loader.smart_load_data(force_refresh=True)
                    if success:
                        self.root.after(0, lambda: self.log_message("✅ Data cache refreshed", "SUCCESS"))
                        self.root.after(0, self.update_cache_status)
                    else:
                        self.root.after(0, lambda: self.log_message("❌ Cache refresh failed", "ERROR"))
                except Exception as e:
                    self.root.after(0, lambda: self.log_message(f"❌ Cache refresh error: {e}", "ERROR"))
                finally:
                    self.root.after(0, lambda: self.refresh_cache_btn.config(state='normal', text="🔄 Refresh Data Cache"))
            
            threading.Thread(target=refresh_worker, daemon=True).start()
            
        except Exception as e:
            self.log_message(f"❌ Refresh cache error: {e}", "ERROR")

    def update_cache_status(self):
        """Update cache status display"""
        try:
            if self.data_loader:
                cache_status = self.data_loader.get_cache_status()
                if cache_status['cache_valid']:
                    status_text = f"✅ Valid ({cache_status['data_rows']:,} rows)"
                    color = 'green'
                elif cache_status['cache_exists']:
                    status_text = f"⚠️ Expired ({cache_status['cache_age_hours']:.1f}h old)"
                    color = 'orange'
                else:
                    status_text = "❌ No cache"
                    color = 'red'
                
                self.cache_status_label.config(text=status_text, foreground=color)
        except Exception as e:
            self.cache_status_label.config(text="❌ Error", foreground='red')

    def update_recovery_status(self):
        """Update recovery status display"""
        try:
            if self.environment and hasattr(self.environment, 'in_recovery_mode'):
                # Recovery mode
                if self.environment.in_recovery_mode:
                    self.recovery_mode_label.config(text="🔄 RECOVERY MODE", foreground='red')
                else:
                    self.recovery_mode_label.config(text="🟢 Normal", foreground='green')
                
                # Recovery level
                recovery_text = f"{self.environment.recovery_level}/{self.environment.max_recovery_levels}"
                self.recovery_level_label.config(text=recovery_text)
                
                # Total P&L
                pnl = self.environment.total_pnl
                pnl_color = 'green' if pnl >= 0 else 'red'
                self.total_pnl_label.config(text=f"${pnl:.2f}", foreground=pnl_color)
                
                # Max Drawdown
                dd = self.environment.max_drawdown
                dd_color = 'red' if dd < -100 else 'orange' if dd < 0 else 'green'
                self.max_drawdown_label.config(text=f"${dd:.2f}", foreground=dd_color)
                
                # Recovery Success Rate
                recovery_attempts = self.environment.recovery_attempts
                successful_recoveries = self.environment.successful_recoveries
                if recovery_attempts > 0:
                    recovery_rate = (successful_recoveries / recovery_attempts) * 100
                    recovery_text = f"{successful_recoveries}/{recovery_attempts} ({recovery_rate:.0f}%)"
                else:
                    recovery_text = "0/0 (0%)"
                self.recovery_success_label.config(text=recovery_text)
                
                # Win Rate
                total_trades = self.environment.total_trades
                winning_trades = self.environment.winning_trades
                if total_trades > 0:
                    win_rate = (winning_trades / total_trades) * 100
                    win_text = f"{winning_trades}/{total_trades} ({win_rate:.0f}%)"
                    win_color = 'green' if win_rate >= 60 else 'orange' if win_rate >= 40 else 'red'
                else:
                    win_text = "0/0 (0%)"
                    win_color = 'gray'
                self.win_rate_label.config(text=win_text, foreground=win_color)
                
        except Exception as e:
            self.log_message(f"❌ Recovery status update error: {e}", "ERROR")

    def update_account_info(self):
        """Update account information display"""
        if not self.is_connected or not self.mt5_interface:
            return
        
        try:
            account_info = self.mt5_interface.get_account_info()
            if account_info:
                balance = account_info.get('balance', 0)
                equity = account_info.get('equity', 0)
                
                self.balance_label.config(text=f"${balance:.2f}")
                self.equity_label.config(text=f"${equity:.2f}")
            
            positions = self.mt5_interface.get_positions()
            self.positions_label.config(text=str(len(positions)))
            
        except Exception as e:
            self.log_message(f"❌ Account update error: {e}", "ERROR")

    def start_update_loop(self):
        """Start GUI update loop"""
        def update_loop():
            if self.running:
                try:
                    if self.is_connected:
                        self.update_account_info()
                        
                    # Update recovery status if environment exists
                    if self.environment:
                        self.update_recovery_status()
                        
                except:
                    pass  # Silent fail for updates
                
                # Schedule next update
                self.root.after(3000, update_loop)  # Update every 3 seconds for recovery info
        
        update_loop()

    def log_message(self, message, level="INFO"):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        try:
            self.log_text.insert(tk.END, log_entry)
            
            # Auto-scroll if enabled
            if self.auto_scroll_var.get():
                self.log_text.see(tk.END)
                
            # Print to console too
            print(log_entry.strip())
            
        except Exception as e:
            print(f"Log error: {e}")

    def clear_logs(self):
        """Clear log display"""
        try:
            self.log_text.delete(1.0, tk.END)
            self.log_message("🗑️ Logs cleared", "INFO")
        except Exception as e:
            print(f"Clear logs error: {e}")

    def on_closing(self):
        """Handle application closing"""
        try:
            self.running = False
            
            # Stop all operations
            self.is_training = False
            self.is_trading = False
            
            # Disconnect MT5
            if self.is_connected and self.mt5_interface:
                self.mt5_interface.disconnect()
            
            # Save config
            self.save_config()
            
            self.log_message("👋 Application closing...", "INFO")
            
        except Exception as e:
            print(f"Closing error: {e}")
        finally:
            self.root.destroy()

    def run(self):
        """Run the GUI application"""
        try:
            # Setup close handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Add welcome message
            self.log_message("🚀 AI Recovery Trading System started", "SUCCESS")
            self.log_message("📋 Connect to MT5 to begin", "INFO")
            
            # Start main loop
            self.root.mainloop()
            
        except Exception as e:
            print(f"GUI run error: {e}")
            raise e
        
    def _check_recovery_mode(self):
        """
        🔄 ตรวจสอบและจัดการ Recovery Mode
        
        เชื่อมกับ:
        - self.environment - recovery state variables
        - self.mt5_interface - current positions และ P&L
        - self.recovery_threshold_var (GUI) - จุดเริ่ม recovery
        
        หน้าที่:
        1. คำนวณ unrealized P&L รวม
        2. ตรวจสอบว่าควรเริ่ม recovery mode หรือไม่
        3. อัพเดท recovery level และ status
        4. แสดงสถานะ recovery ใน log
        """
        try:
            if not self.mt5_interface:
                return
                
            # ดึงข้อมูล positions ปัจจุบัน
            positions = self.mt5_interface.get_positions()
            total_unrealized_pnl = sum(pos.get('profit', 0) for pos in positions)
            
            # ดึงค่า recovery threshold จาก GUI
            recovery_threshold = self.recovery_threshold_var.get()  # เช่น -35.0
            
            # ตรวจสอบเงื่อนไข recovery
            should_enter_recovery = total_unrealized_pnl <= recovery_threshold
            
            # จัดการ recovery state
            if hasattr(self, 'in_recovery_mode'):
                current_recovery = self.in_recovery_mode
            else:
                self.in_recovery_mode = False
                self.recovery_level = 0
                current_recovery = False
            
            # เริ่ม recovery mode
            if should_enter_recovery and not current_recovery:
                self.in_recovery_mode = True
                self.recovery_level = 1
                self.recovery_start_pnl = total_unrealized_pnl
                self.log_message(f"🔄 RECOVERY MODE ACTIVATED - P&L: ${total_unrealized_pnl:.2f}", "WARNING")
                self.log_message(f"🎯 Recovery threshold: ${recovery_threshold:.2f}", "INFO")
                
            # อัพเดท recovery level
            elif self.in_recovery_mode:
                # เพิ่ม recovery level หาก P&L แย่ลง
                if total_unrealized_pnl < self.recovery_start_pnl - 20:  # ทุกๆ $20 ที่แย่ลง
                    max_recovery = self.max_recovery_levels_var.get()
                    if self.recovery_level < max_recovery:
                        self.recovery_level += 1
                        self.log_message(f"⬆️ Recovery Level increased to {self.recovery_level}/{max_recovery}", "WARNING")
                        
                # ตรวจสอบการออกจาก recovery mode
                if total_unrealized_pnl >= -5:  # P&L กลับมาใกล้ break-even
                    self.in_recovery_mode = False
                    self.recovery_level = 0
                    self.log_message(f"✅ RECOVERY SUCCESSFUL - P&L: ${total_unrealized_pnl:.2f}", "SUCCESS")
                    
            # แสดงสถานะ recovery
            if self.in_recovery_mode:
                max_recovery = self.max_recovery_levels_var.get()
                self.log_message(f"🔄 Recovery Mode: Level {self.recovery_level}/{max_recovery} | P&L: ${total_unrealized_pnl:.2f}", "INFO")
                
        except Exception as e:
            self.log_message(f"❌ Recovery mode check error: {e}", "ERROR")


    def _execute_ai_trade(self, action_type, position_size, predicted_action):
        """
        ⚡ ส่งคำสั่งเทรดจริงตาม AI decision
        
        เชื่อมกับ:
        - self.mt5_interface - ส่งคำสั่งไปยัง MT5
        - self.stop_loss_pips_var (GUI) - ค่า stop loss
        - predicted_action - ข้อมูลจาก AI model
        
        หน้าที่:
        1. เตรียมพารามิเตอร์การเทรด
        2. คำนวณ stop loss และ take profit
        3. ส่งคำสั่งไปยัง MT5
        4. ตรวจสอบผลการเทรดและ log
        """
        try:
            symbol = self.config.get('symbol', 'XAUUSD')
            
            # ดึงราคาปัจจุบัน
            current_price = self.mt5_interface.get_current_price(symbol)
            if not current_price:
                self.log_message("❌ Cannot get current price for trading", "ERROR")
                return False
                
            # เตรียมพารามิเตอร์
            sl_pips = self.stop_loss_pips_var.get()
            
            if action_type == 1:  # BUY
                order_type = "buy"
                entry_price = current_price['ask']
                sl_price = entry_price - (sl_pips * 0.01) if sl_pips > 0 else None
                tp_price = entry_price + (sl_pips * 2 * 0.01) if sl_pips > 0 else None  # 1:2 Risk:Reward
                
                self.log_message(f"🟢 BUY Signal: {position_size} lots at ${entry_price:.2f}", "INFO")
                
            elif action_type == 2:  # SELL
                order_type = "sell"
                entry_price = current_price['bid']
                sl_price = entry_price + (sl_pips * 0.01) if sl_pips > 0 else None
                tp_price = entry_price - (sl_pips * 2 * 0.01) if sl_pips > 0 else None  # 1:2 Risk:Reward
                
                self.log_message(f"🔴 SELL Signal: {position_size} lots at ${entry_price:.2f}", "INFO")
                
            else:
                self.log_message(f"❌ Unknown action type: {action_type}", "ERROR")
                return False
            
            # ปรับขนาด position หาก recovery mode
            if hasattr(self, 'in_recovery_mode') and self.in_recovery_mode:
                recovery_multiplier = self.recovery_multiplier_var.get()
                original_size = position_size
                position_size = position_size * (recovery_multiplier ** self.recovery_level)
                
                # จำกัดขนาดสูงสุด
                max_lot = self.max_lot_size_var.get()
                position_size = min(position_size, max_lot)
                
                self.log_message(f"🔄 Recovery Mode: {original_size:.2f} → {position_size:.2f} lots (Level {self.recovery_level})", "WARNING")
            
            # ส่งคำสั่งเทรด
            self.log_message(f"📤 Sending {order_type.upper()} order to MT5...", "INFO")
            
            success = self.mt5_interface.place_order(
                symbol=symbol,
                order_type=order_type,
                volume=position_size,
                price=entry_price,
                sl=sl_price,
                tp=tp_price,
                comment="AI Recovery Trading"
            )
            
            if success:
                self.log_message(f"✅ Order executed successfully!", "SUCCESS")
                
                # Log รายละเอียด
                if sl_price:
                    self.log_message(f"   Stop Loss: ${sl_price:.2f} ({sl_pips} pips)", "INFO")
                if tp_price:
                    self.log_message(f"   Take Profit: ${tp_price:.2f} ({sl_pips*2} pips)", "INFO")
                    
                return True
            else:
                error_msg = self.mt5_interface.get_last_error()
                self.log_message(f"❌ Order failed: {error_msg}", "ERROR")
                return False
                
        except Exception as e:
            self.log_message(f"❌ Execute trade error: {e}", "ERROR")
            return False


    def _close_all_positions(self):
        """
        🚪 ปิด positions ทั้งหมด
        
        เชื่อมกับ:
        - self.mt5_interface - ดึงและปิด positions
        
        หน้าที่:
        1. ดึงรายการ positions ทั้งหมด
        2. ปิดทีละ position
        3. นับจำนวนที่ปิดสำเร็จ
        4. รีเซ็ต recovery mode หากจำเป็น
        """
        try:
            if not self.mt5_interface:
                return 0
                
            positions = self.mt5_interface.get_positions()
            if not positions:
                self.log_message("ℹ️ No positions to close", "INFO")
                return 0
                
            self.log_message(f"🚪 Closing {len(positions)} positions...", "INFO")
            
            closed_count = 0
            total_profit = 0
            
            for position in positions:
                ticket = position.get('ticket')
                profit = position.get('profit', 0)
                symbol = position.get('symbol', '')
                volume = position.get('volume', 0)
                
                self.log_message(f"   Closing: {symbol} {volume} lots (P&L: ${profit:.2f})", "INFO")
                
                success = self.mt5_interface.close_position(ticket)
                
                if success:
                    closed_count += 1
                    total_profit += profit
                    self.log_message(f"   ✅ Position {ticket} closed", "SUCCESS")
                else:
                    error_msg = self.mt5_interface.get_last_error()
                    self.log_message(f"   ❌ Failed to close {ticket}: {error_msg}", "ERROR")
            
            if closed_count > 0:
                self.log_message(f"🏁 Closed {closed_count} positions, Total P&L: ${total_profit:.2f}", "SUCCESS")
                
                # รีเซ็ต recovery mode หากปิดทั้งหมด
                if closed_count == len(positions) and hasattr(self, 'in_recovery_mode'):
                    self.in_recovery_mode = False
                    self.recovery_level = 0
                    self.log_message("🔄 Recovery mode reset", "INFO")
            
            return closed_count
            
        except Exception as e:
            self.log_message(f"❌ Close all positions error: {e}", "ERROR")
            return 0


    def _monitor_open_positions(self):
        """
        👁️ ตรวจสอบ positions ที่เปิดอยู่
        
        เชื่อมกับ:
        - self.mt5_interface - ข้อมูล positions
        - GUI recovery status display
        
        หน้าที่:
        1. ดึงข้อมูล positions ปัจจุบัน
        2. คำนวณ P&L รวม
        3. ตรวจสอบ positions ที่มี risk สูง
        4. อัพเดทสถานะใน GUI
        """
        try:
            if not self.mt5_interface:
                return
                
            positions = self.mt5_interface.get_positions()
            
            if not positions:
                return
                
            total_profit = 0
            risk_positions = []
            
            for position in positions:
                profit = position.get('profit', 0)
                symbol = position.get('symbol', '')
                volume = position.get('volume', 0)
                total_profit += profit
                
                # ตรวจสอบ positions ที่ขาดทุนเกิน threshold
                if profit < -50:  # ขาดทุนเกิน $50
                    risk_positions.append({
                        'symbol': symbol,
                        'volume': volume,
                        'profit': profit,
                        'ticket': position.get('ticket')
                    })
            
            # แสดง warning สำหรับ positions เสี่ยง
            if risk_positions:
                self.log_message(f"⚠️ {len(risk_positions)} high-risk positions (loss > $50)", "WARNING")
                for pos in risk_positions:
                    self.log_message(f"   🔻 {pos['symbol']} {pos['volume']} lots: ${pos['profit']:.2f}", "WARNING")
            
            # อัพเดท recovery status ใน GUI
            self._update_recovery_display(total_profit, len(positions))
            
        except Exception as e:
            self.log_message(f"❌ Monitor positions error: {e}", "ERROR")


    def _update_recovery_display(self, total_pnl, position_count):
        """
        🖥️ อัพเดทการแสดงผล recovery status ใน GUI
        
        เชื่อมกับ:
        - GUI labels (recovery_mode_label, total_pnl_label, etc.)
        
        หน้าที่:
        1. อัพเดทสี recovery mode indicator
        2. แสดง P&L ปัจจุบัน
        3. แสดง recovery level
        4. อัพเดทจำนวน positions
        """
        try:
            # อัพเดท Recovery Mode Status
            if hasattr(self, 'in_recovery_mode') and self.in_recovery_mode:
                recovery_text = f"🔄 RECOVERY L{self.recovery_level}"
                recovery_color = 'red'
            else:
                recovery_text = "🟢 Normal"
                recovery_color = 'green'
                
            if hasattr(self, 'recovery_mode_label'):
                self.recovery_mode_label.config(text=recovery_text, foreground=recovery_color)
            
            # อัพเดท Total P&L
            pnl_color = 'green' if total_pnl >= 0 else 'red'
            if hasattr(self, 'total_pnl_label'):
                self.total_pnl_label.config(text=f"${total_pnl:.2f}", foreground=pnl_color)
            
            # อัพเดท Position Count
            if hasattr(self, 'positions_label'):
                self.positions_label.config(text=str(position_count))
            
            # อัพเดท Recovery Level
            if hasattr(self, 'recovery_level_label'):
                max_recovery = self.max_recovery_levels_var.get()
                level_text = f"{getattr(self, 'recovery_level', 0)}/{max_recovery}"
                self.recovery_level_label.config(text=level_text)
                
        except Exception as e:
            # Silent fail for GUI updates
            pass
    
    def _update_trade_statistics(self, trade_result):
        """
        📈 อัพเดท trading statistics
        
        เชื่อมกับ:
        - _execute_ai_trade() method
        - _close_all_positions() method
        
        หน้าที่:
        1. นับจำนวน trades
        2. แยก winning/losing trades
        3. อัพเดท win rate
        """
        try:
            self.total_trades_today += 1
            
            if trade_result > 0:
                self.winning_trades_today += 1
                self.log_message(f"✅ Winning trade #{self.total_trades_today}: ${trade_result:.2f}", "SUCCESS")
            elif trade_result < 0:
                self.losing_trades_today += 1
                self.log_message(f"❌ Losing trade #{self.total_trades_today}: ${trade_result:.2f}", "ERROR")
            
            # คำนวณ win rate
            if self.total_trades_today > 0:
                win_rate = (self.winning_trades_today / self.total_trades_today) * 100
                self.log_message(f"📊 Today's Win Rate: {self.winning_trades_today}/{self.total_trades_today} ({win_rate:.1f}%)", "INFO")
            
        except Exception as e:
            self.log_message(f"❌ Statistics update error: {e}", "ERROR")

    def show_ai_insights(self):
        """แสดง AI Market Insights"""
        try:
            if self.current_environment_type != "aggressive":
                messagebox.showinfo("Info", "AI Insights ใช้ได้เฉพาะ Aggressive Mode")
                return
            
            if not self.environment or not hasattr(self.environment, 'get_ai_insights'):
                messagebox.showwarning("Warning", "AI Environment not initialized")
                return
            
            # ดึงข้อมูล AI Insights
            insights = self.environment.get_ai_insights()
            
            if not insights:
                messagebox.showwarning("Warning", "No AI insights available")
                return
            
            # สร้าง Insights Window
            self._create_ai_insights_window(insights)
            
        except Exception as e:
            self.log_message(f"❌ Show AI insights error: {e}", "ERROR")

    def _create_ai_insights_window(self, insights):
        """สร้าง AI Insights Window"""
        try:
            # ปิด window เก่า (ถ้ามี)
            if self.ai_insights_window and self.ai_insights_window.winfo_exists():
                self.ai_insights_window.destroy()
            
            # สร้าง window ใหม่
            self.ai_insights_window = tk.Toplevel(self.root)
            self.ai_insights_window.title("🧠 AI Market Insights")
            self.ai_insights_window.geometry("600x500")
            self.ai_insights_window.configure(bg='#2b2b2b')
            
            # Main frame
            main_frame = ttk.Frame(self.ai_insights_window)
            main_frame.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Title
            title_label = ttk.Label(
                main_frame, 
                text="🧠 AI MARKET INSIGHTS", 
                font=self.fonts['header'],
                style='Header.TLabel'
            )
            title_label.pack(pady=(0, 10))
            
            # Insights Content
            content_frame = ttk.Frame(main_frame)
            content_frame.pack(fill='both', expand=True)
            
            # Text widget with scrollbar
            text_widget = tk.Text(
                content_frame, 
                wrap=tk.WORD, 
                font=self.fonts['console'],
                bg='#1e1e1e', 
                fg='white',
                relief='flat',
                borderwidth=1
            )
            scrollbar = ttk.Scrollbar(content_frame, orient='vertical', command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
            
            text_widget.pack(side='left', fill='both', expand=True)
            scrollbar.pack(side='right', fill='y')
            
            # Format insights text
            insights_text = self._format_ai_insights(insights)
            text_widget.insert(tk.END, insights_text)
            text_widget.config(state='disabled')
            
            # Buttons frame
            buttons_frame = ttk.Frame(main_frame)
            buttons_frame.pack(fill='x', pady=(10, 0))
            
            # Refresh button
            refresh_btn = ttk.Button(
                buttons_frame, 
                text="🔄 Refresh", 
                command=lambda: self._refresh_ai_insights(text_widget),
                style='Thai.TButton'
            )
            refresh_btn.pack(side='left', padx=5)
            
            # Close button
            close_btn = ttk.Button(
                buttons_frame, 
                text="❌ Close", 
                command=self.ai_insights_window.destroy,
                style='Thai.TButton'
            )
            close_btn.pack(side='right', padx=5)
            
        except Exception as e:
            self.log_message(f"❌ Create AI insights window error: {e}", "ERROR")

    def _format_ai_insights(self, insights):
        """จัดรูปแบบ AI Insights เป็นข้อความ"""
        try:
            text = f"""
    🧠 AI MARKET INSIGHTS
    {'='*60}

    📊 MARKET ANALYSIS:
    Market Regime: {insights.get('market_regime', 'Unknown')}
    Trading Session: {insights.get('trading_session', 'Unknown')}
    Volatility Level: {insights.get('volatility_level', 'Unknown')}
    Trend Direction: {insights.get('trend_direction', 'Unknown')}
    Confidence Level: {insights.get('confidence_level', 'Unknown')}

    🎯 RECOMMENDED STRATEGIES:
    """
            
            strategies = insights.get('recommended_strategies', [])
            for i, strategy in enumerate(strategies, 1):
                text += f"   {i}. {strategy}\n"
            
            # Warnings
            warnings = insights.get('warnings', [])
            if warnings:
                text += f"\n⚠️ WARNINGS:\n"
                for warning in warnings:
                    text += f"   • {warning}\n"
            
            # Opportunities
            opportunities = insights.get('opportunities', [])
            if opportunities:
                text += f"\n💡 OPPORTUNITIES:\n"
                for opp in opportunities:
                    text += f"   • {opp}\n"
            
            # Additional info
            text += f"\n📈 ADDITIONAL INFO:\n"
            text += f"   Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if hasattr(self.environment, 'get_ai_status'):
                ai_status = self.environment.get_ai_status()
                session_info = ai_status.get('session', {})
                if session_info:
                    text += f"   AI Session: {session_info.get('id', 'Unknown')}\n"
                    text += f"   Total Decisions: {ai_status.get('performance', {}).get('total_decisions', 0)}\n"
            
            return text
            
        except Exception as e:
            return f"❌ Error formatting insights: {e}"

    def _refresh_ai_insights(self, text_widget):
        """รีเฟรช AI Insights"""
        try:
            if not self.environment or not hasattr(self.environment, 'get_ai_insights'):
                return
            
            # ดึงข้อมูลใหม่
            insights = self.environment.get_ai_insights()
            
            if insights:
                # อัปเดต text widget
                text_widget.config(state='normal')
                text_widget.delete(1.0, tk.END)
                
                new_text = self._format_ai_insights(insights)
                text_widget.insert(tk.END, new_text)
                text_widget.config(state='disabled')
                
                self.log_message("🔄 AI Insights refreshed", "INFO")
            
        except Exception as e:
            self.log_message(f"❌ Refresh AI insights error: {e}", "ERROR")

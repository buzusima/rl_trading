import tkinter as tk
from tkinter import ttk, messagebox
import threading
import json
import os
from datetime import datetime

class TradingGUI:
    """
    Recovery Trading GUI - Fixed Version
    - Recovery strategy controls
    - Historical data management  
    - Real-time recovery monitoring
    - Smart cache integration
    """
    
    def __init__(self):
        print("🎨 Initializing Recovery Trading GUI...")
        
        # Main window
        self.root = tk.Tk()
        self.root.title("🤖 AI Recovery Trading System")
        self.root.geometry("900x700")
        self.root.configure(bg='#2b2b2b')
        
        # ✅ Configure Thai fonts for better readability
        self.setup_fonts()
        
        # System state
        self.is_connected = False
        self.is_training = False
        self.is_trading = False
        
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
        """Initialize all GUI variables"""
        # Basic config variables
        self.symbol_var = tk.StringVar(value=self.config.get('symbol', 'XAUUSD'))
        self.lot_size_var = tk.DoubleVar(value=self.config.get('lot_size', 0.01))
        self.max_positions_var = tk.IntVar(value=self.config.get('max_positions', 5))
        self.training_steps_var = tk.IntVar(value=self.config.get('training_steps', 10000))
        self.learning_rate_var = tk.DoubleVar(value=self.config.get('learning_rate', 0.0003))
        
        # Recovery config variables
        self.recovery_multiplier_var = tk.DoubleVar(value=self.config.get('recovery_multiplier', 1.5))
        self.recovery_threshold_var = tk.DoubleVar(value=self.config.get('recovery_threshold', -20.0))
        self.max_recovery_levels_var = tk.IntVar(value=self.config.get('max_recovery_levels', 3))
        # Removed max_drawdown_limit_var
        
        # Auto-scroll variable
        self.auto_scroll_var = tk.BooleanVar(value=True)

    def load_config(self):
        """Load configuration including recovery settings"""
        default_config = {
            'symbol': 'XAUUSD.v',  # ✅ Updated to correct symbol
            'lot_size': 0.02,      # ✅ Updated for $4000 account
            'max_positions': 3,
            'training_steps': 10000,
            'learning_rate': 0.0003,
            # Recovery settings
            'recovery_multiplier': 1.4,     # ✅ Updated for $4000 account
            'recovery_threshold': -35.0,    # ✅ Updated for $4000 account  
            'max_recovery_levels': 3
            # Removed max_drawdown_limit
        }
        
        try:
            config_file = 'config/config.json'
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    print("✅ Configuration loaded (including recovery settings)")
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
        """Setup main control tab with recovery status"""
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

    def setup_training_tab(self):
        """Setup training controls tab with recovery settings"""
        # === CONFIGURATION SECTION ===
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
        ttk.Label(config_grid, text="Training Steps:").grid(row=3, column=0, sticky='w')
        training_steps_entry = ttk.Entry(config_grid, textvariable=self.training_steps_var, width=10)
        training_steps_entry.grid(row=3, column=1, sticky='w', padx=(10, 0))
        
        # Learning rate
        ttk.Label(config_grid, text="Learning Rate:").grid(row=4, column=0, sticky='w')
        lr_entry = ttk.Entry(config_grid, textvariable=self.learning_rate_var, width=10)
        lr_entry.grid(row=4, column=1, sticky='w', padx=(10, 0))
        
        # === RECOVERY STRATEGY SECTION ===
        recovery_frame = ttk.LabelFrame(self.training_frame, text="🔄 Recovery Strategy (แก้ไม้)")
        recovery_frame.pack(fill='x', padx=10, pady=5)
        
        recovery_grid = ttk.Frame(recovery_frame)
        recovery_grid.pack(fill='x', padx=10, pady=10)
        
        # Recovery multiplier
        ttk.Label(recovery_grid, text="Recovery Multiplier:", style='Thai.TLabel').grid(row=0, column=0, sticky='w')
        recovery_mult_entry = ttk.Entry(recovery_grid, textvariable=self.recovery_multiplier_var, width=10, style='Thai.TEntry')
        recovery_mult_entry.grid(row=0, column=1, sticky='w', padx=(10, 0))
        ttk.Label(recovery_grid, text="(1.5 = เพิ่ม lot 1.5 เท่า)", style='Comment.TLabel').grid(row=0, column=2, sticky='w', padx=(5, 0))
        
        # Recovery threshold
        ttk.Label(recovery_grid, text="Recovery Threshold:", style='Thai.TLabel').grid(row=1, column=0, sticky='w')
        recovery_thresh_entry = ttk.Entry(recovery_grid, textvariable=self.recovery_threshold_var, width=10, style='Thai.TEntry')
        recovery_thresh_entry.grid(row=1, column=1, sticky='w', padx=(10, 0))
        ttk.Label(recovery_grid, text="($ ขาดทุนเท่าไหร่เริ่มแก้ไม้)", style='Comment.TLabel').grid(row=1, column=2, sticky='w', padx=(5, 0))
        
        # Max recovery levels
        ttk.Label(recovery_grid, text="Max Recovery Levels:", style='Thai.TLabel').grid(row=2, column=0, sticky='w')
        max_recovery_entry = ttk.Entry(recovery_grid, textvariable=self.max_recovery_levels_var, width=10, style='Thai.TEntry')
        max_recovery_entry.grid(row=2, column=1, sticky='w', padx=(10, 0))
        ttk.Label(recovery_grid, text="(แก้ไม้สูงสุดกี่ระดับ)", style='Comment.TLabel').grid(row=2, column=2, sticky='w', padx=(5, 0))
        
        # Note about no drawdown limit
        ttk.Label(recovery_grid, text="Drawdown Limit:", style='Thai.TLabel').grid(row=3, column=0, sticky='w')
        ttk.Label(recovery_grid, text="🚀 UNLIMITED", foreground='blue', style='Header.TLabel').grid(row=3, column=1, sticky='w', padx=(10, 0))
        ttk.Label(recovery_grid, text="(ไม่มีการหยุดเทรด - แก้ไม้ไม่จำกัด)", style='Comment.TLabel').grid(row=3, column=2, sticky='w', padx=(5, 0))
        
        # === DATA SETTINGS SECTION ===
        data_frame = ttk.LabelFrame(self.training_frame, text="📊 Historical Data Settings")
        data_frame.pack(fill='x', padx=10, pady=5)
        
        data_grid = ttk.Frame(data_frame)
        data_grid.pack(fill='x', padx=10, pady=10)
        
        # Timeframe (readonly - M5 fixed)
        ttk.Label(data_grid, text="Timeframe:").grid(row=0, column=0, sticky='w')
        ttk.Label(data_grid, text="M5 (5 นาที)", foreground='blue').grid(row=0, column=1, sticky='w', padx=(10, 0))
        
        # Lookback period
        ttk.Label(data_grid, text="Lookback Period:").grid(row=1, column=0, sticky='w')
        ttk.Label(data_grid, text="2 Years", foreground='blue').grid(row=1, column=1, sticky='w', padx=(10, 0))
        
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
        
        # === TRAINING CONTROLS SECTION ===
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
        """Save current configuration including recovery settings"""
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
                'max_recovery_levels': self.max_recovery_levels_var.get()
                # Removed max_drawdown_limit
            })
            
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.log_message("✅ Configuration saved (including recovery settings)", "SUCCESS")
            
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
            self.log_message("🏗️ Initializing recovery trading system...", "INFO")
            
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
            
            # Create environment
            from core.environment import Environment
            self.environment = Environment(self.mt5_interface, self.config)
            self.log_message("✅ Recovery environment created", "SUCCESS")
            
            # Create agent
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
                
            if self.agent is None:
                self.initialize_trading_system()
            
            self.log_message("🎓 Starting recovery training...", "INFO")
            self.is_training = True
            self.start_training_btn.config(state='disabled')
            self.stop_training_btn.config(state='normal')
            self.training_status_label.config(text="Training...")
            
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
        """Start live trading with auto-loaded model"""
        try:
            if not self.is_connected:
                messagebox.showwarning("Warning", "Please connect to MT5 first")
                return
            
            if self.agent is None:
                messagebox.showwarning("Warning", "Trading system not initialized")
                return
            
            # Double-check model is loaded
            if not self.agent.is_trained:
                self.log_message("⚠️ No trained model detected, attempting to load...", "WARNING")
                success = self.agent.load_model()
                if not success:
                    messagebox.showerror("Error", "No trained model found. Please train or load a model first.")
                    return
                else:
                    self.model_status_label.config(text="✅ Model Loaded", foreground='green')
            
            # Start trading
            self.is_trading = True
            self.start_trading_btn.config(state='disabled')
            self.stop_trading_btn.config(state='normal')
            self.log_message("🚀 Recovery trading started with loaded model", "SUCCESS")
            
            # Start trading loop
            self.trading_thread = threading.Thread(target=self._trading_worker, daemon=True)
            self.trading_thread.start()
            
        except Exception as e:
            self.log_message(f"❌ Trading start error: {e}", "ERROR")

    def _trading_worker(self):
        """Trading worker thread - implement your trading logic here"""
        try:
            import time
            
            while self.is_trading and self.running:
                # Get current observation
                # Make prediction
                # Execute trade if needed
                # This is where you'd implement the actual trading loop
                
                self.log_message("💹 Trading cycle...", "INFO")
                time.sleep(5)  # Placeholder - replace with actual trading logic
                
        except Exception as e:
            self.log_message(f"❌ Trading worker error: {e}", "ERROR")

    def stop_trading(self):
        """Stop live trading"""
        self.is_trading = False
        self.start_trading_btn.config(state='normal')
        self.stop_trading_btn.config(state='disabled')
        self.log_message("⏹️ Recovery trading stopped", "INFO")

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
            
            self.reset_training_controls()
            self.stop_trading()
            
            self.log_message("🛑 EMERGENCY STOP ACTIVATED", "ERROR")
            messagebox.showwarning("Emergency Stop", "All operations stopped!")
            
        except Exception as e:
            self.log_message(f"❌ Emergency stop error: {e}", "ERROR")

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
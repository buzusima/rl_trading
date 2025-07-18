# main.py - Enhanced Trading GUI Application
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
# แก้ไข import สำหรับ matplotlib เวอร์ชั่นใหม่
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
except ImportError:
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkinter as FigureCanvas
    except ImportError:
        print("Error: Cannot import matplotlib backend for tkinter")
        FigureCanvas = None
                
import numpy as np
import time

from environment import TradingEnvironment, TradingState
from recovery_engine import RecoveryEngine
from mt5_interface import MT5Interface
from rl_agent import RLAgent
from utils.data_handler import DataHandler
from utils.visualizer import Visualizer
from portfolio_manager import AIPortfolioManager

class TradingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 AI Trading System - XAUUSD Pro")
        self.root.geometry("1400x900")
        
        # Initialize components
        self.mt5_interface = MT5Interface()
        self.recovery_engine = RecoveryEngine()
        self.data_handler = DataHandler()
        self.visualizer = Visualizer()
        
        # System state
        self.is_training = False
        self.is_trading = False
        self.is_connected = False
        
        # Configuration
        self.config = self.load_config()
        
        # Portfolio Manager
        self.portfolio_manager = AIPortfolioManager(self.config)
        
        # AI monitoring
        self.ai_decision_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0, 'CLOSE': 0, 'HEDGE': 0}
        self.ai_override_count = 0
        self.ai_total_rewards = []
        self.ai_decision_labels = {}
        self.recent_actions = []
        
        # Trading state tracking
        self.current_trading_state = TradingState.ANALYZE
        self.state_start_time = time.time()
        
        # Initialize GUI components
        self.setup_gui()
        
        # Initialize RL components
        self.trading_env = None
        self.rl_agent = None
        
        # Threading for real-time updates
        self.update_thread = None
        self.training_thread = None
        
        # Start real-time updates
        self.start_real_time_updates()
    
    def setup_gui(self):
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Enhanced Dashboard Tab
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="🎯 Dashboard")
        
        # Configuration Tab
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="⚙️ Configuration")
        
        # Training Tab
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="🎓 Training")
        
        # Performance Tab
        self.performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.performance_frame, text="📊 Performance")
        
        # Portfolio Tab
        self.portfolio_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.portfolio_frame, text="💼 Portfolio")
        
        # Logs Tab
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="📝 Logs")
        
        self.notebook.pack(fill='both', expand=True)
        
        # Setup individual tabs
        self.setup_enhanced_dashboard()
        self.setup_enhanced_configuration()
        self.setup_enhanced_training()
        self.setup_portfolio_tab()
        self.setup_performance_tab()
        self.setup_enhanced_logs()
        
    def setup_enhanced_dashboard(self):
        # Main container with two columns
        main_container = ttk.Frame(self.main_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left column
        left_column = ttk.Frame(main_container)
        left_column.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Right column
        right_column = ttk.Frame(main_container)
        right_column.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # === LEFT COLUMN ===
        
        # Connection & Control Frame
        control_frame = ttk.LabelFrame(left_column, text="🔌 Connection & Control")
        control_frame.pack(fill='x', pady=(0, 10))
        
        # Connection status
        conn_container = ttk.Frame(control_frame)
        conn_container.pack(fill='x', padx=10, pady=5)
        
        self.conn_status = tk.Label(conn_container, text="❌ Disconnected", fg="red", font=('Arial', 12, 'bold'))
        self.conn_status.pack(side='left')
        
        self.connect_btn = tk.Button(conn_container, text="🔗 Connect MT5", 
                                   command=self.connect_mt5, bg='lightblue', font=('Arial', 10, 'bold'))
        self.connect_btn.pack(side='right')
        
        # Trading controls
        trading_container = ttk.Frame(control_frame)
        trading_container.pack(fill='x', padx=10, pady=5)
        
        self.start_trading_btn = tk.Button(trading_container, text="🚀 Start AI Trading", 
                                         command=self.start_trading, state='disabled',
                                         bg='lightgreen', font=('Arial', 10, 'bold'))
        self.start_trading_btn.pack(side='left', padx=5)
        
        self.stop_trading_btn = tk.Button(trading_container, text="⏹️ Stop Trading", 
                                        command=self.stop_trading, state='disabled',
                                        bg='lightcoral', font=('Arial', 10, 'bold'))
        self.stop_trading_btn.pack(side='left', padx=5)
        
        # Trading State Frame
        state_frame = ttk.LabelFrame(left_column, text="🎯 AI Trading State")
        state_frame.pack(fill='x', pady=(0, 10))
        
        # Current state display
        self.current_state_label = tk.Label(state_frame, text="State: ANALYZE", 
                                          font=('Arial', 14, 'bold'), fg='blue')
        self.current_state_label.pack(pady=5)
        
        # State description
        self.state_description = tk.Label(state_frame, text="🔍 Analyzing market for opportunities...", 
                                        font=('Arial', 10), wraplength=300)
        self.state_description.pack(pady=5)
        
        # State timer
        self.state_timer_label = tk.Label(state_frame, text="Time in state: 0s", font=('Arial', 9))
        self.state_timer_label.pack()
        
        # Position Information Frame
        pos_frame = ttk.LabelFrame(left_column, text="📈 Position Information")
        pos_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Position table
        pos_container = ttk.Frame(pos_frame)
        pos_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.pos_tree = ttk.Treeview(pos_container, columns=('Symbol', 'Type', 'Lots', 'Price', 'PnL', 'Target'), show='headings', height=8)
        self.pos_tree.heading('Symbol', text='Symbol')
        self.pos_tree.heading('Type', text='Type')
        self.pos_tree.heading('Lots', text='Lots')
        self.pos_tree.heading('Price', text='Price')
        self.pos_tree.heading('PnL', text='PnL')
        self.pos_tree.heading('Target', text='Target')
        
        # Column widths
        self.pos_tree.column('Symbol', width=80)
        self.pos_tree.column('Type', width=60)
        self.pos_tree.column('Lots', width=60)
        self.pos_tree.column('Price', width=80)
        self.pos_tree.column('PnL', width=80)
        self.pos_tree.column('Target', width=80)
        
        self.pos_tree.pack(side='left', fill='both', expand=True)
        
        # Scrollbar for position table
        pos_scrollbar = ttk.Scrollbar(pos_container, orient='vertical', command=self.pos_tree.yview)
        self.pos_tree.configure(yscrollcommand=pos_scrollbar.set)
        pos_scrollbar.pack(side='right', fill='y')
        
        # === RIGHT COLUMN ===
        
        # AI Intelligence Monitor
        ai_frame = ttk.LabelFrame(right_column, text="🤖 AI Intelligence Monitor")
        ai_frame.pack(fill='x', pady=(0, 10))
        
        # AI decision stats
        ai_stats_frame = ttk.Frame(ai_frame)
        ai_stats_frame.pack(fill='x', padx=10, pady=10)
        
        # Decision counters
        decision_container = ttk.Frame(ai_stats_frame)
        decision_container.pack(fill='x')
        
        for i, (action, count) in enumerate(self.ai_decision_counts.items()):
            row = i // 3
            col = i % 3
            label = tk.Label(decision_container, text=f"{action}: {count}", 
                           relief='sunken', width=12, bg='lightgray', font=('Arial', 9, 'bold'))
            label.grid(row=row, column=col, padx=2, pady=2, sticky='ew')
            self.ai_decision_labels[action] = label
            
        # Configure grid weights
        for i in range(3):
            decision_container.columnconfigure(i, weight=1)
        
        # AI performance metrics
        perf_container = ttk.Frame(ai_stats_frame)
        perf_container.pack(fill='x', pady=(10, 0))
        
        self.ai_override_label = tk.Label(perf_container, text="🧠 Smart Overrides: 0", 
                                        fg='blue', font=('Arial', 10, 'bold'))
        self.ai_override_label.pack(anchor='w')
        
        self.ai_reward_label = tk.Label(perf_container, text="📊 Avg Reward: 0.00", 
                                      fg='green', font=('Arial', 10, 'bold'))
        self.ai_reward_label.pack(anchor='w')
        
        self.ai_status_label = tk.Label(perf_container, text="🤖 AI Status: Not Trained", 
                                      fg='red', font=('Arial', 10, 'bold'))
        self.ai_status_label.pack(anchor='w')
        
        # Account Information Frame
        account_frame = ttk.LabelFrame(right_column, text="💰 Account Information")
        account_frame.pack(fill='x', pady=(0, 10))
        
        account_container = ttk.Frame(account_frame)
        account_container.pack(fill='x', padx=10, pady=10)
        
        # Account metrics in grid
        self.balance_label = tk.Label(account_container, text="Balance: $0.00", font=('Arial', 11, 'bold'))
        self.balance_label.grid(row=0, column=0, sticky='w', padx=5, pady=2)
        
        self.equity_label = tk.Label(account_container, text="Equity: $0.00", font=('Arial', 11, 'bold'))
        self.equity_label.grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        self.margin_label = tk.Label(account_container, text="Margin: $0.00", font=('Arial', 11))
        self.margin_label.grid(row=1, column=0, sticky='w', padx=5, pady=2)
        
        self.free_margin_label = tk.Label(account_container, text="Free: $0.00", font=('Arial', 11))
        self.free_margin_label.grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        # Recovery & Portfolio Status Frame
        status_frame = ttk.LabelFrame(right_column, text="🛡️ Recovery & Portfolio Status")
        status_frame.pack(fill='x', pady=(0, 10))
        
        status_container = ttk.Frame(status_frame)
        status_container.pack(fill='x', padx=10, pady=10)
        
        # Recovery status
        self.recovery_status = tk.Label(status_container, text="Recovery: Inactive", font=('Arial', 11))
        self.recovery_status.grid(row=0, column=0, sticky='w', pady=2)
        
        self.recovery_level = tk.Label(status_container, text="Level: 0", font=('Arial', 11))
        self.recovery_level.grid(row=0, column=1, sticky='w', pady=2)
        
        # Portfolio status
        self.portfolio_heat_label = tk.Label(status_container, text="Portfolio Heat: 0%", font=('Arial', 11))
        self.portfolio_heat_label.grid(row=1, column=0, sticky='w', pady=2)
        
        self.profit_target_label = tk.Label(status_container, text="Profit Target: $10", font=('Arial', 11))
        self.profit_target_label.grid(row=1, column=1, sticky='w', pady=2)
        
        # Market Information Frame
        market_frame = ttk.LabelFrame(right_column, text="📊 Market Information")
        market_frame.pack(fill='both', expand=True)
        
        market_container = ttk.Frame(market_frame)
        market_container.pack(fill='x', padx=10, pady=10)
        
        self.spread_label = tk.Label(market_container, text="Spread: Loading...", font=('Arial', 11))
        self.spread_label.grid(row=0, column=0, sticky='w', pady=2)
        
        self.price_label = tk.Label(market_container, text="Price: Loading...", font=('Arial', 11))
        self.price_label.grid(row=0, column=1, sticky='w', pady=2)
        
        self.net_pnl_label = tk.Label(market_container, text="Net PnL: $0.00", font=('Arial', 11, 'bold'))
        self.net_pnl_label.grid(row=1, column=0, columnspan=2, sticky='w', pady=2)
        
        # Quick Action Buttons
        action_frame = ttk.LabelFrame(right_column, text="⚡ Quick Actions")
        action_frame.pack(fill='x', pady=(10, 0))
        
        action_container = ttk.Frame(action_frame)
        action_container.pack(fill='x', padx=10, pady=10)
        
        close_all_btn = tk.Button(action_container, text="💰 Close All Profitable", 
                                command=self.close_profitable_positions, bg='gold', font=('Arial', 9, 'bold'))
        close_all_btn.pack(side='left', padx=5)
        
        emergency_close_btn = tk.Button(action_container, text="🚨 Emergency Close", 
                                      command=self.emergency_close_all, bg='red', fg='white', font=('Arial', 9, 'bold'))
        emergency_close_btn.pack(side='right', padx=5)
    
    def setup_enhanced_configuration(self):
        # Create scrollable frame
        canvas = tk.Canvas(self.config_frame)
        scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Trading Parameters Frame
        trading_params = ttk.LabelFrame(scrollable_frame, text="📈 Trading Parameters")
        trading_params.pack(fill='x', padx=10, pady=5)
        
        params_grid = ttk.Frame(trading_params)
        params_grid.pack(fill='x', padx=10, pady=10)
        
        # Symbol
        tk.Label(params_grid, text="Symbol:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', padx=5, pady=3)
        self.symbol_var = tk.StringVar(value=self.config.get('symbol', 'XAUUSD'))
        symbol_entry = tk.Entry(params_grid, textvariable=self.symbol_var, font=('Arial', 10))
        symbol_entry.grid(row=0, column=1, padx=5, pady=3, sticky='ew')
        
        # Initial Lot Size
        tk.Label(params_grid, text="Initial Lot Size:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w', padx=5, pady=3)
        self.lot_size_var = tk.DoubleVar(value=self.config.get('initial_lot_size', 0.01))
        lot_entry = tk.Entry(params_grid, textvariable=self.lot_size_var, font=('Arial', 10))
        lot_entry.grid(row=1, column=1, padx=5, pady=3, sticky='ew')
        
        # Max Positions
        tk.Label(params_grid, text="Max Positions:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', padx=5, pady=3)
        self.max_positions_var = tk.IntVar(value=self.config.get('max_positions', 10))
        max_pos_entry = tk.Entry(params_grid, textvariable=self.max_positions_var, font=('Arial', 10))
        max_pos_entry.grid(row=2, column=1, padx=5, pady=3, sticky='ew')
        
        # Configure grid weights
        params_grid.columnconfigure(1, weight=1)
        
        # AI Learning Parameters Frame
        ai_params = ttk.LabelFrame(scrollable_frame, text="🤖 AI Learning Parameters")
        ai_params.pack(fill='x', padx=10, pady=5)
        
        ai_grid = ttk.Frame(ai_params)
        ai_grid.pack(fill='x', padx=10, pady=10)
        
        # Algorithm
        tk.Label(ai_grid, text="Algorithm:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', padx=5, pady=3)
        self.algorithm_var = tk.StringVar(value=self.config.get('algorithm', 'PPO'))
        algo_combo = ttk.Combobox(ai_grid, textvariable=self.algorithm_var, 
                                values=['PPO', 'DQN', 'A2C'], font=('Arial', 10))
        algo_combo.grid(row=0, column=1, padx=5, pady=3, sticky='ew')
        
        # Learning Rate
        tk.Label(ai_grid, text="Learning Rate:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w', padx=5, pady=3)
        self.learning_rate_var = tk.DoubleVar(value=self.config.get('learning_rate', 0.001))
        lr_entry = tk.Entry(ai_grid, textvariable=self.learning_rate_var, font=('Arial', 10))
        lr_entry.grid(row=1, column=1, padx=5, pady=3, sticky='ew')
        
        # Training Steps
        tk.Label(ai_grid, text="Training Steps:", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', padx=5, pady=3)
        self.training_steps_var = tk.IntVar(value=self.config.get('training_steps', 50000))
        steps_entry = tk.Entry(ai_grid, textvariable=self.training_steps_var, font=('Arial', 10))
        steps_entry.grid(row=2, column=1, padx=5, pady=3, sticky='ew')
        
        # Training Mode
        self.training_mode_var = tk.BooleanVar(value=self.config.get('training_mode', True))
        training_check = tk.Checkbutton(ai_grid, text="🎓 Training Mode (Simulation)", 
                                      variable=self.training_mode_var, font=('Arial', 10, 'bold'))
        training_check.grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ai_grid.columnconfigure(1, weight=1)
        
        # Portfolio Management Parameters
        portfolio_params = ttk.LabelFrame(scrollable_frame, text="💼 Portfolio Management")
        portfolio_params.pack(fill='x', padx=10, pady=5)
        
        portfolio_grid = ttk.Frame(portfolio_params)
        portfolio_grid.pack(fill='x', padx=10, pady=10)
        
        # Risk per trade
        tk.Label(portfolio_grid, text="Risk per Trade (%):", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', padx=5, pady=3)
        self.risk_per_trade_var = tk.DoubleVar(value=self.config.get('base_risk_per_trade', 1.0))
        risk_entry = tk.Entry(portfolio_grid, textvariable=self.risk_per_trade_var, font=('Arial', 10))
        risk_entry.grid(row=0, column=1, padx=5, pady=3, sticky='ew')
        
        # Max portfolio risk
        tk.Label(portfolio_grid, text="Max Portfolio Risk (%):", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w', padx=5, pady=3)
        self.max_portfolio_risk_var = tk.DoubleVar(value=self.config.get('max_portfolio_risk', 5.0))
        max_risk_entry = tk.Entry(portfolio_grid, textvariable=self.max_portfolio_risk_var, font=('Arial', 10))
        max_risk_entry.grid(row=1, column=1, padx=5, pady=3, sticky='ew')
        
        # Profit target per lot
        tk.Label(portfolio_grid, text="Profit per 0.01 Lot ($):", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', padx=5, pady=3)
        self.profit_per_lot_var = tk.DoubleVar(value=self.config.get('profit_per_lot_target', 5.0))
        profit_lot_entry = tk.Entry(portfolio_grid, textvariable=self.profit_per_lot_var, font=('Arial', 10))
        profit_lot_entry.grid(row=2, column=1, padx=5, pady=3, sticky='ew')
        
        portfolio_grid.columnconfigure(1, weight=1)
        
        # Recovery Parameters Frame
        recovery_params = ttk.LabelFrame(scrollable_frame, text="🛡️ Recovery Parameters")
        recovery_params.pack(fill='x', padx=10, pady=5)
        
        recovery_grid = ttk.Frame(recovery_params)
        recovery_grid.pack(fill='x', padx=10, pady=10)
        
        # Recovery Type
        tk.Label(recovery_grid, text="Recovery Type:", font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', padx=5, pady=3)
        self.recovery_type_var = tk.StringVar(value=self.config.get('recovery_type', 'combined'))
        recovery_combo = ttk.Combobox(recovery_grid, textvariable=self.recovery_type_var, 
                                    values=['martingale', 'grid', 'hedge', 'combined'], font=('Arial', 10))
        recovery_combo.grid(row=0, column=1, padx=5, pady=3, sticky='ew')
        
        # Martingale Multiplier
        tk.Label(recovery_grid, text="Martingale Multiplier:", font=('Arial', 10, 'bold')).grid(row=1, column=0, sticky='w', padx=5, pady=3)
        self.martingale_mult_var = tk.DoubleVar(value=self.config.get('martingale_multiplier', 2.0))
        mart_entry = tk.Entry(recovery_grid, textvariable=self.martingale_mult_var, font=('Arial', 10))
        mart_entry.grid(row=1, column=1, padx=5, pady=3, sticky='ew')
        
        # Min Profit Target
        tk.Label(recovery_grid, text="Min Profit Target ($):", font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', padx=5, pady=3)
        self.min_profit_var = tk.DoubleVar(value=self.config.get('min_profit_target', 10))
        min_profit_entry = tk.Entry(recovery_grid, textvariable=self.min_profit_var, font=('Arial', 10))
        min_profit_entry.grid(row=2, column=1, padx=5, pady=3, sticky='ew')
        
        recovery_grid.columnconfigure(1, weight=1)
        
        # Quick Profit Mode
        self.quick_profit_var = tk.BooleanVar(value=self.config.get('quick_profit_mode', True))
        quick_profit_check = tk.Checkbutton(recovery_grid, text="⚡ Quick Profit Mode", 
                                          variable=self.quick_profit_var, font=('Arial', 10, 'bold'))
        quick_profit_check.grid(row=3, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        # Configuration Buttons
        config_buttons = ttk.Frame(scrollable_frame)
        config_buttons.pack(fill='x', padx=10, pady=20)
        
        save_btn = tk.Button(config_buttons, text="💾 Save Configuration", 
                           command=self.save_config, bg='lightgreen', font=('Arial', 11, 'bold'))
        save_btn.pack(side='left', padx=5)
        
        load_btn = tk.Button(config_buttons, text="📂 Load Configuration", 
                           command=self.load_config_dialog, bg='lightblue', font=('Arial', 11, 'bold'))
        load_btn.pack(side='left', padx=5)
        
        reset_btn = tk.Button(config_buttons, text="🔄 Reset to Default", 
                            command=self.reset_config, bg='lightyellow', font=('Arial', 11, 'bold'))
        reset_btn.pack(side='left', padx=5)
        
        apply_btn = tk.Button(config_buttons, text="✅ Apply Settings", 
                            command=self.apply_all_settings, bg='orange', font=('Arial', 11, 'bold'))
        apply_btn.pack(side='right', padx=5)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def setup_enhanced_training(self):
        # Training Control Frame
        training_control = ttk.LabelFrame(self.training_frame, text="🎓 Training Control")
        training_control.pack(fill='x', padx=10, pady=10)
        
        control_container = ttk.Frame(training_control)
        control_container.pack(fill='x', padx=10, pady=10)
        
        # Training buttons
        self.start_training_btn = tk.Button(control_container, text="🚀 Start Training", 
                                          command=self.start_training, bg='lightgreen', font=('Arial', 11, 'bold'))
        self.start_training_btn.pack(side='left', padx=5)
        
        self.stop_training_btn = tk.Button(control_container, text="⏹️ Stop Training", 
                                         command=self.stop_training, state='disabled',
                                         bg='lightcoral', font=('Arial', 11, 'bold'))
        self.stop_training_btn.pack(side='left', padx=5)
        
        self.quick_train_btn = tk.Button(control_container, text="⚡ Quick Train (1K)", 
                                       command=self.quick_train, bg='orange', font=('Arial', 11, 'bold'))
        self.quick_train_btn.pack(side='left', padx=5)
        
        self.save_model_btn = tk.Button(control_container, text="💾 Save Model", 
                                      command=self.save_model_manual, bg='gold', font=('Arial', 11, 'bold'))
        self.save_model_btn.pack(side='right', padx=5)
        
        # Training Progress Frame
        progress_frame = ttk.LabelFrame(self.training_frame, text="📊 Training Progress")
        progress_frame.pack(fill='x', padx=10, pady=10)
        
        progress_container = ttk.Frame(progress_frame)
        progress_container.pack(fill='x', padx=10, pady=10)
        
        self.progress_bar = ttk.Progressbar(progress_container, mode='determinate')
        self.progress_bar.pack(fill='x', pady=5)
        
        self.progress_label = tk.Label(progress_container, text="Ready to start training", 
                                     font=('Arial', 11, 'bold'))
        self.progress_label.pack(pady=5)
        
        # Training Statistics Frame
        stats_frame = ttk.LabelFrame(self.training_frame, text="📈 Training Statistics")
        stats_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create text widget with scrollbar
        stats_container = ttk.Frame(stats_frame)
        stats_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.stats_text = tk.Text(stats_container, height=20, font=('Consolas', 10),
                                bg='#f8f9fa', fg='#212529', wrap='word')
        stats_scrollbar = ttk.Scrollbar(stats_container, orient='vertical', command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side='left', fill='both', expand=True)
        stats_scrollbar.pack(side='right', fill='y')
    
    def setup_portfolio_tab(self):
        # Portfolio Overview Frame
        overview_frame = ttk.LabelFrame(self.portfolio_frame, text="💼 Portfolio Overview")
        overview_frame.pack(fill='x', padx=10, pady=10)
        
        overview_container = ttk.Frame(overview_frame)
        overview_container.pack(fill='x', padx=10, pady=10)
        
        # Capital information
        capital_frame = ttk.Frame(overview_container)
        capital_frame.pack(fill='x', pady=5)
        
        self.capital_label = tk.Label(capital_frame, text="Capital: $0", font=('Arial', 14, 'bold'))
        self.capital_label.pack(side='left')
        
        self.drawdown_label = tk.Label(capital_frame, text="Drawdown: 0%", font=('Arial', 12))
        self.drawdown_label.pack(side='right')
        
        # Dynamic targets
        targets_frame = ttk.Frame(overview_container)
        targets_frame.pack(fill='x', pady=5)
        
        self.profit_per_lot_label = tk.Label(targets_frame, text="Profit/Lot: $5.00", font=('Arial', 12))
        self.profit_per_lot_label.pack(side='left')
        
        self.portfolio_target_label = tk.Label(targets_frame, text="Portfolio Target: $25.00", font=('Arial', 12))
        self.portfolio_target_label.pack(side='right')
        
        # Risk Management Frame
        risk_frame = ttk.LabelFrame(self.portfolio_frame, text="⚠️ Risk Management")
        risk_frame.pack(fill='x', padx=10, pady=10)
        
        risk_container = ttk.Frame(risk_frame)
        risk_container.pack(fill='x', padx=10, pady=10)
        
        # Risk indicators
        self.trading_allowed_label = tk.Label(risk_container, text="Trading: ✅ Allowed", 
                                            font=('Arial', 12, 'bold'), fg='green')
        self.trading_allowed_label.pack(anchor='w')
        
        self.risk_mode_label = tk.Label(risk_container, text="Risk Mode: Normal", font=('Arial', 11))
        self.risk_mode_label.pack(anchor='w')
        
        self.daily_pnl_label = tk.Label(risk_container, text="Daily PnL: 0%", font=('Arial', 11))
        self.daily_pnl_label.pack(anchor='w')
        
        # Position Analysis Frame
        analysis_frame = ttk.LabelFrame(self.portfolio_frame, text="📊 Position Analysis")
        analysis_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Text widget for efficiency report
        analysis_container = ttk.Frame(analysis_frame)
        analysis_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.analysis_text = tk.Text(analysis_container, height=15, font=('Consolas', 10),
                                   bg='#f8f9fa', fg='#212529', wrap='word')
        analysis_scrollbar = ttk.Scrollbar(analysis_container, orient='vertical', command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=analysis_scrollbar.set)
        
        self.analysis_text.pack(side='left', fill='both', expand=True)
        analysis_scrollbar.pack(side='right', fill='y')
        
        # Portfolio action buttons
        portfolio_actions = ttk.Frame(self.portfolio_frame)
        portfolio_actions.pack(fill='x', padx=10, pady=10)
        
        refresh_btn = tk.Button(portfolio_actions, text="🔄 Refresh Analysis", 
                              command=self.refresh_portfolio_analysis, bg='lightblue', font=('Arial', 10, 'bold'))
        refresh_btn.pack(side='left', padx=5)
        
        optimize_btn = tk.Button(portfolio_actions, text="🎯 Optimize Portfolio", 
                               command=self.optimize_portfolio, bg='gold', font=('Arial', 10, 'bold'))
        optimize_btn.pack(side='left', padx=5)
        
        reset_daily_btn = tk.Button(portfolio_actions, text="📅 Reset Daily", 
                                  command=self.reset_daily_tracking, bg='orange', font=('Arial', 10, 'bold'))
        reset_daily_btn.pack(side='right', padx=5)
    
    def setup_performance_tab(self):
        # Performance Metrics Frame
        metrics_frame = ttk.LabelFrame(self.performance_frame, text="📊 Performance Metrics")
        metrics_frame.pack(fill='x', padx=10, pady=10)
        
        metrics_container = ttk.Frame(metrics_frame)
        metrics_container.pack(fill='x', padx=10, pady=10)
        
        # Performance indicators
        self.total_return_label = tk.Label(metrics_container, text="Total Return: 0%", font=('Arial', 12, 'bold'))
        self.total_return_label.pack(side='left')
        
        self.sharpe_ratio_label = tk.Label(metrics_container, text="Sharpe Ratio: 0.00", font=('Arial', 12))
        self.sharpe_ratio_label.pack(side='right')
        
        # Trading Statistics Frame
        trading_stats_frame = ttk.LabelFrame(self.performance_frame, text="📈 Trading Statistics")
        trading_stats_frame.pack(fill='x', padx=10, pady=10)
        
        trading_stats_container = ttk.Frame(trading_stats_frame)
        trading_stats_container.pack(fill='x', padx=10, pady=10)
        
        # Stats in grid
        self.win_rate_label = tk.Label(trading_stats_container, text="Win Rate: 0%", font=('Arial', 11))
        self.win_rate_label.grid(row=0, column=0, sticky='w', padx=10)
        
        self.profit_factor_label = tk.Label(trading_stats_container, text="Profit Factor: 0.00", font=('Arial', 11))
        self.profit_factor_label.grid(row=0, column=1, sticky='w', padx=10)
        
        self.total_trades_label = tk.Label(trading_stats_container, text="Total Trades: 0", font=('Arial', 11))
        self.total_trades_label.grid(row=1, column=0, sticky='w', padx=10)
        
        self.avg_trade_label = tk.Label(trading_stats_container, text="Avg Trade: $0.00", font=('Arial', 11))
        self.avg_trade_label.grid(row=1, column=1, sticky='w', padx=10)
        
        # Recovery Performance Frame
        recovery_perf_frame = ttk.LabelFrame(self.performance_frame, text="🛡️ Recovery Performance")
        recovery_perf_frame.pack(fill='x', padx=10, pady=10)
        
        recovery_container = ttk.Frame(recovery_perf_frame)
        recovery_container.pack(fill='x', padx=10, pady=10)
        
        self.recovery_attempts_label = tk.Label(recovery_container, text="Recovery Attempts: 0", font=('Arial', 11))
        self.recovery_attempts_label.pack(side='left')
        
        self.recovery_success_label = tk.Label(recovery_container, text="Success Rate: 0%", font=('Arial', 11))
        self.recovery_success_label.pack(side='right')
        
        # Performance Chart Placeholder
        chart_frame = ttk.LabelFrame(self.performance_frame, text="📈 Performance Chart")
        chart_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Placeholder for matplotlib chart
        chart_label = tk.Label(chart_frame, text="📊 Performance charts will be displayed here\n(Requires matplotlib integration)", 
                             font=('Arial', 12), fg='gray')
        chart_label.pack(expand=True)
        
        # Performance action buttons
        perf_actions = ttk.Frame(self.performance_frame)
        perf_actions.pack(fill='x', padx=10, pady=10)
        
        export_btn = tk.Button(perf_actions, text="📊 Export Report", 
                             command=self.export_performance_report, bg='lightgreen', font=('Arial', 10, 'bold'))
        export_btn.pack(side='left', padx=5)
        
        refresh_perf_btn = tk.Button(perf_actions, text="🔄 Refresh", 
                                   command=self.refresh_performance, bg='lightblue', font=('Arial', 10, 'bold'))
        refresh_perf_btn.pack(side='right', padx=5)
    
    def setup_enhanced_logs(self):
        """Enhanced log setup with better formatting and filtering"""
        
        # Main container frame
        main_container = ttk.Frame(self.logs_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Top control panel
        control_panel = ttk.LabelFrame(main_container, text="📝 Log Controls")
        control_panel.pack(fill='x', pady=(0, 10))
        
        # Filter controls
        filter_frame = ttk.Frame(control_panel)
        filter_frame.pack(fill='x', padx=10, pady=5)
        
        tk.Label(filter_frame, text="Filter:", font=('Arial', 9, 'bold')).pack(side='left', padx=(0, 5))
        
        self.log_filter = tk.StringVar(value="All")
        filter_combo = ttk.Combobox(filter_frame, textvariable=self.log_filter, 
                                values=["All", "AI Decisions", "Trading", "Profits", "Errors", "System"],
                                width=15, state="readonly")
        filter_combo.pack(side='left', padx=(0, 15))
        filter_combo.bind('<<ComboboxSelected>>', self.filter_logs)
        
        # Auto-scroll control
        self.auto_scroll = tk.BooleanVar(value=True)
        scroll_check = tk.Checkbutton(filter_frame, text="Auto Scroll", 
                                    variable=self.auto_scroll, font=('Arial', 9))
        scroll_check.pack(side='left', padx=(0, 15))
        
        # Log level control
        tk.Label(filter_frame, text="Level:", font=('Arial', 9, 'bold')).pack(side='left', padx=(0, 5))
        self.log_level = tk.StringVar(value="All")
        level_combo = ttk.Combobox(filter_frame, textvariable=self.log_level,
                                values=["All", "INFO", "SUCCESS", "WARNING", "ERROR"],
                                width=10, state="readonly")
        level_combo.pack(side='left', padx=(0, 15))
        level_combo.bind('<<ComboboxSelected>>', self.filter_logs)
        
        # Search box
        tk.Label(filter_frame, text="Search:", font=('Arial', 9, 'bold')).pack(side='left', padx=(0, 5))
        self.search_var = tk.StringVar()
        search_entry = tk.Entry(filter_frame, textvariable=self.search_var, width=15)
        search_entry.pack(side='left', padx=(0, 10))
        search_entry.bind('<KeyRelease>', self.search_logs)
        
        # Quick clear search
        clear_search_btn = tk.Button(filter_frame, text="✕", command=self.clear_search,
                                    width=2, height=1, font=('Arial', 8))
        clear_search_btn.pack(side='left', padx=(0, 10))
        
        # Log display frame
        log_display_frame = ttk.LabelFrame(main_container, text="📋 Trading Logs")
        log_display_frame.pack(fill='both', expand=True)
        
        # Create enhanced text widget
        text_frame = ttk.Frame(log_display_frame)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.log_text = tk.Text(text_frame, wrap='word', height=25,
                            bg='#0d1117', fg='#c9d1d9', 
                            font=('Consolas', 10),
                            insertbackground='#c9d1d9',
                            selectbackground='#264f78',
                            relief='flat',
                            borderwidth=1)
        
        # Configure enhanced color tags for better readability
        self.setup_log_tags()
        
        # Scrollbar with custom styling
        log_scrollbar = ttk.Scrollbar(text_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')
        
        # Bottom status and control panel
        bottom_panel = ttk.Frame(main_container)
        bottom_panel.pack(fill='x', pady=(10, 0))
        
        # Status info (left side)
        status_frame = ttk.Frame(bottom_panel)
        status_frame.pack(side='left', fill='x', expand=True)
        
        self.log_status = tk.Label(status_frame, text="Logs: 0 entries | Filtered: 0 | Last: --:--:--", 
                                font=('Arial', 9), fg='#7d8590', anchor='w')
        self.log_status.pack(side='left')
        
        # Control buttons (right side)
        button_frame = ttk.Frame(bottom_panel)
        button_frame.pack(side='right')
        
        # Style buttons with colors
        save_btn = tk.Button(button_frame, text="💾 Save", command=self.save_logs,
                            bg='#238636', fg='white', font=('Arial', 9, 'bold'),
                            relief='flat', padx=10, pady=2)
        save_btn.pack(side='left', padx=(0, 5))
        
        export_btn = tk.Button(button_frame, text="📤 Export", command=self.export_logs,
                            bg='#1f6feb', fg='white', font=('Arial', 9, 'bold'),
                            relief='flat', padx=10, pady=2)
        export_btn.pack(side='left', padx=(0, 5))
        
        clear_btn = tk.Button(button_frame, text="🗑️ Clear", command=self.clear_logs,
                            bg='#da3633', fg='white', font=('Arial', 9, 'bold'),
                            relief='flat', padx=10, pady=2)
        clear_btn.pack(side='left', padx=(0, 5))
        
        # Pause/Resume logging
        self.logging_paused = tk.BooleanVar(value=False)
        pause_btn = tk.Checkbutton(button_frame, text="⏸️ Pause", variable=self.logging_paused,
                                font=('Arial', 9, 'bold'), fg='#f85149')
        pause_btn.pack(side='left', padx=(0, 5))
        
        # Initialize log storage
        self.all_logs = []
        self.filtered_logs = []
        self.log_count = 0
        self.displayed_count = 0
    
    def setup_log_tags(self):
        """Setup color tags for different log types"""
        # AI related - Blue theme
        self.log_text.tag_configure("AI", foreground="#58a6ff", font=('Consolas', 10, 'bold'))
        self.log_text.tag_configure("AI_VALUE", foreground="#79c0ff", font=('Consolas', 10, 'bold'))
        
        # Trading - Green theme  
        self.log_text.tag_configure("TRADE", foreground="#3fb950", font=('Consolas', 10, 'bold'))
        self.log_text.tag_configure("SUCCESS", foreground="#56d364", font=('Consolas', 10, 'bold'))
        
        # Profit/Money - Yellow/Gold theme
        self.log_text.tag_configure("PROFIT", foreground="#ffd33d", font=('Consolas', 10, 'bold'))
        self.log_text.tag_configure("MONEY", foreground="#ffdf5d", font=('Consolas', 10, 'bold'))
        
        # Errors - Red theme
        self.log_text.tag_configure("ERROR", foreground="#f85149", font=('Consolas', 10, 'bold'))
        self.log_text.tag_configure("CRITICAL", foreground="#ff6b6b", font=('Consolas', 11, 'bold'))
        
        # Warnings - Orange theme
        self.log_text.tag_configure("WARNING", foreground="#f0883e", font=('Consolas', 10, 'bold'))
        
        # System - Gray theme
        self.log_text.tag_configure("SYSTEM", foreground="#8b949e", font=('Consolas', 10))
        self.log_text.tag_configure("INFO", foreground="#7d8590", font=('Consolas', 10))
        
        # Timestamp - Muted
        self.log_text.tag_configure("TIME", foreground="#6e7681", font=('Consolas', 9))
        
        # Special highlights
        self.log_text.tag_configure("HIGHLIGHT", background="#404040", foreground="#ffd33d")
        self.log_text.tag_configure("URGENT", background="#3d1a1a", foreground="#f85149")
    
    # ========================= CORE FUNCTIONALITY =========================
    
    def start_real_time_updates(self):
        """Start real-time GUI updates"""
        def update_loop():
            while True:
                try:
                    if self.is_connected:
                        self.root.after(0, self.update_gui)
                    time.sleep(2)  # Update every 2 seconds
                except Exception as e:
                    print(f"Update loop error: {e}")
                    time.sleep(5)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()
    
    def connect_mt5(self):
        """Connect to MetaTrader 5"""
        try:
            if self.mt5_interface.connect():
                self.is_connected = True
                self.conn_status.config(text="✅ Connected", fg="green")
                self.start_trading_btn.config(state='normal')
                
                # Initialize portfolio manager
                self.portfolio_manager.initialize_portfolio(self.mt5_interface)
                
                self.log_message("🔗 MT5 connected successfully", "SUCCESS")
                self.update_account_info()
                
            else:
                self.log_message("❌ Failed to connect to MT5", "ERROR")
                messagebox.showerror("Connection Error", "Failed to connect to MT5")
                
        except Exception as e:
            self.log_message(f"❌ MT5 connection error: {str(e)}", "ERROR")
            messagebox.showerror("Connection Error", f"Error: {str(e)}")
    
    def start_trading(self):
        """Start AI trading"""
        if not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to MT5 first")
            return
            
        try:
            # Apply current configuration
            self.apply_all_settings()
            
            # Set training mode
            is_training_mode = self.training_mode_var.get()
            self.config['training_mode'] = is_training_mode
            self.mt5_interface.set_training_mode(is_training_mode)

            # Initialize trading environment and RL agent
            self.trading_env = TradingEnvironment(self.mt5_interface, self.recovery_engine, self.config)
            self.trading_env.gui_instance = self
            self.rl_agent = RLAgent(self.trading_env, self.config)
            
            # Load trained model if available
            if self.rl_agent.load_model():
                self.log_message("✅ AI Model loaded successfully!", "SUCCESS")
                self.ai_status_label.config(text="🤖 AI Status: Model Loaded", fg='green')
            else:
                self.log_message("⚠️ No trained model found - using random actions", "WARNING")
                result = messagebox.askyesno("Warning", "No AI model found!\nContinue with random actions?")
                if not result:
                    return
                self.ai_status_label.config(text="🤖 AI Status: Random Actions", fg='orange')
                
            self.is_trading = True
            self.start_trading_btn.config(state='disabled')
            self.stop_trading_btn.config(state='normal')
            
            # Reset AI monitoring
            self.reset_ai_monitoring()
            
            # Start trading thread
            self.trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
            self.trading_thread.start()
            
            mode_text = "SIMULATION" if is_training_mode else "LIVE"
            self.log_message(f"🚀 {mode_text} AI Trading Started!", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"❌ Error starting trading: {str(e)}", "ERROR")
            messagebox.showerror("Trading Error", f"Error: {str(e)}")
    
    def stop_trading(self):
        """Stop AI trading"""
        self.is_trading = False
        self.start_trading_btn.config(state='normal')
        self.stop_trading_btn.config(state='disabled')
        self.log_message("⏹️ Trading stopped", "INFO")
    
    def trading_loop(self):
        """Main trading loop with state machine integration"""
        while self.is_trading:
            try:
                # Get observation from environment
                observation = self.trading_env._get_observation()
                
                # Get RL agent decision
                action = self.rl_agent.get_action(observation)
                
                # Force AI diversity to prevent repetitive actions
                action = self.force_ai_diversity(action)
                
                # Convert action to name
                action_name = self.get_action_name_from_value(action[0])
                
                # Execute action through environment (includes state machine)
                observation_new, reward, done, truncated, info = self.trading_env.step(action)
                
                # Update current trading state
                if info and 'trading_state' in info:
                    self.current_trading_state = TradingState(info['trading_state'])
                
                # Check for smart overrides
                override_used = abs(reward) > 2.0  # High reward indicates smart decision
                
                # Update AI monitoring
                self.root.after(0, lambda: self.update_ai_monitoring(action_name, float(reward), override_used))
                
                # Log AI decision
                self.log_message(f"🤖 AI Decision: {action_name} → Reward: {reward:.3f}", "AI")
                
                # Update GUI
                self.root.after(0, self.update_gui)
                
                # Check for episode reset
                if done:
                    self.trading_env.reset()
                    self.log_message("🔄 Episode completed, environment reset", "SYSTEM")
                
                # Sleep for trading interval
                time.sleep(1)
                
            except Exception as e:
                self.log_message(f"❌ Trading loop error: {str(e)}", "ERROR")
                time.sleep(5)
    
    def update_gui(self):
        """Update all GUI components"""
        try:
            # Update position table
            self.update_positions()
            
            # Update account information
            self.update_account_info()
            
            # Update trading state
            self.update_trading_state()
            
            # Update recovery status
            self.update_recovery_status()
            
            # Update portfolio status
            self.update_portfolio_status()
            
            # Update market information
            self.update_market_info()
            
        except Exception as e:
            self.log_message(f"❌ GUI update error: {str(e)}", "ERROR")
    
    def update_positions(self):
        """Update position table with enhanced information"""
        try:
            # Clear existing items
            for item in self.pos_tree.get_children():
                self.pos_tree.delete(item)
                
            # Get current positions from MT5
            positions = self.mt5_interface.get_positions() if self.is_connected else []
            
            for pos in positions:
                # Calculate target profit using portfolio manager
                volume = pos.get('volume', 0.01)
                target_profit = 0
                
                if hasattr(self, 'portfolio_manager'):
                    thresholds = self.portfolio_manager.get_current_thresholds()
                    target_profit = thresholds.get('per_lot', 5.0) * (volume / 0.01)
                
                # Determine row color based on profit vs target
                current_profit = pos.get('profit', 0)
                profit_color = 'green' if current_profit >= target_profit else 'red' if current_profit < 0 else 'black'
                
                # Insert row
                item_id = self.pos_tree.insert('', 'end', values=(
                    pos.get('symbol', ''),
                    'BUY' if pos.get('type', 0) == 0 else 'SELL',
                    f"{pos.get('volume', 0):.2f}",
                    f"{pos.get('price_open', 0):.2f}",
                    f"${current_profit:.2f}",
                    f"${target_profit:.2f}"
                ))
                
                # Color the row based on profit status
                if current_profit >= target_profit:
                    self.pos_tree.item(item_id, tags=('profitable',))
                elif current_profit < 0:
                    self.pos_tree.item(item_id, tags=('losing',))
                    
            # Configure row colors
            self.pos_tree.tag_configure('profitable', background='lightgreen')
            self.pos_tree.tag_configure('losing', background='lightcoral')
            
        except Exception as e:
            self.log_message(f"❌ Error updating positions: {str(e)}", "ERROR")
    
    def update_account_info(self):
        """Update account information display"""
        try:
            if not self.is_connected:
                return
                
            account_info = self.mt5_interface.get_account_info()
            if account_info:
                balance = account_info.get('balance', 0)
                equity = account_info.get('equity', 0)
                margin = account_info.get('margin', 0)
                free_margin = account_info.get('margin_free', 0)
                
                self.balance_label.config(text=f"Balance: ${balance:,.2f}")
                self.equity_label.config(text=f"Equity: ${equity:,.2f}")
                self.margin_label.config(text=f"Margin: ${margin:,.2f}")
                self.free_margin_label.config(text=f"Free: ${free_margin:,.2f}")
                
                # Update portfolio manager
                if hasattr(self, 'portfolio_manager'):
                    self.portfolio_manager.update_capital(equity)
                    
        except Exception as e:
            self.log_message(f"❌ Error updating account info: {str(e)}", "ERROR")
    
    def update_trading_state(self):
        """Update trading state display"""
        try:
            # Update state label
            state_name = self.current_trading_state.value
            self.current_state_label.config(text=f"State: {state_name}")
            
            # Update state description
            descriptions = {
                'ANALYZE': "🔍 Analyzing market for trading opportunities...",
                'ENTRY': "🎯 Executing position entry...",
                'MONITOR': "👀 Monitoring open positions...",
                'RECOVERY': "🛡️ Recovery mode - managing drawdown...",
                'EXIT': "💰 Closing positions and taking profits..."
            }
            
            description = descriptions.get(state_name, "🤖 AI processing...")
            self.state_description.config(text=description)
            
            # Update state timer
            time_in_state = int(time.time() - self.state_start_time)
            self.state_timer_label.config(text=f"Time in state: {time_in_state}s")
            
            # Color coding based on state
            state_colors = {
                'ANALYZE': 'blue',
                'ENTRY': 'orange', 
                'MONITOR': 'green',
                'RECOVERY': 'red',
                'EXIT': 'purple'
            }
            
            color = state_colors.get(state_name, 'black')
            self.current_state_label.config(fg=color)
            
        except Exception as e:
            self.log_message(f"❌ Error updating trading state: {str(e)}", "ERROR")
    
    def update_recovery_status(self):
        """Update recovery status display"""
        try:
            if hasattr(self, 'recovery_engine') and self.recovery_engine:
                recovery_info = self.recovery_engine.get_status()
                
                # Update recovery status
                recovery_active = recovery_info.get('recovery_active', False)
                recovery_level = recovery_info.get('recovery_level', 0)
                
                if recovery_active:
                    recovery_text = f"Recovery: 🔴 Active (L{recovery_level})"
                    self.recovery_status.config(text=recovery_text, fg='red')
                else:
                    recovery_text = "Recovery: 🟢 Inactive"
                    self.recovery_status.config(text=recovery_text, fg='green')
                    
                self.recovery_level.config(text=f"Level: {recovery_level}")
                
                # Update profit settings display
                profit_info = recovery_info.get('profit_settings', {})
                profit_target = profit_info.get('min_profit_target', 10)
                self.profit_target_label.config(text=f"Profit Target: ${profit_target}")
                
        except Exception as e:
            self.log_message(f"❌ Error updating recovery status: {str(e)}", "ERROR")
    
    def update_portfolio_status(self):
        """Update portfolio management display"""
        try:
            if hasattr(self, 'portfolio_manager'):
                # Update portfolio heat
                portfolio_heat = self.portfolio_manager.calculate_portfolio_heat(self.mt5_interface)
                heat_color = 'red' if portfolio_heat > 5.0 else 'orange' if portfolio_heat > 3.0 else 'green'
                self.portfolio_heat_label.config(text=f"Portfolio Heat: {portfolio_heat:.1f}%", fg=heat_color)
                
                # Update dynamic thresholds
                thresholds = self.portfolio_manager.get_current_thresholds()
                profit_per_lot = thresholds.get('per_lot', 5.0)
                
                # Update capital display in portfolio tab
                if hasattr(self, 'capital_label'):
                    capital = thresholds.get('capital', 0)
                    self.capital_label.config(text=f"Capital: ${capital:,.2f}")
                    
                if hasattr(self, 'profit_per_lot_label'):
                    self.profit_per_lot_label.config(text=f"Profit/Lot: ${profit_per_lot:.2f}")
                    
                if hasattr(self, 'portfolio_target_label'):
                    portfolio_target = thresholds.get('portfolio', 25.0)
                    self.portfolio_target_label.config(text=f"Portfolio Target: ${portfolio_target:.2f}")
                
                # Update trading allowed status
                if hasattr(self, 'trading_allowed_label'):
                    trading_allowed = self.portfolio_manager.trading_allowed
                    if trading_allowed:
                        self.trading_allowed_label.config(text="Trading: ✅ Allowed", fg='green')
                    else:
                        self.trading_allowed_label.config(text="Trading: 🚫 Blocked", fg='red')
                
                # Update risk mode
                if hasattr(self, 'risk_mode_label'):
                    if self.portfolio_manager.recovery_mode:
                        self.risk_mode_label.config(text="Risk Mode: 🔴 Recovery", fg='red')
                    elif self.portfolio_manager.risk_reduction_active:
                        self.risk_mode_label.config(text="Risk Mode: 🟡 Reduction", fg='orange')
                    else:
                        self.risk_mode_label.config(text="Risk Mode: 🟢 Normal", fg='green')
                
                # Update daily PnL
                if hasattr(self, 'daily_pnl_label'):
                    daily_pnl = self.portfolio_manager.daily_pnl
                    pnl_color = 'green' if daily_pnl > 0 else 'red' if daily_pnl < 0 else 'black'
                    self.daily_pnl_label.config(text=f"Daily PnL: {daily_pnl:+.2f}%", fg=pnl_color)
                
                # Update drawdown
                if hasattr(self, 'drawdown_label'):
                    drawdown = self.portfolio_manager.current_drawdown
                    dd_color = 'red' if drawdown > 5.0 else 'orange' if drawdown > 2.0 else 'green'
                    self.drawdown_label.config(text=f"Drawdown: {drawdown:.2f}%", fg=dd_color)
                    
        except Exception as e:
            self.log_message(f"❌ Error updating portfolio status: {str(e)}", "ERROR")
    
    def update_market_info(self):
        """Update market information display"""
        try:
            if not self.is_connected:
                return
                
            # Update spread information
            spread_info = self.mt5_interface.get_spread(self.symbol_var.get())
            if spread_info:
                spread_pips = spread_info['spread_pips']
                spread_color = 'red' if spread_pips > 2.0 else 'orange' if spread_pips > 1.0 else 'green'
                self.spread_label.config(text=f"Spread: {spread_pips:.1f} pips", fg=spread_color)
            
            # Update current price
            current_price = self.mt5_interface.get_current_price(self.symbol_var.get())
            if current_price:
                bid = current_price['bid']
                ask = current_price['ask']
                self.price_label.config(text=f"Price: {bid:.2f}/{ask:.2f}")
            
            # Update net PnL (positions profit minus spread costs)
            positions = self.mt5_interface.get_positions()
            if positions:
                total_pnl = sum(pos.get('profit', 0) for pos in positions)
                pnl_color = 'green' if total_pnl > 0 else 'red' if total_pnl < 0 else 'black'
                self.net_pnl_label.config(text=f"Net PnL: ${total_pnl:+.2f}", fg=pnl_color)
            else:
                self.net_pnl_label.config(text="Net PnL: $0.00", fg='black')
                
        except Exception as e:
            self.log_message(f"❌ Error updating market info: {str(e)}", "ERROR")
    
    # ========================= AI MONITORING =========================
    
    def reset_ai_monitoring(self):
        """Reset AI monitoring statistics"""
        try:
            self.ai_decision_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0, 'CLOSE': 0, 'HEDGE': 0}
            self.ai_override_count = 0
            self.ai_total_rewards = []
            
            # Update labels
            for action, label in self.ai_decision_labels.items():
                label.config(text=f"{action}: 0", bg='lightgray')
                
            self.ai_override_label.config(text="🧠 Smart Overrides: 0")
            self.ai_reward_label.config(text="📊 Avg Reward: 0.00")
            self.ai_status_label.config(text="🤖 AI Status: Starting...", fg='orange')
            
        except Exception as e:
            self.log_message(f"❌ Reset monitoring error: {e}", "ERROR")

    def update_ai_monitoring(self, action_taken, reward, override_used=False):
        """Update AI monitoring display"""
        try:
            # Update decision count
            if action_taken in self.ai_decision_counts:
                self.ai_decision_counts[action_taken] += 1
                if action_taken in self.ai_decision_labels:
                    self.ai_decision_labels[action_taken].config(text=f"{action_taken}: {self.ai_decision_counts[action_taken]}")
                    
                    # Color coding by action type
                    action_colors = {
                        'BUY': 'lightgreen',
                        'SELL': 'lightcoral', 
                        'CLOSE': 'gold',
                        'HEDGE': 'lightblue',
                        'HOLD': 'lightgray'
                    }
                    color = action_colors.get(action_taken, 'lightgray')
                    self.ai_decision_labels[action_taken].config(bg=color)
            
            # Update override count
            if override_used:
                self.ai_override_count += 1
                self.ai_override_label.config(text=f"🧠 Smart Overrides: {self.ai_override_count}")
            
            # Update reward tracking
            self.ai_total_rewards.append(reward)
            if len(self.ai_total_rewards) > 100:  # Keep last 100 rewards
                self.ai_total_rewards = self.ai_total_rewards[-100:]
                
            avg_reward = sum(self.ai_total_rewards) / len(self.ai_total_rewards) if self.ai_total_rewards else 0
            self.ai_reward_label.config(text=f"📊 Avg Reward: {avg_reward:.3f}")
            
            # Update AI status based on performance
            if len(self.ai_total_rewards) > 10:
                if avg_reward > 1.0:
                    self.ai_status_label.config(text="🤖 AI Status: 🌟 Excellent", fg='darkgreen')
                elif avg_reward > 0.5:
                    self.ai_status_label.config(text="🤖 AI Status: 📈 Learning Well", fg='green')
                elif avg_reward > 0.0:
                    self.ai_status_label.config(text="🤖 AI Status: 📊 Learning", fg='orange')
                elif avg_reward > -0.5:
                    self.ai_status_label.config(text="🤖 AI Status: ⚠️ Struggling", fg='red')
                else:
                    self.ai_status_label.config(text="🤖 AI Status: 🔴 Poor", fg='darkred')
            else:
                self.ai_status_label.config(text="🤖 AI Status: 📡 Collecting Data", fg='blue')
                    
        except Exception as e:
            self.log_message(f"❌ AI Monitoring update error: {e}", "ERROR")

    def force_ai_diversity(self, action):
        """Force AI to have diverse actions to prevent getting stuck"""
        try:
            # Add to recent actions
            self.recent_actions.append(action[0])
            if len(self.recent_actions) > 10:
                self.recent_actions = self.recent_actions[-10:]
                
            # Check if too repetitive
            if len(self.recent_actions) >= 5:
                action_variance = np.var(self.recent_actions)
                if action_variance < 0.01:  # Too similar
                    self.log_message(f"🎲 Adding AI diversity: variance={action_variance:.4f}", "WARNING")
                    action[0] += np.random.normal(0, 0.3)  # Add noise
                    action[0] = np.clip(action[0], 0, 4)
                    self.ai_override_count += 1
        
            return action
        except:
            return action

    def get_action_name_from_value(self, action_value):
        """Convert action value to action name"""
        try:
            if action_value < 0.3:
                return 'HOLD'
            elif 0.3 <= action_value < 1.7:
                return 'BUY'
            elif 1.7 <= action_value < 2.7:
                return 'SELL'
            elif 2.7 <= action_value < 3.5:
                return 'CLOSE'
            else:
                return 'HEDGE'
        except:
            return 'UNKNOWN'
    
    # ========================= TRAINING FUNCTIONALITY =========================
    
    def start_training(self):
        """Start RL training"""
        if not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to MT5 first")
            return
            
        try:
            # Apply current configuration
            self.apply_all_settings()
            
            # Initialize training environment
            self.trading_env = TradingEnvironment(self.mt5_interface, self.recovery_engine, self.config)
            self.trading_env.gui_instance = self
            self.rl_agent = RLAgent(self.trading_env, self.config)
            
            self.is_training = True
            self.start_training_btn.config(state='disabled')
            self.stop_training_btn.config(state='normal')
            
            # Start training thread with callback
            self.training_thread = threading.Thread(target=self.training_loop_with_callback, daemon=True)
            self.training_thread.start()
            
            self.log_message("🎓 AI Training started", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"❌ Error starting training: {str(e)}", "ERROR")
            messagebox.showerror("Training Error", f"Error: {str(e)}")

    def training_loop_with_callback(self):
        """Training loop with GUI updates"""
        try:
            # Create callback for GUI updates
            def gui_callback(step, total_steps, reward=None):
                progress = (step / total_steps) * 100
                
                # Update progress bar
                self.root.after(0, lambda: self.progress_bar.config(value=progress))
                self.root.after(0, lambda: self.progress_label.config(
                    text=f"🎓 Training: {step:,}/{total_steps:,} ({progress:.1f}%)"
                ))
                
                # Update statistics
                stats_text = f"Training Step: {step:,}/{total_steps:,}\n"
                stats_text += f"Progress: {progress:.1f}%\n"
                if reward:
                    stats_text += f"Current Reward: {reward:.4f}\n"
                stats_text += f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                stats_text += "=" * 50 + "\n"
                
                self.root.after(0, lambda: self.stats_text.insert(tk.END, stats_text))
                self.root.after(0, lambda: self.stats_text.see(tk.END))
            
            # Start training with callback
            success = self.rl_agent.train(
                total_timesteps=self.training_steps_var.get(),
                callback=gui_callback
            )
            
            if success:
                self.log_message("✅ Training completed successfully", "SUCCESS")
                self.root.after(0, lambda: self.progress_bar.config(value=100))
                self.root.after(0, lambda: self.progress_label.config(text="🎉 Training completed!"))
                
                # Auto-save model if enabled
                if hasattr(self, 'auto_save_var') and self.auto_save_var.get():
                    self.save_model_manual()
            else:
                self.log_message("❌ Training failed", "ERROR")
                self.root.after(0, lambda: self.progress_label.config(text="❌ Training failed"))
                
        except Exception as e:
            self.log_message(f"❌ Training error: {str(e)}", "ERROR")
            self.root.after(0, lambda: self.progress_label.config(text="❌ Training error"))
            
        finally:
            self.is_training = False
            self.root.after(0, lambda: self.start_training_btn.config(state='normal'))
            self.root.after(0, lambda: self.stop_training_btn.config(state='disabled'))

    def stop_training(self):
        """Stop training"""
        self.is_training = False
        self.start_training_btn.config(state='normal')
        self.stop_training_btn.config(state='disabled')
        self.log_message("⏹️ Training stopped", "WARNING")
        
    def quick_train(self):
        """Quick training for testing"""
        try:
            # Store original values
            original_steps = self.training_steps_var.get()
            
            # Set quick training parameters
            self.training_steps_var.set(1000)
            
            # Reset monitoring
            self.reset_ai_monitoring()
            
            # Start training
            self.start_training()
            
            # Schedule restore of original values
            def restore_values():
                self.training_steps_var.set(original_steps)
            
            self.root.after(3000, restore_values)  # Restore after 3 seconds
            
            self.log_message("⚡ Quick AI training started (1000 steps)", "INFO")
            
        except Exception as e:
            self.log_message(f"❌ Quick training error: {str(e)}", "ERROR")

    def save_model_manual(self):
        """Save current model manually"""
        try:
            if hasattr(self, 'rl_agent') and self.rl_agent and self.rl_agent.model:
                save_path = self.rl_agent.save_model("manual_save")
                
                if save_path:
                    self.log_message(f"💾 Model saved: {save_path}", "SUCCESS")
                    messagebox.showinfo("Success", f"Model saved!\nPath: {save_path}.zip")
                else:
                    self.log_message("❌ Failed to save model", "ERROR")
                    messagebox.showerror("Error", "Failed to save model")
            else:
                self.log_message("❌ No model to save", "WARNING")
                messagebox.showwarning("Warning", "No trained model found")
                
        except Exception as e:
            self.log_message(f"❌ Save error: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Save error: {str(e)}")
    
    # ========================= PORTFOLIO ACTIONS =========================
    
    def close_profitable_positions(self):
        """Close all profitable positions"""
        try:
            if not self.is_connected:
                messagebox.showwarning("Warning", "Not connected to MT5")
                return
                
            positions = self.mt5_interface.get_positions()
            profitable_positions = [pos for pos in positions if pos.get('profit', 0) > 0]
            
            if not profitable_positions:
                messagebox.showinfo("Info", "No profitable positions to close")
                return
                
            result = messagebox.askyesno("Confirm", f"Close {len(profitable_positions)} profitable positions?")
            if result:
                closed_count = 0
                total_profit = 0
                
                for pos in profitable_positions:
                    if self.mt5_interface.close_position(pos.get('ticket')):
                        closed_count += 1
                        total_profit += pos.get('profit', 0)
                        
                self.log_message(f"💰 Closed {closed_count} positions, profit: ${total_profit:.2f}", "SUCCESS")
                messagebox.showinfo("Success", f"Closed {closed_count} positions\nTotal profit: ${total_profit:.2f}")
                
        except Exception as e:
            self.log_message(f"❌ Error closing positions: {str(e)}", "ERROR")

    def emergency_close_all(self):
        """Emergency close all positions"""
        try:
            if not self.is_connected:
                messagebox.showwarning("Warning", "Not connected to MT5")
                return
                
            result = messagebox.askyesno("🚨 EMERGENCY", 
                                       "Close ALL positions immediately?\nThis cannot be undone!")
            if result:
                success = self.mt5_interface.close_all_positions()
                if success:
                    self.log_message("🚨 EMERGENCY: All positions closed", "CRITICAL")
                    messagebox.showinfo("Emergency", "All positions closed successfully")
                else:
                    self.log_message("❌ Emergency close failed", "ERROR")
                    messagebox.showerror("Error", "Failed to close all positions")
                    
        except Exception as e:
            self.log_message(f"❌ Emergency close error: {str(e)}", "ERROR")
    
    def refresh_portfolio_analysis(self):
        """Refresh portfolio analysis display"""
        try:
            if not hasattr(self, 'portfolio_manager'):
                return
                
            positions = self.mt5_interface.get_positions() if self.is_connected else []
            report = self.portfolio_manager.get_position_efficiency_report(positions)
            
            # Update analysis text
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(1.0, report)
            
            self.log_message("🔄 Portfolio analysis refreshed", "INFO")
            
        except Exception as e:
            self.log_message(f"❌ Error refreshing analysis: {str(e)}", "ERROR")

    def optimize_portfolio(self):
        """Optimize portfolio based on current positions"""
        try:
            if not hasattr(self, 'portfolio_manager'):
                messagebox.showwarning("Warning", "Portfolio manager not available")
                return
                
            positions = self.mt5_interface.get_positions() if self.is_connected else []
            optimization = self.portfolio_manager.optimize_portfolio_allocation(positions)
            
            # Show optimization results
            message = f"Optimization Result: {optimization['action']}\n"
            message += f"Reason: {optimization['reason']}\n\n"
            
            if optimization['close_positions']:
                message += "Recommended closures:\n"
                for pos in optimization['close_positions']:
                    message += f"- Position {pos['ticket']}: {pos['reason']}\n"
                    
            messagebox.showinfo("Portfolio Optimization", message)
            self.log_message(f"🎯 Portfolio optimization: {optimization['action']}", "INFO")
            
        except Exception as e:
            self.log_message(f"❌ Optimization error: {str(e)}", "ERROR")

    def reset_daily_tracking(self):
        """Reset daily tracking for portfolio manager"""
        try:
            if hasattr(self, 'portfolio_manager'):
                self.portfolio_manager.reset_daily_tracking()
                self.log_message("📅 Daily tracking reset", "SUCCESS")
                messagebox.showinfo("Success", "Daily tracking has been reset")
            else:
                messagebox.showwarning("Warning", "Portfolio manager not available")
                
        except Exception as e:
            self.log_message(f"❌ Reset daily tracking error: {str(e)}", "ERROR")
    
    # ========================= CONFIGURATION =========================
    
    def apply_all_settings(self):
        """Apply all configuration settings"""
        try:
            # Update config with current GUI values
            self.config.update({
                'symbol': self.symbol_var.get(),
                'initial_lot_size': self.lot_size_var.get(),
                'max_positions': self.max_positions_var.get(),
                'algorithm': self.algorithm_var.get(),
                'learning_rate': self.learning_rate_var.get(),
                'training_steps': self.training_steps_var.get(),
                'training_mode': self.training_mode_var.get(),
                'base_risk_per_trade': self.risk_per_trade_var.get(),
                'max_portfolio_risk': self.max_portfolio_risk_var.get(),
                'profit_per_lot_target': self.profit_per_lot_var.get(),
                'recovery_type': self.recovery_type_var.get(),
                'martingale_multiplier': self.martingale_mult_var.get(),
                'min_profit_target': self.min_profit_var.get(),
                'quick_profit_mode': self.quick_profit_var.get()
            })
            
            # Apply to recovery engine
            if hasattr(self, 'recovery_engine'):
                self.recovery_engine.update_profit_settings({
                    'min_profit_target': self.min_profit_var.get(),
                    'quick_profit_mode': self.quick_profit_var.get()
                })
            
            # Apply to portfolio manager
            if hasattr(self, 'portfolio_manager'):
                self.portfolio_manager.config.update(self.config)
                
            self.log_message("✅ All settings applied", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"❌ Error applying settings: {str(e)}", "ERROR")

    def save_config(self):
        """Save current configuration"""
        try:
            self.apply_all_settings()
            
            os.makedirs('config', exist_ok=True)
            with open('config/user_config.json', 'w') as f:
                json.dump(self.config, f, indent=4)
                
            self.log_message("💾 Configuration saved", "SUCCESS")
            messagebox.showinfo("Success", "Configuration saved successfully")
            
        except Exception as e:
            self.log_message(f"❌ Save config error: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Save error: {str(e)}")
        
    def load_config_dialog(self):
        """Load configuration from file"""
        try:
            filename = filedialog.askopenfilename(
                title="Load Configuration",
                filetypes=[("JSON files", "*.json")]
            )
            
            if filename:
                with open(filename, 'r') as f:
                    config = json.load(f)
                    
                # Update GUI with loaded config
                self.symbol_var.set(config.get('symbol', 'XAUUSD'))
                self.lot_size_var.set(config.get('initial_lot_size', 0.01))
                self.max_positions_var.set(config.get('max_positions', 10))
                self.algorithm_var.set(config.get('algorithm', 'PPO'))
                self.learning_rate_var.set(config.get('learning_rate', 0.001))
                self.training_steps_var.set(config.get('training_steps', 50000))
                self.training_mode_var.set(config.get('training_mode', True))
                self.risk_per_trade_var.set(config.get('base_risk_per_trade', 1.0))
                self.max_portfolio_risk_var.set(config.get('max_portfolio_risk', 5.0))
                self.profit_per_lot_var.set(config.get('profit_per_lot_target', 5.0))
                self.recovery_type_var.set(config.get('recovery_type', 'combined'))
                self.martingale_mult_var.set(config.get('martingale_multiplier', 2.0))
                self.min_profit_var.set(config.get('min_profit_target', 10))
                self.quick_profit_var.set(config.get('quick_profit_mode', True))
                
                self.config = config
                self.log_message(f"📂 Configuration loaded from {filename}", "SUCCESS")
                messagebox.showinfo("Success", "Configuration loaded successfully")
                
        except Exception as e:
            self.log_message(f"❌ Load config error: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Load error: {str(e)}")
                
    def reset_config(self):
        """Reset configuration to defaults"""
        try:
            # Reset to default values
            self.symbol_var.set('XAUUSD')
            self.lot_size_var.set(0.01)
            self.max_positions_var.set(10)
            self.algorithm_var.set('PPO')
            self.learning_rate_var.set(0.001)
            self.training_steps_var.set(50000)
            self.training_mode_var.set(True)
            self.risk_per_trade_var.set(1.0)
            self.max_portfolio_risk_var.set(5.0)
            self.profit_per_lot_var.set(5.0)
            self.recovery_type_var.set('combined')
            self.martingale_mult_var.set(2.0)
            self.min_profit_var.set(10)
            self.quick_profit_var.set(True)
            
            self.config = self.load_config()  # Load default config
            self.log_message("🔄 Configuration reset to defaults", "INFO")
            messagebox.showinfo("Reset", "Configuration reset to defaults")
            
        except Exception as e:
            self.log_message(f"❌ Reset config error: {str(e)}", "ERROR")
    
    def load_config(self):
        """Load configuration with enhanced defaults"""
        try:
            with open('config/user_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'symbol': 'XAUUSD',
                'initial_lot_size': 0.01,
                'max_positions': 10,
                'algorithm': 'PPO',
                'learning_rate': 0.001,
                'training_steps': 50000,
                'training_mode': True,
                'base_risk_per_trade': 1.0,
                'max_portfolio_risk': 5.0,
                'profit_per_lot_target': 5.0,
                'recovery_type': 'combined',
                'martingale_multiplier': 2.0,
                'min_profit_target': 10,
                'quick_profit_mode': True,
                'max_steps': 500
            }
    
    # ========================= PERFORMANCE & REPORTING =========================
    
    def refresh_performance(self):
        """Refresh performance metrics"""
        try:
            if hasattr(self, 'portfolio_manager'):
                performance = self.portfolio_manager.get_portfolio_performance()
                
                # Update performance labels
                if hasattr(self, 'total_return_label'):
                    total_return = performance.get('total_return', 0)
                    return_color = 'green' if total_return > 0 else 'red' if total_return < 0 else 'black'
                    self.total_return_label.config(text=f"Total Return: {total_return:+.2f}%", fg=return_color)
                
                if hasattr(self, 'sharpe_ratio_label'):
                    sharpe = performance.get('sharpe_ratio', 0)
                    self.sharpe_ratio_label.config(text=f"Sharpe Ratio: {sharpe:.2f}")
            
            # Update trading statistics from data handler
            if hasattr(self, 'data_handler'):
                stats = self.data_handler.calculate_trading_statistics()
                
                if hasattr(self, 'win_rate_label'):
                    win_rate = stats.get('win_rate', 0) * 100
                    self.win_rate_label.config(text=f"Win Rate: {win_rate:.1f}%")
                
                if hasattr(self, 'profit_factor_label'):
                    profit_factor = stats.get('profit_factor', 0)
                    self.profit_factor_label.config(text=f"Profit Factor: {profit_factor:.2f}")
                
                if hasattr(self, 'total_trades_label'):
                    total_trades = stats.get('total_trades', 0)
                    self.total_trades_label.config(text=f"Total Trades: {total_trades}")
                
                if hasattr(self, 'avg_trade_label'):
                    avg_pnl = stats.get('average_pnl', 0)
                    self.avg_trade_label.config(text=f"Avg Trade: ${avg_pnl:.2f}")
            
            # Update recovery statistics
            if hasattr(self, 'recovery_engine'):
                recovery_stats = self.recovery_engine.get_recovery_statistics()
                
                if hasattr(self, 'recovery_attempts_label'):
                    attempts = recovery_stats.get('total_attempts', 0)
                    self.recovery_attempts_label.config(text=f"Recovery Attempts: {attempts}")
                
                if hasattr(self, 'recovery_success_label'):
                    success_rate = recovery_stats.get('success_rate', 0) * 100
                    self.recovery_success_label.config(text=f"Success Rate: {success_rate:.1f}%")
            
            self.log_message("🔄 Performance metrics refreshed", "INFO")
            
        except Exception as e:
            self.log_message(f"❌ Error refreshing performance: {str(e)}", "ERROR")

    def export_performance_report(self):
        """Export comprehensive performance report"""
        try:
            from tkinter import filedialog
            import json
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("Text files", "*.txt")
                ],
                title="Export Performance Report"
            )
            
            if not filename:
                return
                
            # Collect all performance data
            report_data = {
                'export_time': datetime.now().isoformat(),
                'account_info': self.mt5_interface.get_account_info() if self.is_connected else {},
                'portfolio_performance': {},
                'trading_statistics': {},
                'recovery_statistics': {},
                'ai_performance': {
                    'decision_counts': self.ai_decision_counts,
                    'override_count': self.ai_override_count,
                    'average_reward': sum(self.ai_total_rewards) / len(self.ai_total_rewards) if self.ai_total_rewards else 0
                }
            }
            
            # Portfolio performance
            if hasattr(self, 'portfolio_manager'):
                report_data['portfolio_performance'] = self.portfolio_manager.get_portfolio_performance()
            
            # Trading statistics
            if hasattr(self, 'data_handler'):
                report_data['trading_statistics'] = self.data_handler.calculate_trading_statistics()
            
            # Recovery statistics
            if hasattr(self, 'recovery_engine'):
                report_data['recovery_statistics'] = self.recovery_engine.get_recovery_statistics()
            
            # Save report
            if filename.endswith('.json'):
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            else:  # .txt
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("AI Trading System Performance Report\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Generated: {report_data['export_time']}\n\n")
                    
                    # Write each section
                    for section, data in report_data.items():
                        if section != 'export_time':
                            f.write(f"{section.upper()}:\n")
                            f.write(str(data) + "\n\n")
            
            self.log_message(f"📊 Performance report exported: {filename}", "SUCCESS")
            messagebox.showinfo("Success", f"Report exported to:\n{filename}")
            
        except Exception as e:
            self.log_message(f"❌ Export error: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Export error: {str(e)}")
    
    # ========================= ENHANCED LOGGING =========================
    
    def log_message(self, message, level="INFO"):
        """Enhanced log message with formatting and colors"""
        try:
            # Skip if logging is paused
            if hasattr(self, 'logging_paused') and self.logging_paused.get():
                return
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Store in all_logs
            log_entry = {
                'timestamp': timestamp,
                'level': level,
                'message': message
            }
            
            if not hasattr(self, 'all_logs'):
                self.all_logs = []
                self.log_count = 0
                
            self.all_logs.append(log_entry)
            self.log_count += 1
            
            # Apply current filter and display if matches
            if (not hasattr(self, 'log_filter') or 
                self.should_display_log(log_entry)):
                self.add_formatted_log_to_display(log_entry)
                
                # Auto scroll if enabled
                if hasattr(self, 'auto_scroll') and self.auto_scroll.get():
                    self.log_text.see(tk.END)
            
            # Update status
            if hasattr(self, 'update_log_status'):
                self.update_log_status()
                
        except Exception as e:
            print(f"Logging error: {e}")

    def should_display_log(self, log_entry):
        """Check if log should be displayed based on current filters"""
        if not hasattr(self, 'log_filter'):
            return True
            
        filter_type = self.log_filter.get()
        level_filter = self.log_level.get() if hasattr(self, 'log_level') else "All"
        search_term = self.search_var.get().lower() if hasattr(self, 'search_var') else ""
        
        # Apply filters
        if filter_type != "All" and not self.matches_type_filter(log_entry, filter_type):
            return False
            
        if level_filter != "All" and log_entry.get('level', 'INFO') != level_filter:
            return False
            
        if search_term and search_term not in log_entry['message'].lower():
            return False
            
        return True

    def matches_type_filter(self, log_entry, filter_type):
        """Check if log entry matches the type filter"""
        message = log_entry['message'].upper()
        
        if filter_type == "AI Decisions":
            return "AI" in message or "🤖" in message
        elif filter_type == "Trading":
            return any(keyword in message for keyword in ["BUY", "SELL", "CLOSE", "ORDER", "POSITION", "TRADING"])
        elif filter_type == "Profits":
            return any(keyword in message for keyword in ["PROFIT", "PNL", "$", "💰"])
        elif filter_type == "Errors":
            return log_entry.get('level') == 'ERROR' or "ERROR" in message or "❌" in message
        elif filter_type == "System":
            return not any(keyword in message for keyword in ["AI", "BUY", "SELL", "PROFIT", "ERROR"])
        
        return True

    def add_formatted_log_to_display(self, log_entry):
        """Add formatted log entry to display"""
        try:
            timestamp = log_entry['timestamp']
            message = log_entry['message']
            level = log_entry.get('level', 'INFO')
            
            # Insert timestamp
            self.log_text.insert(tk.END, f"[{timestamp}] ", "TIME")
            
            # Format and insert message with appropriate tag
            if "🤖" in message or "AI" in message.upper():
                self.log_text.insert(tk.END, message, "AI")
            elif any(keyword in message.upper() for keyword in ["BUY", "SELL", "CLOSE", "ORDER", "TRADING"]):
                self.log_text.insert(tk.END, message, "TRADE")
            elif any(keyword in message.upper() for keyword in ["PROFIT", "PNL", "$", "💰"]):
                self.log_text.insert(tk.END, message, "PROFIT")
            elif level == "ERROR" or "❌" in message:
                self.log_text.insert(tk.END, message, "ERROR")
            elif level == "WARNING" or "⚠️" in message:
                self.log_text.insert(tk.END, message, "WARNING")
            elif level == "SUCCESS" or "✅" in message:
                self.log_text.insert(tk.END, message, "SUCCESS")
            else:
                self.log_text.insert(tk.END, message, "SYSTEM")
            
            self.log_text.insert(tk.END, "\n")
            
        except Exception as e:
            print(f"Log display error: {e}")

    def filter_logs(self, event=None):
        """Apply current filter settings to logs"""
        try:
            if not hasattr(self, 'all_logs'):
                return
                
            # Clear display
            self.log_text.delete(1.0, tk.END)
            
            # Filter and display logs
            self.filtered_logs = []
            for log_entry in self.all_logs:
                if self.should_display_log(log_entry):
                    self.filtered_logs.append(log_entry)
                    self.add_formatted_log_to_display(log_entry)
            
            # Update status
            self.displayed_count = len(self.filtered_logs)
            self.update_log_status()
            
            # Auto scroll to bottom
            if hasattr(self, 'auto_scroll') and self.auto_scroll.get():
                self.log_text.see(tk.END)
                
        except Exception as e:
            print(f"Filter logs error: {e}")

    def search_logs(self, event=None):
        """Perform real-time search filtering"""
        self.filter_logs()

    def clear_search(self):
        """Clear search term and refresh"""
        self.search_var.set("")
        self.filter_logs()

    def update_log_status(self):
        """Update the status bar with current log statistics"""
        try:
            if hasattr(self, 'filtered_logs'):
                last_time = self.all_logs[-1]['timestamp'] if self.all_logs else "--:--:--"
                status_text = f"Logs: {self.log_count} entries | Filtered: {self.displayed_count} | Last: {last_time}"
                
                # Add filter info if active
                if (self.log_filter.get() != "All" or 
                    self.log_level.get() != "All" or 
                    self.search_var.get()):
                    status_text += " | 🔍 FILTERED"
                    
                self.log_status.config(text=status_text)
        except:
            pass

    def save_logs(self):
        """Save current filtered logs to file"""
        try:
            import os
            from tkinter import messagebox
            
            os.makedirs('logs', exist_ok=True)
            filename = f"logs/trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            logs_to_save = getattr(self, 'filtered_logs', []) or getattr(self, 'all_logs', [])
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"AI Trading System Logs - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Entries: {len(logs_to_save)}\n")
                f.write("=" * 80 + "\n\n")
                
                for log_entry in logs_to_save:
                    f.write(f"[{log_entry['timestamp']}] [{log_entry.get('level', 'INFO')}] {log_entry['message']}\n")
            
            messagebox.showinfo("Success", f"Logs saved to:\n{filename}")
            self.log_message(f"💾 Logs saved: {filename}", "SUCCESS")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save logs:\n{str(e)}")

    def export_logs(self):
        """Export logs in various formats"""
        try:
            from tkinter import filedialog, messagebox
            import json
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[
                    ("JSON files", "*.json"),
                    ("Text files", "*.txt"), 
                    ("CSV files", "*.csv")
                ],
                title="Export Trading Logs"
            )
            
            if not filename:
                return
                
            logs_to_export = getattr(self, 'filtered_logs', []) or getattr(self, 'all_logs', [])
            
            if filename.endswith('.json'):
                with open(filename, 'w', encoding='utf-8') as f:
                    export_data = {
                        'export_time': datetime.now().isoformat(),
                        'total_entries': len(logs_to_export),
                        'logs': logs_to_export
                    }
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                    
            elif filename.endswith('.csv'):
                import csv
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Timestamp', 'Level', 'Message'])
                    for log in logs_to_export:
                        writer.writerow([log['timestamp'], log.get('level', 'INFO'), log['message']])
                        
            else:  # .txt
                with open(filename, 'w', encoding='utf-8') as f:
                    for log in logs_to_export:
                        f.write(f"[{log['timestamp']}] [{log.get('level', 'INFO')}] {log['message']}\n")
            
            messagebox.showinfo("Success", f"Logs exported to:\n{filename}")
            self.log_message(f"📤 Logs exported: {filename}", "SUCCESS")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export logs:\n{str(e)}")

    def clear_logs(self):
        """Enhanced clear logs with confirmation"""
        try:
            from tkinter import messagebox
            
            log_count = getattr(self, 'log_count', 0)
            if log_count > 0:
                result = messagebox.askyesno("Clear Logs", 
                                        f"Clear all {log_count} log entries?\nThis cannot be undone.")
                if not result:
                    return
            
            self.log_text.delete(1.0, tk.END)
            self.all_logs = []
            self.filtered_logs = []
            self.log_count = 0
            self.displayed_count = 0
            
            if hasattr(self, 'update_log_status'):
                self.update_log_status()
                
        except Exception as e:
            print(f"Clear logs error: {e}")
    
    # ========================= MAIN APPLICATION =========================
    
    def run(self):
        """Run the main application"""
        try:
            # Set window icon if available
            try:
                self.root.iconbitmap('icon.ico')  # Optional icon file
            except:
                pass
            
            # Center window on screen
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'{width}x{height}+{x}+{y}')
            
            # Initial log message
            self.log_message("🚀 AI Trading System Started", "SUCCESS")
            self.log_message("💡 Connect to MT5 to begin trading", "INFO")
            
            # Start main loop
            self.root.mainloop()
            
        except Exception as e:
            print(f"Application error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources before exit"""
        try:
            # Stop trading
            self.is_trading = False
            self.is_training = False
            
            # Disconnect MT5
            if hasattr(self, 'mt5_interface'):
                self.mt5_interface.disconnect()
            
            # Save current configuration
            try:
                self.save_config()
            except:
                pass
            
            print("🧹 Cleanup completed")
            
        except Exception as e:
            print(f"Cleanup error: {e}")

if __name__ == "__main__":
    try:
        # Create necessary directories
        os.makedirs('config', exist_ok=True)
        os.makedirs('models/trained_models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('utils', exist_ok=True)
        
        # Run the application
        app = TradingGUI()
        app.run()
        
    except Exception as e:
        print(f"Startup error: {e}")
        import traceback
        traceback.print_exc()
# main.py - Main GUI Application
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

from environment import TradingEnvironment
from recovery_engine import RecoveryEngine
from mt5_interface import MT5Interface
from rl_agent import RLAgent
from utils.data_handler import DataHandler
from utils.visualizer import Visualizer

class TradingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RL Trading System - XAUUSD")
        self.root.geometry("1200x800")
        
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
        
        # Initialize GUI components
        self.setup_gui()
        
        # Initialize RL components
        self.trading_env = None
        self.rl_agent = None
        
        # Threading for real-time updates
        self.update_thread = None
        self.training_thread = None
        self.recent_actions = []  # ← เพิ่มบรรทัดนี้
    
    def force_ai_diversity(self, action):
        """Force AI to have diverse actions"""
        
        # Add randomness if AI is too repetitive
        if hasattr(self, 'recent_actions'):
            self.recent_actions.append(action[0])
            if len(self.recent_actions) > 10:
                self.recent_actions = self.recent_actions[-10:]
                
            # Check if too repetitive
            action_variance = np.var(self.recent_actions)
            if action_variance < 0.01:  # Too similar
                print(f"🎲 Adding diversity: variance={action_variance:.4f}")
                action[0] += np.random.normal(0, 0.2)  # Add noise
                action[0] = np.clip(action[0], 0, 4)
        else:
            self.recent_actions = [action[0]]
        
        return action

    def setup_gui(self):
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        
        # Main Dashboard Tab
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text="Dashboard")
        
        # Configuration Tab
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Configuration")
        
        # Training Tab
        self.training_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.training_frame, text="Training")
        
        # Performance Tab
        self.performance_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.performance_frame, text="Performance")
        
        # Logs Tab
        self.logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.logs_frame, text="Logs")
        
        self.notebook.pack(fill='both', expand=True)
        
        # Setup individual tabs
        self.setup_dashboard()
        self.setup_configuration()
        self.setup_training()
        self.setup_performance()
        self.setup_logs()
        
    def setup_dashboard(self):
        # Connection Status Frame
        conn_frame = ttk.LabelFrame(self.main_frame, text="Connection Status")
        conn_frame.pack(fill='x', padx=10, pady=5)
        
        self.conn_status = tk.Label(conn_frame, text="Disconnected", fg="red")
        self.conn_status.pack(side='left', padx=10)
        
        self.connect_btn = tk.Button(conn_frame, text="Connect MT5", 
                                   command=self.connect_mt5)
        self.connect_btn.pack(side='right', padx=10)
        
        # Trading Control Frame
        control_frame = ttk.LabelFrame(self.main_frame, text="Trading Control")
        control_frame.pack(fill='x', padx=10, pady=5)
        
        self.start_trading_btn = tk.Button(control_frame, text="Start Trading", 
                                         command=self.start_trading, 
                                         state='disabled')
        self.start_trading_btn.pack(side='left', padx=5)
        
        self.stop_trading_btn = tk.Button(control_frame, text="Stop Trading", 
                                        command=self.stop_trading, 
                                        state='disabled')
        self.stop_trading_btn.pack(side='left', padx=5)
        
        # Position Information Frame
        pos_frame = ttk.LabelFrame(self.main_frame, text="Position Information")
        pos_frame.pack(fill='x', padx=10, pady=5)
        
        # Position table
        self.pos_tree = ttk.Treeview(pos_frame, columns=('Symbol', 'Type', 'Lots', 'Price', 'PnL'), show='headings')
        self.pos_tree.heading('Symbol', text='Symbol')
        self.pos_tree.heading('Type', text='Type')
        self.pos_tree.heading('Lots', text='Lots')
        self.pos_tree.heading('Price', text='Price')
        self.pos_tree.heading('PnL', text='PnL')
        self.pos_tree.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Recovery Status Frame
        recovery_frame = ttk.LabelFrame(self.main_frame, text="Recovery Status")
        recovery_frame.pack(fill='x', padx=10, pady=5)
        
        self.recovery_status = tk.Label(recovery_frame, text="No Recovery Active")
        self.recovery_status.pack(side='left', padx=10)
        
        self.recovery_level = tk.Label(recovery_frame, text="Level: 0")
        self.recovery_level.pack(side='right', padx=10)
        
        # Account Information Frame
        account_frame = ttk.LabelFrame(self.main_frame, text="Account Information")
        account_frame.pack(fill='x', padx=10, pady=5)
        
        self.balance_label = tk.Label(account_frame, text="Balance: $0.00")
        self.balance_label.pack(side='left', padx=10)
        
        self.equity_label = tk.Label(account_frame, text="Equity: $0.00")
        self.equity_label.pack(side='left', padx=10)
        
        self.margin_label = tk.Label(account_frame, text="Margin: $0.00")
        self.margin_label.pack(side='left', padx=10)
        
    def setup_configuration(self):
        # Trading Parameters Frame
        trading_params = ttk.LabelFrame(self.config_frame, text="Trading Parameters")
        trading_params.pack(fill='x', padx=10, pady=5)
        
        # Symbol
        tk.Label(trading_params, text="Symbol:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.symbol_var = tk.StringVar(value=self.config.get('symbol', 'XAUUSD'))
        tk.Entry(trading_params, textvariable=self.symbol_var).grid(row=0, column=1, padx=5, pady=2)
        
        # Initial Lot Size
        tk.Label(trading_params, text="Initial Lot Size:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.lot_size_var = tk.DoubleVar(value=self.config.get('initial_lot_size', 0.01))
        tk.Entry(trading_params, textvariable=self.lot_size_var).grid(row=1, column=1, padx=5, pady=2)
        
        # Max Positions
        tk.Label(trading_params, text="Max Positions:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.max_positions_var = tk.IntVar(value=self.config.get('max_positions', 10))
        tk.Entry(trading_params, textvariable=self.max_positions_var).grid(row=2, column=1, padx=5, pady=2)
        
        # Recovery Parameters Frame
        recovery_params = ttk.LabelFrame(self.config_frame, text="Recovery Parameters")
        recovery_params.pack(fill='x', padx=10, pady=5)
        
        # Martingale Multiplier
        tk.Label(recovery_params, text="Martingale Multiplier:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.martingale_mult_var = tk.DoubleVar(value=self.config.get('martingale_multiplier', 2.0))
        tk.Entry(recovery_params, textvariable=self.martingale_mult_var).grid(row=0, column=1, padx=5, pady=2)
        
        # Grid Spacing
        tk.Label(recovery_params, text="Grid Spacing (pips):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.grid_spacing_var = tk.IntVar(value=self.config.get('grid_spacing', 100))
        tk.Entry(recovery_params, textvariable=self.grid_spacing_var).grid(row=1, column=1, padx=5, pady=2)
        
        # Recovery Type
        tk.Label(recovery_params, text="Recovery Type:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.recovery_type_var = tk.StringVar(value=self.config.get('recovery_type', 'Martingale'))
        recovery_combo = ttk.Combobox(recovery_params, textvariable=self.recovery_type_var, 
                                    values=['Martingale', 'Grid', 'Hedge', 'Combined'])
        recovery_combo.grid(row=2, column=1, padx=5, pady=2)
        
        # RL Parameters Frame
        rl_params = ttk.LabelFrame(self.config_frame, text="RL Parameters")
        rl_params.pack(fill='x', padx=10, pady=5)
        
        # Algorithm
        tk.Label(rl_params, text="Algorithm:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.algorithm_var = tk.StringVar(value=self.config.get('algorithm', 'PPO'))
        algo_combo = ttk.Combobox(rl_params, textvariable=self.algorithm_var, 
                                values=['PPO', 'DQN', 'A2C'])
        algo_combo.grid(row=0, column=1, padx=5, pady=2)
        
        # Learning Rate
        tk.Label(rl_params, text="Learning Rate:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.learning_rate_var = tk.DoubleVar(value=self.config.get('learning_rate', 0.0003))
        tk.Entry(rl_params, textvariable=self.learning_rate_var).grid(row=1, column=1, padx=5, pady=2)
        
        # Training Steps
        tk.Label(rl_params, text="Training Steps:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.training_steps_var = tk.IntVar(value=self.config.get('training_steps', 10000))
        tk.Entry(rl_params, textvariable=self.training_steps_var).grid(row=2, column=1, padx=5, pady=2)
        
        # Profit Taking Parameters Frame - เพิ่มใหม่
        profit_params = ttk.LabelFrame(self.config_frame, text="Profit Taking Settings")
        profit_params.pack(fill='x', padx=10, pady=5)
        
        # Min Profit Target
        tk.Label(profit_params, text="Min Profit Target ($):").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.min_profit_var = tk.DoubleVar(value=self.config.get('min_profit_target', 25))
        tk.Entry(profit_params, textvariable=self.min_profit_var).grid(row=0, column=1, padx=5, pady=2)
        
        # Trailing Stop Distance
        tk.Label(profit_params, text="Trailing Stop ($):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.trailing_stop_var = tk.DoubleVar(value=self.config.get('trailing_stop_distance', 15))
        tk.Entry(profit_params, textvariable=self.trailing_stop_var).grid(row=1, column=1, padx=5, pady=2)
        
        # Quick Profit Mode
        self.quick_profit_var = tk.BooleanVar(value=self.config.get('quick_profit_mode', True))
        tk.Checkbutton(profit_params, text="Quick Profit Mode (เก็บกำไรเร็วขึ้น)", 
                      variable=self.quick_profit_var).grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        # Profit Mode Selection
        tk.Label(profit_params, text="Profit Mode:").grid(row=3, column=0, sticky='w', padx=5, pady=2)
        self.profit_mode_var = tk.StringVar(value=self.config.get('profit_mode', 'balanced'))
        profit_mode_combo = ttk.Combobox(profit_params, textvariable=self.profit_mode_var, 
                                       values=['conservative', 'balanced', 'aggressive', 'scalping'])
        profit_mode_combo.grid(row=3, column=1, padx=5, pady=2)
        
        # Apply Profit Settings Button
        tk.Button(profit_params, text="Apply Profit Settings", 
                 command=self.apply_profit_settings, bg='lightgreen').grid(row=4, column=0, columnspan=2, pady=5)
        
        # Configuration Buttons
        config_buttons = ttk.Frame(self.config_frame)
        config_buttons.pack(fill='x', padx=10, pady=10)
        
        tk.Button(config_buttons, text="Save Config", 
                 command=self.save_config).pack(side='left', padx=5)
        tk.Button(config_buttons, text="Load Config", 
                 command=self.load_config_dialog).pack(side='left', padx=5)
        tk.Button(config_buttons, text="Reset to Default", 
                 command=self.reset_config).pack(side='left', padx=5)
                 
    def apply_profit_settings(self):
        """Apply profit settings to recovery engine in real-time"""
        try:
            if hasattr(self, 'recovery_engine') and self.recovery_engine:
                new_settings = {
                    'min_profit_target': self.min_profit_var.get(),
                    'trailing_stop_distance': self.trailing_stop_var.get(),
                    'quick_profit_mode': self.quick_profit_var.get(),
                    'profit_mode': self.profit_mode_var.get()
                }
                
                # เรียก method และเช็ค return value
                success = self.recovery_engine.update_profit_settings(new_settings)
                
                if success:
                    self.log_message(f"Profit settings applied: Target=${new_settings['min_profit_target']:.1f}, Quick={new_settings['quick_profit_mode']}")
                    messagebox.showinfo("Success", f"Profit settings applied!\nTarget: ${new_settings['min_profit_target']:.1f}\nMode: {new_settings['profit_mode']}")
                    
                    # Update config ด้วย
                    self.config.update(new_settings)
                    
                    # Update GUI display
                    self.update_recovery_status()
                    
                else:
                    self.log_message("Failed to apply profit settings")
                    messagebox.showerror("Error", "Failed to apply profit settings")
                    
            else:
                self.log_message("Recovery engine not initialized yet")
                messagebox.showwarning("Warning", "Start trading first to apply profit settings")
                
        except Exception as e:
            self.log_message(f"Error applying profit settings: {str(e)}")
            messagebox.showerror("Error", f"Error applying settings: {str(e)}")            
    
    def update_recovery_status(self):
        """Update recovery status display with profit info"""
        try:
            if hasattr(self, 'recovery_engine') and self.recovery_engine:
                recovery_info = self.recovery_engine.get_status()
                
                # Update recovery status
                recovery_text = f"Recovery: {'Active' if recovery_info.get('recovery_active', False) else 'Inactive'}"
                if recovery_info.get('recovery_active', False):
                    recovery_text += f" (Level {recovery_info.get('recovery_level', 0)})"
                    
                self.recovery_status.config(text=recovery_text)
                self.recovery_level.config(text=f"Level: {recovery_info.get('recovery_level', 0)}")
                
                # Update profit settings display
                profit_info = recovery_info.get('profit_settings', {})
                profit_text = f"Profit: ${profit_info.get('min_profit_target', 25):.0f}"
                if profit_info.get('quick_profit_mode', False):
                    profit_text += " (Quick)"
                    
                # Add profit status to recovery frame
                if hasattr(self, 'profit_status_label'):
                    self.profit_status_label.config(text=profit_text)
                else:
                    # Create profit status label if not exists
                    recovery_frame = self.recovery_status.master
                    self.profit_status_label = tk.Label(recovery_frame, text=profit_text)
                    self.profit_status_label.pack(side='right', padx=10)
            else:
                self.recovery_status.config(text="Recovery: Not Ready")
                self.recovery_level.config(text="Level: 0")
                
        except Exception as e:
            self.log_message(f"Error updating recovery status: {str(e)}")
        
    def setup_training(self):
        # Training Control Frame
        training_control = ttk.LabelFrame(self.training_frame, text="Training Control")
        training_control.pack(fill='x', padx=10, pady=5)
        
        self.start_training_btn = tk.Button(training_control, text="Start Training", 
                                          command=self.start_training)
        self.start_training_btn.pack(side='left', padx=5)
        
        self.stop_training_btn = tk.Button(training_control, text="Stop Training", 
                                         command=self.stop_training, 
                                         state='disabled')
        self.stop_training_btn.pack(side='left', padx=5)
        
        self.pause_training_btn = tk.Button(training_control, text="Pause Training", 
                                          command=self.pause_training, 
                                          state='disabled')
        self.pause_training_btn.pack(side='left', padx=5)
        self.save_model_btn = tk.Button(training_control, text="Save Model Now", 
                                   command=self.save_model_manual,
                                   bg='lightgreen')
        self.save_model_btn.pack(side='left', padx=5)
        # Training Progress Frame
        progress_frame = ttk.LabelFrame(self.training_frame, text="Training Progress")
        progress_frame.pack(fill='x', padx=10, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill='x', padx=5, pady=5)
        
        self.progress_label = tk.Label(progress_frame, text="Ready to start training")
        self.progress_label.pack(pady=5)
        
        # Training Statistics Frame
        stats_frame = ttk.LabelFrame(self.training_frame, text="Training Statistics")
        stats_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=15)
        stats_scrollbar = ttk.Scrollbar(stats_frame, orient='vertical', command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        self.stats_text.pack(side='left', fill='both', expand=True)
        stats_scrollbar.pack(side='right', fill='y')
        
    def setup_performance(self):
        # Performance metrics will be displayed here
        # This would include charts, statistics, etc.
        metrics_frame = ttk.LabelFrame(self.performance_frame, text="Performance Metrics")
        metrics_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Placeholder for performance charts
        self.performance_text = tk.Text(metrics_frame)
        self.performance_text.pack(fill='both', expand=True)
        
    def setup_log_tags(self):
        """Setup color tags for different log types - FIXED VERSION"""
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
        
        # Special highlights - FIXED: ใช้สี RGB ธรรมดา
        self.log_text.tag_configure("HIGHLIGHT", background="#404040", foreground="#ffd33d")
        self.log_text.tag_configure("URGENT", background="#3d1a1a", foreground="#f85149")

    # ========================= COMPLETE FIXED SETUP_LOGS =========================

    def setup_logs(self):
        """Enhanced log setup with better formatting and filtering - FIXED VERSION"""
        
        # Main container frame
        main_container = ttk.Frame(self.logs_frame)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Top control panel
        control_panel = ttk.LabelFrame(main_container, text="Log Controls")
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
        log_display_frame = ttk.LabelFrame(main_container, text="Trading Logs")
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
        save_btn = tk.Button(button_frame, text="Save", command=self.save_logs,
                            bg='#238636', fg='white', font=('Arial', 9, 'bold'),
                            relief='flat', padx=10, pady=2)
        save_btn.pack(side='left', padx=(0, 5))
        
        export_btn = tk.Button(button_frame, text="Export", command=self.export_logs,
                            bg='#1f6feb', fg='white', font=('Arial', 9, 'bold'),
                            relief='flat', padx=10, pady=2)
        export_btn.pack(side='left', padx=(0, 5))
        
        clear_btn = tk.Button(button_frame, text="Clear", command=self.clear_logs,
                            bg='#da3633', fg='white', font=('Arial', 9, 'bold'),
                            relief='flat', padx=10, pady=2)
        clear_btn.pack(side='left', padx=(0, 5))
        
        # Pause/Resume logging
        self.logging_paused = tk.BooleanVar(value=False)
        pause_btn = tk.Checkbutton(button_frame, text="Pause", variable=self.logging_paused,
                                font=('Arial', 9, 'bold'), fg='#f85149')
        pause_btn.pack(side='left', padx=(0, 5))
        
        # Initialize log storage
        self.all_logs = []
        self.filtered_logs = []
        self.log_count = 0
        self.displayed_count = 0

    # ========================= HELPER METHODS =========================

    def filter_logs(self, event=None):
        """Apply current filter settings to logs"""
        if not hasattr(self, 'all_logs'):
            return
            
        # Get filter values
        filter_type = self.log_filter.get()
        level_filter = self.log_level.get()
        search_term = self.search_var.get().lower()
        
        # Clear display
        self.log_text.delete(1.0, tk.END)
        
        # Filter logs
        self.filtered_logs = []
        for log_entry in self.all_logs:
            # Apply type filter
            if filter_type != "All" and not self.matches_type_filter(log_entry, filter_type):
                continue
                
            # Apply level filter  
            if level_filter != "All" and log_entry.get('level', 'INFO') != level_filter:
                continue
                
            # Apply search filter
            if search_term and search_term not in log_entry['message'].lower():
                continue
                
            self.filtered_logs.append(log_entry)
        
        # Display filtered logs
        for log_entry in self.filtered_logs:
            self.add_formatted_log_to_display(log_entry)
        
        # Update status
        self.displayed_count = len(self.filtered_logs)
        self.update_log_status()
        
        # Auto scroll to bottom
        if self.auto_scroll.get():
            self.log_text.see(tk.END)

    def matches_type_filter(self, log_entry, filter_type):
        """Check if log entry matches the type filter"""
        message = log_entry['message'].upper()
        
        if filter_type == "AI Decisions":
            return "AI DECISION" in message or "AI" in message
        elif filter_type == "Trading":
            return any(keyword in message for keyword in ["BUY", "SELL", "CLOSE", "ORDER", "POSITION"])
        elif filter_type == "Profits":
            return any(keyword in message for keyword in ["PROFIT", "PNL", "$"])
        elif filter_type == "Errors":
            return log_entry.get('level') == 'ERROR' or "ERROR" in message
        elif filter_type == "System":
            return not any(keyword in message for keyword in ["AI DECISION", "BUY", "SELL", "PROFIT", "ERROR"])
        
        return True

    def search_logs(self, event=None):
        """Perform real-time search filtering"""
        self.filter_logs()

    def clear_search(self):
        """Clear search term and refresh"""
        self.search_var.set("")
        self.filter_logs()

    def update_log_status(self):
        """Update the status bar with current log statistics"""
        if hasattr(self, 'filtered_logs'):
            last_time = self.all_logs[-1]['timestamp'] if self.all_logs else "--:--:--"
            status_text = f"Logs: {self.log_count} entries | Filtered: {self.displayed_count} | Last: {last_time}"
            
            # Add filter info if active
            if (self.log_filter.get() != "All" or 
                self.log_level.get() != "All" or 
                self.search_var.get()):
                status_text += " | FILTERED"
                
            self.log_status.config(text=status_text)

    def add_formatted_log_to_display(self, log_entry):
        """Add formatted log entry to display"""
        timestamp = log_entry['timestamp']
        message = log_entry['message']
        level = log_entry.get('level', 'INFO')
        
        # Insert timestamp
        self.log_text.insert(tk.END, f"[{timestamp}] ", "TIME")
        
        # Format and insert message with appropriate tag
        if "AI Decision" in message:
            self.log_text.insert(tk.END, "🤖 " + message, "AI")
        elif any(keyword in message.upper() for keyword in ["BUY", "SELL", "CLOSE", "ORDER"]):
            self.log_text.insert(tk.END, "📈 " + message, "TRADE")
        elif any(keyword in message.upper() for keyword in ["PROFIT", "PNL", "$"]):
            self.log_text.insert(tk.END, "💰 " + message, "PROFIT")
        elif level == "ERROR" or "ERROR" in message.upper():
            self.log_text.insert(tk.END, "🚨 " + message, "ERROR")
        elif level == "WARNING" or "WARNING" in message.upper():
            self.log_text.insert(tk.END, "⚠️ " + message, "WARNING")
        elif level == "SUCCESS" or "SUCCESS" in message.upper():
            self.log_text.insert(tk.END, "✅ " + message, "SUCCESS")
        else:
            self.log_text.insert(tk.END, "ℹ️ " + message, "SYSTEM")
        
        self.log_text.insert(tk.END, "\n")

    # ========================= ENHANCED LOG_MESSAGE =========================

    def log_message(self, message, level="INFO"):
        """Enhanced log message with formatting and colors - FIXED VERSION"""
        from datetime import datetime
        
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

    # ========================= EXPORT FUNCTIONS =========================

    def save_logs(self):
        """Save current filtered logs to file"""
        try:
            import os
            from datetime import datetime
            from tkinter import messagebox
            
            os.makedirs('logs', exist_ok=True)
            filename = f"logs/trading_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            logs_to_save = getattr(self, 'filtered_logs', []) or getattr(self, 'all_logs', [])
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Trading System Logs - Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Entries: {len(logs_to_save)}\n")
                f.write("=" * 80 + "\n\n")
                
                for log_entry in logs_to_save:
                    f.write(f"[{log_entry['timestamp']}] [{log_entry.get('level', 'INFO')}] {log_entry['message']}\n")
            
            messagebox.showinfo("Success", f"Logs saved to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save logs:\n{str(e)}")

    def export_logs(self):
        """Export logs in various formats"""
        try:
            from tkinter import filedialog, messagebox
            import json
            from datetime import datetime
            
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
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export logs:\n{str(e)}")

    def clear_logs(self):
        """Enhanced clear logs with confirmation"""
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

    def connect_mt5(self):
        try:
            if self.mt5_interface.connect():
                self.is_connected = True
                self.conn_status.config(text="Connected", fg="green")
                self.start_trading_btn.config(state='normal')
                self.log_message("MT5 connected successfully")
                self.update_account_info()
            else:
                self.log_message("Failed to connect to MT5")
                messagebox.showerror("Connection Error", "Failed to connect to MT5")
        except Exception as e:
            self.log_message(f"MT5 connection error: {str(e)}")
            messagebox.showerror("Connection Error", f"Error: {str(e)}")
            
    def start_trading(self):
        if not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to MT5 first")
            return
            
        try:
            print("DEBUG: Starting LIVE trading...")
            
            # เปลี่ยนเป็น LIVE MODE
            self.config['training_mode'] = False  # ← เพิ่มบรรทัดนี้
            
            # Set live mode for MT5 interface
            self.mt5_interface.set_training_mode(False)  # ← เพิ่มบรรทัดนี้
            
            # Initialize trading environment and RL agent
            self.trading_env = TradingEnvironment(self.mt5_interface, self.recovery_engine, self.config)
            self.rl_agent = RLAgent(self.trading_env, self.config)
            
            # Load trained model if available - ปรับปรุงข้อความ
            if self.rl_agent.load_model():
                self.log_message("✅ AI Model loaded successfully!")
                messagebox.showinfo("Success", "AI Trading Model Loaded!")  # ← เพิ่มบรรทัดนี้
            else:
                self.log_message("⚠️ No trained model found - using random actions")
                result = messagebox.askyesno("Warning", "No AI model found!\nContinue with random actions?")  # ← เพิ่มบรรทัดนี้
                if not result:  # ← เพิ่มบรรทัดนี้
                    return  # ← เพิ่มบรรทัดนี้
                
            self.is_trading = True
            self.start_trading_btn.config(state='disabled')
            self.stop_trading_btn.config(state='normal')
            
            # Start trading thread
            self.trading_thread = threading.Thread(target=self.trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            self.log_message("🚀 LIVE AI Trading Started!")  # ← ปรับปรุงข้อความ
            
        except Exception as e:
            self.log_message(f"Error starting trading: {str(e)}")
            messagebox.showerror("Trading Error", f"Error: {str(e)}")

    def stop_trading(self):
        self.is_trading = False
        self.start_trading_btn.config(state='normal')
        self.stop_trading_btn.config(state='disabled')
        self.log_message("Trading stopped")
        
    def start_training(self):
        if not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to MT5 first")
            return
            
        try:
            print("DEBUG: Starting training...")
            
            # Initialize training environment
            print("DEBUG: Creating environment...")
            self.trading_env = TradingEnvironment(self.mt5_interface, self.recovery_engine, self.config)
            
            print("DEBUG: Creating RL agent...")
            self.rl_agent = RLAgent(self.trading_env, self.config)
            self.config['training_mode'] = False
            self.is_training = False
            self.start_training_btn.config(state='disabled')
            self.stop_training_btn.config(state='normal')
            self.pause_training_btn.config(state='normal')
            
            print("DEBUG: Starting training thread...")
            self.trading_env = TradingEnvironment(self.mt5_interface, self.recovery_engine, self.config)
            # Start training thread with callback
            self.training_thread = threading.Thread(target=self.training_loop_with_callback)
            self.training_thread.daemon = True
            self.training_thread.start()
            
            self.log_message("Training started")
            
        except Exception as e:
            print(f"DEBUG: Error in start_training: {str(e)}")
            self.log_message(f"Error starting training: {str(e)}")
            messagebox.showerror("Training Error", f"Error: {str(e)}")

    def training_loop_with_callback(self):
        try:
            print("DEBUG: Inside training_loop_with_callback")
            
            # Create callback for GUI updates
            def gui_callback(step, total_steps, reward=None):
                progress = (step / total_steps) * 100
                
                # Update progress bar
                self.root.after(0, lambda: self.progress_bar.config(value=progress))
                self.root.after(0, lambda: self.progress_label.config(
                    text=f"Training: {step:,}/{total_steps:,} ({progress:.1f}%)"
                ))
                
                # Update statistics
                stats_text = f"Step: {step:,}/{total_steps:,}\n"
                stats_text += f"Progress: {progress:.1f}%\n"
                if reward:
                    stats_text += f"Current Reward: {reward:.4f}\n"
                stats_text += f"Time: {datetime.now().strftime('%H:%M:%S')}\n"
                stats_text += "=" * 40 + "\n"
                
                self.root.after(0, lambda: self.stats_text.insert(tk.END, stats_text))
                self.root.after(0, lambda: self.stats_text.see(tk.END))
            
            # Start training with callback
            success = self.rl_agent.train(
                total_timesteps=self.training_steps_var.get(),
                callback=gui_callback
            )
            
            print(f"DEBUG: train returned: {success}")
            
            if success:
                self.log_message("Training completed successfully")
                self.root.after(0, lambda: self.progress_bar.config(value=100))
                self.root.after(0, lambda: self.progress_label.config(text="Training completed!"))
            else:
                self.log_message("Training failed - check logs")
                self.root.after(0, lambda: self.progress_label.config(text="Training failed"))
                
        except Exception as e:
            print(f"DEBUG: training_loop error: {str(e)}")
            self.log_message(f"Training error: {str(e)}")
            self.root.after(0, lambda: self.progress_label.config(text="Training error"))
            
        finally:
            print("DEBUG: training_loop finished")
            self.is_training = False
            self.root.after(0, lambda: self.start_training_btn.config(state='normal'))
            self.root.after(0, lambda: self.stop_training_btn.config(state='disabled'))
            self.root.after(0, lambda: self.pause_training_btn.config(state='disabled'))

    def stop_training(self):
        self.is_training = False
        self.start_training_btn.config(state='normal')
        self.stop_training_btn.config(state='disabled')
        self.pause_training_btn.config(state='disabled')
        self.log_message("Training stopped")
        
    def pause_training(self):
        # Implementation for pause functionality
        pass
        
    def trading_loop(self):
        while self.is_trading:
            try:
                self.trading_env.update_market_data()
                # Get observation from environment (แทน market_data)
                observation = self.trading_env._get_observation()
                self.log_message(f"🔍 Observation updated: {observation[:5]}...")  # ← เพิ่ม

                # Get RL agent decision
                action = self.rl_agent.get_action(observation)
                action = self.force_ai_diversity(action)

                action[0] += np.random.normal(0, 0.05)  # Add noise to action
                action[0] = np.clip(action[0], 0, 4)    # Keep in range
                self.log_message(f"🤖 AI Decision: {action}")  # ← เพิ่ม

                # Execute action
                self.execute_action(action)
                
                # Update GUI
                self.root.after(0, self.update_gui)
                
                # Sleep for a bit
                threading.Event().wait(1)
                
            except Exception as e:
                self.log_message(f"Trading loop error: {str(e)}")

    def training_loop(self):
        try:
            print("DEBUG: Inside training_loop")
            
            # Update GUI แสดงว่าเริ่มเทรน
            self.root.after(0, lambda: self.progress_label.config(text="Training in progress..."))
            self.root.after(0, lambda: self.progress_bar.config(value=1))
            
            print("DEBUG: Calling train...")
            success = self.rl_agent.train(
                total_timesteps=self.training_steps_var.get()
            )
            
            print(f"DEBUG: train returned: {success}")
            
            if success:
                self.log_message("Training completed successfully")
                # Update progress to 100%
                self.root.after(0, lambda: self.progress_bar.config(value=100))
                self.root.after(0, lambda: self.progress_label.config(text="Training completed!"))
            else:
                self.log_message("Training failed - check logs")
                self.root.after(0, lambda: self.progress_label.config(text="Training failed"))
                
        except Exception as e:
            print(f"DEBUG: training_loop error: {str(e)}")
            self.log_message(f"Training error: {str(e)}")
            self.root.after(0, lambda: self.progress_label.config(text="Training error"))
            
        finally:
            print("DEBUG: training_loop finished")
            self.is_training = False
            self.root.after(0, lambda: self.start_training_btn.config(state='normal'))
            self.root.after(0, lambda: self.stop_training_btn.config(state='disabled'))
            self.root.after(0, lambda: self.pause_training_btn.config(state='disabled'))

    def training_callback(self, locals_dict, globals_dict):
        # Update training progress
        if 'self' in locals_dict:
            step = locals_dict.get('step', 0)
            total_steps = self.training_steps_var.get()
            progress = (step / total_steps) * 100
            
            self.root.after(0, lambda: self.progress_bar.config(value=progress))
            self.root.after(0, lambda: self.progress_label.config(text=f"Step {step}/{total_steps}"))
            
        return self.is_training
        
    def execute_action(self, action):
        """Execute trading action through environment"""
        try:
            # ส่ง action ไปที่ environment เพื่อประมวลผล
            observation, reward, done, truncated, info = self.trading_env.step(action)
            
            # Log ผลลัพธ์
            self.log_message(f"⚡ Action executed: reward={reward:.3f}, done={done}")
            
            # Update info
            if info:
                self.log_message(f"📊 Info: {info}")
                
            return reward
            
        except Exception as e:
            self.log_message(f"❌ Execute action error: {str(e)}")
            return 0.0        
    
    def update_gui(self):
        # Update position table
        self.update_positions()
        
        # Update account information
        self.update_account_info()
        
        # Update recovery status
        self.update_recovery_status()
        
    def update_positions(self):
        # Clear existing items
        for item in self.pos_tree.get_children():
            self.pos_tree.delete(item)
            
        # Get current positions from MT5
        positions = self.mt5_interface.get_positions()
        
        for pos in positions:
            self.pos_tree.insert('', 'end', values=(
                pos.get('symbol', ''),
                pos.get('type', ''),
                pos.get('volume', ''),
                pos.get('price_open', ''),
                pos.get('profit', '')
            ))
            
    def update_account_info(self):
        account_info = self.mt5_interface.get_account_info()
        if account_info:
            self.balance_label.config(text=f"Balance: ${account_info.get('balance', 0):.2f}")
            self.equity_label.config(text=f"Equity: ${account_info.get('equity', 0):.2f}")
            self.margin_label.config(text=f"Margin: ${account_info.get('margin', 0):.2f}")
            
    def update_recovery_status(self):
        # Update recovery status display
        recovery_info = self.recovery_engine.get_status()
        self.recovery_status.config(text=recovery_info.get('status', 'No Recovery Active'))
        self.recovery_level.config(text=f"Level: {recovery_info.get('level', 0)}")
        
    def log_message(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
    def clear_logs(self):
        self.log_text.delete(1.0, tk.END)

    def save_model_manual(self):
        """Save current model manually"""
        try:
            if hasattr(self, 'rl_agent') and self.rl_agent and self.rl_agent.model:
                # Save model
                save_path = self.rl_agent.save_model("manual_save")
                
                if save_path:
                    self.log_message(f"✅ Model saved successfully: {save_path}")
                    messagebox.showinfo("Success", f"Model saved!\nPath: {save_path}.zip")
                else:
                    self.log_message("❌ Failed to save model")
                    messagebox.showerror("Error", "Failed to save model")
            else:
                self.log_message("❌ No model to save")
                messagebox.showwarning("Warning", "No trained model found in memory")
                
        except Exception as e:
            self.log_message(f"Save error: {str(e)}")
            messagebox.showerror("Error", f"Save error: {str(e)}")

    def save_config(self):
        """Save current configuration including profit settings"""
        config = {
            'symbol': self.symbol_var.get(),
            'initial_lot_size': self.lot_size_var.get(),
            'max_positions': self.max_positions_var.get(),
            'martingale_multiplier': self.martingale_mult_var.get(),
            'grid_spacing': self.grid_spacing_var.get(),
            'recovery_type': self.recovery_type_var.get(),
            'algorithm': self.algorithm_var.get(),
            'learning_rate': self.learning_rate_var.get(),
            'training_steps': self.training_steps_var.get(),
            # เพิ่ม profit settings
            'min_profit_target': self.min_profit_var.get(),
            'trailing_stop_distance': self.trailing_stop_var.get(),
            'quick_profit_mode': self.quick_profit_var.get(),
            'profit_mode': self.profit_mode_var.get()
        }
        
        os.makedirs('config', exist_ok=True)
        with open('config/user_config.json', 'w') as f:
            json.dump(config, f, indent=4)
            
        self.log_message("Configuration saved with profit settings")
        messagebox.showinfo("Success", "Configuration saved successfully")
        
    def load_config(self):
        """Load configuration including profit settings"""
        try:
            with open('config/user_config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default config with profit settings
            return {
                'symbol': 'XAUUSD',
                'initial_lot_size': 0.01,
                'max_positions': 10,
                'martingale_multiplier': 2.0,
                'grid_spacing': 100,
                'recovery_type': 'Martingale',
                'algorithm': 'PPO',
                'learning_rate': 0.0003,
                'training_steps': 1000,
                'training_mode': False,
                # Default profit settings - เก็บกำไรเร็วขึ้น
                'min_profit_target': 25,
                'trailing_stop_distance': 15,
                'quick_profit_mode': True,
                'profit_mode': 'balanced'
            }
            
    def load_config_dialog(self):
        """Load configuration from file dialog"""
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    config = json.load(f)
                    
                # Update GUI with loaded config including profit settings
                self.symbol_var.set(config.get('symbol', 'XAUUSD'))
                self.lot_size_var.set(config.get('initial_lot_size', 0.01))
                self.max_positions_var.set(config.get('max_positions', 10))
                self.martingale_mult_var.set(config.get('martingale_multiplier', 2.0))
                self.grid_spacing_var.set(config.get('grid_spacing', 100))
                self.recovery_type_var.set(config.get('recovery_type', 'Martingale'))
                self.algorithm_var.set(config.get('algorithm', 'PPO'))
                self.learning_rate_var.set(config.get('learning_rate', 0.0003))
                self.training_steps_var.set(config.get('training_steps', 10000))
                
                # Load profit settings
                self.min_profit_var.set(config.get('min_profit_target', 25))
                self.trailing_stop_var.set(config.get('trailing_stop_distance', 15))
                self.quick_profit_var.set(config.get('quick_profit_mode', True))
                self.profit_mode_var.set(config.get('profit_mode', 'balanced'))
                
                self.log_message(f"Configuration loaded from {filename}")
                messagebox.showinfo("Success", "Configuration loaded successfully")
                
            except Exception as e:
                self.log_message(f"Error loading configuration: {str(e)}")
                messagebox.showerror("Error", f"Error loading configuration: {str(e)}")
                
    def reset_config(self):
        """Reset to default values including profit settings"""
        # Reset trading parameters
        self.symbol_var.set('XAUUSD')

        self.lot_size_var.set(0.01)
        self.max_positions_var.set(10)
        self.martingale_mult_var.set(2.0)
        self.grid_spacing_var.set(100)
        self.recovery_type_var.set('Martingale')
        self.algorithm_var.set('PPO')
        self.learning_rate_var.set(0.0003)
        self.training_steps_var.set(10000)
        
        # Reset profit settings to quick profit defaults
        self.min_profit_var.set(25)  # เริ่มต้นที่ $25
        self.trailing_stop_var.set(15)  # Trailing stop $15
        self.quick_profit_var.set(True)  # เปิด quick mode
        self.profit_mode_var.set('balanced')  # โหมด balanced
        
        self.log_message("Configuration reset to defaults (Quick Profit Mode)")
        messagebox.showinfo("Reset", "Configuration reset to quick profit defaults")
        
    def run(self):
        self.root.mainloop()
        
    def __del__(self):
        # Cleanup
        if hasattr(self, 'mt5_interface'):
            self.mt5_interface.disconnect()

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs('config', exist_ok=True)
    os.makedirs('models/trained_models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('utils', exist_ok=True)
    
    # Run the application
    app = TradingGUI()
    app.run()
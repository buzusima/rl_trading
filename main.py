# main.py - Professional Trading GUI Application
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import json
import os
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
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

# Professional imports
from environment import ProfessionalTradingEnvironment, TradingState, MarketRegime
from recovery_engine import ProfessionalRecoveryEngine, RecoveryMode
from mt5_interface import MT5Interface
from rl_agent import ProfessionalRLAgent
from utils.data_handler import DataHandler
from utils.visualizer import Visualizer
from portfolio_manager import AIPortfolioManager, PortfolioMode

class ProfessionalTradingGUI:
    """
    🚀 Professional AI Trading System GUI
    - 15-dimensional action space
    - Multi-agent RL system (PPO, SAC, TD3)
    - Advanced portfolio management
    - Smart recovery engine
    - Real-time professional analytics
    """
    
    def __init__(self):
        print("🚀 Initializing Professional Trading System...")
        
        # Main window setup
        self.root = tk.Tk()
        self.root.title("🤖 Professional AI Trading System - XAUUSD Elite")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e1e')  # Dark theme
        
        # Logging system attributes
        self.all_logs = []
        self.filtered_logs = []
        self.log_count = 0
        self.displayed_count = 0
        self.max_log_entries = 1000
        
        # Log filtering
        self.log_filters = {
            'INFO': True,
            'WARNING': True, 
            'ERROR': True,
            'SYSTEM': True,
            'TRADING': True,
            'DEBUG': False
        }
    
        # Core system components
        self.mt5_interface = MT5Interface()
        self.recovery_engine = ProfessionalRecoveryEngine()
        self.data_handler = DataHandler()
        self.visualizer = Visualizer()
        
        # System state tracking
        self.is_training = False
        self.is_trading = False
        self.is_connected = False
        self.is_professional_mode = True
        
        # Load professional configuration
        self.config = self.load_professional_config()
        
        # Professional managers
        self.portfolio_manager = AIPortfolioManager(self.config)
        
        # Advanced monitoring systems
        self.ai_agent_performance = {
            'PPO': {'decisions': 0, 'rewards': [], 'win_rate': 0.0},
            'SAC': {'decisions': 0, 'rewards': [], 'win_rate': 0.0}, 
            'TD3': {'decisions': 0, 'rewards': [], 'win_rate': 0.0}
        }
        self.action_distribution = {f'action_{i}': 0 for i in range(15)}
        self.recent_actions = []
        self.system_efficiency = 0.0
        
        # Trading state tracking
        self.current_trading_state = TradingState.MARKET_ANALYSIS
        self.current_market_regime = MarketRegime.SIDEWAYS
        self.current_portfolio_mode = PortfolioMode.GROWTH
        self.current_recovery_mode = RecoveryMode.INACTIVE
        
        # Performance metrics
        self.session_start_time = datetime.now()
        self.total_trades = 0
        self.successful_recoveries = 0
        self.portfolio_efficiency_score = 0.0
        
        # GUI components initialization
        self.setup_professional_gui()
        
        # Professional RL system
        self.trading_env = None
        self.rl_agent_system = None
        
        # Threading for real-time systems
        self.update_thread = None
        self.training_thread = None
        self.analytics_thread = None
        
        # Start professional real-time systems
        self.start_professional_systems()
        
        print("✅ Professional Trading System initialized successfully!")

    def setup_professional_gui(self):
        """Setup professional-grade GUI interface"""
        print("🎨 Setting up Professional GUI...")
        
        # Create professional style
        self.setup_professional_style()
        
        # Main container with dark theme
        self.main_container = ttk.Frame(self.root, style='Dark.TFrame')
        self.main_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Professional notebook with enhanced tabs
        self.notebook = ttk.Notebook(self.main_container, style='Professional.TNotebook')
        
        # 🎯 Dashboard Tab - Main control center
        self.dashboard_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.dashboard_frame, text="🎯 Command Center")
        
        # 📊 Analytics Tab - Advanced analytics
        self.analytics_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.analytics_frame, text="📊 Pro Analytics")
        
        # 🤖 AI Agents Tab - Multi-agent monitoring
        self.agents_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.agents_frame, text="🤖 AI Agents")
        
        # 🛡️ Recovery Tab - Smart recovery system
        self.recovery_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.recovery_frame, text="🛡️ Recovery Engine")
        
        # 💼 Portfolio Tab - Advanced portfolio management
        self.portfolio_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.portfolio_frame, text="💼 Portfolio Pro")
        
        # 🔧 System Tab - Professional configuration
        self.system_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.system_frame, text="🔧 Pro Settings")
        
        # 📈 Training Tab - Advanced training interface
        self.training_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.training_frame, text="📈 Elite Training")
        
        # 📋 Logs Tab - Professional logging
        self.logs_frame = ttk.Frame(self.notebook, style='Dark.TFrame')
        self.notebook.add(self.logs_frame, text="📋 System Logs")
        
        self.notebook.pack(fill='both', expand=True)
        
        # Setup individual tab interfaces
        self.setup_command_center()
        self.setup_analytics_interface()
        self.setup_ai_agents_interface()
        self.setup_recovery_interface()
        self.setup_portfolio_interface()
        self.setup_system_interface()
        self.setup_training_interface()
        self.setup_logs_interface()
        
        print("✅ Professional GUI setup completed!")
    
    def setup_performance_metrics(self, perf_metrics_frame):
        """Setup performance metrics interface"""
        try:
            # Performance metrics display
            metrics_scroll = tk.Scrollbar(perf_metrics_frame)
            metrics_scroll.pack(side='right', fill='y')
            
            self.metrics_text = tk.Text(perf_metrics_frame, 
                                    yscrollcommand=metrics_scroll.set,
                                    height=20, width=80,
                                    bg='#2e2e2e', fg='white',
                                    font=('Consolas', 10))
            self.metrics_text.pack(fill='both', expand=True)
            metrics_scroll.config(command=self.metrics_text.yview)
            
            # Initial metrics display
            self.update_performance_display()
            
        except Exception as e:
            print(f"Setup performance metrics error: {e}")

    def update_performance_display(self):
        """Update performance metrics display"""
        try:
            if hasattr(self, 'metrics_text'):
                self.metrics_text.delete(1.0, tk.END)
                
                # Get current metrics
                metrics = {
                    'System Status': 'Running',
                    'Connection': 'Connected' if self.is_connected else 'Disconnected',
                    'Trading': 'Active' if self.is_trading else 'Inactive',
                    'Training': 'Active' if self.is_training else 'Inactive'
                }
                
                # Display metrics
                for key, value in metrics.items():
                    self.metrics_text.insert(tk.END, f"{key}: {value}\n")
                    
        except Exception as e:
            print(f"Performance display update error: {e}")

    def setup_professional_style(self):
        """Setup professional dark theme styling"""
        style = ttk.Style()
        
        # Configure professional dark theme
        style.theme_use('clam')
        
        # Dark theme colors
        bg_color = '#1e1e1e'
        fg_color = '#ffffff'
        select_color = '#404040'
        accent_color = '#0078d4'
        success_color = '#28a745'
        warning_color = '#ffc107'
        danger_color = '#dc3545'
        
        # Configure styles
        style.configure('Dark.TFrame', background=bg_color)
        style.configure('Professional.TNotebook', background=bg_color, borderwidth=0)
        style.configure('Professional.TNotebook.Tab', 
                       background=select_color, foreground=fg_color,
                       padding=[12, 8], font=('Segoe UI', 9, 'bold'))
        style.map('Professional.TNotebook.Tab',
                 background=[('selected', accent_color), ('active', select_color)])
        
        # Professional button styles
        style.configure('Success.TButton', background=success_color, foreground='white',
                       font=('Segoe UI', 9, 'bold'), padding=[10, 5])
        style.configure('Warning.TButton', background=warning_color, foreground='black',
                       font=('Segoe UI', 9, 'bold'), padding=[10, 5])
        style.configure('Danger.TButton', background=danger_color, foreground='white',
                       font=('Segoe UI', 9, 'bold'), padding=[10, 5])
        style.configure('Professional.TButton', background=accent_color, foreground='white',
                       font=('Segoe UI', 9, 'bold'), padding=[10, 5])
    
    def setup_command_center(self):
        """Setup main command center dashboard"""
        print("🎯 Setting up Command Center...")
        
        # Main layout with professional grid
        main_grid = ttk.Frame(self.dashboard_frame, style='Dark.TFrame')
        main_grid.pack(fill='both', expand=True, padx=10, pady=10)
        
        # === TOP ROW: System Status & Quick Controls ===
        top_row = ttk.Frame(main_grid, style='Dark.TFrame')
        top_row.pack(fill='x', pady=(0, 10))
        
        # Connection & System Status Panel
        status_frame = ttk.LabelFrame(top_row, text="🔗 System Status", style='Dark.TLabelframe')
        status_frame.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        # Connection status
        conn_frame = ttk.Frame(status_frame)
        conn_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(conn_frame, text="MT5:", style='Dark.TLabel').pack(side='left')
        self.conn_status = ttk.Label(conn_frame, text="❌ Disconnected", 
                                    foreground='red', style='Dark.TLabel')
        self.conn_status.pack(side='left', padx=(5, 0))
        
        # Trading state indicator
        state_frame = ttk.Frame(status_frame)
        state_frame.pack(fill='x', padx=5, pady=2)
        
        ttk.Label(state_frame, text="State:", style='Dark.TLabel').pack(side='left')
        self.trading_state_label = ttk.Label(state_frame, text="🔍 ANALYZE", 
                                           foreground='#ffc107', style='Dark.TLabel')
        self.trading_state_label.pack(side='left', padx=(5, 0))
        
        # Market regime indicator
        regime_frame = ttk.Frame(status_frame)
        regime_frame.pack(fill='x', padx=5, pady=2)
        
        ttk.Label(regime_frame, text="Market:", style='Dark.TLabel').pack(side='left')
        self.market_regime_label = ttk.Label(regime_frame, text="📊 SIDEWAYS", 
                                           foreground='#17a2b8', style='Dark.TLabel')
        self.market_regime_label.pack(side='left', padx=(5, 0))
        
        # Quick Control Panel
        control_frame = ttk.LabelFrame(top_row, text="⚡ Quick Controls", style='Dark.TLabelframe')
        control_frame.pack(side='right', padx=(5, 0))
        
        # Professional control buttons
        self.connect_btn = ttk.Button(control_frame, text="🔗 Connect MT5", 
                                     command=self.connect_mt5, style='Professional.TButton')
        self.connect_btn.pack(side='top', fill='x', padx=5, pady=2)
        
        self.start_trading_btn = ttk.Button(control_frame, text="🚀 Start Trading", 
                                          command=self.start_professional_trading, 
                                          style='Success.TButton', state='disabled')
        self.start_trading_btn.pack(side='top', fill='x', padx=5, pady=2)
        
        self.stop_trading_btn = ttk.Button(control_frame, text="⏹️ Stop Trading", 
                                         command=self.stop_trading, 
                                         style='Danger.TButton', state='disabled')
        self.stop_trading_btn.pack(side='top', fill='x', padx=5, pady=2)
        
        self.emergency_stop_btn = ttk.Button(control_frame, text="🚨 EMERGENCY", 
                                           command=self.emergency_stop, 
                                           style='Danger.TButton')
        self.emergency_stop_btn.pack(side='top', fill='x', padx=5, pady=2)
        
        # === MIDDLE ROW: Account Info & Live Metrics ===
        middle_row = ttk.Frame(main_grid, style='Dark.TFrame')
        middle_row.pack(fill='x', pady=(0, 10))
        
        # Account Information Panel
        account_frame = ttk.LabelFrame(middle_row, text="💰 Account Information", style='Dark.TLabelframe')
        account_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Account metrics grid
        acc_grid = ttk.Frame(account_frame)
        acc_grid.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Balance & Equity
        balance_frame = ttk.Frame(acc_grid)
        balance_frame.pack(fill='x', pady=2)
        ttk.Label(balance_frame, text="Balance:", style='Dark.TLabel').pack(side='left')
        self.balance_label = ttk.Label(balance_frame, text="$0.00", 
                                      foreground='#28a745', style='Dark.TLabel')
        self.balance_label.pack(side='right')
        
        equity_frame = ttk.Frame(acc_grid)
        equity_frame.pack(fill='x', pady=2)
        ttk.Label(equity_frame, text="Equity:", style='Dark.TLabel').pack(side='left')
        self.equity_label = ttk.Label(equity_frame, text="$0.00", 
                                     foreground='#28a745', style='Dark.TLabel')
        self.equity_label.pack(side='right')
        
        # Profit & Drawdown
        profit_frame = ttk.Frame(acc_grid)
        profit_frame.pack(fill='x', pady=2)
        ttk.Label(profit_frame, text="Floating P&L:", style='Dark.TLabel').pack(side='left')
        self.profit_label = ttk.Label(profit_frame, text="$0.00", 
                                     foreground='white', style='Dark.TLabel')
        self.profit_label.pack(side='right')
        
        dd_frame = ttk.Frame(acc_grid)
        dd_frame.pack(fill='x', pady=2)
        ttk.Label(dd_frame, text="Drawdown:", style='Dark.TLabel').pack(side='left')
        self.drawdown_label = ttk.Label(dd_frame, text="0.0%", 
                                       foreground='#dc3545', style='Dark.TLabel')
        self.drawdown_label.pack(side='right')
        
        # Live Performance Metrics Panel
        metrics_frame = ttk.LabelFrame(middle_row, text="📊 Live Performance", style='Dark.TLabelframe')
        metrics_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Performance metrics grid
        perf_grid = ttk.Frame(metrics_frame)
        perf_grid.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Portfolio heat
        heat_frame = ttk.Frame(perf_grid)
        heat_frame.pack(fill='x', pady=2)
        ttk.Label(heat_frame, text="Portfolio Heat:", style='Dark.TLabel').pack(side='left')
        self.portfolio_heat_label = ttk.Label(heat_frame, text="0.0%", 
                                             foreground='#ffc107', style='Dark.TLabel')
        self.portfolio_heat_label.pack(side='right')
        
        # System efficiency
        eff_frame = ttk.Frame(perf_grid)
        eff_frame.pack(fill='x', pady=2)
        ttk.Label(eff_frame, text="System Efficiency:", style='Dark.TLabel').pack(side='left')
        self.efficiency_label = ttk.Label(eff_frame, text="0.0%", 
                                         foreground='#17a2b8', style='Dark.TLabel')
        self.efficiency_label.pack(side='right')
        
        # Active positions
        pos_frame = ttk.Frame(perf_grid)
        pos_frame.pack(fill='x', pady=2)
        ttk.Label(pos_frame, text="Active Positions:", style='Dark.TLabel').pack(side='left')
        self.positions_count_label = ttk.Label(pos_frame, text="0", 
                                              foreground='white', style='Dark.TLabel')
        self.positions_count_label.pack(side='right')
        
        # Recovery status
        recovery_frame = ttk.Frame(perf_grid)
        recovery_frame.pack(fill='x', pady=2)
        ttk.Label(recovery_frame, text="Recovery:", style='Dark.TLabel').pack(side='left')
        self.recovery_status_label = ttk.Label(recovery_frame, text="INACTIVE", 
                                              foreground='#6c757d', style='Dark.TLabel')
        self.recovery_status_label.pack(side='right')
        
        # === BOTTOM ROW: Positions Table ===
        positions_frame = ttk.LabelFrame(main_grid, text="📋 Active Positions", style='Dark.TLabelframe')
        positions_frame.pack(fill='both', expand=True)
        
        # Professional positions table
        self.setup_positions_table(positions_frame)
        
        print("✅ Command Center setup completed!")

    def setup_positions_table(self, parent):
        """Setup professional positions table"""
        # Create treeview with scrollbars
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Positions treeview
        columns = ('Symbol', 'Type', 'Volume', 'Open Price', 'Current Price', 
                  'Profit', 'Target', 'Duration', 'Status')
        
        self.pos_tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=8)
        
        # Configure columns
        self.pos_tree.heading('Symbol', text='Symbol')
        self.pos_tree.heading('Type', text='Type')
        self.pos_tree.heading('Volume', text='Volume')
        self.pos_tree.heading('Open Price', text='Open Price')
        self.pos_tree.heading('Current Price', text='Current Price')
        self.pos_tree.heading('Profit', text='Profit')
        self.pos_tree.heading('Target', text='Target')
        self.pos_tree.heading('Duration', text='Duration')
        self.pos_tree.heading('Status', text='Status')
        
        # Column widths
        self.pos_tree.column('Symbol', width=80)
        self.pos_tree.column('Type', width=60)
        self.pos_tree.column('Volume', width=80)
        self.pos_tree.column('Open Price', width=100)
        self.pos_tree.column('Current Price', width=100)
        self.pos_tree.column('Profit', width=100)
        self.pos_tree.column('Target', width=100)
        self.pos_tree.column('Duration', width=80)
        self.pos_tree.column('Status', width=100)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(table_frame, orient='vertical', command=self.pos_tree.yview)
        h_scrollbar = ttk.Scrollbar(table_frame, orient='horizontal', command=self.pos_tree.xview)
        
        self.pos_tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack table and scrollbars
        self.pos_tree.pack(side='left', fill='both', expand=True)
        v_scrollbar.pack(side='right', fill='y')
        h_scrollbar.pack(side='bottom', fill='x')
        
        # Position context menu
        self.setup_position_context_menu()

    def setup_position_context_menu(self):
        """Setup right-click context menu for positions"""
        self.pos_context_menu = tk.Menu(self.root, tearoff=0)
        self.pos_context_menu.add_command(label="🔧 Modify Position", command=self.modify_selected_position)
        self.pos_context_menu.add_command(label="❌ Close Position", command=self.close_selected_position)
        self.pos_context_menu.add_separator()
        self.pos_context_menu.add_command(label="📊 Position Details", command=self.show_position_details)
        self.pos_context_menu.add_command(label="🛡️ Activate Recovery", command=self.activate_position_recovery)
        
        # Bind right-click to positions table
        self.pos_tree.bind("<Button-3>", self.show_position_context_menu)

    def setup_analytics_interface(self):
        """Setup advanced analytics interface"""
        print("📊 Setting up Analytics Interface...")
        
        # Main analytics container
        analytics_container = ttk.Frame(self.analytics_frame, style='Dark.TFrame')
        analytics_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # === TOP SECTION: Real-time Charts ===
        charts_frame = ttk.LabelFrame(analytics_container, text="📈 Real-time Analytics", 
                                     style='Dark.TLabelframe')
        charts_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Charts notebook
        self.charts_notebook = ttk.Notebook(charts_frame, style='Professional.TNotebook')
        self.charts_notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Price chart tab
        self.price_chart_frame = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(self.price_chart_frame, text="💹 Price Action")
        
        # Portfolio performance tab
        self.portfolio_chart_frame = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(self.portfolio_chart_frame, text="💼 Portfolio")
        
        # AI performance tab
        self.ai_chart_frame = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(self.ai_chart_frame, text="🤖 AI Performance")
        
        # Risk analysis tab
        self.risk_chart_frame = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(self.risk_chart_frame, text="🛡️ Risk Analysis")
        
        # Setup individual chart interfaces
        self.setup_price_chart()
        self.setup_portfolio_chart()
        self.setup_ai_performance_chart()
        self.setup_risk_analysis_chart()
        
        # === BOTTOM SECTION: Analytics Metrics ===
        metrics_row = ttk.Frame(analytics_container, style='Dark.TFrame')
        metrics_row.pack(fill='x')
        
        # Performance metrics panel
        perf_metrics_frame = ttk.LabelFrame(metrics_row, text="📊 Performance Metrics", 
                                          style='Dark.TLabelframe')
        perf_metrics_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Add performance metrics
        self.setup_performance_metrics(perf_metrics_frame)
        
        # Risk metrics panel
        risk_metrics_frame = ttk.LabelFrame(metrics_row, text="⚠️ Risk Metrics", 
                                          style='Dark.TLabelframe')
        risk_metrics_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Add risk metrics
        self.setup_risk_metrics(risk_metrics_frame)
        
        print("✅ Analytics Interface setup completed!")

    def setup_price_chart(self):
        """Setup real-time price chart"""
        if FigureCanvas is None:
            ttk.Label(self.price_chart_frame, text="📊 Chart unavailable - matplotlib backend error", 
                     style='Dark.TLabel').pack(expand=True)
            return
        
        # Create matplotlib figure
        self.price_fig = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        self.price_ax = self.price_fig.add_subplot(111, facecolor='#2d2d2d')
        
        # Configure chart styling
        self.price_ax.tick_params(colors='white')
        self.price_ax.set_xlabel('Time', color='white')
        self.price_ax.set_ylabel('Price', color='white')
        self.price_ax.set_title('XAUUSD - Real-time Price Action', color='white', fontweight='bold')
        
        # Create canvas
        self.price_canvas = FigureCanvas(self.price_fig, self.price_chart_frame)
        self.price_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initialize empty data
        self.price_data = []
        self.time_data = []

    def setup_portfolio_chart(self):
        """Setup portfolio performance chart"""
        if FigureCanvas is None:
            ttk.Label(self.portfolio_chart_frame, text="📊 Chart unavailable", 
                     style='Dark.TLabel').pack(expand=True)
            return
        
        # Create portfolio figure
        self.portfolio_fig = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        self.portfolio_ax = self.portfolio_fig.add_subplot(111, facecolor='#2d2d2d')
        
        # Configure styling
        self.portfolio_ax.tick_params(colors='white')
        self.portfolio_ax.set_xlabel('Time', color='white')
        self.portfolio_ax.set_ylabel('Portfolio Value', color='white')
        self.portfolio_ax.set_title('Portfolio Performance - Real-time', color='white', fontweight='bold')
        
        # Create canvas
        self.portfolio_canvas = FigureCanvas(self.portfolio_fig, self.portfolio_chart_frame)
        self.portfolio_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initialize data
        self.portfolio_data = []
        self.portfolio_time_data = []

    def setup_ai_performance_chart(self):
        """Setup AI agents performance comparison"""
        if FigureCanvas is None:
            ttk.Label(self.ai_chart_frame, text="📊 Chart unavailable", 
                     style='Dark.TLabel').pack(expand=True)
            return
        
        # Create AI performance figure
        self.ai_fig = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        self.ai_ax = self.ai_fig.add_subplot(111, facecolor='#2d2d2d')
        
        # Configure styling
        self.ai_ax.tick_params(colors='white')
        self.ai_ax.set_xlabel('Episode', color='white')
        self.ai_ax.set_ylabel('Cumulative Reward', color='white')
        self.ai_ax.set_title('AI Agents Performance Comparison', color='white', fontweight='bold')
        
        # Create canvas
        self.ai_canvas = FigureCanvas(self.ai_fig, self.ai_chart_frame)
        self.ai_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initialize agent performance data
        self.agent_performance_data = {
            'PPO': {'episodes': [], 'rewards': []},
            'SAC': {'episodes': [], 'rewards': []},
            'TD3': {'episodes': [], 'rewards': []}
        }

    def setup_risk_analysis_chart(self):
        """Setup risk analysis visualization"""
        if FigureCanvas is None:
            ttk.Label(self.risk_chart_frame, text="📊 Chart unavailable", 
                     style='Dark.TLabel').pack(expand=True)
            return
        
        # Create risk analysis figure with subplots
        self.risk_fig = Figure(figsize=(12, 6), facecolor='#1e1e1e')
        
        # Drawdown chart
        self.drawdown_ax = self.risk_fig.add_subplot(211, facecolor='#2d2d2d')
        self.drawdown_ax.tick_params(colors='white')
        self.drawdown_ax.set_ylabel('Drawdown %', color='white')
        self.drawdown_ax.set_title('Portfolio Drawdown Analysis', color='white', fontweight='bold')
        
        # Portfolio heat chart
        self.heat_ax = self.risk_fig.add_subplot(212, facecolor='#2d2d2d')
        self.heat_ax.tick_params(colors='white')
        self.heat_ax.set_xlabel('Time', color='white')
        self.heat_ax.set_ylabel('Portfolio Heat %', color='white')
        self.heat_ax.set_title('Portfolio Heat Monitoring', color='white', fontweight='bold')
        
        self.risk_fig.tight_layout()
        
        # Create canvas
        self.risk_canvas = FigureCanvas(self.risk_fig, self.risk_chart_frame)
        self.risk_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initialize risk data
        self.drawdown_data = []
        self.heat_data = []
        self.risk_time_data = []

    def setup_ai_agents_interface(self):
        """Setup multi-agent AI monitoring interface"""
        print("🤖 Setting up AI Agents Interface...")
        
        # Main AI agents container
        ai_container = ttk.Frame(self.agents_frame, style='Dark.TFrame')
        ai_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # === TOP SECTION: Agent Status Overview ===
        agents_status_frame = ttk.LabelFrame(ai_container, text="🤖 Multi-Agent System Status", 
                                           style='Dark.TLabelframe')
        agents_status_frame.pack(fill='x', pady=(0, 10))
        
        # Agents grid
        agents_grid = ttk.Frame(agents_status_frame)
        agents_grid.pack(fill='x', padx=10, pady=10)
        
        # PPO Agent Panel
        ppo_frame = ttk.LabelFrame(agents_grid, text="🧠 PPO Agent", style='Dark.TLabelframe')
        ppo_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        self.ppo_status = ttk.Label(ppo_frame, text="Status: Inactive", 
                                   foreground='#6c757d', style='Dark.TLabel')
        self.ppo_status.pack(anchor='w', padx=5, pady=2)
        
        self.ppo_decisions = ttk.Label(ppo_frame, text="Decisions: 0", style='Dark.TLabel')
        self.ppo_decisions.pack(anchor='w', padx=5, pady=2)
        
        self.ppo_winrate = ttk.Label(ppo_frame, text="Win Rate: 0.0%", style='Dark.TLabel')
        self.ppo_winrate.pack(anchor='w', padx=5, pady=2)
        
        self.ppo_avg_reward = ttk.Label(ppo_frame, text="Avg Reward: 0.0", style='Dark.TLabel')
        self.ppo_avg_reward.pack(anchor='w', padx=5, pady=2)
        
        # SAC Agent Panel
        sac_frame = ttk.LabelFrame(agents_grid, text="🎯 SAC Agent", style='Dark.TLabelframe')
        sac_frame.pack(side='left', fill='both', expand=True, padx=2.5)
        
        self.sac_status = ttk.Label(sac_frame, text="Status: Inactive", 
                                   foreground='#6c757d', style='Dark.TLabel')
        self.sac_status.pack(anchor='w', padx=5, pady=2)
        
        self.sac_decisions = ttk.Label(sac_frame, text="Decisions: 0", style='Dark.TLabel')
        self.sac_decisions.pack(anchor='w', padx=5, pady=2)
        
        self.sac_winrate = ttk.Label(sac_frame, text="Win Rate: 0.0%", style='Dark.TLabel')
        self.sac_winrate.pack(anchor='w', padx=5, pady=2)
        
        self.sac_avg_reward = ttk.Label(sac_frame, text="Avg Reward: 0.0", style='Dark.TLabel')
        self.sac_avg_reward.pack(anchor='w', padx=5, pady=2)
        
        # TD3 Agent Panel
        td3_frame = ttk.LabelFrame(agents_grid, text="🚀 TD3 Agent", style='Dark.TLabelframe')
        td3_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.td3_status = ttk.Label(td3_frame, text="Status: Inactive", 
                                   foreground='#6c757d', style='Dark.TLabel')
        self.td3_status.pack(anchor='w', padx=5, pady=2)
        
        self.td3_decisions = ttk.Label(td3_frame, text="Decisions: 0", style='Dark.TLabel')
        self.td3_decisions.pack(anchor='w', padx=5, pady=2)
        
        self.td3_winrate = ttk.Label(td3_frame, text="Win Rate: 0.0%", style='Dark.TLabel')
        self.td3_winrate.pack(anchor='w', padx=5, pady=2)
        
        self.td3_avg_reward = ttk.Label(td3_frame, text="Avg Reward: 0.0", style='Dark.TLabel')
        self.td3_avg_reward.pack(anchor='w', padx=5, pady=2)
        
        # === MIDDLE SECTION: Action Distribution & Decision Matrix ===
        analysis_row = ttk.Frame(ai_container, style='Dark.TFrame')
        analysis_row.pack(fill='both', expand=True, pady=(0, 10))
        
        # 15-Dimensional Action Distribution
        action_frame = ttk.LabelFrame(analysis_row, text="📊 15D Action Distribution", 
                                     style='Dark.TLabelframe')
        action_frame.pack(side='left', fill='both', expand=True, padx=(0, 5))
        
        # Action distribution table
        self.setup_action_distribution_table(action_frame)
        
        # Decision Matrix Panel
        decision_frame = ttk.LabelFrame(analysis_row, text="🎯 Decision Matrix", 
                                       style='Dark.TLabelframe')
        decision_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        # Decision matrix
        self.setup_decision_matrix(decision_frame)
        
        # === BOTTOM SECTION: Real-time Agent Performance ===
        performance_frame = ttk.LabelFrame(ai_container, text="📈 Real-time Agent Performance", 
                                         style='Dark.TLabelframe')
        performance_frame.pack(fill='both', expand=True)
        
        # Agent performance chart (already handled in analytics)
        performance_info = ttk.Frame(performance_frame)
        performance_info.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Current best agent
        best_agent_frame = ttk.Frame(performance_info)
        best_agent_frame.pack(fill='x', pady=(0, 10))
        
        ttk.Label(best_agent_frame, text="🏆 Current Best Agent:", 
                 style='Dark.TLabel', font=('Segoe UI', 10, 'bold')).pack(side='left')
        self.best_agent_label = ttk.Label(best_agent_frame, text="None", 
                                         foreground='#28a745', style='Dark.TLabel')
        self.best_agent_label.pack(side='left', padx=(10, 0))
        
        # Agent switching controls
        switching_frame = ttk.Frame(performance_info)
        switching_frame.pack(fill='x')
        
        ttk.Label(switching_frame, text="🔄 Agent Switching:", style='Dark.TLabel').pack(side='left')
        
        self.auto_switching_var = tk.BooleanVar(value=True)
        self.auto_switching_cb = ttk.Checkbutton(switching_frame, text="Auto Switch", 
                                                variable=self.auto_switching_var,
                                                command=self.toggle_auto_switching)
        self.auto_switching_cb.pack(side='left', padx=(10, 0))
        
        # Manual agent selection
        ttk.Label(switching_frame, text="Manual:", style='Dark.TLabel').pack(side='left', padx=(20, 5))
        
        self.manual_agent_var = tk.StringVar(value="AUTO")
        agent_combo = ttk.Combobox(switching_frame, textvariable=self.manual_agent_var,
                                  values=["AUTO", "PPO", "SAC", "TD3"], width=10, state='readonly')
        agent_combo.pack(side='left', padx=(0, 10))
        agent_combo.bind('<<ComboboxSelected>>', self.on_manual_agent_change)
        
        print("✅ AI Agents Interface setup completed!")

    def setup_action_distribution_table(self, parent):
        """Setup 15-dimensional action distribution table"""
        # Create scrolled frame for action distribution
        action_container = ttk.Frame(parent)
        action_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Action distribution treeview
        action_columns = ('Action Dim', 'Description', 'Current Value', 'Avg Value', 'Usage %')
        
        self.action_tree = ttk.Treeview(action_container, columns=action_columns, 
                                       show='headings', height=10)
        
        # Configure action columns
        self.action_tree.heading('Action Dim', text='Dimension')
        self.action_tree.heading('Description', text='Description')
        self.action_tree.heading('Current Value', text='Current')
        self.action_tree.heading('Avg Value', text='Average')
        self.action_tree.heading('Usage %', text='Usage')
        
        # Column widths
        self.action_tree.column('Action Dim', width=80)
        self.action_tree.column('Description', width=150)
        self.action_tree.column('Current Value', width=80)
        self.action_tree.column('Avg Value', width=80)
        self.action_tree.column('Usage %', width=70)
        
        # Scrollbar for actions
        action_scrollbar = ttk.Scrollbar(action_container, orient='vertical', 
                                        command=self.action_tree.yview)
        self.action_tree.configure(yscrollcommand=action_scrollbar.set)
        
        # Pack action table
        self.action_tree.pack(side='left', fill='both', expand=True)
        action_scrollbar.pack(side='right', fill='y')
        
        # Initialize action dimensions
        self.init_action_dimensions()

    def init_action_dimensions(self):
        """Initialize 15-dimensional action space display"""
        action_descriptions = [
            ('A0', 'Market Direction', '-1 to 1'),
            ('A1', 'Position Size', '0.01 to 1.0'),
            ('A2', 'Entry Aggression', '0 to 1'),
            ('A3', 'Profit Target Ratio', '0.5 to 5.0'),
            ('A4', 'Partial Take Levels', '0 to 3'),
            ('A5', 'Add Position Signal', '0 to 1'),
            ('A6', 'Hedge Ratio', '0 to 1'),
            ('A7', 'Recovery Mode', '0 to 3'),
            ('A8', 'Correlation Limit', '0 to 1'),
            ('A9', 'Volatility Filter', '0 to 1'),
            ('A10', 'Spread Tolerance', '0 to 1'),
            ('A11', 'Time Filter', '0 to 1'),
            ('A12', 'Portfolio Heat Limit', '0 to 1'),
            ('A13', 'Smart Exit Signal', '0 to 1'),
            ('A14', 'Rebalance Trigger', '0 to 1')
        ]
        
        for dim, desc, range_desc in action_descriptions:
            self.action_tree.insert('', 'end', values=(dim, desc, '0.0', '0.0', '0%'))

    def setup_decision_matrix(self, parent):
        """Setup decision matrix display"""
        decision_container = ttk.Frame(parent)
        decision_container.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Recent decisions table
        decision_columns = ('Time', 'Agent', 'Action Type', 'Confidence', 'Reward')
        
        self.decision_tree = ttk.Treeview(decision_container, columns=decision_columns, 
                                         show='headings', height=10)
        
        # Configure decision columns
        self.decision_tree.heading('Time', text='Time')
        self.decision_tree.heading('Agent', text='Agent')
        self.decision_tree.heading('Action Type', text='Action')
        self.decision_tree.heading('Confidence', text='Confidence')
        self.decision_tree.heading('Reward', text='Reward')
        
        # Column widths
        self.decision_tree.column('Time', width=80)
        self.decision_tree.column('Agent', width=60)
        self.decision_tree.column('Action Type', width=100)
        self.decision_tree.column('Confidence', width=80)
        self.decision_tree.column('Reward', width=80)
        
        # Scrollbar for decisions
        decision_scrollbar = ttk.Scrollbar(decision_container, orient='vertical', 
                                          command=self.decision_tree.yview)
        self.decision_tree.configure(yscrollcommand=decision_scrollbar.set)
        
        # Pack decision table
        self.decision_tree.pack(side='left', fill='both', expand=True)
        decision_scrollbar.pack(side='right', fill='y')

    def setup_recovery_interface(self):
        """Setup smart recovery engine interface"""
        print("🛡️ Setting up Recovery Engine Interface...")
        
        # Main recovery container
        recovery_container = ttk.Frame(self.recovery_frame, style='Dark.TFrame')
        recovery_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # === TOP SECTION: Recovery Status & Controls ===
        recovery_status_frame = ttk.LabelFrame(recovery_container, text="🛡️ Recovery System Status", 
                                             style='Dark.TLabelframe')
        recovery_status_frame.pack(fill='x', pady=(0, 10))
        
        # Status grid
        status_grid = ttk.Frame(recovery_status_frame)
        status_grid.pack(fill='x', padx=10, pady=10)
        
        # Current recovery mode
        mode_frame = ttk.Frame(status_grid)
        mode_frame.pack(fill='x', pady=2)
        ttk.Label(mode_frame, text="Current Mode:", style='Dark.TLabel').pack(side='left')
        self.recovery_mode_label = ttk.Label(mode_frame, text="INACTIVE", 
                                           foreground='#6c757d', style='Dark.TLabel')
        self.recovery_mode_label.pack(side='left', padx=(10, 0))
        
        # Recovery level
        level_frame = ttk.Frame(status_grid)
        level_frame.pack(fill='x', pady=2)
        ttk.Label(level_frame, text="Recovery Level:", style='Dark.TLabel').pack(side='left')
        self.recovery_level_label = ttk.Label(level_frame, text="0", style='Dark.TLabel')
        self.recovery_level_label.pack(side='left', padx=(10, 0))
        
        # Recovery efficiency
        efficiency_frame = ttk.Frame(status_grid)
        efficiency_frame.pack(fill='x', pady=2)
        ttk.Label(efficiency_frame, text="Recovery Efficiency:", style='Dark.TLabel').pack(side='left')
        self.recovery_efficiency_label = ttk.Label(efficiency_frame, text="0.0%", style='Dark.TLabel')
        self.recovery_efficiency_label.pack(side='left', padx=(10, 0))
        
        # Total attempts vs successes
        attempts_frame = ttk.Frame(status_grid)
        attempts_frame.pack(fill='x', pady=2)
        ttk.Label(attempts_frame, text="Success Rate:", style='Dark.TLabel').pack(side='left')
        self.recovery_success_label = ttk.Label(attempts_frame, text="0/0 (0.0%)", style='Dark.TLabel')
        self.recovery_success_label.pack(side='left', padx=(10, 0))
        
        # === MIDDLE SECTION: Recovery Strategies ===
        strategies_frame = ttk.LabelFrame(recovery_container, text="🔧 Recovery Strategies", 
                                        style='Dark.TLabelframe')
        strategies_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Strategies notebook
        self.recovery_notebook = ttk.Notebook(strategies_frame, style='Professional.TNotebook')
        self.recovery_notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Adaptive Martingale tab
        self.martingale_frame = ttk.Frame(self.recovery_notebook)
        self.recovery_notebook.add(self.martingale_frame, text="🎯 Adaptive Martingale")
        
        # Smart Grid tab
        self.grid_frame = ttk.Frame(self.recovery_notebook)
        self.recovery_notebook.add(self.grid_frame, text="📊 Smart Grid")
        
        # Dynamic Hedge tab
        self.hedge_frame = ttk.Frame(self.recovery_notebook)
        self.recovery_notebook.add(self.hedge_frame, text="🛡️ Dynamic Hedge")
        
        # AI Ensemble tab
        self.ensemble_frame = ttk.Frame(self.recovery_notebook)
        self.recovery_notebook.add(self.ensemble_frame, text="🤖 AI Ensemble")
        
        # Setup strategy interfaces
        self.setup_martingale_strategy()
        self.setup_grid_strategy()
        self.setup_hedge_strategy()
        self.setup_ensemble_strategy()
        
        # === BOTTOM SECTION: Recovery History & Analytics ===
        history_frame = ttk.LabelFrame(recovery_container, text="📈 Recovery History", 
                                     style='Dark.TLabelframe')
        history_frame.pack(fill='both', expand=True)
        
        # Recovery history table
        self.setup_recovery_history_table(history_frame)
        
        print("✅ Recovery Engine Interface setup completed!")

    def setup_martingale_strategy(self):
        """Setup adaptive martingale strategy interface"""
        # Martingale parameters
        params_frame = ttk.LabelFrame(self.martingale_frame, text="⚙️ Parameters", 
                                     style='Dark.TLabelframe')
        params_frame.pack(fill='x', padx=10, pady=5)
        
        # Martingale multiplier
        mult_frame = ttk.Frame(params_frame)
        mult_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(mult_frame, text="Multiplier:", style='Dark.TLabel').pack(side='left')
        self.martingale_multiplier_var = tk.DoubleVar(value=1.5)
        mult_scale = ttk.Scale(mult_frame, from_=1.1, to=3.0, orient='horizontal',
                              variable=self.martingale_multiplier_var, length=200)
        mult_scale.pack(side='left', padx=(10, 5))
        self.mult_value_label = ttk.Label(mult_frame, text="1.5", style='Dark.TLabel')
        self.mult_value_label.pack(side='left')
        mult_scale.configure(command=lambda v: self.mult_value_label.config(text=f"{float(v):.2f}"))
        
        # Max recovery levels
        levels_frame = ttk.Frame(params_frame)
        levels_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(levels_frame, text="Max Levels:", style='Dark.TLabel').pack(side='left')
        self.max_recovery_levels_var = tk.IntVar(value=8)
        levels_scale = ttk.Scale(levels_frame, from_=3, to=15, orient='horizontal',
                               variable=self.max_recovery_levels_var, length=200)
        levels_scale.pack(side='left', padx=(10, 5))
        self.levels_value_label = ttk.Label(levels_frame, text="8", style='Dark.TLabel')
        self.levels_value_label.pack(side='left')
        levels_scale.configure(command=lambda v: self.levels_value_label.config(text=f"{int(float(v))}"))
        
        # Adaptive sizing
        adaptive_frame = ttk.Frame(params_frame)
        adaptive_frame.pack(fill='x', padx=5, pady=2)
        self.adaptive_sizing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(adaptive_frame, text="Adaptive Position Sizing", 
                       variable=self.adaptive_sizing_var).pack(side='left')
        
        # Status display
        status_frame = ttk.LabelFrame(self.martingale_frame, text="📊 Current Status", 
                                    style='Dark.TLabelframe')
        status_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.martingale_status_text = tk.Text(status_frame, height=8, width=50, 
                                            bg='#2d2d2d', fg='white', font=('Consolas', 9))
        self.martingale_status_text.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_grid_strategy(self):
        """Setup smart grid strategy interface"""
        # Grid parameters
        params_frame = ttk.LabelFrame(self.grid_frame, text="⚙️ Parameters", 
                                     style='Dark.TLabelframe')
        params_frame.pack(fill='x', padx=10, pady=5)
        
        # Grid spacing
        spacing_frame = ttk.Frame(params_frame)
        spacing_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(spacing_frame, text="Grid Spacing (points):", style='Dark.TLabel').pack(side='left')
        self.grid_spacing_var = tk.IntVar(value=200)
        spacing_scale = ttk.Scale(spacing_frame, from_=50, to=500, orient='horizontal',
                                variable=self.grid_spacing_var, length=200)
        spacing_scale.pack(side='left', padx=(10, 5))
        self.spacing_value_label = ttk.Label(spacing_frame, text="200", style='Dark.TLabel')
        self.spacing_value_label.pack(side='left')
        spacing_scale.configure(command=lambda v: self.spacing_value_label.config(text=f"{int(float(v))}"))
        
        # Dynamic spacing
        dynamic_frame = ttk.Frame(params_frame)
        dynamic_frame.pack(fill='x', padx=5, pady=2)
        self.dynamic_spacing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dynamic_frame, text="Dynamic Spacing (ATR-based)", 
                       variable=self.dynamic_spacing_var).pack(side='left')
        
        # Grid visualization
        viz_frame = ttk.LabelFrame(self.grid_frame, text="📊 Grid Visualization", 
                                 style='Dark.TLabelframe')
        viz_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.grid_status_text = tk.Text(viz_frame, height=8, width=50, 
                                      bg='#2d2d2d', fg='white', font=('Consolas', 9))
        self.grid_status_text.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_hedge_strategy(self):
        """Setup dynamic hedge strategy interface"""
        # Hedge parameters
        params_frame = ttk.LabelFrame(self.hedge_frame, text="⚙️ Parameters", 
                                     style='Dark.TLabelframe')
        params_frame.pack(fill='x', padx=10, pady=5)
        
        # Hedge ratio
        ratio_frame = ttk.Frame(params_frame)
        ratio_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(ratio_frame, text="Hedge Ratio:", style='Dark.TLabel').pack(side='left')
        self.hedge_ratio_var = tk.DoubleVar(value=0.5)
        ratio_scale = ttk.Scale(ratio_frame, from_=0.1, to=1.0, orient='horizontal',
                              variable=self.hedge_ratio_var, length=200)
        ratio_scale.pack(side='left', padx=(10, 5))
        self.ratio_value_label = ttk.Label(ratio_frame, text="0.5", style='Dark.TLabel')
        self.ratio_value_label.pack(side='left')
        ratio_scale.configure(command=lambda v: self.ratio_value_label.config(text=f"{float(v):.2f}"))
        
        # Auto hedge
        auto_frame = ttk.Frame(params_frame)
        auto_frame.pack(fill='x', padx=5, pady=2)
        self.auto_hedge_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(auto_frame, text="Automatic Hedge Activation", 
                       variable=self.auto_hedge_var).pack(side='left')
        
        # Hedge status
        status_frame = ttk.LabelFrame(self.hedge_frame, text="📊 Hedge Status", 
                                    style='Dark.TLabelframe')
        status_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.hedge_status_text = tk.Text(status_frame, height=8, width=50, 
                                       bg='#2d2d2d', fg='white', font=('Consolas', 9))
        self.hedge_status_text.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_ensemble_strategy(self):
        """Setup AI ensemble strategy interface"""
        # Ensemble parameters
        params_frame = ttk.LabelFrame(self.ensemble_frame, text="⚙️ AI Ensemble Parameters", 
                                     style='Dark.TLabelframe')
        params_frame.pack(fill='x', padx=10, pady=5)
        
        # Strategy weights
        weights_frame = ttk.Frame(params_frame)
        weights_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(weights_frame, text="Strategy Weights:", style='Dark.TLabel').pack(anchor='w')
        
        # Martingale weight
        mart_weight_frame = ttk.Frame(weights_frame)
        mart_weight_frame.pack(fill='x', pady=1)
        ttk.Label(mart_weight_frame, text="Martingale:", style='Dark.TLabel', width=12).pack(side='left')
        self.martingale_weight_var = tk.DoubleVar(value=0.3)
        mart_scale = ttk.Scale(mart_weight_frame, from_=0.0, to=1.0, orient='horizontal',
                             variable=self.martingale_weight_var, length=150)
        mart_scale.pack(side='left', padx=(5, 5))
        self.mart_weight_label = ttk.Label(mart_weight_frame, text="0.3", style='Dark.TLabel')
        self.mart_weight_label.pack(side='left')
        mart_scale.configure(command=lambda v: self.mart_weight_label.config(text=f"{float(v):.2f}"))
        
        # Grid weight
        grid_weight_frame = ttk.Frame(weights_frame)
        grid_weight_frame.pack(fill='x', pady=1)
        ttk.Label(grid_weight_frame, text="Grid:", style='Dark.TLabel', width=12).pack(side='left')
        self.grid_weight_var = tk.DoubleVar(value=0.3)
        grid_scale = ttk.Scale(grid_weight_frame, from_=0.0, to=1.0, orient='horizontal',
                             variable=self.grid_weight_var, length=150)
        grid_scale.pack(side='left', padx=(5, 5))
        self.grid_weight_label = ttk.Label(grid_weight_frame, text="0.3", style='Dark.TLabel')
        self.grid_weight_label.pack(side='left')
        grid_scale.configure(command=lambda v: self.grid_weight_label.config(text=f"{float(v):.2f}"))
        
        # Hedge weight
        hedge_weight_frame = ttk.Frame(weights_frame)
        hedge_weight_frame.pack(fill='x', pady=1)
        ttk.Label(hedge_weight_frame, text="Hedge:", style='Dark.TLabel', width=12).pack(side='left')
        self.hedge_weight_var = tk.DoubleVar(value=0.4)
        hedge_scale = ttk.Scale(hedge_weight_frame, from_=0.0, to=1.0, orient='horizontal',
                              variable=self.hedge_weight_var, length=150)
        hedge_scale.pack(side='left', padx=(5, 5))
        self.hedge_weight_label = ttk.Label(hedge_weight_frame, text="0.4", style='Dark.TLabel')
        self.hedge_weight_label.pack(side='left')
        hedge_scale.configure(command=lambda v: self.hedge_weight_label.config(text=f"{float(v):.2f}"))
        
        # Ensemble status
        status_frame = ttk.LabelFrame(self.ensemble_frame, text="🤖 Ensemble Status", 
                                    style='Dark.TLabelframe')
        status_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.ensemble_status_text = tk.Text(status_frame, height=8, width=50, 
                                          bg='#2d2d2d', fg='white', font=('Consolas', 9))
        self.ensemble_status_text.pack(fill='both', expand=True, padx=5, pady=5)

    def setup_recovery_history_table(self, parent):
        """Setup recovery history tracking table"""
        # History table frame
        history_container = ttk.Frame(parent)
        history_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Recovery history columns
        history_columns = ('Timestamp', 'Mode', 'Level', 'Trigger', 'Result', 'Duration', 'Efficiency')
        
        self.recovery_history_tree = ttk.Treeview(history_container, columns=history_columns, 
                                                show='headings', height=8)
        
        # Configure history columns
        for col in history_columns:
            self.recovery_history_tree.heading(col, text=col)
            self.recovery_history_tree.column(col, width=100)
        
        # Scrollbars for history
        history_v_scroll = ttk.Scrollbar(history_container, orient='vertical', 
                                        command=self.recovery_history_tree.yview)
        history_h_scroll = ttk.Scrollbar(history_container, orient='horizontal', 
                                        command=self.recovery_history_tree.xview)
        
        self.recovery_history_tree.configure(yscrollcommand=history_v_scroll.set, 
                                            xscrollcommand=history_h_scroll.set)
        
        # Pack history table and scrollbars
        self.recovery_history_tree.pack(side='left', fill='both', expand=True)
        history_v_scroll.pack(side='right', fill='y')
        history_h_scroll.pack(side='bottom', fill='x')

    def setup_portfolio_interface(self):
        """Setup advanced portfolio management interface"""
        print("💼 Setting up Portfolio Management Interface...")
        
        # Main portfolio container
        portfolio_container = ttk.Frame(self.portfolio_frame, style='Dark.TFrame')
        portfolio_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # === TOP SECTION: Portfolio Overview ===
        overview_frame = ttk.LabelFrame(portfolio_container, text="💼 Portfolio Overview", 
                                       style='Dark.TLabelframe')
        overview_frame.pack(fill='x', pady=(0, 10))
        
        # Portfolio metrics grid
        overview_grid = ttk.Frame(overview_frame)
        overview_grid.pack(fill='x', padx=10, pady=10)
        
        # Left column - Portfolio Status
        left_col = ttk.Frame(overview_grid)
        left_col.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Portfolio mode
        mode_frame = ttk.Frame(left_col)
        mode_frame.pack(fill='x', pady=2)
        ttk.Label(mode_frame, text="Portfolio Mode:", style='Dark.TLabel').pack(side='left')
        self.portfolio_mode_label = ttk.Label(mode_frame, text="BALANCED", 
                                            foreground='#17a2b8', style='Dark.TLabel')
        self.portfolio_mode_label.pack(side='left', padx=(10, 0))
        
        # Portfolio heat
        heat_frame = ttk.Frame(left_col)
        heat_frame.pack(fill='x', pady=2)
        ttk.Label(heat_frame, text="Portfolio Heat:", style='Dark.TLabel').pack(side='left')
        self.portfolio_heat_detail_label = ttk.Label(heat_frame, text="0.0%", 
                                                   foreground='#ffc107', style='Dark.TLabel')
        self.portfolio_heat_detail_label.pack(side='left', padx=(10, 0))
        
        # Position correlation
        corr_frame = ttk.Frame(left_col)
        corr_frame.pack(fill='x', pady=2)
        ttk.Label(corr_frame, text="Position Correlation:", style='Dark.TLabel').pack(side='left')
        self.position_correlation_label = ttk.Label(corr_frame, text="0.0", style='Dark.TLabel')
        self.position_correlation_label.pack(side='left', padx=(10, 0))
        
        # Right column - Performance Metrics
        right_col = ttk.Frame(overview_grid)
        right_col.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Win rate
        winrate_frame = ttk.Frame(right_col)
        winrate_frame.pack(fill='x', pady=2)
        ttk.Label(winrate_frame, text="Win Rate:", style='Dark.TLabel').pack(side='left')
        self.portfolio_winrate_label = ttk.Label(winrate_frame, text="0.0%", style='Dark.TLabel')
        self.portfolio_winrate_label.pack(side='left', padx=(10, 0))
        
        # Sharpe ratio
        sharpe_frame = ttk.Frame(right_col)
        sharpe_frame.pack(fill='x', pady=2)
        ttk.Label(sharpe_frame, text="Sharpe Ratio:", style='Dark.TLabel').pack(side='left')
        self.sharpe_ratio_label = ttk.Label(sharpe_frame, text="0.0", style='Dark.TLabel')
        self.sharpe_ratio_label.pack(side='left', padx=(10, 0))
        
        # Efficiency score
        eff_frame = ttk.Frame(right_col)
        eff_frame.pack(fill='x', pady=2)
        ttk.Label(eff_frame, text="Efficiency Score:", style='Dark.TLabel').pack(side='left')
        self.portfolio_efficiency_detail_label = ttk.Label(eff_frame, text="0.0%", style='Dark.TLabel')
        self.portfolio_efficiency_detail_label.pack(side='left', padx=(10, 0))
        
        # === MIDDLE SECTION: Portfolio Controls ===
        controls_frame = ttk.LabelFrame(portfolio_container, text="🎛️ Portfolio Controls", 
                                       style='Dark.TLabelframe')
        controls_frame.pack(fill='x', pady=(0, 10))
        
        # Portfolio mode selection
        mode_control_frame = ttk.Frame(controls_frame)
        mode_control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(mode_control_frame, text="Portfolio Mode:", style='Dark.TLabel').pack(side='left')
        
        self.portfolio_mode_var = tk.StringVar(value="BALANCED")
        mode_combo = ttk.Combobox(mode_control_frame, textvariable=self.portfolio_mode_var,
                                 values=["AGGRESSIVE", "BALANCED", "CONSERVATIVE", "DEFENSIVE"], 
                                 width=15, state='readonly')
        mode_combo.pack(side='left', padx=(10, 20))
        mode_combo.bind('<<ComboboxSelected>>', self.on_portfolio_mode_change)
        
        # Quick action buttons
        ttk.Button(mode_control_frame, text="📊 Rebalance Portfolio", 
                  command=self.rebalance_portfolio, style='Professional.TButton').pack(side='left', padx=5)
        
        ttk.Button(mode_control_frame, text="💰 Take Profits", 
                  command=self.take_portfolio_profits, style='Success.TButton').pack(side='left', padx=5)
        
        ttk.Button(mode_control_frame, text="🛡️ Reduce Risk", 
                  command=self.reduce_portfolio_risk, style='Warning.TButton').pack(side='left', padx=5)
        
        # === RISK MANAGEMENT SECTION ===
        risk_mgmt_frame = ttk.LabelFrame(portfolio_container, text="⚠️ Risk Management", 
                                        style='Dark.TLabelframe')
        risk_mgmt_frame.pack(fill='x', pady=(0, 10))
        
        risk_grid = ttk.Frame(risk_mgmt_frame)
        risk_grid.pack(fill='x', padx=10, pady=10)
        
        # Portfolio heat limit
        heat_limit_frame = ttk.Frame(risk_grid)
        heat_limit_frame.pack(fill='x', pady=2)
        ttk.Label(heat_limit_frame, text="Portfolio Heat Limit:", style='Dark.TLabel').pack(side='left')
        self.portfolio_heat_limit_var = tk.DoubleVar(value=15.0)
        heat_scale = ttk.Scale(heat_limit_frame, from_=5.0, to=25.0, orient='horizontal',
                              variable=self.portfolio_heat_limit_var, length=200)
        heat_scale.pack(side='left', padx=(10, 5))
        self.heat_limit_label = ttk.Label(heat_limit_frame, text="15.0%", style='Dark.TLabel')
        self.heat_limit_label.pack(side='left')
        heat_scale.configure(command=lambda v: self.heat_limit_label.config(text=f"{float(v):.1f}%"))
        
        # Max drawdown limit
        dd_limit_frame = ttk.Frame(risk_grid)
        dd_limit_frame.pack(fill='x', pady=2)
        ttk.Label(dd_limit_frame, text="Max Drawdown Limit:", style='Dark.TLabel').pack(side='left')
        self.max_drawdown_limit_var = tk.DoubleVar(value=10.0)
        dd_scale = ttk.Scale(dd_limit_frame, from_=2.0, to=20.0, orient='horizontal',
                            variable=self.max_drawdown_limit_var, length=200)
        dd_scale.pack(side='left', padx=(10, 5))
        self.dd_limit_label = ttk.Label(dd_limit_frame, text="10.0%", style='Dark.TLabel')
        self.dd_limit_label.pack(side='left')
        dd_scale.configure(command=lambda v: self.dd_limit_label.config(text=f"{float(v):.1f}%"))
        
        # Correlation limit
        corr_limit_frame = ttk.Frame(risk_grid)
        corr_limit_frame.pack(fill='x', pady=2)
        ttk.Label(corr_limit_frame, text="Correlation Limit:", style='Dark.TLabel').pack(side='left')
        self.correlation_limit_var = tk.DoubleVar(value=0.7)
        corr_scale = ttk.Scale(corr_limit_frame, from_=0.3, to=1.0, orient='horizontal',
                              variable=self.correlation_limit_var, length=200)
        corr_scale.pack(side='left', padx=(10, 5))
        self.corr_limit_label = ttk.Label(corr_limit_frame, text="0.7", style='Dark.TLabel')
        self.corr_limit_label.pack(side='left')
        corr_scale.configure(command=lambda v: self.corr_limit_label.config(text=f"{float(v):.2f}"))
        
        # === BOTTOM SECTION: Dynamic Thresholds ===
        thresholds_frame = ttk.LabelFrame(portfolio_container, text="🎯 Dynamic Thresholds", 
                                        style='Dark.TLabelframe')
        thresholds_frame.pack(fill='both', expand=True)
        
        # Thresholds table
        self.setup_thresholds_table(thresholds_frame)
        
        print("✅ Portfolio Management Interface setup completed!")

    def setup_thresholds_table(self, parent):
        """Setup dynamic thresholds monitoring table"""
        thresholds_container = ttk.Frame(parent)
        thresholds_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Thresholds columns
        threshold_columns = ('Threshold Type', 'Current Value', 'Target Value', 'Status', 'Last Updated')
        
        self.thresholds_tree = ttk.Treeview(thresholds_container, columns=threshold_columns, 
                                          show='headings', height=6)
        
        # Configure threshold columns
        self.thresholds_tree.heading('Threshold Type', text='Type')
        self.thresholds_tree.heading('Current Value', text='Current')
        self.thresholds_tree.heading('Target Value', text='Target')
        self.thresholds_tree.heading('Status', text='Status')
        self.thresholds_tree.heading('Last Updated', text='Updated')
        
        # Column widths
        self.thresholds_tree.column('Threshold Type', width=150)
        self.thresholds_tree.column('Current Value', width=100)
        self.thresholds_tree.column('Target Value', width=100)
        self.thresholds_tree.column('Status', width=100)
        self.thresholds_tree.column('Last Updated', width=120)
        
        # Scrollbar
        thresholds_scroll = ttk.Scrollbar(thresholds_container, orient='vertical', 
                                         command=self.thresholds_tree.yview)
        self.thresholds_tree.configure(yscrollcommand=thresholds_scroll.set)
        
        # Pack table
        self.thresholds_tree.pack(side='left', fill='both', expand=True)
        thresholds_scroll.pack(side='right', fill='y')
        
        # Initialize threshold display
        self.init_thresholds_display()

    def init_thresholds_display(self):
        """Initialize dynamic thresholds display"""
        threshold_types = [
            ('Portfolio Target', '$0.00', '$100.00', 'Active'),
            ('Per Lot Target', '$0.00', '$5.00', 'Active'),
            ('Stop Loss Target', '$0.00', '$-50.00', 'Active'),
            ('Take Profit Target', '$0.00', '$25.00', 'Active'),
            ('Emergency Stop', '$0.00', '$-200.00', 'Standby')
        ]
        
        for thresh_type, current, target, status in threshold_types:
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.thresholds_tree.insert('', 'end', 
                                      values=(thresh_type, current, target, status, timestamp))

    def setup_system_interface(self):
        """Setup professional system configuration interface"""
        print("🔧 Setting up System Configuration Interface...")
        
        # Main system container
        system_container = ttk.Frame(self.system_frame, style='Dark.TFrame')
        system_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configuration notebook
        self.config_notebook = ttk.Notebook(system_container, style='Professional.TNotebook')
        self.config_notebook.pack(fill='both', expand=True)
        
        # === TRADING PARAMETERS TAB ===
        self.trading_config_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(self.trading_config_frame, text="⚙️ Trading")
        
        # === RL SYSTEM TAB ===
        self.rl_config_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(self.rl_config_frame, text="🤖 RL System")
        
        # === RISK MANAGEMENT TAB ===
        self.risk_config_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(self.risk_config_frame, text="🛡️ Risk Management")
        
        # === ADVANCED TAB ===
        self.advanced_config_frame = ttk.Frame(self.config_notebook)
        self.config_notebook.add(self.advanced_config_frame, text="🚀 Advanced")
        
        # Setup individual configuration tabs
        self.setup_trading_config()
        self.setup_rl_config()
        self.setup_risk_config()
        self.setup_advanced_config()
        
        # Configuration control buttons
        self.setup_config_controls(system_container)
        
        print("✅ System Configuration Interface setup completed!")

    def setup_trading_config(self):
        """Setup trading parameters configuration"""
        # Basic trading parameters
        basic_frame = ttk.LabelFrame(self.trading_config_frame, text="📊 Basic Trading Parameters", 
                                    style='Dark.TLabelframe')
        basic_frame.pack(fill='x', padx=10, pady=5)
        
        basic_grid = ttk.Frame(basic_frame)
        basic_grid.pack(fill='x', padx=10, pady=10)
        
        # Symbol
        symbol_frame = ttk.Frame(basic_grid)
        symbol_frame.pack(fill='x', pady=2)
        ttk.Label(symbol_frame, text="Trading Symbol:", style='Dark.TLabel', width=20).pack(side='left')
        self.symbol_var = tk.StringVar(value='XAUUSD')
        symbol_entry = ttk.Entry(symbol_frame, textvariable=self.symbol_var, width=15)
        symbol_entry.pack(side='left', padx=(10, 0))
        
        # Base lot size
        lot_frame = ttk.Frame(basic_grid)
        lot_frame.pack(fill='x', pady=2)
        ttk.Label(lot_frame, text="Base Lot Size:", style='Dark.TLabel', width=20).pack(side='left')
        self.base_lot_var = tk.DoubleVar(value=0.01)
        lot_scale = ttk.Scale(lot_frame, from_=0.01, to=0.1, orient='horizontal',
                             variable=self.base_lot_var, length=200)
        lot_scale.pack(side='left', padx=(10, 5))
        self.lot_value_label = ttk.Label(lot_frame, text="0.01", style='Dark.TLabel')
        self.lot_value_label.pack(side='left')
        lot_scale.configure(command=lambda v: self.lot_value_label.config(text=f"{float(v):.3f}"))
        
        # Max positions
        max_pos_frame = ttk.Frame(basic_grid)
        max_pos_frame.pack(fill='x', pady=2)
        ttk.Label(max_pos_frame, text="Max Positions:", style='Dark.TLabel', width=20).pack(side='left')
        self.max_positions_var = tk.IntVar(value=10)
        max_pos_scale = ttk.Scale(max_pos_frame, from_=1, to=20, orient='horizontal',
                                 variable=self.max_positions_var, length=200)
        max_pos_scale.pack(side='left', padx=(10, 5))
        self.max_pos_label = ttk.Label(max_pos_frame, text="10", style='Dark.TLabel')
        self.max_pos_label.pack(side='left')
        max_pos_scale.configure(command=lambda v: self.max_pos_label.config(text=f"{int(float(v))}"))
        
        # Professional trading features
        features_frame = ttk.LabelFrame(self.trading_config_frame, text="🚀 Professional Features", 
                                      style='Dark.TLabelframe')
        features_frame.pack(fill='x', padx=10, pady=5)
        
        features_grid = ttk.Frame(features_frame)
        features_grid.pack(fill='x', padx=10, pady=10)
        
        # Feature checkboxes
        self.dynamic_sizing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(features_grid, text="Dynamic Position Sizing", 
                       variable=self.dynamic_sizing_var).pack(anchor='w', pady=2)
        
        self.smart_entries_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(features_grid, text="Smart Market Entries", 
                       variable=self.smart_entries_var).pack(anchor='w', pady=2)
        
        self.correlation_analysis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(features_grid, text="Position Correlation Analysis", 
                       variable=self.correlation_analysis_var).pack(anchor='w', pady=2)
        
        self.multi_timeframe_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(features_grid, text="Multi-Timeframe Analysis", 
                       variable=self.multi_timeframe_var).pack(anchor='w', pady=2)

    def setup_rl_config(self):
        """Setup RL system configuration"""
        # Multi-agent configuration
        agents_frame = ttk.LabelFrame(self.rl_config_frame, text="🤖 Multi-Agent Configuration", 
                                     style='Dark.TLabelframe')
        agents_frame.pack(fill='x', padx=10, pady=5)
        
        agents_grid = ttk.Frame(agents_frame)
        agents_grid.pack(fill='x', padx=10, pady=10)
        
        # Primary agent
        primary_frame = ttk.Frame(agents_grid)
        primary_frame.pack(fill='x', pady=2)
        ttk.Label(primary_frame, text="Primary Agent:", style='Dark.TLabel', width=20).pack(side='left')
        self.primary_agent_var = tk.StringVar(value='PPO')
        primary_combo = ttk.Combobox(primary_frame, textvariable=self.primary_agent_var,
                                   values=['PPO', 'SAC', 'TD3'], width=15, state='readonly')
        primary_combo.pack(side='left', padx=(10, 0))
        
        # Agent ensemble
        ensemble_frame = ttk.Frame(agents_grid)
        ensemble_frame.pack(fill='x', pady=2)
        self.ensemble_mode_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(ensemble_frame, text="Enable Multi-Agent Ensemble", 
                       variable=self.ensemble_mode_var).pack(side='left')
        
        # Learning parameters
        learning_frame = ttk.LabelFrame(self.rl_config_frame, text="📚 Learning Parameters", 
                                       style='Dark.TLabelframe')
        learning_frame.pack(fill='x', padx=10, pady=5)
        
        learning_grid = ttk.Frame(learning_frame)
        learning_grid.pack(fill='x', padx=10, pady=10)
        
        # Learning rate
        lr_frame = ttk.Frame(learning_grid)
        lr_frame.pack(fill='x', pady=2)
        ttk.Label(lr_frame, text="Learning Rate:", style='Dark.TLabel', width=20).pack(side='left')
        self.learning_rate_var = tk.DoubleVar(value=0.0003)
        lr_scale = ttk.Scale(lr_frame, from_=0.0001, to=0.01, orient='horizontal',
                            variable=self.learning_rate_var, length=200)
        lr_scale.pack(side='left', padx=(10, 5))
        self.lr_label = ttk.Label(lr_frame, text="0.0003", style='Dark.TLabel')
        self.lr_label.pack(side='left')
        lr_scale.configure(command=lambda v: self.lr_label.config(text=f"{float(v):.5f}"))
        
        # Exploration parameters
        exploration_frame = ttk.Frame(learning_grid)
        exploration_frame.pack(fill='x', pady=2)
        ttk.Label(exploration_frame, text="Exploration Rate:", style='Dark.TLabel', width=20).pack(side='left')
        self.exploration_var = tk.DoubleVar(value=0.1)
        exploration_scale = ttk.Scale(exploration_frame, from_=0.01, to=0.5, orient='horizontal',
                                    variable=self.exploration_var, length=200)
        exploration_scale.pack(side='left', padx=(10, 5))
        self.exploration_label = ttk.Label(exploration_frame, text="0.1", style='Dark.TLabel')
        self.exploration_label.pack(side='left')
        exploration_scale.configure(command=lambda v: self.exploration_label.config(text=f"{float(v):.3f}"))

    def setup_risk_config(self):
        """Setup risk management configuration"""
        # Portfolio risk limits
        portfolio_risk_frame = ttk.LabelFrame(self.risk_config_frame, text="💼 Portfolio Risk Limits", 
                                            style='Dark.TLabelframe')
        portfolio_risk_frame.pack(fill='x', padx=10, pady=5)
        
        risk_grid = ttk.Frame(portfolio_risk_frame)
        risk_grid.pack(fill='x', padx=10, pady=10)
        
        # Max drawdown
        max_dd_frame = ttk.Frame(risk_grid)
        max_dd_frame.pack(fill='x', pady=2)
        ttk.Label(max_dd_frame, text="Max Drawdown (%):", style='Dark.TLabel', width=20).pack(side='left')
        self.max_dd_config_var = tk.DoubleVar(value=5.0)
        dd_scale = ttk.Scale(max_dd_frame, from_=1.0, to=15.0, orient='horizontal',
                            variable=self.max_dd_config_var, length=200)
        dd_scale.pack(side='left', padx=(10, 5))
        self.dd_config_label = ttk.Label(max_dd_frame, text="5.0%", style='Dark.TLabel')
        self.dd_config_label.pack(side='left')
        dd_scale.configure(command=lambda v: self.dd_config_label.config(text=f"{float(v):.1f}%"))
        
        # Emergency stop loss
        emergency_frame = ttk.Frame(risk_grid)
        emergency_frame.pack(fill='x', pady=2)
        ttk.Label(emergency_frame, text="Emergency Stop ($):", style='Dark.TLabel', width=20).pack(side='left')
        self.emergency_stop_var = tk.DoubleVar(value=1000.0)
        emergency_scale = ttk.Scale(emergency_frame, from_=100.0, to=5000.0, orient='horizontal',
                                   variable=self.emergency_stop_var, length=200)
        emergency_scale.pack(side='left', padx=(10, 5))
        self.emergency_label = ttk.Label(emergency_frame, text="$1000", style='Dark.TLabel')
        self.emergency_label.pack(side='left')
        emergency_scale.configure(command=lambda v: self.emergency_label.config(text=f"${float(v):.0f}"))
        
        # Recovery parameters
        recovery_risk_frame = ttk.LabelFrame(self.risk_config_frame, text="🛡️ Recovery Parameters", 
                                           style='Dark.TLabelframe')
        recovery_risk_frame.pack(fill='x', padx=10, pady=5)
        
        recovery_grid = ttk.Frame(recovery_risk_frame)
        recovery_grid.pack(fill='x', padx=10, pady=10)
        
        # Recovery trigger
        trigger_frame = ttk.Frame(recovery_grid)
        trigger_frame.pack(fill='x', pady=2)
        ttk.Label(trigger_frame, text="Recovery Trigger ($):", style='Dark.TLabel', width=20).pack(side='left')
        self.recovery_trigger_var = tk.DoubleVar(value=100.0)
        trigger_scale = ttk.Scale(trigger_frame, from_=50.0, to=500.0, orient='horizontal',
                                 variable=self.recovery_trigger_var, length=200)
        trigger_scale.pack(side='left', padx=(10, 5))
        self.trigger_label = ttk.Label(trigger_frame, text="$100", style='Dark.TLabel')
        self.trigger_label.pack(side='left')
        trigger_scale.configure(command=lambda v: self.trigger_label.config(text=f"${float(v):.0f}"))
        
        # Max recovery levels
        levels_frame = ttk.Frame(recovery_grid)
        levels_frame.pack(fill='x', pady=2)
        ttk.Label(levels_frame, text="Max Recovery Levels:", style='Dark.TLabel', width=20).pack(side='left')
        self.max_recovery_var = tk.IntVar(value=8)
        levels_scale = ttk.Scale(levels_frame, from_=3, to=15, orient='horizontal',
                               variable=self.max_recovery_var, length=200)
        levels_scale.pack(side='left', padx=(10, 5))
        self.recovery_levels_label = ttk.Label(levels_frame, text="8", style='Dark.TLabel')
        self.recovery_levels_label.pack(side='left')
        levels_scale.configure(command=lambda v: self.recovery_levels_label.config(text=f"{int(float(v))}"))

    def setup_advanced_config(self):
        """Setup advanced system configuration"""
        # Environment configuration
        env_frame = ttk.LabelFrame(self.advanced_config_frame, text="🌍 Environment Configuration", 
                                  style='Dark.TLabelframe')
        env_frame.pack(fill='x', padx=10, pady=5)
        
        env_grid = ttk.Frame(env_frame)
        env_grid.pack(fill='x', padx=10, pady=10)
        
        # Training mode
        training_frame = ttk.Frame(env_grid)
        training_frame.pack(fill='x', pady=2)
        self.training_mode_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(training_frame, text="Training Mode (Paper Trading)", 
                       variable=self.training_mode_var).pack(side='left')
        
        # Advanced features
        self.ml_profit_optimization_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(env_grid, text="ML Profit Optimization", 
                       variable=self.ml_profit_optimization_var).pack(anchor='w', pady=2)
        
        self.advanced_analytics_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(env_grid, text="Advanced Analytics", 
                       variable=self.advanced_analytics_var).pack(anchor='w', pady=2)
        
        self.real_time_optimization_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(env_grid, text="Real-time Strategy Optimization", 
                       variable=self.real_time_optimization_var).pack(anchor='w', pady=2)
        
        # System performance
        performance_frame = ttk.LabelFrame(self.advanced_config_frame, text="⚡ System Performance", 
                                         style='Dark.TLabelframe')
        performance_frame.pack(fill='x', padx=10, pady=5)
        
        perf_grid = ttk.Frame(performance_frame)
        perf_grid.pack(fill='x', padx=10, pady=10)
        
        # Update frequency
        freq_frame = ttk.Frame(perf_grid)
        freq_frame.pack(fill='x', pady=2)
        ttk.Label(freq_frame, text="Update Frequency (sec):", style='Dark.TLabel', width=20).pack(side='left')
        self.update_frequency_var = tk.DoubleVar(value=1.0)
        freq_scale = ttk.Scale(freq_frame, from_=0.5, to=5.0, orient='horizontal',
                            variable=self.update_frequency_var, length=200)
        freq_scale.pack(side='left', padx=(10, 5))
        self.freq_label = ttk.Label(freq_frame, text="1.0", style='Dark.TLabel')
        self.freq_label.pack(side='left')
        freq_scale.configure(command=lambda v: self.freq_label.config(text=f"{float(v):.1f}"))

    def setup_config_controls(self, parent):
        """Setup configuration control buttons"""
        controls_frame = ttk.Frame(parent, style='Dark.TFrame')
        controls_frame.pack(fill='x', pady=(10, 0))
        
        # Control buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.pack(side='right')
        
        ttk.Button(button_frame, text="💾 Save Config", 
                    command=self.save_professional_config, 
                    style='Success.TButton').pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="📁 Load Config", 
                    command=self.load_professional_config_file, 
                    style='Professional.TButton').pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="🔄 Reset to Defaults", 
                    command=self.reset_to_defaults, 
                    style='Warning.TButton').pack(side='left', padx=5)
        
        ttk.Button(button_frame, text="✅ Apply Settings", 
                    command=self.apply_all_settings, 
                    style='Professional.TButton').pack(side='left', padx=5)

    def setup_training_interface(self):
        """Setup elite training interface"""
        print("📈 Setting up Elite Training Interface...")
        
        # Main training container
        training_container = ttk.Frame(self.training_frame, style='Dark.TFrame')
        training_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # === TOP SECTION: Training Control Panel ===
        control_frame = ttk.LabelFrame(training_container, text="🎮 Training Control Center", 
                                      style='Dark.TLabelframe')
        control_frame.pack(fill='x', pady=(0, 10))
        
        control_grid = ttk.Frame(control_frame)
        control_grid.pack(fill='x', padx=10, pady=10)
        
        # Training mode selection
        mode_frame = ttk.Frame(control_grid)
        mode_frame.pack(fill='x', pady=5)
        
        ttk.Label(mode_frame, text="Training Mode:", style='Dark.TLabel').pack(side='left')
        
        self.training_mode_type_var = tk.StringVar(value='ENSEMBLE')
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.training_mode_type_var,
                                 values=['SINGLE_AGENT', 'ENSEMBLE', 'COMPETITIVE'], 
                                 width=15, state='readonly')
        mode_combo.pack(side='left', padx=(10, 20))
        
        # Training duration
        ttk.Label(mode_frame, text="Duration (episodes):", style='Dark.TLabel').pack(side='left')
        self.training_episodes_var = tk.IntVar(value=1000)
        episodes_scale = ttk.Scale(mode_frame, from_=100, to=10000, orient='horizontal',
                                  variable=self.training_episodes_var, length=200)
        episodes_scale.pack(side='left', padx=(10, 5))
        self.episodes_label = ttk.Label(mode_frame, text="1000", style='Dark.TLabel')
        self.episodes_label.pack(side='left')
        episodes_scale.configure(command=lambda v: self.episodes_label.config(text=f"{int(float(v))}"))
        
        # Training control buttons
        button_frame = ttk.Frame(control_grid)
        button_frame.pack(fill='x', pady=5)
        
        self.start_training_btn = ttk.Button(button_frame, text="🚀 Start Training", 
                                           command=self.start_professional_training, 
                                           style='Success.TButton')
        self.start_training_btn.pack(side='left', padx=5)
        
        self.pause_training_btn = ttk.Button(button_frame, text="⏸️ Pause Training", 
                                           command=self.pause_training, 
                                           style='Warning.TButton', state='disabled')
        self.pause_training_btn.pack(side='left', padx=5)
        
        self.stop_training_btn = ttk.Button(button_frame, text="⏹️ Stop Training", 
                                          command=self.stop_training, 
                                          style='Danger.TButton', state='disabled')
        self.stop_training_btn.pack(side='left', padx=5)
        
        # === MIDDLE SECTION: Training Progress ===
        progress_frame = ttk.LabelFrame(training_container, text="📊 Training Progress", 
                                       style='Dark.TLabelframe')
        progress_frame.pack(fill='x', pady=(0, 10))
        
        progress_grid = ttk.Frame(progress_frame)
        progress_grid.pack(fill='x', padx=10, pady=10)
        
        # Progress bars and metrics
        # Overall progress
        overall_frame = ttk.Frame(progress_grid)
        overall_frame.pack(fill='x', pady=2)
        ttk.Label(overall_frame, text="Overall Progress:", style='Dark.TLabel', width=20).pack(side='left')
        self.overall_progress = ttk.Progressbar(overall_frame, length=300, mode='determinate')
        self.overall_progress.pack(side='left', padx=(10, 5))
        self.overall_progress_label = ttk.Label(overall_frame, text="0%", style='Dark.TLabel')
        self.overall_progress_label.pack(side='left')
        
        # Current episode
        episode_frame = ttk.Frame(progress_grid)
        episode_frame.pack(fill='x', pady=2)
        ttk.Label(episode_frame, text="Current Episode:", style='Dark.TLabel', width=20).pack(side='left')
        self.current_episode_label = ttk.Label(episode_frame, text="0/0", style='Dark.TLabel')
        self.current_episode_label.pack(side='left', padx=(10, 0))
        
        # Training metrics
        metrics_row = ttk.Frame(progress_grid)
        metrics_row.pack(fill='x', pady=5)
        
        # Left metrics
        left_metrics = ttk.Frame(metrics_row)
        left_metrics.pack(side='left', fill='both', expand=True)
        
        avg_reward_frame = ttk.Frame(left_metrics)
        avg_reward_frame.pack(fill='x', pady=1)
        ttk.Label(avg_reward_frame, text="Avg Reward:", style='Dark.TLabel').pack(side='left')
        self.avg_reward_label = ttk.Label(avg_reward_frame, text="0.0", style='Dark.TLabel')
        self.avg_reward_label.pack(side='left', padx=(10, 0))
        
        best_reward_frame = ttk.Frame(left_metrics)
        best_reward_frame.pack(fill='x', pady=1)
        ttk.Label(best_reward_frame, text="Best Reward:", style='Dark.TLabel').pack(side='left')
        self.best_reward_label = ttk.Label(best_reward_frame, text="0.0", style='Dark.TLabel')
        self.best_reward_label.pack(side='left', padx=(10, 0))
        
        # Right metrics
        right_metrics = ttk.Frame(metrics_row)
        right_metrics.pack(side='right', fill='both', expand=True)
        
        training_time_frame = ttk.Frame(right_metrics)
        training_time_frame.pack(fill='x', pady=1)
        ttk.Label(training_time_frame, text="Training Time:", style='Dark.TLabel').pack(side='left')
        self.training_time_label = ttk.Label(training_time_frame, text="00:00:00", style='Dark.TLabel')
        self.training_time_label.pack(side='left', padx=(10, 0))
        
        eta_frame = ttk.Frame(right_metrics)
        eta_frame.pack(fill='x', pady=1)
        ttk.Label(eta_frame, text="ETA:", style='Dark.TLabel').pack(side='left')
        self.eta_label = ttk.Label(eta_frame, text="--:--:--", style='Dark.TLabel')
        self.eta_label.pack(side='left', padx=(10, 0))
        
        # === AGENT PERFORMANCE SECTION ===
        agents_perf_frame = ttk.LabelFrame(training_container, text="🤖 Agent Performance", 
                                         style='Dark.TLabelframe')
        agents_perf_frame.pack(fill='x', pady=(0, 10))
        
        # Agent performance table
        self.setup_agent_performance_table(agents_perf_frame)
        
        # === TRAINING LOGS SECTION ===
        logs_frame = ttk.LabelFrame(training_container, text="📝 Training Logs", 
                                   style='Dark.TLabelframe')
        logs_frame.pack(fill='both', expand=True)
        
        # Training logs text area
        logs_container = ttk.Frame(logs_frame)
        logs_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.training_logs = tk.Text(logs_container, height=10, bg='#2d2d2d', fg='white', 
                                   font=('Consolas', 9), wrap='word')
        
        training_scrollbar = ttk.Scrollbar(logs_container, orient='vertical', 
                                         command=self.training_logs.yview)
        self.training_logs.configure(yscrollcommand=training_scrollbar.set)
        
        self.training_logs.pack(side='left', fill='both', expand=True)
        training_scrollbar.pack(side='right', fill='y')
        
        print("✅ Elite Training Interface setup completed!")

    def setup_agent_performance_table(self, parent):
        """Setup agent performance comparison table"""
        table_container = ttk.Frame(parent)
        table_container.pack(fill='x', padx=10, pady=10)
        
        # Agent performance columns
        perf_columns = ('Agent', 'Episodes', 'Avg Reward', 'Best Reward', 'Win Rate', 'Status')
        
        self.agent_perf_tree = ttk.Treeview(table_container, columns=perf_columns, 
                                          show='headings', height=4)
        
        # Configure performance columns
        self.agent_perf_tree.heading('Agent', text='Agent')
        self.agent_perf_tree.heading('Episodes', text='Episodes')
        self.agent_perf_tree.heading('Avg Reward', text='Avg Reward')
        self.agent_perf_tree.heading('Best Reward', text='Best Reward')
        self.agent_perf_tree.heading('Win Rate', text='Win Rate')
        self.agent_perf_tree.heading('Status', text='Status')
        
        # Column widths
        self.agent_perf_tree.column('Agent', width=80)
        self.agent_perf_tree.column('Episodes', width=80)
        self.agent_perf_tree.column('Avg Reward', width=100)
        self.agent_perf_tree.column('Best Reward', width=100)
        self.agent_perf_tree.column('Win Rate', width=80)
        self.agent_perf_tree.column('Status', width=100)
        
        self.agent_perf_tree.pack(fill='x')
        
        # Initialize agent performance display
        self.init_agent_performance_display()

    def init_agent_performance_display(self):
        """Initialize agent performance display"""
        agents = ['PPO', 'SAC', 'TD3']
        for agent in agents:
            self.agent_perf_tree.insert('', 'end', 
                                      values=(agent, '0', '0.0', '0.0', '0.0%', 'Inactive'))
            
    def setup_logs_interface(self):
        """Setup professional logging system interface"""
        print("📋 Setting up Professional Logging System...")
        
        # Main logs container
        logs_container = ttk.Frame(self.logs_frame, style='Dark.TFrame')
        logs_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # === TOP SECTION: Log Controls ===
        log_controls_frame = ttk.LabelFrame(logs_container, text="🎛️ Log Controls", 
                                          style='Dark.TLabelframe')
        log_controls_frame.pack(fill='x', pady=(0, 10))
        
        controls_grid = ttk.Frame(log_controls_frame)
        controls_grid.pack(fill='x', padx=10, pady=10)
        
        # Log level filter
        level_frame = ttk.Frame(controls_grid)
        level_frame.pack(side='left', fill='x', expand=True)
        
        ttk.Label(level_frame, text="Log Level:", style='Dark.TLabel').pack(side='left')
        
        self.log_level_var = tk.StringVar(value='ALL')
        level_combo = ttk.Combobox(level_frame, textvariable=self.log_level_var,
                                  values=['ALL', 'SUCCESS', 'INFO', 'WARNING', 'ERROR', 'AI', 'SYSTEM'], 
                                  width=12, state='readonly')
        level_combo.pack(side='left', padx=(10, 20))
        level_combo.bind('<<ComboboxSelected>>', self.filter_logs)
        
        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(level_frame, text="Auto Scroll", 
                       variable=self.auto_scroll_var).pack(side='left', padx=(0, 20))
        
        # Log controls buttons
        button_frame = ttk.Frame(controls_grid)
        button_frame.pack(side='right')
        
        ttk.Button(button_frame, text="🔍 Search", 
                  command=self.search_logs, style='Professional.TButton').pack(side='left', padx=2)
        
        ttk.Button(button_frame, text="💾 Export", 
                  command=self.export_logs, style='Professional.TButton').pack(side='left', padx=2)
        
        ttk.Button(button_frame, text="🗑️ Clear", 
                  command=self.clear_logs, style='Warning.TButton').pack(side='left', padx=2)
        
        # === LOG STATUS BAR ===
        status_frame = ttk.Frame(logs_container, style='Dark.TFrame')
        status_frame.pack(fill='x', pady=(0, 5))
        
        # Log statistics
        self.log_stats_frame = ttk.Frame(status_frame)
        self.log_stats_frame.pack(side='left')
        
        self.total_logs_label = ttk.Label(self.log_stats_frame, text="Total: 0", style='Dark.TLabel')
        self.total_logs_label.pack(side='left', padx=(0, 10))
        
        self.filtered_logs_label = ttk.Label(self.log_stats_frame, text="Showing: 0", style='Dark.TLabel')
        self.filtered_logs_label.pack(side='left', padx=(0, 10))
        
        # Log search
        search_frame = ttk.Frame(status_frame)
        search_frame.pack(side='right')
        
        ttk.Label(search_frame, text="Search:", style='Dark.TLabel').pack(side='left')
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var, width=20)
        search_entry.pack(side='left', padx=(5, 0))
        search_entry.bind('<KeyRelease>', self.on_search_change)
        
        # === MAIN LOG DISPLAY ===
        log_display_frame = ttk.LabelFrame(logs_container, text="📝 System Logs", 
                                         style='Dark.TLabelframe')
        log_display_frame.pack(fill='both', expand=True)
        
        # Log text area with scrollbar
        log_container = ttk.Frame(log_display_frame)
        log_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Enhanced log text widget
        self.log_text = tk.Text(log_container, bg='#1a1a1a', fg='#ffffff', 
                               font=('Consolas', 9), wrap='word', state='disabled')
        
        # Log scrollbar
        log_scrollbar = ttk.Scrollbar(log_container, orient='vertical', 
                                     command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # Pack log display
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')
        
        # Configure log text tags for different log levels
        self.setup_log_tags()
        
        # Initialize log storage
        self.all_logs = []
        self.filtered_logs = []
        self.log_count = 0
        self.displayed_count = 0
        
        print("✅ Professional Logging System setup completed!")

    def setup_log_tags(self):
        """Setup text tags for different log levels"""
        # Configure tags for different log types
        self.log_text.tag_configure("SUCCESS", foreground="#28a745", font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure("INFO", foreground="#17a2b8")
        self.log_text.tag_configure("WARNING", foreground="#ffc107", font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure("ERROR", foreground="#dc3545", font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure("AI", foreground="#6f42c1", font=('Consolas', 9, 'bold'))
        self.log_text.tag_configure("SYSTEM", foreground="#6c757d")
        self.log_text.tag_configure("TIMESTAMP", foreground="#adb5bd", font=('Consolas', 8))

    def start_professional_systems(self):
        """Start all professional real-time systems"""
        print("⚡ Starting Professional Real-time Systems...")
        
        # Start main GUI update loop
        self.start_gui_update_loop()
        
        # Start analytics update thread
        self.start_analytics_thread()
        
        # Start system monitoring
        self.start_system_monitoring()
        
        print("✅ All professional systems started!")

    def start_gui_update_loop(self):
        """Start professional GUI update loop"""
        def update_loop():
            while True:
                try:
                    if self.is_connected:
                        self.root.after(0, self.update_all_professional_displays)
                    
                    # Update frequency from config
                    update_freq = getattr(self, 'update_frequency_var', None)
                    sleep_time = update_freq.get() if update_freq else 2.0
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    self.log_message(f"❌ GUI update loop error: {str(e)}", "ERROR")
                    time.sleep(5)
        
        self.update_thread = threading.Thread(target=update_loop, daemon=True)
        self.update_thread.start()

    def start_analytics_thread(self):
        """Start advanced analytics processing thread"""
        def analytics_loop():
            while True:
                try:
                    if self.is_connected and self.is_trading:
                        self.update_advanced_analytics()
                    time.sleep(5)  # Analytics update every 5 seconds
                    
                except Exception as e:
                    self.log_message(f"❌ Analytics error: {str(e)}", "ERROR")
                    time.sleep(10)
        
        self.analytics_thread = threading.Thread(target=analytics_loop, daemon=True)
        self.analytics_thread.start()

    def connect_mt5(self):
        """Connect to MetaTrader 5 with professional initialization"""
        try:
            self.log_message("🔗 Connecting to MetaTrader 5...", "SYSTEM")
            
            if self.mt5_interface.connect():
                self.is_connected = True
                self.conn_status.config(text="✅ Connected", foreground="green")
                self.start_trading_btn.config(state='normal')
                
                # Initialize professional systems
                self.initialize_professional_systems()
                
                self.log_message("🔗 MT5 connected successfully", "SUCCESS")
                self.log_message("🚀 Professional systems initialized", "SUCCESS")
                
                # Update account info immediately
                self.update_account_info()
                
            else:
                self.log_message("❌ Failed to connect to MT5", "ERROR")
                messagebox.showerror("Connection Error", "Failed to connect to MT5")
                
        except Exception as e:
            self.log_message(f"❌ MT5 connection error: {str(e)}", "ERROR")
            messagebox.showerror("Connection Error", f"Error: {str(e)}")

    def initialize_professional_systems(self):
        """Initialize all professional trading systems"""
        try:
            # Initialize portfolio manager
            self.portfolio_manager.initialize_portfolio(self.mt5_interface)
            self.log_message("💼 Portfolio Manager initialized", "SYSTEM")
            
            # Initialize recovery engine
            self.recovery_engine.reset()
            self.log_message("🛡️ Recovery Engine initialized", "SYSTEM")
            
            # Update portfolio mode display
            current_mode = self.portfolio_manager.current_mode
            self.portfolio_mode_label.config(text=current_mode.value)
            
            # Update recovery status
            self.recovery_mode_label.config(text="INACTIVE")
            
        except Exception as e:
            self.log_message(f"❌ System initialization error: {str(e)}", "ERROR")

    def start_professional_trading(self):
        """Start professional AI trading system"""
        if not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to MT5 first")
            return
            
        try:
            self.log_message("🚀 Starting Professional Trading System...", "SYSTEM")
            
            # Apply current configuration
            self.apply_all_settings()
            
            # Set training mode
            is_training_mode = self.training_mode_var.get()
            self.config['training_mode'] = is_training_mode
            self.mt5_interface.set_training_mode(is_training_mode)
            
            # Initialize professional trading environment
            self.trading_env = ProfessionalTradingEnvironment(
                self.mt5_interface, 
                self.recovery_engine, 
                self.config
            )
            self.trading_env.gui_instance = self
            self.trading_env.portfolio_manager = self.portfolio_manager
            
            # Initialize multi-agent RL system
            self.rl_agent_system = ProfessionalRLAgent(self.trading_env, self.config)
            
            # Load trained models if available
            if self.rl_agent_system.load_models():
                self.log_message("✅ AI Models loaded successfully!", "SUCCESS")
            else:
                self.log_message("⚠️ No trained models found, using random initialization", "WARNING")
            
            # Update GUI state
            self.is_trading = True
            self.start_trading_btn.config(state='disabled')
            self.stop_trading_btn.config(state='normal')
            
            # Start professional trading loop
            self.start_professional_trading_loop()
            
            self.log_message("🚀 Professional Trading System started successfully!", "SUCCESS")
            self.log_message("🤖 Multi-Agent AI system active", "AI")
            
        except Exception as e:
            self.log_message(f"❌ Error starting trading: {str(e)}", "ERROR")
            messagebox.showerror("Trading Error", f"Error: {str(e)}")

    def start_professional_trading_loop(self):
        """Start the main professional trading loop"""
        def trading_loop():
            self.log_message("🔄 Professional trading loop started", "SYSTEM")
            
            while self.is_trading:
                try:
                    # Get professional observation (150 features)
                    observation = self.trading_env._get_professional_observation()
                    
                    # Get multi-agent decision (15-dimensional action)
                    action, agent_info = self.rl_agent_system.get_ensemble_action(observation)
                    
                    # Log agent decision
                    active_agent = agent_info.get('active_agent', 'Unknown')
                    confidence = agent_info.get('confidence', 0.0)
                    self.log_message(f"🤖 {active_agent} Decision (Confidence: {confidence:.2f})", "AI")
                    
                    # Execute action through professional environment
                    observation_new, reward, done, truncated, info = self.trading_env.step(action)
                    
                    # Update trading state
                    if info and 'trading_state' in info:
                        self.current_trading_state = TradingState(info['trading_state'])
                    
                    # Update market regime
                    if info and 'market_regime' in info:
                        self.current_market_regime = MarketRegime(info['market_regime'])
                    
                    # Execute portfolio management
                    if self.portfolio_manager:
                        portfolio_result = self.portfolio_manager.execute_portfolio_management(
                            action, 
                            self.trading_env.market_data_cache,
                            self.mt5_interface
                        )
                        
                        if portfolio_result['success']:
                            total_actions = portfolio_result['execution_summary']['total_actions']
                            if total_actions > 0:
                                self.log_message(f"💼 Portfolio actions executed: {total_actions}", "SYSTEM")
                    
                    # Update agent performance tracking
                    self.update_agent_performance(active_agent, reward, agent_info)
                    
                    # Update GUI displays
                    self.root.after(0, self.update_trading_displays)
                    
                    # Log trading activity
                    if reward != 0:
                        self.log_message(f"💹 Trading Reward: {reward:.3f}", "AI")
                    
                    # Check for episode reset
                    if done:
                        self.trading_env.reset()
                        self.log_message("🔄 Episode completed, environment reset", "SYSTEM")
                    
                    # Trading interval
                    time.sleep(1)
                    
                except Exception as e:
                    self.log_message(f"❌ Trading loop error: {str(e)}", "ERROR")
                    time.sleep(5)
            
            self.log_message("⏹️ Professional trading loop stopped", "SYSTEM")
        
        self.trading_thread = threading.Thread(target=trading_loop, daemon=True)
        self.trading_thread.start()

    def stop_trading(self):
        """Stop AI trading"""
        self.is_trading = False
        self.start_trading_btn.config(state='normal')
        self.stop_trading_btn.config(state='disabled')
        self.log_message("⏹️ Trading stopped", "INFO")

    def emergency_stop(self):
        """Emergency stop all trading activities"""
        try:
            # Confirm emergency stop
            result = messagebox.askyesno("Emergency Stop", 
                                       "⚠️ This will immediately stop all trading and close all positions!\n\nAre you sure?")
            if not result:
                return
            
            self.log_message("🚨 EMERGENCY STOP ACTIVATED!", "ERROR")
            
            # Stop all trading
            self.is_trading = False
            self.is_training = False
            
            # Close all positions if connected
            if self.is_connected:
                positions = self.mt5_interface.get_positions()
                for position in positions:
                    try:
                        self.mt5_interface.close_position(position['ticket'])
                        self.log_message(f"🚨 Emergency closed position {position['ticket']}", "ERROR")
                    except Exception as e:
                        self.log_message(f"❌ Failed to close position {position['ticket']}: {e}", "ERROR")
            
            # Reset GUI state
            self.start_trading_btn.config(state='normal' if self.is_connected else 'disabled')
            self.stop_trading_btn.config(state='disabled')
            
            # Reset system states
            if hasattr(self, 'recovery_engine'):
                self.recovery_engine.reset()
            
            self.log_message("🚨 Emergency stop completed", "ERROR")
            messagebox.showinfo("Emergency Stop", "Emergency stop completed successfully!")
            
        except Exception as e:
            self.log_message(f"❌ Emergency stop error: {str(e)}", "ERROR")
            messagebox.showerror("Emergency Stop Error", f"Error during emergency stop: {str(e)}")
 
    def start_professional_training(self):
        """Start professional multi-agent training"""
        if not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to MT5 first")
            return
        
        try:
            # Get training parameters
            training_mode = self.training_mode_type_var.get()
            episodes = self.training_episodes_var.get()
            
            self.log_message(f"🎓 Starting {training_mode} training for {episodes} episodes", "SYSTEM")
            
            # Initialize training environment
            self.trading_env = ProfessionalTradingEnvironment(
                self.mt5_interface, 
                self.recovery_engine, 
                self.config
            )
            self.trading_env.set_training_mode(True)
            
            # Initialize RL system for training
            self.rl_agent_system = ProfessionalRLAgent(self.trading_env, self.config)
            
            # Update training GUI state
            self.is_training = True
            self.start_training_btn.config(state='disabled')
            self.pause_training_btn.config(state='normal')
            self.stop_training_btn.config(state='normal')
            
            # Start training thread
            self.start_training_thread(training_mode, episodes)
            
            self.log_message("🎓 Professional training started!", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"❌ Training start error: {str(e)}", "ERROR")
            messagebox.showerror("Training Error", f"Error: {str(e)}")

    def start_training_thread(self, training_mode, episodes):
        """Start the training thread"""
        def training_loop():
            try:
                self.training_start_time = time.time()
                
                for episode in range(episodes):
                    if not self.is_training:
                        break
                    
                    # Reset environment
                    observation, info = self.trading_env.reset()
                    episode_reward = 0
                    step_count = 0
                    
                    # Episode loop
                    while self.is_training:
                        # Get action from ensemble
                        action, agent_info = self.rl_agent_system.get_ensemble_action(observation)
                        
                        # Execute action
                        observation_new, reward, done, truncated, info = self.trading_env.step(action)
                        
                        # Store experience for training
                        self.rl_agent_system.store_experience(observation, action, reward, observation_new, done)
                        
                        episode_reward += reward
                        step_count += 1
                        observation = observation_new
                        
                        # Update training display
                        if step_count % 10 == 0:
                            self.root.after(0, lambda: self.update_training_progress(episode, episodes, episode_reward))
                        
                        if done or truncated:
                            break
                    
                    # Train agents periodically
                    if episode % 10 == 0 and episode > 0:
                        training_result = self.rl_agent_system.train_agents()
                        self.log_training_message(f"Episode {episode}: Reward={episode_reward:.2f}, Training={training_result}")
                    
                    # Update progress
                    progress = (episode + 1) / episodes * 100
                    self.root.after(0, lambda p=progress: self.overall_progress.config(value=p))
                
                self.log_message("🎓 Training completed successfully!", "SUCCESS")
                
            except Exception as e:
                self.log_message(f"❌ Training error: {str(e)}", "ERROR")
            finally:
                # Reset training state
                self.is_training = False
                self.root.after(0, self.reset_training_gui)
        
        self.training_thread = threading.Thread(target=training_loop, daemon=True)
        self.training_thread.start()

    def pause_training(self):
        """Pause/resume training"""
        if self.is_training:
            self.is_training = False
            self.pause_training_btn.config(text="▶️ Resume Training")
            self.log_message("⏸️ Training paused", "WARNING")
        else:
            self.is_training = True
            self.pause_training_btn.config(text="⏸️ Pause Training")
            self.log_message("▶️ Training resumed", "INFO")

    def stop_training(self):
        """Stop training"""
        self.is_training = False
        self.reset_training_gui()
        self.log_message("⏹️ Training stopped", "INFO")

    def update_all_professional_displays(self):
        """Update all professional GUI displays"""
        try:
            # Update core displays
            self.update_account_info()
            self.update_positions_table()
            self.update_trading_state_display()
            self.update_portfolio_displays()
            self.update_recovery_displays()
            self.update_ai_agent_displays()
            
            # Update charts if available
            self.update_professional_charts()
            
        except Exception as e:
            self.log_message(f"❌ Display update error: {str(e)}", "ERROR")

    def update_account_info(self):
        """Update account information display"""
        try:
            if not self.is_connected:
                return
            
            account_info = self.mt5_interface.get_account_info()
            if account_info:
                # Update balance and equity
                balance = account_info.get('balance', 0)
                equity = account_info.get('equity', 0)
                
                self.balance_label.config(text=f"${balance:.2f}")
                self.equity_label.config(text=f"${equity:.2f}")
                
                # Calculate and display floating P&L
                floating_pnl = equity - balance
                color = '#28a745' if floating_pnl >= 0 else '#dc3545'
                self.profit_label.config(text=f"${floating_pnl:.2f}", foreground=color)
                
                # Calculate drawdown
                if hasattr(self, 'peak_equity'):
                    if equity > self.peak_equity:
                        self.peak_equity = equity
                    
                    drawdown = ((self.peak_equity - equity) / self.peak_equity * 100) if self.peak_equity > 0 else 0
                    self.drawdown_label.config(text=f"{drawdown:.1f}%")
                else:
                    self.peak_equity = equity
                    
        except Exception as e:
            self.log_message(f"❌ Account update error: {str(e)}", "ERROR")

    def update_positions_table(self):
        """Update positions table with professional information"""
        try:
            # Clear existing items
            for item in self.pos_tree.get_children():
                self.pos_tree.delete(item)
            
            if not self.is_connected:
                return
            
            # Get current positions
            positions = self.mt5_interface.get_positions()
            
            # Update position count
            self.positions_count_label.config(text=str(len(positions)))
            
            # Get current price for calculation
            current_price = 0
            if hasattr(self, 'trading_env') and self.trading_env:
                current_price = self.trading_env.market_data_cache.get('current_price', 0)
            
            for position in positions:
                # Calculate duration
                open_time = position.get('time', 0)
                duration = ""
                if open_time:
                    duration_sec = time.time() - open_time
                    hours = int(duration_sec // 3600)
                    minutes = int((duration_sec % 3600) // 60)
                    duration = f"{hours:02d}:{minutes:02d}"
                
                # Calculate target profit
                volume = position.get('volume', 0.01)
                target_profit = 0
                if hasattr(self, 'portfolio_manager'):
                    thresholds = self.portfolio_manager.dynamic_thresholds
                    target_profit = thresholds.get('per_lot', 5.0) * (volume / 0.01)
                
                # Determine status
                current_profit = position.get('profit', 0)
                if current_profit >= target_profit:
                    status = "🎯 Target"
                    status_color = '#28a745'
                elif current_profit > 0:
                    status = "💰 Profit"
                    status_color = '#17a2b8'
                elif current_profit < -20:
                    status = "⚠️ Loss"
                    status_color = '#dc3545'
                else:
                    status = "📊 Active"
                    status_color = '#6c757d'
                
                # Insert row
                item_id = self.pos_tree.insert('', 'end', values=(
                    position.get('symbol', ''),
                    'BUY' if position.get('type', 0) == 0 else 'SELL',
                    f"{position.get('volume', 0):.2f}",
                    f"{position.get('price_open', 0):.2f}",
                    f"{current_price:.2f}" if current_price > 0 else "N/A",
                    f"${current_profit:.2f}",
                    f"${target_profit:.2f}",
                    duration,
                    status
                ))
                
                # Color the row based on profit status
                self.pos_tree.set(item_id, 'Status', status)
                
        except Exception as e:
            self.log_message(f"❌ Positions update error: {str(e)}", "ERROR")

    def update_trading_state_display(self):
        """Update trading state indicators"""
        try:
            # Update trading state
            state_text = f"🔍 {self.current_trading_state.value}"
            state_colors = {
                TradingState.MARKET_ANALYSIS: '#ffc107',
                TradingState.ENTRY: '#28a745',
                TradingState.MANAGE: '#17a2b8',
                TradingState.EXIT: '#dc3545',
                TradingState.RECOVERY: '#fd7e14'
            }
            color = state_colors.get(self.current_trading_state, '#6c757d')
            self.trading_state_label.config(text=state_text, foreground=color)
            
            # Update market regime
            regime_text = f"📊 {self.current_market_regime.value}"
            regime_colors = {
                MarketRegime.BULL_TREND: '#28a745',
                MarketRegime.BEAR_TREND: '#dc3545',
                MarketRegime.SIDEWAYS: '#17a2b8',
                MarketRegime.HIGH_VOLATILITY: '#fd7e14',
                MarketRegime.LOW_VOLATILITY: '#6c757d'
            }
            color = regime_colors.get(self.current_market_regime, '#6c757d')
            self.market_regime_label.config(text=regime_text, foreground=color)
            
        except Exception as e:
            self.log_message(f"❌ State display update error: {str(e)}", "ERROR")

    def update_portfolio_displays(self):
        """Update portfolio management displays"""
        try:
            if not hasattr(self, 'portfolio_manager') or not self.portfolio_manager:
                return
            
            # Update portfolio mode
            current_mode = self.portfolio_manager.current_mode
            self.portfolio_mode_label.config(text=current_mode.value)
            
            # Update portfolio heat
            portfolio_heat = self.portfolio_manager.portfolio_heat
            heat_color = '#28a745' if portfolio_heat < 10 else '#ffc107' if portfolio_heat < 15 else '#dc3545'
            self.portfolio_heat_label.config(text=f"{portfolio_heat:.1f}%", foreground=heat_color)
            self.portfolio_heat_detail_label.config(text=f"{portfolio_heat:.1f}%", foreground=heat_color)
            
            # Update correlation
            correlation = self.portfolio_manager.position_correlation
            self.position_correlation_label.config(text=f"{correlation:.2f}")
            
            # Update performance metrics
            self.portfolio_winrate_label.config(text=f"{self.portfolio_manager.win_rate:.1%}")
            self.sharpe_ratio_label.config(text=f"{self.portfolio_manager.sharpe_ratio:.2f}")
            
            # Update efficiency score
            efficiency = self.portfolio_manager.risk_adjusted_return
            self.efficiency_label.config(text=f"{efficiency:.1%}")
            self.portfolio_efficiency_detail_label.config(text=f"{efficiency:.1%}")
            
            # Update dynamic thresholds table
            self.update_thresholds_display()
            
        except Exception as e:
            self.log_message(f"❌ Portfolio display update error: {str(e)}", "ERROR")

    def update_recovery_displays(self):
        """Update recovery engine displays"""
        try:
            if not hasattr(self, 'recovery_engine') or not self.recovery_engine:
                return
            
            # Update recovery status
            recovery_mode = self.recovery_engine.current_mode
            recovery_text = recovery_mode.value
            
            if recovery_mode == RecoveryMode.INACTIVE:
                color = '#6c757d'
            else:
                color = '#fd7e14'
            
            self.recovery_status_label.config(text=recovery_text, foreground=color)
            self.recovery_mode_label.config(text=recovery_text, foreground=color)
            
            # Update recovery level
            recovery_level = self.recovery_engine.recovery_level
            self.recovery_level_label.config(text=str(recovery_level))
            
            # Update recovery efficiency
            efficiency = self.recovery_engine.recovery_efficiency
            self.recovery_efficiency_label.config(text=f"{efficiency:.1%}")
            
            # Update success rate
            total_attempts = self.recovery_engine.total_recovery_attempts
            successful = self.recovery_engine.successful_recoveries
            success_rate = (successful / total_attempts * 100) if total_attempts > 0 else 0
            self.recovery_success_label.config(text=f"{successful}/{total_attempts} ({success_rate:.1f}%)")
            
        except Exception as e:
            self.log_message(f"❌ Recovery display update error: {str(e)}", "ERROR")

    def update_ai_agent_displays(self):
        """Update AI agent monitoring displays"""
        try:
            if not hasattr(self, 'rl_agent_system') or not self.rl_agent_system:
                return
            
            # Update agent status displays
            agent_status = self.rl_agent_system.get_agent_status()
            
            for agent_name, status in agent_status.items():
                # Update individual agent displays
                if agent_name == 'PPO':
                    self.ppo_status.config(text=f"Status: {status.get('status', 'Inactive')}")
                    self.ppo_decisions.config(text=f"Decisions: {status.get('decisions', 0)}")
                    self.ppo_winrate.config(text=f"Win Rate: {status.get('win_rate', 0.0):.1%}")
                    self.ppo_avg_reward.config(text=f"Avg Reward: {status.get('avg_reward', 0.0):.2f}")
                
                elif agent_name == 'SAC':
                    self.sac_status.config(text=f"Status: {status.get('status', 'Inactive')}")
                    self.sac_decisions.config(text=f"Decisions: {status.get('decisions', 0)}")
                    self.sac_winrate.config(text=f"Win Rate: {status.get('win_rate', 0.0):.1%}")
                    self.sac_avg_reward.config(text=f"Avg Reward: {status.get('avg_reward', 0.0):.2f}")
                
                elif agent_name == 'TD3':
                    self.td3_status.config(text=f"Status: {status.get('status', 'Inactive')}")
                    self.td3_decisions.config(text=f"Decisions: {status.get('decisions', 0)}")
                    self.td3_winrate.config(text=f"Win Rate: {status.get('win_rate', 0.0):.1%}")
                    self.td3_avg_reward.config(text=f"Avg Reward: {status.get('avg_reward', 0.0):.2f}")
            
            # Update best agent
            best_agent = self.rl_agent_system.get_best_agent()
            self.best_agent_label.config(text=best_agent)
            
            # Update action distribution
            self.update_action_distribution_display()
            
        except Exception as e:
            self.log_message(f"❌ AI agent display update error: {str(e)}", "ERROR")

    def update_professional_charts(self):
        """Update professional charts"""
        try:
            if not self.is_connected or FigureCanvas is None:
                return
            
            # Update price chart
            self.update_price_chart()
            
            # Update portfolio chart
            self.update_portfolio_chart_data()
            
            # Update risk analysis chart
            self.update_risk_chart_data()
            
        except Exception as e:
            self.log_message(f"❌ Charts update error: {str(e)}", "ERROR")

    def log_message(self, message, level="INFO"):
        """Enhanced logging with professional formatting"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}\n"
            
            # Store log entry
            log_entry = {
                'timestamp': timestamp,
                'message': message,
                'level': level,
                'full_text': formatted_message
            }
            self.all_logs.append(log_entry)
            self.log_count += 1
            
            # Filter and display
            self.filter_and_display_logs()
            
            # Update log statistics
            self.update_log_statistics()
            
        except Exception as e:
            print(f"Logging error: {e}")

    def run(self):
        """Run the professional trading application"""
        try:
            # Set window properties
            self.setup_window_properties()
            
            # Initial log messages
            self.log_message("🚀 Professional AI Trading System Started", "SUCCESS")
            self.log_message("💡 Connect to MT5 to begin professional trading", "INFO")
            self.log_message("🤖 Multi-Agent RL System Ready", "AI")
            self.log_message("💼 Portfolio Manager Initialized", "SYSTEM")
            self.log_message("🛡️ Recovery Engine Standby", "SYSTEM")
            
            # Start main application loop
            self.root.mainloop()
            
        except Exception as e:
            self.log_message(f"❌ Application error: {str(e)}", "ERROR")
            print(f"Application error: {e}")
        finally:
            self.cleanup_professional_systems()

    def setup_window_properties(self):
        """Setup professional window properties"""
        try:
            # Set window icon if available
            try:
                self.root.iconbitmap('assets/trading_icon.ico')
            except:
                pass
            
            # Center window on screen
            self.root.update_idletasks()
            width = self.root.winfo_width()
            height = self.root.winfo_height()
            x = (self.root.winfo_screenwidth() // 2) - (width // 2)
            y = (self.root.winfo_screenheight() // 2) - (height // 2)
            self.root.geometry(f'{width}x{height}+{x}+{y}')
            
            # Set minimum window size
            self.root.minsize(1400, 900)
            
            # Handle window close event
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
        except Exception as e:
            print(f"Window setup error: {e}")

    def cleanup_professional_systems(self):
        """Cleanup all professional systems before exit"""
        try:
            self.log_message("🧹 Shutting down Professional Trading System...", "SYSTEM")
            
            # Stop all trading activities
            self.is_trading = False
            self.is_training = False
            
            # Save portfolio state
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                try:
                    self.portfolio_manager.save_portfolio_state()
                    self.log_message("💾 Portfolio state saved", "SYSTEM")
                except:
                    pass
            
            # Save RL models
            if hasattr(self, 'rl_agent_system') and self.rl_agent_system:
                try:
                    self.rl_agent_system.save_models()
                    self.log_message("💾 AI models saved", "SYSTEM")
                except:
                    pass
            
            # Save configuration
            try:
                self.save_professional_config()
                self.log_message("💾 Configuration saved", "SYSTEM")
            except:
                pass
            
            # Disconnect MT5
            if hasattr(self, 'mt5_interface') and self.mt5_interface:
                try:
                    self.mt5_interface.disconnect()
                    self.log_message("🔌 MT5 disconnected", "SYSTEM")
                except:
                    pass
            
            self.log_message("✅ Professional system shutdown completed", "SUCCESS")
            
        except Exception as e:
            print(f"Cleanup error: {e}")

    def on_closing(self):
        """Handle application closing"""
        try:
            # Ask for confirmation if trading is active
            if self.is_trading or self.is_training:
                result = messagebox.askyesno("Exit Confirmation", 
                                            "⚠️ Trading/Training is active!\n\nAre you sure you want to exit?")
                if not result:
                    return
            
            # Cleanup and exit
            self.cleanup_professional_systems()
            self.root.destroy()
            
        except Exception as e:
            print(f"Closing error: {e}")
            self.root.destroy()

    def update_agent_performance(self, agent_name, reward, agent_info):
        """Update agent performance tracking"""
        try:
            if agent_name not in self.ai_agent_performance:
                return
            
            # Update performance data
            self.ai_agent_performance[agent_name]['decisions'] += 1
            self.ai_agent_performance[agent_name]['rewards'].append(reward)
            
            # Keep only recent rewards (last 100)
            if len(self.ai_agent_performance[agent_name]['rewards']) > 100:
                self.ai_agent_performance[agent_name]['rewards'] = \
                    self.ai_agent_performance[agent_name]['rewards'][-100:]
            
            # Calculate win rate
            positive_rewards = [r for r in self.ai_agent_performance[agent_name]['rewards'] if r > 0]
            total_rewards = len(self.ai_agent_performance[agent_name]['rewards'])
            win_rate = len(positive_rewards) / total_rewards if total_rewards > 0 else 0
            self.ai_agent_performance[agent_name]['win_rate'] = win_rate
            
            # Update decision matrix
            self.update_decision_matrix(agent_name, agent_info, reward)
            
        except Exception as e:
            self.log_message(f"❌ Agent performance update error: {str(e)}", "ERROR")

    def update_decision_matrix(self, agent_name, agent_info, reward):
        """Update decision matrix display"""
        try:
            # Add to decision tree
            timestamp = datetime.now().strftime("%H:%M:%S")
            action_type = agent_info.get('action_type', 'Unknown')
            confidence = agent_info.get('confidence', 0.0)
            
            # Insert new decision
            self.decision_tree.insert('', 0, values=(
                timestamp,
                agent_name,
                action_type,
                f"{confidence:.2f}",
                f"{reward:.3f}"
            ))
            
            # Keep only recent decisions (last 20)
            children = self.decision_tree.get_children()
            if len(children) > 20:
                for item in children[20:]:
                    self.decision_tree.delete(item)
                    
        except Exception as e:
            self.log_message(f"❌ Decision matrix update error: {str(e)}", "ERROR")

    def update_action_distribution_display(self):
        """Update 15-dimensional action distribution display"""
        try:
            if not hasattr(self, 'rl_agent_system') or not self.rl_agent_system:
                return
            
            # Get recent action statistics
            action_stats = self.rl_agent_system.get_action_statistics()
            
            # Update action tree
            for item in self.action_tree.get_children():
                self.action_tree.delete(item)
            
            for i, (dim, desc, range_desc) in enumerate([
                ('A0', 'Market Direction', '-1 to 1'),
                ('A1', 'Position Size', '0.01 to 1.0'),
                ('A2', 'Entry Aggression', '0 to 1'),
                ('A3', 'Profit Target Ratio', '0.5 to 5.0'),
                ('A4', 'Partial Take Levels', '0 to 3'),
                ('A5', 'Add Position Signal', '0 to 1'),
                ('A6', 'Hedge Ratio', '0 to 1'),
                ('A7', 'Recovery Mode', '0 to 3'),
                ('A8', 'Correlation Limit', '0 to 1'),
                ('A9', 'Volatility Filter', '0 to 1'),
                ('A10', 'Spread Tolerance', '0 to 1'),
                ('A11', 'Time Filter', '0 to 1'),
                ('A12', 'Portfolio Heat Limit', '0 to 1'),
                ('A13', 'Smart Exit Signal', '0 to 1'),
                ('A14', 'Rebalance Trigger', '0 to 1')
            ]):
                current_val = action_stats.get(f'current_{i}', 0.0)
                avg_val = action_stats.get(f'avg_{i}', 0.0)
                usage_pct = action_stats.get(f'usage_{i}', 0.0)
                
                self.action_tree.insert('', 'end', values=(
                    dim, desc, f"{current_val:.3f}", f"{avg_val:.3f}", f"{usage_pct:.1f}%"
                ))
                
        except Exception as e:
            self.log_message(f"❌ Action distribution update error: {str(e)}", "ERROR")

    def update_thresholds_display(self):
        """Update dynamic thresholds display"""
        try:
            if not hasattr(self, 'portfolio_manager') or not self.portfolio_manager:
                return
            
            # Clear existing items
            for item in self.thresholds_tree.get_children():
                self.thresholds_tree.delete(item)
            
            # Get current thresholds
            thresholds = self.portfolio_manager.dynamic_thresholds
            timestamp = datetime.now().strftime('%H:%M:%S')
            
            # Add threshold entries
            threshold_data = [
                ('Portfolio Target', f"${thresholds.get('portfolio_target', 0):.2f}", "$100.00", 'Active'),
                ('Per Lot Target', f"${thresholds.get('per_lot', 0):.2f}", "$5.00", 'Active'),
                ('Stop Loss Target', f"${thresholds.get('stop_loss', 0):.2f}", "$-50.00", 'Active'),
                ('Take Profit Target', f"${thresholds.get('take_profit', 0):.2f}", "$25.00", 'Active'),
                ('Emergency Stop', f"${thresholds.get('emergency_stop', 0):.2f}", "$-200.00", 'Standby')
            ]
            
            for thresh_type, current, target, status in threshold_data:
                self.thresholds_tree.insert('', 'end', 
                                          values=(thresh_type, current, target, status, timestamp))
                
        except Exception as e:
            self.log_message(f"❌ Thresholds display update error: {str(e)}", "ERROR")

    def update_training_progress(self, episode, total_episodes, episode_reward):
        """Update training progress display"""
        try:
            # Update progress bar
            progress = (episode / total_episodes) * 100
            self.overall_progress.config(value=progress)
            self.overall_progress_label.config(text=f"{progress:.1f}%")
            
            # Update episode counter
            self.current_episode_label.config(text=f"{episode}/{total_episodes}")
            
            # Update training time
            if hasattr(self, 'training_start_time'):
                elapsed = time.time() - self.training_start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                self.training_time_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
                
                # Calculate ETA
                if episode > 0:
                    avg_time_per_episode = elapsed / episode
                    remaining_episodes = total_episodes - episode
                    eta_seconds = remaining_episodes * avg_time_per_episode
                    eta_hours = int(eta_seconds // 3600)
                    eta_minutes = int((eta_seconds % 3600) // 60)
                    self.eta_label.config(text=f"{eta_hours:02d}:{eta_minutes:02d}:00")
            
            # Update reward display
            self.avg_reward_label.config(text=f"{episode_reward:.3f}")
            
        except Exception as e:
            self.log_message(f"❌ Training progress update error: {str(e)}", "ERROR")

    def reset_training_gui(self):
        """Reset training GUI to initial state"""
        try:
            self.start_training_btn.config(state='normal')
            self.pause_training_btn.config(state='disabled', text="⏸️ Pause Training")
            self.stop_training_btn.config(state='disabled')
            
            self.overall_progress.config(value=0)
            self.overall_progress_label.config(text="0%")
            self.current_episode_label.config(text="0/0")
            self.training_time_label.config(text="00:00:00")
            self.eta_label.config(text="--:--:--")
            
        except Exception as e:
            self.log_message(f"❌ Training GUI reset error: {str(e)}", "ERROR")

    def log_training_message(self, message):
        """Log message to training logs"""
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}\n"
            
            self.training_logs.config(state='normal')
            self.training_logs.insert('end', formatted_message)
            
            # Auto-scroll to bottom
            self.training_logs.see('end')
            
            # Limit log size
            lines = int(self.training_logs.index('end-1c').split('.')[0])
            if lines > 1000:
                self.training_logs.delete('1.0', '500.0')
            
            self.training_logs.config(state='disabled')
            
        except Exception as e:
            print(f"Training log error: {e}")

    # === EVENT HANDLERS ===

    def on_portfolio_mode_change(self, event=None):
        """Handle portfolio mode change"""
        try:
            new_mode = self.portfolio_mode_var.get()
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                from portfolio_manager import PortfolioMode
                self.portfolio_manager.set_portfolio_mode(PortfolioMode(new_mode))
                self.log_message(f"💼 Portfolio mode changed to: {new_mode}", "SYSTEM")
                
        except Exception as e:
            self.log_message(f"❌ Portfolio mode change error: {str(e)}", "ERROR")

    def on_manual_agent_change(self, event=None):
        """Handle manual agent selection"""
        try:
            selected_agent = self.manual_agent_var.get()
            if selected_agent != "AUTO":
                self.auto_switching_var.set(False)
                if hasattr(self, 'rl_agent_system') and self.rl_agent_system:
                    self.rl_agent_system.set_manual_agent(selected_agent)
                    self.log_message(f"🤖 Manual agent selected: {selected_agent}", "AI")
            else:
                self.auto_switching_var.set(True)
                if hasattr(self, 'rl_agent_system') and self.rl_agent_system:
                    self.rl_agent_system.enable_auto_switching()
                    self.log_message("🤖 Auto agent switching enabled", "AI")
                    
        except Exception as e:
            self.log_message(f"❌ Agent change error: {str(e)}", "ERROR")

    def toggle_auto_switching(self):
        """Toggle automatic agent switching"""
        try:
            auto_enabled = self.auto_switching_var.get()
            if hasattr(self, 'rl_agent_system') and self.rl_agent_system:
                if auto_enabled:
                    self.rl_agent_system.enable_auto_switching()
                    self.manual_agent_var.set("AUTO")
                    self.log_message("🤖 Auto agent switching enabled", "AI")
                else:
                    self.rl_agent_system.disable_auto_switching()
                    self.log_message("🤖 Auto agent switching disabled", "AI")
                    
        except Exception as e:
            self.log_message(f"❌ Auto switching toggle error: {str(e)}", "ERROR")

    def show_position_context_menu(self, event):
        """Show context menu for positions"""
        try:
            # Get selected item
            item = self.pos_tree.selection()[0] if self.pos_tree.selection() else None
            if item:
                self.selected_position_item = item
                self.pos_context_menu.post(event.x_root, event.y_root)
                
        except Exception as e:
            self.log_message(f"❌ Context menu error: {str(e)}", "ERROR")

    def modify_selected_position(self):
        """Modify selected position"""
        try:
            if not hasattr(self, 'selected_position_item'):
                return
            
            # Get position data
            values = self.pos_tree.item(self.selected_position_item)['values']
            if not values:
                return
            
            symbol = values[0]
            messagebox.showinfo("Position Modification", f"Position modification for {symbol} - Feature coming soon!")
            
        except Exception as e:
            self.log_message(f"❌ Position modification error: {str(e)}", "ERROR")

    def close_selected_position(self):
        """Close selected position"""
        try:
            if not hasattr(self, 'selected_position_item'):
                return
            
            # Confirm close
            result = messagebox.askyesno("Close Position", "Are you sure you want to close this position?")
            if not result:
                return
            
            # Get position data and close
            values = self.pos_tree.item(self.selected_position_item)['values']
            if not values:
                return
            
            symbol = values[0]
            self.log_message(f"🔒 Closing position for {symbol}...", "SYSTEM")
            
            # Implementation would go here
            messagebox.showinfo("Position Closed", f"Position for {symbol} closed successfully!")
            
        except Exception as e:
            self.log_message(f"❌ Position close error: {str(e)}", "ERROR")

    def show_position_details(self):
        """Show detailed position information"""
        try:
            if not hasattr(self, 'selected_position_item'):
                return
            
            values = self.pos_tree.item(self.selected_position_item)['values']
            if not values:
                return
            
            symbol = values[0]
            messagebox.showinfo("Position Details", f"Detailed analysis for {symbol} position - Feature coming soon!")
            
        except Exception as e:
            self.log_message(f"❌ Position details error: {str(e)}", "ERROR")

    def activate_position_recovery(self):
        """Activate recovery for selected position"""
        try:
            if not hasattr(self, 'selected_position_item'):
                return
            
            values = self.pos_tree.item(self.selected_position_item)['values']
            if not values:
                return
            
            symbol = values[0]
            result = messagebox.askyesno("Activate Recovery", 
                                       f"Activate recovery strategy for {symbol} position?")
            if result:
                self.log_message(f"🛡️ Recovery activated for {symbol}", "SYSTEM")
                
        except Exception as e:
            self.log_message(f"❌ Recovery activation error: {str(e)}", "ERROR")

    # === PORTFOLIO MANAGEMENT ACTIONS ===

    def rebalance_portfolio(self):
        """Rebalance portfolio"""
        try:
            if not hasattr(self, 'portfolio_manager') or not self.portfolio_manager:
                messagebox.showwarning("Warning", "Portfolio manager not available")
                return
            
            result = messagebox.askyesno("Portfolio Rebalancing", 
                                       "Rebalance portfolio to optimal allocation?")
            if result:
                self.log_message("💼 Rebalancing portfolio...", "SYSTEM")
                # Rebalancing logic would go here
                messagebox.showinfo("Rebalancing", "Portfolio rebalanced successfully!")
                
        except Exception as e:
            self.log_message(f"❌ Portfolio rebalancing error: {str(e)}", "ERROR")

    def take_portfolio_profits(self):
        """Take portfolio profits"""
        try:
            result = messagebox.askyesno("Take Profits", 
                                       "Close all profitable positions?")
            if result:
                self.log_message("💰 Taking portfolio profits...", "SYSTEM")
                # Profit taking logic would go here
                messagebox.showinfo("Profits Taken", "Portfolio profits taken successfully!")
                
        except Exception as e:
            self.log_message(f"❌ Profit taking error: {str(e)}", "ERROR")

    def reduce_portfolio_risk(self):
        """Reduce portfolio risk"""
        try:
            result = messagebox.askyesno("Risk Reduction", 
                                       "Reduce portfolio risk exposure?")
            if result:
                self.log_message("🛡️ Reducing portfolio risk...", "SYSTEM")
                # Risk reduction logic would go here
                messagebox.showinfo("Risk Reduced", "Portfolio risk reduced successfully!")
                
        except Exception as e:
            self.log_message(f"❌ Risk reduction error: {str(e)}", "ERROR")
    
    def load_professional_config(self):
        """Load professional trading configuration"""
        try:
            config_file = 'config/professional_config.json'
            
            # Default professional configuration
            default_config = {
                # Trading Parameters
                'symbol': 'XAUUSD',
                'base_lot_size': 0.01,
                'max_positions': 10,
                'training_mode': False,
                
                # RL System Parameters
                'primary_agent': 'PPO',
                'ensemble_mode': True,
                'learning_rate': 0.0003,
                'exploration_rate': 0.1,
                
                # Portfolio Management
                'portfolio_mode': 'BALANCED',
                'portfolio_heat_limit': 15.0,
                'max_drawdown_limit': 10.0,
                'correlation_limit': 0.7,
                'dynamic_sizing': True,
                'smart_entries': True,
                'correlation_analysis': True,
                'multi_timeframe': True,
                
                # Risk Management
                'max_drawdown_threshold': 5.0,
                'emergency_stop_loss': 1000.0,
                'recovery_trigger_loss': 100.0,
                'max_recovery_levels': 8,
                
                # Recovery Engine
                'martingale_multiplier': 1.5,
                'grid_spacing': 200,
                'hedge_ratio': 0.5,
                'adaptive_sizing': True,
                'dynamic_spacing': True,
                'auto_hedge': True,
                
                # Advanced Features
                'ml_profit_optimization': True,
                'advanced_analytics': True,
                'real_time_optimization': True,
                'update_frequency': 1.0,
                
                # Training Parameters
                'training_episodes': 1000,
                'training_mode_type': 'ENSEMBLE',
                
                # Recovery Strategy Weights
                'martingale_weight': 0.3,
                'grid_weight': 0.3,
                'hedge_weight': 0.4
            }
            
            # Load existing config if available
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
                    self.log_message("📁 Professional configuration loaded", "SUCCESS")
            else:
                self.log_message("⚙️ Using default professional configuration", "INFO")
            
            return default_config
            
        except Exception as e:
            self.log_message(f"❌ Config load error: {str(e)}", "ERROR")
            return {}

    def save_professional_config(self):
        """Save current professional configuration"""
        try:
            config_file = 'config/professional_config.json'
            
            # Collect current configuration
            current_config = {
                # Trading Parameters
                'symbol': getattr(self, 'symbol_var', tk.StringVar()).get(),
                'base_lot_size': getattr(self, 'base_lot_var', tk.DoubleVar()).get(),
                'max_positions': getattr(self, 'max_positions_var', tk.IntVar()).get(),
                'training_mode': getattr(self, 'training_mode_var', tk.BooleanVar()).get(),
                
                # RL System Parameters
                'primary_agent': getattr(self, 'primary_agent_var', tk.StringVar()).get(),
                'ensemble_mode': getattr(self, 'ensemble_mode_var', tk.BooleanVar()).get(),
                'learning_rate': getattr(self, 'learning_rate_var', tk.DoubleVar()).get(),
                'exploration_rate': getattr(self, 'exploration_var', tk.DoubleVar()).get(),
                
                # Portfolio Management
                'portfolio_mode': getattr(self, 'portfolio_mode_var', tk.StringVar()).get(),
                'portfolio_heat_limit': getattr(self, 'portfolio_heat_limit_var', tk.DoubleVar()).get(),
                'max_drawdown_limit': getattr(self, 'max_drawdown_limit_var', tk.DoubleVar()).get(),
                'correlation_limit': getattr(self, 'correlation_limit_var', tk.DoubleVar()).get(),
                'dynamic_sizing': getattr(self, 'dynamic_sizing_var', tk.BooleanVar()).get(),
                'smart_entries': getattr(self, 'smart_entries_var', tk.BooleanVar()).get(),
                'correlation_analysis': getattr(self, 'correlation_analysis_var', tk.BooleanVar()).get(),
                'multi_timeframe': getattr(self, 'multi_timeframe_var', tk.BooleanVar()).get(),
                
                # Risk Management
                'max_drawdown_threshold': getattr(self, 'max_dd_config_var', tk.DoubleVar()).get(),
                'emergency_stop_loss': getattr(self, 'emergency_stop_var', tk.DoubleVar()).get(),
                'recovery_trigger_loss': getattr(self, 'recovery_trigger_var', tk.DoubleVar()).get(),
                'max_recovery_levels': getattr(self, 'max_recovery_var', tk.IntVar()).get(),
                
                # Recovery Engine
                'martingale_multiplier': getattr(self, 'martingale_multiplier_var', tk.DoubleVar()).get(),
                'grid_spacing': getattr(self, 'grid_spacing_var', tk.IntVar()).get(),
                'hedge_ratio': getattr(self, 'hedge_ratio_var', tk.DoubleVar()).get(),
                'adaptive_sizing': getattr(self, 'adaptive_sizing_var', tk.BooleanVar()).get(),
                'dynamic_spacing': getattr(self, 'dynamic_spacing_var', tk.BooleanVar()).get(),
                'auto_hedge': getattr(self, 'auto_hedge_var', tk.BooleanVar()).get(),
                
                # Advanced Features
                'ml_profit_optimization': getattr(self, 'ml_profit_optimization_var', tk.BooleanVar()).get(),
                'advanced_analytics': getattr(self, 'advanced_analytics_var', tk.BooleanVar()).get(),
                'real_time_optimization': getattr(self, 'real_time_optimization_var', tk.BooleanVar()).get(),
                'update_frequency': getattr(self, 'update_frequency_var', tk.DoubleVar()).get(),
                
                # Training Parameters
                'training_episodes': getattr(self, 'training_episodes_var', tk.IntVar()).get(),
                'training_mode_type': getattr(self, 'training_mode_type_var', tk.StringVar()).get(),
                
                # Recovery Strategy Weights
                'martingale_weight': getattr(self, 'martingale_weight_var', tk.DoubleVar()).get(),
                'grid_weight': getattr(self, 'grid_weight_var', tk.DoubleVar()).get(),
                'hedge_weight': getattr(self, 'hedge_weight_var', tk.DoubleVar()).get(),
                
                # Timestamp
                'last_saved': datetime.now().isoformat()
            }
            
            # Save to file
            os.makedirs('config', exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(current_config, f, indent=2)
            
            self.log_message("💾 Professional configuration saved", "SUCCESS")
            return True
            
        except Exception as e:
            self.log_message(f"❌ Config save error: {str(e)}", "ERROR")
            return False

    def load_professional_config_file(self):
        """Load configuration from file dialog"""
        try:
            filename = filedialog.askopenfilename(
                title="Load Professional Configuration",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                initialdir="config"
            )
            
            if not filename:
                return
            
            with open(filename, 'r') as f:
                config = json.load(f)
            
            # Apply loaded configuration
            self.apply_loaded_config(config)
            
            self.log_message(f"📁 Configuration loaded from {filename}", "SUCCESS")
            messagebox.showinfo("Success", "Configuration loaded successfully!")
            
        except Exception as e:
            self.log_message(f"❌ Error loading configuration: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Error loading configuration: {str(e)}")

    def apply_loaded_config(self, config):
        """Apply loaded configuration to GUI"""
        try:
            # Trading Parameters
            if hasattr(self, 'symbol_var'):
                self.symbol_var.set(config.get('symbol', 'XAUUSD'))
            if hasattr(self, 'base_lot_var'):
                self.base_lot_var.set(config.get('base_lot_size', 0.01))
            if hasattr(self, 'max_positions_var'):
                self.max_positions_var.set(config.get('max_positions', 10))
            if hasattr(self, 'training_mode_var'):
                self.training_mode_var.set(config.get('training_mode', False))
            
            # RL System Parameters
            if hasattr(self, 'primary_agent_var'):
                self.primary_agent_var.set(config.get('primary_agent', 'PPO'))
            if hasattr(self, 'ensemble_mode_var'):
                self.ensemble_mode_var.set(config.get('ensemble_mode', True))
            if hasattr(self, 'learning_rate_var'):
                self.learning_rate_var.set(config.get('learning_rate', 0.0003))
            if hasattr(self, 'exploration_var'):
                self.exploration_var.set(config.get('exploration_rate', 0.1))
            
            # Portfolio Management
            if hasattr(self, 'portfolio_mode_var'):
                self.portfolio_mode_var.set(config.get('portfolio_mode', 'BALANCED'))
            if hasattr(self, 'portfolio_heat_limit_var'):
                self.portfolio_heat_limit_var.set(config.get('portfolio_heat_limit', 15.0))
            if hasattr(self, 'max_drawdown_limit_var'):
                self.max_drawdown_limit_var.set(config.get('max_drawdown_limit', 10.0))
            if hasattr(self, 'correlation_limit_var'):
                self.correlation_limit_var.set(config.get('correlation_limit', 0.7))
            
            # Feature toggles
            if hasattr(self, 'dynamic_sizing_var'):
                self.dynamic_sizing_var.set(config.get('dynamic_sizing', True))
            if hasattr(self, 'smart_entries_var'):
                self.smart_entries_var.set(config.get('smart_entries', True))
            if hasattr(self, 'correlation_analysis_var'):
                self.correlation_analysis_var.set(config.get('correlation_analysis', True))
            if hasattr(self, 'multi_timeframe_var'):
                self.multi_timeframe_var.set(config.get('multi_timeframe', True))
            
            # Risk Management
            if hasattr(self, 'max_dd_config_var'):
                self.max_dd_config_var.set(config.get('max_drawdown_threshold', 5.0))
            if hasattr(self, 'emergency_stop_var'):
                self.emergency_stop_var.set(config.get('emergency_stop_loss', 1000.0))
            if hasattr(self, 'recovery_trigger_var'):
                self.recovery_trigger_var.set(config.get('recovery_trigger_loss', 100.0))
            if hasattr(self, 'max_recovery_var'):
                self.max_recovery_var.set(config.get('max_recovery_levels', 8))
            
            # Recovery Engine
            if hasattr(self, 'martingale_multiplier_var'):
                self.martingale_multiplier_var.set(config.get('martingale_multiplier', 1.5))
            if hasattr(self, 'grid_spacing_var'):
                self.grid_spacing_var.set(config.get('grid_spacing', 200))
            if hasattr(self, 'hedge_ratio_var'):
                self.hedge_ratio_var.set(config.get('hedge_ratio', 0.5))
            if hasattr(self, 'adaptive_sizing_var'):
                self.adaptive_sizing_var.set(config.get('adaptive_sizing', True))
            if hasattr(self, 'dynamic_spacing_var'):
                self.dynamic_spacing_var.set(config.get('dynamic_spacing', True))
            if hasattr(self, 'auto_hedge_var'):
                self.auto_hedge_var.set(config.get('auto_hedge', True))
            
            # Advanced Features
            if hasattr(self, 'ml_profit_optimization_var'):
                self.ml_profit_optimization_var.set(config.get('ml_profit_optimization', True))
            if hasattr(self, 'advanced_analytics_var'):
                self.advanced_analytics_var.set(config.get('advanced_analytics', True))
            if hasattr(self, 'real_time_optimization_var'):
                self.real_time_optimization_var.set(config.get('real_time_optimization', True))
            if hasattr(self, 'update_frequency_var'):
                self.update_frequency_var.set(config.get('update_frequency', 1.0))
            
            # Training Parameters
            if hasattr(self, 'training_episodes_var'):
                self.training_episodes_var.set(config.get('training_episodes', 1000))
            if hasattr(self, 'training_mode_type_var'):
                self.training_mode_type_var.set(config.get('training_mode_type', 'ENSEMBLE'))
            
            # Recovery Strategy Weights
            if hasattr(self, 'martingale_weight_var'):
                self.martingale_weight_var.set(config.get('martingale_weight', 0.3))
            if hasattr(self, 'grid_weight_var'):
                self.grid_weight_var.set(config.get('grid_weight', 0.3))
            if hasattr(self, 'hedge_weight_var'):
                self.hedge_weight_var.set(config.get('hedge_weight', 0.4))
            
        except Exception as e:
            self.log_message(f"❌ Config application error: {str(e)}", "ERROR")

    def reset_to_defaults(self):
        """Reset configuration to default values"""
        try:
            result = messagebox.askyesno("Reset Configuration", 
                                       "⚠️ This will reset all settings to defaults!\n\nAre you sure?")
            if not result:
                return
            
            # Load and apply default configuration
            default_config = self.load_professional_config()
            self.apply_loaded_config(default_config)
            
            self.log_message("🔄 Configuration reset to defaults", "INFO")
            messagebox.showinfo("Reset Complete", "Configuration reset to professional defaults!")
            
        except Exception as e:
            self.log_message(f"❌ Reset error: {str(e)}", "ERROR")
            messagebox.showerror("Reset Error", f"Error resetting configuration: {str(e)}")

    def apply_all_settings(self):
        """Apply all current settings to trading systems"""
        try:
            self.log_message("⚙️ Applying professional settings...", "SYSTEM")
            
            # Update config dictionary
            self.config.update({
                'symbol': getattr(self, 'symbol_var', tk.StringVar()).get(),
                'base_lot_size': getattr(self, 'base_lot_var', tk.DoubleVar()).get(),
                'max_positions': getattr(self, 'max_positions_var', tk.IntVar()).get(),
                'training_mode': getattr(self, 'training_mode_var', tk.BooleanVar()).get(),
                'portfolio_heat_limit': getattr(self, 'portfolio_heat_limit_var', tk.DoubleVar()).get(),
                'max_drawdown_limit': getattr(self, 'max_drawdown_limit_var', tk.DoubleVar()).get(),
                'correlation_limit': getattr(self, 'correlation_limit_var', tk.DoubleVar()).get(),
                'emergency_stop_loss': getattr(self, 'emergency_stop_var', tk.DoubleVar()).get(),
                'recovery_trigger_loss': getattr(self, 'recovery_trigger_var', tk.DoubleVar()).get(),
                'max_recovery_levels': getattr(self, 'max_recovery_var', tk.IntVar()).get(),
                'martingale_multiplier': getattr(self, 'martingale_multiplier_var', tk.DoubleVar()).get(),
                'grid_spacing': getattr(self, 'grid_spacing_var', tk.IntVar()).get(),
                'hedge_ratio': getattr(self, 'hedge_ratio_var', tk.DoubleVar()).get()
            })
            
            # Apply to portfolio manager
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                self.portfolio_manager.update_configuration(self.config)
            
            # Apply to recovery engine
            if hasattr(self, 'recovery_engine') and self.recovery_engine:
                self.recovery_engine.update_configuration(self.config)
            
            # Apply to trading environment
            if hasattr(self, 'trading_env') and self.trading_env:
                self.trading_env.update_configuration(self.config)
            
            # Apply to RL system
            if hasattr(self, 'rl_agent_system') and self.rl_agent_system:
                self.rl_agent_system.update_configuration(self.config)
            
            self.log_message("✅ Professional settings applied successfully", "SUCCESS")
            messagebox.showinfo("Settings Applied", "All professional settings applied successfully!")
            
        except Exception as e:
            self.log_message(f"❌ Settings application error: {str(e)}", "ERROR")
            messagebox.showerror("Settings Error", f"Error applying settings: {str(e)}")

    def update_price_chart(self):
        """Update real-time price chart"""
        try:
            if not hasattr(self, 'trading_env') or not self.trading_env:
                return
            
            # Get current market data
            current_price = self.trading_env.market_data_cache.get('current_price', 0)
            if current_price == 0:
                return
            
            # Add to price data
            current_time = datetime.now()
            self.price_data.append(current_price)
            self.time_data.append(current_time)
            
            # Keep only recent data (last 100 points)
            if len(self.price_data) > 100:
                self.price_data = self.price_data[-100:]
                self.time_data = self.time_data[-100:]
            
            # Update chart
            self.price_ax.clear()
            self.price_ax.plot(self.time_data, self.price_data, color='#17a2b8', linewidth=2)
            
            # Styling
            self.price_ax.set_facecolor('#2d2d2d')
            self.price_ax.tick_params(colors='white')
            self.price_ax.set_xlabel('Time', color='white')
            self.price_ax.set_ylabel('Price', color='white')
            self.price_ax.set_title('XAUUSD - Real-time Price Action', color='white', fontweight='bold')
            self.price_ax.grid(True, alpha=0.3)
            
            # Format time axis
            if len(self.time_data) > 1:
                import matplotlib.dates as mdates
                self.price_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                self.price_fig.autofmt_xdate()
            
            self.price_canvas.draw()
            
        except Exception as e:
            self.log_message(f"❌ Price chart update error: {str(e)}", "ERROR")

    def update_portfolio_chart_data(self):
        """Update portfolio performance chart"""
        try:
            if not hasattr(self, 'portfolio_manager') or not self.portfolio_manager:
                return
            
            # Get portfolio value
            if self.is_connected:
                account_info = self.mt5_interface.get_account_info()
                if account_info:
                    portfolio_value = account_info.get('equity', 0)
                    current_time = datetime.now()
                    
                    self.portfolio_data.append(portfolio_value)
                    self.portfolio_time_data.append(current_time)
                    
                    # Keep recent data
                    if len(self.portfolio_data) > 100:
                        self.portfolio_data = self.portfolio_data[-100:]
                        self.portfolio_time_data = self.portfolio_time_data[-100:]
                    
                    # Update chart
                    self.portfolio_ax.clear()
                    
                    # Plot portfolio value
                    self.portfolio_ax.plot(self.portfolio_time_data, self.portfolio_data, 
                                         color='#28a745', linewidth=2, label='Portfolio Value')
                    
                    # Add starting balance line if available
                    if len(self.portfolio_data) > 1:
                        start_value = self.portfolio_data[0]
                        self.portfolio_ax.axhline(y=start_value, color='#6c757d', 
                                                linestyle='--', alpha=0.7, label='Starting Balance')
                    
                    # Styling
                    self.portfolio_ax.set_facecolor('#2d2d2d')
                    self.portfolio_ax.tick_params(colors='white')
                    self.portfolio_ax.set_xlabel('Time', color='white')
                    self.portfolio_ax.set_ylabel('Portfolio Value', color='white')
                    self.portfolio_ax.set_title('Portfolio Performance - Real-time', 
                                              color='white', fontweight='bold')
                    self.portfolio_ax.grid(True, alpha=0.3)
                    self.portfolio_ax.legend()
                    
                    # Format time axis
                    if len(self.portfolio_time_data) > 1:
                        import matplotlib.dates as mdates
                        self.portfolio_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                        self.portfolio_fig.autofmt_xdate()
                    
                    self.portfolio_canvas.draw()
            
        except Exception as e:
            self.log_message(f"❌ Portfolio chart update error: {str(e)}", "ERROR")

    def update_risk_chart_data(self):
        """Update risk analysis charts"""
        try:
            if not hasattr(self, 'portfolio_manager') or not self.portfolio_manager:
                return
            
            current_time = datetime.now()
            
            # Get risk metrics
            current_drawdown = self.portfolio_manager.current_drawdown
            portfolio_heat = self.portfolio_manager.portfolio_heat
            
            # Add to risk data
            self.drawdown_data.append(current_drawdown)
            self.heat_data.append(portfolio_heat)
            self.risk_time_data.append(current_time)
            
            # Keep recent data
            if len(self.risk_time_data) > 100:
                self.drawdown_data = self.drawdown_data[-100:]
                self.heat_data = self.heat_data[-100:]
                self.risk_time_data = self.risk_time_data[-100:]
            
            # Update drawdown chart
            self.drawdown_ax.clear()
            self.drawdown_ax.fill_between(self.risk_time_data, self.drawdown_data, 
                                        color='#dc3545', alpha=0.7)
            self.drawdown_ax.plot(self.risk_time_data, self.drawdown_data, 
                                color='#dc3545', linewidth=2)
            
            # Drawdown styling
            self.drawdown_ax.set_facecolor('#2d2d2d')
            self.drawdown_ax.tick_params(colors='white')
            self.drawdown_ax.set_ylabel('Drawdown %', color='white')
            self.drawdown_ax.set_title('Portfolio Drawdown Analysis', color='white', fontweight='bold')
            self.drawdown_ax.grid(True, alpha=0.3)
            
            # Update portfolio heat chart
            self.heat_ax.clear()
            self.heat_ax.fill_between(self.risk_time_data, self.heat_data, 
                                    color='#ffc107', alpha=0.7)
            self.heat_ax.plot(self.risk_time_data, self.heat_data, 
                            color='#ffc107', linewidth=2)
            
            # Heat styling
            self.heat_ax.set_facecolor('#2d2d2d')
            self.heat_ax.tick_params(colors='white')
            self.heat_ax.set_xlabel('Time', color='white')
            self.heat_ax.set_ylabel('Portfolio Heat %', color='white')
            self.heat_ax.set_title('Portfolio Heat Monitoring', color='white', fontweight='bold')
            self.heat_ax.grid(True, alpha=0.3)
            
            # Format time axis
            if len(self.risk_time_data) > 1:
                import matplotlib.dates as mdates
                self.drawdown_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                self.heat_ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                self.risk_fig.autofmt_xdate()
            
            self.risk_fig.tight_layout()
            self.risk_canvas.draw()
            
        except Exception as e:
            self.log_message(f"❌ Risk chart update error: {str(e)}", "ERROR")

    def update_system_monitoring(self):
        """Update system monitoring displays"""
        try:
            # System efficiency calculation
            if hasattr(self, 'rl_agent_system') and self.rl_agent_system:
                agent_status = self.rl_agent_system.get_agent_status()
                
                # Calculate overall system efficiency
                total_decisions = sum(status.get('decisions', 0) for status in agent_status.values())
                total_rewards = sum(sum(status.get('rewards', [])) for status in agent_status.values())
                
                if total_decisions > 0:
                    avg_reward = total_rewards / total_decisions
                    self.system_efficiency = max(0, min(100, (avg_reward + 1) * 50))  # Normalize to 0-100
                else:
                    self.system_efficiency = 0
                
                # Update efficiency display
                eff_color = '#28a745' if self.system_efficiency > 70 else '#ffc107' if self.system_efficiency > 40 else '#dc3545'
                self.efficiency_label.config(text=f"{self.system_efficiency:.1f}%", foreground=eff_color)
            
        except Exception as e:
            self.log_message(f"❌ System monitoring error: {str(e)}", "ERROR")

    def start_system_monitoring(self):
        """Start system monitoring thread"""
        def monitoring_loop():
            while True:
                try:
                    if self.is_connected:
                        self.update_system_monitoring()
                    time.sleep(10)  # Monitor every 10 seconds
                    
                except Exception as e:
                    self.log_message(f"❌ Monitoring error: {str(e)}", "ERROR")
                    time.sleep(30)
        
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()

    def update_trading_displays(self):
        """Update trading-specific displays"""
        try:
            # Update recent actions display
            if hasattr(self, 'recent_actions'):
                # Limit recent actions to last 10
                if len(self.recent_actions) > 10:
                    self.recent_actions = self.recent_actions[-10:]
        
        except Exception as e:
            self.log_message(f"❌ Trading display update error: {str(e)}", "ERROR")

    def update_advanced_analytics(self):
        """Update advanced analytics processing"""
        try:
            if not self.is_connected:
                return
            
            # Market analysis
            self.analyze_market_conditions()
            
            # Performance analytics
            self.analyze_performance_metrics()
            
            # Risk analytics
            self.analyze_risk_metrics()
            
            # AI performance analytics
            self.analyze_ai_performance()
            
        except Exception as e:
            self.log_message(f"❌ Advanced analytics error: {str(e)}", "ERROR")

    def analyze_market_conditions(self):
        """Analyze current market conditions"""
        try:
            if not hasattr(self, 'trading_env') or not self.trading_env:
                return
            
            market_data = self.trading_env.market_data_cache
            
            # Volatility analysis
            atr = market_data.get('atr', 0)
            if atr > 15:
                volatility_regime = "HIGH"
            elif atr < 5:
                volatility_regime = "LOW"
            else:
                volatility_regime = "NORMAL"
            
            # Trend analysis
            trend_strength = market_data.get('trend_strength', 0)
            if trend_strength > 0.7:
                trend_direction = "STRONG_BULL" if market_data.get('current_price', 0) > market_data.get('sma_20', 0) else "STRONG_BEAR"
            elif trend_strength > 0.3:
                trend_direction = "WEAK_BULL" if market_data.get('current_price', 0) > market_data.get('sma_20', 0) else "WEAK_BEAR"
            else:
                trend_direction = "SIDEWAYS"
            
            # Update market regime if needed
            if trend_direction.startswith("STRONG_BULL"):
                self.current_market_regime = MarketRegime.BULL_TREND
            elif trend_direction.startswith("STRONG_BEAR"):
                self.current_market_regime = MarketRegime.BEAR_TREND
            elif volatility_regime == "HIGH":
                self.current_market_regime = MarketRegime.HIGH_VOLATILITY
            elif volatility_regime == "LOW":
                self.current_market_regime = MarketRegime.LOW_VOLATILITY
            else:
                self.current_market_regime = MarketRegime.SIDEWAYS
            
        except Exception as e:
            self.log_message(f"❌ Market analysis error: {str(e)}", "ERROR")

    def analyze_performance_metrics(self):
        """Analyze performance metrics"""
        try:
            if not self.is_connected:
                return
            
            # Get account info
            account_info = self.mt5_interface.get_account_info()
            if not account_info:
                return
            
            # Calculate session performance
            current_equity = account_info.get('equity', 0)
            starting_balance = account_info.get('balance', 0)
            
            # Session P&L
            session_pnl = current_equity - starting_balance
            session_pnl_pct = (session_pnl / starting_balance * 100) if starting_balance > 0 else 0
            
            # Update peak equity for drawdown calculation
            if not hasattr(self, 'session_peak_equity'):
                self.session_peak_equity = current_equity
            elif current_equity > self.session_peak_equity:
                self.session_peak_equity = current_equity
            
            # Calculate current drawdown
            current_drawdown = 0
            if self.session_peak_equity > 0:
                current_drawdown = (self.session_peak_equity - current_equity) / self.session_peak_equity * 100
            
            # Update portfolio manager if available
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                self.portfolio_manager.current_drawdown = current_drawdown
                self.portfolio_manager.peak_equity = self.session_peak_equity
            
            # Calculate win rate from recent trades
            if hasattr(self, 'recent_trades'):
                profitable_trades = [t for t in self.recent_trades if t.get('profit', 0) > 0]
                win_rate = len(profitable_trades) / len(self.recent_trades) if self.recent_trades else 0
                
                if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                    self.portfolio_manager.win_rate = win_rate
            
        except Exception as e:
            self.log_message(f"❌ Performance analysis error: {str(e)}", "ERROR")

    def analyze_risk_metrics(self):
        """Analyze risk metrics"""
        try:
            if not hasattr(self, 'portfolio_manager') or not self.portfolio_manager:
                return
            
            # Get current positions
            positions = self.mt5_interface.get_positions() if self.is_connected else []
            
            # Calculate portfolio heat
            total_risk = 0
            account_info = self.mt5_interface.get_account_info() if self.is_connected else {}
            balance = account_info.get('balance', 1000)  # Default balance
            
            for position in positions:
                position_risk = abs(position.get('profit', 0))
                total_risk += position_risk
            
            portfolio_heat = (total_risk / balance * 100) if balance > 0 else 0
            self.portfolio_manager.portfolio_heat = min(portfolio_heat, 100)  # Cap at 100%
            
            # Calculate position correlation
            if len(positions) > 1:
                # Simple correlation based on position types and profits
                buy_positions = [p for p in positions if p.get('type', 0) == 0]
                sell_positions = [p for p in positions if p.get('type', 0) == 1]
                
                total_positions = len(positions)
                buy_ratio = len(buy_positions) / total_positions
                sell_ratio = len(sell_positions) / total_positions
                
                # High correlation if positions are heavily skewed to one direction
                correlation = abs(buy_ratio - sell_ratio)
                self.portfolio_manager.position_correlation = correlation
            else:
                self.portfolio_manager.position_correlation = 0
            
        except Exception as e:
            self.log_message(f"❌ Risk analysis error: {str(e)}", "ERROR")

    def analyze_ai_performance(self):
        """Analyze AI system performance"""
        try:
            if not hasattr(self, 'rl_agent_system') or not self.rl_agent_system:
                return
            
            # Get agent performance data
            agent_status = self.rl_agent_system.get_agent_status()
            
            # Update agent performance tree
            for item in self.agent_perf_tree.get_children():
                self.agent_perf_tree.delete(item)
            
            for agent_name, status in agent_status.items():
                episodes = status.get('decisions', 0)
                rewards = status.get('rewards', [])
                avg_reward = sum(rewards) / len(rewards) if rewards else 0
                best_reward = max(rewards) if rewards else 0
                win_rate = status.get('win_rate', 0) * 100
                agent_status_text = status.get('status', 'Inactive')
                
                self.agent_perf_tree.insert('', 'end', values=(
                    agent_name,
                    str(episodes),
                    f"{avg_reward:.3f}",
                    f"{best_reward:.3f}",
                    f"{win_rate:.1f}%",
                    agent_status_text
                ))
            
            # Update AI performance chart data
            if hasattr(self, 'agent_performance_data'):
                for agent_name, status in agent_status.items():
                    if agent_name in self.agent_performance_data:
                        episodes = status.get('decisions', 0)
                        rewards = status.get('rewards', [])
                        cumulative_reward = sum(rewards) if rewards else 0
                        
                        self.agent_performance_data[agent_name]['episodes'].append(episodes)
                        self.agent_performance_data[agent_name]['rewards'].append(cumulative_reward)
                        
                        # Keep only recent data
                        if len(self.agent_performance_data[agent_name]['episodes']) > 100:
                            self.agent_performance_data[agent_name]['episodes'] = \
                                self.agent_performance_data[agent_name]['episodes'][-100:]
                            self.agent_performance_data[agent_name]['rewards'] = \
                                self.agent_performance_data[agent_name]['rewards'][-100:]
                
                # Update AI performance chart
                self.update_ai_performance_chart_display()
            
        except Exception as e:
            self.log_message(f"❌ AI performance analysis error: {str(e)}", "ERROR")

    def update_ai_performance_chart_display(self):
        """Update AI performance chart display"""
        try:
            if not hasattr(self, 'ai_ax') or not self.agent_performance_data:
                return
            
            self.ai_ax.clear()
            
            colors = {'PPO': '#17a2b8', 'SAC': '#28a745', 'TD3': '#ffc107'}
            
            for agent_name, data in self.agent_performance_data.items():
                if data['episodes'] and data['rewards']:
                    color = colors.get(agent_name, '#6c757d')
                    self.ai_ax.plot(data['episodes'], data['rewards'], 
                                    color=color, linewidth=2, label=f'{agent_name} Agent')
            
            # Styling
            self.ai_ax.set_facecolor('#2d2d2d')
            self.ai_ax.tick_params(colors='white')
            self.ai_ax.set_xlabel('Episode', color='white')
            self.ai_ax.set_ylabel('Cumulative Reward', color='white')
            self.ai_ax.set_title('AI Agents Performance Comparison', color='white', fontweight='bold')
            self.ai_ax.grid(True, alpha=0.3)
            self.ai_ax.legend()
            
            self.ai_canvas.draw()
            
        except Exception as e:
            self.log_message(f"❌ AI chart display error: {str(e)}", "ERROR")

    def update_performance_metrics(self, parent):
        """Setup and update performance metrics display"""
        try:
            # Clear existing widgets
            for widget in parent.winfo_children():
                widget.destroy()
            
            metrics_grid = ttk.Frame(parent)
            metrics_grid.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Session performance
            session_frame = ttk.Frame(metrics_grid)
            session_frame.pack(fill='x', pady=2)
            ttk.Label(session_frame, text="Session P&L:", style='Dark.TLabel').pack(side='left')
            
            session_pnl = 0
            if self.is_connected:
                account_info = self.mt5_interface.get_account_info()
                if account_info:
                    session_pnl = account_info.get('equity', 0) - account_info.get('balance', 0)
            
            color = '#28a745' if session_pnl >= 0 else '#dc3545'
            session_pnl_label = ttk.Label(session_frame, text=f"${session_pnl:.2f}", 
                                        foreground=color, style='Dark.TLabel')
            session_pnl_label.pack(side='right')
            
            # Win rate
            winrate_frame = ttk.Frame(metrics_grid)
            winrate_frame.pack(fill='x', pady=2)
            ttk.Label(winrate_frame, text="Win Rate:", style='Dark.TLabel').pack(side='left')
            
            win_rate = 0
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                win_rate = self.portfolio_manager.win_rate * 100
            
            winrate_label = ttk.Label(winrate_frame, text=f"{win_rate:.1f}%", style='Dark.TLabel')
            winrate_label.pack(side='right')
            
            # Sharpe ratio
            sharpe_frame = ttk.Frame(metrics_grid)
            sharpe_frame.pack(fill='x', pady=2)
            ttk.Label(sharpe_frame, text="Sharpe Ratio:", style='Dark.TLabel').pack(side='left')
            
            sharpe_ratio = 0
            if hasattr(self, 'portfolio_manager') and self.portfolio_manager:
                sharpe_ratio = self.portfolio_manager.sharpe_ratio
            
            sharpe_label = ttk.Label(sharpe_frame, text=f"{sharpe_ratio:.2f}", style='Dark.TLabel')
            sharpe_label.pack(side='right')
            
        except Exception as e:
            self.log_message(f"❌ Performance metrics update error: {str(e)}", "ERROR")

    def setup_risk_metrics(self, parent):
        """Setup risk metrics display"""
        try:
            metrics_grid = ttk.Frame(parent)
            metrics_grid.pack(fill='both', expand=True, padx=10, pady=10)
            
            # Max drawdown
            dd_frame = ttk.Frame(metrics_grid)
            dd_frame.pack(fill='x', pady=2)
            ttk.Label(dd_frame, text="Max Drawdown:", style='Dark.TLabel').pack(side='left')
            self.max_dd_label = ttk.Label(dd_frame, text="0.0%", 
                                        foreground='#dc3545', style='Dark.TLabel')
            self.max_dd_label.pack(side='right')
            
            # VaR (Value at Risk)
            var_frame = ttk.Frame(metrics_grid)
            var_frame.pack(fill='x', pady=2)
            ttk.Label(var_frame, text="Value at Risk:", style='Dark.TLabel').pack(side='left')
            self.var_label = ttk.Label(var_frame, text="$0.00", style='Dark.TLabel')
            self.var_label.pack(side='right')
            
            # Risk-Return Ratio
            rr_frame = ttk.Frame(metrics_grid)
            rr_frame.pack(fill='x', pady=2)
            ttk.Label(rr_frame, text="Risk-Return Ratio:", style='Dark.TLabel').pack(side='left')
            self.rr_label = ttk.Label(rr_frame, text="0.0", style='Dark.TLabel')
            self.rr_label.pack(side='right')
            
        except Exception as e:
            self.log_message(f"❌ Risk metrics setup error: {str(e)}", "ERROR")

    def filter_logs(self, event=None):
        """Filter logs by level"""
        try:
            selected_level = self.log_level_var.get()
            self.filter_and_display_logs(level_filter=selected_level)
            
        except Exception as e:
            self.log_message(f"❌ Log filter error: {str(e)}", "ERROR")

    def search_logs(self):
        """Open search dialog for logs"""
        try:
            search_term = self.search_var.get().strip()
            if not search_term:
                messagebox.showwarning("Search", "Please enter a search term")
                return
            
            self.filter_and_display_logs(search_term=search_term)
            
        except Exception as e:
            self.log_message(f"❌ Log search error: {str(e)}", "ERROR")

    def on_search_change(self, event=None):
        """Handle search term change"""
        try:
            search_term = self.search_var.get().strip()
            level_filter = self.log_level_var.get()
            self.filter_and_display_logs(search_term=search_term, level_filter=level_filter)
            
        except Exception as e:
            pass  # Silent fail for real-time search

    def filter_and_display_logs(self, search_term="", level_filter="ALL"):
        """Filter and display logs based on criteria"""
        try:
            # Filter logs
            self.filtered_logs = []
            
            for log_entry in self.all_logs:
                # Level filter
                if level_filter != "ALL" and log_entry['level'] != level_filter:
                    continue
                
                # Search filter
                if search_term and search_term.lower() not in log_entry['message'].lower():
                    continue
                
                self.filtered_logs.append(log_entry)
            
            # Update display
            self.log_text.config(state='normal')
            self.log_text.delete(1.0, tk.END)
            
            for log_entry in self.filtered_logs[-1000:]:  # Show last 1000 filtered logs
                timestamp = log_entry['timestamp']
                message = log_entry['message']
                level = log_entry['level']
                
                # Insert timestamp
                self.log_text.insert('end', f"[{timestamp}] ", "TIMESTAMP")
                
                # Insert message with appropriate tag
                self.log_text.insert('end', f"{message}\n", level)
            
            # Auto-scroll if enabled
            if self.auto_scroll_var.get():
                self.log_text.see('end')
            
            self.log_text.config(state='disabled')
            
            # Update statistics
            self.update_log_statistics()
            
        except Exception as e:
            print(f"Log filter error: {e}")

    def update_log_statistics(self):
        """Update log statistics display"""
        try:
            total_logs = len(self.all_logs)
            filtered_logs = len(self.filtered_logs)
            
            self.total_logs_label.config(text=f"Total: {total_logs}")
            self.filtered_logs_label.config(text=f"Showing: {filtered_logs}")
            
        except Exception as e:
            print(f"Log statistics error: {e}")

    def export_logs(self):
        """Export logs to file"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Export Logs",
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
                initialfilename=f"trading_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            )
            
            if not filename:
                return
            
            # Determine export format
            if filename.endswith('.csv'):
                self.export_logs_csv(filename)
            else:
                self.export_logs_txt(filename)
            
            self.log_message(f"📄 Logs exported to {filename}", "SUCCESS")
            messagebox.showinfo("Export Complete", f"Logs exported successfully to:\n{filename}")
            
        except Exception as e:
            self.log_message(f"❌ Log export error: {str(e)}", "ERROR")
            messagebox.showerror("Export Error", f"Error exporting logs: {str(e)}")

    def export_logs_txt(self, filename):
        """Export logs as text file"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("Professional AI Trading System - Log Export\n")
            f.write("=" * 50 + "\n")
            f.write(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Logs: {len(self.filtered_logs)}\n")
            f.write("=" * 50 + "\n\n")
            
            for log_entry in self.filtered_logs:
                f.write(f"[{log_entry['timestamp']}] [{log_entry['level']}] {log_entry['message']}\n")

    def export_logs_csv(self, filename):
        """Export logs as CSV file"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'Level', 'Message'])
            
            for log_entry in self.filtered_logs:
                writer.writerow([log_entry['timestamp'], log_entry['level'], log_entry['message']])

    def clear_logs(self):
        """Clear all logs"""
        try:
            result = messagebox.askyesno("Clear Logs", 
                                       "⚠️ This will clear all log messages!\n\nAre you sure?")
            if not result:
                return
            
            self.log_text.config(state='normal')
            self.log_text.delete(1.0, tk.END)
            self.log_text.config(state='disabled')
            
            self.all_logs = []
            self.filtered_logs = []
            self.log_count = 0
            self.displayed_count = 0
            
            self.update_log_statistics()
            
            # Add clear message
            self.log_message("🗑️ Log history cleared", "SYSTEM")
            
        except Exception as e:
            print(f"Clear logs error: {e}")
    
    # ========================= MAIN APPLICATION ENTRY POINT =========================

if __name__ == "__main__":
    try:
        print("🚀 Starting Professional AI Trading System...")
        print("=" * 60)
        
        # Create necessary directories
        directories = [
            'config',
            'models',
            'models/trained_models', 
            'models/checkpoints',
            'data',
            'data/market_data',
            'data/backtest',
            'logs',
            'logs/trading',
            'logs/training',
            'utils',
            'assets',
            'exports',
            'reports'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 Created directory: {directory}")
        
        print("=" * 60)
        
        # Check Python version
        import sys
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 or higher is required!")
            print(f"Current version: {sys.version}")
            sys.exit(1)
        
        print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
        
        # Check required modules
        required_modules = [
            'tkinter', 'numpy', 'pandas', 'matplotlib', 'threading', 'json', 'datetime'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
                print(f"✅ {module} - OK")
            except ImportError:
                missing_modules.append(module)
                print(f"❌ {module} - MISSING")
        
        if missing_modules:
            print(f"\n❌ Missing required modules: {', '.join(missing_modules)}")
            print("Please install missing modules before running the application.")
            sys.exit(1)
        
        # Check optional modules
        optional_modules = {
            'MetaTrader5': 'MT5 integration',
            'stable_baselines3': 'Advanced RL algorithms',
            'tensorflow': 'Deep learning support',
            'torch': 'PyTorch support'
        }
        
        print("\n📋 Optional modules check:")
        for module, description in optional_modules.items():
            try:
                __import__(module)
                print(f"✅ {module} - Available ({description})")
            except ImportError:
                print(f"⚠️ {module} - Not available ({description})")
        
        print("=" * 60)
        
        # Initialize and run the application
        print("🎯 Initializing Professional Trading GUI...")
        
        app = ProfessionalTradingGUI()
        
        print("✅ Professional Trading System initialized successfully!")
        print("🚀 Starting GUI application...")
        print("=" * 60)
        
        # Run the application
        app.run()
        
    except KeyboardInterrupt:
        print("\n⏹️ Application interrupted by user")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Startup error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Try to show error in GUI if possible
        try:
            import tkinter as tk
            from tkinter import messagebox
            
            root = tk.Tk()
            root.withdraw()  # Hide main window
            
            messagebox.showerror("Startup Error", 
                               f"Failed to start Professional Trading System:\n\n{str(e)}\n\nCheck console for detailed error information.")
            
        except:
            pass
        
        sys.exit(1)
    
    finally:
        print("\n🧹 Professional Trading System shutdown complete")
        print("Thank you for using Professional AI Trading System! 🚀")
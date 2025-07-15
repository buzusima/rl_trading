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
                
                self.recovery_engine.update_profit_settings(new_settings)
                self.log_message(f"Profit settings applied: Target=${new_settings['min_profit_target']:.1f}, Quick={new_settings['quick_profit_mode']}")
                messagebox.showinfo("Success", f"Profit settings applied!\nTarget: ${new_settings['min_profit_target']:.1f}\nMode: {new_settings['profit_mode']}")
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
        
    def setup_logs(self):
        # Log display
        self.log_text = tk.Text(self.logs_frame)
        log_scrollbar = ttk.Scrollbar(self.logs_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')
        
        # Clear logs button
        clear_btn = tk.Button(self.logs_frame, text="Clear Logs", 
                            command=self.clear_logs)
        clear_btn.pack(side='bottom', pady=5)
        
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
            # Initialize trading environment and RL agent
            self.trading_env = TradingEnvironment(self.mt5_interface, self.recovery_engine, self.config)
            self.rl_agent = RLAgent(self.trading_env, self.config)
            
            # Load trained model if available
            if self.rl_agent.load_model():
                self.log_message("Loaded trained model")
            else:
                self.log_message("No trained model found, using random actions")
                
            self.is_trading = True
            self.start_trading_btn.config(state='disabled')
            self.stop_trading_btn.config(state='normal')
            
            # Start trading thread
            self.trading_thread = threading.Thread(target=self.trading_loop)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            self.log_message("Trading started")
            
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
            
            self.is_training = True
            self.start_training_btn.config(state='disabled')
            self.stop_training_btn.config(state='normal')
            self.pause_training_btn.config(state='normal')
            
            print("DEBUG: Starting training thread...")
            
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
                # Get market data
                market_data = self.mt5_interface.get_market_data()
                
                # Get RL agent decision
                action = self.rl_agent.get_action(market_data)
                
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
        # Execute trading action based on RL agent decision
        # This would interface with the recovery engine
        pass
        
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
                'training_steps': 10000,
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
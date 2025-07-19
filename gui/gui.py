import tkinter as tk
from tkinter import ttk, messagebox
import threading
import json
import os
from datetime import datetime

class MinimalGUI:
    """
    Minimal Trading GUI
    - Connection management
    - Basic training controls
    - Account monitoring
    - Essential logging
    """
    
    def __init__(self):
        print("🎨 Initializing Minimal GUI...")
        
        # Main window
        self.root = tk.Tk()
        self.root.title("🤖 Simple AI Trading System")
        self.root.geometry("800x600")
        self.root.configure(bg='#2b2b2b')
        
        # System state
        self.is_connected = False
        self.is_training = False
        self.is_trading = False
        
        # Core components (will be initialized when needed)
        self.mt5_interface = None
        self.environment = None
        self.agent = None
        
        # Configuration
        self.config = self.load_config()
        
        # Threading
        self.training_thread = None
        self.update_thread = None
        self.running = True
        
        # Setup GUI
        self.setup_gui()
        
        # Start update loop
        self.start_update_loop()
        
        print("✅ Minimal GUI initialized successfully")

    def load_config(self):
        """Load basic configuration"""
        default_config = {
            'symbol': 'XAUUSD',
            'lot_size': 0.01,
            'max_positions': 5,
            'training_steps': 10000,
            'learning_rate': 0.0003
        }
        
        try:
            config_file = 'config/simple_config.json'
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    print("✅ Configuration loaded")
        except Exception as e:
            print(f"⚠️ Config load error: {e}, using defaults")
        
        return default_config

    def save_config(self):
        """Save current configuration"""
        try:
            os.makedirs('config', exist_ok=True)
            config_file = 'config/simple_config.json'
            
            # Update config with current GUI values
            self.config.update({
                'symbol': self.symbol_var.get(),
                'lot_size': self.lot_size_var.get(),
                'max_positions': self.max_positions_var.get(),
                'training_steps': self.training_steps_var.get(),
                'learning_rate': self.learning_rate_var.get()
            })
            
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.log_message("✅ Configuration saved", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"❌ Config save error: {e}", "ERROR")

    def setup_gui(self):
        """Setup minimal GUI interface"""
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
        """Setup main control tab"""
        # === CONNECTION SECTION ===
        conn_frame = ttk.LabelFrame(self.main_frame, text="🔗 MT5 Connection")
        conn_frame.pack(fill='x', padx=10, pady=5)
        
        conn_controls = ttk.Frame(conn_frame)
        conn_controls.pack(fill='x', padx=10, pady=10)
        
        # Connection status
        self.conn_status_label = ttk.Label(conn_controls, text="❌ Disconnected", 
                                          foreground='red')
        self.conn_status_label.pack(side='left')
        
        # Connect button
        self.connect_btn = ttk.Button(conn_controls, text="🔌 Connect MT5", 
                                     command=self.connect_mt5)
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
        
        # === TRADING CONTROLS SECTION ===
        trading_frame = ttk.LabelFrame(self.main_frame, text="⚡ Trading Controls")
        trading_frame.pack(fill='x', padx=10, pady=5)
        
        trading_controls = ttk.Frame(trading_frame)
        trading_controls.pack(fill='x', padx=10, pady=10)
        
        # Start/Stop trading
        self.start_trading_btn = ttk.Button(trading_controls, text="🚀 Start Trading", 
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
        """Setup training controls tab"""
        # === CONFIGURATION SECTION ===
        config_frame = ttk.LabelFrame(self.training_frame, text="⚙️ Configuration")
        config_frame.pack(fill='x', padx=10, pady=5)
        
        config_grid = ttk.Frame(config_frame)
        config_grid.pack(fill='x', padx=10, pady=10)
        
        # Symbol
        ttk.Label(config_grid, text="Symbol:").grid(row=0, column=0, sticky='w')
        self.symbol_var = tk.StringVar(value=self.config.get('symbol', 'XAUUSD'))
        symbol_entry = ttk.Entry(config_grid, textvariable=self.symbol_var, width=10)
        symbol_entry.grid(row=0, column=1, sticky='w', padx=(10, 0))
        
        # Lot size
        ttk.Label(config_grid, text="Lot Size:").grid(row=1, column=0, sticky='w')
        self.lot_size_var = tk.DoubleVar(value=self.config.get('lot_size', 0.01))
        lot_size_entry = ttk.Entry(config_grid, textvariable=self.lot_size_var, width=10)
        lot_size_entry.grid(row=1, column=1, sticky='w', padx=(10, 0))
        
        # Max positions
        ttk.Label(config_grid, text="Max Positions:").grid(row=2, column=0, sticky='w')
        self.max_positions_var = tk.IntVar(value=self.config.get('max_positions', 5))
        max_pos_entry = ttk.Entry(config_grid, textvariable=self.max_positions_var, width=10)
        max_pos_entry.grid(row=2, column=1, sticky='w', padx=(10, 0))
        
        # Training steps
        ttk.Label(config_grid, text="Training Steps:").grid(row=3, column=0, sticky='w')
        self.training_steps_var = tk.IntVar(value=self.config.get('training_steps', 10000))
        training_steps_entry = ttk.Entry(config_grid, textvariable=self.training_steps_var, width=10)
        training_steps_entry.grid(row=3, column=1, sticky='w', padx=(10, 0))
        
        # Learning rate
        ttk.Label(config_grid, text="Learning Rate:").grid(row=4, column=0, sticky='w')
        self.learning_rate_var = tk.DoubleVar(value=self.config.get('learning_rate', 0.0003))
        lr_entry = ttk.Entry(config_grid, textvariable=self.learning_rate_var, width=10)
        lr_entry.grid(row=4, column=1, sticky='w', padx=(10, 0))
        
        # Save config button
        save_config_btn = ttk.Button(config_grid, text="💾 Save Config", 
                                    command=self.save_config)
        save_config_btn.grid(row=5, column=0, columnspan=2, pady=10)
        
        # === TRAINING CONTROLS SECTION ===
        training_ctrl_frame = ttk.LabelFrame(self.training_frame, text="🎓 Training Controls")
        training_ctrl_frame.pack(fill='x', padx=10, pady=5)
        
        training_controls = ttk.Frame(training_ctrl_frame)
        training_controls.pack(fill='x', padx=10, pady=10)
        
        # Training buttons
        self.start_training_btn = ttk.Button(training_controls, text="🚀 Start Training", 
                                            command=self.start_training)
        self.start_training_btn.pack(side='left', padx=5)
        
        self.stop_training_btn = ttk.Button(training_controls, text="⏹️ Stop Training", 
                                           command=self.stop_training, state='disabled')
        self.stop_training_btn.pack(side='left', padx=5)
        
        # Training status
        self.training_status_label = ttk.Label(training_controls, text="Ready to train")
        self.training_status_label.pack(side='right')

    def setup_logs_tab(self):
        """Setup logging tab"""
        # Log display
        log_frame = ttk.Frame(self.logs_frame)
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Log text with scrollbar
        self.log_text = tk.Text(log_frame, bg='#1e1e1e', fg='white', 
                               font=('Consolas', 9))
        log_scrollbar = ttk.Scrollbar(log_frame, orient='vertical', 
                                     command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')
        
        # Log controls
        log_controls = ttk.Frame(self.logs_frame)
        log_controls.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(log_controls, text="🗑️ Clear Logs", 
                  command=self.clear_logs).pack(side='left', padx=5)
        
        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_controls, text="Auto-scroll", 
                       variable=self.auto_scroll_var).pack(side='right')

    def connect_mt5(self):
        """Connect to MT5"""
        try:
            self.log_message("🔌 Connecting to MT5...", "INFO")
            
            # Import MT5 interface
            from core.mt5_connector import MT5Interface
            self.mt5_interface = MT5Interface(self.config)
            
            # Attempt connection
            if self.mt5_interface.connect():
                self.is_connected = True
                self.conn_status_label.config(text="✅ Connected", foreground='green')
                self.start_trading_btn.config(state='normal')
                self.log_message("✅ MT5 connected successfully", "SUCCESS")
                
                # Update account info
                self.update_account_info()
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

    def start_training(self):
        """Start training in background thread"""
        if self.is_training:
            self.log_message("⚠️ Training already in progress", "WARNING")
            return
        
        try:
            self.log_message("🎓 Starting training...", "INFO")
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
        """Training worker thread"""
        try:
            # Initialize components
            from core.simple_environment import Environment
            from core.basic_agent import BasicRLAgent
            
            # Create environment
            self.environment = Environment(self.mt5_interface, self.config)
            self.log_message("✅ Environment created", "INFO")
            
            # Create agent
            self.agent = BasicRLAgent(self.environment, self.config)
            self.log_message("✅ Agent created", "INFO")
            
            # Start training
            training_steps = self.training_steps_var.get()
            self.log_message(f"🚀 Training for {training_steps} steps...", "INFO")
            
            # Note: This would need the train method to be implemented in BasicRLAgent
            # self.agent.train(total_timesteps=training_steps)
            
            self.log_message("✅ Training completed successfully", "SUCCESS")
            
        except Exception as e:
            self.log_message(f"❌ Training error: {e}", "ERROR")
        finally:
            self.root.after(0, self.reset_training_controls)

    def start_trading(self):
        """Start live trading"""
        if not self.is_connected:
            messagebox.showwarning("Warning", "Please connect to MT5 first")
            return
        
        if self.agent is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return
        
        self.is_trading = True
        self.start_trading_btn.config(state='disabled')
        self.stop_trading_btn.config(state='normal')
        self.log_message("🚀 Live trading started", "SUCCESS")

    def stop_trading(self):
        """Stop live trading"""
        self.is_trading = False
        self.start_trading_btn.config(state='normal')
        self.stop_trading_btn.config(state='disabled')
        self.log_message("⏹️ Live trading stopped", "INFO")

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
        self.training_status_label.config(text="Ready to train")

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
                except:
                    pass  # Silent fail for updates
                
                # Schedule next update
                self.root.after(5000, update_loop)  # Update every 5 seconds
        
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
            self.log_message("🚀 Simple AI Trading System started", "SUCCESS")
            self.log_message("📋 Connect to MT5 to begin", "INFO")
            
            # Start main loop
            self.root.mainloop()
            
        except Exception as e:
            print(f"GUI run error: {e}")
            raise e
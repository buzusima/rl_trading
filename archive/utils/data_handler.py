# utils/data_handler.py - Data Management Utilities
import pandas as pd
import numpy as np
import json
import csv
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import pickle

class DataHandler:
    """
    Utility class for handling trading data, configurations, and persistence
    """
    
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, 'trading_data.db')
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
    def init_database(self):
        """
        Initialize SQLite database for trading data
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    symbol TEXT,
                    action TEXT,
                    volume REAL,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    duration_minutes INTEGER,
                    recovery_level INTEGER,
                    recovery_type TEXT,
                    rl_action TEXT,
                    market_conditions TEXT
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    balance REAL,
                    equity REAL,
                    total_pnl REAL,
                    drawdown REAL,
                    recovery_active BOOLEAN,
                    open_positions INTEGER,
                    win_rate REAL,
                    profit_factor REAL
                )
            ''')
            
            # RL training sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time DATETIME,
                    end_time DATETIME,
                    algorithm TEXT,
                    total_timesteps INTEGER,
                    final_reward REAL,
                    learning_rate REAL,
                    model_path TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Database initialization error: {str(e)}")
            
    def save_trade(self, trade_data: Dict):
        """
        Save trade data to database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    timestamp, symbol, action, volume, entry_price, 
                    exit_price, pnl, duration_minutes, recovery_level,
                    recovery_type, rl_action, market_conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('timestamp', datetime.now()),
                trade_data.get('symbol', ''),
                trade_data.get('action', ''),
                trade_data.get('volume', 0),
                trade_data.get('entry_price', 0),
                trade_data.get('exit_price', 0),
                trade_data.get('pnl', 0),
                trade_data.get('duration_minutes', 0),
                trade_data.get('recovery_level', 0),
                trade_data.get('recovery_type', ''),
                trade_data.get('rl_action', ''),
                json.dumps(trade_data.get('market_conditions', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving trade: {str(e)}")
            
    def save_performance_metrics(self, metrics: Dict):
        """
        Save performance metrics to database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (
                    timestamp, balance, equity, total_pnl, drawdown,
                    recovery_active, open_positions, win_rate, profit_factor
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics.get('timestamp', datetime.now()),
                metrics.get('balance', 0),
                metrics.get('equity', 0),
                metrics.get('total_pnl', 0),
                metrics.get('drawdown', 0),
                metrics.get('recovery_active', False),
                metrics.get('open_positions', 0),
                metrics.get('win_rate', 0),
                metrics.get('profit_factor', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving performance metrics: {str(e)}")
            
    def get_trade_history(self, symbol: str = None, days: int = 30):
        """
        Get trade history from database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM trades 
                WHERE timestamp >= datetime('now', '-{} days')
            '''.format(days)
            
            if symbol:
                query += " AND symbol = ?"
                df = pd.read_sql_query(query, conn, params=[symbol])
            else:
                df = pd.read_sql_query(query, conn)
                
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            return df
            
        except Exception as e:
            print(f"Error getting trade history: {str(e)}")
            return pd.DataFrame()
            
    def get_performance_history(self, days: int = 30):
        """
        Get performance metrics history
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM performance_metrics 
                WHERE timestamp >= datetime('now', '-{} days')
                ORDER BY timestamp
            '''.format(days)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            return df
            
        except Exception as e:
            print(f"Error getting performance history: {str(e)}")
            return pd.DataFrame()
            
    def calculate_trading_statistics(self, days: int = 30):
        """
        Calculate comprehensive trading statistics
        """
        try:
            trade_df = self.get_trade_history(days=days)
            
            if trade_df.empty:
                return {}
                
            stats = {}
            
            # Basic statistics
            stats['total_trades'] = len(trade_df)
            stats['winning_trades'] = len(trade_df[trade_df['pnl'] > 0])
            stats['losing_trades'] = len(trade_df[trade_df['pnl'] < 0])
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades'] if stats['total_trades'] > 0 else 0
            
            # PnL statistics
            stats['total_pnl'] = trade_df['pnl'].sum()
            stats['average_pnl'] = trade_df['pnl'].mean()
            stats['best_trade'] = trade_df['pnl'].max()
            stats['worst_trade'] = trade_df['pnl'].min()
            
            # Win/Loss analysis
            winning_trades = trade_df[trade_df['pnl'] > 0]['pnl']
            losing_trades = trade_df[trade_df['pnl'] < 0]['pnl']
            
            stats['average_win'] = winning_trades.mean() if len(winning_trades) > 0 else 0
            stats['average_loss'] = losing_trades.mean() if len(losing_trades) > 0 else 0
            
            # Profit factor
            gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
            stats['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Recovery analysis
            recovery_trades = trade_df[trade_df['recovery_level'] > 0]
            stats['recovery_trades'] = len(recovery_trades)
            stats['recovery_success_rate'] = len(recovery_trades[recovery_trades['pnl'] > 0]) / len(recovery_trades) if len(recovery_trades) > 0 else 0
            
            # Drawdown analysis
            cumulative_pnl = trade_df['pnl'].cumsum()
            running_max = cumulative_pnl.cummax()
            drawdown = running_max - cumulative_pnl
            stats['max_drawdown'] = drawdown.max()
            stats['current_drawdown'] = drawdown.iloc[-1] if len(drawdown) > 0 else 0
            
            # Time analysis
            trade_df['duration_hours'] = trade_df['duration_minutes'] / 60
            stats['average_trade_duration'] = trade_df['duration_hours'].mean()
            stats['longest_trade'] = trade_df['duration_hours'].max()
            stats['shortest_trade'] = trade_df['duration_hours'].min()
            
            return stats
            
        except Exception as e:
            print(f"Error calculating trading statistics: {str(e)}")
            return {}
            
    def export_data_to_csv(self, data_type: str, filename: str = None):
        """
        Export data to CSV file
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{data_type}_{timestamp}.csv"
                
            filepath = os.path.join(self.data_dir, filename)
            
            if data_type == 'trades':
                df = self.get_trade_history(days=365)  # Get full year
            elif data_type == 'performance':
                df = self.get_performance_history(days=365)
            else:
                print(f"Unknown data type: {data_type}")
                return None
                
            if not df.empty:
                df.to_csv(filepath, index=False)
                print(f"Data exported to: {filepath}")
                return filepath
            else:
                print("No data to export")
                return None
                
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            return None
            
    def import_data_from_csv(self, filepath: str, data_type: str):
        """
        Import data from CSV file
        """
        try:
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                return False
                
            df = pd.read_csv(filepath)
            
            if data_type == 'trades':
                for _, row in df.iterrows():
                    trade_data = row.to_dict()
                    self.save_trade(trade_data)
                    
            elif data_type == 'performance':
                for _, row in df.iterrows():
                    metrics = row.to_dict()
                    self.save_performance_metrics(metrics)
                    
            else:
                print(f"Unknown data type: {data_type}")
                return False
                
            print(f"Data imported from: {filepath}")
            return True
            
        except Exception as e:
            print(f"Error importing data: {str(e)}")
            return False
            
    def backup_database(self, backup_path: str = None):
        """
        Create backup of database
        """
        try:
            if backup_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(self.data_dir, f"backup_trading_data_{timestamp}.db")
                
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            print(f"Database backed up to: {backup_path}")
            return backup_path
            
        except Exception as e:
            print(f"Error backing up database: {str(e)}")
            return None
            
    def restore_database(self, backup_path: str):
        """
        Restore database from backup
        """
        try:
            if not os.path.exists(backup_path):
                print(f"Backup file not found: {backup_path}")
                return False
                
            # Copy backup to current database
            import shutil
            shutil.copy2(backup_path, self.db_path)
            
            print(f"Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            print(f"Error restoring database: {str(e)}")
            return False
            
    def save_config(self, config: Dict, filename: str = None):
        """
        Save configuration to JSON file
        """
        try:
            if filename is None:
                filename = 'current_config.json'
                
            filepath = os.path.join(self.data_dir, filename)
            
            # Add timestamp to config
            config['saved_at'] = datetime.now().isoformat()
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4, default=str)
                
            print(f"Configuration saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving config: {str(e)}")
            return None
            
    def load_config(self, filename: str = None):
        """
        Load configuration from JSON file
        """
        try:
            if filename is None:
                filename = 'current_config.json'
                
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Config file not found: {filepath}")
                return None
                
            with open(filepath, 'r') as f:
                config = json.load(f)
                
            print(f"Configuration loaded from: {filepath}")
            return config
            
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return None
            
    def save_model_state(self, model_data: Dict, filename: str = None):
        """
        Save model state using pickle
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_state_{timestamp}.pkl"
                
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
                
            print(f"Model state saved to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"Error saving model state: {str(e)}")
            return None
            
    def load_model_state(self, filename: str):
        """
        Load model state from pickle file
        """
        try:
            filepath = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"Model state file not found: {filepath}")
                return None
                
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            print(f"Model state loaded from: {filepath}")
            return model_data
            
        except Exception as e:
            print(f"Error loading model state: {str(e)}")
            return None
            
    def get_data_summary(self):
        """
        Get summary of all stored data
        """
        try:
            summary = {}
            
            # Database summary
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count trades
            cursor.execute("SELECT COUNT(*) FROM trades")
            summary['total_trades'] = cursor.fetchone()[0]
            
            # Count performance records
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            summary['performance_records'] = cursor.fetchone()[0]
            
            # Count training sessions
            cursor.execute("SELECT COUNT(*) FROM training_sessions")
            summary['training_sessions'] = cursor.fetchone()[0]
            
            # Date range
            cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM trades")
            date_range = cursor.fetchone()
            summary['trade_date_range'] = date_range
            
            conn.close()
            
            # File summary
            files_summary = {}
            if os.path.exists(self.data_dir):
                for filename in os.listdir(self.data_dir):
                    filepath = os.path.join(self.data_dir, filename)
                    if os.path.isfile(filepath):
                        file_size = os.path.getsize(filepath)
                        files_summary[filename] = {
                            'size_bytes': file_size,
                            'size_mb': file_size / (1024 * 1024),
                            'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
                        }
                        
            summary['files'] = files_summary
            
            return summary
            
        except Exception as e:
            print(f"Error getting data summary: {str(e)}")
            return {}
            
    def clean_old_data(self, days_to_keep: int = 90):
        """
        Clean old data to save space
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old trades
            cursor.execute("DELETE FROM trades WHERE timestamp < ?", (cutoff_date,))
            trades_deleted = cursor.rowcount
            
            # Delete old performance metrics
            cursor.execute("DELETE FROM performance_metrics WHERE timestamp < ?", (cutoff_date,))
            metrics_deleted = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            print(f"Cleaned {trades_deleted} old trades and {metrics_deleted} old metrics")
            
            # Clean old backup files
            backup_files_deleted = 0
            if os.path.exists(self.data_dir):
                for filename in os.listdir(self.data_dir):
                    if filename.startswith('backup_') and filename.endswith('.db'):
                        filepath = os.path.join(self.data_dir, filename)
                        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                        
                        if file_time < cutoff_date:
                            os.remove(filepath)
                            backup_files_deleted += 1
                            
            print(f"Cleaned {backup_files_deleted} old backup files")
            
            return {
                'trades_deleted': trades_deleted,
                'metrics_deleted': metrics_deleted,
                'backup_files_deleted': backup_files_deleted
            }
            
        except Exception as e:
            print(f"Error cleaning old data: {str(e)}")
            return None
            
    def optimize_database(self):
        """
        Optimize database performance
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create indexes for better performance
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
                "CREATE INDEX IF NOT EXISTS idx_trades_pnl ON trades(pnl)",
                "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)",
                "CREATE INDEX IF NOT EXISTS idx_training_start_time ON training_sessions(start_time)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
                
            # Vacuum database to reclaim space
            cursor.execute("VACUUM")
            
            conn.commit()
            conn.close()
            
            print("Database optimized successfully")
            return True
            
        except Exception as e:
            print(f"Error optimizing database: {str(e)}")
            return False
            
    def validate_data_integrity(self):
        """
        Validate data integrity and consistency
        """
        try:
            issues = []
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for negative volumes
            cursor.execute("SELECT COUNT(*) FROM trades WHERE volume <= 0")
            if cursor.fetchone()[0] > 0:
                issues.append("Found trades with invalid volume")
                
            # Check for missing timestamps
            cursor.execute("SELECT COUNT(*) FROM trades WHERE timestamp IS NULL")
            if cursor.fetchone()[0] > 0:
                issues.append("Found trades with missing timestamps")
                
            # Check for extremely large PnL values (potential data corruption)
            cursor.execute("SELECT COUNT(*) FROM trades WHERE ABS(pnl) > 100000")
            if cursor.fetchone()[0] > 0:
                issues.append("Found trades with extremely large PnL values")
                
            # Check performance metrics consistency
            cursor.execute("SELECT COUNT(*) FROM performance_metrics WHERE balance < 0")
            if cursor.fetchone()[0] > 0:
                issues.append("Found performance records with negative balance")
                
            conn.close()
            
            if issues:
                print("Data integrity issues found:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("Data integrity validation passed")
                
            return len(issues) == 0
            
        except Exception as e:
            print(f"Error validating data integrity: {str(e)}")
            return False
            
    def get_recovery_performance(self):
        """
        Analyze recovery system performance
        """
        try:
            trade_df = self.get_trade_history(days=365)  # Get full year
            
            if trade_df.empty:
                return {}
                
            # Filter recovery trades
            recovery_trades = trade_df[trade_df['recovery_level'] > 0]
            
            if recovery_trades.empty:
                return {'message': 'No recovery trades found'}
                
            recovery_stats = {}
            
            # Basic recovery statistics
            recovery_stats['total_recovery_trades'] = len(recovery_trades)
            recovery_stats['recovery_success_rate'] = len(recovery_trades[recovery_trades['pnl'] > 0]) / len(recovery_trades)
            
            # Recovery by type
            recovery_by_type = recovery_trades.groupby('recovery_type').agg({
                'pnl': ['count', 'sum', 'mean'],
                'recovery_level': 'mean'
            })
            recovery_stats['by_type'] = recovery_by_type.to_dict()
            
            # Recovery by level
            recovery_by_level = recovery_trades.groupby('recovery_level').agg({
                'pnl': ['count', 'sum', 'mean']
            })
            recovery_stats['by_level'] = recovery_by_level.to_dict()
            
            # Average recovery time
            recovery_stats['average_recovery_time'] = recovery_trades['duration_minutes'].mean()
            
            # Recovery efficiency (profit per recovery level)
            recovery_stats['efficiency'] = recovery_trades['pnl'].sum() / recovery_trades['recovery_level'].sum()
            
            return recovery_stats
            
        except Exception as e:
            print(f"Error analyzing recovery performance: {str(e)}")
            return {}
            
    def generate_daily_report(self, date: datetime = None):
        """
        Generate daily trading report
        """
        try:
            if date is None:
                date = datetime.now().date()
                
            # Get trades for the day
            start_time = datetime.combine(date, datetime.min.time())
            end_time = datetime.combine(date, datetime.max.time())
            
            conn = sqlite3.connect(self.db_path)
            
            trades_query = '''
                SELECT * FROM trades 
                WHERE timestamp BETWEEN ? AND ?
            '''
            trade_df = pd.read_sql_query(trades_query, conn, params=[start_time, end_time])
            
            # Get performance metrics for the day
            performance_query = '''
                SELECT * FROM performance_metrics 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp
            '''
            performance_df = pd.read_sql_query(performance_query, conn, params=[start_time, end_time])
            
            conn.close()
            
            # Generate report
            report = {
                'date': date.isoformat(),
                'trades': {
                    'total': len(trade_df),
                    'winning': len(trade_df[trade_df['pnl'] > 0]) if not trade_df.empty else 0,
                    'losing': len(trade_df[trade_df['pnl'] < 0]) if not trade_df.empty else 0,
                    'total_pnl': trade_df['pnl'].sum() if not trade_df.empty else 0,
                    'best_trade': trade_df['pnl'].max() if not trade_df.empty else 0,
                    'worst_trade': trade_df['pnl'].min() if not trade_df.empty else 0
                }
            }
            
            if not performance_df.empty:
                report['performance'] = {
                    'starting_balance': performance_df['balance'].iloc[0],
                    'ending_balance': performance_df['balance'].iloc[-1],
                    'max_drawdown': performance_df['drawdown'].max(),
                    'max_positions': performance_df['open_positions'].max()
                }
                
            # Save report
            report_filename = f"daily_report_{date.strftime('%Y%m%d')}.json"
            report_path = os.path.join(self.data_dir, report_filename)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4, default=str)
                
            print(f"Daily report generated: {report_path}")
            return report
            
        except Exception as e:
            print(f"Error generating daily report: {str(e)}")
            return {}
            
    def get_system_health(self):
        """
        Check overall system health and data status
        """
        try:
            health = {
                'timestamp': datetime.now().isoformat(),
                'database_size_mb': 0,
                'data_age_days': 0,
                'integrity_ok': False,
                'backup_exists': False,
                'storage_usage': {}
            }
            
            # Database size
            if os.path.exists(self.db_path):
                db_size = os.path.getsize(self.db_path)
                health['database_size_mb'] = db_size / (1024 * 1024)
                
            # Data age
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(timestamp) FROM trades")
            latest_trade = cursor.fetchone()[0]
            
            if latest_trade:
                latest_date = datetime.fromisoformat(latest_trade.replace('Z', '+00:00'))
                health['data_age_days'] = (datetime.now() - latest_date).days
                
            conn.close()
            
            # Data integrity
            health['integrity_ok'] = self.validate_data_integrity()
            
            # Check for backups
            backup_files = [f for f in os.listdir(self.data_dir) 
                          if f.startswith('backup_') and f.endswith('.db')]
            health['backup_exists'] = len(backup_files) > 0
            health['backup_count'] = len(backup_files)
            
            # Storage usage
            total_size = 0
            for filename in os.listdir(self.data_dir):
                filepath = os.path.join(self.data_dir, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
                    
            health['storage_usage'] = {
                'total_mb': total_size / (1024 * 1024),
                'files_count': len(os.listdir(self.data_dir))
            }
            
            return health
            
        except Exception as e:
            print(f"Error checking system health: {str(e)}")
            return {'error': str(e)}
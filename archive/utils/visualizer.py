# utils/visualizer.py - Data Visualization Utilities
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# แก้ไข import สำหรับ matplotlib เวอร์ชั่นใหม่
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvasTkinter
except ImportError:
    try:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkinter
    except ImportError:
        print("Warning: Cannot import matplotlib tkinter backend")
        FigureCanvasTkinter = None
        
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import tkinter as tk

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Visualizer:
    """
    Utility class for creating trading visualizations and charts
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'profit': '#2E8B57',      # Sea Green
            'loss': '#DC143C',        # Crimson
            'neutral': '#4682B4',     # Steel Blue
            'recovery': '#FF8C00',    # Dark Orange
            'background': '#F8F8FF'   # Ghost White
        }
        
    def create_pnl_chart(self, trade_data: pd.DataFrame, title: str = "PnL Performance"):
        """
        Create PnL performance chart
        """
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 1])
            
            if trade_data.empty:
                ax1.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax1.transAxes)
                ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax2.transAxes)
                return fig
                
            # Ensure timestamp is datetime
            if 'timestamp' in trade_data.columns:
                trade_data['timestamp'] = pd.to_datetime(trade_data['timestamp'])
                trade_data = trade_data.sort_values('timestamp')
                
            # Calculate cumulative PnL
            trade_data['cumulative_pnl'] = trade_data['pnl'].cumsum()
            
            # Main PnL chart
            ax1.plot(trade_data['timestamp'], trade_data['cumulative_pnl'], 
                    linewidth=2, color=self.colors['neutral'], label='Cumulative PnL')
            
            # Color positive and negative areas
            ax1.fill_between(trade_data['timestamp'], trade_data['cumulative_pnl'], 0,
                           where=(trade_data['cumulative_pnl'] >= 0), color=self.colors['profit'], alpha=0.3)
            ax1.fill_between(trade_data['timestamp'], trade_data['cumulative_pnl'], 0,
                           where=(trade_data['cumulative_pnl'] < 0), color=self.colors['loss'], alpha=0.3)
            
            # Mark recovery trades
            recovery_trades = trade_data[trade_data['recovery_level'] > 0]
            if not recovery_trades.empty:
                ax1.scatter(recovery_trades['timestamp'], recovery_trades['cumulative_pnl'],
                          color=self.colors['recovery'], s=50, alpha=0.7, label='Recovery Trades')
                
            ax1.set_title(title, fontsize=14, fontweight='bold')
            ax1.set_ylabel('Cumulative PnL ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Individual trade PnL bars
            colors = [self.colors['profit'] if pnl >= 0 else self.colors['loss'] for pnl in trade_data['pnl']]
            ax2.bar(trade_data['timestamp'], trade_data['pnl'], color=colors, alpha=0.7, width=0.8)
            
            ax2.set_xlabel('Time', fontsize=12)
            ax2.set_ylabel('Individual PnL ($)', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating PnL chart: {str(e)}")
            return None
            
    def create_drawdown_chart(self, performance_data: pd.DataFrame):
        """
        Create drawdown analysis chart
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            if performance_data.empty:
                ax.text(0.5, 0.5, 'No performance data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return fig
                
            # Ensure timestamp is datetime
            performance_data['timestamp'] = pd.to_datetime(performance_data['timestamp'])
            performance_data = performance_data.sort_values('timestamp')
            
            # Plot equity curve
            ax.plot(performance_data['timestamp'], performance_data['equity'], 
                   linewidth=2, label='Account Equity', color=self.colors['neutral'])
            
            # Plot balance line
            ax.plot(performance_data['timestamp'], performance_data['balance'], 
                   linewidth=2, label='Account Balance', color=self.colors['profit'], linestyle='--')
            
            # Calculate and plot drawdown
            if 'drawdown' in performance_data.columns:
                ax2 = ax.twinx()
                ax2.fill_between(performance_data['timestamp'], performance_data['drawdown'], 0,
                               color=self.colors['loss'], alpha=0.3, label='Drawdown')
                ax2.set_ylabel('Drawdown ($)', fontsize=12, color=self.colors['loss'])
                ax2.tick_params(axis='y', labelcolor=self.colors['loss'])
                
            ax.set_title('Account Performance & Drawdown', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel('Account Value ($)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating drawdown chart: {str(e)}")
            return None
            
    def create_recovery_analysis(self, trade_data: pd.DataFrame):
        """
        Create recovery system analysis charts
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            if trade_data.empty:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
                return fig
                
            recovery_trades = trade_data[trade_data['recovery_level'] > 0]
            
            # Recovery success rate by level
            if not recovery_trades.empty:
                recovery_stats = recovery_trades.groupby('recovery_level').agg({
                    'pnl': ['count', lambda x: (x > 0).sum()]
                })
                recovery_stats.columns = ['total', 'successful']
                recovery_stats['success_rate'] = recovery_stats['successful'] / recovery_stats['total']
                
                ax1.bar(recovery_stats.index, recovery_stats['success_rate'], 
                       color=self.colors['recovery'], alpha=0.7)
                ax1.set_title('Recovery Success Rate by Level', fontweight='bold')
                ax1.set_xlabel('Recovery Level')
                ax1.set_ylabel('Success Rate')
                ax1.set_ylim(0, 1)
                ax1.grid(True, alpha=0.3)
                
                # Recovery PnL distribution
                ax2.hist(recovery_trades['pnl'], bins=20, color=self.colors['recovery'], alpha=0.7, edgecolor='black')
                ax2.axvline(recovery_trades['pnl'].mean(), color='red', linestyle='--', label='Average')
                ax2.set_title('Recovery Trade PnL Distribution', fontweight='bold')
                ax2.set_xlabel('PnL ($)')
                ax2.set_ylabel('Frequency')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Recovery type performance
                if 'recovery_type' in recovery_trades.columns:
                    type_stats = recovery_trades.groupby('recovery_type').agg({
                        'pnl': ['count', 'sum', 'mean']
                    })
                    type_stats.columns = ['count', 'total_pnl', 'avg_pnl']
                    
                    ax3.bar(type_stats.index, type_stats['avg_pnl'], 
                           color=[self.colors['profit'] if x > 0 else self.colors['loss'] for x in type_stats['avg_pnl']])
                    ax3.set_title('Average PnL by Recovery Type', fontweight='bold')
                    ax3.set_xlabel('Recovery Type')
                    ax3.set_ylabel('Average PnL ($)')
                    ax3.grid(True, alpha=0.3)
                    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
                    
                # Recovery time analysis
                if 'duration_minutes' in recovery_trades.columns:
                    recovery_trades['duration_hours'] = recovery_trades['duration_minutes'] / 60
                    ax4.scatter(recovery_trades['recovery_level'], recovery_trades['duration_hours'],
                              c=recovery_trades['pnl'], cmap='RdYlGn', alpha=0.7)
                    ax4.set_title('Recovery Time vs Level (Color = PnL)', fontweight='bold')
                    ax4.set_xlabel('Recovery Level')
                    ax4.set_ylabel('Duration (Hours)')
                    ax4.grid(True, alpha=0.3)
                    
                    # Add colorbar
                    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
                    cbar.set_label('PnL ($)')
                    
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating recovery analysis: {str(e)}")
            return None
            
    def create_trading_statistics_dashboard(self, stats: Dict):
        """
        Create comprehensive trading statistics dashboard
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Win/Loss pie chart
            if stats.get('winning_trades', 0) + stats.get('losing_trades', 0) > 0:
                sizes = [stats.get('winning_trades', 0), stats.get('losing_trades', 0)]
                labels = ['Winning Trades', 'Losing Trades']
                colors = [self.colors['profit'], self.colors['loss']]
                
                ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax1.set_title(f"Win Rate: {stats.get('win_rate', 0):.1%}", fontweight='bold')
                
            # PnL metrics bar chart
            pnl_metrics = {
                'Total PnL': stats.get('total_pnl', 0),
                'Average Win': stats.get('average_win', 0),
                'Average Loss': stats.get('average_loss', 0),
                'Best Trade': stats.get('best_trade', 0),
                'Worst Trade': stats.get('worst_trade', 0)
            }
            
            bars = ax2.bar(pnl_metrics.keys(), pnl_metrics.values(),
                          color=[self.colors['profit'] if v >= 0 else self.colors['loss'] for v in pnl_metrics.values()])
            ax2.set_title('PnL Metrics', fontweight='bold')
            ax2.set_ylabel('Amount ($)')
            ax2.grid(True, alpha=0.3)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, pnl_metrics.values()):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 * max(abs(v) for v in pnl_metrics.values())),
                        f'${value:.2f}', ha='center', va='bottom' if value >= 0 else 'top')
                        
            # Risk metrics
            risk_data = {
                'Profit Factor': stats.get('profit_factor', 0),
                'Max Drawdown': stats.get('max_drawdown', 0),
                'Recovery Rate': stats.get('recovery_success_rate', 0)
            }
            
            # Create horizontal bar chart for risk metrics
            y_pos = np.arange(len(risk_data))
            values = list(risk_data.values())
            
            bars = ax3.barh(y_pos, values, color=self.colors['neutral'], alpha=0.7)
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(risk_data.keys())
            ax3.set_title('Risk Metrics', fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, values)):
                ax3.text(value + 0.01 * max(abs(v) for v in values), bar.get_y() + bar.get_height()/2,
                        f'{value:.2f}', ha='left', va='center')
                        
            # Time analysis
            time_data = {
                'Total Trades': stats.get('total_trades', 0),
                'Recovery Trades': stats.get('recovery_trades', 0),
                'Avg Duration (h)': stats.get('average_trade_duration', 0)
            }
            
            ax4.bar(time_data.keys(), time_data.values(), color=self.colors['recovery'], alpha=0.7)
            ax4.set_title('Trading Activity', fontweight='bold')
            ax4.set_ylabel('Count / Hours')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating statistics dashboard: {str(e)}")
            return None
            
    def create_rl_training_progress(self, training_data: List[Dict]):
        """
        Create RL training progress visualization
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            if not training_data:
                for ax in [ax1, ax2, ax3, ax4]:
                    ax.text(0.5, 0.5, 'No training data available', 
                           ha='center', va='center', transform=ax.transAxes)
                return fig
                
            df = pd.DataFrame(training_data)
            
            # Training rewards over time
            if 'episode_rewards' in df.columns:
                episodes = range(len(df))
                ax1.plot(episodes, df['episode_rewards'], color=self.colors['neutral'], alpha=0.7)
                
                # Add moving average
                window = min(50, len(df) // 10)
                if window > 1:
                    moving_avg = df['episode_rewards'].rolling(window=window).mean()
                    ax1.plot(episodes, moving_avg, color=self.colors['profit'], linewidth=2, label=f'MA({window})')
                    ax1.legend()
                    
                ax1.set_title('Training Rewards Progress', fontweight='bold')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Reward')
                ax1.grid(True, alpha=0.3)
                
            # Episode lengths
            if 'episode_lengths' in df.columns:
                ax2.plot(df['episode_lengths'], color=self.colors['recovery'], alpha=0.7)
                ax2.set_title('Episode Lengths', fontweight='bold')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Steps')
                ax2.grid(True, alpha=0.3)
                
            # Learning rate schedule
            if 'learning_rate' in df.columns:
                ax3.plot(df['learning_rate'], color=self.colors['loss'], linewidth=2)
                ax3.set_title('Learning Rate Schedule', fontweight='bold')
                ax3.set_xlabel('Episode')
                ax3.set_ylabel('Learning Rate')
                ax3.set_yscale('log')
                ax3.grid(True, alpha=0.3)
                
            # Loss values
            if 'loss' in df.columns:
                ax4.plot(df['loss'], color=self.colors['neutral'], alpha=0.7)
                ax4.set_title('Training Loss', fontweight='bold')
                ax4.set_xlabel('Episode')
                ax4.set_ylabel('Loss')
                ax4.grid(True, alpha=0.3)
                
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating training progress chart: {str(e)}")
            return None
            
    def create_market_analysis_chart(self, market_data: pd.DataFrame, positions: List[Dict] = None):
        """
        Create market analysis with positions overlay
        """
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] + 2), 
                                              height_ratios=[3, 1, 1])
            
            if market_data.empty:
                ax1.text(0.5, 0.5, 'No market data available', 
                        ha='center', va='center', transform=ax1.transAxes)
                return fig
                
            # Ensure timestamp is datetime
            if 'time' in market_data.columns:
                market_data['time'] = pd.to_datetime(market_data['time'])
                market_data = market_data.sort_values('time')
                
            # Candlestick chart (simplified)
            for i, row in market_data.iterrows():
                color = self.colors['profit'] if row['close'] >= row['open'] else self.colors['loss']
                
                # Draw high-low line
                ax1.plot([row['time'], row['time']], [row['low'], row['high']], 
                        color='black', linewidth=1, alpha=0.7)
                
                # Draw open-close box
                height = abs(row['close'] - row['open'])
                bottom = min(row['open'], row['close'])
                
                ax1.bar(row['time'], height, bottom=bottom, width=timedelta(minutes=1), 
                       color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
                       
            # Add moving averages
            if len(market_data) >= 20:
                sma_20 = market_data['close'].rolling(20).mean()
                ax1.plot(market_data['time'], sma_20, color='blue', linewidth=1, 
                        alpha=0.8, label='SMA(20)')
                        
            if len(market_data) >= 50:
                sma_50 = market_data['close'].rolling(50).mean()
                ax1.plot(market_data['time'], sma_50, color='red', linewidth=1, 
                        alpha=0.8, label='SMA(50)')
                        
            # Mark positions
            if positions:
                for pos in positions:
                    if pos.get('type') == 0:  # Buy position
                        ax1.scatter(pd.to_datetime(pos.get('time_open', datetime.now())), 
                                  pos.get('price_open', 0), color='green', marker='^', s=100, 
                                  alpha=0.8, label='Buy' if 'Buy' not in ax1.get_legend_handles_labels()[1] else "")
                    else:  # Sell position
                        ax1.scatter(pd.to_datetime(pos.get('time_open', datetime.now())), 
                                  pos.get('price_open', 0), color='red', marker='v', s=100, 
                                  alpha=0.8, label='Sell' if 'Sell' not in ax1.get_legend_handles_labels()[1] else "")
                        
            ax1.set_title('Market Analysis with Positions', fontweight='bold')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume chart
            if 'volume' in market_data.columns:
                ax2.bar(market_data['time'], market_data['volume'], 
                       color=self.colors['neutral'], alpha=0.6, width=timedelta(minutes=1))
                ax2.set_ylabel('Volume')
                ax2.grid(True, alpha=0.3)
                
            # RSI indicator
            if len(market_data) >= 14:
                rsi = self.calculate_rsi(market_data['close'])
                ax3.plot(market_data['time'], rsi, color=self.colors['recovery'], linewidth=2)
                ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
                ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
                ax3.set_ylabel('RSI')
                ax3.set_ylim(0, 100)
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
            # Format x-axis
            for ax in [ax1, ax2, ax3]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
            ax3.set_xlabel('Time')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating market analysis chart: {str(e)}")
            return None
            
    def calculate_rsi(self, prices: pd.Series, period: int = 14):
        """
        Calculate RSI indicator
        """
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            print(f"Error calculating RSI: {str(e)}")
            return pd.Series()
            
    def create_correlation_heatmap(self, features_data: pd.DataFrame):
        """
        Create correlation heatmap for features analysis
        """
        try:
            fig, ax = plt.subplots(figsize=self.figsize)
            
            if features_data.empty:
                ax.text(0.5, 0.5, 'No features data available', 
                       ha='center', va='center', transform=ax.transAxes)
                return fig
                
            # Calculate correlation matrix
            corr_matrix = features_data.corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
                       square=True, ax=ax, cbar_kws={'shrink': 0.8})
            
            ax.set_title('Features Correlation Matrix', fontweight='bold')
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"Error creating correlation heatmap: {str(e)}")
            return None
            
    def create_risk_analysis(self, trade_data: pd.DataFrame, performance_data: pd.DataFrame):
        """
        Create comprehensive risk analysis dashboard
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Monthly returns distribution
            if not trade_data.empty:
                trade_data['timestamp'] = pd.to_datetime(trade_data['timestamp'])
                monthly_returns = trade_data.groupby(trade_data['timestamp'].dt.to_period('M'))['pnl'].sum()
                
                ax1.hist(monthly_returns, bins=20, color=self.colors['neutral'], alpha=0.7, edgecolor='black')
                ax1.axvline(monthly_returns.mean(), color='red', linestyle='--', label='Mean')
                ax1.axvline(0, color='black', linestyle='-', alpha=0.5)
                ax1.set_title('Monthly Returns Distribution', fontweight='bold')
                ax1.set_xlabel('Monthly PnL ($)')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
            # Risk-Return scatter
            if not performance_data.empty:
                performance_data['timestamp'] = pd.to_datetime(performance_data['timestamp'])
                performance_data['returns'] = performance_data['equity'].pct_change()
                
                # Calculate rolling statistics
                window = 30
                rolling_return = performance_data['returns'].rolling(window).mean() * 252  # Annualized
                rolling_vol = performance_data['returns'].rolling(window).std() * np.sqrt(252)  # Annualized
                
                scatter = ax2.scatter(rolling_vol, rolling_return, 
                                    c=performance_data['timestamp'].iloc[window:].astype(int), 
                                    cmap='viridis', alpha=0.6)
                ax2.set_title('Risk-Return Profile', fontweight='bold')
                ax2.set_xlabel('Volatility (Annualized)')
                ax2.set_ylabel('Return (Annualized)')
                ax2.grid(True, alpha=0.3)
                
                # Add colorbar for time
                cbar = plt.colorbar(scatter, ax=ax2)
                cbar.set_label('Time')
                
            # Drawdown duration analysis
            if not performance_data.empty and 'drawdown' in performance_data.columns:
                # Find drawdown periods
                is_drawdown = performance_data['drawdown'] > 0
                drawdown_periods = []
                start_idx = None
                
                for i, in_dd in enumerate(is_drawdown):
                    if in_dd and start_idx is None:
                        start_idx = i
                    elif not in_dd and start_idx is not None:
                        duration = i - start_idx
                        max_dd = performance_data['drawdown'].iloc[start_idx:i].max()
                        drawdown_periods.append({'duration': duration, 'max_drawdown': max_dd})
                        start_idx = None
                        
                if drawdown_periods:
                    durations = [dd['duration'] for dd in drawdown_periods]
                    max_dds = [dd['max_drawdown'] for dd in drawdown_periods]
                    
                    ax3.scatter(durations, max_dds, color=self.colors['loss'], alpha=0.7)
                    ax3.set_title('Drawdown Duration vs Magnitude', fontweight='bold')
                    ax3.set_xlabel('Duration (periods)')
                    ax3.set_ylabel('Max Drawdown ($)')
                    ax3.grid(True, alpha=0.3)
                    
            # Value at Risk (VaR) analysis
            if not trade_data.empty:
                returns = trade_data['pnl']
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
                
                ax4.hist(returns, bins=50, color=self.colors['neutral'], alpha=0.7, edgecolor='black')
                ax4.axvline(var_95, color='orange', linestyle='--', label=f'VaR 95%: ${var_95:.2f}')
                ax4.axvline(var_99, color='red', linestyle='--', label=f'VaR 99%: ${var_99:.2f}')
                ax4.set_title('Value at Risk Analysis', fontweight='bold')
                ax4.set_xlabel('Trade PnL ($)')
                ax4.set_ylabel('Frequency')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating risk analysis: {str(e)}")
            return None
            
    def create_realtime_dashboard(self, current_data: Dict):
        """
        Create real-time trading dashboard
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Current positions pie chart
            if current_data.get('positions'):
                positions = current_data['positions']
                buy_volume = sum(pos.get('volume', 0) for pos in positions if pos.get('type') == 0)
                sell_volume = sum(pos.get('volume', 0) for pos in positions if pos.get('type') == 1)
                
                if buy_volume + sell_volume > 0:
                    sizes = [buy_volume, sell_volume]
                    labels = ['Buy Positions', 'Sell Positions']
                    colors = [self.colors['profit'], self.colors['loss']]
                    
                    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.2f', startangle=90)
                    ax1.set_title('Current Position Distribution', fontweight='bold')
                    
            # PnL gauge
            current_pnl = current_data.get('total_pnl', 0)
            max_range = max(abs(current_pnl) * 2, 1000)
            
            # Create semi-circular gauge
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            ax2.plot(theta, r, 'k-', linewidth=8)  # Outer arc
            
            # PnL indicator
            pnl_angle = np.pi * (0.5 + current_pnl / (2 * max_range))  # Map PnL to angle
            pnl_angle = max(0, min(np.pi, pnl_angle))  # Clamp to valid range
            
            ax2.plot([pnl_angle, pnl_angle], [0, 1], 
                    color=self.colors['profit'] if current_pnl >= 0 else self.colors['loss'], 
                    linewidth=4)
            
            ax2.text(np.pi/2, 0.5, f'${current_pnl:.2f}', ha='center', va='center', 
                    fontsize=14, fontweight='bold')
            ax2.set_xlim(0, np.pi)
            ax2.set_ylim(0, 1.2)
            ax2.set_title('Current PnL', fontweight='bold')
            ax2.axis('off')
            
            # Account metrics
            account_data = current_data.get('account_info', {})
            metrics = ['Balance', 'Equity', 'Margin', 'Free Margin']
            values = [
                account_data.get('balance', 0),
                account_data.get('equity', 0),
                account_data.get('margin', 0),
                account_data.get('margin_free', 0)
            ]
            
            bars = ax3.bar(metrics, values, color=self.colors['neutral'], alpha=0.7)
            ax3.set_title('Account Metrics', fontweight='bold')
            ax3.set_ylabel('Amount ($)')
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                        f'${value:.2f}', ha='center', va='bottom')
                        
            # Recovery status
            recovery_data = current_data.get('recovery_info', {})
            recovery_active = recovery_data.get('active', False)
            recovery_level = recovery_data.get('level', 0)
            
            if recovery_active:
                # Show recovery progress
                levels = list(range(1, 6))  # Show up to 5 levels
                colors = [self.colors['recovery'] if i <= recovery_level else 'lightgray' for i in levels]
                
                ax4.bar(levels, [1] * len(levels), color=colors, alpha=0.7)
                ax4.set_title(f'Recovery Active - Level {recovery_level}', fontweight='bold')
                ax4.set_xlabel('Recovery Level')
                ax4.set_ylabel('Status')
                ax4.set_ylim(0, 1.2)
            else:
                ax4.text(0.5, 0.5, 'No Recovery Active', ha='center', va='center', 
                        transform=ax4.transAxes, fontsize=14, fontweight='bold')
                ax4.set_title('Recovery Status', fontweight='bold')
                
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error creating realtime dashboard: {str(e)}")
            return None
            
    def embed_chart_in_tkinter(self, fig, parent_widget):
        """
        Embed matplotlib figure in tkinter widget
        """
        try:
            if FigureCanvasTkinter is None:
                print("Error: matplotlib tkinter backend not available")
                return None
                
            canvas = FigureCanvasTkinter(fig, parent_widget)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            return canvas
            
        except Exception as e:
            print(f"Error embedding chart: {str(e)}")
            return None
            
    def save_chart(self, fig, filename: str, dpi: int = 300):
        """
        Save chart to file
        """
        try:
            if not filename.endswith(('.png', '.jpg', '.pdf', '.svg')):
                filename += '.png'
                
            fig.savefig(filename, dpi=dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            print(f"Chart saved: {filename}")
            return filename
            
        except Exception as e:
            print(f"Error saving chart: {str(e)}")
            return None
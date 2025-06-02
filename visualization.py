import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinancialVisualizer:
    """Class for creating financial data visualizations"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_stock_price(self, data: pd.DataFrame, title: str = "Stock Price Analysis",
                        save_path: Optional[str] = None) -> None:
        """Plot stock price with volume and technical indicators"""
        
        fig, axes = plt.subplots(3, 1, figsize=(self.figsize[0], self.figsize[1] * 1.2))
        
        # Price and moving averages
        axes[0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
        if 'SMA_20' in data.columns:
            axes[0].plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
        if 'SMA_50' in data.columns:
            axes[0].plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7)
        
        axes[0].set_title(f'{title} - Price Chart')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume
        axes[1].bar(data.index, data['Volume'], alpha=0.6, color='gray')
        axes[1].set_title('Trading Volume')
        axes[1].set_ylabel('Volume')
        axes[1].grid(True, alpha=0.3)
        
        # Returns and volatility
        if 'Returns' in data.columns:
            axes[2].plot(data.index, data['Returns'], label='Daily Returns', alpha=0.7)
        if 'Volatility' in data.columns:
            ax2_twin = axes[2].twinx()
            ax2_twin.plot(data.index, data['Volatility'], 
                         label='Volatility (20d)', color='red', alpha=0.7)
            ax2_twin.set_ylabel('Volatility')
            ax2_twin.legend(loc='upper right')
        
        axes[2].set_title('Returns and Volatility')
        axes[2].set_ylabel('Returns')
        axes[2].set_xlabel('Date')
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Stock price chart saved to {save_path}")
        
        plt.show()
    
    def plot_option_surface(self, options_data: pd.DataFrame, option_type: str = 'call',
                           save_path: Optional[str] = None) -> None:
        """Plot 3D option price surface"""
        
        # Prepare data
        price_col = f'{option_type.title()}_Price'
        pivot_data = options_data.pivot_table(
            values=price_col, 
            index='Strike', 
            columns='Time_to_Maturity'
        )
        
        # Create 3D surface plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = np.meshgrid(pivot_data.columns, pivot_data.index)
        Z = pivot_data.values
        
        surface = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Time to Maturity')
        ax.set_ylabel('Strike Price')
        ax.set_zlabel(f'{option_type.title()} Option Price')
        ax.set_title(f'{option_type.title()} Option Price Surface')
        
        # Add colorbar
        fig.colorbar(surface, shrink=0.5, aspect=5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Option surface plot saved to {save_path}")
        
        plt.show()
    
    def plot_greeks_heatmap(self, options_data: pd.DataFrame, greek: str = 'delta',
                           option_type: str = 'call', save_path: Optional[str] = None) -> None:
        """Plot heatmap of option Greeks"""
        
        greek_col = f'{option_type.title()}_{greek.title()}'
        
        # Create pivot table
        pivot_data = options_data.pivot_table(
            values=greek_col,
            index='Strike',
            columns='Time_to_Maturity'
        )
        
        # Create heatmap
        plt.figure(figsize=self.figsize)
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0)
        plt.title(f'{option_type.title()} Option {greek.title()} Heatmap')
        plt.xlabel('Time to Maturity')
        plt.ylabel('Strike Price')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Greeks heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_monte_carlo_paths(self, paths: np.ndarray, title: str = "Monte Carlo Simulation",
                              n_paths_display: int = 100, save_path: Optional[str] = None) -> None:
        """Plot Monte Carlo simulation paths"""
        
        plt.figure(figsize=self.figsize)
        
        # Plot subset of paths
        n_display = min(n_paths_display, paths.shape[0])
        time_steps = np.arange(paths.shape[1])
        
        for i in range(n_display):
            plt.plot(time_steps, paths[i], alpha=0.3, linewidth=0.5, color='blue')
        
        # Plot mean path
        mean_path = np.mean(paths, axis=0)
        plt.plot(time_steps, mean_path, color='red', linewidth=2, label='Mean Path')
        
        # Plot confidence intervals
        std_path = np.std(paths, axis=0)
        plt.fill_between(time_steps, 
                        mean_path - 2*std_path, 
                        mean_path + 2*std_path, 
                        alpha=0.2, color='red', label='95% Confidence Interval')
        
        plt.title(f'{title} ({paths.shape[0]} paths)')
        plt.xlabel('Time Steps')
        plt.ylabel('Asset Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Monte Carlo paths plot saved to {save_path}")
        
        plt.show()
    
    def plot_portfolio_performance(self, portfolio_data: pd.DataFrame, 
                                 benchmark_data: Optional[pd.DataFrame] = None,
                                 save_path: Optional[str] = None) -> None:
        """Plot portfolio performance analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(portfolio_data.index, portfolio_data['Portfolio_Value'], 
                       label='Portfolio', linewidth=2)
        if benchmark_data is not None:
            axes[0, 0].plot(benchmark_data.index, benchmark_data['Portfolio_Value'], 
                           label='Benchmark', linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[0, 1].hist(portfolio_data['Portfolio_Returns'].dropna(), 
                       bins=50, alpha=0.7, density=True, label='Portfolio')
        if benchmark_data is not None:
            axes[0, 1].hist(benchmark_data['Portfolio_Returns'].dropna(), 
                           bins=50, alpha=0.5, density=True, label='Benchmark')
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Daily Returns')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = portfolio_data['Portfolio_Returns'].rolling(window=30).std() * np.sqrt(252)
        axes[1, 0].plot(portfolio_data.index[29:], rolling_vol.dropna(), 
                       label='Portfolio Volatility', linewidth=2)
        if benchmark_data is not None:
            benchmark_vol = benchmark_data['Portfolio_Returns'].rolling(window=30).std() * np.sqrt(252)
            axes[1, 0].plot(benchmark_data.index[29:], benchmark_vol.dropna(), 
                           label='Benchmark Volatility', linewidth=2, alpha=0.7)
        axes[1, 0].set_title('Rolling 30-Day Volatility')
        axes[1, 0].set_ylabel('Annualized Volatility')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Drawdown analysis
        portfolio_cumulative = (1 + portfolio_data['Portfolio_Returns']).cumprod()
        running_max = portfolio_cumulative.expanding().max()
        drawdown = (portfolio_cumulative - running_max) / running_max
        
        axes[1, 1].fill_between(portfolio_data.index, drawdown, 0, 
                               alpha=0.7, color='red', label='Drawdown')
        axes[1, 1].set_title('Portfolio Drawdown')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Portfolio performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_training_history(self, history: Dict, save_path: Optional[str] = None) -> None:
        """Plot model training history"""
        
        # Determine number of subplots based on available metrics
        metrics = list(history.keys())
        n_metrics = len(metrics)
        
        if n_metrics <= 2:
            fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
            if n_metrics == 1:
                axes = [axes]
        else:
            n_rows = (n_metrics + 1) // 2
            fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5*n_rows))
            axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                axes[i].plot(history[metric], label=f'Training {metric}', linewidth=2)
                
                # Plot validation metric if available
                val_metric = f'val_{metric}'
                if val_metric in history:
                    axes[i].plot(history[val_metric], label=f'Validation {metric}', linewidth=2)
                
                axes[i].set_title(f'{metric.title()} Over Epochs')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric.title())
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 title: str = "Predictions vs Actual",
                                 save_path: Optional[str] = None) -> None:
        """Plot predictions vs actual values"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(f'{title} - Scatter Plot')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Time series comparison (if applicable)
        time_steps = np.arange(len(y_true))
        axes[1].plot(time_steps, y_true, label='Actual', linewidth=2, alpha=0.8)
        axes[1].plot(time_steps, y_pred, label='Predicted', linewidth=2, alpha=0.8)
        axes[1].set_xlabel('Time Steps')
        axes[1].set_ylabel('Values')
        axes[1].set_title(f'{title} - Time Series')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, data: pd.DataFrame, title: str = "Correlation Matrix",
                               save_path: Optional[str] = None) -> None:
        """Plot correlation matrix heatmap"""
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=self.figsize)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdYlBu_r', center=0, square=True, cbar_kws={"shrink": .8})
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation matrix saved to {save_path}")
        
        plt.show()
    
    def plot_risk_return_scatter(self, returns: pd.DataFrame, 
                                title: str = "Risk-Return Analysis",
                                save_path: Optional[str] = None) -> None:
        """Plot risk-return scatter plot"""
        
        # Calculate metrics for each asset
        mean_returns = returns.mean() * 252  # Annualized
        volatilities = returns.std() * np.sqrt(252)  # Annualized
        
        plt.figure(figsize=self.figsize)
        
        # Scatter plot
        for i, asset in enumerate(returns.columns):
            plt.scatter(volatilities[asset], mean_returns[asset], 
                       s=100, alpha=0.7, label=asset, color=self.colors[i % len(self.colors)])
        
        plt.xlabel('Volatility (Annualized)')
        plt.ylabel('Expected Return (Annualized)')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add efficient frontier if multiple assets
        if len(returns.columns) > 1:
            self._add_efficient_frontier(returns)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Risk-return scatter plot saved to {save_path}")
        
        plt.show()
    
    def _add_efficient_frontier(self, returns: pd.DataFrame, n_portfolios: int = 100) -> None:
        """Add efficient frontier to risk-return plot"""
        
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        n_assets = len(returns.columns)
        
        # Generate random portfolio weights
        results = np.zeros((4, n_portfolios))
        
        for i in range(n_portfolios):
            # Random weights
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            # Portfolio return and risk
            portfolio_return = np.sum(weights * mean_returns)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Sharpe ratio
            sharpe_ratio = portfolio_return / portfolio_risk
            
            results[0, i] = portfolio_return
            results[1, i] = portfolio_risk
            results[2, i] = sharpe_ratio
            results[3, i] = i
        
        # Plot efficient frontier
        plt.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.5, s=10)
        plt.colorbar(label='Sharpe Ratio')
    
    def create_interactive_dashboard(self, data: Dict, save_path: Optional[str] = None) -> None:
        """Create interactive dashboard using Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Price Chart', 'Volume', 'Returns Distribution', 'Volatility'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Assuming data contains stock information
        if 'stock_data' in data:
            stock_data = data['stock_data']
            
            # Price chart
            fig.add_trace(
                go.Scatter(x=stock_data.index, y=stock_data['Close'], 
                          name='Close Price', line=dict(width=2)),
                row=1, col=1
            )
            
            # Volume
            fig.add_trace(
                go.Bar(x=stock_data.index, y=stock_data['Volume'], 
                      name='Volume', opacity=0.6),
                row=1, col=2
            )
            
            # Returns distribution
            fig.add_trace(
                go.Histogram(x=stock_data['Returns'].dropna(), 
                           name='Returns Distribution', nbinsx=50),
                row=2, col=1
            )
            
            # Volatility
            if 'Volatility' in stock_data.columns:
                fig.add_trace(
                    go.Scatter(x=stock_data.index, y=stock_data['Volatility'], 
                              name='Volatility', line=dict(color='red')),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Financial Data Dashboard",
            showlegend=True,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
            print(f"Interactive dashboard saved to {save_path}")
        
        fig.show()

# Example usage and testing
if __name__ == "__main__":
    print("Testing Financial Visualization Tools...")
    
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Sample stock data
    stock_data = pd.DataFrame({
        'Close': 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 252)),
        'Volume': np.random.randint(1000000, 5000000, 252),
        'Returns': np.random.normal(0.001, 0.02, 252),
        'Volatility': np.random.uniform(0.15, 0.35, 252)
    }, index=dates)
    
    stock_data['SMA_20'] = stock_data['Close'].rolling(20).mean()
    stock_data['SMA_50'] = stock_data['Close'].rolling(50).mean()
    
    # Initialize visualizer
    viz = FinancialVisualizer()
    
    # Test stock price plot
    print("\n1. Testing stock price visualization...")
    viz.plot_stock_price(stock_data, "Sample Stock Analysis", "results/stock_analysis.png")
    
    # Test Monte Carlo paths
    print("\n2. Testing Monte Carlo paths visualization...")
    paths = np.random.randn(1000, 100).cumsum(axis=1) + 100
    viz.plot_monte_carlo_paths(paths, "Sample Monte Carlo Paths", save_path="results/monte_carlo.png")
    
    # Test training history
    print("\n3. Testing training history visualization...")
    sample_history = {
        'loss': np.exp(-np.linspace(0, 3, 50)) + np.random.normal(0, 0.01, 50),
        'val_loss': np.exp(-np.linspace(0, 2.5, 50)) + np.random.normal(0, 0.02, 50),
        'mae': np.exp(-np.linspace(0, 2, 50)) + np.random.normal(0, 0.005, 50)
    }
    viz.plot_model_training_history(sample_history, "results/training_history.png")
    
    # Test prediction comparison
    print("\n4. Testing prediction vs actual visualization...")
    y_true = np.random.randn(100)
    y_pred = y_true + np.random.normal(0, 0.1, 100)  # Add some noise
    viz.plot_prediction_vs_actual(y_true, y_pred, "Sample Predictions", "results/predictions.png")
    
    # Test correlation matrix
    print("\n5. Testing correlation matrix...")
    sample_returns = pd.DataFrame(np.random.randn(252, 5), 
                                 columns=['Asset_A', 'Asset_B', 'Asset_C', 'Asset_D', 'Asset_E'])
    viz.plot_correlation_matrix(sample_returns, "Sample Correlation Matrix", "results/correlation.png")
    
    print("\nAll visualization tests completed successfully!")
    print("Check the 'results/' directory for saved plots.")
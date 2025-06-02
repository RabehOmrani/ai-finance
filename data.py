import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, Dict, List, Optional
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FinancialDataLoader:
    """Class for loading and preprocessing financial data"""
    
    def __init__(self):
        self.data_cache = {}
        
    def load_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load stock data from Yahoo Finance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'SPY')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        
        Returns:
            DataFrame with stock data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            # Calculate additional features
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            
            # Remove NaN values
            data = data.dropna()
            
            self.data_cache[symbol] = data
            return data
            
        except Exception as e:
            print(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_synthetic_options_data(self, stock_data: pd.DataFrame, 
                                      strikes: List[float], 
                                      maturities: List[float],
                                      risk_free_rate: float = 0.05) -> pd.DataFrame:
        """
        Generate synthetic options data using Black-Scholes model
        
        Args:
            stock_data: Stock price data
            strikes: List of strike prices
            maturities: List of time to maturity (in years)
            risk_free_rate: Risk-free interest rate
        
        Returns:
            DataFrame with options data
        """
        from utils import black_scholes_call, black_scholes_put, calculate_greeks
        
        options_data = []
        
        for idx, row in stock_data.iterrows():
            S = row['Close']
            sigma = row['Volatility'] if not pd.isna(row['Volatility']) else 0.2
            
            for K in strikes:
                for T in maturities:
                    if T > 0:  # Only positive time to maturity
                        # Calculate option prices
                        call_price = black_scholes_call(S, K, T, risk_free_rate, sigma)
                        put_price = black_scholes_put(S, K, T, risk_free_rate, sigma)
                        
                        # Calculate Greeks
                        call_greeks = calculate_greeks(S, K, T, risk_free_rate, sigma, 'call')
                        put_greeks = calculate_greeks(S, K, T, risk_free_rate, sigma, 'put')
                        
                        # Moneyness
                        moneyness = S / K
                        
                        options_data.append({
                            'Date': idx,
                            'Underlying_Price': S,
                            'Strike': K,
                            'Time_to_Maturity': T,
                            'Volatility': sigma,
                            'Risk_Free_Rate': risk_free_rate,
                            'Call_Price': call_price,
                            'Put_Price': put_price,
                            'Moneyness': moneyness,
                            'Call_Delta': call_greeks['delta'],
                            'Call_Gamma': call_greeks['gamma'],
                            'Call_Theta': call_greeks['theta'],
                            'Call_Vega': call_greeks['vega'],
                            'Call_Rho': call_greeks['rho'],
                            'Put_Delta': put_greeks['delta'],
                            'Put_Gamma': put_greeks['gamma'],
                            'Put_Theta': put_greeks['theta'],
                            'Put_Vega': put_greeks['vega'],
                            'Put_Rho': put_greeks['rho']
                        })
        
        return pd.DataFrame(options_data)
    
    def create_portfolio_data(self, symbols: List[str], start_date: str, end_date: str,
                            weights: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Create portfolio data from multiple assets
        
        Args:
            symbols: List of stock symbols
            start_date: Start date
            end_date: End date
            weights: Portfolio weights (equal weights if None)
        
        Returns:
            DataFrame with portfolio data
        """
        if weights is None:
            weights = [1.0 / len(symbols)] * len(symbols)
        
        portfolio_data = pd.DataFrame()
        
        for i, symbol in enumerate(symbols):
            stock_data = self.load_stock_data(symbol, start_date, end_date)
            if not stock_data.empty:
                portfolio_data[f'{symbol}_Price'] = stock_data['Close']
                portfolio_data[f'{symbol}_Returns'] = stock_data['Returns']
                portfolio_data[f'{symbol}_Weight'] = weights[i]
        
        # Calculate portfolio metrics
        if not portfolio_data.empty:
            # Portfolio returns
            return_cols = [col for col in portfolio_data.columns if 'Returns' in col]
            weight_cols = [col for col in portfolio_data.columns if 'Weight' in col]
            
            portfolio_returns = 0
            for i, ret_col in enumerate(return_cols):
                portfolio_returns += portfolio_data[ret_col] * weights[i]
            
            portfolio_data['Portfolio_Returns'] = portfolio_returns
            portfolio_data['Portfolio_Value'] = (1 + portfolio_data['Portfolio_Returns']).cumprod()
            portfolio_data['Portfolio_Volatility'] = portfolio_data['Portfolio_Returns'].rolling(window=20).std() * np.sqrt(252)
        
        return portfolio_data.dropna()

class SyntheticDataGenerator:
    """Generate synthetic financial data for testing"""
    
    @staticmethod
    def generate_gbm_paths(S0: float, mu: float, sigma: float, T: float, 
                          n_steps: int, n_paths: int) -> np.ndarray:
        """Generate Geometric Brownian Motion paths"""
        from utils import monte_carlo_paths
        return monte_carlo_paths(S0, mu, sigma, T, n_steps, n_paths)
    
    @staticmethod
    def generate_heston_paths(S0: float, v0: float, kappa: float, theta: float,
                            sigma: float, rho: float, r: float, T: float,
                            n_steps: int, n_paths: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate Heston model paths"""
        from utils import heston_monte_carlo
        return heston_monte_carlo(S0, v0, kappa, theta, sigma, rho, r, T, n_steps, n_paths)
    
    @staticmethod
    def generate_jump_diffusion_paths(S0: float, mu: float, sigma: float, 
                                    lambda_jump: float, mu_jump: float, sigma_jump: float,
                                    T: float, n_steps: int, n_paths: int) -> np.ndarray:
        """Generate jump-diffusion (Merton) model paths"""
        dt = T / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        
        for t in range(1, n_steps + 1):
            # Brownian motion component
            Z = np.random.standard_normal(n_paths)
            
            # Jump component
            N_jumps = np.random.poisson(lambda_jump * dt, n_paths)
            jump_sizes = np.zeros(n_paths)
            
            for i in range(n_paths):
                if N_jumps[i] > 0:
                    jumps = np.random.normal(mu_jump, sigma_jump, N_jumps[i])
                    jump_sizes[i] = np.sum(jumps)
            
            # Update paths
            paths[:, t] = paths[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + jump_sizes
            )
        
        return paths

class DataPreprocessor:
    """Preprocess financial data for machine learning"""
    
    def __init__(self):
        self.scalers = {}
    
    def prepare_lstm_data(self, data: pd.DataFrame, target_col: str, 
                         feature_cols: List[str], sequence_length: int,
                         train_ratio: float = 0.8) -> Dict:
        """
        Prepare data for LSTM training
        
        Args:
            data: Input DataFrame
            target_col: Target column name
            feature_cols: List of feature column names
            sequence_length: Length of input sequences
            train_ratio: Ratio of training data
        
        Returns:
            Dictionary with prepared data
        """
        from utils import normalize_data, create_sequences
        
        # Select features and target
        features = data[feature_cols].values
        target = data[target_col].values
        
        # Normalize features
        features_norm, feature_params = normalize_data(features, method='zscore')
        target_norm, target_params = normalize_data(target.reshape(-1, 1), method='zscore')
        target_norm = target_norm.flatten()
        
        # Store scaling parameters
        self.scalers['features'] = feature_params
        self.scalers['target'] = target_params
        
        # Combine features and target
        combined_data = np.column_stack([features_norm, target_norm])
        
        # Create sequences
        X, y = create_sequences(combined_data, sequence_length, target_column=-1)
        
        # Split into train and test
        split_idx = int(len(X) * train_ratio)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_params': feature_params,
            'target_params': target_params,
            'sequence_length': sequence_length
        }
    
    def prepare_options_data(self, options_data: pd.DataFrame) -> Dict:
        """Prepare options data for neural network training"""
        from utils import normalize_data
        
        # Define features for option pricing
        feature_cols = ['Underlying_Price', 'Strike', 'Time_to_Maturity', 
                       'Volatility', 'Risk_Free_Rate', 'Moneyness']
        
        target_cols = ['Call_Price', 'Put_Price']
        
        # Extract features and targets
        X = options_data[feature_cols].values
        y_call = options_data['Call_Price'].values
        y_put = options_data['Put_Price'].values
        
        # Normalize data
        X_norm, X_params = normalize_data(X, method='zscore')
        y_call_norm, y_call_params = normalize_data(y_call.reshape(-1, 1), method='zscore')
        y_put_norm, y_put_params = normalize_data(y_put.reshape(-1, 1), method='zscore')
        
        # Store scaling parameters
        self.scalers['options_features'] = X_params
        self.scalers['call_prices'] = y_call_params
        self.scalers['put_prices'] = y_put_params
        
        return {
            'X': X_norm,
            'y_call': y_call_norm.flatten(),
            'y_put': y_put_norm.flatten(),
            'feature_names': feature_cols,
            'scalers': self.scalers
        }

# Example usage and testing
if __name__ == "__main__":
    print("Testing Financial Data Loader...")
    
    # Initialize data loader
    loader = FinancialDataLoader()
    
    # Load sample stock data
    end_date = '2010-01-01'
    start_date = '2024-01-01'
    
    print(f"Loading SPY data from {start_date} to {end_date}")
    spy_data = loader.load_stock_data('SPY', start_date, end_date)
    
    if not spy_data.empty:
        print(f"Loaded {len(spy_data)} days of data")
        print(f"Columns: {list(spy_data.columns)}")
        print(f"Date range: {spy_data.index[0]} to {spy_data.index[-1]}")
        
        # Generate synthetic options data
        strikes = [spy_data['Close'].iloc[-1] * k for k in [0.9, 0.95, 1.0, 1.05, 1.1]]
        maturities = [0.25, 0.5, 1.0]  # 3 months, 6 months, 1 year
        
        print("\nGenerating synthetic options data...")
        options_data = loader.generate_synthetic_options_data(spy_data, strikes, maturities)
        print(f"Generated {len(options_data)} option contracts")
        
        # Test data preprocessing
        print("\nTesting data preprocessing...")
        preprocessor = DataPreprocessor()
        
        # Prepare LSTM data
        feature_cols = ['Close', 'Volume', 'Volatility', 'RSI']
        lstm_data = preprocessor.prepare_lstm_data(
            spy_data, 'Close', feature_cols, sequence_length=30
        )
        
        print(f"LSTM training data shape: {lstm_data['X_train'].shape}")
        print(f"LSTM test data shape: {lstm_data['X_test'].shape}")
        
        # Prepare options data
        options_prepared = preprocessor.prepare_options_data(options_data)
        print(f"Options features shape: {options_prepared['X'].shape}")
        
        print("\nData loading and preprocessing completed successfully!")
    
    else:
        print("Failed to load stock data. Using synthetic data for demonstration...")
        
        # Generate synthetic data
        generator = SyntheticDataGenerator()
        
        # GBM paths
        gbm_paths = generator.generate_gbm_paths(
            S0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=1000
        )
        print(f"Generated GBM paths shape: {gbm_paths.shape}")
        
        # Heston paths
        heston_S, heston_v = generator.generate_heston_paths(
            S0=100, v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, 
            rho=-0.7, r=0.05, T=1.0, n_steps=252, n_paths=1000
        )
        print(f"Generated Heston stock paths shape: {heston_S.shape}")
        print(f"Generated Heston variance paths shape: {heston_v.shape}")
        
        print("Synthetic data generation completed successfully!")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional, Union
import pickle
import os
from datetime import datetime
import logging

# Set up logging
def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('neural_finance')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    fh = logging.FileHandler(f'{log_dir}/neural_finance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    fh.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Financial utility functions
def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    Calculate Black-Scholes call option price
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration
        r: Risk-free rate
        sigma: Volatility
    
    Returns:
        Call option price
    """
    from scipy.stats import norm
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Calculate Black-Scholes put option price"""
    from scipy.stats import norm
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price

def heston_characteristic_function(phi: complex, S0: float, v0: float, kappa: float, 
                                 theta: float, sigma: float, rho: float, r: float, T: float) -> complex:
    """
    Heston model characteristic function for option pricing
    
    Args:
        phi: Complex frequency parameter
        S0: Initial stock price
        v0: Initial variance
        kappa: Mean reversion speed
        theta: Long-term variance
        sigma: Volatility of volatility
        rho: Correlation between stock and variance
        r: Risk-free rate
        T: Time to maturity
    
    Returns:
        Characteristic function value
    """
    xi = kappa - sigma * rho * phi * 1j
    d = np.sqrt(xi**2 + sigma**2 * (phi * 1j + phi**2))
    
    A1 = phi * 1j * (np.log(S0) + r * T)
    A2 = kappa * theta / sigma**2 * (xi - d) * T
    A3 = -kappa * theta / sigma**2 * np.log((1 - (xi - d) / (xi + d) * np.exp(-d * T)) / 2)
    A4 = -(phi * 1j + phi**2) * v0 / (xi + d) * (1 - np.exp(-d * T)) / (1 - (xi - d) / (xi + d) * np.exp(-d * T))
    
    return np.exp(A1 + A2 + A3 + A4)

def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Dict[str, float]:
    """
    Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho)
    
    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
    
    Returns:
        Dictionary containing Greeks
    """
    from scipy.stats import norm
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    # Gamma
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    # Theta
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    
    # Vega
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    
    # Rho
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

# Model evaluation metrics
def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error"""
    return np.mean((y_true - y_pred)**2)

def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe ratio"""
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown"""
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)

# Data preprocessing utilities
def normalize_data(data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Dict]:
    """
    Normalize data using specified method
    
    Args:
        data: Input data array
        method: 'minmax', 'zscore', or 'robust'
    
    Returns:
        Normalized data and scaling parameters
    """
    if method == 'minmax':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized = (data - min_val) / (max_val - min_val)
        params = {'min': min_val, 'max': max_val, 'method': method}
    
    elif method == 'zscore':
        mean_val = np.mean(data, axis=0)
        std_val = np.std(data, axis=0)
        normalized = (data - mean_val) / std_val
        params = {'mean': mean_val, 'std': std_val, 'method': method}
    
    elif method == 'robust':
        median_val = np.median(data, axis=0)
        mad_val = np.median(np.abs(data - median_val), axis=0)
        normalized = (data - median_val) / mad_val
        params = {'median': median_val, 'mad': mad_val, 'method': method}
    
    return normalized, params

def denormalize_data(normalized_data: np.ndarray, params: Dict) -> np.ndarray:
    """Denormalize data using stored parameters"""
    method = params['method']
    
    if method == 'minmax':
        return normalized_data * (params['max'] - params['min']) + params['min']
    elif method == 'zscore':
        return normalized_data * params['std'] + params['mean']
    elif method == 'robust':
        return normalized_data * params['mad'] + params['median']

def create_sequences(data: np.ndarray, sequence_length: int, target_column: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series modeling
    
    Args:
        data: Input data array
        sequence_length: Length of input sequences
        target_column: Column index for target variable
    
    Returns:
        Input sequences and targets
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[i + sequence_length, target_column])
    
    return np.array(X), np.array(y)

# Checkpoint and model saving utilities
def save_checkpoint(model_state: Dict, filepath: str) -> None:
    """Save model checkpoint"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model_state, f)

def load_checkpoint(filepath: str) -> Dict:
    """Load model checkpoint"""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_results(results: Dict, filepath: str) -> None:
    """Save results to file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if filepath.endswith('.pkl'):
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    elif filepath.endswith('.csv'):
        pd.DataFrame(results).to_csv(filepath, index=False)

# Monte Carlo simulation utilities
def monte_carlo_paths(S0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int) -> np.ndarray:
    """
    Generate Monte Carlo paths for geometric Brownian motion
    
    Args:
        S0: Initial stock price
        mu: Drift parameter
        sigma: Volatility
        T: Time horizon
        n_steps: Number of time steps
        n_paths: Number of simulation paths
    
    Returns:
        Array of simulated paths
    """
    dt = T / n_steps
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    
    for t in range(1, n_steps + 1):
        Z = np.random.standard_normal(n_paths)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    return paths

def heston_monte_carlo(S0: float, v0: float, kappa: float, theta: float, sigma: float, 
                      rho: float, r: float, T: float, n_steps: int, n_paths: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Monte Carlo paths using Heston stochastic volatility model
    
    Returns:
        Stock price paths and variance paths
    """
    dt = T / n_steps
    
    S_paths = np.zeros((n_paths, n_steps + 1))
    v_paths = np.zeros((n_paths, n_steps + 1))
    
    S_paths[:, 0] = S0
    v_paths[:, 0] = v0
    
    for t in range(1, n_steps + 1):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_paths)
        
        # Variance process (with Feller condition)
        v_paths[:, t] = np.maximum(
            v_paths[:, t-1] + kappa * (theta - v_paths[:, t-1]) * dt + 
            sigma * np.sqrt(v_paths[:, t-1] * dt) * Z2, 0
        )
        
        # Stock price process
        S_paths[:, t] = S_paths[:, t-1] * np.exp(
            (r - 0.5 * v_paths[:, t-1]) * dt + 
            np.sqrt(v_paths[:, t-1] * dt) * Z1
        )
    
    return S_paths, v_paths

print("Utils module loaded successfully!")
print("Available functions:")
print("- Financial: black_scholes_call/put, heston_characteristic_function, calculate_greeks")
print("- Evaluation: calculate_mse/mae/mape/r2, calculate_sharpe_ratio, calculate_max_drawdown")
print("- Data processing: normalize_data, create_sequences")
print("- Monte Carlo: monte_carlo_paths, heston_monte_carlo")
print("- Utilities: save_checkpoint, load_checkpoint, setup_logging")
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from utils import setup_logging, calculate_mse, calculate_mae, calculate_r2, save_checkpoint, save_results
from data import FinancialDataLoader, SyntheticDataGenerator, DataPreprocessor
from models import ModelFactory, LSTMPricePredictor, PhysicsInformedNN, NeuralSDE
from visualization import FinancialVisualizer

class NeuralFinanceExperiment:
    """Main experiment class for neural finance modeling"""
    
    def __init__(self, experiment_name: str = "neural_finance_experiment"):
        self.experiment_name = experiment_name
        self.logger = setup_logging()
        self.results_dir = f"results/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.create_directories()
        
        # Initialize components
        self.data_loader = FinancialDataLoader()
        self.data_preprocessor = DataPreprocessor()
        self.visualizer = FinancialVisualizer()
        
        # Storage for results
        self.results = {}
        self.models = {}
        
        self.logger.info(f"Initialized experiment: {experiment_name}")
    
    def create_directories(self):
        """Create necessary directories for results"""
        directories = [
            self.results_dir,
            f"{self.results_dir}/models",
            f"{self.results_dir}/plots",
            f"{self.results_dir}/data",
            f"{self.results_dir}/checkpoints"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def load_financial_data(self, symbols: list = ['SPY'], 
                           start_date: str = None, end_date: str = None):
        """Load financial data for experiments"""
        
        if start_date is None:
            start_date = '2019-01-01'
        if end_date is None:
            end_date = '2024-01-01'
        
        self.logger.info(f"Loading data for symbols: {symbols}")
        
        all_data = {}
        for symbol in symbols:
            try:
                data = self.data_loader.load_stock_data(symbol, start_date, end_date)
                if not data.empty:
                    all_data[symbol] = data
                    self.logger.info(f"Loaded {len(data)} days of data for {symbol}")
                else:
                    self.logger.warning(f"No data loaded for {symbol}")
            except Exception as e:
                self.logger.error(f"Error loading data for {symbol}: {e}")
        
        self.financial_data = all_data
        return all_data
    
    def generate_synthetic_data(self):
        """Generate synthetic financial data for testing"""
        
        self.logger.info("Generating synthetic financial data...")
        
        generator = SyntheticDataGenerator()
        
        # Generate GBM paths
        gbm_paths = generator.generate_gbm_paths(
            S0=100, mu=0.05, sigma=0.2, T=1.0, n_steps=252, n_paths=1000
        )
        
        # Generate Heston paths
        heston_S, heston_v = generator.generate_heston_paths(
            S0=100, v0=0.04, kappa=2.0, theta=0.04, sigma=0.3,
            rho=-0.7, r=0.05, T=1.0, n_steps=252, n_paths=1000
        )
        
        # Generate jump-diffusion paths
        jump_paths = generator.generate_jump_diffusion_paths(
            S0=100, mu=0.05, sigma=0.2, lambda_jump=0.1, 
            mu_jump=-0.05, sigma_jump=0.1, T=1.0, n_steps=252, n_paths=1000
        )
        
        self.synthetic_data = {
            'gbm_paths': gbm_paths,
            'heston_stock': heston_S,
            'heston_variance': heston_v,
            'jump_diffusion': jump_paths
        }
        
        self.logger.info("Synthetic data generation completed")
        return self.synthetic_data
    
    def experiment_lstm_prediction(self, symbol: str = 'SPY'):
        """Run LSTM stock price prediction experiment"""
        
        self.logger.info(f"Starting LSTM prediction experiment for {symbol}")
        
        if symbol not in self.financial_data:
            self.logger.error(f"No data available for {symbol}")
            return None
        
        # Prepare data
        stock_data = self.financial_data[symbol]
        feature_cols = ['Close', 'Volume', 'Returns', 'Volatility']
        
        # Remove any columns that don't exist
        available_cols = [col for col in feature_cols if col in stock_data.columns]
        
        lstm_data = self.data_preprocessor.prepare_lstm_data(
            stock_data, 'Close', available_cols, sequence_length=30
        )
        
        # Build and train LSTM model - FIX HERE: Use the actual number of features from the data
        n_features = lstm_data['X_train'].shape[2]  # Get the actual number of features
        
        lstm_model = LSTMPricePredictor(
            sequence_length=30, 
            n_features=n_features  # Use the actual number of features
        )
        
        lstm_model.build_model(
            lstm_units=[64, 32],
            dense_units=[16],
            dropout_rate=0.2,
            learning_rate=0.001
        )

        
        # Train model
        history = lstm_model.train(
            lstm_data['X_train'], lstm_data['y_train'],
            X_val=lstm_data['X_test'], y_val=lstm_data['y_test'],
            epochs=50, batch_size=32, verbose=1
        )
        
        # Make predictions
        y_pred_train = lstm_model.predict(lstm_data['X_train'])
        y_pred_test = lstm_model.predict(lstm_data['X_test'])
        
        # Calculate metrics
        train_mse = calculate_mse(lstm_data['y_train'], y_pred_train.flatten())
        test_mse = calculate_mse(lstm_data['y_test'], y_pred_test.flatten())
        train_r2 = calculate_r2(lstm_data['y_train'], y_pred_train.flatten())
        test_r2 = calculate_r2(lstm_data['y_test'], y_pred_test.flatten())
        
        # Store results
        lstm_results = {
            'model_type': 'LSTM',
            'symbol': symbol,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'history': history.history,
            'predictions_train': y_pred_train,
            'predictions_test': y_pred_test,
            'actual_train': lstm_data['y_train'],
            'actual_test': lstm_data['y_test']
        }
        
        self.results[f'lstm_{symbol}'] = lstm_results
        self.models[f'lstm_{symbol}'] = lstm_model
        
        # Save model and results
        model_path = f"{self.results_dir}/models/lstm_{symbol}.h5"
        lstm_model.save_model(model_path)
        
        results_path = f"{self.results_dir}/data/lstm_{symbol}_results.pkl"
        save_results(lstm_results, results_path)
        
        # Create visualizations
        self.visualizer.plot_model_training_history(
            history.history, 
            f"{self.results_dir}/plots/lstm_{symbol}_training.png"
        )
        
        self.visualizer.plot_prediction_vs_actual(
            lstm_data['y_test'], y_pred_test.flatten(),
            f"LSTM Predictions - {symbol}",
            f"{self.results_dir}/plots/lstm_{symbol}_predictions.png"
        )
        
        self.logger.info(f"LSTM experiment completed for {symbol}")
        self.logger.info(f"Test MSE: {test_mse:.6f}, Test R²: {test_r2:.4f}")
        
        return lstm_results
    
    def experiment_pinn_option_pricing(self):
        """Run simplified option pricing experiment using real market data"""
    
        self.logger.info("Starting simplified option pricing experiment")
        
        # Use real SPY data if available, otherwise create realistic synthetic data
        if hasattr(self, 'financial_data') and 'SPY' in self.financial_data:
            spy_data = self.financial_data['SPY'].tail(100)  # Use recent data
            current_price = spy_data['Close'].iloc[-1]
            current_vol = spy_data['Volatility'].iloc[-1] if not pd.isna(spy_data['Volatility'].iloc[-1]) else 0.2
        else:
            current_price = 100.0
            current_vol = 0.2
        
        # Create realistic option pricing dataset
        np.random.seed(42)
        n_samples = 1000
        
        # Generate realistic market scenarios
        S_values = np.random.uniform(current_price * 0.8, current_price * 1.2, n_samples)
        K_values = np.random.uniform(current_price * 0.9, current_price * 1.1, n_samples)
        T_values = np.random.uniform(0.1, 1.0, n_samples)  # 1 month to 1 year
        r_values = np.random.uniform(0.02, 0.06, n_samples)  # 2% to 6%
        sigma_values = np.random.uniform(0.15, 0.35, n_samples)  # 15% to 35% vol
        
        # Calculate Black-Scholes prices as targets
        from utils import black_scholes_call
        call_prices = []
        
        for i in range(n_samples):
            try:
                price = black_scholes_call(S_values[i], K_values[i], T_values[i], r_values[i], sigma_values[i])
                call_prices.append(price)
            except:
                call_prices.append(max(S_values[i] - K_values[i], 0))  # Intrinsic value fallback
        
        # Prepare data
        X = np.column_stack([S_values, K_values, T_values, r_values, sigma_values]).astype(np.float32)
        y = np.array(call_prices).astype(np.float32)
        
        # Split data
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        # Build simple neural network for option pricing
        from tensorflow.keras import layers, Model, optimizers
        
        inputs = layers.Input(shape=(5,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(16, activation='relu')(x)
        outputs = layers.Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            verbose=1
        )
        
        # Make predictions
        y_pred_test = model.predict(X_test).flatten()
        
        # Calculate metrics
        test_mse = calculate_mse(y_test, y_pred_test)
        test_mae = calculate_mae(y_test, y_pred_test)
        test_r2 = calculate_r2(y_test, y_pred_test)
        
        # Store results
        pinn_results = {
            'model_type': 'Option_Pricing_NN',
            'test_mse': test_mse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'history': history.history,
            'predictions': y_pred_test,
            'actual': y_test,
            'current_price': current_price
        }
        
        self.results['option_pricing'] = pinn_results
        self.models['option_pricing'] = model
        
        # Save results
        results_path = f"{self.results_dir}/data/option_pricing_results.pkl"
        save_results(pinn_results, results_path)
        
        # Create visualizations
        self.visualizer.plot_model_training_history(
            history.history,
            f"{self.results_dir}/plots/option_pricing_training.png"
        )
        
        self.visualizer.plot_prediction_vs_actual(
            y_test, y_pred_test,
            "Option Pricing Neural Network",
            f"{self.results_dir}/plots/option_pricing_predictions.png"
        )
        
        self.logger.info("Option pricing experiment completed")
        self.logger.info(f"Test MSE: {test_mse:.6f}, Test R²: {test_r2:.4f}")
        
        return pinn_results
    
    def experiment_neural_sde(self):
        """Run Neural SDE experiment"""
        
        self.logger.info("Starting Neural SDE experiment")
        
        # Generate training data for Neural SDE
        np.random.seed(42)
        n_samples = 1000
        state_dim = 2
        
        # Generate synthetic SDE data
        X_data = np.random.randn(n_samples, state_dim + 1)  # [X_t, t]
        
        # True drift and diffusion functions (for synthetic data)
        drift_targets = 0.05 * X_data[:, :state_dim] + 0.1 * np.random.randn(n_samples, state_dim)
        diffusion_targets = 0.2 * np.ones((n_samples, state_dim)) + 0.05 * np.random.randn(n_samples, state_dim)
        
        # Build and train Neural SDE
        nsde_model = NeuralSDE(state_dim=state_dim)
        nsde_model.build_model(
            hidden_units=[64, 32],
            learning_rate=0.001
        )
        
        # Train model
        history = nsde_model.train(
            X_data, drift_targets, diffusion_targets,
            epochs=100, batch_size=32, verbose=1
        )
        
        # Simulate paths
        X0 = np.array([100.0, 0.04])  # Initial stock price and volatility
        simulated_paths = nsde_model.simulate_path(X0, T=1.0, n_steps=252)
        
        # Store results
        nsde_results = {
            'model_type': 'Neural_SDE',
            'history': history.history,
            'simulated_paths': simulated_paths,
            'initial_state': X0
        }
        
        self.results['neural_sde'] = nsde_results
        self.models['neural_sde'] = nsde_model
        
        # Save results
        results_path = f"{self.results_dir}/data/neural_sde_results.pkl"
        save_results(nsde_results, results_path)
        
        # Create visualizations
        self.visualizer.plot_model_training_history(
            history.history,
            f"{self.results_dir}/plots/neural_sde_training.png"
        )
        
        # Plot simulated paths
        paths_for_plot = simulated_paths[:, 0].reshape(1, -1)  # Stock price component
        self.visualizer.plot_monte_carlo_paths(
            paths_for_plot, "Neural SDE Simulation", 1,
            f"{self.results_dir}/plots/neural_sde_paths.png"
        )
        
        self.logger.info("Neural SDE experiment completed")
        
        return nsde_results
    
    def run_comprehensive_experiment(self):
        """Run all experiments in sequence using real-world data"""
    
        self.logger.info("Starting comprehensive neural finance experiment with real data")
        
        try:
            # Load real financial data first
            self.logger.info("Loading real market data...")
            financial_data = self.load_financial_data(['SPY', 'AAPL', 'MSFT'])
            
            if not financial_data:
                self.logger.warning("No real data loaded, generating synthetic data...")
                self.generate_synthetic_data()
            
            # Run experiments on real data
            results_summary = {}
            
            # 1. LSTM experiment on multiple stocks
            for symbol in ['SPY', 'AAPL', 'MSFT']:
                if symbol in self.financial_data:
                    self.logger.info(f"Running LSTM experiment for {symbol}")
                    lstm_results = self.experiment_lstm_prediction(symbol)
                    results_summary[f'LSTM_{symbol}'] = lstm_results
            
            # 2. Option pricing experiment
            self.logger.info("Running option pricing experiment")
            option_results = self.experiment_pinn_option_pricing()
            results_summary['Option_Pricing'] = option_results
            
            # 3. Neural SDE experiment
            self.logger.info("Running Neural SDE experiment")
            nsde_results = self.experiment_neural_sde()
            results_summary['Neural_SDE'] = nsde_results
            
            # Generate comprehensive report
            self.generate_experiment_report()
            
            self.logger.info("All experiments completed successfully")
            return results_summary
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive experiment: {e}")
            # Continue with synthetic data if real data fails
            self.logger.info("Falling back to synthetic data experiments...")
            self.generate_synthetic_data()
            
            # Run simplified experiments
            if hasattr(self, 'synthetic_data'):
                nsde_results = self.experiment_neural_sde()
                self.generate_experiment_report()
    
    def generate_experiment_report(self):
        """Generate comprehensive experiment report"""
        
        self.logger.info("Generating experiment report")
        
        report = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'results_summary': {},
            'model_performance': {}
        }
        
        # Summarize results
        for exp_name, results in self.results.items():
            if 'test_mse' in results:
                report['model_performance'][exp_name] = {
                    'test_mse': results['test_mse'],
                    'test_r2': results.get('test_r2', 'N/A'),
                    'model_type': results.get('model_type', 'Unknown')
                }
        
        # Save report
        report_path = f"{self.results_dir}/experiment_report.pkl"
        save_results(report, report_path)
        
        # Create summary visualization
        if len(self.results) > 1:
            self.create_results_summary_plot()
        
        # Save detailed stats to text file - ADD THIS LINE
        self.save_experiment_stats_to_txt()
        
        self.logger.info(f"Experiment report saved to {report_path}")
        
        return report
    
    def create_results_summary_plot(self):
        """Create summary plot of all experiment results"""
        
        import matplotlib.pyplot as plt
        
        # Extract performance metrics
        models = []
        mse_scores = []
        r2_scores = []
        
        for exp_name, results in self.results.items():
            if 'test_mse' in results:
                models.append(exp_name)
                mse_scores.append(results['test_mse'])
                r2_scores.append(results.get('test_r2', 0))
        
        if len(models) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # MSE comparison
            ax1.bar(models, mse_scores, alpha=0.7)
            ax1.set_title('Model Performance - MSE')
            ax1.set_ylabel('Mean Squared Error')
            ax1.tick_params(axis='x', rotation=45)
            
            # R² comparison
            ax2.bar(models, r2_scores, alpha=0.7, color='green')
            ax2.set_title('Model Performance - R²')
            ax2.set_ylabel('R² Score')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/plots/results_summary.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
    def save_experiment_stats_to_txt(self):
        """Save comprehensive experiment statistics to a text file for easy copy-paste to README"""
        
        stats_file = f"{self.results_dir}/experiment_stats.txt"
        
        with open(stats_file, 'w') as f:
            f.write("# Neural Finance ML - Experiment Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Experiment Name: {self.experiment_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Results Directory: {self.results_dir}\n\n")
            
            # Overall Summary
            f.write("## Overall Performance Summary\n")
            f.write("-" * 40 + "\n")
            
            total_models = len(self.results)
            successful_models = sum(1 for r in self.results.values() if 'test_r2' in r and r['test_r2'] > 0.5)
            
            f.write(f"Total Models Trained: {total_models}\n")
            f.write(f"Successful Models (R² > 0.5): {successful_models}\n")
            f.write(f"Success Rate: {(successful_models/total_models)*100:.1f}%\n\n")
            
            # Detailed Results for Each Model
            f.write("## Detailed Model Performance\n")
            f.write("-" * 40 + "\n\n")
            
            for exp_name, results in self.results.items():
                f.write(f"### {exp_name.upper().replace('_', ' ')}\n")
                f.write(f"Model Type: {results.get('model_type', 'Unknown')}\n")
                
                if 'symbol' in results:
                    f.write(f"Symbol: {results['symbol']}\n")
                
                # Training Metrics
                if 'train_mse' in results and 'train_r2' in results:
                    f.write(f"Training MSE: {results['train_mse']:.6f}\n")
                    f.write(f"Training R²: {results['train_r2']:.4f}\n")
                
                # Validation/Test Metrics
                if 'test_mse' in results:
                    f.write(f"Test MSE: {results['test_mse']:.6f}\n")
                if 'test_mae' in results:
                    f.write(f"Test MAE: {results['test_mae']:.6f}\n")
                if 'test_r2' in results:
                    f.write(f"Test R²: {results['test_r2']:.4f}\n")
                
                # Model-specific metrics
                if results.get('model_type') == 'LSTM':
                    # Calculate additional LSTM metrics
                    if 'history' in results:
                        history = results['history']
                        final_train_loss = history['loss'][-1] if 'loss' in history else 'N/A'
                        final_val_loss = history['val_loss'][-1] if 'val_loss' in history else 'N/A'
                        best_val_loss = min(history['val_loss']) if 'val_loss' in history else 'N/A'
                        
                        f.write(f"Final Training Loss: {final_train_loss:.6f}\n")
                        f.write(f"Final Validation Loss: {final_val_loss:.6f}\n")
                        f.write(f"Best Validation Loss: {best_val_loss:.6f}\n")
                        
                        # Convergence info
                        if 'val_loss' in history and len(history['val_loss']) > 10:
                            epochs_trained = len(history['val_loss'])
                            f.write(f"Epochs Trained: {epochs_trained}\n")
                            
                            # Check if model converged (last 5 epochs have similar loss)
                            last_5_losses = history['val_loss'][-5:]
                            loss_std = np.std(last_5_losses)
                            converged = "Yes" if loss_std < 0.001 else "No"
                            f.write(f"Converged: {converged}\n")
                
                # Performance Rating
                if 'test_r2' in results:
                    r2 = results['test_r2']
                    if r2 > 0.9:
                        rating = "Excellent"
                    elif r2 > 0.8:
                        rating = "Very Good"
                    elif r2 > 0.7:
                        rating = "Good"
                    elif r2 > 0.5:
                        rating = "Fair"
                    else:
                        rating = "Poor"
                    f.write(f"Performance Rating: {rating}\n")
                
                f.write("\n")
            
            # Best Performing Models
            f.write("## Best Performing Models\n")
            f.write("-" * 40 + "\n")
            
            # Sort models by R²
            models_with_r2 = [(name, results) for name, results in self.results.items() 
                            if 'test_r2' in results]
            models_with_r2.sort(key=lambda x: x[1]['test_r2'], reverse=True)
            
            for i, (name, results) in enumerate(models_with_r2[:3], 1):
                f.write(f"{i}. {name}: R² = {results['test_r2']:.4f}, MSE = {results['test_mse']:.6f}\n")
            
            f.write("\n")
            
            # Data Statistics
            f.write("## Data Statistics\n")
            f.write("-" * 40 + "\n")
            
            if hasattr(self, 'financial_data'):
                f.write("Real Market Data Used:\n")
                for symbol, data in self.financial_data.items():
                    f.write(f"- {symbol}: {len(data)} days ({data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')})\n")
                    f.write(f"  Price Range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}\n")
                    f.write(f"  Avg Daily Return: {data['Returns'].mean()*100:.3f}%\n")
                    f.write(f"  Volatility: {data['Returns'].std()*np.sqrt(252)*100:.1f}%\n")
            
            f.write("\n")
            
            # Technical Details
            f.write("## Technical Configuration\n")
            f.write("-" * 40 + "\n")
            f.write("LSTM Configuration:\n")
            f.write("- Sequence Length: 30 days\n")
            f.write("- LSTM Units: [64, 32]\n")
            f.write("- Dense Units: [16]\n")
            f.write("- Dropout Rate: 0.2\n")
            f.write("- Learning Rate: 0.001\n")
            f.write("- Batch Size: 32\n")
            f.write("- Max Epochs: 50\n\n")
            
            f.write("Option Pricing NN Configuration:\n")
            f.write("- Hidden Layers: [64, 32, 16]\n")
            f.write("- Activation: ReLU\n")
            f.write("- Output Activation: Linear\n")
            f.write("- Optimizer: Adam\n")
            f.write("- Loss Function: MSE\n")
            f.write("- Epochs: 100\n\n")
            
            # Files Generated
            f.write("## Generated Files\n")
            f.write("-" * 40 + "\n")
            f.write("Models:\n")
            for model_name in self.models.keys():
                f.write(f"- {model_name}.h5\n")
            
            f.write("\nPlots:\n")
            plot_files = [
                "training_history.png",
                "predictions_vs_actual.png", 
                "results_summary.png",
                "monte_carlo_paths.png"
            ]
            for plot in plot_files:
                f.write(f"- {plot}\n")
            
            f.write("\nData Files:\n")
            f.write("- experiment_report.pkl\n")
            f.write("- model_results.pkl (for each model)\n")
            f.write("- experiment_stats.txt (this file)\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("End of Experiment Report\n")
        
        self.logger.info(f"Experiment statistics saved to {stats_file}")
        print(f"\n📊 Detailed stats saved to: {stats_file}")
        print("📋 You can copy-paste this content directly to your README.md file!")
        
        return stats_file
def main():
    """Main execution function"""
    
    print("=" * 60)
    print("NEURAL FINANCE ML - COMPREHENSIVE EXPERIMENT")
    print("=" * 60)
    
    # Initialize experiment
    experiment = NeuralFinanceExperiment("comprehensive_neural_finance")
    
    try:
        # Run comprehensive experiment
        experiment.run_comprehensive_experiment()
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results saved to: {experiment.results_dir}")
        print("\nExperiment Summary:")
        
        for exp_name, results in experiment.results.items():
            print(f"\n{exp_name.upper()}:")
            if 'test_mse' in results:
                print(f"  - Test MSE: {results['test_mse']:.6f}")
                if 'test_r2' in results:
                    print(f"  - Test R²: {results['test_r2']:.4f}")
            print(f"  - Model Type: {results.get('model_type', 'N/A')}")
        
    except Exception as e:
        print(f"\nError in experiment: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
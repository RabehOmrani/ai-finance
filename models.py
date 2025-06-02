import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, optimizers, callbacks
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class BaseNeuralModel(ABC):
    """Abstract base class for neural network models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.history = None
        self.is_trained = False
    
    @abstractmethod
    def build_model(self, **kwargs):
        """Build the neural network architecture"""
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """Train the model"""
        pass
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not built yet")
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True

class LSTMPricePredictor(BaseNeuralModel):
    """LSTM model for stock price prediction"""
    
    def __init__(self, sequence_length: int, n_features: int, name: str = "LSTM_Predictor"):
        super().__init__(name)
        self.sequence_length = sequence_length
        self.n_features = n_features
    
    def build_model(self, lstm_units: List[int] = [50, 50], 
                   dense_units: List[int] = [25], 
                   dropout_rate: float = 0.2,
                   learning_rate: float = 0.001):
        """Build LSTM architecture"""
        
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        x = inputs
        
        # LSTM layers
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1
            x = layers.LSTM(units, return_sequences=return_sequences, 
                           dropout=dropout_rate, recurrent_dropout=dropout_rate)(x)
            x = layers.BatchNormalization()(x)
        
        # Dense layers
        for units in dense_units:
            x = layers.Dense(units, activation='relu')(x)
            x = layers.Dropout(dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='linear')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs: int = 100, batch_size: int = 32, 
              early_stopping_patience: int = 10,
              reduce_lr_patience: int = 5,
              verbose: int = 1):
        """Train the LSTM model"""
        
        if self.model is None:
            self.build_model()
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-7
            )
        ]
        
        # Validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        self.is_trained = True
        return self.history

class PhysicsInformedNN(BaseNeuralModel):
    """Physics-Informed Neural Network for option pricing"""
    
    def __init__(self, name: str = "PINN_OptionPricing"):
        super().__init__(name)
        self.pde_weight = 1.0
        self.boundary_weight = 1.0
        self.data_weight = 1.0
    
    def build_model(self, hidden_layers: List[int] = [50, 50, 50],
                   activation: str = 'tanh',
                   learning_rate: float = 0.001):
        """Build PINN architecture"""
        
        # Input: [S, K, T, r, sigma] - stock price, strike, time, rate, volatility
        inputs = layers.Input(shape=(6,), name='market_inputs')
        x = inputs
        
        # Hidden layers
        for units in hidden_layers:
            x = layers.Dense(units, activation=activation)(x)
        
        # Output: option price
        outputs = layers.Dense(1, activation='softplus', name='option_price')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        
        # Custom optimizer
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
        
        return self.model
    
    def black_scholes_pde_loss(self, inputs, predictions):
        """Calculate Black-Scholes PDE residual loss"""
        S, K, T, r, sigma, moneyness = tf.split(inputs, 6, axis=1)
        V = predictions
        
        # Calculate derivatives using automatic differentiation
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(inputs)
                V_pred = self.model(inputs)
            
            # First derivatives
            dV_dS = tape1.gradient(V_pred, inputs)[:, 0:1]  # dV/dS
            dV_dT = tape1.gradient(V_pred, inputs)[:, 2:3]  # dV/dT
        
        # Second derivative
        d2V_dS2 = tape2.gradient(dV_dS, inputs)[:, 0:1]  # d²V/dS²
        
        # Black-Scholes PDE: dV/dt + 0.5*σ²*S²*d²V/dS² + r*S*dV/dS - r*V = 0
        pde_residual = (dV_dT + 0.5 * sigma**2 * S**2 * d2V_dS2 + 
                       r * S * dV_dS - r * V)
        
        return tf.reduce_mean(tf.square(pde_residual))
    
    def boundary_condition_loss(self, inputs, predictions):
        """Calculate boundary condition loss"""
        S, K, T, r, sigma, moneyness = tf.split(inputs, 6, axis=1)
        V = predictions
        
        # Terminal condition: V(S, T=0) = max(S - K, 0) for call option
        terminal_mask = tf.less(T, 0.01)  # Near expiration
        terminal_payoff = tf.maximum(S - K, 0.0)
        terminal_loss = tf.reduce_mean(
            tf.square((V - terminal_payoff) * tf.cast(terminal_mask, tf.float32))
        )
        
        # Boundary conditions at S=0 and S→∞
        zero_price_mask = tf.less(S, 0.01)
        zero_price_loss = tf.reduce_mean(
            tf.square(V * tf.cast(zero_price_mask, tf.float32))
        )
        
        return terminal_loss + zero_price_loss
    
    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs: int = 1000, batch_size: int = 256,
              pde_weight: float = 1.0, boundary_weight: float = 1.0,
              verbose: int = 1):
        """Train PINN with physics constraints"""
        
        if self.model is None:
            self.build_model()
        
        self.pde_weight = pde_weight
        self.boundary_weight = boundary_weight
        
        # Custom training loop
        optimizer = self.model.optimizer
        
        @tf.function
        def train_step(x_batch, y_batch):
            with tf.GradientTape() as tape:
                predictions = self.model(x_batch, training=True)
                
                # Data loss
                data_loss = tf.reduce_mean(tf.square(predictions - y_batch))
                
                # Physics losses
                pde_loss = self.black_scholes_pde_loss(x_batch, predictions)
                boundary_loss = self.boundary_condition_loss(x_batch, predictions)
                
                # Total loss
                total_loss = (self.data_weight * data_loss + 
                            self.pde_weight * pde_loss + 
                            self.boundary_weight * boundary_loss)
            
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            return total_loss, data_loss, pde_loss, boundary_loss
        
        # Training loop
        history = {'loss': [], 'data_loss': [], 'pde_loss': [], 'boundary_loss': []}
        
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        dataset = dataset.batch(batch_size).shuffle(1000)
        
        for epoch in range(epochs):
            epoch_losses = {'total': [], 'data': [], 'pde': [], 'boundary': []}
            
            for x_batch, y_batch in dataset:
                total_loss, data_loss, pde_loss, boundary_loss = train_step(x_batch, y_batch)
                
                epoch_losses['total'].append(total_loss.numpy())
                epoch_losses['data'].append(data_loss.numpy())
                epoch_losses['pde'].append(pde_loss.numpy())
                epoch_losses['boundary'].append(boundary_loss.numpy())
            
            # Record epoch averages
            for key in epoch_losses:
                if key == 'total':
                    history['loss'].append(np.mean(epoch_losses[key]))
                else:
                    history[f'{key}_loss'].append(np.mean(epoch_losses[key]))
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {history['loss'][-1]:.6f}, "
                      f"Data = {history['data_loss'][-1]:.6f}, "
                      f"PDE = {history['pde_loss'][-1]:.6f}, "
                      f"Boundary = {history['boundary_loss'][-1]:.6f}")
        
        self.history = history
        self.is_trained = True
        return history

class NeuralSDE(BaseNeuralModel):
    """Neural Stochastic Differential Equation model"""
    
    def __init__(self, state_dim: int, name: str = "Neural_SDE"):
        super().__init__(name)
        self.state_dim = state_dim
        self.drift_net = None
        self.diffusion_net = None
    
    def build_model(self, hidden_units: List[int] = [64, 64],
                   activation: str = 'relu',
                   learning_rate: float = 0.001):
        """Build Neural SDE architecture"""
        
        # Drift network μ(X_t, t)
        drift_input = layers.Input(shape=(self.state_dim + 1,), name='drift_input')  # [X_t, t]
        x_drift = drift_input
        
        for units in hidden_units:
            x_drift = layers.Dense(units, activation=activation)(x_drift)
        
        drift_output = layers.Dense(self.state_dim, name='drift_output')(x_drift)
        self.drift_net = Model(inputs=drift_input, outputs=drift_output, name='drift_network')
        
        # Diffusion network σ(X_t, t)
        diffusion_input = layers.Input(shape=(self.state_dim + 1,), name='diffusion_input')
        x_diff = diffusion_input
        
        for units in hidden_units:
            x_diff = layers.Dense(units, activation=activation)(x_diff)
        
        diffusion_output = layers.Dense(self.state_dim, activation='softplus', name='diffusion_output')(x_diff)
        self.diffusion_net = Model(inputs=diffusion_input, outputs=diffusion_output, name='diffusion_network')
        
        # Combined model for training
        combined_input = layers.Input(shape=(self.state_dim + 1,))
        drift_pred = self.drift_net(combined_input)
        diffusion_pred = self.diffusion_net(combined_input)
        
        self.model = Model(inputs=combined_input, 
                          outputs=[drift_pred, diffusion_pred], 
                          name=self.name)
        
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=['mse', 'mse'])
        
        return self.model
    
    def simulate_path(self, X0: np.ndarray, T: float, n_steps: int) -> np.ndarray:
        """Simulate SDE path using Euler-Maruyama scheme"""
        if not self.is_trained:
            raise ValueError("Model must be trained before simulation")
        
        dt = T / n_steps
        path = np.zeros((n_steps + 1, self.state_dim))
        path[0] = X0
        
        for i in range(n_steps):
            t = i * dt
            X_t = path[i]
            
            # Prepare input [X_t, t]
            input_data = np.concatenate([X_t, [t]]).reshape(1, -1)
            
            # Get drift and diffusion
            drift = self.drift_net.predict(input_data, verbose=0)[0]
            diffusion = self.diffusion_net.predict(input_data, verbose=0)[0]
            
            # Euler-Maruyama step
            dW = np.random.normal(0, np.sqrt(dt), self.state_dim)
            path[i + 1] = X_t + drift * dt + diffusion * dW
        
        return path
    
    def train(self, X_data, drift_targets, diffusion_targets,
              epochs: int = 100, batch_size: int = 32, verbose: int = 1):
        """Train Neural SDE"""
        if self.model is None:
            self.build_model()
        
        self.history = self.model.fit(
            X_data, [drift_targets, diffusion_targets],
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        self.is_trained = True
        return self.history

class PortfolioOptimizer(BaseNeuralModel):
    """Neural network for portfolio optimization"""
    
    def __init__(self, n_assets: int, name: str = "Portfolio_Optimizer"):
        super().__init__(name)
        self.n_assets = n_assets
    
    def build_model(self, hidden_layers: List[int] = [128, 64, 32],
                   activation: str = 'relu',
                   learning_rate: float = 0.001):
        """Build portfolio optimization network"""
        
        # Input: market features (returns, volatilities, correlations, etc.)
        inputs = layers.Input(shape=(self.n_assets * 3,), name='market_features')  # returns, vol, momentum
        x = inputs
        
        # Hidden layers
        for units in hidden_layers:
            x = layers.Dense(units, activation=activation)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
        
        # Output: portfolio weights (softmax for sum to 1 constraint)
        outputs = layers.Dense(self.n_assets, activation='softmax', name='portfolio_weights')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        
        # Custom loss function for portfolio optimization
        def portfolio_loss(y_true, y_pred):
            # y_true contains [expected_returns, covariance_matrix_flattened]
            # y_pred contains portfolio weights
            
            # Expected portfolio return
            expected_returns = y_true[:, :self.n_assets]
            portfolio_return = tf.reduce_sum(y_pred * expected_returns, axis=1)
            
            # Portfolio variance (simplified)
            portfolio_variance = tf.reduce_sum(tf.square(y_pred), axis=1)
            
            # Sharpe ratio maximization (minimize negative Sharpe)
            sharpe_ratio = portfolio_return / tf.sqrt(portfolio_variance + 1e-8)
            
            return -tf.reduce_mean(sharpe_ratio)
        
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=portfolio_loss)
        
        return self.model
    
    def train(self, X_train, y_train, epochs: int = 100, batch_size: int = 32, verbose: int = 1):
        """Train portfolio optimizer"""
        if self.model is None:
            self.build_model()
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        self.is_trained = True
        return self.history

# Model factory
class ModelFactory:
    """Factory class for creating different types of models"""
    
    @staticmethod
    def create_model(model_type: str, **kwargs):
        """Create a model of specified type"""
        
        if model_type.lower() == 'lstm':
            return LSTMPricePredictor(**kwargs)
        elif model_type.lower() == 'pinn':
            return PhysicsInformedNN(**kwargs)
        elif model_type.lower() == 'neural_sde':
            return NeuralSDE(**kwargs)
        elif model_type.lower() == 'portfolio':
            return PortfolioOptimizer(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Example usage and testing
if __name__ == "__main__":
    print("Testing Neural Network Models...")
    
    # Test LSTM model
    print("\n1. Testing LSTM Price Predictor...")
    lstm_model = LSTMPricePredictor(sequence_length=30, n_features=4)
    lstm_model.build_model()
    print(f"LSTM model built: {lstm_model.model.summary()}")
    
    # Test PINN model
    print("\n2. Testing Physics-Informed Neural Network...")
    pinn_model = PhysicsInformedNN()
    pinn_model.build_model()
    print(f"PINN model built with {pinn_model.model.count_params()} parameters")
    
    # Test Neural SDE
    print("\n3. Testing Neural SDE...")
    nsde_model = NeuralSDE(state_dim=2)
    nsde_model.build_model()
    print(f"Neural SDE built with drift and diffusion networks")
    
    # Test Portfolio Optimizer
    print("\n4. Testing Portfolio Optimizer...")
    portfolio_model = PortfolioOptimizer(n_assets=5)
    portfolio_model.build_model()
    print(f"Portfolio optimizer built for {portfolio_model.n_assets} assets")
    
    # Test model factory
    print("\n5. Testing Model Factory...")
    factory_lstm = ModelFactory.create_model('lstm', sequence_length=20, n_features=3)
    print(f"Factory created LSTM: {type(factory_lstm).__name__}")
    
    print("\nAll models tested successfully!")
    print("Models are ready for training with real financial data.")
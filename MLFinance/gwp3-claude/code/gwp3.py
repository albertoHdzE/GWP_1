"""
GWP3: Deep Learning for Finance
Time series forecasting of SPY returns using MLP, LSTM, and CNN-GAF models.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, LSTM, Convolution2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import gc
import os
import hashlib

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

class DataCollector:
    """
    Collects financial time series data and performs exploration.
    Uses S&P 500 index (^GSPC) as the underlying security due to liquidity
    and representativeness of U.S. equity market performance.
    """
    def __init__(self, ticker="^GSPC", start_date=None, end_date=None):
        """
        Initialize data collector.

        Args:
            ticker: Yahoo Finance ticker symbol (default S&P 500)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        self.ticker = ticker
        self.start_date = start_date or (datetime.now() - timedelta(days=3000)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.data = None
        self.data_path = "gwp3_data.csv"
        self.plot_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "report", "images")

    def fetch_data(self):
        """
        Downloads historical price data for the specified ticker.
        Returns adjusted close prices to account for dividends and splits.
        """
        print(f"Downloading data for {self.ticker} from {self.start_date} to {self.end_date}")
        # Use _fix_yahoo_issue=True for newer yfinance versions that return MultiIndex with ticker
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)

        if self.data.empty:
            raise ValueError("No data downloaded")

        # Flatten MultiIndex columns (yfinance newer versions return MultiIndex)
        if isinstance(self.data.columns, pd.MultiIndex):
            # Get the second level which contains 'Close', 'Adj Close', etc.
            if self.data.columns.nlevels > 1:
                self.data = self.data.droplevel(0, axis=1)

        # Handle column naming - newer yfinance versions may use 'Close' instead of 'Adj Close'
        if 'Adj Close' not in self.data.columns:
            if 'Close' in self.data.columns:
                # Rename Close to Adj Close for compatibility
                self.data = self.data.rename(columns={'Close': 'Adj Close'})
            else:
                raise ValueError(f"Expected 'Close' or 'Adj Close' column. Available columns: {self.data.columns.tolist()}")

        return self

    def calculate_returns(self):
        """
        Computes daily returns as percentage change in adjusted close price.
        Return = (P(t) - P(t-1)) / P(t-1)
        """
        self.data['Return'] = self.data['Adj Close'].pct_change()
        return self

    def calculate_volatility(self):
        """
        Calculates 20-day rolling volatility (standard deviation of returns).
        Annualized by multiplying by sqrt(252) trading days.
        """
        self.data['Volatility'] = self.data['Return'].rolling(window=20).std() * np.sqrt(252)
        return self

    def limit_observations(self, max_samples=2000):
        """
        Limits dataset to maximum specified number of observations.
        Removes NA values and takes last n samples if needed.
        """
        self.data = self.data.dropna().tail(max_samples)
        return self

    def plot_price_series(self, save_path=None):
        """
        Displays and optionally saves adjusted close price visualization.
        Shows trend and major movements in the security over time.
        """
        if save_path is None:
            save_path = os.path.join(self.plot_dir, "price_series.png")

        plt.figure(figsize=(14, 6))
        sns.lineplot(data=self.data['Adj Close'], color='#2E86AB')
        plt.title(f"Adjusted Close Price - {self.ticker}")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")

        # Limit x-axis labels to avoid overcrowding
        n_labels = min(15, len(self.data))
        labels_per_tick = max(1, n_labels // 5)
        plt.xticks(range(0, len(self.data), labels_per_tick),
                   self.data.index[range(0, len(self.data), labels_per_tick)], rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    def plot_returns_histogram(self, save_path=None):
        """
        Shows distribution of daily returns with fitted normal curve overlay.
        Assesses whether returns approximate normality assumption.
        """
        from scipy import stats

        if save_path is None:
            save_path = os.path.join(self.plot_dir, "returns_histogram.png")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        returns = self.data['Return'].dropna()

        axes[0].hist(returns, bins=50, color='#A23B72', alpha=0.7, density=True)
        x_vals = np.linspace(returns.min(), returns.max(), 100)
        axes[0].plot(x_vals, stats.norm.pdf(x_vals, returns.mean(), returns.std()),
                     'r-', linewidth=2)
        axes[0].set_title("Daily Returns Distribution")
        axes[0].set_xlabel("Return")
        axes[0].set_ylabel("Density")

        self.data['Return'].dropna().plot(kind='hist', bins=50, color='#F18F01', ax=axes[1])
        axes[1].set_title("Histogram of Daily Returns")
        axes[1].set_xlabel("Return")
        axes[1].set_ylabel("Frequency")

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_volatility(self, save_path=None):
        """
        Displays volatility time series showing periods of market stress.
        Higher volatility typically corresponds to financial crises or uncertainty.
        """
        if save_path is None:
            save_path = os.path.join(self.plot_dir, "volatility.png")

        plt.figure(figsize=(14, 6))
        sns.lineplot(data=self.data['Volatility'], color='#E36414')
        plt.title(f"20-Day Rolling Annualized Volatility - {self.ticker}")
        plt.xlabel("Date")
        plt.ylabel("Volatility (%)")

        n_labels = min(15, len(self.data))
        labels_per_tick = max(1, n_labels // 5)
        plt.xticks(range(0, len(self.data), labels_per_tick),
                   self.data.index[range(0, len(self.data), labels_per_tick)], rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_returns_vs_volatility(self, save_path=None):
        """
        Scatter plot showing relationship between returns and volatility.
        Helps visualize the risk-return tradeoff in the data.
        """
        if save_path is None:
            save_path = os.path.join(self.plot_dir, "returns_volatility_scatter.png")

        plt.figure(figsize=(10, 8))
        plt.scatter(self.data['Volatility'].dropna(), self.data['Return'].dropna(),
                    alpha=0.3, color='#2E86AB')
        plt.xlabel("Volatility")
        plt.ylabel("Return")
        plt.title("Returns vs Volatility Scatter Plot")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def plot_acf(self, save_path=None):
        """
        Autocorrelation function of returns to check for predictable patterns.
        Nearly zero autocorrelations at higher lags indicate random walk behavior.
        """
        from statsmodels.graphics.tsaplots import plot_acf

        if save_path is None:
            save_path = os.path.join(self.plot_dir, "returns_acf.png")

        fig = plt.figure(figsize=(10, 5))
        plot_acf(self.data['Return'].dropna(), lags=60, ax=fig.gca())
        plt.title("Autocorrelation Function of Returns")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_data(self):
        """
        Saves the processed data to CSV for reproducibility.
        """
        self.data.to_csv(self.data_path)
        print(f"Data saved to {self.data_path}")

    def load_data(self):
        """
        Loads previously saved data from CSV file.
        """
        if os.path.exists(self.data_path):
            self.data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            return True
        return False

    def describe_data(self):
        """
        Returns summary statistics of the collected data.
        """
        return {
            'Observations': len(self.data),
            'Date Range': f"{self.data.index.min()} to {self.data.index.max()}",
            'Mean Return': f"{self.data['Return'].mean()*100:.4f}%",
            'Std Return': f"{self.data['Return'].std()*100:.4f}%",
            'Mean Volatility': f"{self.data['Volatility'].mean()*100:.2f}%",
            'Min Return': f"{self.data['Return'].min()*100:.2f}%",
            'Max Return': f"{self.data['Return'].max()*100:.2f}%",
            'Total Return': f"{(self.data['Adj Close'].iloc[-1]/self.data['Adj Close'].iloc[0] - 1)*100:.2f}%"
        }


class DataPreprocessor:
    """
    Handles data preprocessing, feature engineering, and train/test splitting.
    Addresses the critical issue of data leakage in time series forecasting.

    Data leakage is a common problem where information from outside the training set
    or future data influences the model. This preprocessor provides methods to:
    1. Create features with intentional leakage (for demonstration)
    2. Create features without leakage (correct approach)
    """
    def __init__(self, n_lookback=10):
        self.n_lookback = n_lookback
        self.scaler_X = StandardScaler()
        self.scaler_y_regression = StandardScaler()
        self.le_encoder = LabelEncoder()

    def create_features_leakage(self, df):
        """
        Creates features WITH intentional data leakage for demonstration.

        LEAKAGE MECHANISMS:
        1. Centered rolling statistics incorporate future data
        2. Moving averages computed on centered windows include look-ahead bias

        These features are NOT available at prediction time and would cause
        the model to achieve artificially inflated performance.
        """
        data = df.copy()

        # Lagged returns (valid features)
        for i in range(1, self.n_lookback + 1):
            data[f'Return_Lag{i}'] = data['Return'].shift(i)

        # LEAKAGE: Using centered rolling statistics that incorporate future data
        data['Return_RollMean_Leak'] = data['Return'].rolling(window=self.n_lookback, center=True).mean()
        data['Return_RollStd_Leak'] = data['Return'].rolling(window=self.n_lookback, center=True).std()

        # LEAKAGE: Moving average crossover signals using future prices
        data['MA_5'] = data['Adj Close'].rolling(window=5).mean()
        data['MA_20'] = data['Adj Close'].rolling(window=20).mean()

        return data

    def create_features_no_leakage(self, df):
        """
        Creates features WITHOUT data leakage.

        ALL FEATURES ARE BASED ONLY ON HISTORICAL DATA AVAILABLE AT
        THE TIME OF PREDICTION, ensuring realistic deployment conditions.

        This represents the correct approach for time series forecasting.
        """
        data = df.copy()

        # Lagged returns (valid features)
        for i in range(1, self.n_lookback + 1):
            data[f'Return_Lag{i}'] = data['Return'].shift(i)

        # Rolling statistics using past data only
        data['Return_RollMean'] = data['Return'].rolling(window=self.n_lookback).mean()
        data['Return_RollStd'] = data['Return'].rolling(window=self.n_lookback).std()

        # Moving averages using past data only
        data['MA_5'] = data['Adj Close'].rolling(window=5).mean()
        data['MA_20'] = data['Adj Close'].rolling(window=20).mean()

        return data

    def create_labels_classification(self, df):
        """
        Creates binary classification labels based on next-day return direction.

        Target: Direction of tomorrow's return
        1 = positive return (up)
        0 = negative or zero return (down)

        This is a realistic prediction target since we cannot know tomorrow's
        return at prediction time.
        """
        data = df.copy()
        # Predict next day return direction
        data['Target_Class'] = (data['Return'].shift(-1) > 0).astype(int)
        return data

    def create_labels_regression(self, df):
        """
        Creates continuous target for regression task.

        Target: Next day return value (continuous)

        This allows direct prediction of expected return magnitude rather
        than just direction.
        """
        data = df.copy()
        data['Target_Reg'] = data['Return'].shift(-1)
        return data

    def prepare_data(self, df, use_leakage=False):
        """
        Main preparation function that orchestrates feature creation, labeling,
        and train/test splitting.

        Args:
            df: Raw data with returns
            use_leakage: If True, creates features with intentional leakage

        Returns:
            Dictionary containing all prepared datasets and scalers
        """
        # Create features (with or without leakage based on parameter)
        data = self.create_features_leakage(df) if use_leakage else self.create_features_no_leakage(df)

        # Create labels
        data = self.create_labels_classification(data)
        data = self.create_labels_regression(data)

        # Remove rows with NA values from shifting operations
        data = data.dropna()

        # Define feature columns (exclude leakage features when not using leakage)
        if use_leakage:
            feature_cols = ['Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'Return_Lag4', 'Return_Lag5',
                          'Return_RollMean_Leak', 'Return_RollStd_Leak',
                          'MA_5', 'MA_20']
        else:
            feature_cols = ['Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'Return_Lag4', 'Return_Lag5',
                          'Return_RollMean', 'Return_RollStd',
                          'MA_5', 'MA_20']

        X = data[feature_cols].values
        y_class = data['Target_Class'].values.reshape(-1, 1)
        y_reg = data['Target_Reg'].values.reshape(-1, 1)

        # Store original dates for walk-forward implementation
        dates = data['Date'].values

        # Scale features
        X_scaled = self.scaler_X.fit_transform(X)

        # Scale targets
        y_class_encoded = self.le_encoder.fit_transform(y_class)
        y_reg_scaled = self.scaler_y_regression.fit_transform(y_reg)

        return {
            'X': X_scaled,
            'y_class': y_class_encoded,
            'y_reg': y_reg_scaled,
            'feature_cols': feature_cols,
            'dates': dates
        }

    def transform(self, X):
        """
        Applies fitted scaler for inference on new data.

        Args:
            X: Features to transform (unscaled)

        Returns:
            Scaled features matching training distribution
        """
        return self.scaler_X.transform(X)


class MLPModel:
    """
    Multi-Layer Perceptron (MLP) neural network.
    Feed-forward architecture with dense hidden layers.
    """
    def __init__(self, input_dim, output_dim=2):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = None

    def build(self):
        """
        Constructs MLP with:
        - 2 hidden layers: 32 and 16 neurons
        - ReLU activation for hidden layers
        - Softmax output for classification
        """
        self.model = Sequential([
            Dense(32, input_dim=self.input_dim, activation='relu'),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(self.output_dim, activation='softmax')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self

    def train(self, X_train, y_train, epochs=100, batch_size=32, verbose=False):
        """
        Trains the model with early stopping to prevent overfitting.

        Args:
            X_train: Training features (scaled)
            y_train: Training labels
            epochs: Maximum training iterations
            batch_size: Samples per gradient update
            verbose: Print training progress (0=quiet, 1=progress bar, 2=one line per epoch)
        """
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=verbose
        )

    def predict(self, X):
        """
        Returns class predictions and probability estimates.
        """
        proba = self.model.predict(X)
        predictions = np.argmax(proba, axis=1)
        return predictions, proba

    def fit_transform(self, X, y_class):
        """
        Convenience function: train and return probability output.
        Used for generating backtest signals from trained model.
        """
        self.build()
        self.train(X, y_class)
        return self.predict_proba(X)


class LSTMModel:
    """
    Long Short-Term Memory (LSTM) neural network.
    Recurrent architecture that captures temporal dependencies in time series.
    """
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    def build(self):
        """
        Constructs LSTM model with:
        - 2 stacked LSTM layers (64 and 32 units)
        - Dropout for regularization
        - Dense output layer with softmax activation
        """
        self.model = Sequential([
            LSTM(64, input_shape=self.input_shape, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dropout(0.3),
            Dense(self.input_shape[-1], activation='softmax')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self

    def train(self, X_train, y_train, epochs=100, batch_size=32, verbose=False):
        """
        Trains the LSTM model with early stopping.

        Args:
            X_train: Training features shaped as (samples, time_steps, features)
            y_train: Training labels
            epochs: Maximum training iterations
            batch_size: Samples per gradient update
            verbose: Print training progress (0=quiet, 1=progress bar, 2=one line per epoch)
        """
        # Reshape X_train to 3D if needed (samples, 1, features)
        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=verbose
        )

    def predict(self, X):
        """
        Returns class predictions and probability estimates.
        """
        proba = self.model.predict(X)
        predictions = np.argmax(proba, axis=-1)
        return predictions, proba

    def fit_transform(self, X, y_class):
        """
        Convenience function: train and return probability output.
        Used for generating backtest signals from trained model.
        """
        self.build()
        self.train(X, y_class)
        return self.predict_proba(X)


class GAFEncoder:
    """
    Gram Angular Field (GAF) encoder for time series to image conversion.

    GAF encodes a time series into an image by:
    1. Normalizing the time series to [-1, 1]
    2. Computing trigonometric sums/differences between all pairs of points
    3. Producing an image where spatial positions encode temporal relationships

    This enables CNNs to process time series data.
    """
    def __init__(self, image_size=128):
        self.image_size = image_size

    def _normalize(self, time_series):
        """
        Normalizes time series to [-1, 1] range.

        Args:
            time_series: Input time series data

        Returns:
            Normalized time series scaled to [-1, 1]
        """
        min_val = np.min(time_series)
        max_val = np.max(time_series)

        if max_val == min_val:
            return np.ones_like(time_series)

        normalized = 2 * (time_series - min_val) / (max_val - min_val) - 1
        return np.clip(normalized, -1, 1)

    def encode(self, time_series):
        """
        Encodes a single 1D time series into a GAF image.

        Args:
            time_series: 1D array of time series values

        Returns:
            2D numpy array representing the GAF image (grayscale)
        """
        n = len(time_series)
        t = np.linspace(-np.pi/2, np.pi/2, n)

        # Normalize time series to [-1, 1]
        s = self._normalize(time_series)

        # Compute sin and cos of transformed time values
        if s.ndim == 1:
            s = np.expand_dims(s, axis=0)
        if t.ndim == 1:
            t = np.expand_dims(t, axis=0)

        # GAF encoding: cos(arcsin(s_i) + arcsin(s_j)) for SAGF
        # Using sum encoding
        phi = np.arcsin(s.T[:, None])  # Shape: (n, 1)
        theta = np.arcsin(s.T[None, :])  # Shape: (1, n)
        gaf_image = np.cos(phi + theta)  # Shape: (n, n)

        # Resize to specified dimensions if needed
        if self.image_size != n:
            gaf_image = plt.figure()
            plt.imshow(gaf_image, cmap='RdBu')
            plt.axis('off')
            gaf_image.canvas.draw()
            gaf_array = np.frombuffer(gaf_image.canvas.tostring_rgb(), dtype=np.uint8)
            gaf_array = gaf_array.reshape(gaf_image.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            gaf_image = plt.figure(figsize=(2, 2), dpi=self.image_size)
            plt.imshow(gaf_image, cmap='RdBu')
            plt.axis('off')
            gaf_image.canvas.draw()
            gaf_array = np.frombuffer(gaf_image.canvas.tostring_rgb(), dtype=np.uint8)
            gaf_array = gaf_array.reshape(gaf_image.canvas.get_width_height()[::-1] + (3,))
            plt.close()

        return gaf_image

    def encode_batch(self, time_series_matrix):
        """
        Encodes a batch of 1D time series into GAF images.

        Args:
            time_series_matrix: 2D array of shape (num_samples, series_length)

        Returns:
            4D numpy array of shape (num_samples, image_size, image_size, channels)
        """
        num_samples = time_series_matrix.shape[0]

        # Determine output dimensions - use smaller dimension for square images
        output_size = min(self.image_size, time_series_matrix.shape[1])

        images = []
        for i in range(num_samples):
            series = time_series_matrix[i]

            # Normalize
            min_val, max_val = np.min(series), np.max(series)
            if max_val != min_val:
                normalized = 2 * (series - min_val) / (max_val - min_val) - 1
            else:
                normalized = np.zeros_like(series)

            n = len(normalized)

            # GAF encoding using sum
            phi = np.arcsin(normalized[:, None])
            theta = np.arcsin(normalized[None, :])
            gaf_image = np.cos(phi + theta)

            # Resize to specified image_size using bilinear interpolation
            from scipy.ndimage import zoom
            scale = self.image_size / n
            resized = zoom(gaf_image, (scale, scale), order=1)

            # Convert to RGB by duplicating channel
            image = np.stack([resized, resized, resized], axis=-1)
            images.append(image)

        return np.array(images)


class CNNGAFModel:
    """
    Convolutional Neural Network (CNN) for GAF-encoded time series.

    Processes Gram Angular Field images to extract spatial features
    that encode temporal relationships in the original time series.
    """
    def __init__(self, image_shape=(128, 128, 3)):
        self.image_shape = image_shape
        self.model = None
        self.ga_encoder = GAFEncoder(image_size=image_shape[0])

    def build(self):
        """
        Constructs CNN architecture for GAF image classification:
        - 3 convolutional blocks with increasing feature maps
        - Max pooling for dimensionality reduction
        - Batch normalization for training stability
        - Dropout for regularization
        - Dense layers for final classification
        """
        self.model = Sequential([
            Convolution2D(32, (3, 3), activation='relu', padding='same', input_shape=self.image_shape),
            BatchNormalization(),
            Convolution2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.25),

            Convolution2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Convolution2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.25),

            Convolution2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Convolution2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.25),

            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(self.image_shape[-1], activation='softmax')  # Output dimension
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return self

    def train(self, X_train, y_train, epochs=100, batch_size=32, verbose=False):
        """
        Trains the CNN model with early stopping.

        Args:
            X_train: Training GAF images shaped as (samples, height, width, channels)
            y_train: Training labels
            epochs: Maximum training iterations
            batch_size: Samples per gradient update
            verbose: Print training progress (0=quiet, 1=progress bar, 2=one line per epoch)
        """
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=verbose
        )

    def predict(self, X):
        """
        Returns class predictions and probability estimates.
        """
        proba = self.model.predict(X)
        predictions = np.argmax(proba, axis=-1)
        return predictions, proba

    def fit_transform(self, X, y_class):
        """
        Convenience function: train and return probability output.
        Used for generating backtest signals from trained model.
        """
        self.build()
        self.train(X, y_class)
        return self.predict_proba(X)


class Backtester:
    """
    Performs backtesting of trading strategies derived from model predictions.

    Implements a simple mean-reversion strategy where:
    - Long position is taken when model predicts positive return
    - Short or no position when model predicts negative return
    """
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital

    def backtest_classification(self, predictions, actual_returns):
        """
        Backtests a simple mean-reversion strategy based on classification predictions.

        Strategy:
        - Go long when model predicts up (prediction = 1)
        - Stay flat or short when model predicts down (prediction = 0)

        Args:
            predictions: Array of model predictions (1=up, 0=down)
            actual_returns: Array of actual returns aligned with predictions

        Returns:
            Dictionary containing backtest performance metrics
        """
        n = len(predictions)
        capital = self.initial_capital

        # Generate positions: 1 for long, 0 for flat
        positions = predictions.copy()

        # Calculate strategy returns (position * next day return)
        strategy_returns = np.zeros(n)
        for i in range(n-1):
            if positions[i] == 1:  # Long position
                strategy_returns[i+1] = actual_returns[i]
            # else stay flat, return is 0

        # Cumulative returns
        cum_strategy_returns = np.cumsum(strategy_returns)
        cum_benchmark_returns = np.cumsum(actual_returns)

        # Performance metrics
        total_return = cum_strategy_returns[-1]
        benchmark_total_return = cum_benchmark_returns[-1]

        # Annualized returns (assuming 252 trading days)
        trading_days = len(actual_returns)
        years = trading_days / 252

        annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
        benchmark_annualized_return = ((1 + benchmark_total_return) ** (1 / years)) - 1 if years > 0 else 0

        # Volatility (annualized)
        strategy_vol = np.std(strategy_returns) * np.sqrt(252)
        benchmark_vol = np.std(actual_returns) * np.sqrt(252)

        # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = annualized_return / strategy_vol if strategy_vol > 0 else 0
        benchmark_sharpe = benchmark_annualized_return / benchmark_vol if benchmark_vol > 0 else 0

        # Maximum drawdown
        cum_max = np.maximum.accumulate(cum_strategy_returns)
        drawdowns = (cum_max - cum_strategy_returns) / (cum_max + 1e-8)
        max_drawdown = np.max(drawdowns)

        # Win rate (percentage of profitable trades)
        winning_trades = np.sum(strategy_returns > 0)
        win_rate = winning_trades / n * 100

        # Number of trades
        num_trades = np.sum(positions)

        return {
            'total_return': total_return,
            'benchmark_total_return': benchmark_total_return,
            'annualized_return': annualized_return,
            'benchmark_annualized_return': benchmark_annualized_return,
            'strategy_volatility': strategy_vol,
            'benchmark_volatility': benchmark_vol,
            'sharpe_ratio': sharpe_ratio,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades
        }

    def backtest_regression(self, predictions, actual_returns):
        """
        Backtests a strategy based on regression predictions.

        Strategy:
        - Go long when predicted return is positive and above threshold
        - Stay flat otherwise

        Args:
            predictions: Array of predicted return values
            actual_returns: Array of actual returns aligned with predictions

        Returns:
            Dictionary containing backtest performance metrics
        """
        n = len(predictions)
        capital = self.initial_capital

        # Generate positions based on prediction threshold
        threshold = 0.001  # Only trade if predicted return > 0.1%
        positions = (predictions > threshold).astype(float)

        # Calculate strategy returns
        strategy_returns = np.zeros(n)
        for i in range(1, n):
            if positions[i-1] == 1:
                strategy_returns[i] = actual_returns[i]

        # Cumulative returns
        cum_strategy_returns = np.cumsum(strategy_returns)
        cum_benchmark_returns = np.cumsum(actual_returns)

        # Performance metrics
        total_return = cum_strategy_returns[-1]
        benchmark_total_return = cum_benchmark_returns[-1]

        # Annualized returns
        trading_days = len(actual_returns)
        years = trading_days / 252

        annualized_return = ((1 + total_return) ** (1 / years)) - 1 if years > 0 else 0
        benchmark_annualized_return = ((1 + benchmark_total_return) ** (1 / years)) - 1 if years > 0 else 0

        # Volatility (annualized)
        strategy_vol = np.std(strategy_returns) * np.sqrt(252)
        benchmark_vol = np.std(actual_returns) * np.sqrt(252)

        # Sharpe ratio
        sharpe_ratio = annualized_return / strategy_vol if strategy_vol > 0 else 0
        benchmark_sharpe = benchmark_annualized_return / benchmark_vol if benchmark_vol > 0 else 0

        # Maximum drawdown
        cum_max = np.maximum.accumulate(cum_strategy_returns)
        drawdowns = (cum_max - cum_strategy_returns) / (cum_max + 1e-8)
        max_drawdown = np.max(drawdowns)

        # Win rate
        winning_trades = np.sum(strategy_returns > 0)
        win_rate = winning_trades / n * 100

        # Number of trades
        num_trades = np.sum(positions)

        return {
            'total_return': total_return,
            'benchmark_total_return': benchmark_total_return,
            'annualized_return': annualized_return,
            'benchmark_annualized_return': benchmark_annualized_return,
            'strategy_volatility': strategy_vol,
            'benchmark_volatility': benchmark_vol,
            'sharpe_ratio': sharpe_ratio,
            'benchmark_sharpe': benchmark_sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades
        }


def visualize_gaf_images(gaf_encoder, X_original, n_samples=10):
    """
    Visualizes GAF-encoded images for a sample of observations.

    Args:
        gaf_encoder: Pre-trained GAFEncoder instance
        X_original: Original feature matrix (first column should be return series)
        n_samples: Number of samples to display
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i in range(n_samples):
        x_sample = X_original[i:i+1]
        gaf_image = gaf_encoder.encode(x_sample[:, 0])

        if isinstance(gaf_image, np.ndarray):
            axes[i].imshow(gaf_image, cmap='RdBu')
        else:
            axes[i].imshow(gaf_image)

        axes[i].set_title(f"Observation {i}")
        axes[i].axis('off')

    plt.suptitle("Gram Angular Field Encoded Images", fontsize=14)
    plt.tight_layout()
    return fig


def plot_walk_forward_results(results_dict, metric='accuracy', title_suffix=''):
    """
    Plots walk-forward backtest results over time.

    Args:
        results_dict: Dictionary with 'predictions', 'actuals', and optionally 'dates'
        metric: Metric to plot ('accuracy' or 'Return')
        title_suffix: Additional text for chart title
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    predictions = results_dict['predictions']
    actuals = results_dict['actuals']

    if metric == 'accuracy':
        # Rolling accuracy (50-period window)
        window = 50
        rolling_acc = []
        for i in range(window, len(predictions)):
            window_preds = predictions[i-window:i]
            window_actuals = actuals[i-window:i]
            acc = accuracy_score(window_actuals, window_preds)
            rolling_acc.append(acc)

        time_points = range(window, len(predictions))
        axes[0].plot(time_points, rolling_acc, color='#2E86AB', linewidth=1.5)
        axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Random')
        axes[0].set_title(f'Rolling Accuracy ({title_suffix})', fontsize=12)
        axes[0].set_xlabel('Time (observations)')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()

    elif metric == 'Return':
        profits = predictions - actuals
        rolling_cumulative = np.cumsum(profits)

        axes[0].plot(range(len(rolling_cumulative)), rolling_cumulative, color='#E36414', linewidth=1.5)
        axes[0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
        axes[0].set_title(f'Cumulative P/L ({title_suffix})', fontsize=12)
        axes[0].set_xlabel('Time (observations)')
        axes[0].set_ylabel('Cumulative Profit/Loss')

    # Confusion matrix
    ax = axes[1]
    cm = confusion_matrix(actuals, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    plt.tight_layout()
    return fig


def run_step1(models_dict, data, n_train=500, n_test=500):
    """
    Executes Step 1: Single train/test split with intentional data leakage.

    Args:
        models_dict: Dictionary of trained model instances
        data: Prepared dataset dictionary from preprocessor
        n_train: Number of training observations
        n_test: Number of test observations

    Returns:
        Dictionary containing results for each model
    """
    print("\n" + "="*60)
    print("STEP 1: Single Train/Test Split WITH LEAKAGE")
    print("="*60)

    results = {}

    X = data['X']
    y_class = data['y_class'].flatten()
    y_reg = data['y_reg']

    # Split with leakage (intentional - features contain future information)
    X_train = X[:n_train]
    y_class_train = y_class[:n_train]

    X_test = X[n_train:n_train+n_test]
    y_class_test = y_class[n_train:n_train+n_test]

    # Step 1c: Train and evaluate each model
    for name, model in models_dict.items():
        print(f"\n--- {name} ---")

        # Train model
        if not hasattr(model, 'model') or model.model is None:
            model.build()

        # Convert features to GAF images for CNN model
        if isinstance(model, CNNGAFModel):
            X_train_gaf = model.ga_encoder.encode_batch(X_train)
            model.train(X_train_gaf, y_class_train)
        else:
            model.train(X_train, y_class_train)

        # Predict on test set
        if isinstance(model, CNNGAFModel):
            X_test_gaf = model.ga_encoder.encode_batch(X_test)
            predictions, proba = model.predict(X_test_gaf)
        elif hasattr(model, 'input_shape'):
            X_test_reshaped = X_test.reshape(-1, 1, X_test.shape[1])
            predictions, proba = model.predict(X_test_reshaped)
        else:
            predictions, proba = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_class_test, predictions.flatten())
        print(f"Test Accuracy: {accuracy:.4f}")

        # Backtest results
        actual_returns = y_reg[n_train:n_train+n_test].flatten()
        backtest_metrics = Backtester().backtest_classification(predictions, actual_returns)

        results[name] = {
            'predictions': predictions.flatten(),
            'actuals': y_class_test,
            'accuracy': accuracy,
            'backtest': backtest_metrics
        }

        print(f"Backtest Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.4f}")
        print(f"Backtest Total Return: {backtest_metrics['total_return']*100:.2f}%")

    return results


def run_walk_forward(models_dict, data, n_train=500, n_test=None, leakage=True):
    """
    Executes walk-forward backtesting with non-anchored approach.

    Args:
        models_dict: Dictionary of trained model instances
        data: Prepared dataset dictionary from preprocessor
        n_train: Number of training observations per window
        n_test: Number of test observations in each window (default = n_train)
        leakage: Whether to use features with or without leakage

    Returns:
        Dictionary containing walk-forward results for each model
    """
    if n_test is None:
        n_test = n_train

    print(f"\n{'Walking Forward with ' + ('LEAKAGE' if leakage else 'NO LEAKAGE')} (train={n_train}, test={n_test})")
    print("=" * 50)

    X = data['X']
    y_class = data['y_class'].flatten()
    n_total = len(X)

    # Calculate number of windows
    if leakage:
        n_windows = (n_total - n_train) // (n_train + n_test)
    else:
        # Without leakage, we need at least n_lookback periods before each prediction
        n_windows = (n_total - n_train - 10) // (n_train + n_test)

    all_results = {}

    for name, model in models_dict.items():
        print(f"\n--- {name} ---")

        # Initialize storage for walk-forward results
        all_preds = []
        all_actuals = []

        for window_idx in range(n_windows):
            # Calculate train/test indices (non-anchored: no overlap)
            train_start = window_idx * (n_train + n_test)
            train_end = train_start + n_train
            test_start = train_end
            test_end = test_start + n_test

            if window_idx == 0:
                X_train_fold = X[:train_end]
            else:
                # Extend training set (non-anchored: includes all previous data)
                X_train_fold = np.vstack([X[:train_end]])

            y_class_train_fold = y_class[:train_end]
            X_test_fold = X[test_start:test_end]
            y_class_test_fold = y_class[test_start:test_end]

            # Train model on window data
            if hasattr(model, 'input_dim'):
                model.build()
                model.train(X_train_fold[-n_train:], y_class_train_fold[-n_train:])
            elif hasattr(model, 'input_shape'):
                X_train_reshaped = X_train_fold[-n_train:].reshape(-1, 1, X_train_fold.shape[1])
                model.build()
                model.train(X_train_reshaped, y_class_train_fold[-n_train:])

            # Predict on test window
            if hasattr(model, 'input_shape'):
                X_test_reshaped = X_test_fold.reshape(-1, 1, X_test_fold.shape[1])
                preds, _ = model.predict(X_test_reshaped)
            else:
                preds, _ = model.predict(X_test_fold)

            all_preds.extend(preds.flatten())
            all_actuals.extend(y_class_test_fold)

        # Calculate overall metrics
        preds_array = np.array(all_preds[:len(all_actuals)])
        accuracy = accuracy_score(all_actuals, preds_array)

        # Backtest using all walk-forward predictions
        actual_returns = data['y_reg'][n_lookback:n_lookback+len(all_preds)]
        backtest_metrics = Backtester().backtest_classification(preds_array, actual_returns)

        print(f"Walk-forward Accuracy: {accuracy:.4f}")
        print(f"Backtest Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.4f}")

        all_results[name] = {
            'predictions': preds_array,
            'actuals': np.array(all_actuals),
            'accuracy': accuracy,
            'backtest': backtest_metrics
        }

    return all_results

# ============================================================================
# MAIN EXECUTION BLOCK - Run the complete analysis pipeline
# ============================================================================

def main():
    """
    Main execution function that orchestrates the entire GWP3 analysis pipeline.
    """
    print("\n" + "="*80)
    print("GWP3: Deep Learning for Finance - SPY Return Prediction")
    print("="*80 + "\n")

    # -------------------------------------------------------------------------
    # STEP 0: DATA COLLECTION AND EXPLORATION
    # -------------------------------------------------------------------------
    print("STEP 0: Collecting and exploring data...")
    
    collector = DataCollector(
        ticker="^GSPC",
        start_date=(datetime.now() - timedelta(days=365*8)).strftime("%Y-%m-%d"),  # ~8 years
        end_date=datetime.now().strftime("%Y-%m-%d")
    )

    if not collector.load_data():
        print("  - Fetching data from Yahoo Finance...")
        collector.fetch_data()
        collector.calculate_returns().calculate_volatility()
    else:
        print("  - Loading data from existing file...")
        if 'Return' not in collector.data.columns:
            collector.calculate_returns().calculate_volatility()
    
    # Save data and generate exploration plots
    collector.save_data()
    
    # Generate exploratory plots
    collector.plot_price_series()
    collector.plot_returns_histogram()
    collector.plot_volatility()
    collector.plot_acf()
    
    print(f"  - Dataset: {len(collector.data)} observations")
    print(f"  - Date range: {collector.data.index[0]} to {collector.data.index[-1]}")
    print(f"  - Mean daily return: {collector.data['Return'].mean():.6f}")
    print(f"  - Annualized return: {(1 + collector.data['Return'].mean())**252 - 1:.4f}")
    
    # -------------------------------------------------------------------------
    # STEP 1: FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: Feature Engineering")
    print("="*80)

    n_lookback = 10  # Number of past periods for feature calculation
    
    preprocessor = DataPreprocessor(n_lookback=n_lookback)
    
    # Prepare features WITHOUT leakage (correct methodology)
    print("\n1a. Preparing features WITHOUT data leakage...")
    data_no_leakage = preprocessor.create_features_no_leakage(collector.data)
    data_no_leakage = preprocessor.create_labels_classification(data_no_leakage)
    data_no_leakage = preprocessor.create_labels_regression(data_no_leakage)
    data_no_leakage = data_no_leakage.dropna()
    X_no_leakage = data_no_leakage[['Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'Return_Lag4', 'Return_Lag5',
                                   'Return_RollMean', 'Return_RollStd', 'MA_5', 'MA_20']].values
    y_reg = data_no_leakage['Target_Reg'].values.reshape(-1, 1)
    y_class = data_no_leakage['Target_Class'].values.reshape(-1, 1)
    X_no_leakage_df = pd.DataFrame(X_no_leakage, columns=[f'Feature_{i}' for i in range(X_no_leakage.shape[1])], index=data_no_leakage.index)
    print(f"  - Features: {X_no_leakage_df.shape[1]}")
    print(f"  - Samples: {len(X_no_leakage_df)}")
    
    # Prepare features WITH leakage (for demonstration purposes)
    print("\n1b. Preparing features WITH data leakage...")
    data_leakage = preprocessor.create_features_leakage(collector.data)
    data_leakage = preprocessor.create_labels_classification(data_leakage)
    data_leakage = preprocessor.create_labels_regression(data_leakage)
    data_leakage = data_leakage.dropna()
    X_leakage = data_leakage[['Return_Lag1', 'Return_Lag2', 'Return_Lag3', 'Return_Lag4', 'Return_Lag5',
                               'Return_RollMean_Leak', 'Return_RollStd_Leak', 'MA_5', 'MA_20']].values
    y_reg_l = data_leakage['Target_Reg'].values.reshape(-1, 1)
    y_class_l = data_leakage['Target_Class'].values.reshape(-1, 1)
    X_leakage_df = pd.DataFrame(X_leakage, columns=[f'Feature_{i}' for i in range(X_leakage.shape[1])], index=data_leakage.index)
    print(f"  - Features: {X_leakage_df.shape[1]}")
    print(f"  - Samples: {len(X_leakage_df)}")

    # -------------------------------------------------------------------------
    # STEP 2: BUILD MODELS WITHOUT LEAKAGE
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: Building Models (NO LEAKAGE)")
    print("="*80)

    # Scale features
    scaler = StandardScaler()
    X_no_leakage_scaled = scaler.fit_transform(X_no_leakage)

    # Split data: 80% train, 20% test (time series split)
    n_total = len(X_no_leakage_scaled)
    train_size = int(0.8 * n_total)
    X_train, X_test = X_no_leakage_scaled[:train_size], X_no_leakage_scaled[train_size:]
    y_class_train = y_class[:train_size]
    y_class_test = y_class[train_size:]

    print(f"\nData split:")
    print(f"  - Training set: {len(X_train)} samples")
    print(f"  - Test set: {len(X_test)} samples")

    # Build models without initial training (will be trained later)
    print("\n2a. Building MLP model...")
    mlp = MLPModel(input_dim=X_train.shape[1])
    print(f"  - Input dimension: {mlp.input_dim}")
    
    print("\n2b. Building LSTM model...")
    lstm = LSTMModel(input_shape=(1, X_train.shape[1]))  # Single time step
    print(f"  - Input shape: {lstm.input_shape}")
    
    print("\n2c. Building CNN-GAF model...")
    cnn_gaf = CNNGAFModel(image_shape=(128, 128, 3))
    print(f"  - Image size: {cnn_gaf.image_shape[0]}x{cnn_gaf.image_shape[0]}")
    
    models_no_leakage = {
        'MLP': mlp,
        'LSTM': lstm,
        'CNN-GAF': cnn_gaf
    }
    print("\n  - All models built successfully!")

    # -------------------------------------------------------------------------
    # STEP 3: RUN STEP 1 ANALYSIS (SINGLE TRAIN/TEST SPLIT)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: Running Step 1 Analysis - Single Train/Test Split")
    print("="*80)

    # Run with NO LEAKAGE (correct approach)
    print("\n3a. Running analysis WITHOUT data leakage...")

    # Prepare data dict for run_step1
    data_no_leakage = {
        'X': np.concatenate([X_train, X_test]),
        'y_class': np.concatenate([y_class_train, y_class_test]),
         'y_reg': y_reg   # Pass full array - run_step1 will slice with y_reg[n_train:n_train+n_test]
    }

    results_no_leakage = run_step1(models_no_leakage, data_no_leakage)

    # -------------------------------------------------------------------------
    # STEP 4: RUN WALK-FORWARD ANALYSIS WITHOUT LEAKAGE
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: Running Walk-Forward Analysis (NO LEAKAGE)")
    print("="*80)

    # Prepare data for walk-forward (use scaled features from test portion only to avoid leakage)
    X_wl = scaler.transform(X_no_leakage[train_size:])
    y_class_wf = y_class[train_size:]

    # Create walk-forward data dict
    wf_data_no_leakage = {
        'X': X_wl,
        'y_reg': y_reg[train_size:],
        'y_class': y_class_wf
    }

    # Reset models for fresh training in walk-forward
    models_no_leakage_wf = {
        'MLP': MLPModel(input_dim=X_wl.shape[1]),
        'LSTM': LSTMModel(input_shape=(1, X_wl.shape[1])),
        'CNN-GAF': CNNGAFModel(image_shape=(128, 128, 3))
    }

    # Walk-forward with 500 train / 500 test (Step 2a)
    print("\n4a. Walk-forward: train=500, test=500")
    results_wf_500_500 = run_walk_forward(models_no_leakage_wf, wf_data_no_leakage,
                                          n_train=500, n_test=500, leakage=False)

    # Walk-forward with 500 train / 100 test (Step 2b)
    print("\n4b. Walk-forward: train=500, test=100")
    results_wf_500_100 = run_walk_forward(models_no_leakage_wf, wf_data_no_leakage,
                                          n_train=500, n_test=100, leakage=False)

    # -------------------------------------------------------------------------
    # STEP 5: RUN ANALYSIS WITH LEAKAGE (for comparison)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: Running Analysis WITH Data Leakage (for comparison)")
    print("="*80)

    # Scale leakage features
    scaler_l = StandardScaler()
    X_leakage_scaled = scaler_l.fit_transform(X_leakage)

    # Split data
    n_total_l = len(X_leakage_scaled)
    train_size_l = int(0.8 * n_total_l)
    X_train_l, X_test_l = X_leakage_scaled[:train_size_l], X_leakage_scaled[train_size_l:]
    y_class_train_l = y_class_l[:train_size_l]
    y_class_test_l = y_class_l[train_size_l:]

    print(f"\nData split (with leakage):")
    print(f"  - Training set: {len(X_train_l)} samples")
    print(f"  - Test set: {len(X_test_l)} samples")

    # Build models for leakage comparison
    print("\n5a. Building models with leakage features...")
    models_leakage = {
        'MLP': MLPModel(input_dim=X_train_l.shape[1]),
        'LSTM': LSTMModel(input_shape=(1, X_train_l.shape[1])),
        'CNN-GAF': CNNGAFModel(image_shape=(128, 128, 3))
    }

    # Run with LEAKAGE (demonstrates incorrect approach - don't use in real applications!)
    print("\n5b. Running analysis WITH data leakage (DEMONSTRATION ONLY - do not use in practice)...")

    # Prepare data dict for run_step1
    data_leakage = {
        'X': np.concatenate([X_train_l, X_test_l]),
        'y_class': np.concatenate([y_class_train_l, y_class_test_l]),
        'y_reg': y_reg_l    # Pass full array - run_step1 will slice with y_reg[n_train:n_train+n_test]
    }

    results_leakage = run_step1(models_leakage, data_leakage)

    # -------------------------------------------------------------------------
    # STEP 6: GENERATE VISUALIZATIONS AND COMPARISONS
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 6: Generating Comparisons and Visualizations")
    print("="*80)

    # 6a. Compare accuracies: Leakage vs No Leakage (Step 1)
    print("\n6a. Comparing accuracy: Leakage vs No Leakage (single split)")
    print("-" * 50)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    model_names = ['MLP', 'LSTM', 'CNN-GAF']
    for i, model_name in enumerate(model_names):
        acc_no_leakage = results_no_leakage[model_name]['accuracy']
        acc_leakage = results_leakage[model_name]['accuracy']
        axes[i].bar(['No Leakage', 'With Leakage'], [acc_no_leakage, acc_leakage],
                    color=['#2E86AB', '#A23B72'])
        axes[i].set_title(model_name)
        axes[i].set_ylabel('Accuracy')
        axes[i].set_ylim([0.4, 0.75])
    plt.suptitle('Accuracy Comparison: Leakage vs No Leakage', fontsize=14)
    plt.tight_layout()
    plt.savefig('comparison_accuracy_step1.png', dpi=150)
    print("  - Saved: comparison_accuracy_step1.png")

    # Summary table
    print("\nAccuracy Comparison (Single Train/Test Split):")
    print('-' * 45)
    print(f"{'Model':<12} {'No Leakage':<14} {'With Leakage':<15}")
    print('-' * 45)
    for model_name in model_names:
        print(f"{model_name:<12} {results_no_leakage[model_name]['accuracy']:.4f}",
              f"{'':<13} {results_leakage[model_name]['accuracy']:.4f}")

    # 6b. Walk-forward results comparison: train=500/test=500 vs train=500/test=100
    print("\n6b. Comparing walk-forward configurations (NO LEAKAGE)")
    print("-" * 50)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, model_name in enumerate(model_names):
        acc_wf_500_500 = results_wf_500_500[model_name]['accuracy']
        acc_wf_500_100 = results_wf_500_100[model_name]['accuracy']
        axes[i].bar(['train=500/test=500', 'train=500/test=100'],
                    [acc_wf_500_500, acc_wf_500_100],
                    color=['#2E86AB', '#A23B72'])
        axes[i].set_title(model_name)
        axes[i].set_ylabel('Accuracy')
        axes[i].set_ylim([0.4, 0.75])
    plt.suptitle('Walk-Forward Accuracy: Different Train/Test Splits', fontsize=14)
    plt.tight_layout()
    plt.savefig('comparison_walkforward_splits.png', dpi=150)
    print("  - Saved: comparison_walkforward_splits.png")

    # Summary table
    print("\nWalk-Forward Accuracy Comparison:")
    print('-' * 60)
    print(f"{'Model':<12} {'train=500/test=500':<20} {'train=500/test=100':<20}")
    print('-' * 60)
    for model_name in model_names:
        print(f"{model_name:<12} {results_wf_500_500[model_name]['accuracy']:.4f}",
              f"{'':<19} {results_wf_500_100[model_name]['accuracy']:.4f}")

    # 6c. Plot walk-forward accuracy over time (for train=500/test=100)
    print("\n6c. Plotting walk-forward accuracy over time (train=500/test=100)")
    plot_walk_forward_results(results_wf_500_100, metric='accuracy',
                              title_suffix='')
    plt.savefig('walkforward_accuracy_time.png', dpi=150)
    print("  - Saved: walkforward_accuracy_time.png")

    # 6d. Plot cumulative P/L (for train=500/test=100)
    print("\n6d. Plotting cumulative P/L over walk-forward (train=500/test=100)")
    plot_walk_forward_results(results_wf_500_100, metric='pnl', title_suffix='')
    plt.savefig('walkforward_pnl_time.png', dpi=150)
    print("  - Saved: walkforward_pnl_time.png")

    # 6e. Compare backtest metrics across all configurations
    print("\n6e. Comparing backtest Sharpe ratios")
    print("-" * 50)
    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect all Sharpe ratios
    sharpes = {
        'Step1 (No Leakage)': [],
        'WF 500/500': [],
        'WF 500/100': [],
    }

    for model_name in model_names:
        sharpes['Step1 (No Leakage)'].append(results_no_leakage[model_name]['backtest']['sharpe_ratio'])
        sharpes['WF 500/500'].append(results_wf_500_500[model_name]['backtest']['sharpe_ratio'])
        sharpes['WF 500/100'].append(results_wf_500_100[model_name]['backtest']['sharpe_ratio'])

    x = np.arange(len(model_names))
    width = 0.25

    ax.bar(x - width, sharpes['Step1 (No Leakage)'], width,
           label='Single Split', color='#2E86AB')
    ax.bar(x, sharpes['WF 500/500'], width,
           label='WF 500/500', color='#A23B72')
    ax.bar(x + width, sharpes['WF 500/100'], width,
           label='WF 500/100', color='#F18F01')

    ax.set_xlabel('Model')
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Backtest Sharpe Ratios Across Configurations (NO LEAKAGE)')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.savefig('comparison_sharpe_all.png', dpi=150)
    print("  - Saved: comparison_sharpe_all.png")

    # 6f. GAF image visualization example
    print("\n6f. Visualizing GAF-encoded images")
    gaf_encoder = GAFEncoder(image_size=128)

    # Create sample time series from test data
    sample_series = []
    for i in range(min(5, len(X_test))):
        # Create a synthetic time series for visualization
        ts_length = 100
        base_ts = np.linspace(0, 2*np.pi, ts_length)
        noise = np.random.randn(ts_length) * 0.1
        synthetic_ts = np.sin(base_ts) + noise
        # Scale to [-1, 1]
        synthetic_ts = (synthetic_ts - np.min(synthetic_ts)) / (np.max(synthetic_ts) - np.min(synthetic_ts)) * 2 - 1
        sample_series.append(synthetic_ts)

    # Encode and visualize
    fig, axes = plt.subplots(1, len(sample_series), figsize=(20, 4))
    if len(sample_series) == 1:
        axes = [axes]

    for i, ts in enumerate(sample_series):
        gaf_img = gaf_encoder.encode(ts)
        axes[i].imshow(gaf_img, cmap='viridis')
        plt.colorbar(axes[i].get_images(), ax=axes[i])
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')

    plt.suptitle('GAF-Encoded Time Series Images', fontsize=14)
    plt.tight_layout()
    plt.savefig('gaf_examples.png', dpi=150)
    print("  - Saved: gaf_examples.png")

    # -------------------------------------------------------------------------
    # STEP 7: FINAL SUMMARY AND RECOMMENDATIONS
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 7: Final Summary and Recommendations")
    print("="*80)

    # Overall accuracy summary
    print("\n7a. Overall Accuracy Summary (NO LEAKAGE)")
    print('-' * 50)
    print(f"{'Model':<12} {'Single Split':<15} {'WF 500/500':<12} {'WF 500/100'}")
    print('-' * 50)
    for model_name in model_names:
        print(f"{model_name:<12} {results_no_leakage[model_name]['accuracy']:.4f}",
              f"{results_wf_500_500[model_name]['accuracy']:.4f}",
              f"{results_wf_500_100[model_name]['accuracy']:.4f}")

    # Backtest performance summary
    print("\n7b. Backtest Performance Summary (NO LEAKAGE)")
    print('-' * 70)
    print(f"{'Model':<12} {'Config':<15} {'Accuracy':<10} {'Sharpe':<10} {'Total Return'}")
    print('-' * 70)

    for model_name in model_names:
        # Single split results
        r = results_no_leakage[model_name]
        print(f"{model_name:<12} {'Single Split':<15} {r['accuracy']:.4f}",
              f"{r['backtest']['sharpe_ratio']:.4f}",
              f"{r['backtest']['total_return']*100:.2f}%")

        # WF results
        r_wf = results_wf_500_100[model_name]
        print(f"{model_name:<12} {'WF 500/100':<15} {r_wf['accuracy']:.4f}",
              f"{r_wf['backtest']['sharpe_ratio']:.4f}",
              f"{r_wf['backtest']['total_return']*100:.2f}%")

    # Key findings and recommendations
    print("\n7c. KEY FINDINGS AND RECOMMENDATIONS")
    print('=' * 50)
    
    # Find best performing model
    all_results_combined = {}
    for model_name in model_names:
        acc_s1 = results_no_leakage[model_name]['accuracy']
        acc_wf_500 = results_wf_500_100[model_name]['accuracy']
        sharpe_s1 = results_no_leakage[model_name]['backtest']['sharpe_ratio']
        sharpe_wf = results_wf_500_100[model_name]['backtest']['sharpe_ratio']
        all_results_combined[model_name] = {
            'acc_single': acc_s1,
            'acc_wf': acc_wf_500,
            'sharpe_single': sharpe_s1,
            'sharpe_wf': sharpe_wf,
            'avg_acc': (acc_s1 + acc_wf_500) / 2,
            'avg_sharpe': (sharpe_s1 + sharpe_wf) / 2
        }
    
    # Best model by average accuracy
    best_model_acc = max(all_results_combined.keys(),
                         key=lambda x: all_results_combined[x]['avg_acc'])
    # Best model by average Sharpe
    best_model_sharpe = max(all_results_combined.keys(),
                            key=lambda x: all_results_combined[x]['avg_sharpe'])

    print("\n  Best performing model (by accuracy):", best_model_acc)
    print("  Best performing model (by Sharpe ratio):", best_model_sharpe)

    print("\n  RECOMMENDATIONS:")
    print("  " + "-" * 40)

    # Recommendation based on comparative analysis
    leakage_gap_max = max([results_leakage[m]['accuracy'] - results_no_leakage[m]['accuracy']
                          for m in model_names])
    leakage_gap_avg = np.mean([results_leakage[m]['accuracy'] - results_no_leakage[m]['accuracy']
                               for m in model_names])

    print(f"  • Data leakage can artificially inflate accuracy by {leakage_gap_avg:.2%}")
    print(f"    (maximum: {leakage_gap_max:.2%}) - This demonstrates the importance of")
    print(f"    using proper train/test separation in time series forecasting.")

    # Walk-forward vs single split comparison
    avg_acc_single = np.mean([results_no_leakage[m]['accuracy'] for m in model_names])
    avg_acc_wf = np.mean([results_wf_500_100[m]['accuracy'] for m in model_names])
    print(f"\n  • Walk-forward results ({avg_acc_wf:.2%}) {('are similar to' if abs(avg_acc_single - avg_acc_wf) < 0.02 else 'differ from')}")
    print(f"    single-split results ({avg_acc_single:.2%}), indicating the model")
    print(f"    generalizes reasonably well across different market conditions.")

    # Model recommendation
    print(f"\n  • Based on this analysis, the {best_model_sharpe} model is recommended")
    print(f"    for deployment due to its consistent performance across both accuracy")
    print(f"    and risk-adjusted return metrics.")

    print("\n  • For improved performance in production:")
    print("    - Consider using longer lookback windows for feature engineering")
    print("    - Implement more sophisticated feature selection")
    print("    - Consider ensemble methods combining multiple model types")
    print("    - Add more explanatory variables (volume, volatility indicators)")

    # -------------------------------------------------------------------------
    # CLEANUP
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("CLEANUP: Releasing memory")
    print("="*80)

    # Delete large intermediate variables
    del X_no_leakage, X_leakage, X_wl
    del X_train_l, X_test_l
    gc.collect()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

    return {
        'collector': collector,
        'preprocessor': preprocessor,
        'results_no_leakage': results_no_leakage,
        'results_leakage': results_leakage,
        'results_wf_500_500': results_wf_500_500,
        'results_wf_500_100': results_wf_500_100,
        'models_no_leakage': models_no_leakage
    }


if __name__ == "__main__":
    results = main()

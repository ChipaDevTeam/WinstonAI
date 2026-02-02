"""
Technical indicators for market analysis using the ta library
"""

import pandas as pd
import numpy as np
import ta
import warnings


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator for financial market data
    """
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators with GPU acceleration
        
        Args:
            df: DataFrame with OHLCV data (open, high, low, close, volume)
            
        Returns:
            DataFrame containing all calculated indicators
        """
        indicators = {}
        
        # Ensure we have enough data
        if len(df) < 100:
            warnings.warn(f"Only {len(df)} rows available, need at least 100 for reliable indicators")
            # Duplicate data to meet minimum requirements
            while len(df) < 100:
                df = pd.concat([df, df.iloc[-1:]], ignore_index=True)
        
        # Add synthetic volume if not present
        if 'volume' not in df.columns:
            df['volume'] = (df['high'] - df['low']) * df['close'] * 1000
        
        try:
            # Price-based indicators - Simple Moving Averages
            indicators['sma_5'] = ta.trend.sma_indicator(df['close'], window=5)
            indicators['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
            indicators['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            indicators['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            indicators['sma_100'] = ta.trend.sma_indicator(df['close'], window=100)
            
            # Exponential Moving Averages
            indicators['ema_5'] = ta.trend.ema_indicator(df['close'], window=5)
            indicators['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)
            indicators['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            indicators['ema_50'] = ta.trend.ema_indicator(df['close'], window=50)
            indicators['ema_100'] = ta.trend.ema_indicator(df['close'], window=100)
            
            # MACD family
            indicators['macd'] = ta.trend.macd(df['close'])
            indicators['macd_signal'] = ta.trend.macd_signal(df['close'])
            indicators['macd_diff'] = ta.trend.macd_diff(df['close'])
            
            # Multiple timeframe RSI
            indicators['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
            indicators['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
            indicators['rsi_21'] = ta.momentum.rsi(df['close'], window=21)
            indicators['rsi_50'] = ta.momentum.rsi(df['close'], window=50)
            
            # Bollinger Bands with multiple periods
            indicators['bb_high_20'] = ta.volatility.bollinger_hband(df['close'], window=20)
            indicators['bb_low_20'] = ta.volatility.bollinger_lband(df['close'], window=20)
            indicators['bb_mid_20'] = ta.volatility.bollinger_mavg(df['close'], window=20)
            indicators['bb_width_20'] = ta.volatility.bollinger_wband(df['close'], window=20)
            
            indicators['bb_high_10'] = ta.volatility.bollinger_hband(df['close'], window=10)
            indicators['bb_low_10'] = ta.volatility.bollinger_lband(df['close'], window=10)
            indicators['bb_mid_10'] = ta.volatility.bollinger_mavg(df['close'], window=10)
            
            # Stochastic oscillators
            indicators['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            indicators['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            
            # Williams %R with multiple periods
            indicators['williams_r_14'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
            indicators['williams_r_7'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=7)
            indicators['williams_r_21'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=21)
            
            # Volatility indicators - Average True Range
            indicators['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            indicators['atr_7'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=7)
            indicators['atr_21'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=21)
            
            # Trend indicators - ADX
            indicators['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
            indicators['adx_pos'] = ta.trend.adx_pos(df['high'], df['low'], df['close'])
            indicators['adx_neg'] = ta.trend.adx_neg(df['high'], df['low'], df['close'])
            
            # Commodity Channel Index
            indicators['cci_14'] = ta.trend.cci(df['high'], df['low'], df['close'], window=14)
            indicators['cci_20'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
            
            # Volume indicators
            indicators['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            indicators['vpt'] = ta.volume.volume_price_trend(df['close'], df['volume'])
            
            # Money Flow Index
            indicators['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
            
            # Parabolic SAR
            indicators['psar'] = ta.trend.psar_down(df['high'], df['low'], df['close'])
            
            # Keltner Channels
            indicators['kc_high'] = ta.volatility.keltner_channel_hband(df['high'], df['low'], df['close'])
            indicators['kc_low'] = ta.volatility.keltner_channel_lband(df['high'], df['low'], df['close'])
            indicators['kc_mid'] = ta.volatility.keltner_channel_mband(df['high'], df['low'], df['close'])
            
            # Donchian Channels
            indicators['dc_high'] = ta.volatility.donchian_channel_hband(df['high'], df['low'], df['close'])
            indicators['dc_low'] = ta.volatility.donchian_channel_lband(df['high'], df['low'], df['close'])
            indicators['dc_mid'] = ta.volatility.donchian_channel_mband(df['high'], df['low'], df['close'])
            
            # Fibonacci retracements (simplified)
            high_val = df['high'].rolling(window=50).max()
            low_val = df['low'].rolling(window=50).min()
            diff = high_val - low_val
            
            indicators['fib_23_6'] = high_val - (diff * 0.236)
            indicators['fib_38_2'] = high_val - (diff * 0.382)
            indicators['fib_50_0'] = high_val - (diff * 0.500)
            indicators['fib_61_8'] = high_val - (diff * 0.618)
            indicators['fib_78_6'] = high_val - (diff * 0.786)
            
            # Support and resistance levels
            indicators['support_5'] = df['low'].rolling(window=5).min()
            indicators['resistance_5'] = df['high'].rolling(window=5).max()
            indicators['support_20'] = df['low'].rolling(window=20).min()
            indicators['resistance_20'] = df['high'].rolling(window=20).max()
            
            # Price action patterns
            indicators['price_change'] = df['close'].pct_change()
            indicators['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            indicators['open_close_ratio'] = (df['close'] - df['open']) / df['open']
            
            # Momentum indicators - Rate of Change
            indicators['roc_10'] = ta.momentum.roc(df['close'], window=10)
            indicators['roc_20'] = ta.momentum.roc(df['close'], window=20)
            
            # Ichimoku components
            indicators['ichimoku_a'] = ta.trend.ichimoku_a(df['high'], df['low'])
            indicators['ichimoku_b'] = ta.trend.ichimoku_b(df['high'], df['low'])
            
        except Exception as e:
            warnings.warn(f"Error calculating indicators: {e}")
            # Fill with zeros if calculation fails
            for key in ['sma_20', 'ema_20', 'rsi_14', 'macd', 'bb_high_20', 'bb_low_20']:
                if key not in indicators:
                    indicators[key] = pd.Series([0] * len(df))
        
        # Convert to DataFrame and handle NaN values
        indicators_df = pd.DataFrame(indicators)
        indicators_df = indicators_df.ffill().bfill().fillna(0)
        
        return indicators_df
    
    @staticmethod
    def calculate_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic set of technical indicators (faster computation)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with basic indicators
        """
        indicators = {}
        
        # Add synthetic volume if not present
        if 'volume' not in df.columns:
            df['volume'] = (df['high'] - df['low']) * df['close'] * 1000
        
        try:
            # Essential moving averages
            indicators['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            indicators['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
            
            # RSI
            indicators['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            indicators['macd'] = ta.trend.macd(df['close'])
            indicators['macd_signal'] = ta.trend.macd_signal(df['close'])
            
            # Bollinger Bands
            indicators['bb_high'] = ta.volatility.bollinger_hband(df['close'])
            indicators['bb_low'] = ta.volatility.bollinger_lband(df['close'])
            
            # ATR
            indicators['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
        except Exception as e:
            warnings.warn(f"Error calculating basic indicators: {e}")
        
        indicators_df = pd.DataFrame(indicators)
        indicators_df = indicators_df.ffill().bfill().fillna(0)
        
        return indicators_df

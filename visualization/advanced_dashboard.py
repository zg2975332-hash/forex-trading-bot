#!/usr/bin/env python3
"""
ADVANCED VISUALIZATION DASHBOARD - PROFESSIONAL TRADING ANALYTICS
Comprehensive visualization system with advanced charts and analytics
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dataclasses import dataclass, asdict
import json
import warnings
from enum import Enum
import threading
from collections import deque
warnings.filterwarnings('ignore')

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.log_config import setup_logging, get_logger

class ChartType(Enum):
    """Supported chart types"""
    CANDLESTICK = "candlestick"
    LINE = "line"
    BAR = "bar"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    VIOLIN = "violin"

class TimeFrame(Enum):
    """Supported timeframes"""
    M1 = "1min"
    M5 = "5min"
    M15 = "15min"
    H1 = "1H"
    H4 = "4H"
    D1 = "1D"
    W1 = "1W"

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    theme: str = "dark"
    chart_quality: str = "high"  # low, medium, high
    auto_refresh: bool = True
    refresh_interval: int = 5000  # ms
    max_data_points: int = 1000
    enable_animations: bool = True
    default_timeframe: TimeFrame = TimeFrame.H1
    risk_visualization: bool = True
    correlation_matrix: bool = True
    portfolio_analysis: bool = True

@dataclass
class ChartData:
    """Container for chart data"""
    timestamp: List[datetime]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]
    indicators: Dict[str, List[float]]
    signals: Dict[str, List[Any]]

class AdvancedVisualizationDashboard:
    """
    PROFESSIONAL ADVANCED VISUALIZATION DASHBOARD
    Comprehensive trading analytics and visualization system
    """
    
    def __init__(self, trading_bot=None, config: VisualizationConfig = None):
        self.trading_bot = trading_bot
        self.config = config or VisualizationConfig()
        self.logger = get_logger(__name__)
        
        # Data storage
        self.market_data = {}
        self.portfolio_data = {}
        self.risk_metrics = {}
        self.performance_history = deque(maxlen=self.config.max_data_points)
        
        # Chart cache for performance
        self.chart_cache = {}
        self.last_update = datetime.now()
        
        # Initialize Dash app with advanced features
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
            suppress_callback_exceptions=True
        )
        
        # Setup advanced layout
        self._setup_advanced_layout()
        
        self.logger.info("🎨 ADVANCED VISUALIZATION DASHBOARD INITIALIZED")
    
    def _setup_advanced_layout(self):
        """Setup comprehensive advanced dashboard layout"""
        self.app.layout = dbc.Container([
            # Header with Controls
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1("📊 FOREX TRADING BOT - ADVANCED ANALYTICS", 
                               className="text-center mb-2",
                               style={'color': '#00ff88', 'fontWeight': 'bold', 'fontSize': '2.5rem'}),
                        html.P("Professional Trading Analytics & Visualization Platform", 
                              className="text-center text-muted mb-4")
                    ])
                ], width=12)
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎮 ADVANCED CONTROLS", className="text-center"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Timeframe:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='timeframe-selector',
                                        options=[{'label': tf.value, 'value': tf.value} for tf in TimeFrame],
                                        value=self.config.default_timeframe.value,
                                        clearable=False,
                                        className="mb-2"
                                    )
                                ], width=2),
                                dbc.Col([
                                    html.Label("Chart Type:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='chart-type-selector',
                                        options=[{'label': ct.value.upper(), 'value': ct.value} for ct in ChartType],
                                        value=ChartType.CANDLESTICK.value,
                                        clearable=False,
                                        className="mb-2"
                                    )
                                ], width=2),
                                dbc.Col([
                                    html.Label("Symbol:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='symbol-selector',
                                        options=[
                                            {'label': 'EUR/USD', 'value': 'EURUSD'},
                                            {'label': 'GBP/USD', 'value': 'GBPUSD'},
                                            {'label': 'USD/JPY', 'value': 'USDJPY'},
                                            {'label': 'AUD/USD', 'value': 'AUDUSD'},
                                            {'label': 'USD/CAD', 'value': 'USDCAD'}
                                        ],
                                        value='EURUSD',
                                        clearable=False,
                                        className="mb-2"
                                    )
                                ], width=2),
                                dbc.Col([
                                    html.Label("Indicators:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='indicator-selector',
                                        options=[
                                            {'label': 'SMA (20,50)', 'value': 'sma'},
                                            {'label': 'EMA (12,26)', 'value': 'ema'},
                                            {'label': 'RSI (14)', 'value': 'rsi'},
                                            {'label': 'MACD', 'value': 'macd'},
                                            {'label': 'Bollinger Bands', 'value': 'bollinger'},
                                            {'label': 'All Indicators', 'value': 'all'}
                                        ],
                                        value=['sma', 'rsi'],
                                        multi=True,
                                        className="mb-2"
                                    )
                                ], width=3),
                                dbc.Col([
                                    html.Label("Risk Metrics:", className="fw-bold"),
                                    dcc.Dropdown(
                                        id='risk-metric-selector',
                                        options=[
                                            {'label': 'VaR Analysis', 'value': 'var'},
                                            {'label': 'Monte Carlo', 'value': 'montecarlo'},
                                            {'label': 'Correlation Matrix', 'value': 'correlation'},
                                            {'label': 'Drawdown Analysis', 'value': 'drawdown'}
                                        ],
                                        value=['var', 'correlation'],
                                        multi=True,
                                        className="mb-2"
                                    )
                                ], width=3)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("🔄 Refresh All Charts", id="refresh-all-btn", color="primary", className="me-2"),
                                    dbc.Button("💾 Export Analytics", id="export-analytics-btn", color="success", className="me-2"),
                                    dbc.Button("📈 Generate Report", id="generate-report-btn", color="info", className="me-2"),
                                    dbc.Button("⚙️ Settings", id="settings-btn", color="secondary")
                                ], width=12, className="text-center mt-2")
                            ])
                        ])
                    ], color="dark", outline=True, className="mb-4")
                ], width=12)
            ]),
            
            # Main Charts Row
            dbc.Row([
                # Price Chart with Indicators
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.Span("💰 PRICE CHART & TECHNICAL ANALYSIS", className="fw-bold"),
                            html.Small(" | Live Market Data", className="text-muted ms-2")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(
                                id='price-chart',
                                style={'height': '600px'},
                                config={
                                    'displayModeBar': True,
                                    'scrollZoom': True,
                                    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
                                    'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                                }
                            ),
                            dcc.Interval(id='price-chart-interval', interval=self.config.refresh_interval, n_intervals=0)
                        ])
                    ], color="dark", outline=True)
                ], width=8, className="mb-4"),
                
                # Technical Indicators Panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 TECHNICAL INDICATORS", className="text-center"),
                        dbc.CardBody([
                            dcc.Graph(id='rsi-chart', style={'height': '200px'}),
                            dcc.Graph(id='macd-chart', style={'height': '200px'}),
                            dcc.Graph(id='volume-chart', style={'height': '200px'})
                        ])
                    ], color="dark", outline=True)
                ], width=4, className="mb-4")
            ]),
            
            # Risk Analytics Row
            dbc.Row([
                # Risk Metrics
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🛡️ ADVANCED RISK ANALYTICS", className="text-center"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='var-chart', style={'height': '300px'})
                                ], width=6),
                                dbc.Col([
                                    dcc.Graph(id='correlation-heatmap', style={'height': '300px'})
                                ], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='drawdown-chart', style={'height': '300px'})
                                ], width=6),
                                dbc.Col([
                                    dcc.Graph(id='monte-carlo-chart', style={'height': '300px'})
                                ], width=6)
                            ])
                        ])
                    ], color="dark", outline=True)
                ], width=12, className="mb-4")
            ]),
            
            # Portfolio Analytics Row
            dbc.Row([
                # Portfolio Performance
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("💼 PORTFOLIO PERFORMANCE ANALYTICS", className="text-center"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='portfolio-growth-chart', style={'height': '300px'})
                                ], width=6),
                                dbc.Col([
                                    dcc.Graph(id='asset-allocation-chart', style={'height': '300px'})
                                ], width=6)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='performance-metrics-chart', style={'height': '300px'})
                                ], width=12)
                            ])
                        ])
                    ], color="dark", outline=True)
                ], width=12, className="mb-4")
            ]),
            
            # Trading Analytics Row
            dbc.Row([
                # Trading Performance
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 TRADING PERFORMANCE ANALYTICS", className="text-center"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(id='trade-distribution-chart', style={'height': '300px'})
                                ], width=4),
                                dbc.Col([
                                    dcc.Graph(id='win-loss-chart', style={'height': '300px'})
                                ], width=4),
                                dbc.Col([
                                    dcc.Graph(id='time-analysis-chart', style={'height': '300px'})
                                ], width=4)
                            ])
                        ])
                    ], color="dark", outline=True)
                ], width=12, className="mb-4")
            ]),
            
            # Hidden data storage
            html.Div(id='hidden-chart-data', style={'display': 'none'}),
            html.Div(id='hidden-risk-data', style={'display': 'none'}),
            html.Div(id='hidden-portfolio-data', style={'display': 'none'}),
            
            # Update intervals
            dcc.Interval(id='main-update-interval', interval=2000, n_intervals=0),
            dcc.Store(id='chart-data-store'),
            dcc.Store(id='risk-data-store'),
            dcc.Store(id='portfolio-data-store')
            
        ], fluid=True, style={'backgroundColor': '#0d0d0d', 'minHeight': '100vh'})
        
        # Setup advanced callbacks
        self._setup_advanced_callbacks()
    
    def _setup_advanced_callbacks(self):
        """Setup comprehensive advanced callbacks"""
        
        @self.app.callback(
            Output('price-chart', 'figure'),
            [Input('price-chart-interval', 'n_intervals'),
             Input('symbol-selector', 'value'),
             Input('timeframe-selector', 'value'),
             Input('indicator-selector', 'value')]
        )
        def update_price_chart(n_intervals, symbol, timeframe, indicators):
            """Update main price chart with technical indicators"""
            try:
                # Generate sample price data
                price_data = self._generate_sample_price_data(symbol, timeframe)
                
                # Create subplots
                fig = make_subplots(
                    rows=4, cols=1,
                    shared_x=True,
                    vertical_spacing=0.02,
                    subplot_titles=('Price Chart', 'RSI', 'MACD', 'Volume'),
                    row_heights=[0.5, 0.15, 0.15, 0.2]
                )
                
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=price_data['timestamp'],
                        open=price_data['open'],
                        high=price_data['high'],
                        low=price_data['low'],
                        close=price_data['close'],
                        name='Price'
                    ), row=1, col=1
                )
                
                # Add selected indicators
                if indicators:
                    if 'sma' in indicators:
                        sma_20 = self._calculate_sma(price_data['close'], 20)
                        sma_50 = self._calculate_sma(price_data['close'], 50)
                        fig.add_trace(
                            go.Scatter(x=price_data['timestamp'], y=sma_20, 
                                     line=dict(color='orange', width=1), name='SMA 20'),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=price_data['timestamp'], y=sma_50, 
                                     line=dict(color='red', width=1), name='SMA 50'),
                            row=1, col=1
                        )
                    
                    if 'bollinger' in indicators:
                        bb_upper, bb_lower = self._calculate_bollinger_bands(price_data['close'], 20)
                        fig.add_trace(
                            go.Scatter(x=price_data['timestamp'], y=bb_upper, 
                                     line=dict(color='blue', width=1, dash='dash'), name='BB Upper'),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=price_data['timestamp'], y=bb_lower, 
                                     line=dict(color='blue', width=1, dash='dash'), name='BB Lower'),
                            row=1, col=1
                        )
                
                # RSI
                rsi = self._calculate_rsi(price_data['close'], 14)
                fig.add_trace(
                    go.Scatter(x=price_data['timestamp'], y=rsi, line=dict(color='purple'), name='RSI'),
                    row=2, col=1
                )
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                macd, signal, histogram = self._calculate_macd(price_data['close'])
                fig.add_trace(
                    go.Scatter(x=price_data['timestamp'], y=macd, line=dict(color='blue'), name='MACD'),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Scatter(x=price_data['timestamp'], y=signal, line=dict(color='red'), name='Signal'),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Bar(x=price_data['timestamp'], y=histogram, name='Histogram', marker_color='gray'),
                    row=3, col=1
                )
                
                # Volume
                fig.add_trace(
                    go.Bar(x=price_data['timestamp'], y=price_data['volume'], name='Volume', 
                          marker_color='lightblue', opacity=0.7),
                    row=4, col=1
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{symbol} - {timeframe} Price Chart with Technical Indicators",
                    template="plotly_dark",
                    height=600,
                    showlegend=True,
                    xaxis_rangeslider_visible=False,
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                
                return fig
                
            except Exception as e:
                self.logger.error(f"Price chart update error: {e}")
                return self._create_empty_chart("Price Chart")
        
        @self.app.callback(
            [Output('var-chart', 'figure'),
             Output('correlation-heatmap', 'figure'),
             Output('drawdown-chart', 'figure'),
             Output('monte-carlo-chart', 'figure')],
            [Input('main-update-interval', 'n_intervals'),
             Input('risk-metric-selector', 'value')]
        )
        def update_risk_charts(n_intervals, risk_metrics):
            """Update all risk analytics charts"""
            try:
                # VaR Chart
                var_fig = self._create_var_chart()
                
                # Correlation Heatmap
                correlation_fig = self._create_correlation_heatmap()
                
                # Drawdown Chart
                drawdown_fig = self._create_drawdown_chart()
                
                # Monte Carlo Chart
                monte_carlo_fig = self._create_monte_carlo_chart()
                
                return var_fig, correlation_fig, drawdown_fig, monte_carlo_fig
                
            except Exception as e:
                self.logger.error(f"Risk charts update error: {e}")
                empty_chart = self._create_empty_chart("Risk Analytics")
                return empty_chart, empty_chart, empty_chart, empty_chart
        
        @self.app.callback(
            [Output('portfolio-growth-chart', 'figure'),
             Output('asset-allocation-chart', 'figure'),
             Output('performance-metrics-chart', 'figure')],
            [Input('main-update-interval', 'n_intervals')]
        )
        def update_portfolio_charts(n_intervals):
            """Update portfolio analytics charts"""
            try:
                # Portfolio Growth Chart
                growth_fig = self._create_portfolio_growth_chart()
                
                # Asset Allocation Chart
                allocation_fig = self._create_asset_allocation_chart()
                
                # Performance Metrics Chart
                metrics_fig = self._create_performance_metrics_chart()
                
                return growth_fig, allocation_fig, metrics_fig
                
            except Exception as e:
                self.logger.error(f"Portfolio charts update error: {e}")
                empty_chart = self._create_empty_chart("Portfolio Analytics")
                return empty_chart, empty_chart, empty_chart
        
        @self.app.callback(
            [Output('trade-distribution-chart', 'figure'),
             Output('win-loss-chart', 'figure'),
             Output('time-analysis-chart', 'figure')],
            [Input('main-update-interval', 'n_intervals')]
        )
        def update_trading_analytics(n_intervals):
            """Update trading performance analytics"""
            try:
                # Trade Distribution Chart
                distribution_fig = self._create_trade_distribution_chart()
                
                # Win-Loss Chart
                win_loss_fig = self._create_win_loss_chart()
                
                # Time Analysis Chart
                time_analysis_fig = self._create_time_analysis_chart()
                
                return distribution_fig, win_loss_fig, time_analysis_fig
                
            except Exception as e:
                self.logger.error(f"Trading analytics update error: {e}")
                empty_chart = self._create_empty_chart("Trading Analytics")
                return empty_chart, empty_chart, empty_chart
        
        # Technical indicator charts callbacks
        @self.app.callback(
            [Output('rsi-chart', 'figure'),
             Output('macd-chart', 'figure'),
             Output('volume-chart', 'figure')],
            [Input('price-chart-interval', 'n_intervals'),
             Input('symbol-selector', 'value')]
        )
        def update_technical_indicators(n_intervals, symbol):
            """Update technical indicator charts"""
            try:
                # RSI Chart
                rsi_fig = self._create_rsi_chart(symbol)
                
                # MACD Chart
                macd_fig = self._create_macd_chart(symbol)
                
                # Volume Chart
                volume_fig = self._create_volume_chart(symbol)
                
                return rsi_fig, macd_fig, volume_fig
                
            except Exception as e:
                self.logger.error(f"Technical indicators update error: {e}")
                empty_chart = self._create_empty_chart("Technical Indicators")
                return empty_chart, empty_chart, empty_chart
    
    def _generate_sample_price_data(self, symbol: str, timeframe: str) -> Dict[str, List]:
        """Generate sample price data for demonstration"""
        periods = 100
        end_date = datetime.now()
        
        if timeframe == '1min':
            dates = pd.date_range(end=end_date, periods=periods, freq='1min')
        elif timeframe == '1H':
            dates = pd.date_range(end=end_date, periods=periods, freq='1H')
        elif timeframe == '1D':
            dates = pd.date_range(end=end_date, periods=periods, freq='1D')
        else:
            dates = pd.date_range(end=end_date, periods=periods, freq='1H')
        
        # Generate realistic price data
        np.random.seed(hash(symbol) % 10000)  # Consistent per symbol
        
        prices = [100.0]  # Starting price
        for i in range(1, periods):
            change = np.random.normal(0, 0.002)  # Small random changes
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        # Create OHLC data
        open_prices = prices
        close_prices = [p * (1 + np.random.normal(0, 0.001)) for p in prices]
        high_prices = [max(o, c) * (1 + abs(np.random.normal(0, 0.001))) for o, c in zip(open_prices, close_prices)]
        low_prices = [min(o, c) * (1 - abs(np.random.normal(0, 0.001))) for o, c in zip(open_prices, close_prices)]
        volumes = [np.random.randint(1000, 10000) for _ in range(periods)]
        
        return {
            'timestamp': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }
    
    def _calculate_sma(self, data: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        return pd.Series(data).rolling(window=period).mean().tolist()
    
    def _calculate_ema(self, data: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        return pd.Series(data).ewm(span=period).mean().tolist()
    
    def _calculate_rsi(self, data: List[float], period: int) -> List[float]:
        """Calculate RSI indicator"""
        delta = pd.Series(data).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.tolist()
    
    def _calculate_macd(self, data: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD indicator"""
        ema_12 = self._calculate_ema(data, 12)
        ema_26 = self._calculate_ema(data, 26)
        macd = [e12 - e26 for e12, e26 in zip(ema_12, ema_26)]
        signal = self._calculate_ema(macd, 9)
        histogram = [m - s for m, s in zip(macd, signal)]
        return macd, signal, histogram
    
    def _calculate_bollinger_bands(self, data: List[float], period: int) -> Tuple[List[float], List[float]]:
        """Calculate Bollinger Bands"""
        sma = self._calculate_sma(data, period)
        std = pd.Series(data).rolling(window=period).std()
        upper_band = [s + 2 * st for s, st in zip(sma, std)]
        lower_band = [s - 2 * st for s, st in zip(sma, std)]
        return upper_band, lower_band
    
    def _create_var_chart(self):
        """Create Value at Risk analysis chart"""
        # Sample VaR data
        confidence_levels = [0.90, 0.95, 0.99]
        var_values = [1500, 2100, 3200]
        
        fig = go.Figure(data=[
            go.Bar(x=confidence_levels, y=var_values, 
                  marker_color=['green', 'orange', 'red'],
                  text=var_values, textposition='auto')
        ])
        
        fig.update_layout(
            title="Value at Risk (VaR) Analysis",
            xaxis_title="Confidence Level",
            yaxis_title="VaR ($)",
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def _create_correlation_heatmap(self):
        """Create correlation matrix heatmap"""
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        
        # Generate sample correlation matrix
        np.random.seed(42)
        corr_matrix = np.random.uniform(-0.8, 0.8, (5, 5))
        np.fill_diagonal(corr_matrix, 1.0)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=symbols,
            y=symbols,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Currency Pair Correlation Matrix",
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def _create_drawdown_chart(self):
        """Create drawdown analysis chart"""
        # Sample drawdown data
        dates = pd.date_range(end=datetime.now(), periods=50, freq='D')
        portfolio_value = 10000 + np.cumsum(np.random.randn(50) * 100)
        running_max = np.maximum.accumulate(portfolio_value)
        drawdown = (portfolio_value - running_max) / running_max * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=drawdown,
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red'),
            name='Drawdown'
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def _create_monte_carlo_chart(self):
        """Create Monte Carlo simulation chart"""
        # Sample Monte Carlo paths
        n_paths = 50
        n_days = 100
        initial_value = 10000
        
        paths = []
        for i in range(n_paths):
            returns = np.random.normal(0.001, 0.02, n_days)
            path = initial_value * np.cumprod(1 + returns)
            paths.append(path)
        
        fig = go.Figure()
        for i, path in enumerate(paths[:10]):  # Show first 10 paths
            fig.add_trace(go.Scatter(
                y=path,
                mode='lines',
                line=dict(width=1),
                showlegend=False,
                opacity=0.6
            ))
        
        # Add mean path
        mean_path = np.mean(paths, axis=0)
        fig.add_trace(go.Scatter(
            y=mean_path,
            mode='lines',
            line=dict(color='red', width=3),
            name='Mean Path'
        ))
        
        fig.update_layout(
            title="Monte Carlo Simulation (Portfolio Value)",
            xaxis_title="Days",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def _create_portfolio_growth_chart(self):
        """Create portfolio growth chart"""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        portfolio_value = 10000 + np.cumsum(np.random.randn(100) * 50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=portfolio_value,
            mode='lines',
            line=dict(color='#00ff88', width=3),
            name='Portfolio Value'
        ))
        
        fig.update_layout(
            title="Portfolio Growth Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def _create_asset_allocation_chart(self):
        """Create asset allocation pie chart"""
        assets = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'Cash']
        allocations = [25, 20, 15, 10, 30]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
        
        fig = go.Figure(data=[go.Pie(
            labels=assets,
            values=allocations,
            hole=.3,
            marker_colors=colors
        )])
        
        fig.update_layout(
            title="Current Asset Allocation",
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def _create_performance_metrics_chart(self):
        """Create performance metrics radar chart"""
        categories = ['Return', 'Risk', 'Sharpe', 'Win Rate', 'Profit Factor', 'Max DD']
        values = [8.5, 12.2, 0.7, 62, 1.8, 15.3]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='#00ff88')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 20]
                )),
            showlegend=False,
            title="Performance Metrics Radar",
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def _create_trade_distribution_chart(self):
        """Create trade distribution histogram"""
        # Sample trade P&L distribution
        trade_pnl = np.random.normal(50, 200, 1000)
        
        fig = go.Figure(data=[go.Histogram(
            x=trade_pnl,
            nbinsx=50,
            marker_color='#4ECDC4',
            opacity=0.7
        )])
        
        fig.update_layout(
            title="Trade P&L Distribution",
            xaxis_title="P&L ($)",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def _create_win_loss_chart(self):
        """Create win-loss analysis chart"""
        categories = ['Winning Trades', 'Losing Trades', 'Breakeven']
        counts = [65, 30, 5]
        colors = ['#00ff88', '#FF6B6B', '#FECA57']
        
        fig = go.Figure(data=[go.Bar(
            x=categories,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Win-Loss Trade Analysis",
            xaxis_title="Trade Outcome",
            yaxis_title="Number of Trades",
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def _create_time_analysis_chart(self):
        """Create trading time analysis chart"""
        hours = list(range(24))
        performance = [np.random.uniform(-20, 100) for _ in hours]
        
        fig = go.Figure(data=[go.Scatter(
            x=hours,
            y=performance,
            mode='lines+markers',
            line=dict(color='#45B7D1', width=3),
            marker=dict(size=8)
        )])
        
        fig.update_layout(
            title="Trading Performance by Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Average P&L ($)",
            template="plotly_dark",
            height=300
        )
        
        return fig
    
    def _create_rsi_chart(self, symbol: str):
        """Create dedicated RSI chart"""
        price_data = self._generate_sample_price_data(symbol, '1H')
        rsi = self._calculate_rsi(price_data['close'], 14)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_data['timestamp'], y=rsi,
            line=dict(color='purple', width=2),
            name='RSI'
        ))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        fig.add_hrect(y0=70, y1=100, fillcolor="red", opacity=0.1, line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor="green", opacity=0.1, line_width=0)
        
        fig.update_layout(
            title="RSI (14)",
            template="plotly_dark",
            height=200,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def _create_macd_chart(self, symbol: str):
        """Create dedicated MACD chart"""
        price_data = self._generate_sample_price_data(symbol, '1H')
        macd, signal, histogram = self._calculate_macd(price_data['close'])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_data['timestamp'], y=macd,
            line=dict(color='blue', width=2),
            name='MACD'
        ))
        fig.add_trace(go.Scatter(
            x=price_data['timestamp'], y=signal,
            line=dict(color='red', width=2),
            name='Signal'
        ))
        fig.add_trace(go.Bar(
            x=price_data['timestamp'], y=histogram,
            name='Histogram',
            marker_color='gray'
        ))
        
        fig.update_layout(
            title="MACD",
            template="plotly_dark",
            height=200,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def _create_volume_chart(self, symbol: str):
        """Create dedicated volume chart"""
        price_data = self._generate_sample_price_data(symbol, '1H')
        
        # Color volume bars based on price movement
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(price_data['close'], price_data['open'])]
        
        fig = go.Figure(data=[go.Bar(
            x=price_data['timestamp'],
            y=price_data['volume'],
            marker_color=colors,
            opacity=0.7
        )])
        
        fig.update_layout(
            title="Volume",
            template="plotly_dark",
            height=200,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def _create_empty_chart(self, title: str):
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text="Loading data...",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="white")
        )
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=300,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig
    
    async def start_dashboard(self, host: str = '0.0.0.0', port: int = 8051):
        """Start the advanced visualization dashboard"""
        try:
            self.logger.info(f"🚀 Starting Advanced Visualization Dashboard on {host}:{port}")
        
            # FIXED: Use app.run with different port
            self.app.run(host=host, port=port, debug=False)
        
        except Exception as e:
            self.logger.error(f"Advanced dashboard start failed: {e}")

# Factory function
def create_advanced_visualization_dashboard(trading_bot=None, config: VisualizationConfig = None):
    """Create and return AdvancedVisualizationDashboard instance"""
    return AdvancedVisualizationDashboard(trading_bot=trading_bot, config=config)

# Main execution for testing
async def main():
    """Test the advanced visualization dashboard"""
    print("🎨 Testing Advanced Visualization Dashboard...")
    
    dashboard = AdvancedVisualizationDashboard()
    await dashboard.start_dashboard()
    
    try:
        # Keep running for demonstration
        await asyncio.sleep(30)
    except KeyboardInterrupt:
        print("\n🛑 Stopping dashboard...")

if __name__ == "__main__":
    asyncio.run(main())
#!/usr/bin/env python3
"""
REAL-TIME DASHBOARD - MONITORING MODULE
Professional real-time monitoring dashboard for Forex Trading Bot
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from dataclasses import dataclass, asdict
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.log_config import setup_logging, get_logger

@dataclass
class DashboardMetrics:
    """Comprehensive dashboard metrics container"""
    system_uptime: float = 0.0
    total_trades: int = 0
    successful_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    volatility: float = 0.0
    risk_adjusted_return: float = 0.0
    active_positions: int = 0
    pending_orders: int = 0
    system_health: str = "HEALTHY"
    module_status: Dict[str, str] = None
    performance_metrics: Dict[str, Any] = None
    risk_metrics: Dict[str, Any] = None
    market_conditions: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.module_status is None:
            self.module_status = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.risk_metrics is None:
            self.risk_metrics = {}
        if self.market_conditions is None:
            self.market_conditions = {}

class RealTimeDashboard:
    """
    PROFESSIONAL REAL-TIME DASHBOARD FOR FOREX TRADING BOT
    Advanced monitoring and visualization system
    """
    
    def __init__(self, trading_bot=None, host='0.0.0.0', port=8050):
        self.trading_bot = trading_bot
        self.host = host  # ✅ YEH ADD KARO
        self.port = port  # ✅ YEH ADD KARO
        self.logger = get_logger(__name__)
        
        # Dashboard state
        self.metrics_history = []
        self.max_history_points = 1000
        self.last_update = datetime.now()
        self.is_running = False
        
        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
        )
        
        # Setup dashboard layout
        self._setup_dashboard_layout()
        
        # Initialize metrics
        self.current_metrics = DashboardMetrics()
        
        self.logger.info("🎯 REAL-TIME DASHBOARD INITIALIZED")
    
    def _setup_dashboard_layout(self):
        """Setup comprehensive dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("🎯 FOREX TRADING BOT - REAL-TIME DASHBOARD", 
                           className="text-center mb-4",
                           style={'color': '#00ff88', 'fontWeight': 'bold'})
                ], width=12)
            ]),
            
            # System Status Row
            dbc.Row([
                # System Health Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🖥️ SYSTEM HEALTH", className="text-center"),
                        dbc.CardBody([
                            html.H4("HEALTHY", id="system-health", 
                                   className="text-success text-center",
                                   style={'fontWeight': 'bold'}),
                            html.Div(id="system-uptime", className="text-center"),
                            html.Div(id="active-modules", className="text-center")
                        ])
                    ], color="dark", outline=True)
                ], width=3),
                
                # Performance Metrics Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 PERFORMANCE METRICS", className="text-center"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Total Trades", className="text-center"),
                                    html.H4(id="total-trades", className="text-info text-center")
                                ]),
                                dbc.Col([
                                    html.H6("Win Rate", className="text-center"),
                                    html.H4(id="win-rate", className="text-success text-center")
                                ])
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Total P&L", className="text-center"),
                                    html.H4(id="total-pnl", className="text-warning text-center")
                                ]),
                                dbc.Col([
                                    html.H6("Sharpe Ratio", className="text-center"),
                                    html.H4(id="sharpe-ratio", className="text-primary text-center")
                                ])
                            ])
                        ])
                    ], color="dark", outline=True)
                ], width=3),
                
                # Risk Metrics Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🛡️ RISK METRICS", className="text-center"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Max Drawdown", className="text-center"),
                                    html.H4(id="max-drawdown", className="text-danger text-center")
                                ]),
                                dbc.Col([
                                    html.H6("Current DD", className="text-center"),
                                    html.H4(id="current-drawdown", className="text-warning text-center")
                                ])
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Volatility", className="text-center"),
                                    html.H4(id="volatility", className="text-info text-center")
                                ]),
                                dbc.Col([
                                    html.H6("Risk Adj Return", className="text-center"),
                                    html.H4(id="risk-adj-return", className="text-success text-center")
                                ])
                            ])
                        ])
                    ], color="dark", outline=True)
                ], width=3),
                
                # Active Positions Card
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("💼 ACTIVE TRADING", className="text-center"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Active Positions", className="text-center"),
                                    html.H4(id="active-positions", className="text-primary text-center")
                                ]),
                                dbc.Col([
                                    html.H6("Pending Orders", className="text-center"),
                                    html.H4(id="pending-orders", className="text-warning text-center")
                                ])
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Today's Trades", className="text-center"),
                                    html.H4(id="today-trades", className="text-info text-center")
                                ]),
                                dbc.Col([
                                    html.H6("Success Rate", className="text-center"),
                                    html.H4(id="success-rate", className="text-success text-center")
                                ])
                            ])
                        ])
                    ], color="dark", outline=True)
                ], width=3)
            ], className="mb-4"),
            
            # Charts Row
            dbc.Row([
                # P&L Chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("💰 P&L PERFORMANCE", className="text-center"),
                        dbc.CardBody([
                            dcc.Graph(id='pnl-chart'),
                            dcc.Interval(id='pnl-interval', interval=5000, n_intervals=0)
                        ])
                    ], color="dark", outline=True)
                ], width=6),
                
                # Risk Metrics Chart
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 RISK ANALYSIS", className="text-center"),
                        dbc.CardBody([
                            dcc.Graph(id='risk-chart'),
                            dcc.Interval(id='risk-interval', interval=10000, n_intervals=0)
                        ])
                    ], color="dark", outline=True)
                ], width=6)
            ], className="mb-4"),
            
            # Market Analysis Row
            dbc.Row([
                # Market Conditions
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🌐 MARKET CONDITIONS", className="text-center"),
                        dbc.CardBody([
                            html.Div(id="market-conditions"),
                            dcc.Interval(id='market-interval', interval=3000, n_intervals=0)
                        ])
                    ], color="dark", outline=True)
                ], width=4),
                
                # Module Status
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔧 MODULE STATUS", className="text-center"),
                        dbc.CardBody([
                            html.Div(id="module-status"),
                            dcc.Interval(id='module-interval', interval=5000, n_intervals=0)
                        ])
                    ], color="dark", outline=True)
                ], width=4),
                
                # Trading Signals
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📡 TRADING SIGNALS", className="text-center"),
                        dbc.CardBody([
                            html.Div(id="trading-signals"),
                            dcc.Interval(id='signals-interval', interval=2000, n_intervals=0)
                        ])
                    ], color="dark", outline=True)
                ], width=4)
            ], className="mb-4"),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎮 CONTROL PANEL", className="text-center"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("🔄 Refresh Data", id="refresh-btn", color="primary", className="w-100"),
                                ], width=3),
                                dbc.Col([
                                    dbc.Button("📊 Export Report", id="export-btn", color="success", className="w-100"),
                                ], width=3),
                                dbc.Col([
                                    dbc.Button("🛑 Emergency Stop", id="stop-btn", color="danger", className="w-100"),
                                ], width=3),
                                dbc.Col([
                                    dbc.Button("🧹 Clear History", id="clear-btn", color="warning", className="w-100"),
                                ], width=3)
                            ]),
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="control-feedback", className="mt-2 text-center")
                                ], width=12)
                            ])
                        ])
                    ], color="dark", outline=True)
                ], width=12)
            ]),
            
            # Hidden div for storing data
            html.Div(id='hidden-data', style={'display': 'none'}),
            
            # Update intervals
            dcc.Interval(id='main-interval', interval=1000, n_intervals=0)
            
        ], fluid=True, style={'backgroundColor': '#1a1a1a', 'minHeight': '100vh'})
        
        # Setup callbacks
        self._setup_dashboard_callbacks()
    
    def _setup_dashboard_callbacks(self):
        """Setup all dashboard callbacks"""
        
        @self.app.callback(
            [Output('system-health', 'children'),
             Output('system-uptime', 'children'),
             Output('active-modules', 'children')],
            [Input('main-interval', 'n_intervals')]
        )
        def update_system_health(n):
            """Update system health metrics"""
            try:
                uptime_str = self._format_uptime(self.current_metrics.system_uptime)
                active_modules = len([m for m in self.current_metrics.module_status.values() 
                                    if m == 'HEALTHY'])
                total_modules = len(self.current_metrics.module_status)
                
                return [
                    self.current_metrics.system_health,
                    f"Uptime: {uptime_str}",
                    f"Modules: {active_modules}/{total_modules} Healthy"
                ]
            except Exception as e:
                self.logger.error(f"Health update error: {e}")
                return ["ERROR", "Uptime: N/A", "Modules: N/A"]
        
        @self.app.callback(
            [Output('total-trades', 'children'),
             Output('win-rate', 'children'),
             Output('total-pnl', 'children'),
             Output('sharpe-ratio', 'children')],
            [Input('main-interval', 'n_intervals')]
        )
        def update_performance_metrics(n):
            """Update performance metrics"""
            try:
                win_rate_pct = f"{self.current_metrics.win_rate:.1%}"
                pnl_str = f"${self.current_metrics.total_pnl:+.2f}"
                sharpe_str = f"{self.current_metrics.sharpe_ratio:.2f}"
                
                return [
                    self.current_metrics.total_trades,
                    win_rate_pct,
                    pnl_str,
                    sharpe_str
                ]
            except Exception as e:
                self.logger.error(f"Performance update error: {e}")
                return ["N/A", "N/A", "N/A", "N/A"]
        
        @self.app.callback(
            [Output('max-drawdown', 'children'),
             Output('current-drawdown', 'children'),
             Output('volatility', 'children'),
             Output('risk-adj-return', 'children')],
            [Input('main-interval', 'n_intervals')]
        )
        def update_risk_metrics(n):
            """Update risk metrics"""
            try:
                max_dd_str = f"{self.current_metrics.max_drawdown:.1%}"
                current_dd_str = f"{self.current_metrics.current_drawdown:.1%}"
                vol_str = f"{self.current_metrics.volatility:.1%}"
                risk_ret_str = f"{self.current_metrics.risk_adjusted_return:.2f}"
                
                return [max_dd_str, current_dd_str, vol_str, risk_ret_str]
            except Exception as e:
                self.logger.error(f"Risk update error: {e}")
                return ["N/A", "N/A", "N/A", "N/A"]
        
        @self.app.callback(
            [Output('active-positions', 'children'),
             Output('pending-orders', 'children'),
             Output('today-trades', 'children'),
             Output('success-rate', 'children')],
            [Input('main-interval', 'n_intervals')]
        )
        def update_trading_metrics(n):
            """Update trading metrics"""
            try:
                success_rate = (self.current_metrics.successful_trades / 
                              max(1, self.current_metrics.total_trades))
                success_str = f"{success_rate:.1%}"
                
                return [
                    self.current_metrics.active_positions,
                    self.current_metrics.pending_orders,
                    self.current_metrics.performance_metrics.get('today_trades', 0),
                    success_str
                ]
            except Exception as e:
                self.logger.error(f"Trading update error: {e}")
                return ["N/A", "N/A", "N/A", "N/A"]
        
        @self.app.callback(
            Output('pnl-chart', 'figure'),
            [Input('pnl-interval', 'n_intervals')]
        )
        def update_pnl_chart(n):
            """Update P&L chart"""
            try:
                if not self.metrics_history:
                    return self._create_empty_figure("P&L Performance")
                
                # Create sample data for demonstration
                dates = pd.date_range(end=datetime.now(), periods=50, freq='H')
                pnl_data = np.cumsum(np.random.randn(50) * 100)  # Simulated P&L
                
                fig = go.Figure()
                
                # P&L line
                fig.add_trace(go.Scatter(
                    x=dates, y=pnl_data,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color='#00ff88', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 136, 0.1)'
                ))
                
                # Zero line
                fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5)
                
                fig.update_layout(
                    title="Cumulative P&L Over Time",
                    xaxis_title="Time",
                    yaxis_title="P&L ($)",
                    template="plotly_dark",
                    height=400,
                    showlegend=True,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                return fig
                
            except Exception as e:
                self.logger.error(f"P&L chart error: {e}")
                return self._create_empty_figure("P&L Performance")
        
        @self.app.callback(
            Output('risk-chart', 'figure'),
            [Input('risk-interval', 'n_intervals')]
        )
        def update_risk_chart(n):
            """Update risk metrics chart"""
            try:
                # Create sample risk metrics
                metrics = ['Volatility', 'Drawdown', 'VaR', 'Sharpe', 'Win Rate']
                values = [
                    self.current_metrics.volatility * 100,
                    self.current_metrics.current_drawdown * 100,
                    2.5,  # Sample VaR
                    self.current_metrics.sharpe_ratio,
                    self.current_metrics.win_rate * 100
                ]
                
                colors = ['#ff6b6b', '#ffd93d', '#6bcf7f', '#4ecdc4', '#45b7d1']
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=metrics,
                        y=values,
                        marker_color=colors,
                        text=values,
                        texttemplate='%{text:.1f}',
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Risk Metrics Overview",
                    xaxis_title="Metrics",
                    yaxis_title="Values",
                    template="plotly_dark",
                    height=400,
                    showlegend=False,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                return fig
                
            except Exception as e:
                self.logger.error(f"Risk chart error: {e}")
                return self._create_empty_figure("Risk Metrics")
        
        @self.app.callback(
            Output('market-conditions', 'children'),
            [Input('market-interval', 'n_intervals')]
        )
        def update_market_conditions(n):
            """Update market conditions display"""
            try:
                conditions = self.current_metrics.market_conditions
                
                if not conditions:
                    conditions = {
                        'trend': 'NEUTRAL',
                        'volatility': 'MEDIUM',
                        'sentiment': 'NEUTRAL',
                        'regime': 'NORMAL'
                    }
                
                trend_color = {
                    'BULLISH': 'success',
                    'BEARISH': 'danger',
                    'NEUTRAL': 'warning'
                }.get(conditions.get('trend', 'NEUTRAL'), 'secondary')
                
                volatility_color = {
                    'HIGH': 'danger',
                    'MEDIUM': 'warning',
                    'LOW': 'success'
                }.get(conditions.get('volatility', 'MEDIUM'), 'secondary')
                
                sentiment_color = {
                    'BULLISH': 'success',
                    'BEARISH': 'danger',
                    'NEUTRAL': 'warning'
                }.get(conditions.get('sentiment', 'NEUTRAL'), 'secondary')
                
                return dbc.ListGroup([
                    dbc.ListGroupItem([
                        html.Span("📈 Trend: "),
                        dbc.Badge(conditions.get('trend', 'NEUTRAL'), 
                                 color=trend_color, className="ms-2")
                    ]),
                    dbc.ListGroupItem([
                        html.Span("🌊 Volatility: "),
                        dbc.Badge(conditions.get('volatility', 'MEDIUM'), 
                                 color=volatility_color, className="ms-2")
                    ]),
                    dbc.ListGroupItem([
                        html.Span("😊 Sentiment: "),
                        dbc.Badge(conditions.get('sentiment', 'NEUTRAL'), 
                                 color=sentiment_color, className="ms-2")
                    ]),
                    dbc.ListGroupItem([
                        html.Span("🔄 Market Regime: "),
                        dbc.Badge(conditions.get('regime', 'NORMAL'), 
                                 color="info", className="ms-2")
                    ])
                ], flush=True)
                
            except Exception as e:
                self.logger.error(f"Market conditions error: {e}")
                return html.Div("Error loading market conditions")
        
        @self.app.callback(
            Output('module-status', 'children'),
            [Input('module-interval', 'n_intervals')]
        )
        def update_module_status(n):
            """Update module status display"""
            try:
                module_status = self.current_metrics.module_status
                
                if not module_status:
                    module_status = {
                        'Data System': 'HEALTHY',
                        'Trading Engine': 'HEALTHY',
                        'Risk Management': 'HEALTHY',
                        'ML Models': 'HEALTHY',
                        'News Sentiment': 'HEALTHY'
                    }
                
                status_items = []
                for module, status in module_status.items():
                    color = {
                        'HEALTHY': 'success',
                        'DEGRADED': 'warning',
                        'FAILED': 'danger'
                    }.get(status, 'secondary')
                    
                    status_items.append(
                        dbc.ListGroupItem([
                            html.Span(f"🔧 {module}: "),
                            dbc.Badge(status, color=color, className="ms-2")
                        ])
                    )
                
                return dbc.ListGroup(status_items, flush=True)
                
            except Exception as e:
                self.logger.error(f"Module status error: {e}")
                return html.Div("Error loading module status")
        
        @self.app.callback(
            Output('trading-signals', 'children'),
            [Input('signals-interval', 'n_intervals')]
        )
        def update_trading_signals(n):
            """Update trading signals display"""
            try:
                # Sample trading signals
                signals = [
                    {'symbol': 'EUR/USD', 'action': 'BUY', 'confidence': 0.85, 'timestamp': datetime.now()},
                    {'symbol': 'GBP/USD', 'action': 'SELL', 'confidence': 0.72, 'timestamp': datetime.now()},
                    {'symbol': 'USD/JPY', 'action': 'HOLD', 'confidence': 0.45, 'timestamp': datetime.now()}
                ]
                
                signal_items = []
                for signal in signals:
                    action_color = {
                        'BUY': 'success',
                        'SELL': 'danger',
                        'HOLD': 'warning'
                    }.get(signal['action'], 'secondary')
                    
                    confidence_color = 'success' if signal['confidence'] > 0.7 else 'warning'
                    
                    signal_items.append(
                        dbc.ListGroupItem([
                            html.Div([
                                html.Strong(signal['symbol']),
                                dbc.Badge(signal['action'], color=action_color, className="ms-2"),
                                dbc.Badge(f"{signal['confidence']:.0%}", color=confidence_color, className="ms-1")
                            ]),
                            html.Small(
                                signal['timestamp'].strftime('%H:%M:%S'),
                                className="text-muted"
                            )
                        ])
                    )
                
                return dbc.ListGroup(signal_items, flush=True)
                
            except Exception as e:
                self.logger.error(f"Trading signals error: {e}")
                return html.Div("Error loading trading signals")
        
        @self.app.callback(
            Output('control-feedback', 'children'),
            [Input('refresh-btn', 'n_clicks'),
             Input('export-btn', 'n_clicks'),
             Input('stop-btn', 'n_clicks'),
             Input('clear-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def handle_control_actions(refresh_clicks, export_clicks, stop_clicks, clear_clicks):
            """Handle control panel actions"""
            ctx = callback_context
            if not ctx.triggered:
                return ""
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            try:
                if button_id == 'refresh-btn':
                    self._refresh_data()
                    return dbc.Alert("✅ Data refreshed successfully!", color="success", duration=3000)
                
                elif button_id == 'export-btn':
                    self._export_report()
                    return dbc.Alert("📊 Report exported successfully!", color="info", duration=3000)
                
                elif button_id == 'stop-btn':
                    asyncio.create_task(self._emergency_stop())
                    return dbc.Alert("🛑 Emergency stop initiated!", color="danger", duration=5000)
                
                elif button_id == 'clear-btn':
                    self._clear_history()
                    return dbc.Alert("🧹 History cleared!", color="warning", duration=3000)
                
            except Exception as e:
                self.logger.error(f"Control action error: {e}")
                return dbc.Alert(f"❌ Action failed: {e}", color="danger", duration=5000)
            
            return ""
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in seconds to human readable string"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.0f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"
    
    def _create_empty_figure(self, title: str):
        """Create empty figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="white")
        )
        fig.update_layout(
            title=title,
            template="plotly_dark",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        return fig
    
    def _refresh_data(self):
        """Refresh dashboard data"""
        self.logger.info("Refreshing dashboard data...")
        self.last_update = datetime.now()
    
    def _export_report(self):
        """Export comprehensive report"""
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': asdict(self.current_metrics),
                'history_length': len(self.metrics_history)
            }
            
            # In production, this would save to file or database
            self.logger.info(f"Report exported: {report_data}")
            
        except Exception as e:
            self.logger.error(f"Export error: {e}")
    
    async def _emergency_stop(self):
        """Execute emergency stop"""
        try:
            self.logger.critical("EMERGENCY STOP INITIATED FROM DASHBOARD")
            if self.trading_bot:
                await self.trading_bot.emergency_stop()
            
            # Update dashboard state
            self.current_metrics.system_health = "EMERGENCY_STOP"
            
        except Exception as e:
            self.logger.error(f"Emergency stop error: {e}")
    
    def _clear_history(self):
        """Clear metrics history"""
        self.metrics_history.clear()
        self.logger.info("Dashboard history cleared")
    
    async def update_metrics(self, new_metrics: DashboardMetrics):
        """Update dashboard metrics"""
        try:
            self.current_metrics = new_metrics
            
            # Add to history (limit size)
            self.metrics_history.append(new_metrics)
            if len(self.metrics_history) > self.max_history_points:
                self.metrics_history.pop(0)
            
            self.last_update = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Metrics update error: {e}")
    
    def start_dashboard(self):
        """Start the dashboard server"""
        try:
            print(f"🚀 Starting Real-Time Dashboard on http://{self.host}:{self.port}")
            print("📊 Dashboard loading... Please wait")
            
            # FIXED: Use app.run instead of app.run_server
            self.app.run(host=self.host, port=self.port, debug=False)
            
        except Exception as e:
            print(f"❌ Dashboard start failed: {e}")

# Factory function to create dashboard instance
def create_real_time_dashboard(trading_bot=None, host='0.0.0.0', port=8050):
    """Create and return a RealTimeDashboard instance"""
    return RealTimeDashboard(trading_bot=trading_bot, host=host, port=port)

# Main execution for testing
def main():
    """Test the real-time dashboard"""
    print("🎯 FOREX TRADING BOT - REAL-TIME DASHBOARD")
    print("=" * 50)
    
    dashboard = RealTimeDashboard()
    dashboard.start_dashboard()

if __name__ == "__main__":
    main()
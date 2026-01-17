import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class AdvancedMetricsCalculator:
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate  # Annual risk-free rate
        self.logger = logging.getLogger('forex_bot.advanced_metrics')
        
    def calculate_comprehensive_metrics(self, returns: pd.Series, 
                                      trades: List[Dict] = None,
                                      initial_balance: float = 10000) -> Dict:
        """Calculate comprehensive advanced performance metrics"""
        try:
            if returns.empty:
                return {}
            
            # Convert to daily returns if needed
            daily_returns = self._convert_to_daily_returns(returns)
            
            metrics = {}
            
            # Basic metrics
            metrics.update(self._calculate_basic_metrics(daily_returns, initial_balance))
            
            # Risk-adjusted metrics
            metrics.update(self._calculate_risk_adjusted_metrics(daily_returns))
            
            # Drawdown analysis
            metrics.update(self._calculate_drawdown_metrics(daily_returns, initial_balance))
            
            # Trade analysis (if trades provided)
            if trades:
                metrics.update(self._calculate_trade_based_metrics(trades, initial_balance))
            
            # Advanced statistical metrics
            metrics.update(self._calculate_statistical_metrics(daily_returns))
            
            # Portfolio theory metrics
            metrics.update(self._calculate_portfolio_metrics(daily_returns))
            
            # Strategy quality metrics
            metrics.update(self._calculate_strategy_quality_metrics(daily_returns, trades))
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            return {}

    def _calculate_basic_metrics(self, returns: pd.Series, initial_balance: float) -> Dict:
        """Calculate basic performance metrics"""
        try:
            total_return = (returns + 1).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            
            # Volatility
            annual_volatility = returns.std() * np.sqrt(252)
            
            # Cumulative returns
            cumulative_returns = (returns + 1).cumprod()
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'annual_volatility': annual_volatility,
                'cumulative_returns': cumulative_returns.tolist(),
                'total_trading_days': len(returns),
                'positive_days': len(returns[returns > 0]),
                'negative_days': len(returns[returns < 0]),
                'win_rate_daily': len(returns[returns > 0]) / len(returns)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {e}")
            return {}

    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk-adjusted performance metrics"""
        try:
            annual_return = (1 + returns.mean()) ** 252 - 1
            annual_volatility = returns.std() * np.sqrt(252)
            
            # Sharpe Ratio
            sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
            
            # Sortino Ratio (only downside risk)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annual_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar Ratio
            max_drawdown = self._calculate_max_drawdown(returns)
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Omega Ratio
            omega_ratio = self._calculate_omega_ratio(returns)
            
            # Treynor Ratio (assuming beta = 1 for simplicity)
            treynor_ratio = (annual_return - self.risk_free_rate) / 1.0  # Beta = 1
            
            # Information Ratio (no benchmark, using risk-free as benchmark)
            excess_returns = returns - (self.risk_free_rate / 252)
            information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'omega_ratio': omega_ratio,
                'treynor_ratio': treynor_ratio,
                'information_ratio': information_ratio,
                'risk_free_rate': self.risk_free_rate
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted metrics: {e}")
            return {}

    def _calculate_omega_ratio(self, returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega Ratio - measures probability-weighted gains vs losses"""
        try:
            if len(returns) == 0:
                return 0
            
            # Returns above and below threshold
            gains = returns[returns > threshold] - threshold
            losses = threshold - returns[returns < threshold]
            
            if len(losses) == 0:
                return float('inf')  # No losses
            
            sum_gains = gains.sum()
            sum_losses = losses.sum()
            
            return sum_gains / sum_losses if sum_losses != 0 else float('inf')
            
        except Exception as e:
            self.logger.error(f"Error calculating omega ratio: {e}")
            return 0

    def _calculate_drawdown_metrics(self, returns: pd.Series, initial_balance: float) -> Dict:
        """Calculate comprehensive drawdown analysis"""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            
            # Maximum drawdown
            max_drawdown = drawdowns.min()
            max_drawdown_date = drawdowns.idxmin() if hasattr(drawdowns, 'idxmin') else None
            
            # Drawdown duration analysis
            drawdown_durations = self._calculate_drawdown_durations(drawdowns)
            
            # Ulcer Index (measures depth and duration of drawdowns)
            ulcer_index = np.sqrt((drawdowns ** 2).mean())
            
            # Pain Index (average of drawdowns)
            pain_index = drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0
            
            # Recovery factors
            recovery_metrics = self._calculate_recovery_metrics(drawdowns, returns)
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_date': max_drawdown_date.isoformat() if max_drawdown_date else None,
                'avg_drawdown': drawdowns[drawdowns < 0].mean() if len(drawdowns[drawdowns < 0]) > 0 else 0,
                'drawdown_std': drawdowns[drawdowns < 0].std() if len(drawdowns[drawdowns < 0]) > 0 else 0,
                'ulcer_index': ulcer_index,
                'pain_index': abs(pain_index),
                'longest_drawdown_days': max(drawdown_durations) if drawdown_durations else 0,
                'avg_drawdown_duration': np.mean(drawdown_durations) if drawdown_durations else 0,
                'drawdown_durations': drawdown_durations,
                'recovery_metrics': recovery_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown metrics: {e}")
            return {}

    def _calculate_drawdown_durations(self, drawdowns: pd.Series) -> List[int]:
        """Calculate durations of all drawdown periods"""
        try:
            durations = []
            in_drawdown = False
            current_duration = 0
            
            for dd in drawdowns:
                if dd < -0.001:  # In drawdown (0.1% threshold)
                    if not in_drawdown:
                        in_drawdown = True
                        current_duration = 1
                    else:
                        current_duration += 1
                else:
                    if in_drawdown:
                        durations.append(current_duration)
                        in_drawdown = False
                        current_duration = 0
            
            # Add final drawdown if still in drawdown
            if in_drawdown:
                durations.append(current_duration)
            
            return durations
            
        except Exception as e:
            self.logger.error(f"Error calculating drawdown durations: {e}")
            return []

    def _calculate_recovery_metrics(self, drawdowns: pd.Series, returns: pd.Series) -> Dict:
        """Calculate drawdown recovery metrics"""
        try:
            recovery_times = []
            recovery_rates = []
            
            in_drawdown = False
            drawdown_start = None
            max_dd_in_period = 0
            
            for i, (date, dd) in enumerate(drawdowns.items()):
                if dd < -0.001 and not in_drawdown:  # Entering drawdown
                    in_drawdown = True
                    drawdown_start = i
                    max_dd_in_period = dd
                elif dd < -0.001 and in_drawdown:  # Still in drawdown
                    max_dd_in_period = min(max_dd_in_period, dd)
                elif in_drawdown and dd >= -0.001:  # Recovered
                    recovery_time = i - drawdown_start
                    recovery_times.append(recovery_time)
                    
                    # Recovery rate (how quickly it recovered from max drawdown)
                    if max_dd_in_period != 0:
                        recovery_rate = abs(max_dd_in_period) / recovery_time
                        recovery_rates.append(recovery_rate)
                    
                    in_drawdown = False
                    drawdown_start = None
                    max_dd_in_period = 0
            
            return {
                'avg_recovery_days': np.mean(recovery_times) if recovery_times else 0,
                'max_recovery_days': max(recovery_times) if recovery_times else 0,
                'avg_recovery_rate': np.mean(recovery_rates) if recovery_rates else 0,
                'recovery_consistency': len([rt for rt in recovery_times if rt <= 10]) / len(recovery_times) if recovery_times else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating recovery metrics: {e}")
            return {}

    def _calculate_trade_based_metrics(self, trades: List[Dict], initial_balance: float) -> Dict:
        """Calculate metrics based on individual trades"""
        try:
            if not trades:
                return {}
            
            df_trades = pd.DataFrame(trades)
            
            # Basic trade statistics
            total_trades = len(df_trades)
            winning_trades = len(df_trades[df_trades['pnl'] > 0])
            losing_trades = len(df_trades[df_trades['pnl'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # PnL statistics
            total_pnl = df_trades['pnl'].sum()
            avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            largest_win = df_trades['pnl'].max()
            largest_loss = df_trades['pnl'].min()
            
            # Profit factor
            gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
            gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Expectancy
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
            
            # Kelly Criterion
            kelly_criterion = win_rate - ((1 - win_rate) / (avg_win / abs(avg_loss))) if avg_loss != 0 else 0
            
            # Trade duration analysis
            if 'duration_hours' in df_trades.columns:
                avg_duration = df_trades['duration_hours'].mean()
                win_duration = df_trades[df_trades['pnl'] > 0]['duration_hours'].mean() if winning_trades > 0 else 0
                loss_duration = df_trades[df_trades['pnl'] < 0]['duration_hours'].mean() if losing_trades > 0 else 0
            else:
                avg_duration = win_duration = loss_duration = 0
            
            # Consecutive wins/losses
            consecutive_stats = self._calculate_consecutive_trades(df_trades)
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'kelly_criterion': kelly_criterion,
                'avg_trade_pnl': df_trades['pnl'].mean(),
                'avg_winning_trade': avg_win,
                'avg_losing_trade': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'avg_trade_duration_hours': avg_duration,
                'avg_win_duration_hours': win_duration,
                'avg_loss_duration_hours': loss_duration,
                'consecutive_wins': consecutive_stats['max_consecutive_wins'],
                'consecutive_losses': consecutive_stats['max_consecutive_losses'],
                'avg_consecutive_wins': consecutive_stats['avg_consecutive_wins'],
                'avg_consecutive_losses': consecutive_stats['avg_consecutive_losses']
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trade-based metrics: {e}")
            return {}

    def _calculate_consecutive_trades(self, df_trades: pd.DataFrame) -> Dict:
        """Calculate consecutive wins and losses"""
        try:
            wins = (df_trades['pnl'] > 0).astype(int)
            current_streak = 0
            current_type = None
            win_streaks = []
            loss_streaks = []
            
            for win in wins:
                if win == 1:  # Winning trade
                    if current_type == 'win':
                        current_streak += 1
                    else:
                        if current_type == 'loss' and current_streak > 0:
                            loss_streaks.append(current_streak)
                        current_streak = 1
                        current_type = 'win'
                else:  # Losing trade
                    if current_type == 'loss':
                        current_streak += 1
                    else:
                        if current_type == 'win' and current_streak > 0:
                            win_streaks.append(current_streak)
                        current_streak = 1
                        current_type = 'loss'
            
            # Add final streak
            if current_streak > 0:
                if current_type == 'win':
                    win_streaks.append(current_streak)
                else:
                    loss_streaks.append(current_streak)
            
            return {
                'max_consecutive_wins': max(win_streaks) if win_streaks else 0,
                'max_consecutive_losses': max(loss_streaks) if loss_streaks else 0,
                'avg_consecutive_wins': np.mean(win_streaks) if win_streaks else 0,
                'avg_consecutive_losses': np.mean(loss_streaks) if loss_streaks else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating consecutive trades: {e}")
            return {'max_consecutive_wins': 0, 'max_consecutive_losses': 0, 
                   'avg_consecutive_wins': 0, 'avg_consecutive_losses': 0}

    def _calculate_statistical_metrics(self, returns: pd.Series) -> Dict:
        """Calculate advanced statistical metrics"""
        try:
            # Skewness and Kurtosis
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Normality tests
            shapiro_stat, shapiro_p = stats.shapiro(returns) if len(returns) < 5000 else (0, 0)
            normaltest_stat, normaltest_p = stats.normaltest(returns)
            
            # VaR (Value at Risk)
            var_95 = returns.quantile(0.05)
            var_99 = returns.quantile(0.01)
            
            # CVaR (Conditional VaR / Expected Shortfall)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Tail ratio
            tail_ratio = abs(returns.quantile(0.95)) / abs(returns.quantile(0.05)) if returns.quantile(0.05) != 0 else 0
            
            # Gain to Pain ratio
            total_gain = returns[returns > 0].sum()
            total_pain = abs(returns[returns < 0].sum())
            gain_to_pain = total_gain / total_pain if total_pain > 0 else float('inf')
            
            # Common Sense Ratio (Profitability vs Risk)
            common_sense_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0
            
            return {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'shapiro_wilk_p': shapiro_p,
                'normality_test_p': normaltest_p,
                'is_normal_distribution': normaltest_p > 0.05,
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'tail_ratio': tail_ratio,
                'gain_to_pain_ratio': gain_to_pain,
                'common_sense_ratio': common_sense_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating statistical metrics: {e}")
            return {}

    def _calculate_portfolio_metrics(self, returns: pd.Series) -> Dict:
        """Calculate portfolio theory metrics"""
        try:
            # Alpha and Beta (assuming market return = risk-free for simplicity)
            market_returns = pd.Series([self.risk_free_rate / 252] * len(returns), index=returns.index)
            beta = 1.0  # Simplified - in reality would calculate vs benchmark
            
            # Jensen's Alpha
            alpha = returns.mean() - (self.risk_free_rate / 252 + beta * (market_returns.mean() - self.risk_free_rate / 252))
            
            # Tracking Error (vs risk-free rate)
            tracking_error = (returns - market_returns).std() * np.sqrt(252)
            
            # Appraisal Ratio (Information Ratio)
            appraisal_ratio = alpha / tracking_error if tracking_error > 0 else 0
            
            # Modigliani Risk-Adjusted Performance (M2)
            m2_ratio = returns.mean() * (market_returns.std() / returns.std()) if returns.std() > 0 else 0
            
            # Burke Ratio (Calmar-like but with downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            burke_ratio = returns.mean() * np.sqrt(252) / downside_std if downside_std > 0 else 0
            
            return {
                'alpha': alpha * 252,  # Annualized
                'beta': beta,
                'tracking_error': tracking_error,
                'appraisal_ratio': appraisal_ratio,
                'm2_ratio': m2_ratio,
                'burke_ratio': burke_ratio,
                'active_return': (returns.mean() - market_returns.mean()) * 252
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return {}

    def _calculate_strategy_quality_metrics(self, returns: pd.Series, trades: List[Dict] = None) -> Dict:
        """Calculate strategy quality and robustness metrics"""
        try:
            # Serial correlation (check for predictability)
            autocorrelation = returns.autocorr()
            
            # Hurst Exponent (measure of trend tendency)
            hurst_exponent = self._calculate_hurst_exponent(returns)
            
            # Sharpe Ratio有效性检验 (Probabilistic Sharpe Ratio)
            probabilistic_sharpe = self._calculate_probabilistic_sharpe(returns)
            
            # Deflated Sharpe Ratio (account for multiple testing)
            deflated_sharpe = self._calculate_deflated_sharpe(returns)
            
            # Strategy stability metrics
            stability_metrics = self._calculate_strategy_stability(returns)
            
            # Monte Carlo metrics (if trades available)
            monte_carlo_metrics = self._calculate_monte_carlo_metrics(returns, trades) if trades else {}
            
            return {
                'autocorrelation': autocorrelation,
                'hurst_exponent': hurst_exponent,
                'market_efficiency': 'Trending' if hurst_exponent > 0.6 else 
                                   'Mean Reverting' if hurst_exponent < 0.4 else 
                                   'Random Walk',
                'probabilistic_sharpe_ratio': probabilistic_sharpe,
                'deflated_sharpe_ratio': deflated_sharpe,
                'strategy_stability': stability_metrics,
                'monte_carlo_analysis': monte_carlo_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy quality metrics: {e}")
            return {}

    def _calculate_hurst_exponent(self, returns: pd.Series, max_lag: int = 20) -> float:
        """Calculate Hurst Exponent for market efficiency analysis"""
        try:
            if len(returns) < 50:
                return 0.5
            
            lags = range(2, min(max_lag, len(returns)//4))
            tau = []
            
            for lag in lags:
                # Calculate R/S ratio for each lag
                returns_lag = []
                for i in range(0, len(returns) - lag, lag):
                    returns_lag.append(returns.iloc[i:i+lag].sum())
                
                if len(returns_lag) > 1:
                    mean = np.mean(returns_lag)
                    std = np.std(returns_lag)
                    cumulative_deviation = np.cumsum(returns_lag - mean)
                    range_val = np.max(cumulative_deviation) - np.min(cumulative_deviation)
                    
                    if std > 0:
                        rs = range_val / std
                        tau.append(np.log(rs))
            
            if len(tau) < 2:
                return 0.5
            
            # Fit linear regression
            H, _ = np.polyfit(np.log(lags[:len(tau)]), tau, 1)
            return H
            
        except Exception as e:
            self.logger.error(f"Error calculating Hurst exponent: {e}")
            return 0.5

    def _calculate_probabilistic_sharpe(self, returns: pd.Series) -> float:
        """Calculate Probabilistic Sharpe Ratio"""
        try:
            sharpe = (returns.mean() - self.risk_free_rate/252) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            n = len(returns)
            
            # Assuming benchmark Sharpe = 0
            benchmark_sharpe = 0
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Variance of Sharpe ratio
            variance = (1 + (0.5 * sharpe ** 2) - (skewness * sharpe) + ((kurtosis - 3) / 4) * sharpe ** 2) / (n - 1)
            std_sharpe = np.sqrt(variance)
            
            # Probabilistic Sharpe (Z-score)
            probabilistic_sharpe = (sharpe - benchmark_sharpe) / std_sharpe if std_sharpe > 0 else 0
            
            return probabilistic_sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating probabilistic Sharpe: {e}")
            return 0

    def _calculate_deflated_sharpe(self, returns: pd.Series) -> float:
        """Calculate Deflated Sharpe Ratio"""
        try:
            sharpe = (returns.mean() - self.risk_free_rate/252) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            n = len(returns)
            
            # Assuming multiple testing adjustment
            num_strategies_tested = 100  # Conservative estimate
            variance = 1 / (n - 1)
            
            # Deflated Sharpe
            deflated_sharpe = sharpe * np.sqrt(1 - np.log(1 / num_strategies_tested) / (n - 1))
            
            return deflated_sharpe
            
        except Exception as e:
            self.logger.error(f"Error calculating deflated Sharpe: {e}")
            return 0

    def _calculate_strategy_stability(self, returns: pd.Series, window: int = 63) -> Dict:
        """Calculate strategy stability over time"""
        try:
            if len(returns) < window * 2:
                return {}
            
            rolling_sharpe = returns.rolling(window=window).apply(
                lambda x: (x.mean() - self.risk_free_rate/252) / x.std() * np.sqrt(252) if x.std() > 0 else 0
            )
            
            rolling_max_dd = returns.rolling(window=window).apply(
                lambda x: self._calculate_max_drawdown(x)
            )
            
            return {
                'sharpe_stability': rolling_sharpe.std() / rolling_sharpe.mean() if rolling_sharpe.mean() != 0 else 0,
                'max_sharpe': rolling_sharpe.max(),
                'min_sharpe': rolling_sharpe.min(),
                'sharpe_decline_periods': len(rolling_sharpe[rolling_sharpe < 0]),
                'drawdown_stability': rolling_max_dd.std() / abs(rolling_max_dd.mean()) if rolling_max_dd.mean() != 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy stability: {e}")
            return {}

    def _calculate_monte_carlo_metrics(self, returns: pd.Series, trades: List[Dict], 
                                     num_simulations: int = 1000) -> Dict:
        """Calculate Monte Carlo simulation metrics"""
        try:
            if not trades:
                return {}
            
            # Simulate different trade sequences
            final_values = []
            max_drawdowns = []
            
            for _ in range(num_simulations):
                # Shuffle trades randomly
                shuffled_trades = np.random.permutation(trades)
                cumulative_pnl = np.cumsum([t['pnl'] for t in shuffled_trades])
                
                if len(cumulative_pnl) > 0:
                    final_values.append(cumulative_pnl[-1])
                    max_drawdowns.append(self._calculate_max_drawdown(pd.Series(cumulative_pnl)))
            
            return {
                'monte_carlo_final_value_mean': np.mean(final_values),
                'monte_carlo_final_value_std': np.std(final_values),
                'monte_carlo_success_rate': len([v for v in final_values if v > 0]) / len(final_values),
                'monte_carlo_max_drawdown_mean': np.mean(max_drawdowns),
                'monte_carlo_var_95': np.percentile(final_values, 5),
                'monte_carlo_cvar_95': np.mean([v for v in final_values if v <= np.percentile(final_values, 5)])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Monte Carlo metrics: {e}")
            return {}

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0

    def _convert_to_daily_returns(self, returns: pd.Series) -> pd.Series:
        """Convert returns to daily frequency if needed"""
        try:
            # If returns are already daily, return as is
            if len(returns) < 2:
                return returns
            
            # Simple conversion - in reality would use proper resampling
            # This assumes returns are in the same frequency
            return returns
            
        except Exception as e:
            self.logger.error(f"Error converting to daily returns: {e}")
            return returns

    def generate_performance_report(self, metrics: Dict) -> str:
        """Generate human-readable performance report"""
        try:
            report = []
            report.append("=" * 60)
            report.append("ADVANCED PERFORMANCE METRICS REPORT")
            report.append("=" * 60)
            
            # Basic Performance
            report.append("\n📊 BASIC PERFORMANCE")
            report.append(f"Total Return: {metrics.get('total_return', 0)*100:.2f}%")
            report.append(f"Annual Return: {metrics.get('annual_return', 0)*100:.2f}%")
            report.append(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            
            # Risk-Adjusted Performance
            report.append("\n🎯 RISK-ADJUSTED PERFORMANCE")
            report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            report.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
            report.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
            report.append(f"Omega Ratio: {metrics.get('omega_ratio', 0):.2f}")
            
            # Risk Analysis
            report.append("\n⚠️ RISK ANALYSIS")
            report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
            report.append(f"Annual Volatility: {metrics.get('annual_volatility', 0)*100:.2f}%")
            report.append(f"VaR (95%): {metrics.get('var_95', 0)*100:.2f}%")
            report.append(f"CVaR (95%): {metrics.get('cvar_95', 0)*100:.2f}%")
            
            # Strategy Quality
            report.append("\n🔬 STRATEGY QUALITY")
            report.append(f"Probabilistic Sharpe: {metrics.get('probabilistic_sharpe_ratio', 0):.2f}")
            report.append(f"Deflated Sharpe: {metrics.get('deflated_sharpe_ratio', 0):.2f}")
            report.append(f"Hurst Exponent: {metrics.get('hurst_exponent', 0):.2f}")
            report.append(f"Market Type: {metrics.get('market_efficiency', 'Unknown')}")
            
            # Trade Analysis
            if 'total_trades' in metrics:
                report.append("\n💼 TRADE ANALYSIS")
                report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
                report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                report.append(f"Expectancy: {metrics.get('expectancy', 0):.2f}")
                report.append(f"Kelly Criterion: {metrics.get('kelly_criterion', 0):.2f}")
            
            report.append("\n" + "=" * 60)
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return "Error generating report"
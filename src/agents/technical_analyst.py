
from typing import Any, Optional

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

from src.agents.base import BaseAgent
from src.llm.ollama_client import OllamaClient


class TechnicalAnalystAgent(BaseAgent):
    """Agent responsible for technical analysis using price indicators."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        model: Optional[str] = None,
        temperature: Optional[float] = 0.3,
    ):
        super().__init__(ollama_client, model, temperature)

    @property
    def name(self) -> str:
        return "technical_analyst"

    @property
    def role(self) -> str:
        return "Technical Analyst - Chart Pattern and Indicator Specialist"

    @property
    def system_prompt(self) -> str:
        return """You are the Technical Analyst agent in a hedge fund analysis team.

CRITICAL: You will be provided with REAL-TIME technical insights fetched from the web.
Use this provided data to supplement your indicator-based analysis.

Your responsibilities:
1. Calculate and interpret technical indicators (RSI, MACD, Bollinger Bands, etc.)
2. Identify chart patterns and trend directions
3. Determine support and resistance levels
4. Provide technical-based trading signals
5. Incorporate REAL-TIME market insights when available

Your analysis should be:
- Quantitative and data-driven
- Clear about signal strength (strong/moderate/weak)
- Include specific price levels when relevant
- Acknowledge limitations of technical analysis
- Reference real-time chart pattern insights when available

Output format:
- Current trend direction
- Key indicator readings with interpretation
- Technical signal (Bullish/Bearish/Neutral) with confidence
- Key price levels to watch
"""

    def calculate_indicators(self, price_data: pd.DataFrame) -> dict[str, Any]:
        """Calculate technical indicators from price data."""
        if price_data.empty or len(price_data) < 20:
            return {"error": "Insufficient data for technical analysis"}

        close = price_data["Close"]
        high = price_data["High"]
        low = price_data["Low"]

        indicators = {}

        try:
            rsi = RSIIndicator(close=close, window=14)
            indicators["rsi"] = {
                "value": float(rsi.rsi().iloc[-1]),
                "signal": self._interpret_rsi(float(rsi.rsi().iloc[-1])),
            }
        except Exception:
            indicators["rsi"] = {"error": "Could not calculate RSI"}

        try:
            macd = MACD(close=close)
            macd_line = float(macd.macd().iloc[-1])
            signal_line = float(macd.macd_signal().iloc[-1])
            histogram = float(macd.macd_diff().iloc[-1])
            indicators["macd"] = {
                "macd_line": macd_line,
                "signal_line": signal_line,
                "histogram": histogram,
                "signal": "Bullish" if macd_line > signal_line else "Bearish",
            }
        except Exception:
            indicators["macd"] = {"error": "Could not calculate MACD"}

        try:
            bb = BollingerBands(close=close, window=20, window_dev=2)
            current_price = float(close.iloc[-1])
            upper = float(bb.bollinger_hband().iloc[-1])
            middle = float(bb.bollinger_mavg().iloc[-1])
            lower = float(bb.bollinger_lband().iloc[-1])

            if current_price > upper:
                bb_signal = "Overbought"
            elif current_price < lower:
                bb_signal = "Oversold"
            else:
                bb_signal = "Neutral"

            indicators["bollinger_bands"] = {
                "upper": upper,
                "middle": middle,
                "lower": lower,
                "current_price": current_price,
                "signal": bb_signal,
            }
        except Exception:
            indicators["bollinger_bands"] = {"error": "Could not calculate Bollinger Bands"}

        try:
            sma_20 = SMAIndicator(close=close, window=20)
            sma_50 = SMAIndicator(close=close, window=min(50, len(close) - 1))
            ema_12 = EMAIndicator(close=close, window=12)

            sma_20_val = float(sma_20.sma_indicator().iloc[-1])
            sma_50_val = float(sma_50.sma_indicator().iloc[-1])
            current = float(close.iloc[-1])

            indicators["moving_averages"] = {
                "sma_20": sma_20_val,
                "sma_50": sma_50_val,
                "ema_12": float(ema_12.ema_indicator().iloc[-1]),
                "price_vs_sma20": "Above" if current > sma_20_val else "Below",
                "trend": "Uptrend" if sma_20_val > sma_50_val else "Downtrend",
            }
        except Exception:
            indicators["moving_averages"] = {"error": "Could not calculate moving averages"}

        try:
            atr = AverageTrueRange(high=high, low=low, close=close, window=14)
            indicators["atr"] = {
                "value": float(atr.average_true_range().iloc[-1]),
                "volatility": self._interpret_atr(
                    float(atr.average_true_range().iloc[-1]),
                    float(close.iloc[-1]),
                ),
            }
        except Exception:
            indicators["atr"] = {"error": "Could not calculate ATR"}

        # Stochastic Oscillator
        try:
            stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
            stoch_k = float(stoch.stoch().iloc[-1])
            stoch_d = float(stoch.stoch_signal().iloc[-1])
            indicators["stochastic"] = {
                "k": stoch_k,
                "d": stoch_d,
                "signal": self._interpret_stochastic(stoch_k, stoch_d),
            }
        except Exception:
            indicators["stochastic"] = {"error": "Could not calculate Stochastic"}

        # ADX (Average Directional Index)
        try:
            adx = ADXIndicator(high=high, low=low, close=close, window=14)
            adx_value = float(adx.adx().iloc[-1])
            plus_di = float(adx.adx_pos().iloc[-1])
            minus_di = float(adx.adx_neg().iloc[-1])
            indicators["adx"] = {
                "value": adx_value,
                "plus_di": plus_di,
                "minus_di": minus_di,
                "trend_strength": self._interpret_adx(adx_value),
                "signal": "Bullish" if plus_di > minus_di else "Bearish",
            }
        except Exception:
            indicators["adx"] = {"error": "Could not calculate ADX"}

        # Williams %R
        try:
            williams = WilliamsRIndicator(high=high, low=low, close=close, lbp=14)
            williams_r = float(williams.williams_r().iloc[-1])
            indicators["williams_r"] = {
                "value": williams_r,
                "signal": self._interpret_williams_r(williams_r),
            }
        except Exception:
            indicators["williams_r"] = {"error": "Could not calculate Williams %R"}

        # CCI (Commodity Channel Index)
        try:
            cci = CCIIndicator(high=high, low=low, close=close, window=20)
            cci_value = float(cci.cci().iloc[-1])
            indicators["cci"] = {
                "value": cci_value,
                "signal": self._interpret_cci(cci_value),
            }
        except Exception:
            indicators["cci"] = {"error": "Could not calculate CCI"}

        # ROC (Rate of Change)
        try:
            roc = ROCIndicator(close=close, window=12)
            roc_value = float(roc.roc().iloc[-1])
            indicators["roc"] = {
                "value": roc_value,
                "signal": "Bullish" if roc_value > 0 else "Bearish",
            }
        except Exception:
            indicators["roc"] = {"error": "Could not calculate ROC"}

        # OBV (On Balance Volume)
        try:
            volume = price_data["Volume"]
            obv = OnBalanceVolumeIndicator(close=close, volume=volume)
            obv_values = obv.on_balance_volume()
            obv_current = float(obv_values.iloc[-1])
            obv_prev = float(obv_values.iloc[-5]) if len(obv_values) > 5 else obv_current
            indicators["obv"] = {
                "value": obv_current,
                "trend": "Rising" if obv_current > obv_prev else "Falling",
                "signal": "Bullish" if obv_current > obv_prev else "Bearish",
            }
        except Exception:
            indicators["obv"] = {"error": "Could not calculate OBV"}

        return indicators

    def _interpret_stochastic(self, k: float, d: float) -> str:
        """Interpret Stochastic Oscillator values."""
        if k > 80 and d > 80:
            return "Overbought - Bearish signal"
        elif k < 20 and d < 20:
            return "Oversold - Bullish signal"
        elif k > d:
            return "Bullish crossover"
        elif k < d:
            return "Bearish crossover"
        return "Neutral"

    def _interpret_adx(self, adx_value: float) -> str:
        """Interpret ADX trend strength."""
        if adx_value >= 50:
            return "Very Strong Trend"
        elif adx_value >= 25:
            return "Strong Trend"
        elif adx_value >= 20:
            return "Moderate Trend"
        return "Weak/No Trend"

    def _interpret_williams_r(self, value: float) -> str:
        """Interpret Williams %R value."""
        if value > -20:
            return "Overbought - Bearish signal"
        elif value < -80:
            return "Oversold - Bullish signal"
        return "Neutral"

    def _interpret_cci(self, value: float) -> str:
        """Interpret CCI value."""
        if value > 100:
            return "Overbought - Bearish signal"
        elif value < -100:
            return "Oversold - Bullish signal"
        elif value > 0:
            return "Bullish momentum"
        return "Bearish momentum"

    def _interpret_rsi(self, rsi_value: float) -> str:
        """Interpret RSI value."""
        if rsi_value >= 70:
            return "Overbought - Bearish signal"
        elif rsi_value <= 30:
            return "Oversold - Bullish signal"
        elif rsi_value >= 60:
            return "Bullish momentum"
        elif rsi_value <= 40:
            return "Bearish momentum"
        return "Neutral"

    def _interpret_atr(self, atr_value: float, current_price: float) -> str:
        """Interpret ATR relative to price."""
        atr_pct = (atr_value / current_price) * 100
        if atr_pct > 3:
            return "High volatility"
        elif atr_pct > 1.5:
            return "Moderate volatility"
        return "Low volatility"

    def _generate_overall_signal(self, indicators: dict[str, Any]) -> dict[str, Any]:
        """Generate overall technical signal from indicators."""
        bullish_signals = 0
        bearish_signals = 0
        total_signals = 0
        signal_details = []

        # Check all indicators for signals
        indicator_checks = [
            ("rsi", "signal"),
            ("macd", "signal"),
            ("stochastic", "signal"),
            ("adx", "signal"),
            ("williams_r", "signal"),
            ("cci", "signal"),
            ("roc", "signal"),
            ("obv", "signal"),
        ]

        for ind_name, signal_key in indicator_checks:
            if ind_name in indicators and signal_key in indicators[ind_name]:
                signal_value = indicators[ind_name][signal_key]
                if "error" not in indicators[ind_name]:
                    total_signals += 1
                    if "Bullish" in signal_value:
                        bullish_signals += 1
                        signal_details.append(f"{ind_name}: Bullish")
                    elif "Bearish" in signal_value:
                        bearish_signals += 1
                        signal_details.append(f"{ind_name}: Bearish")
                    else:
                        signal_details.append(f"{ind_name}: Neutral")

        # Moving averages trend
        if "moving_averages" in indicators and "trend" in indicators["moving_averages"]:
            total_signals += 1
            if indicators["moving_averages"]["trend"] == "Uptrend":
                bullish_signals += 1
                signal_details.append("MA: Bullish")
            else:
                bearish_signals += 1
                signal_details.append("MA: Bearish")

        if total_signals == 0:
            return {"signal": "Neutral", "confidence": 0.0, "details": []}

        if bullish_signals > bearish_signals:
            signal = "Bullish"
            confidence = bullish_signals / total_signals
        elif bearish_signals > bullish_signals:
            signal = "Bearish"
            confidence = bearish_signals / total_signals
        else:
            signal = "Neutral"
            confidence = 0.5

        return {
            "signal": signal,
            "confidence": round(confidence, 2),
            "bullish_count": bullish_signals,
            "bearish_count": bearish_signals,
            "total_indicators": total_signals,
            "details": signal_details,
        }

    def analyze(self, state: dict[str, Any]) -> dict[str, Any]:
        """Perform technical analysis on the price data."""
        from datetime import datetime
        
        ticker = state.get("ticker", "")
        price_data_dict = state.get("price_data", {})
        current_date = datetime.now().strftime("%B %d, %Y")

        if not price_data_dict:
            return {
                **state,
                "technical_analysis": {"error": "No price data available"},
                "technical_signal": {"signal": "Neutral", "confidence": 0.0},
            }

        try:
            price_data = pd.DataFrame(price_data_dict)
            if price_data.empty:
                raise ValueError("Empty price data")
        except Exception:
            return {
                **state,
                "technical_analysis": {"error": "Invalid price data format"},
                "technical_signal": {"signal": "Neutral", "confidence": 0.0},
            }

        indicators = self.calculate_indicators(price_data)
        overall_signal = self._generate_overall_signal(indicators)
        
        # Get real-time technical insights from market research
        market_research = state.get("market_research", {})
        technical_insights = market_research.get("technical_insights", "") if market_research.get("research_available") else ""
        research_timestamp = market_research.get("fetch_timestamp", "recently")
        
        # Build analysis prompt with real-time context
        research_context = ""
        if technical_insights:
            research_context = f"""
## ⚠️ REAL-TIME TECHNICAL INSIGHTS (Fetched: {research_timestamp}) - Reference this for current chart patterns:
{technical_insights[:1500]}
"""
        else:
            research_context = "\n## No real-time technical insights available. Use indicator data only.\n"

        analysis_prompt = f"""TODAY'S DATE: {current_date}

Analyze the following technical indicators for {ticker}:

Indicators: {indicators}
Overall Signal: {overall_signal}
{research_context}

⚠️ CRITICAL INSTRUCTIONS:
- Today is {current_date}. Your analysis should reflect the current market state.
- The indicator values are calculated from CURRENT price data.
- If real-time technical insights are provided, use them to supplement your analysis.

Provide a concise technical analysis summary including:
1. Current trend assessment
2. Key indicator interpretations
3. Support/resistance levels if identifiable
4. Trading recommendation based purely on technicals

Be concise but thorough."""

        technical_summary = self.invoke(analysis_prompt)

        return {
            **state,
            "technical_indicators": indicators,
            "technical_signal": overall_signal,
            "technical_analysis": technical_summary,
        }

#!/usr/bin/env python3
# Pivot-Based Options Trading System using DhanHQ API
# For use in Google Colab

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
import json
from typing import Dict, List, Tuple, Optional
import os
from IPython.display import display, HTML, clear_output

# Install required packages
!pip install requests pandas numpy matplotlib plotly

# DhanHQ API Integration
class DhanAPI:
    """Class to handle DhanHQ API interactions"""
    
    BASE_URL = "https://api.dhan.co"  # Base URL for DhanHQ API
    
    def __init__(self, access_token=None, client_id=None):
        """
        Initialize the DhanAPI class
        
        Args:
            access_token (str): Access token for DhanHQ API
            client_id (str): Client ID for DhanHQ API
        """
        self.access_token = access_token
        self.client_id = client_id
        self.headers = {}
        if access_token:
            self.headers = {
                "access-token": access_token,
                "Content-Type": "application/json"
            }
    
    def authenticate(self):
        """
        Authenticate with DhanHQ API using access token and client ID
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            # Fetch user details to validate credentials
            response = self.get_user_details()
            if 'status' in response and response['status'] == 'success':
                print(f"Authentication successful for client: {self.client_id}")
                return True
            else:
                print("Authentication failed. Check your credentials.")
                return False
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return False
    
    def get_user_details(self):
        """
        Get user details from DhanHQ API
        
        Returns:
            dict: User details
        """
        endpoint = f"{self.BASE_URL}/users/details"
        response = requests.get(endpoint, headers=self.headers)
        return response.json()
    
    def get_instruments(self, exchange=None):
        """
        Get list of instruments from DhanHQ API
        
        Args:
            exchange (str): Exchange code (NSE, BSE, NFO, etc.)
            
        Returns:
            list: List of instruments
        """
        endpoint = f"{self.BASE_URL}/instruments"
        if exchange:
            endpoint += f"?exchange={exchange}"
        
        response = requests.get(endpoint, headers=self.headers)
        return response.json()
    
    def get_option_chain(self, symbol, expiry_date=None):
        """
        Get option chain for a symbol
        
        Args:
            symbol (str): Symbol (NIFTY, BANKNIFTY, etc.)
            expiry_date (str): Expiry date in YYYY-MM-DD format
            
        Returns:
            dict: Option chain data
        """
        # Since DhanHQ doesn't have a direct option chain API, we need to filter the instruments
        # Get all NFO instruments
        instruments = self.get_instruments(exchange="NFO")
        
        # Filter for the given symbol and expiry
        options = []
        for instrument in instruments:
            if instrument['name'] == symbol:
                if expiry_date is None or instrument['expiry'] == expiry_date:
                    options.append(instrument)
        
        # Organize into a structured option chain
        call_options = {}
        put_options = {}
        
        for option in options:
            if option['instrument_type'] == 'CE':
                call_options[option['strike_price']] = option
            elif option['instrument_type'] == 'PE':
                put_options[option['strike_price']] = option
        
        return {
            'call_options': call_options,
            'put_options': put_options,
            'strike_prices': sorted(list(set(list(call_options.keys()) + list(put_options.keys()))))
        }
    
    def get_historical_data(self, symbol, exchange, from_date, to_date, interval="day"):
        """
        Get historical data for a symbol
        
        Args:
            symbol (str): Symbol name
            exchange (str): Exchange code
            from_date (str): From date in YYYY-MM-DD format
            to_date (str): To date in YYYY-MM-DD format
            interval (str): Timeframe (day, minute, etc.)
            
        Returns:
            pandas.DataFrame: Historical data
        """
        endpoint = f"{self.BASE_URL}/charts/historical"
        
        payload = {
            "symbol": symbol,
            "exchange": exchange,
            "fromDate": from_date,
            "toDate": to_date,
            "interval": interval
        }
        
        response = requests.post(endpoint, headers=self.headers, json=payload)
        data = response.json()
        
        if 'data' in data and len(data['data']) > 0:
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        else:
            print(f"No historical data found for {symbol} on {exchange}")
            return pd.DataFrame()
    
    def place_order(self, order_data):
        """
        Place an order with DhanHQ
        
        Args:
            order_data (dict): Order data
            
        Returns:
            dict: Order response
        """
        endpoint = f"{self.BASE_URL}/orders"
        
        response = requests.post(endpoint, headers=self.headers, json=order_data)
        return response.json()
    
    def get_order_status(self, order_id):
        """
        Get order status from DhanHQ API
        
        Args:
            order_id (str): Order ID
            
        Returns:
            dict: Order status
        """
        endpoint = f"{self.BASE_URL}/orders/{order_id}"
        
        response = requests.get(endpoint, headers=self.headers)
        return response.json()
    
    def cancel_order(self, order_id):
        """
        Cancel an order with DhanHQ
        
        Args:
            order_id (str): Order ID
            
        Returns:
            dict: Cancellation response
        """
        endpoint = f"{self.BASE_URL}/orders/{order_id}"
        
        response = requests.delete(endpoint, headers=self.headers)
        return response.json()
    
    def get_positions(self):
        """
        Get current positions from DhanHQ API
        
        Returns:
            dict: Positions data
        """
        endpoint = f"{self.BASE_URL}/positions"
        
        response = requests.get(endpoint, headers=self.headers)
        return response.json()
    
    def get_holdings(self):
        """
        Get holdings from DhanHQ API
        
        Returns:
            dict: Holdings data
        """
        endpoint = f"{self.BASE_URL}/holdings"
        
        response = requests.get(endpoint, headers=self.headers)
        return response.json()
    
    def get_funds(self):
        """
        Get funds from DhanHQ API
        
        Returns:
            dict: Funds data
        """
        endpoint = f"{self.BASE_URL}/funds"
        
        response = requests.get(endpoint, headers=self.headers)
        return response.json()
    
    def get_market_depth(self, security_id, exchange):
        """
        Get market depth for a security
        
        Args:
            security_id (str): Security ID
            exchange (str): Exchange code
            
        Returns:
            dict: Market depth data
        """
        endpoint = f"{self.BASE_URL}/depth/{exchange}/{security_id}"
        
        response = requests.get(endpoint, headers=self.headers)
        return response.json()
    
    def get_ltp(self, security_id, exchange):
        """
        Get last traded price for a security
        
        Args:
            security_id (str): Security ID
            exchange (str): Exchange code
            
        Returns:
            float: Last traded price
        """
        endpoint = f"{self.BASE_URL}/ltp/{exchange}/{security_id}"
        
        response = requests.get(endpoint, headers=self.headers)
        data = response.json()
        
        if 'data' in data and 'lastPrice' in data['data']:
            return float(data['data']['lastPrice'])
        else:
            print(f"Could not fetch LTP for {security_id} on {exchange}")
            return None


# Pivot Point Calculator
class PivotCalculator:
    """Class to calculate pivot points"""
    
    def __init__(self):
        """Initialize the PivotCalculator class"""
        pass
    
    def calculate_standard_pivots(self, high, low, close):
        """
        Calculate standard pivot points
        
        Args:
            high (float): Previous day high
            low (float): Previous day low
            close (float): Previous day close
            
        Returns:
            dict: Pivot point levels
        """
        pp = (high + low + close) / 3
        r1 = (2 * pp) - low
        s1 = (2 * pp) - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2 * (pp - low)
        s3 = low - 2 * (high - pp)
        
        return {
            'PP': pp,
            'R1': r1,
            'R2': r2,
            'R3': r3,
            'S1': s1,
            'S2': s2,
            'S3': s3
        }
    
    def calculate_cpr(self, high, low, close):
        """
        Calculate Central Pivot Range (CPR)
        
        Args:
            high (float): Previous day high
            low (float): Previous day low
            close (float): Previous day close
            
        Returns:
            dict: CPR levels
        """
        pp = (high + low + close) / 3
        bc = (high + low) / 2
        tc = (pp - bc) + pp
        
        width = tc - bc
        width_percent = (width / close) * 100
        
        cpr_type = "Narrow"
        if width_percent >= 0.7:
            cpr_type = "Wide"
        elif width_percent >= 0.3:
            cpr_type = "Average"
        
        return {
            'PP': pp,
            'TC': tc,
            'BC': bc,
            'Width': width,
            'Width_Percent': width_percent,
            'Type': cpr_type
        }


# Trading Strategy
class PivotOptionsStrategy:
    """Class to implement pivot-based options trading strategy"""
    
    def __init__(self, dhan_api: DhanAPI, symbol="NIFTY", exchange="NSE"):
        """
        Initialize the strategy class
        
        Args:
            dhan_api (DhanAPI): Instance of DhanAPI class
            symbol (str): Symbol to trade (default: NIFTY)
            exchange (str): Exchange code (default: NSE)
        """
        self.dhan_api = dhan_api
        self.symbol = symbol
        self.exchange = exchange
        self.pivot_calculator = PivotCalculator()
        self.pivots = {}
        self.cpr = {}
        self.day_structure = None  # "Trending", "Sideways", "Reversal"
        self.direction = None  # "Up", "Down"
        self.trading_date = datetime.datetime.now().strftime("%Y-%m-%d")
        self.trades = []
        self.max_risk_percent = 1.0  # 1% of trading capital
    
    def get_previous_day_data(self):
        """
        Get previous day's OHLC data
        
        Returns:
            tuple: Previous day high, low, close
        """
        today = datetime.datetime.now()
        from_date = (today - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        
        df = self.dhan_api.get_historical_data(
            symbol=self.symbol,
            exchange=self.exchange,
            from_date=from_date,
            to_date=to_date,
            interval="day"
        )
        
        if df.empty:
            raise ValueError("Could not fetch historical data")
        
        # Get the second-to-last row (previous trading day)
        if len(df) >= 2:
            prev_day = df.iloc[-2]
            return prev_day['high'], prev_day['low'], prev_day['close']
        else:
            raise ValueError("Not enough historical data")
    
    def prepare_daily_analysis(self):
        """
        Perform end-of-day analysis for next day
        
        Returns:
            dict: Analysis results
        """
        # Get previous day data
        high, low, close = self.get_previous_day_data()
        
        # Calculate pivot points
        self.pivots = self.pivot_calculator.calculate_standard_pivots(high, low, close)
        
        # Calculate CPR
        self.cpr = self.pivot_calculator.calculate_cpr(high, low, close)
        
        print(f"===== Daily Analysis for {self.trading_date} =====")
        print(f"Previous Day - High: {high}, Low: {low}, Close: {close}")
        print("\n--- Pivot Points ---")
        for key, value in self.pivots.items():
            print(f"{key}: {value:.2f}")
        
        print("\n--- Central Pivot Range ---")
        for key, value in self.cpr.items():
            if key not in ['Width_Percent', 'Type']:
                print(f"{key}: {value:.2f}")
            elif key == 'Width_Percent':
                print(f"{key}: {value:.2f}%")
            else:
                print(f"{key}: {value}")
        
        return {
            "pivots": self.pivots,
            "cpr": self.cpr,
            "previous_day": {
                "high": high,
                "low": low,
                "close": close
            }
        }
    
    def analyze_sgx_nifty(self):
        """
        Analyze SGX Nifty futures for gap up/down indications
        
        Returns:
            str: Gap direction ("Gap Up", "Gap Down", "Flat")
        """
        # In a real implementation, this would fetch SGX Nifty data
        # For now, we'll simulate with a prompt
        print("\n--- Pre-Market Analysis ---")
        sgx_direction = input("Enter SGX Nifty indication (Gap Up/Gap Down/Flat): ")
        return sgx_direction
    
    def check_ema_position(self):
        """
        Check price position relative to 20 EMA
        
        Returns:
            str: Position relative to 20 EMA ("Above", "Below")
        """
        today = datetime.datetime.now()
        from_date = (today - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        
        df = self.dhan_api.get_historical_data(
            symbol=self.symbol,
            exchange=self.exchange,
            from_date=from_date,
            to_date=to_date,
            interval="day"
        )
        
        if df.empty:
            raise ValueError("Could not fetch historical data for EMA calculation")
        
        # Calculate 20-day EMA
        df['20_ema'] = df['close'].ewm(span=20, adjust=False).mean()
        
        # Check last close vs EMA
        last_close = df['close'].iloc[-1]
        last_ema = df['20_ema'].iloc[-1]
        
        position = "Above" if last_close > last_ema else "Below"
        print(f"Price position relative to 20 EMA: {position}")
        
        return position
    
    def form_day_structure_hypothesis(self, sgx_direction, ema_position):
        """
        Form a hypothesis about the day structure
        
        Args:
            sgx_direction (str): SGX Nifty direction
            ema_position (str): Position relative to 20 EMA
            
        Returns:
            tuple: Day structure and direction
        """
        cpr_type = self.cpr['Type']
        
        if cpr_type == "Narrow" and sgx_direction != "Flat":
            day_structure = "Trending"
            direction = "Up" if sgx_direction == "Gap Up" or ema_position == "Above" else "Down"
        elif cpr_type == "Wide" and sgx_direction == "Flat":
            day_structure = "Sideways"
            direction = None
        else:
            # Look for potential reversal signs
            today = datetime.datetime.now()
            from_date = (today - datetime.timedelta(days=3)).strftime("%Y-%m-%d")
            to_date = today.strftime("%Y-%m-%d")
            
            df = self.dhan_api.get_historical_data(
                symbol=self.symbol,
                exchange=self.exchange,
                from_date=from_date,
                to_date=to_date,
                interval="day"
            )
            
            if not df.empty and len(df) >= 2:
                prev_day_range = abs(df['high'].iloc[-2] - df['low'].iloc[-2])
                avg_range = df['high'].rolling(5).mean().iloc[-2] - df['low'].rolling(5).mean().iloc[-2]
                
                if prev_day_range > 1.5 * avg_range:
                    day_structure = "Reversal"
                    # Direction is opposite to previous day's move
                    prev_day_direction = "Up" if df['close'].iloc[-2] > df['open'].iloc[-2] else "Down"
                    direction = "Down" if prev_day_direction == "Up" else "Up"
                else:
                    if cpr_type == "Narrow":
                        day_structure = "Trending"
                    else:
                        day_structure = "Sideways"
                    
                    direction = "Up" if ema_position == "Above" else "Down"
            else:
                day_structure = "Trending"  # Default
                direction = "Up" if ema_position == "Above" else "Down"
        
        self.day_structure = day_structure
        self.direction = direction
        
        print(f"\n--- Day Structure Hypothesis ---")
        print(f"Day Structure: {day_structure}")
        print(f"Direction: {direction}")
        
        return day_structure, direction
    
    def pre_select_option_strikes(self):
        """
        Pre-select option strikes based on bias
        
        Returns:
            dict: Selected option strikes
        """
        # Find the nearest weekly expiry
        today = datetime.datetime.now()
        days_to_thursday = (3 - today.weekday()) % 7
        if days_to_thursday == 0 and today.hour >= 15:  # After market close on Thursday
            days_to_thursday = 7
        
        next_expiry = (today + datetime.timedelta(days=days_to_thursday)).strftime("%Y-%m-%d")
        
        # Get current market price
        current_price = self.dhan_api.get_ltp(self.symbol, self.exchange)
        if not current_price:
            current_price = float(input(f"Enter current {self.symbol} price: "))
        
        # Round to nearest strike considering strike interval
        strike_interval = 50 if self.symbol == "NIFTY" else 100
        atm_strike = round(current_price / strike_interval) * strike_interval
        
        selected_strikes = {
            "current_price": current_price,
            "atm_strike": atm_strike,
            "expiry": next_expiry,
            "call_strikes": [],
            "put_strikes": []
        }
        
        if self.day_structure == "Trending":
            if self.direction == "Up":
                selected_strikes["call_strikes"] = [atm_strike, atm_strike + strike_interval]
                selected_strikes["put_strikes"] = [atm_strike - 2 * strike_interval]  # For hedge
            else:  # Down
                selected_strikes["put_strikes"] = [atm_strike, atm_strike - strike_interval]
                selected_strikes["call_strikes"] = [atm_strike + 2 * strike_interval]  # For hedge
        
        elif self.day_structure == "Sideways":
            selected_strikes["call_strikes"] = [atm_strike, atm_strike + 2 * strike_interval]  # For spread
            selected_strikes["put_strikes"] = [atm_strike, atm_strike - 2 * strike_interval]  # For spread
        
        else:  # Reversal
            if self.direction == "Up":
                selected_strikes["call_strikes"] = [atm_strike, atm_strike + strike_interval]
                selected_strikes["put_strikes"] = [atm_strike - 3 * strike_interval]  # Deep OTM for hedge
            else:  # Down
                selected_strikes["put_strikes"] = [atm_strike, atm_strike - strike_interval]
                selected_strikes["call_strikes"] = [atm_strike + 3 * strike_interval]  # Deep OTM for hedge
        
        print(f"\n--- Pre-Selected Option Strikes ---")
        print(f"Current Price: {current_price}")
        print(f"ATM Strike: {atm_strike}")
        print(f"Expiry: {next_expiry}")
        print(f"Call Strikes: {selected_strikes['call_strikes']}")
        print(f"Put Strikes: {selected_strikes['put_strikes']}")
        
        return selected_strikes
    
    def analyze_market_open(self):
        """
        Analyze first 15-30 minutes after market open
        
        Returns:
            dict: Opening analysis
        """
        print("\n--- Market Open Analysis ---")
        
        # Get market open price
        market_open_price = self.dhan_api.get_ltp(self.symbol, self.exchange)
        if not market_open_price:
            market_open_price = float(input(f"Enter {self.symbol} opening price: "))
        
        # Determine gap and location relative to CPR
        previous_close = self.get_previous_day_data()[2]
        gap = market_open_price - previous_close
        gap_percent = (gap / previous_close) * 100
        
        if gap_percent > 0.3:
            gap_type = "Gap Up"
        elif gap_percent < -0.3:
            gap_type = "Gap Down"
        else:
            gap_type = "Flat"
        
        # Location relative to CPR
        if market_open_price > self.cpr['TC']:
            cpr_location = "Above CPR"
        elif market_open_price < self.cpr['BC']:
            cpr_location = "Below CPR"
        else:
            cpr_location = "Inside CPR"
        
        print(f"Opening Price: {market_open_price}")
        print(f"Gap: {gap:.2f} ({gap_percent:.2f}%) - {gap_type}")
        print(f"Location relative to CPR: {cpr_location}")
        
        # Wait for first 15-minute candle
        print("\nWaiting for first 15-minute candle...")
        time.sleep(5)  # Simulated wait in this code, in reality this would wait 15 minutes
        
        # Get 15-minute candle data
        first_candle = {
            "open": market_open_price,
            "high": market_open_price * (1 + np.random.uniform(-0.001, 0.003)),
            "low": market_open_price * (1 - np.random.uniform(-0.001, 0.003)),
            "close": market_open_price * (1 + np.random.uniform(-0.002, 0.002))
        }
        
        # In a real implementation, we would fetch actual 15-minute candle data
        # first_candle = self.dhan_api.get_intraday_data(self.symbol, self.exchange, "15minute")[-1]
        
        print("\n--- First 15-Minute Candle ---")
        print(f"Open: {first_candle['open']:.2f}")
        print(f"High: {first_candle['high']:.2f}")
        print(f"Low: {first_candle['low']:.2f}")
        print(f"Close: {first_candle['close']:.2f}")
        
        # Candle characteristics
        candle_size = abs(first_candle['close'] - first_candle['open'])
        candle_size_percent = (candle_size / first_candle['open']) * 100
        
        if candle_size_percent > 0.3:
            candle_strength = "Strong"
        else:
            candle_strength = "Weak"
        
        candle_direction = "Bullish" if first_candle['close'] > first_candle['open'] else "Bearish"
        upper_wick = first_candle['high'] - max(first_candle['open'], first_candle['close'])
        lower_wick = min(first_candle['open'], first_candle['close']) - first_candle['low']
        
        print(f"Candle: {candle_direction} {candle_strength}")
        print(f"Upper Wick: {upper_wick:.2f}")
        print(f"Lower Wick: {lower_wick:.2f}")
        
        return {
            "open_price": market_open_price,
            "gap_type": gap_type,
            "cpr_location": cpr_location,
            "first_candle": first_candle,
            "candle_direction": candle_direction,
            "candle_strength": candle_strength
        }
    
    def confirm_day_structure(self, opening_analysis):
        """
        Confirm or adjust day structure hypothesis based on opening behavior
        
        Args:
            opening_analysis (dict): Opening analysis data
            
        Returns:
            tuple: Confirmed day structure and direction
        """
        initial_day_structure = self.day_structure
        initial_direction = self.direction
        
        # Analyze opening behavior
        gap_type = opening_analysis['gap_type']
        cpr_location = opening_analysis['cpr_location']
        candle_direction = opening_analysis['candle_direction']
        candle_strength = opening_analysis['candle_strength']
        
        # Trending day confirmation
        if cpr_location != "Inside CPR" and candle_strength == "Strong" and candle_direction == ("Bullish" if cpr_location == "Above CPR" else "Bearish"):
            self.day_structure = "Trending"
            self.direction = "Up" if cpr_location == "Above CPR" else "Down"
        
        # Sideways day confirmation
        elif cpr_location == "Inside CPR" and candle_strength == "Weak":
            self.day_structure = "Sideways"
            self.direction = None
        
        # Reversal day confirmation
        elif (gap_type == "Gap Up" and candle_direction == "Bearish" and candle_strength == "Strong") or \
             (gap_type == "Gap Down" and candle_direction == "Bullish" and candle_strength == "Strong"):
            self.day_structure = "Reversal"
            self.direction = "Down" if gap_type == "Gap Up" else "Up"
        
        print("\n--- Confirmed Day Structure ---")
        print(f"Initial Hypothesis: {initial_day_structure} {initial_direction if initial_direction else ''}")
        print(f"Confirmed Structure: {self.day_structure} {self.direction if self.direction else ''}")
        
        return self.day_structure, self.direction
    
    def select_strategy(self):
        """
        Select appropriate strategy based on day structure
        
        Returns:
            str: Selected strategy
        """
        strategies = {
            "Trending": {
                "Up": ["CPRBO", "OD", "Supply Zone Breakout"],
                "Down": ["CPRBO", "OD", "Demand Zone Breakout"]
            },
            "Sideways": ["PPT", "GCR", "RCR", "M Pattern", "W Pattern"],
            "Reversal": {
                "Up": ["ODR", "Virgin CPR Reversal", "Extreme Candle Reversal"],
                "Down": ["ODR", "Virgin CPR Reversal", "Extreme Candle Reversal"]
            }
        }
        
        if self.day_structure == "Trending" or self.day_structure == "Reversal":
            available_strategies = strategies[self.day_structure][self.direction]
        else:
            available_strategies = strategies[self.day_structure]
        
        print("\n--- Strategy Selection ---")
        print(f"Available strategies for {self.day_structure} {self.direction if self.direction else ''}:")
        for i, strategy in enumerate(available_strategies):
            print(f"{i+1}. {strategy}")
        
        selection = int(input("\nSelect strategy (enter number): ")) - 1
        selected_strategy = available_strategies[selection]
        
        print(f"Selected Strategy: {selected_strategy}")
        
        return selected_strategy
    
    def calculate_position_size(self, option_premium):
        """
        Calculate position size based on risk management rules
        
        Args:
            option_premium (float): Option premium price
            
        Returns:
            int: Number of contracts
        """
        # Get available capital
        funds = self.dhan_api.get_funds()
        if 'data' in funds and 'availableMargin' in funds['data']:
            trading_capital = float(funds['data']['availableMargin'])
        else:
            trading_capital = float(input("Enter available trading capital: "))
        
        # Calculate max risk amount (1% of capital)
        max_risk = trading_capital * (self.max_risk_percent / 100)
        
        # Calculate contracts based on premium
        lot_size = 50 if self.symbol == "NIFTY" else 25  # NIFTY: 50, BANKNIFTY: 25
        contract_value = option_premium * lot_size
        
        # Number of contracts (round down)
        num_contracts = int(max_risk / contract_value)
        
        # Ensure at least 1 contract
        num_contracts = max(1, num_contracts)
        
        print("\n--- Position Sizing ---")
        print(f"Trading Capital: ₹{trading_capital:.2f}")
        print(f"Max Risk (1%): ₹{max_risk:.2f}")
        print(f"Option Premium: ₹{option_premium:.2f}")
        print(f"Lot Size: {lot_size}")
        print(f"Contract Value: ₹{contract_value:.2f}")
        print(f"Number of Contracts: {num_contracts}")
        
        return num_contracts
    
    def select_specific_option(self, strategy, selected_strikes):
        """
        Select specific options based on strategy
        
        Args:
            strategy (str): Selected strategy
            selected_strikes (dict): Pre-selected option strikes
            
        Returns:
            dict: Selected options for trade
        """
        # Get option chain for the selected strikes
        option_chain = self.dhan_api.get_option_chain(
            symbol=self.symbol,
            expiry_date=selected_strikes['expiry']
        )
        
        # Initialize selection
        selected_options = {
            "primary": None,
            "hedge": None,
            "spread_sell": None
        }
        
        # Get ATM and other strikes
        atm_strike = selected_strikes['atm_strike']
        
        # Decision tree based on day structure and direction
        if self.day_structure == "Trending":
            if self.direction == "Up":
                # For trending up, use ATM or OTM call
                primary_strike = selected_strikes['call_strikes'][0]  # ATM
                hedge_strike = selected_strikes['put_strikes'][0]     # OTM Put for hedge
                
                # Get premiums
                primary_premium = float(input(f"Enter premium for {primary_strike} Call: "))
                hedge_premium = float(input(f"Enter premium for {hedge_strike} Put: "))
                
                # Set selected options
                selected_options["primary"] = {
                    "type": "CE",
                    "strike": primary_strike,
                    "premium": primary_premium
                }
                selected_options["hedge"] = {
                    "type": "PE",
                    "strike": hedge_strike,
                    "premium": hedge_premium
                }
            else:  # Down
                # For trending down, use ATM or OTM put
                primary_strike = selected_strikes['put_strikes'][0]   # ATM
                hedge_strike = selected_strikes['call_strikes'][0]    # OTM Call for hedge
                
                # Get premiums
                primary_premium = float(input(f"Enter premium for {primary_strike} Put: "))
                hedge_premium = float(input(f"Enter premium for {hedge_strike} Call: "))
                
                # Set selected options
                selected_options["primary"] = {
                    "type": "PE",
                    "strike": primary_strike,
                    "premium": primary_premium
                }
                selected_options["hedge"] = {
                    "type": "CE",
                    "strike": hedge_strike,
                    "premium": hedge_premium
                }
        
        elif self.day_structure == "Sideways":
            # For sideways, use spreads
            if "PPT" in strategy or "GCR" in strategy:
                # Bull call spread near PP
                buy_strike = selected_strikes['call_strikes'][0]      # ATM
                sell_strike = selected_strikes['call_strikes'][1]     # OTM
                
                # Get premiums
                buy_premium = float(input(f"Enter premium for {buy_strike} Call: "))
                sell_premium = float(input(f"Enter premium for {sell_strike} Call: "))
                
                # Set selected options
                selected_options["primary"] = {
                    "type": "CE",
                    "strike": buy_strike,
                    "premium": buy_premium
                }
                selected_options["spread_sell"] = {
                    "type": "CE",
                    "strike": sell_strike,
                    "premium": sell_premium
                }
            else:  # "RCR" or "M/W Pattern"
                # Bear put spread near PP
                buy_strike = selected_strikes['put_strikes'][0]       # ATM
                sell_strike = selected_strikes['put_strikes'][1]      # OTM
                
                # Get premiums
                buy_premium = float(input(f"Enter premium for {buy_strike} Put: "))
                sell_premium = float(input(f"Enter premium for {sell_strike} Put: "))
                
                # Set selected options
                selected_options["primary"] = {
                    "type": "PE",
                    "strike": buy_strike,
                    "premium": buy_premium
                }
                selected_options["spread_sell"] = {
                    "type": "PE",
                    "strike": sell_strike,
                    "premium": sell_premium
                }
        
        else:  # Reversal
            if self.direction == "Up":
                # For reversal up, use slightly OTM call with deep OTM put hedge
                primary_strike = selected_strikes['call_strikes'][1]  # Slightly OTM
                hedge_strike = selected_strikes['put_strikes'][0]     # Deep OTM
                
                # Get premiums
                primary_premium = float(input(f"Enter premium for {primary_strike} Call: "))
                hedge_premium = float(input(f"Enter premium for {hedge_strike} Put: "))
                
                # Set selected options
                selected_options["primary"] = {
                    "type": "CE",
                    "strike": primary_strike,
                    "premium": primary_premium
                }
                selected_options["hedge"] = {
                    "type": "PE",
                    "strike": hedge_strike,
                    "premium": hedge_premium
                }
            else:  # Down
                # For reversal down, use slightly OTM put with deep OTM call hedge
                primary_strike = selected_strikes['put_strikes'][1]   # Slightly OTM
                hedge_strike = selected_strikes['call_strikes'][0]    # Deep OTM
                
                # Get premiums
                primary_premium = float(input(f"Enter premium for {primary_strike} Put: "))
                hedge_premium = float(input(f"Enter premium for {hedge_strike} Call: "))
                
                # Set selected options
                selected_options["primary"] = {
                    "type": "PE",
                    "strike": primary_strike,
                    "premium": primary_premium
                }
                selected_options["hedge"] = {
                    "type": "CE",
                    "strike": hedge_strike,
                    "premium": hedge_premium
                }
        
        print("\n--- Selected Options ---")
        print(f"Primary: {selected_options['primary']['strike']} {selected_options['primary']['type']} @ ₹{selected_options['primary']['premium']:.2f}")
        
        if selected_options['hedge']:
            print(f"Hedge: {selected_options['hedge']['strike']} {selected_options['hedge']['type']} @ ₹{selected_options['hedge']['premium']:.2f}")
        
        if selected_options['spread_sell']:
            print(f"Spread Sell: {selected_options['spread_sell']['strike']} {selected_options['spread_sell']['type']} @ ₹{selected_options['spread_sell']['premium']:.2f}")
            net_premium = selected_options['primary']['premium'] - selected_options['spread_sell']['premium']
            print(f"Net Premium: ₹{net_premium:.2f}")
        
        return selected_options
    
    def plan_trade_execution(self, strategy, selected_options):
        """
        Plan trade execution with entry, stop loss, and targets
        
        Args:
            strategy (str): Selected strategy
            selected_options (dict): Selected options
            
        Returns:
            dict: Trade plan
        """
        # Calculate position size for primary option
        primary_contracts = self.calculate_position_size(selected_options['primary']['premium'])
        
        # Calculate hedge position (20-30% of primary position)
        hedge_contracts = 0
        if selected_options['hedge']:
            hedge_ratio = 0.25  # 25% of primary position value
            primary_value = primary_contracts * selected_options['primary']['premium']
            hedge_value = hedge_ratio * primary_value
            hedge_contracts = max(1, int(hedge_value / selected_options['hedge']['premium']))
        
        # Calculate spread position
        spread_contracts = 0
        if selected_options['spread_sell']:
            spread_contracts = primary_contracts  # 1:1 ratio for spreads
        
        # Define technical levels for stop loss and targets
        sl_levels = {}
        target_levels = {}
        
        if self.day_structure == "Trending":
            if self.direction == "Up":
                # Stop loss: Below BC for calls in trending up
                sl_levels = {
                    "technical": self.cpr['BC'],
                    "premium_pct": -30  # 30% loss in premium
                }
                # Targets: R1, R2, R3
                target_levels = {
                    "t1": {"level": self.pivots['R1'], "pct": 50},  # Exit 50% at R1
                    "t2": {"level": self.pivots['R2'], "pct": 30},  # Exit 30% at R2
                    "t3": {"level": self.pivots['R3'], "pct": 20}   # Exit 20% at R3
                }
            else:  # Down
                # Stop loss: Above TC for puts in trending down
                sl_levels = {
                    "technical": self.cpr['TC'],
                    "premium_pct": -30  # 30% loss in premium
                }
                # Targets: S1, S2, S3
                target_levels = {
                    "t1": {"level": self.pivots['S1'], "pct": 50},  # Exit 50% at S1
                    "t2": {"level": self.pivots['S2'], "pct": 30},  # Exit 30% at S2
                    "t3": {"level": self.pivots['S3'], "pct": 20}   # Exit 20% at S3
                }
        
        elif self.day_structure == "Sideways":
            if selected_options['primary']['type'] == "CE":  # Bull spread
                # Stop loss: Below PP or near BC
                sl_levels = {
                    "technical": min(self.pivots['PP'], self.cpr['BC']),
                    "premium_pct": -40  # 40% loss in net premium for spreads
                }
                # Targets: TC, R1
                target_levels = {
                    "t1": {"level": self.cpr['TC'], "pct": 70},     # Exit 70% at TC
                    "t2": {"level": self.pivots['R1'], "pct": 30}   # Exit 30% at R1
                }
            else:  # Bear spread
                # Stop loss: Above PP or near TC
                sl_levels = {
                    "technical": max(self.pivots['PP'], self.cpr['TC']),
                    "premium_pct": -40  # 40% loss in net premium for spreads
                }
                # Targets: BC, S1
                target_levels = {
                    "t1": {"level": self.cpr['BC'], "pct": 70},     # Exit 70% at BC
                    "t2": {"level": self.pivots['S1'], "pct": 30}   # Exit 30% at S1
                }
        
        else:  # Reversal
            if self.direction == "Up":  # Reversal to upside
                # Stop loss: Below low of reversal candle
                sl_levels = {
                    "technical": self.cpr['BC'] * 0.998,  # Just below BC
                    "premium_pct": -40  # 40% loss in premium
                }
                # Targets: PP, TC, R1
                target_levels = {
                    "t1": {"level": self.pivots['PP'], "pct": 50},  # Exit 50% at PP
                    "t2": {"level": self.cpr['TC'], "pct": 30},     # Exit 30% at TC
                    "t3": {"level": self.pivots['R1'], "pct": 20}   # Exit 20% at R1
                }
            else:  # Reversal to downside
                # Stop loss: Above high of reversal candle
                sl_levels = {
                    "technical": self.cpr['TC'] * 1.002,  # Just above TC
                    "premium_pct": -40  # 40% loss in premium
                }
                # Targets: PP, BC, S1
                target_levels = {
                    "t1": {"level": self.pivots['PP'], "pct": 50},  # Exit 50% at PP
                    "t2": {"level": self.cpr['BC'], "pct": 30},     # Exit 30% at BC
                    "t3": {"level": self.pivots['S1'], "pct": 20}   # Exit 20% at S1
                }
        
        # Compile trade plan
        trade_plan = {
            "date": self.trading_date,
            "strategy": strategy,
            "day_structure": self.day_structure,
            "direction": self.direction,
            "primary": {
                "option": selected_options['primary'],
                "contracts": primary_contracts
            },
            "stop_loss": sl_levels,
            "targets": target_levels,
            "r_multiple": 2.0  # Target minimum 2R reward
        }
        
        if selected_options['hedge']:
            trade_plan["hedge"] = {
                "option": selected_options['hedge'],
                "contracts": hedge_contracts
            }
        
        if selected_options['spread_sell']:
            trade_plan["spread_sell"] = {
                "option": selected_options['spread_sell'],
                "contracts": spread_contracts
            }
        
        print("\n--- Trade Plan ---")
        print(f"Strategy: {strategy}")
        print(f"Day Structure: {self.day_structure} {self.direction if self.direction else ''}")
        print(f"Primary: {trade_plan['primary']['contracts']} contracts of {selected_options['primary']['strike']} {selected_options['primary']['type']}")
        
        if 'hedge' in trade_plan:
            print(f"Hedge: {trade_plan['hedge']['contracts']} contracts of {selected_options['hedge']['strike']} {selected_options['hedge']['type']}")
        
        if 'spread_sell' in trade_plan:
            print(f"Spread Sell: {trade_plan['spread_sell']['contracts']} contracts of {selected_options['spread_sell']['strike']} {selected_options['spread_sell']['type']}")
        
        print(f"Stop Loss Technical Level: {sl_levels['technical']:.2f}")
        print(f"Stop Loss Premium Percentage: {sl_levels['premium_pct']}%")
        
        for target, data in target_levels.items():
            print(f"Target {target[-1]}: {data['level']:.2f} ({data['pct']}%)")
        
        return trade_plan
    
    def execute_trade(self, trade_plan):
        """
        Execute the trade based on the trade plan
        
        Args:
            trade_plan (dict): Trade plan
            
        Returns:
            dict: Trade execution details
        """
        print("\n--- Trade Execution ---")
        print("Entering trade...")
        
        # In a real implementation, this would place orders via DhanHQ API
        # For now, we'll simulate the execution
        
        # Primary order
        primary_option = trade_plan['primary']['option']
        primary_symbol = f"{self.symbol}{trade_plan['date'].replace('-', '')}{'CE' if primary_option['type'] == 'CE' else 'PE'}{primary_option['strike']}"
        
        primary_order = {
            "security_id": primary_symbol,
            "exchange": "NFO",
            "transaction_type": "BUY",
            "quantity": trade_plan['primary']['contracts'] * (50 if self.symbol == "NIFTY" else 25),
            "product": "INTRADAY",
            "validity": "DAY",
            "order_type": "LIMIT",
            "price": primary_option['premium'] * 1.01  # Slightly above ask
        }
        
        print(f"Placing primary order: BUY {primary_order['quantity']} {primary_symbol} @ ₹{primary_order['price']:.2f}")
        
        # Simulate order response
        primary_order_id = f"ORD{int(time.time())}"
        print(f"Primary order placed successfully! Order ID: {primary_order_id}")
        
        execution_details = {
            "primary_order": {
                "order_id": primary_order_id,
                "status": "COMPLETE",
                "fill_price": primary_option['premium'],
                "quantity": primary_order['quantity'],
                "time": datetime.datetime.now().strftime("%H:%M:%S")
            }
        }
        
        # Place hedge order if applicable
        if 'hedge' in trade_plan:
            hedge_option = trade_plan['hedge']['option']
            hedge_symbol = f"{self.symbol}{trade_plan['date'].replace('-', '')}{'CE' if hedge_option['type'] == 'CE' else 'PE'}{hedge_option['strike']}"
            
            hedge_order = {
                "security_id": hedge_symbol,
                "exchange": "NFO",
                "transaction_type": "BUY",
                "quantity": trade_plan['hedge']['contracts'] * (50 if self.symbol == "NIFTY" else 25),
                "product": "INTRADAY",
                "validity": "DAY",
                "order_type": "LIMIT",
                "price": hedge_option['premium'] * 1.01  # Slightly above ask
            }
            
            print(f"Placing hedge order: BUY {hedge_order['quantity']} {hedge_symbol} @ ₹{hedge_order['price']:.2f}")
            
            # Simulate order response
            hedge_order_id = f"ORD{int(time.time()) + 1}"
            print(f"Hedge order placed successfully! Order ID: {hedge_order_id}")
            
            execution_details["hedge_order"] = {
                "order_id": hedge_order_id,
                "status": "COMPLETE",
                "fill_price": hedge_option['premium'],
                "quantity": hedge_order['quantity'],
                "time": datetime.datetime.now().strftime("%H:%M:%S")
            }
        
        # Place spread sell order if applicable
        if 'spread_sell' in trade_plan:
            spread_option = trade_plan['spread_sell']['option']
            spread_symbol = f"{self.symbol}{trade_plan['date'].replace('-', '')}{'CE' if spread_option['type'] == 'CE' else 'PE'}{spread_option['strike']}"
            
            spread_order = {
                "security_id": spread_symbol,
                "exchange": "NFO",
                "transaction_type": "SELL",
                "quantity": trade_plan['spread_sell']['contracts'] * (50 if self.symbol == "NIFTY" else 25),
                "product": "INTRADAY",
                "validity": "DAY",
                "order_type": "LIMIT",
                "price": spread_option['premium'] * 0.99  # Slightly below bid
            }
            
            print(f"Placing spread sell order: SELL {spread_order['quantity']} {spread_symbol} @ ₹{spread_order['price']:.2f}")
            
            # Simulate order response
            spread_order_id = f"ORD{int(time.time()) + 2}"
            print(f"Spread sell order placed successfully! Order ID: {spread_order_id}")
            
            execution_details["spread_order"] = {
                "order_id": spread_order_id,
                "status": "COMPLETE",
                "fill_price": spread_option['premium'],
                "quantity": spread_order['quantity'],
                "time": datetime.datetime.now().strftime("%H:%M:%S")
            }
        
        # Place stop loss orders
        # In practice, you might use GTT (Good Till Triggered) orders or track manually
        print("\nSetting up stop loss alerts...")
        print(f"Stop Loss Level: {trade_plan['stop_loss']['technical']:.2f}")
        print(f"Premium Stop Loss: {trade_plan['stop_loss']['premium_pct']}%")
        
        # Set target alerts
        print("\nSetting up target alerts...")
        for target, data in trade_plan['targets'].items():
            print(f"Target {target[-1]}: {data['level']:.2f} ({data['pct']}%)")
        
        # Add trade to the list
        self.trades.append({
            "trade_plan": trade_plan,
            "execution": execution_details,
            "status": "ACTIVE",
            "entry_time": datetime.datetime.now().strftime("%H:%M:%S"),
            "exit_time": None,
            "pnl": None
        })
        
        return execution_details
    
    def manage_trade(self, trade_index=0):
        """
        Manage an active trade
        
        Args:
            trade_index (int): Index of the trade to manage
            
        Returns:
            dict: Trade management details
        """
        if trade_index >= len(self.trades) or self.trades[trade_index]['status'] != "ACTIVE":
            print("No active trade to manage.")
            return None
        
        trade = self.trades[trade_index]
        trade_plan = trade['trade_plan']
        
        print("\n--- Trade Management ---")
        print(f"Managing trade: {trade_plan['strategy']} ({trade_plan['direction'] if trade_plan['direction'] else 'Sideways'})")
        
        # In a real implementation, this would monitor price levels and update positions
        # For now, we'll simulate the management process
        
        # Get current price
        current_price = self.dhan_api.get_ltp(self.symbol, self.exchange)
        if not current_price:
            current_price = float(input(f"Enter current {self.symbol} price: "))
        
        print(f"Current price: {current_price:.2f}")
        
        # Get current option premiums
        primary_option = trade_plan['primary']['option']
        current_primary_premium = float(input(f"Enter current premium for {primary_option['strike']} {primary_option['type']}: "))
        
        primary_pnl_pct = ((current_primary_premium / primary_option['premium']) - 1) * 100
        print(f"Primary option P&L: {primary_pnl_pct:.2f}%")
        
        # Check stop loss conditions
        if current_price <= trade_plan['stop_loss']['technical'] and trade_plan['direction'] == "Up":
            print("Stop loss hit (technical level) - Exiting trade")
            return self.exit_trade(trade_index, exit_type="Stop Loss")
        
        if current_price >= trade_plan['stop_loss']['technical'] and trade_plan['direction'] == "Down":
            print("Stop loss hit (technical level) - Exiting trade")
            return self.exit_trade(trade_index, exit_type="Stop Loss")
        
        if primary_pnl_pct <= trade_plan['stop_loss']['premium_pct']:
            print("Stop loss hit (premium percentage) - Exiting trade")
            return self.exit_trade(trade_index, exit_type="Stop Loss")
        
        # Check target conditions
        if 'targets' in trade_plan:
            # Target 1
            if (trade_plan['direction'] == "Up" and current_price >= trade_plan['targets']['t1']['level']) or \
               (trade_plan['direction'] == "Down" and current_price <= trade_plan['targets']['t1']['level']):
                print(f"Target 1 reached - Exiting {trade_plan['targets']['t1']['pct']}% of position")
                return self.partial_exit_trade(trade_index, target="t1")
            
            # Target 2 (if applicable)
            if 't2' in trade_plan['targets']:
                if (trade_plan['direction'] == "Up" and current_price >= trade_plan['targets']['t2']['level']) or \
                   (trade_plan['direction'] == "Down" and current_price <= trade_plan['targets']['t2']['level']):
                    print(f"Target 2 reached - Exiting {trade_plan['targets']['t2']['pct']}% of position")
                    return self.partial_exit_trade(trade_index, target="t2")
            
            # Target 3 (if applicable)
            if 't3' in trade_plan['targets']:
                if (trade_plan['direction'] == "Up" and current_price >= trade_plan['targets']['t3']['level']) or \
                   (trade_plan['direction'] == "Down" and current_price <= trade_plan['targets']['t3']['level']):
                    print(f"Target 3 reached - Exiting {trade_plan['targets']['t3']['pct']}% of position")
                    return self.exit_trade(trade_index, exit_type="Target 3")
        
        # Manage hedge
        if 'hedge' in trade_plan and primary_pnl_pct > 0:
            hedge_option = trade_plan['hedge']['option']
            current_hedge_premium = float(input(f"Enter current premium for {hedge_option['strike']} {hedge_option['type']}: "))
            
            hedge_pnl_pct = ((current_hedge_premium / hedge_option['premium']) - 1) * 100
            print(f"Hedge option P&L: {hedge_pnl_pct:.2f}%")
            
            # Reduce hedge if in profit
            if primary_pnl_pct >= 50:
                print("Primary position in good profit - Exiting hedge")
                return self.exit_hedge(trade_index)
        
        # Move stop to breakeven after 1R move
        if primary_pnl_pct >= 30:  # Approximately 1R move in premium
            print("Moving stop loss to breakeven")
            trade_plan['stop_loss']['premium_pct'] = -5  # Small buffer below breakeven
            print(f"New stop loss: {trade_plan['stop_loss']['premium_pct']}% premium loss")
        
        # Trailing stop for runner portion
        if primary_pnl_pct >= 100:  # 2R move
            print("Activating trailing stop for runner portion")
            # Set trailing stop at previous pivot level
            if trade_plan['direction'] == "Up":
                # Find nearest pivot level below current price
                for level in [self.pivots['R1'], self.pivots['PP'], self.cpr['TC']]:
                    if level < current_price:
                        trade_plan['stop_loss']['technical'] = level
                        break
            else:  # Down
                # Find nearest pivot level above current price
                for level in [self.pivots['S1'], self.pivots['PP'], self.cpr['BC']]:
                    if level > current_price:
                        trade_plan['stop_loss']['technical'] = level
                        break
                        
            print(f"New trailing stop level: {trade_plan['stop_loss']['technical']:.2f}")
        
        return {
            "status": "ACTIVE",
            "current_price": current_price,
            "primary_premium": current_primary_premium,
            "primary_pnl_pct": primary_pnl_pct
        }
    
    def partial_exit_trade(self, trade_index, target):
        """
        Partially exit a trade at target
        
        Args:
            trade_index (int): Index of the trade to exit
            target (str): Target identifier
            
        Returns:
            dict: Partial exit details
        """
        if trade_index >= len(self.trades) or self.trades[trade_index]['status'] != "ACTIVE":
            print("No active trade to exit.")
            return None
        
        trade = self.trades[trade_index]
        trade_plan = trade['trade_plan']
        
        # Get exit percentage for this target
        exit_pct = trade_plan['targets'][target]['pct']
        
        # Calculate quantity to exit
        primary_option = trade_plan['primary']['option']
        primary_total_qty = trade_plan['primary']['contracts'] * (50 if self.symbol == "NIFTY" else 25)
        exit_qty = int(primary_total_qty * (exit_pct / 100))
        
        # Get current premium
        current_premium = float(input(f"Enter current premium for {primary_option['strike']} {primary_option['type']}: "))
        
        # Calculate P&L
        pnl = (current_premium - primary_option['premium']) * exit_qty
        pnl_pct = ((current_premium / primary_option['premium']) - 1) * 100
        
        # Create exit order
        primary_symbol = f"{self.symbol}{trade_plan['date'].replace('-', '')}{'CE' if primary_option['type'] == 'CE' else 'PE'}{primary_option['strike']}"
        
        exit_order = {
            "security_id": primary_symbol,
            "exchange": "NFO",
            "transaction_type": "SELL",
            "quantity": exit_qty,
            "product": "INTRADAY",
            "validity": "DAY",
            "order_type": "LIMIT",
            "price": current_premium * 0.99  # Slightly below bid
        }
        
        print(f"Placing partial exit order: SELL {exit_order['quantity']} {primary_symbol} @ ₹{exit_order['price']:.2f}")
        
        # Simulate order response
        exit_order_id = f"ORD{int(time.time())}"
        print(f"Partial exit order placed successfully! Order ID: {exit_order_id}")
        
        # Update position size
        trade_plan['primary']['contracts'] = trade_plan['primary']['contracts'] * (1 - exit_pct / 100)
        
        # If spread is involved, exit proportionate quantity
        if 'spread_sell' in trade_plan:
            spread_option = trade_plan['spread_sell']['option']
            spread_exit_qty = int(exit_qty)  # 1:1 ratio
            
            spread_symbol = f"{self.symbol}{trade_plan['date'].replace('-', '')}{'CE' if spread_option['type'] == 'CE' else 'PE'}{spread_option['strike']}"
            
            current_spread_premium = float(input(f"Enter current premium for {spread_option['strike']} {spread_option['type']}: "))
            
            spread_exit_order = {
                "security_id": spread_symbol,
                "exchange": "NFO",
                "transaction_type": "BUY",  # Buy to close sold option
                "quantity": spread_exit_qty,
                "product": "INTRADAY",
                "validity": "DAY",
                "order_type": "LIMIT",
                "price": current_spread_premium * 1.01  # Slightly above ask
            }
            
            print(f"Placing spread exit order: BUY {spread_exit_order['quantity']} {spread_symbol} @ ₹{spread_exit_order['price']:.2f}")
            
            # Simulate
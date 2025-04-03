import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import time
import datetime
import subprocess
import threading
import multiprocessing
import queue
import sys
import os
import math

# Import from the scraper file
from asyncio_scraping import LVMHPriceScraper

class LVMHPriceForecaster:
    def __init__(self, forecast_horizon=5, model_update_interval=10):
        # Configuration
        self.forecast_horizon = forecast_horizon  # Number of steps to forecast ahead
        self.model_update_interval = model_update_interval  # How many new data points before retraining
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.new_data_count = 0
        self.data = pd.DataFrame(columns=['timestamp', 'price', 'variation'])
        self.sequence_length = 10  # How many previous prices to use for prediction
        self.data_queue = multiprocessing.Queue()  # Queue for sharing data between processes
        
        # For saving and evaluating forecasts
        self.forecasts_df = pd.DataFrame(columns=['timestamp', 'actual_price', 'forecasted_price'])
        self.rmse_history = []
        self.rmse_timestamps = []
        
        # Set up figure for real-time plotting (prices and forecast)
        plt.ion()
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = self.fig.add_gridspec(2, 1, height_ratios=[3, 1])
        
        # Price plot
        self.ax1 = self.fig.add_subplot(self.gs[0])
        self.ax1.set_title('LVMH Stock Price - Real-time Forecast')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Price')
        self.price_line, = self.ax1.plot([], [], 'b-', label='Actual Price')
        self.forecast_line, = self.ax1.plot([], [], 'r--', label='Forecast')
        self.ax1.legend()
        
        # RMSE plot
        self.ax2 = self.fig.add_subplot(self.gs[1])
        self.ax2.set_title('Forecast Error (RMSE)')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('RMSE')
        self.rmse_line, = self.ax2.plot([], [], 'g-', linewidth=2)
        
        # Tighten layout
        self.fig.tight_layout()
        
    def build_model(self):
        """Build and compile the LSTM model"""
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.sequence_length, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(self.forecast_horizon))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def preprocess_data(self):
        """Preprocess the data for the LSTM model"""
        if len(self.data) < self.sequence_length + self.forecast_horizon:
            return None, None
            
        # Extract price data and normalize
        prices = self.data['price'].values.reshape(-1, 1)
        scaled_prices = self.scaler.fit_transform(prices)
        
        # Create sequences for training
        X, y = [], []
        for i in range(len(scaled_prices) - self.sequence_length - self.forecast_horizon + 1):
            X.append(scaled_prices[i:(i + self.sequence_length)])
            y.append(scaled_prices[i + self.sequence_length:i + self.sequence_length + self.forecast_horizon])
        
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Train or update the LSTM model with current data"""
        X, y = self.preprocess_data()
        if X is None or len(X) < 2:
            print("Not enough data to train model yet")
            return False
            
        if self.model is None:
            print("Building new model...")
            self.model = self.build_model()
        
        print(f"Training model on {len(X)} sequences...")
        self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)
        print("Model training complete")
        return True
    
    def make_forecast(self):
        """Generate forecast for the next 'forecast_horizon' time steps"""
        if self.model is None or len(self.data) < self.sequence_length:
            return None
            
        # Prepare the most recent sequence for prediction
        latest_data = self.data['price'].values[-self.sequence_length:].reshape(-1, 1)
        scaled_latest = self.scaler.transform(latest_data)
        
        # Reshape for LSTM input [samples, time steps, features]
        X_latest = scaled_latest.reshape(1, self.sequence_length, 1)
        
        # Predict the next steps
        scaled_forecast = self.model.predict(X_latest, verbose=0)
        
        # Inverse transform to get actual price values
        forecast = self.scaler.inverse_transform(scaled_forecast)[0]
        
        return forecast
    
    def calculate_rmse(self):
        """Calculate RMSE between forecasted and actual prices"""
        if len(self.forecasts_df) < 2:
            return None
            
        # Use only rows where we have both actual and forecasted prices
        valid_rows = self.forecasts_df.dropna(subset=['actual_price', 'forecasted_price'])
        
        if len(valid_rows) < 1:
            return None
            
        # Calculate RMSE
        rmse = math.sqrt(mean_squared_error(
            valid_rows['actual_price'],
            valid_rows['forecasted_price']
        ))
        
        # Add to history
        self.rmse_history.append(rmse)
        self.rmse_timestamps.append(datetime.datetime.now())
        
        print(f"Current RMSE: {rmse:.4f}")
        return rmse
    
    def save_forecast_data(self, forecast):
        """Save actual and forecasted data for comparison"""
        if forecast is None or len(self.data) < self.forecast_horizon:
            return
            
        current_time = datetime.datetime.now()
        
        # For the initial forecast, we can only store the prediction
        if len(self.forecasts_df) < self.forecast_horizon:
            for i in range(self.forecast_horizon):
                # We only have forecasts, no actuals to compare yet
                new_row = pd.DataFrame({
                    'timestamp': [current_time + datetime.timedelta(minutes=i)],
                    'forecasted_price': [forecast[i]],
                    'actual_price': [None]
                })
                self.forecasts_df = pd.concat([self.forecasts_df, new_row], ignore_index=True)
        else:
            # Update the oldest forecasts with actual values
            latest_actual = self.data['price'].iloc[-1]
            timestamp = self.data['timestamp'].iloc[-1]
            
            # Find the row to update
            for i, row in self.forecasts_df.iterrows():
                if pd.isna(row['actual_price']):
                    self.forecasts_df.at[i, 'actual_price'] = latest_actual
                    self.forecasts_df.at[i, 'timestamp'] = timestamp
                    break
            
            # Add new forecast
            new_row = pd.DataFrame({
                'timestamp': [current_time + datetime.timedelta(minutes=self.forecast_horizon)],
                'forecasted_price': [forecast[-1]],
                'actual_price': [None]
            })
            self.forecasts_df = pd.concat([self.forecasts_df, new_row], ignore_index=True)
    
    def export_data(self):
        """Save the collected data and forecasts to CSV files"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save actual data
        if not self.data.empty:
            data_filename = f"lvmh_actual_data_{timestamp}.csv"
            self.data.to_csv(data_filename, index=False)
            print(f"Actual data saved to {data_filename}")
        
        # Save forecast comparison data
        if not self.forecasts_df.empty:
            forecast_filename = f"lvmh_forecast_comparison_{timestamp}.csv"
            self.forecasts_df.to_csv(forecast_filename, index=False)
            print(f"Forecast comparison data saved to {forecast_filename}")
        
        # Save RMSE history
        if self.rmse_history:
            rmse_df = pd.DataFrame({
                'timestamp': self.rmse_timestamps,
                'rmse': self.rmse_history
            })
            rmse_filename = f"lvmh_rmse_history_{timestamp}.csv"
            rmse_df.to_csv(rmse_filename, index=False)
            print(f"RMSE history saved to {rmse_filename}")
    
    def update_plot(self, forecast=None):
        """Update the real-time plot with new data, forecast and RMSE"""
        if len(self.data) < 2:
            return
            
        # Get timestamps and prices for plotting
        timestamps = self.data['timestamp']
        prices = self.data['price']
        
        # Update actual price line
        self.price_line.set_xdata(range(len(timestamps)))
        self.price_line.set_ydata(prices)
        
        # Update forecast line if available
        if forecast is not None and len(prices) > 0:
            # Create forecast timestamps (future points)
            last_idx = len(timestamps) - 1
            forecast_indices = range(last_idx, last_idx + self.forecast_horizon)
            
            # Create extended line that connects actual data with forecast
            extended_indices = list(range(last_idx - min(3, last_idx), last_idx)) + list(forecast_indices)
            extended_values = list(prices.iloc[-min(3, len(prices)):]) + list(forecast)
            
            self.forecast_line.set_xdata(extended_indices)
            self.forecast_line.set_ydata(extended_values)
        
        # Adjust price plot limits
        self.ax1.set_xlim(0, len(timestamps) + self.forecast_horizon)
        if len(prices) > 0:
            min_price = min(prices.min(), (forecast.min() if forecast is not None else float('inf')))
            max_price = max(prices.max(), (forecast.max() if forecast is not None else float('-inf')))
            margin = (max_price - min_price) * 0.1 if max_price > min_price else 1.0
            self.ax1.set_ylim(min_price - margin, max_price + margin)
        
        # Update x-axis labels to show times
        if len(timestamps) > 0:
            # Show fewer tick labels to avoid overcrowding
            tick_indices = list(range(0, len(timestamps), max(1, len(timestamps) // 10)))
            if len(timestamps) - 1 not in tick_indices:
                tick_indices.append(len(timestamps) - 1)
                
            self.ax1.set_xticks(tick_indices)
            self.ax1.set_xticklabels([timestamps.iloc[i].strftime('%H:%M:%S') for i in tick_indices], rotation=45)
        
        # Update RMSE plot if we have data
        if self.rmse_history:
            self.rmse_line.set_xdata(range(len(self.rmse_history)))
            self.rmse_line.set_ydata(self.rmse_history)
            
            # Adjust RMSE plot limits
            self.ax2.set_xlim(0, max(10, len(self.rmse_history)))
            if len(self.rmse_history) > 0:
                min_rmse = min(self.rmse_history)
                max_rmse = max(self.rmse_history)
                margin = (max_rmse - min_rmse) * 0.1 if max_rmse > min_rmse else 0.1
                self.ax2.set_ylim(max(0, min_rmse - margin), max_rmse + margin)
                
            # Add more informative text to RMSE plot
            if len(self.rmse_history) > 0:
                current_rmse = self.rmse_history[-1]
                avg_rmse = sum(self.rmse_history) / len(self.rmse_history)
                self.ax2.set_title(f'Forecast Error (RMSE) - Current: {current_rmse:.4f}, Avg: {avg_rmse:.4f}')
        
        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def scraper_process(self):
        """This function runs in a separate process to collect price data"""
        try:
            # Create a modified version of the original price scraper
            scraper = LVMHPriceScraper()
            
            # Override the update_dataframe method to also send data to the queue
            original_update = scraper.update_dataframe
            
            def new_update_dataframe(current_price, variation):
                # Call the original method to update scraper's dataframe
                original_update(current_price, variation)
                
                # Send the new data point to the forecaster process
                timestamp = datetime.datetime.now()
                self.data_queue.put({
                    'timestamp': timestamp,
                    'price': current_price,
                    'variation': variation
                })
                
            # Replace the method
            scraper.update_dataframe = new_update_dataframe
            
            # Start the monitoring (this is a blocking call)
            import asyncio
            asyncio.run(scraper.monitor_price())
            
        except KeyboardInterrupt:
            print("Scraper process received interrupt signal")
        except Exception as e:
            print(f"Error in scraper process: {e}")
            
    def check_for_new_data(self):
        """Check for new data in the queue from the scraper process"""
        new_data_received = False
        
        # Get all available data from the queue
        while not self.data_queue.empty():
            try:
                # Get data point from the queue
                data_point = self.data_queue.get(block=False)
                
                # Add to our dataframe
                new_row = pd.DataFrame([data_point])
                if self.data.empty:
                    self.data = new_row
                else:
                    self.data = pd.concat([self.data, new_row], ignore_index=True)
                
                self.new_data_count += 1
                new_data_received = True
                
            except queue.Empty:
                break
                
        return new_data_received
    
    def run(self):
        """Main loop for the forecaster"""
        # Start the data collection process
        print("Starting data collection process...")
        scraper_proc = multiprocessing.Process(target=self.scraper_process)
        scraper_proc.start()
            
        print(f"Forecaster started with horizon = {self.forecast_horizon} steps")
        print("Waiting for initial data...")
        
        try:
            while True:
                # Check for new data
                if self.check_for_new_data():
                    print(f"New data received: {len(self.data)} total points")
                    
                    # If we have enough new data, retrain the model
                    if self.new_data_count >= self.model_update_interval:
                        self.train_model()
                        self.new_data_count = 0
                    
                    # Make a forecast if possible
                    forecast = self.make_forecast()
                    if forecast is not None:
                        print(f"Forecast for next {self.forecast_horizon} steps: {forecast}")
                        
                        # Save forecast data for comparison
                        self.save_forecast_data(forecast)
                        
                        # Calculate RMSE if we have enough data
                        self.calculate_rmse()
                    
                    # Update the plot
                    self.update_plot(forecast)
                
                # Wait before checking again
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nStopping forecaster...")
        finally:
            # Export data before shutting down
            self.export_data()
            
            # Clean up
            print("Terminating data collection process...")
            if scraper_proc.is_alive():
                scraper_proc.terminate()
                scraper_proc.join(timeout=5)
            
            plt.ioff()
            plt.close('all')
            print("Forecaster stopped.")

# Modified version of the scraper for direct integration
def create_direct_integrated_scraper():
    """Create a version of the scraper file that can be imported without running"""
    # Check if the integration file already exists
    if os.path.exists("asyncio_scraping.py"):
        return

    # Read the original scraper file
    with open("asyncio scraping.py", "r") as f:
        content = f.read()
    
    # Modify the content to prevent auto-execution
    modified_content = content.replace(
        'if __name__ == "__main__":', 
        'if __name__ == "__main__" and False:'  # This prevents auto-execution
    )
    
    # Write to the new file
    with open("asyncio_scraping.py", "w") as f:
        f.write(modified_content)

if __name__ == "__main__":
    # Create the integration file
    create_direct_integrated_scraper()
    
    # Allow command-line argument for forecast horizon
    forecast_horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    # Create and run the forecaster
    forecaster = LVMHPriceForecaster(
        forecast_horizon=forecast_horizon,
        model_update_interval=10
    )
    forecaster.run()
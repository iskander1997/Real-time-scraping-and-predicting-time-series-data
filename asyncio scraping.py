import asyncio
import pandas as pd
import aiohttp
import datetime
import json
import signal
import sys

class LVMHPriceScraper:
    def __init__(self):
        # MC.PA is the ticker for LVMH on Yahoo Finance (Euronext Paris)
        self.ticker = "MC.PA"
        self.url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.ticker}?interval=1m"
        self.df = pd.DataFrame(columns=['timestamp', 'price', 'variation'])
        self.first_price = None
        self.last_price = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.running = True
        
    async def fetch_price(self, session):
        try:
            async with session.get(self.url, headers=self.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Extract the most recent price
                    result = data['chart']['result'][0]
                    meta = result['meta']
                    current_price = meta.get('regularMarketPrice', None)
                    
                    if current_price:
                        return float(current_price)
                    else:
                        print("Price data not found in response")
                        return None
                else:
                    print(f"Failed to fetch data: Status code {response.status}")
                    return None
        except Exception as e:
            print(f"Error fetching price: {e}")
            return None
    
    def calculate_variation(self, current_price):
        if self.first_price is None:
            self.first_price = current_price
            variation = 0.0
        else:
            variation = (current_price / self.first_price) - 1
        
        return variation
    
    def update_dataframe(self, current_price, variation):
        timestamp = datetime.datetime.now()
        new_row = pd.DataFrame({
            'timestamp': [timestamp],
            'price': [current_price],
            'variation': [variation]
        })
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        print(f"Price: ${current_price:.2f} | Variation: {variation:.2%} | {timestamp}")
    
    def save_data(self):
        """Save the collected data to a CSV file"""
        if not self.df.empty:
            csv_filename = f"lvmh_price_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.df.to_csv(csv_filename, index=False)
            print(f"\nData saved to {csv_filename}")
        else:
            print("\nNo data to save")
        
    async def monitor_price(self):
        async with aiohttp.ClientSession() as session:
            while self.running:
                try:
                    current_price = await self.fetch_price(session)
                    
                    if current_price is not None:
                        if self.last_price is None or current_price != self.last_price:
                            variation = self.calculate_variation(current_price)
                            self.update_dataframe(current_price, variation)
                            self.last_price = current_price
                    
                    # Wait before the next request
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    # This will be raised when the task is cancelled
                    break
                except Exception as e:
                    print(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(20)  # Wait longer if there's an error

def setup_signal_handlers(scraper):
    """Set up signal handlers for graceful shutdown"""
    def signal_handler(sig, frame):
        print("\nStopping the script gracefully...")
        scraper.running = False
        scraper.save_data()
        print("Script stopped.")
        sys.exit(0)
    
    # Register the handler for SIGINT (Ctrl+C)
    signal.signal(signal.SIGINT, signal_handler)

async def main():
    try:
        scraper = LVMHPriceScraper()
        # Set up the signal handler
        setup_signal_handlers(scraper)
        
        print("Starting LVMH price monitoring. Press Ctrl+C to stop.")
        await scraper.monitor_price()
    except KeyboardInterrupt:
        print("\nMonitoring stopped due to keyboard interrupt.")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # This will execute even if there's an exception
        if 'scraper' in locals():
            scraper.save_data()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This will catch Ctrl+C at the outermost level
        pass
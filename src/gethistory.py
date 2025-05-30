from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
import pandas as pd
import asyncio

# List of assets to fetch data for
asset_list_raw = """#AAPL
#AAPL_otc
#AXP
#AXP_otc
#BA
#BA_otc
#CSCO
#CSCO_otc
#FB
#FB_otc
#INTC
#INTC_otc
#JNJ
#JNJ_otc
#JPM
#MCD
#MCD_otc
#MSFT
#MSFT_otc
#PFE
#PFE_otc
#TSLA
#TSLA_otc
#XOM
#XOM_otc
100GBP
100GBP_otc
ADA-USD_otc
AEDCNY_otc
AEX25
AMZN_otc
AUDCAD
AUDCAD_otc
AUDCHF
AUDCHF_otc
AUDJPY
AUDJPY_otc
AUDNZD_otc
AUDUSD
AUDUSD_otc
AUS200
AUS200_otc
AVAX_otc
BABA
BABA_otc
BCHEUR
BCHGBP
BCHJPY
BHDCNY_otc
BITB_otc
BNB-USD_otc
BTCGBP
BTCJPY
BTCUSD
BTCUSD_otc
CAC40
CADCHF
CADCHF_otc
CADJPY
CADJPY_otc
CHFJPY
CHFJPY_otc
CHFNOK_otc
CITI
CITI_otc
D30EUR
D30EUR_otc
DASH_USD
DJI30
DJI30_otc
DOGE_otc
DOTUSD_otc
E35EUR
E35EUR_otc
E50EUR
E50EUR_otc
ETHUSD
ETHUSD_otc
EURAUD
EURCAD
EURCHF
EURCHF_otc
EURGBP
EURGBP_otc
EURHUF_otc
EURJPY
EURJPY_otc
EURNZD_otc
EURRUB_otc
EURTRY_otc
EURUSD
EURUSD_otc
F40EUR
F40EUR_otc
FDX_otc
GBPAUD
GBPAUD_otc
GBPCAD
GBPCHF
GBPJPY
GBPJPY_otc
GBPUSD
GBPUSD_otc
H33HKD
IRRUSD_otc
JODCNY_otc
JPN225
JPN225_otc
LBPUSD_otc
LINK_otc
LNKUSD
LTCUSD_otc
MADUSD_otc
MATIC_otc
NASUSD
NASUSD_otc
NFLX
NFLX_otc
NZDJPY_otc
NZDUSD_otc
OMRCNY_otc
QARCNY_otc
SARCNY_otc
SMI20
SOL-USD_otc
SP500
SP500_otc
SYPUSD_otc
TNDUSD_otc
TON-USD_otc
TRX-USD_otc
TWITTER
TWITTER_otc
UKBrent
UKBrent_otc
USCrude
USCrude_otc
USDARS_otc
USDBDT_otc
USDBRL_otc
USDCAD
USDCAD_otc
USDCHF
USDCHF_otc
USDCLP_otc
USDCNH_otc
USDCOP_otc
USDDZD_otc
USDEGP_otc
USDIDR_otc
USDINR_otc
USDJPY
USDJPY_otc
USDMXN_otc
USDMYR_otc
USDPHP_otc
USDPKR_otc
USDRUB_otc
USDSGD_otc
USDTHB_otc
USDVND_otc
VISA_otc
XAGEUR
XAGUSD
XAGUSD_otc
XAUEUR
XAUUSD
XAUUSD_otc
XNGUSD
XNGUSD_otc
XPDUSD
XPDUSD_otc
XPTUSD
XPTUSD_otc
XRPUSD_otc
YERUSD_otc"""

assets = [name.lstrip('#') for name in asset_list_raw.splitlines() if name.strip()]

# Main part of the code
async def main(ssid: str):
    # The api automatically detects if the 'ssid' is for real or demo account
    api = PocketOptionAsync(ssid)
    print("Attempting to connect and initialize API...")
    await asyncio.sleep(5) # Allow time for API initialization/connection
    print("API potentially initialized.")

    all_asset_dataframes = []
    
    # Candle parameters (period: 1 sec, count: very large number)
    candle_period = 1
    # Note: 36,000,000,000 candles is an extremely large number.
    # The API will likely return its maximum available candles or cap this.
    candle_count = 36000000000 

    for asset_name in assets:
        print(f"Fetching candles for {asset_name}...")
        try:
            candles = await api.get_candles(asset_name, candle_period, candle_count)
            
            if candles:
                # print(f"Raw Candles for {asset_name}: {candles[:2]}") # Print first 2 candles for brevity
                candles_pd = pd.DataFrame.from_dict(candles)
                candles_pd['asset'] = asset_name  # Add a column for the asset name
                all_asset_dataframes.append(candles_pd)
                print(f"Successfully fetched {len(candles)} candles for {asset_name}.")
            else:
                print(f"No candles returned for {asset_name}.")
                
        except Exception as e:
            print(f"Error fetching data for {asset_name}: {e}")
        
        await asyncio.sleep(1) # Small delay to be polite to the API between requests

    if all_asset_dataframes:
        print("Combining data for all assets...")
        combined_df = pd.concat(all_asset_dataframes, ignore_index=True)
        
        output_filename = 'all_assets_candles2.csv'
        combined_df.to_csv(output_filename, index=False)
        print(f"All fetched candle data saved to {output_filename}")
    else:
        print("No data was fetched for any asset. CSV file not created.")
    
    # It's good practice to close the session if the API provides a method,
    # though it's not explicitly shown in the original snippet or typical for this library.
    # If `api.close()` or similar exists, call it here.
    # For PocketOptionAsync, often connections are managed by an underlying websocket
    # that might close on program exit or if the `api` object is garbage collected.

if __name__ == '__main__':
    ssid = input('Please enter your ssid: ')
    asyncio.run(main(ssid))
import pandas as pd
import numpy as np

class RealCarbonLoader:
    def __init__(self, key_csv="carbon_intensity_2024_06_15.csv"):
        # Load AUTHENTIC CSV data
        self.df = pd.read_csv(key_csv)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        
    def get_trace(self, region="caiso"):
        if region == "caiso":
            region_key = "CAISO_North"
        elif region == "uk":
            region_key = "UK_Grid"
        else:
            raise ValueError(f"Region {region} not found in real dataset.")
            
        # Filter and extract raw values
        trace = self.df[self.df["Region"] == region_key]["CarbonIntensity_gCO2_kWh"].values.tolist()
        return trace

if __name__ == "__main__":
    loader = RealCarbonLoader()
    print("Loaded CAISO Trace (First 5):", loader.get_trace("caiso")[:5])
    print("Loaded UK Trace (First 5):", loader.get_trace("uk")[:5])

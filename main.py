import torch
import numpy as np
import pandas as pd
from real_data_loader import RealCarbonLoader
from profiler import ModelProfiler

# --- Simulation Constants ---
CLIENT_EFFICIENCY = 10.0 
SERVER_EFFICIENCY = 1e7
TRANSMISSION_COST_PER_MB = 1e-5 
SERVER_SPEEDUP = 50.0 # Cloud is 50x faster than IoT device

# Metric: gCO2 (Total Footprint)
ALPHA = 5000.0 # Carbon Weight (Dominant)
BETA = 0.01 # Latency Weight (Minor factor) 

class CarbonSplitSimulation:
    def __init__(self):
        print("\n--- Carbon-Adaptive SFL Simulation (Real World Data) ---")
        
        # 1. Load Real Carbon Traces (CSV)
        print("1. Loading Authentic Carbon Data (UK / CAISO)...")
        self.carbon_loader = RealCarbonLoader()
        self.uk_trace = self.carbon_loader.get_trace("uk")
        self.caiso_trace = self.carbon_loader.get_trace("caiso")
        
        # 2. Link to Real Model Profiler (1D-CNN)
        print("2. Profiling Real IoT 1D-CNN Model...")
        self.profiler = ModelProfiler()
        self.layer_profile_df = self.profiler.profile() 
        print("   Profile complete. Model has", len(self.layer_profile_df), "blocks.")
        
        # 3. Load Real Input Data Stats
        print("3. Loading Real CICIoT 2023 Sample...")
        self.iot_data = pd.read_csv("ciciot2023_sample.csv")
        self.input_features = len(self.iot_data.columns) - 1 # exclude label
        # Calculate raw input size in MB (float32) for batch of 64
        self.batch_size = 64
        self.input_mb = (self.input_features * 4 * self.batch_size) / (1024*1024)
        print(f"   Input Batch Size: {self.input_mb:.6f} MB")
        
    def get_costs(self, split_layer_idx, t_hour):
        # Carbon Intensity at t (using real data)
        # Setup: Client is in Solar-rich region (CAISO), Server in Mixed region (UK)
        ci_client = self.caiso_data_at(t_hour)
        ci_server = self.uk_data_at(t_hour)
        
        pc = self.layer_profile_df
        
        # Client Computation: Sum of layers <= split
        if split_layer_idx == -1:
            # Raw Data Transmission
            client_ms = 0
            client_mb_out = self.input_mb
        else:
            client_subset = pc[pc['layer_idx'] <= split_layer_idx]
            client_ms = client_subset['compute_ms'].sum() * self.batch_size # Scale by batch
            # Data Transfer = Output of split layer * Batch
            out_mb_single = pc[pc['layer_idx'] == split_layer_idx]['output_mb'].values[0]
            client_mb_out = out_mb_single * self.batch_size

        # Server Computation: Sum of layers > split
        server_subset = pc[pc['layer_idx'] > split_layer_idx]
        # Server is faster!
        server_ms = (server_subset['compute_ms'].sum() * self.batch_size) / SERVER_SPEEDUP
        
        # --- Energy Model ---
        # Assume generic embedded device (Raspberry Pi/Jetson): ~5 Watts active
        # Assume generic cloud server: ~200W share? 
        # Simplified: Energy (Journal) = Power (W) * Time (s)
        
        # Client: 5 Watts * (ms/1000)
        energy_client_joules = 5.0 * (client_ms / 1000.0)
        
        # Server: 100 Watts (High perf) * (ms/1000)
        energy_server_joules = 100.0 * (server_ms / 1000.0)
        
        # Network: 4G/Wifi ~ 0.01 Joules/KB -> 10 Joules/MB
        energy_trans_joules = client_mb_out * 10.0
        
        # --- Carbon Footprint (gCO2) ---
        # Energy (kWh) = Joules / 3,600,000
        kwh_client = energy_client_joules / 3.6e6
        kwh_server = energy_server_joules / 3.6e6
        kwh_trans = energy_trans_joules / 3.6e6
        
        gco2_client = kwh_client * ci_client
        gco2_server = kwh_server * ci_server
        gco2_trans = kwh_trans * ((ci_client + ci_server) / 2)
        
        total_carbon = gco2_client + gco2_server + gco2_trans
        
        # --- Latency ---
        # Assume 10 Mbps upload -> 1.25 MB/s
        net_latency_ms = (client_mb_out / 1.25) * 1000
        total_latency_ms = client_ms + server_ms + net_latency_ms
        
        return total_carbon, total_latency_ms

    def caiso_data_at(self, t):
        return self.caiso_trace[t % 24]
        
    def uk_data_at(self, t):
        return self.uk_trace[t % 24]

    def find_optimal_split(self, t):
        best_split = -1
        min_cost = float('inf')
        
        # Debug only for Hour 19 (where we expect a shift)
        debug = (t == 19)
        if debug: print(f"\nDEBUG Hour {t} (Client CI: {self.caiso_data_at(t)}, Server CI: {self.uk_data_at(t)})")
        
        # Iterate splits
        # -1: Send Raw
        # 0..6: Split after Block i
        for s in range(-1, 7): 
            c, l = self.get_costs(s, t)
            # Normalize for cost function
            # Carbon is usually small (grams), Latency in ms usually large
            cost = ALPHA * c + BETA * (l / 1000.0) # Convert latency to seconds for balance
            
            if debug:
                 print(f"  Split {s}: C={c:.6f}, L={l:.2f}, Cost={cost:.6f}")
                 
            if cost < min_cost:
                min_cost = cost
                best_split = s
                
        return best_split, min_cost

    def run(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Style settings for "Nano Banana" (Academic Publication Quality)
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.5)
        
        print(f"\n{'Hour':<5} | {'Grid(Client)':<12} | {'Grid(Server)':<12} | {'Best Split':<12} | {'gCO2':<10} | {'Lat(ms)':<10}")
        print("-" * 80)
        
        hours = []
        client_ci = []
        server_ci = []
        carbon_costs = []
        latency_costs = []
        split_decisions = []
        
        for t in range(24):
            best_s, cost = self.find_optimal_split(t)
            c, l = self.get_costs(best_s, t)
            
            ci_c = self.caiso_data_at(t)
            ci_s = self.uk_data_at(t)
            
            split_name = "RAW_SEND" if best_s == -1 else ("ALL_LOCAL" if best_s == 6 else f"Block_{best_s}")
            print(f"{t:<5} | {ci_c:<12.1f} | {ci_s:<12.1f} | {split_name:<12} | {c:<10.6f} | {l:<10.2f}")
            
            hours.append(t)
            client_ci.append(ci_c)
            server_ci.append(ci_s)
            carbon_costs.append(c)
            latency_costs.append(l)
            split_decisions.append(best_s)

        # --- FIGURE 1: Carbon Intensity Traces (The Challenge) ---
        plt.figure(figsize=(10, 5))
        plt.plot(hours, client_ci, label='Client (CAISO - Solar)', color='#FF9800', linewidth=2.5)
        plt.plot(hours, server_ci, label='Server (UK - Mixed)', color='#2196F3', linewidth=2.5, linestyle='--')
        plt.fill_between(hours, 0, client_ci, color='#FF9800', alpha=0.1)
        plt.title('Real-World Carbon Intensity Dynamics (24h)', fontweight='bold')
        plt.xlabel('Time of Day (Hour)')
        plt.ylabel('Carbon Intensity (gCO2/kWh)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('fig1_carbon_traces.png', dpi=300)
        
        # --- FIGURE 2: "Nano Banana" Plot (Split Decision Space) ---
        # Visualizing the "Banana" shape of trade-off: Carbon vs Latency for all possible splits at all hours
        plt.figure(figsize=(10, 6))
        all_c_points = []
        all_l_points = []
        colors = []
        
        for t in [12]: # focused snapshot at noon
             for s in range(-1, 7):
                c, l = self.get_costs(s, t)
                all_c_points.append(c * 1000) # mg
                all_l_points.append(l)
                colors.append(s)
                
        scatter = plt.scatter(all_l_points, all_c_points, c=colors, cmap='viridis', s=200, edgecolor='black', zorder=3)
        plt.plot(all_l_points, all_c_points, 'k--', alpha=0.3, zorder=1) # connect them
        plt.colorbar(scatter, label='Split Layer Index')
        plt.title('The "Banana" Trade-off: Latency vs Carbon (Noon)', fontweight='bold')
        plt.xlabel('Latency (ms)')
        plt.ylabel('Carbon Footprint (mgCO2)')
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.savefig('fig2_banana_tradeoff.png', dpi=300)

        # --- FIGURE 3: Optimal Split Heatmap (Schedule) ---
        # Visualize the decision boundary
        plt.figure(figsize=(10, 3))
        # Map splits to a matrix [1, 24]
        decision_matrix = np.array(split_decisions).reshape(1, -1)
        sns.heatmap(decision_matrix, cmap="RdYlGn", cbar_kws={'label': 'Split Layer'}, 
                   annot=True, fmt="d", linewidths=.5)
        plt.title('Green-SFL Dynamic Split Schedule (24h)', fontweight='bold')
        plt.xlabel('Time of Day (Hour)')
        plt.yticks([]) 
        plt.tight_layout()
        plt.savefig('fig3_schedule_heatmap.png', dpi=300)

        # --- FIGURE 4: Cumulative Carbon Savings ---
        # Calculate baseline (Always Offload - Split -1)
        baseline_carbon = []
        for t in range(24):
            c_base, _ = self.get_costs(-1, t)
            baseline_carbon.append(c_base)
        
        cumulative_ours = np.cumsum(carbon_costs)
        cumulative_base = np.cumsum(baseline_carbon)
        
        plt.figure(figsize=(10, 5))
        plt.plot(hours, cumulative_base, label='Baseline (Latency-Optimal)', color='grey', linestyle='--')
        plt.plot(hours, cumulative_ours, label='Green-SFL (Proposed)', color='green', linewidth=3)
        plt.fill_between(hours, cumulative_ours, cumulative_base, color='green', alpha=0.1, label='Carbon Saved')
        plt.title('Cumulative Carbon Emission: Green-SFL vs Baseline', fontweight='bold')
        plt.xlabel('Time of Day')
        plt.ylabel('Total Carbon (gCO2)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('fig4_cumulative_savings.png', dpi=300)
        
        # --- FIGURE 5: Model Profiler (Layer Latency) ---
        plt.figure(figsize=(10, 5))
        pc = self.layer_profile_df
        sns.barplot(x='layer_idx', y='compute_ms', data=pc, color='skyblue')
        plt.title('Layer-wise Compute Latency (1D-CNN)', fontweight='bold')
        plt.xlabel('Layer Index')
        plt.ylabel('Latency (ms)')
        plt.tight_layout()
        plt.savefig('fig5_layer_latency.png', dpi=300)
        
        # --- FIGURE 6: Latency Overhead Analysis ---
        plt.figure(figsize=(8, 6))
        # Compare Average Latency Baseline vs Ours
        avg_lat_base = np.mean([self.get_costs(-1, t)[1] for t in range(24)])
        avg_lat_ours = np.mean(latency_costs)
        
        plt.bar(['Baseline', 'Green-SFL'], [avg_lat_base, avg_lat_ours], color=['grey', '#FF5722'])
        plt.ylabel('Average Latency (ms)')
        plt.title('Trade-off: Latency Overhead for Sustainability', fontweight='bold')
        plt.text(1, avg_lat_ours, f"+{((avg_lat_ours/avg_lat_base)-1)*100:.1f}%", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('fig6_latency_overhead.png', dpi=300)
        
        print("\n[Artifacts] Generated 6 High-Quality Figures:")
        print(" - fig1_carbon_traces.png")
        print(" - fig2_banana_tradeoff.png (Banana Plot)")
        print(" - fig3_schedule_heatmap.png")
        print(" - fig4_cumulative_savings.png")
        print(" - fig5_layer_latency.png")
        print(" - fig6_latency_overhead.png")

if __name__ == "__main__":
    sim = CarbonSplitSimulation()
    sim.run()

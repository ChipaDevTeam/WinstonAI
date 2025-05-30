"""
GPU Performance Monitor for WinstonAI Training
Monitors GPU utilization, memory usage, and training performance in real-time
"""

import psutil
import time
import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from datetime import datetime
import pandas as pd

try:
    import GPUtil
    GPU_UTILS_AVAILABLE = True
except ImportError:
    print("GPUtil not found. Install with: pip install GPUtil")
    GPU_UTILS_AVAILABLE = False

try:
    import nvidia_ml_py3 as nvml
    nvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    print("nvidia-ml-py not found. Install with: pip install nvidia-ml-py3")
    NVML_AVAILABLE = False

class GPUMonitor:
    """Real-time GPU performance monitoring"""
    
    def __init__(self, log_file="gpu_performance.json"):
        self.log_file = log_file
        self.monitoring = False
        self.data = {
            'timestamps': [],
            'gpu_utilization': [],
            'gpu_memory_used': [],
            'gpu_memory_total': [],
            'gpu_temperature': [],
            'cpu_utilization': [],
            'ram_utilization': [],
            'training_episodes': [],
            'training_rewards': []
        }
        
        if NVML_AVAILABLE:
            self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(0)
    
    def get_gpu_stats(self):
        """Get current GPU statistics"""
        stats = {
            'utilization': 0,
            'memory_used': 0,
            'memory_total': 0,
            'temperature': 0
        }
        
        if GPU_UTILS_AVAILABLE:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                stats['utilization'] = gpu.load * 100
                stats['memory_used'] = gpu.memoryUsed
                stats['memory_total'] = gpu.memoryTotal
                stats['temperature'] = gpu.temperature
        
        elif NVML_AVAILABLE:
            try:
                # GPU utilization
                util = nvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                stats['utilization'] = util.gpu
                
                # Memory info
                mem_info = nvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                stats['memory_used'] = mem_info.used / 1024**2  # MB
                stats['memory_total'] = mem_info.total / 1024**2  # MB
                
                # Temperature
                temp = nvml.nvmlDeviceGetTemperature(self.gpu_handle, nvml.NVML_TEMPERATURE_GPU)
                stats['temperature'] = temp
                
            except Exception as e:
                print(f"Error getting GPU stats: {e}")
        
        return stats
    
    def get_system_stats(self):
        """Get system CPU and RAM statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'ram_percent': psutil.virtual_memory().percent
        }
    
    def start_monitoring(self, interval=1.0):
        """Start monitoring in a separate thread"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(f"🖥️ GPU monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop monitoring and save data"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
        self.save_data()
        print("[STATS] GPU monitoring stopped and data saved")
    
    def _monitor_loop(self, interval):
        """Main monitoring loop"""
        while self.monitoring:
            timestamp = datetime.now().isoformat()
            gpu_stats = self.get_gpu_stats()
            system_stats = self.get_system_stats()
            
            self.data['timestamps'].append(timestamp)
            self.data['gpu_utilization'].append(gpu_stats['utilization'])
            self.data['gpu_memory_used'].append(gpu_stats['memory_used'])
            self.data['gpu_memory_total'].append(gpu_stats['memory_total'])
            self.data['gpu_temperature'].append(gpu_stats['temperature'])
            self.data['cpu_utilization'].append(system_stats['cpu_percent'])
            self.data['ram_utilization'].append(system_stats['ram_percent'])
            
            time.sleep(interval)
    
    def update_training_metrics(self, episode, reward):
        """Update training metrics"""
        if self.monitoring:
            self.data['training_episodes'].append(episode)
            self.data['training_rewards'].append(reward)
    
    def save_data(self):
        """Save monitoring data to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def load_data(self):
        """Load monitoring data from file"""
        try:
            with open(self.log_file, 'r') as f:
                self.data = json.load(f)
            return True
        except FileNotFoundError:
            return False
    
    def plot_performance(self, save_plot=True):
        """Plot performance metrics"""
        if not self.data['timestamps']:
            print("No data to plot")
            return
        
        # Convert timestamps to datetime
        timestamps = pd.to_datetime(self.data['timestamps'])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # GPU Utilization
        axes[0, 0].plot(timestamps, self.data['gpu_utilization'], color='blue', linewidth=2)
        axes[0, 0].set_title('GPU Utilization (%)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Utilization (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 100)
        
        # GPU Memory Usage
        if self.data['gpu_memory_total']:
            memory_percent = [(used/total)*100 if total > 0 else 0 
                            for used, total in zip(self.data['gpu_memory_used'], self.data['gpu_memory_total'])]
            axes[0, 1].plot(timestamps, memory_percent, color='red', linewidth=2)
            axes[0, 1].set_title('GPU Memory Usage (%)', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('Memory Usage (%)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim(0, 100)
        
        # GPU Temperature
        axes[1, 0].plot(timestamps, self.data['gpu_temperature'], color='orange', linewidth=2)
        axes[1, 0].set_title('GPU Temperature (°C)', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('Temperature (°C)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # System Resources
        axes[1, 1].plot(timestamps, self.data['cpu_utilization'], label='CPU %', color='green', linewidth=2)
        axes[1, 1].plot(timestamps, self.data['ram_utilization'], label='RAM %', color='purple', linewidth=2)
        axes[1, 1].set_title('System Resources (%)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Utilization (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 100)
        
        # Format x-axis for all subplots
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('WinstonAI GPU Performance Monitor', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('gpu_performance_report.png', dpi=300, bbox_inches='tight')
            print("[STATS] Performance plot saved as: gpu_performance_report.png")
        
        plt.show()
    
    def print_current_stats(self):
        """Print current performance statistics"""
        gpu_stats = self.get_gpu_stats()
        system_stats = self.get_system_stats()
        
        print("\n" + "="*50)
        print("🖥️  CURRENT GPU PERFORMANCE")
        print("="*50)
        print(f"GPU Utilization:    {gpu_stats['utilization']:.1f}%")
        print(f"GPU Memory Used:    {gpu_stats['memory_used']:.0f} MB")
        print(f"GPU Memory Total:   {gpu_stats['memory_total']:.0f} MB")
        if gpu_stats['memory_total'] > 0:
            memory_percent = (gpu_stats['memory_used'] / gpu_stats['memory_total']) * 100
            print(f"GPU Memory Usage:   {memory_percent:.1f}%")
        print(f"GPU Temperature:    {gpu_stats['temperature']:.0f}°C")
        print(f"CPU Utilization:    {system_stats['cpu_percent']:.1f}%")
        print(f"RAM Utilization:    {system_stats['ram_percent']:.1f}%")
        print("="*50)
    
    def get_performance_summary(self):
        """Get performance summary statistics"""
        if not self.data['gpu_utilization']:
            return {}
        
        summary = {
            'avg_gpu_utilization': sum(self.data['gpu_utilization']) / len(self.data['gpu_utilization']),
            'max_gpu_utilization': max(self.data['gpu_utilization']),
            'avg_gpu_memory_percent': 0,
            'max_gpu_temperature': max(self.data['gpu_temperature']) if self.data['gpu_temperature'] else 0,
            'avg_cpu_utilization': sum(self.data['cpu_utilization']) / len(self.data['cpu_utilization']),
            'avg_ram_utilization': sum(self.data['ram_utilization']) / len(self.data['ram_utilization'])
        }
        
        if self.data['gpu_memory_total']:
            memory_percents = [(used/total)*100 if total > 0 else 0 
                             for used, total in zip(self.data['gpu_memory_used'], self.data['gpu_memory_total'])]
            summary['avg_gpu_memory_percent'] = sum(memory_percents) / len(memory_percents)
        
        return summary

def monitor_training_session():
    """Monitor a training session interactively"""
    monitor = GPUMonitor()
    
    print("[GPU] Starting GPU Performance Monitor")
    print("Commands: 'stats' - show current stats, 'plot' - show plot, 'quit' - exit")
    
    monitor.start_monitoring(interval=2.0)
    
    try:
        while True:
            command = input("\nEnter command (stats/plot/quit): ").strip().lower()
            
            if command == 'stats':
                monitor.print_current_stats()
            elif command == 'plot':
                monitor.plot_performance(save_plot=False)
            elif command == 'quit':
                break
            else:
                print("Unknown command. Use: stats, plot, or quit")
                
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    monitor.stop_monitoring()
    
    # Show final summary
    summary = monitor.get_performance_summary()
    if summary:
        print("\n[STATS] TRAINING SESSION SUMMARY")
        print("="*50)
        print(f"Avg GPU Utilization: {summary['avg_gpu_utilization']:.1f}%")
        print(f"Max GPU Utilization: {summary['max_gpu_utilization']:.1f}%")
        print(f"Avg GPU Memory:      {summary['avg_gpu_memory_percent']:.1f}%")
        print(f"Max GPU Temperature: {summary['max_gpu_temperature']:.0f}°C")
        print(f"Avg CPU Utilization: {summary['avg_cpu_utilization']:.1f}%")
        print(f"Avg RAM Utilization: {summary['avg_ram_utilization']:.1f}%")
        print("="*50)
    
    monitor.plot_performance()

if __name__ == "__main__":
    print("WinstonAI GPU Performance Monitor")
    print("Install dependencies with: pip install GPUtil nvidia-ml-py3")
    
    monitor_training_session()

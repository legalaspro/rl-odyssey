import torch
import multiprocessing as mp

class SharedRunningMeanStd:
    def __init__(self, shape=(), num_workers=1):
        # Shared statistics
        self.mean = torch.zeros(shape, dtype=torch.float32).share_memory_()
        self.var = torch.ones(shape, dtype=torch.float32).share_memory_()
        self.count = mp.Value('l', 0)  # Shared total count
        self.lock = mp.Lock()
        # Track each worker's last synced count
        self.last_worker_counts = {i: 0 for i in range(num_workers)}

    def sync_worker(self, worker_id, env_wrapper):
        """Update shared stats with new observations from a worker"""
        with self.lock:
            try:
                # Get worker's current stats (as numpy arrays)
                worker_count = int(env_wrapper.obs_rms.count)
                worker_mean = env_wrapper.obs_rms.mean.copy()
                worker_var = env_wrapper.obs_rms.var.copy()
                
                # Calculate observations since last sync
                last_count = self.last_worker_counts[worker_id]
                new_count = worker_count - last_count
                
                if new_count > 0:
                    # Convert everything to PyTorch tensors for consistent calculations
                    worker_mean_tensor = torch.from_numpy(worker_mean).float()
                    worker_var_tensor = torch.from_numpy(worker_var).float()
                    
                    total_count = self.count.value + new_count
                    
                    if self.count.value == 0:
                        # First update
                        self.mean.copy_(worker_mean_tensor)
                        self.var.copy_(worker_var_tensor)
                    else:
                        # Calculate weights for weighted average
                        old_weight = float(self.count.value) / total_count
                        new_weight = float(new_count) / total_count
                        
                        # Update mean
                        new_mean = self.mean * old_weight + worker_mean_tensor * new_weight
                        
                        # Update variance (keep everything as tensors)
                        new_var = (self.var * old_weight + worker_var_tensor * new_weight)
                        
                        # Copy to shared tensors
                        self.mean.copy_(new_mean)
                        self.var.copy_(new_var)
                    
                    # Update counts
                    self.count.value = int(total_count)
                    self.last_worker_counts[worker_id] = worker_count
                
                return True
                
            except Exception as e:
                print(f"Error in sync_worker: {e}")
                return False
    
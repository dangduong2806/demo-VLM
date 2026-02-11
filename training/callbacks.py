from transformers import TrainerCallback
import torch
import gc
from typing import Dict, Any

class MemoryOptimizationCallback(TrainerCallback):
    """Callback for VRAM management"""
    
    def __init__(self, clear_cache_steps: int = 25, log_memory_steps: int = 10):
        self.clear_cache_steps = clear_cache_steps
        self.log_memory_steps = log_memory_steps
        self.step_count = 0
    
    def on_step_begin(self, args, state, control, **kwargs):
        """Monitor inputs before step"""
        step = state.global_step
        
        if step % self.log_memory_steps == 0:
            if 'inputs' in kwargs and kwargs['inputs'] is not None:
                inputs = kwargs['inputs']
                
                if isinstance(inputs, dict) and 'pixel_values' in inputs:
                    pv = inputs['pixel_values']
                    pv_gb = pv.numel() * pv.element_size() / 1e9
                    
                    print(f"  [Step {step}] Input pixel_values: {pv.shape} ({pv_gb:.2f}GB)")
                    
                    if pv_gb > 1.5:
                        print(f"  ⚠️ Large pixel_values detected!")
        
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        """Clear cache and log memory after step"""
        self.step_count += 1
        step = state.global_step
        
        # Clear cache periodically
        if self.step_count % self.clear_cache_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()
            
            if self.step_count % self.log_memory_steps == 0:
                print(f"  [Step {step}] Cache cleared")
        
        # Log memory usage
        if self.step_count % self.log_memory_steps == 0:
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            reserved = torch.cuda.memory_reserved(0) / 1024**3
            
            print(f"  [Step {step}] VRAM: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
            
            # Warning if approaching limit
            if allocated > 13.0:
                print(f" VRAM high! Forcing cache clear...")
                torch.cuda.empty_cache()
                gc.collect()
        
        return control
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Aggressive cleanup at epoch end"""
        torch.cuda.empty_cache()
        gc.collect()
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"\n  [Epoch {int(state.epoch)}] Epoch ended")
        print(f"  VRAM after cleanup: {allocated:.2f}GB\n")
        
        return control
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Initial cleanup before training"""
        print("\n" + "="*80)
        print("MEMORY OPTIMIZATION CALLBACK ACTIVATED")
        print("="*80)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Total VRAM: {total:.2f}GB")
        print(f"  Used before training: {allocated:.2f}GB")
        print(f"  Free: {total - allocated:.2f}GB")
        print("="*80 + "\n")
        
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """Final cleanup"""
        print("\n" + "="*80)
        print("TRAINING ENDED - FINAL CLEANUP")
        print("="*80)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  VRAM after training: {allocated:.2f}GB")
        print("="*80 + "\n")
        
        return control


class ExperimentTrackingCallback(TrainerCallback):
    """Callback for MLflow/WandB tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.backend = config['tracking']['backend']
        self.run = None
        
        if self.backend == 'mlflow':
            import mlflow
            self.mlflow = mlflow
        elif self.backend == 'wandb':
            import wandb
            self.wandb = wandb
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Start tracking run"""
        
        if self.backend == 'mlflow':
            self.mlflow.start_run(run_name=self.config['experiment']['name'])
            self.mlflow.log_params(self.config)
            
        elif self.backend == 'wandb':
            self.run = self.wandb.init(
                project=self.config['tracking']['project_name'],
                name=self.config['experiment']['name'],
                config=self.config
            )
        
        print(f"✓ Experiment tracking started ({self.backend})")
        
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics"""
        
        if logs is None:
            return control
        
        if self.backend == 'mlflow':
            self.mlflow.log_metrics(logs, step=state.global_step)
            
        elif self.backend == 'wandb':
            self.wandb.log(logs, step=state.global_step)
        
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        """End tracking run"""
        
        if self.backend == 'mlflow':
            self.mlflow.end_run()
            
        elif self.backend == 'wandb':
            self.run.finish()
        
        print(f"✓ Experiment tracking ended")
        
        return control
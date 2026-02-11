from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch.nn as nn

class BaseVLM(ABC):
    """Abstract base class for all VLM models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.processor = None
        self.tokenizer = None
    
    @abstractmethod
    def load_model(self):
        """Load the model architecture"""
        pass
    
    @abstractmethod
    def load_processor(self):
        """Load the processor/tokenizer"""
        pass
    
    @abstractmethod
    def prepare_inputs(self, batch: Dict) -> Dict:
        """Prepare inputs for the model"""
        pass
    
    @abstractmethod
    def apply_lora(self, lora_config: Dict):
        """Apply LoRA adapters"""
        pass
    
    @abstractmethod
    def freeze_vision_encoder(self):
        """Freeze vision encoder layers"""
        pass
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Return trainable vs total parameters"""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {"trainable": trainable, "total": total, "percentage": 100 * trainable / total}
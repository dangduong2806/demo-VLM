from typing import Dict, Type
from .base_vlm import BaseVLM
from .llava_wrapper import LLaVAModel
from .qwen_wrapper import QwenVLModel  # Implement similarly

MODEL_REGISTRY: Dict[str, Type[BaseVLM]] = {
    "llava": LLaVAModel,
    "qwen": QwenVLModel,
    # Add more models here
}

def build_model(config: Dict) -> BaseVLM:
    """Factory function to build any VLM model"""
    
    architecture = config['model']['architecture']
    
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {architecture}. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_class = MODEL_REGISTRY[architecture]
    vlm = model_class(config)
    
    # Load components
    vlm.load_model()
    vlm.load_processor()
    
    if config['model']['lora']['enabled']:
        vlm.apply_lora(config['model']['lora'])
    
    if config['model']['vision']['freeze_encoder']:
        vlm.freeze_vision_encoder()
    
    return vlm
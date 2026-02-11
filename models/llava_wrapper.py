from transformers import (
    AutoConfig, AutoProcessor, AutoTokenizer, 
    AutoModelForImageTextToText, BitsAndBytesConfig
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
from src.models.base_vlm import BaseVLM

class LLaVAModel(BaseVLM):
    """LLaVA-OneVision model wrapper"""
    
    def load_model(self):
        # Quantization config
        if self.config['model']['quantization']['enabled']:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config['model']['quantization']['type'],
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=self.config['model']['quantization']['double_quant']
            )
        else:
            bnb_config = None
        
        # Model config
        config = AutoConfig.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True
        )
        config.use_cache = False
        
        # Load model
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config['model']['name'],
            quantization_config=bnb_config,
            config=config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.float16
        )
        
        # Prepare for training
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=self.config['training']['gradient_checkpointing']
        )
        
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        
        return self.model
    
    # Check VLM (cắt ảnh)
    def load_processor(self):
        vision_cfg = self.config['model']['vision']
        
        self.processor = AutoProcessor.from_pretrained(
            self.config['model']['name'],
            trust_remote_code=True,
            vision_feature_select_strategy="default",
            max_image_tiles=vision_cfg['max_tiles'],
            image_processor_kwargs={
                "size": {"height": vision_cfg['image_size'][0], "width": vision_cfg['image_size'][1]},
                "do_resize": True,
                "do_center_crop": False,
                "do_image_splitting": True
            },
            use_fast=True,
        )
        
        self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else \
                        AutoTokenizer.from_pretrained(self.config['model']['name'])
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        return self.processor, self.tokenizer
    
    def apply_lora(self, lora_config: dict):
        lora_cfg = LoraConfig(
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            target_modules=lora_config['target_modules'],
            lora_dropout=lora_config['dropout'],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_cfg)
        self.model.print_trainable_parameters()
        
        return self.model
    
    def freeze_vision_encoder(self):
        frozen_count = 0
        vision_keywords = ['vision_tower', 'vision_model', 'visual', 'vit']
        skip_keywords = ['merger', 'projector', 'mm_projector']
        
        for name, param in self.model.named_parameters():
            if any(kw in name.lower() for kw in vision_keywords):
                if not any(skip in name.lower() for skip in skip_keywords):
                    param.requires_grad = False
                    frozen_count += 1
        
        print(f"✓ Frozen {frozen_count} vision parameters")
    
    def prepare_inputs(self, batch: dict) -> dict:
        """LLaVA-specific input preparation"""
        return batch  # LLaVA uses standard format
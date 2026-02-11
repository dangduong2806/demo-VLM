import json
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class POLMData:
    """POLM structure from bbox annotations"""
    object_type: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    
    def to_text(self) -> str:
        """Convert POLM to text for prompt"""
        return (
            f"Object: {self.object_type}, "
            f"BBox: [{self.bbox[0]:.1f}, {self.bbox[1]:.1f}, {self.bbox[2]:.1f}, {self.bbox[3]:.1f}], "
            f"Confidence: {self.confidence:.2f}"
        )

@dataclass
class GroundTruthData:
    """Ground truth for training"""
    location: str
    weather: str
    traffic: str
    scene: str
    instruction: str
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({
            'location': self.location,
            'weather': self.weather,
            'traffic': self.traffic,
            'scene': self.scene,
            'instruction': self.instruction
        }, ensure_ascii=False)


def construct_prompt(polm_list: List[POLMData], num_images: int = 1) -> str:
    """Construct model input prompt"""
    
    polm_text = "\n".join([polm.to_text() for polm in polm_list])
    image_tokens = "<image>" * num_images
    
    prompt = f"""{image_tokens}
You are a navigation assistant for blind people.

Detected objects in the scene:
{polm_text}

Follow Chain-of-Thought reasoning:
1. Perception: What objects and environment do you see?
2. Context: Location type, weather, traffic level?
3. Decision: What guidance should be given?

Respond in JSON: {{"location": "...", "weather": "...", "traffic": "...", "scene": "...", "instruction": "..."}}"""
    
    return prompt


def map_metadata_to_ground_truth(metadata: Dict) -> GroundTruthData:
    """Map WAD metadata to ground truth format"""
    
    # Location mapping
    area_map = {
        'Pedestrian Path': 'pedestrian_path',
        'Road': 'road',
        'Corridor': 'corridor',
        'Busy Street': 'busy_street',
        'Shopping Mall': 'shopping_mall',
        'Bicycle Lane': 'bicycle_lane',
        'Restaurant': 'restaurant',
        'Other': 'other'
    }
    
    # Weather mapping
    weather_map = {
        'Sunny': 'sunny',
        'Overcast': 'overcast',
        'Cloudy': 'cloudy',
        'Night': 'night',
        'Indoor': 'indoor',
        'Other': 'other'
    }
    
    # Traffic mapping
    traffic_map = {
        'High': 'high',
        'Mid': 'moderate',
        'Low': 'low'
    }
    
    location = area_map.get(metadata.get('area_type', 'Other'), 'other')
    weather = weather_map.get(metadata.get('weather_condition', 'Other'), 'other')
    traffic = traffic_map.get(metadata.get('traffic_flow_rating', 'Low'), 'low')
    scene = metadata.get('summary', '')
    
    # Instruction: alter or QA answer
    if metadata.get('alter'):
        instruction = metadata['alter']
    elif metadata.get('QA') and isinstance(metadata['QA'], dict):
        instruction = metadata['QA'].get('A', '')
    else:
        instruction = ''
    
    return GroundTruthData(
        location=location,
        weather=weather,
        traffic=traffic,
        scene=scene,
        instruction=instruction
    )
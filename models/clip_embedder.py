from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from typing import Union
import io

class CLIPEmbedder:
    """
    Wrapper for CLIP model to generate image embeddings.
    Uses SentenceTransformer's CLIP implementation for consistency.
    """
    
    def __init__(self, model_name: str = 'clip-ViT-B-32'):
        """
        Initialize the CLIP model.
        
        Args:
            model_name: Name of the pre-trained CLIP model
        """
        print(f"Loading CLIP model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("CLIP model loaded successfully!")
        
    def embed_image(self, image: Union[Image.Image, bytes, str]) -> np.ndarray:
        """
        Generate embedding for an image.
        
        Args:
            image: PIL Image, image bytes, or path to image file
            
        Returns:
            numpy array of embeddings (512 dimensions for ViT-B/32)
        """
        # Convert bytes to PIL Image if needed
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, str):
            image = Image.open(image)
            
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Generate embedding
        embedding = self.model.encode(image, convert_to_numpy=True)
        
        return embedding
    
    def embed_images_batch(self, images: list) -> np.ndarray:
        """
        Generate embeddings for multiple images.
        
        Args:
            images: List of PIL Images, image bytes, or paths
            
        Returns:
            numpy array of embeddings (batch_size x 512)
        """
        # Process all images to PIL format
        pil_images = []
        for img in images:
            if isinstance(img, bytes):
                img = Image.open(io.BytesIO(img))
            elif isinstance(img, str):
                img = Image.open(img)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            pil_images.append(img)
        
        # Generate embeddings in batch
        embeddings = self.model.encode(pil_images, convert_to_numpy=True)
        
        return embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()


# Global instance to avoid reloading model on each request
_clip_embedder_instance = None

def get_clip_embedder() -> CLIPEmbedder:
    """
    Get or create the global CLIP embedder instance.
    This ensures the model is only loaded once.
    """
    global _clip_embedder_instance
    
    if _clip_embedder_instance is None:
        _clip_embedder_instance = CLIPEmbedder()
        
    return _clip_embedder_instance

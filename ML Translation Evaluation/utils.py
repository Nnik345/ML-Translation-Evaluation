"""
Utility functions for the MT evaluation pipeline.
"""
import os
import gc
import shutil
import torch
import warnings
from pathlib import Path
from config import SUPPRESS_WARNINGS


def setup_environment() -> None:
    """
    Set up the environment for evaluation.
    Configures warnings and checks GPU availability.
    """
    if SUPPRESS_WARNINGS:
        warnings.filterwarnings('ignore')
    
    print_gpu_info()


def print_gpu_info() -> None:
    """Print information about GPU availability."""
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
        print(f"Using device: cuda")
        print(f"GPU: {device}")
    else:
        print("Using device: cpu")
        print("Note: GPU not available, evaluation will run on CPU")


def cleanup_models(
    comet_model=None,
    comet_qe_model=None,
    comet_model_path: str = None,
    comet_qe_model_path: str = None,
    delete_nltk_data: bool = False,
    delete_model_cache: bool = False
) -> None:
    """
    Clean up model objects, model caches, and NLTK data to free memory.
    
    Args:
        comet_model: COMET model instance to delete
        comet_qe_model: COMET-QE model instance to delete
        comet_model_path: Path to COMET model cache directory
        comet_qe_model_path: Path to COMET-QE model cache directory
        delete_nltk_data: If True, delete NLTK data directory
        delete_model_cache: If True, delete COMET model cache directories
    """
    # Delete model objects from memory
    if comet_model is not None:
        del comet_model
        print("Deleted COMET model from memory")
    
    if comet_qe_model is not None:
        del comet_qe_model
        print("Deleted COMET-QE model from memory")
    
    # Run garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache")
    
    # Delete model cache directories
    if delete_model_cache:
        for model_path in [comet_model_path, comet_qe_model_path]:
            if model_path and os.path.exists(model_path):
                try:
                    shutil.rmtree(model_path, ignore_errors=True)
                    print(f"Deleted model cache: {model_path}")
                except Exception as e:
                    print(f"Warning: Could not delete {model_path}: {e}")
    
    # Delete NLTK data directory
    if delete_nltk_data:
        nltk_dir = Path.home() / "nltk_data"
        if nltk_dir.exists():
            try:
                shutil.rmtree(nltk_dir, ignore_errors=True)
                print(f"Deleted NLTK data: {nltk_dir}")
            except Exception as e:
                print(f"Warning: Could not delete NLTK data: {e}")
    
    print("Cleanup complete!")


def cleanup_all(
    comet_model=None,
    comet_qe_model=None,
    comet_model_path: str = None,
    comet_qe_model_path: str = None
) -> None:
    """
    Thorough cleanup: Delete models, model caches, AND NLTK data.
    Use this for complete cleanup after evaluation is done.
    
    Args:
        comet_model: COMET model instance
        comet_qe_model: COMET-QE model instance
        comet_model_path: Path to COMET model cache
        comet_qe_model_path: Path to COMET-QE model cache
    """
    cleanup_models(
        comet_model=comet_model,
        comet_qe_model=comet_qe_model,
        comet_model_path=comet_model_path,
        comet_qe_model_path=comet_qe_model_path,
        delete_nltk_data=True,
        delete_model_cache=True
    )

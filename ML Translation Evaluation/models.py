"""
Model initialization module.
Handles downloading and loading of COMET models and NLTK data.
"""
import os
import sys
import nltk
from comet import download_model, load_from_checkpoint
from typing import Tuple, Any
from config import COMET_MODEL, COMET_QE_MODEL, NLTK_PACKAGES

# Suppress PyTorch Lightning warnings and logging BEFORE importing models
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import warnings
warnings.filterwarnings('ignore')

# Redirect stderr to suppress Lightning output
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)


def download_nltk_data(quiet: bool = True) -> None:
    """
    Download required NLTK data packages.
    
    Args:
        quiet: If True, suppress download messages
    """
    for package in NLTK_PACKAGES:
        nltk.download(package, quiet=quiet)
    print("NLTK data downloaded successfully!")


def load_comet_models(verbose: bool = True) -> Tuple[Any, Any, str, str]:
    """
    Download and load COMET models.
    
    Args:
        verbose: If True, print progress messages
    
    Returns:
        Tuple of (comet_model, comet_qe_model, comet_model_path, comet_qe_model_path)
    """
    if verbose:
        print("Downloading COMET models (this may take a few minutes)...")
    
    # Suppress all output during model loading
    original_stderr = sys.stderr
    if not verbose:
        sys.stderr = open(os.devnull, 'w')
    
    try:
        # Reference-based COMET model
        comet_model_path = download_model(COMET_MODEL)
        comet_model = load_from_checkpoint(comet_model_path)
        
        # Quality Estimation COMET model (reference-free)
        comet_qe_model_path = download_model(COMET_QE_MODEL)
        comet_qe_model = load_from_checkpoint(comet_qe_model_path)
    finally:
        if not verbose:
            sys.stderr.close()
            sys.stderr = original_stderr
    
    if verbose:
        print("COMET models loaded successfully!")
    
    return comet_model, comet_qe_model, comet_model_path, comet_qe_model_path


def initialize_all_models(verbose: bool = True) -> Tuple[Any, Any, str, str]:
    """
    Initialize all required models and data.
    
    Args:
        verbose: If True, print progress messages
    
    Returns:
        Tuple of (comet_model, comet_qe_model, comet_model_path, comet_qe_model_path)
    """
    download_nltk_data(quiet=not verbose)
    return load_comet_models(verbose=verbose)
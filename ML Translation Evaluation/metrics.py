"""
Metrics module for MT evaluation.
Contains functions to compute BLEU, METEOR, COMET, and COMET-QE scores.
"""
import os
import sys
import numpy as np
import torch
from typing import List, Any
from sacrebleu.metrics import BLEU
from nltk.translate.meteor_score import meteor_score
from config import COMET_BATCH_SIZE, COMET_USE_GPU


def compute_bleu(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute corpus-level BLEU score.
    
    Args:
        references: List of reference translations
        hypotheses: List of hypothesis translations
    
    Returns:
        BLEU score (0-100 scale)
    """
    bleu = BLEU()
    # sacrebleu expects references as list of lists
    score = bleu.corpus_score(hypotheses, [references])
    return score.score


def compute_meteor(references: List[str], hypotheses: List[str]) -> float:
    """
    Compute corpus-level METEOR score.
    
    Args:
        references: List of reference translations
        hypotheses: List of hypothesis translations
    
    Returns:
        Average METEOR score (0-1 scale)
    """
    scores = []
    for ref, hyp in zip(references, hypotheses):
        # Tokenize for METEOR
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        score = meteor_score([ref_tokens], hyp_tokens)
        scores.append(score)
    
    return np.mean(scores)


def compute_comet(
    sources: List[str],
    references: List[str],
    hypotheses: List[str],
    model: Any,
    batch_size: int = COMET_BATCH_SIZE,
    use_gpu: bool = COMET_USE_GPU
) -> float:
    """
    Compute corpus-level COMET score.
    
    Args:
        sources: List of source texts
        references: List of reference translations
        hypotheses: List of hypothesis translations
        model: COMET model instance
        batch_size: Batch size for prediction
        use_gpu: Whether to use GPU if available
    
    Returns:
        COMET score
    """
    data = []
    for src, ref, hyp in zip(sources, references, hypotheses):
        data.append({
            "src": src,
            "mt": hyp,
            "ref": ref
        })
    
    # Use GPU if available and requested
    gpus = 1 if (use_gpu and torch.cuda.is_available()) else 0
    
    # Suppress all output during prediction
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        output = model.predict(data, batch_size=batch_size, gpus=gpus)
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr
    
    return output.system_score


def compute_comet_qe(
    sources: List[str],
    hypotheses: List[str],
    model: Any,
    batch_size: int = COMET_BATCH_SIZE,
    use_gpu: bool = COMET_USE_GPU
) -> float:
    """
    Compute corpus-level COMET-QE score (reference-free).
    
    Args:
        sources: List of source texts
        hypotheses: List of hypothesis translations
        model: COMET-QE model instance
        batch_size: Batch size for prediction
        use_gpu: Whether to use GPU if available
    
    Returns:
        COMET-QE score
    """
    data = []
    for src, hyp in zip(sources, hypotheses):
        data.append({
            "src": src,
            "mt": hyp
        })
    
    # Use GPU if available and requested
    gpus = 1 if (use_gpu and torch.cuda.is_available()) else 0
    
    # Suppress all output during prediction
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        output = model.predict(data, batch_size=batch_size, gpus=gpus)
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr
    
    return output.system_score


def compute_all_metrics(
    sources: List[str],
    references: List[str],
    hypotheses: List[str],
    comet_model: Any,
    comet_qe_model: Any
) -> dict:
    """
    Compute all evaluation metrics for a given dataset.
    
    Args:
        sources: List of source texts
        references: List of reference translations
        hypotheses: List of hypothesis translations
        comet_model: COMET model instance
        comet_qe_model: COMET-QE model instance
    
    Returns:
        Dictionary containing all metric scores
    """
    return {
        'BLEU': compute_bleu(references, hypotheses),
        'METEOR': compute_meteor(references, hypotheses),
        'COMET': compute_comet(sources, references, hypotheses, comet_model),
        'COMET_QE': compute_comet_qe(sources, hypotheses, comet_qe_model)
    }
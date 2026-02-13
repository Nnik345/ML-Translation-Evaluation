"""
Evaluator module.
Orchestrates the evaluation process for MT systems.
"""
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from data_loader import MTDataset
from metrics import compute_all_metrics
from config import OUTPUT_DECIMAL_PLACES


class MTEvaluator:
    """
    Class to evaluate MT systems and store results.
    """
    
    def __init__(self, comet_model: Any, comet_qe_model: Any):
        """
        Initialize evaluator with COMET models.
        
        Args:
            comet_model: COMET model instance
            comet_qe_model: COMET-QE model instance
        """
        self.comet_model = comet_model
        self.comet_qe_model = comet_qe_model
        self.results = []
    
    def evaluate_system(
        self,
        language: str,
        mt_system: str,
        sources: List[str],
        references: List[str],
        hypotheses: List[str],
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate a single MT system.
        
        Args:
            language: Language name
            mt_system: MT system name
            sources: List of source texts
            references: List of reference translations
            hypotheses: List of hypothesis translations
            verbose: If True, print progress messages
        
        Returns:
            Dictionary with evaluation results
        """
        if verbose:
            print(f"  Evaluating {mt_system}...")
        
        # Compute all metrics
        scores = compute_all_metrics(
            sources,
            references,
            hypotheses,
            self.comet_model,
            self.comet_qe_model
        )
        
        # Round scores according to configuration
        for metric, value in scores.items():
            decimals = OUTPUT_DECIMAL_PLACES.get(metric, 4)
            scores[metric] = round(value, decimals)
        
        # Create result entry
        result = {
            'Language': language,
            'MT_System': mt_system,
            **scores,
            'Num_Samples': len(hypotheses)
        }
        
        if verbose:
            print(f"    ✓ BLEU: {scores['BLEU']:.2f}, "
                  f"METEOR: {scores['METEOR']:.4f}, "
                  f"COMET: {scores['COMET']:.4f}, "
                  f"COMET-QE: {scores['COMET_QE']:.4f}")
        
        return result
    
    def evaluate_dataset(
        self,
        language: str,
        dataset: MTDataset,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Evaluate all MT systems in a dataset.
        
        Args:
            language: Language name
            dataset: MTDataset instance
            verbose: If True, print progress messages
        
        Returns:
            List of result dictionaries
        """
        if verbose:
            print(f"Processing {language}...")
        
        mt_systems = dataset.get_mt_systems()
        
        if verbose:
            print(f"  Found {len(mt_systems)} MT system(s): {', '.join(mt_systems)}")
        
        dataset_results = []
        
        for mt_system in mt_systems:
            try:
                sources, references, hypotheses = dataset.prepare_data(mt_system)
                
                result = self.evaluate_system(
                    language,
                    mt_system,
                    sources,
                    references,
                    hypotheses,
                    verbose=verbose
                )
                
                dataset_results.append(result)
                self.results.append(result)
                
            except ValueError as e:
                if verbose:
                    print(f"    Warning: {e}")
                continue
            except Exception as e:
                if verbose:
                    print(f"    Error evaluating {mt_system}: {e}")
                continue
        
        if verbose:
            print(f"  Completed {language}\n")
        
        return dataset_results
    
    def evaluate_multiple_datasets(
        self,
        datasets: Dict[str, MTDataset],
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate multiple datasets.
        
        Args:
            datasets: Dictionary mapping language names to MTDataset objects
            verbose: If True, print progress messages
        
        Returns:
            DataFrame with all results
        """
        if verbose:
            print("Starting evaluation...\n")
        
        for language, dataset in datasets.items():
            self.evaluate_dataset(language, dataset, verbose=verbose)
        
        if verbose:
            print("Evaluation complete!")
        
        return self.get_results_dataframe()
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame.
        
        Returns:
            DataFrame with all evaluation results
        """
        return pd.DataFrame(self.results)
    
    def save_results(
        self,
        output_path: Path,
        verbose: bool = True
    ) -> None:
        """
        Save results to CSV file.
        
        Args:
            output_path: Path to output CSV file
            verbose: If True, print confirmation message
        """
        results_df = self.get_results_dataframe()
        
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(output_path, index=False)
        
        if verbose:
            print(f"\nResults saved to: {output_path}")
            print(f"Total evaluations: {len(results_df)}")
    
    def print_results(self) -> None:
        """Print formatted results table."""
        results_df = self.get_results_dataframe()
        
        print("\n" + "=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80 + "\n")
        print(results_df.to_string(index=False))
        print("\n" + "=" * 80)

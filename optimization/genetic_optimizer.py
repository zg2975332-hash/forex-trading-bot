"""
Advanced Genetic Optimizer for FOREX TRADING BOT
Evolutionary algorithm for optimizing trading strategies and parameters
"""

import logging
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import time
import json
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import pickle
import warnings

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    MAXIMIZE_PROFIT = "maximize_profit"
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_CALMAR = "maximize_calmar"
    MAXIMIZE_WIN_RATE = "maximize_win_rate"
    MINIMIZE_VOLATILITY = "minimize_volatility"

class GeneType(Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    INTEGER = "integer"
    CATEGORICAL = "categorical"

class SelectionMethod(Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"
    ELITISM = "elitism"

@dataclass
class Gene:
    """Genetic algorithm gene definition"""
    name: str
    gene_type: GeneType
    min_value: float
    max_value: float
    step: float = None
    categories: List[Any] = field(default_factory=list)
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1

@dataclass
class Chromosome:
    """Individual solution in genetic algorithm"""
    genes: Dict[str, float]
    fitness: float = 0.0
    fitness_components: Dict[str, float] = field(default_factory=dict)
    age: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GenerationStats:
    """Statistics for a generation"""
    generation: int
    best_fitness: float
    average_fitness: float
    worst_fitness: float
    diversity: float
    convergence: float
    timestamp: datetime

@dataclass
class GeneticOptimizerConfig:
    """Configuration for genetic optimizer"""
    # Population settings
    population_size: int = 100
    max_generations: int = 200
    elite_count: int = 5
    mutation_rate: float = 0.15
    crossover_rate: float = 0.85
    
    # Selection methods
    selection_method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 3
    
    # Optimization objectives
    objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'sharpe_ratio': 0.4,
        'max_drawdown': -0.3,
        'profit_factor': 0.2,
        'win_rate': 0.1
    })
    
    # Convergence criteria
    early_stopping_patience: int = 20
    fitness_tolerance: float = 1e-6
    min_diversity: float = 0.01
    
    # Parallel processing
    enable_parallel: bool = True
    max_workers: int = None
    chunk_size: int = 10
    
    # Exploration vs Exploitation
    adaptive_mutation: bool = True
    diversity_preservation: bool = True
    age_penalty: float = 0.01
    
    # Logging and output
    save_checkpoints: bool = True
    checkpoint_interval: int = 10
    verbose: bool = True

class AdvancedGeneticOptimizer:
    """
    Advanced genetic algorithm optimizer for trading strategy parameters
    """
    
    def __init__(self, genes: List[Gene], fitness_function: Callable, config: GeneticOptimizerConfig = None):
        self.genes = {gene.name: gene for gene in genes}
        self.fitness_function = fitness_function
        self.config = config or GeneticOptimizerConfig()
        
        # Population and evolution state
        self.population: List[Chromosome] = []
        self.generation = 0
        self.best_chromosome: Optional[Chromosome] = None
        self.generation_history: List[GenerationStats] = []
        
        # Multi-processing
        self.max_workers = self.config.max_workers or multiprocessing.cpu_count()
        
        # Adaptive parameters
        self.mutation_rate_history = deque(maxlen=50)
        self.diversity_history = deque(maxlen=50)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialize population
        self._initialize_population()
        
        logger.info("AdvancedGeneticOptimizer initialized successfully")

    def _initialize_population(self):
        """Initialize the population with random chromosomes"""
        self.population = []
        
        for i in range(self.config.population_size):
            chromosome = self._create_random_chromosome()
            self.population.append(chromosome)
        
        logger.info(f"Initialized population with {len(self.population)} individuals")

    def _create_random_chromosome(self) -> Chromosome:
        """Create a random chromosome"""
        genes = {}
        
        for gene_name, gene in self.genes.items():
            if gene.gene_type == GeneType.CONTINUOUS:
                value = random.uniform(gene.min_value, gene.max_value)
            elif gene.gene_type == GeneType.DISCRETE:
                steps = int((gene.max_value - gene.min_value) / gene.step) + 1
                value = gene.min_value + random.randint(0, steps - 1) * gene.step
            elif gene.gene_type == GeneType.INTEGER:
                value = random.randint(int(gene.min_value), int(gene.max_value))
            elif gene.gene_type == GeneType.CATEGORICAL:
                value = random.choice(gene.categories)
            else:
                value = random.uniform(gene.min_value, gene.max_value)
            
            genes[gene_name] = value
        
        return Chromosome(genes=genes)

    def optimize(self, callback: Callable = None) -> Chromosome:
        """
        Run the genetic optimization process
        """
        logger.info("Starting genetic optimization...")
        start_time = time.time()
        
        try:
            for generation in range(self.config.max_generations):
                self.generation = generation
                
                # Evaluate fitness
                self._evaluate_population()
                
                # Calculate generation statistics
                stats = self._calculate_generation_stats()
                self.generation_history.append(stats)
                
                # Update best chromosome
                current_best = max(self.population, key=lambda x: x.fitness)
                if self.best_chromosome is None or current_best.fitness > self.best_chromosome.fitness:
                    self.best_chromosome = current_best
                
                # Log progress
                if self.config.verbose and generation % 10 == 0:
                    self._log_generation_stats(stats)
                
                # Callback for external monitoring
                if callback:
                    callback(self, generation, stats)
                
                # Check convergence
                if self._check_convergence():
                    logger.info(f"Optimization converged at generation {generation}")
                    break
                
                # Create new generation
                self._evolve_population()
                
                # Save checkpoint
                if (self.config.save_checkpoints and 
                    generation % self.config.checkpoint_interval == 0):
                    self._save_checkpoint(generation)
            
            # Final evaluation
            self._evaluate_population()
            final_stats = self._calculate_generation_stats()
            self.generation_history.append(final_stats)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Optimization completed in {elapsed_time:.2f} seconds")
            logger.info(f"Best fitness: {self.best_chromosome.fitness:.6f}")
            
            return self.best_chromosome
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise

    def _evaluate_population(self):
        """Evaluate fitness of all chromosomes in population"""
        try:
            if self.config.enable_parallel and len(self.population) > 1:
                self._evaluate_population_parallel()
            else:
                self._evaluate_population_serial()
                
        except Exception as e:
            logger.error(f"Population evaluation failed: {e}")
            # Fallback to serial evaluation
            self._evaluate_population_serial()

    def _evaluate_population_serial(self):
        """Evaluate population serially"""
        for chromosome in self.population:
            if chromosome.fitness == 0:  # Only evaluate if not already evaluated
                try:
                    fitness_result = self.fitness_function(chromosome.genes)
                    if isinstance(fitness_result, dict):
                        chromosome.fitness = self._calculate_composite_fitness(fitness_result)
                        chromosome.fitness_components = fitness_result
                    else:
                        chromosome.fitness = float(fitness_result)
                except Exception as e:
                    logger.warning(f"Fitness evaluation failed for chromosome: {e}")
                    chromosome.fitness = -float('inf')

    def _evaluate_population_parallel(self):
        """Evaluate population using parallel processing"""
        try:
            # Prepare chromosomes for evaluation
            chromosomes_to_evaluate = [
                chrom for chrom in self.population 
                if chrom.fitness == 0
            ]
            
            if not chromosomes_to_evaluate:
                return
            
            # Use ProcessPoolExecutor for CPU-bound fitness evaluation
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit evaluation tasks
                future_to_chromosome = {
                    executor.submit(self.fitness_function, chrom.genes): chrom 
                    for chrom in chromosomes_to_evaluate
                }
                
                # Collect results
                for future in future_to_chromosome:
                    chromosome = future_to_chromosome[future]
                    try:
                        fitness_result = future.result(timeout=300)  # 5-minute timeout
                        if isinstance(fitness_result, dict):
                            chromosome.fitness = self._calculate_composite_fitness(fitness_result)
                            chromosome.fitness_components = fitness_result
                        else:
                            chromosome.fitness = float(fitness_result)
                    except Exception as e:
                        logger.warning(f"Parallel fitness evaluation failed: {e}")
                        chromosome.fitness = -float('inf')
                        
        except Exception as e:
            logger.error(f"Parallel evaluation failed: {e}")
            self._evaluate_population_serial()

    def _calculate_composite_fitness(self, fitness_components: Dict[str, float]) -> float:
        """Calculate composite fitness from multiple components"""
        composite_fitness = 0.0
        
        for component, value in fitness_components.items():
            weight = self.config.objective_weights.get(component, 0.0)
            
            # Handle different optimization objectives
            if component == 'max_drawdown' and value < 0:
                # For drawdown, we want to minimize (less negative is better)
                composite_fitness += weight * (1.0 + value)  # Convert to positive scaling
            else:
                composite_fitness += weight * value
        
        return composite_fitness

    def _calculate_generation_stats(self) -> GenerationStats:
        """Calculate statistics for current generation"""
        fitness_values = [chrom.fitness for chrom in self.population]
        
        best_fitness = max(fitness_values)
        average_fitness = np.mean(fitness_values)
        worst_fitness = min(fitness_values)
        
        # Calculate diversity (standard deviation of fitness)
        diversity = np.std(fitness_values) if len(fitness_values) > 1 else 0.0
        
        # Calculate convergence (how close to best)
        convergence = 1.0 - (diversity / (abs(best_fitness) + 1e-10)) if best_fitness != 0 else 0.0
        
        stats = GenerationStats(
            generation=self.generation,
            best_fitness=best_fitness,
            average_fitness=average_fitness,
            worst_fitness=worst_fitness,
            diversity=diversity,
            convergence=convergence,
            timestamp=datetime.now()
        )
        
        return stats

    def _log_generation_stats(self, stats: GenerationStats):
        """Log generation statistics"""
        logger.info(
            f"Generation {stats.generation:3d} | "
            f"Best: {stats.best_fitness:8.4f} | "
            f"Avg: {stats.average_fitness:8.4f} | "
            f"Worst: {stats.worst_fitness:8.4f} | "
            f"Div: {stats.diversity:6.4f} | "
            f"Conv: {stats.convergence:6.4f}"
        )

    def _check_convergence(self) -> bool:
        """Check if optimization has converged"""
        if len(self.generation_history) < self.config.early_stopping_patience:
            return False
        
        # Check fitness improvement
        recent_best = [stats.best_fitness for stats in self.generation_history[-self.config.early_stopping_patience:]]
        if max(recent_best) - min(recent_best) < self.config.fitness_tolerance:
            return True
        
        # Check diversity
        current_stats = self.generation_history[-1]
        if current_stats.diversity < self.config.min_diversity:
            return True
        
        return False

    def _evolve_population(self):
        """Create new generation through selection, crossover, and mutation"""
        new_population = []
        
        # Elitism: keep best individuals
        elites = self._select_elites()
        new_population.extend(elites)
        
        # Create rest of population through reproduction
        while len(new_population) < self.config.population_size:
            # Selection
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2
            
            # Mutation
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < self.config.population_size:
                new_population.append(child2)
        
        # Update population
        self.population = new_population[:self.config.population_size]
        
        # Update ages
        for chromosome in self.population:
            chromosome.age += 1

    def _select_elites(self) -> List[Chromosome]:
        """Select elite individuals to preserve"""
        sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return [self._copy_chromosome(chrom) for chrom in sorted_population[:self.config.elite_count]]

    def _select_parent(self) -> Chromosome:
        """Select parent using configured selection method"""
        if self.config.selection_method == SelectionMethod.TOURNAMENT:
            return self._tournament_selection()
        elif self.config.selection_method == SelectionMethod.ROULETTE:
            return self._roulette_selection()
        elif self.config.selection_method == SelectionMethod.RANK:
            return self._rank_selection()
        else:
            return self._tournament_selection()

    def _tournament_selection(self) -> Chromosome:
        """Tournament selection"""
        tournament = random.sample(self.population, self.config.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def _roulette_selection(self) -> Chromosome:
        """Roulette wheel selection"""
        # Convert fitness to positive values for selection
        min_fitness = min(chrom.fitness for chrom in self.population)
        adjusted_fitness = [chrom.fitness - min_fitness + 1e-10 for chrom in self.population]
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness == 0:
            return random.choice(self.population)
        
        # Select based on fitness proportion
        pick = random.uniform(0, total_fitness)
        current = 0
        for chrom in self.population:
            current += (chrom.fitness - min_fitness + 1e-10)
            if current > pick:
                return chrom
        
        return self.population[-1]  # Fallback

    def _rank_selection(self) -> Chromosome:
        """Rank-based selection"""
        sorted_population = sorted(self.population, key=lambda x: x.fitness)
        ranks = list(range(1, len(self.population) + 1))
        total_rank = sum(ranks)
        
        pick = random.uniform(0, total_rank)
        current = 0
        for chrom, rank in zip(sorted_population, ranks):
            current += rank
            if current > pick:
                return chrom
        
        return sorted_population[-1]  # Fallback

    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Perform crossover between two parents"""
        child1_genes = {}
        child2_genes = {}
        
        for gene_name in self.genes.keys():
            gene = self.genes[gene_name]
            
            if gene.gene_type == GeneType.CONTINUOUS:
                # Arithmetic crossover for continuous values
                alpha = random.random()
                child1_value = alpha * parent1.genes[gene_name] + (1 - alpha) * parent2.genes[gene_name]
                child2_value = alpha * parent2.genes[gene_name] + (1 - alpha) * parent1.genes[gene_name]
                
                # Ensure within bounds
                child1_value = max(gene.min_value, min(gene.max_value, child1_value))
                child2_value = max(gene.min_value, min(gene.max_value, child2_value))
                
            elif gene.gene_type in [GeneType.DISCRETE, GeneType.INTEGER]:
                # Single-point crossover
                if random.random() < 0.5:
                    child1_value = parent1.genes[gene_name]
                    child2_value = parent2.genes[gene_name]
                else:
                    child1_value = parent2.genes[gene_name]
                    child2_value = parent1.genes[gene_name]
                    
            elif gene.gene_type == GeneType.CATEGORICAL:
                # Uniform crossover for categorical
                if random.random() < 0.5:
                    child1_value = parent1.genes[gene_name]
                    child2_value = parent2.genes[gene_name]
                else:
                    child1_value = parent2.genes[gene_name]
                    child2_value = parent1.genes[gene_name]
            
            else:
                # Default to parent values
                child1_value = parent1.genes[gene_name]
                child2_value = parent2.genes[gene_name]
            
            child1_genes[gene_name] = child1_value
            child2_genes[gene_name] = child2_value
        
        child1 = Chromosome(genes=child1_genes)
        child2 = Chromosome(genes=child2_genes)
        
        return child1, child2

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        """Apply mutation to chromosome"""
        mutated_genes = chromosome.genes.copy()
        
        for gene_name, gene in self.genes.items():
            if random.random() < gene.mutation_rate:
                if gene.gene_type == GeneType.CONTINUOUS:
                    # Gaussian mutation
                    mutation = random.gauss(0, gene.mutation_strength * (gene.max_value - gene.min_value))
                    mutated_value = mutated_genes[gene_name] + mutation
                    # Ensure within bounds
                    mutated_value = max(gene.min_value, min(gene.max_value, mutated_value))
                    
                elif gene.gene_type == GeneType.DISCRETE:
                    # Random step mutation
                    steps = int((gene.max_value - gene.min_value) / gene.step)
                    current_step = int((mutated_genes[gene_name] - gene.min_value) / gene.step)
                    new_step = current_step + random.randint(-2, 2)
                    new_step = max(0, min(steps, new_step))
                    mutated_value = gene.min_value + new_step * gene.step
                    
                elif gene.gene_type == GeneType.INTEGER:
                    # Random integer mutation
                    mutation = random.randint(-2, 2)
                    mutated_value = mutated_genes[gene_name] + mutation
                    mutated_value = int(max(gene.min_value, min(gene.max_value, mutated_value)))
                    
                elif gene.gene_type == GeneType.CATEGORICAL:
                    # Random category mutation
                    mutated_value = random.choice(gene.categories)
                
                else:
                    mutated_value = mutated_genes[gene_name]
                
                mutated_genes[gene_name] = mutated_value
        
        # Create new chromosome with mutated genes
        mutated_chromosome = Chromosome(genes=mutated_genes)
        
        # Apply age penalty for diversity preservation
        if self.config.diversity_preservation:
            age_penalty = chromosome.age * self.config.age_penalty
            mutated_chromosome.fitness = max(-1e10, chromosome.fitness - age_penalty)
        else:
            mutated_chromosome.fitness = 0  # Need re-evaluation
        
        return mutated_chromosome

    def _copy_chromosome(self, chromosome: Chromosome) -> Chromosome:
        """Create a copy of chromosome"""
        return Chromosome(
            genes=chromosome.genes.copy(),
            fitness=chromosome.fitness,
            fitness_components=chromosome.fitness_components.copy(),
            age=chromosome.age,
            metadata=chromosome.metadata.copy()
        )

    def _save_checkpoint(self, generation: int):
        """Save optimization checkpoint"""
        try:
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)
            
            checkpoint_data = {
                'generation': generation,
                'population': self.population,
                'best_chromosome': self.best_chromosome,
                'generation_history': self.generation_history,
                'config': self.config,
                'timestamp': datetime.now()
            }
            
            checkpoint_file = checkpoint_dir / f"genetic_optimizer_gen_{generation}.pkl"
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.debug(f"Checkpoint saved for generation {generation}")
            
        except Exception as e:
            logger.error(f"Checkpoint saving failed: {e}")

    def load_checkpoint(self, checkpoint_file: str) -> bool:
        """Load optimization from checkpoint"""
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            self.generation = checkpoint_data['generation']
            self.population = checkpoint_data['population']
            self.best_chromosome = checkpoint_data['best_chromosome']
            self.generation_history = checkpoint_data['generation_history']
            
            logger.info(f"Checkpoint loaded from generation {self.generation}")
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint loading failed: {e}")
            return False

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if not self.generation_history:
            return {}
        
        latest_stats = self.generation_history[-1]
        first_stats = self.generation_history[0]
        
        improvement = latest_stats.best_fitness - first_stats.best_fitness
        improvement_pct = (improvement / abs(first_stats.best_fitness)) * 100 if first_stats.best_fitness != 0 else 0
        
        summary = {
            'total_generations': len(self.generation_history),
            'best_fitness': latest_stats.best_fitness,
            'best_chromosome': self.best_chromosome.genes if self.best_chromosome else {},
            'fitness_improvement': improvement,
            'fitness_improvement_pct': improvement_pct,
            'final_diversity': latest_stats.diversity,
            'convergence_status': self._check_convergence(),
            'optimization_time': (latest_stats.timestamp - first_stats.timestamp).total_seconds(),
            'fitness_components': self.best_chromosome.fitness_components if self.best_chromosome else {}
        }
        
        return summary

    def plot_optimization_history(self, save_path: str = None):
        """Plot optimization history (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            generations = [stats.generation for stats in self.generation_history]
            best_fitness = [stats.best_fitness for stats in self.generation_history]
            avg_fitness = [stats.average_fitness for stats in self.generation_history]
            diversity = [stats.diversity for stats in self.generation_history]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Fitness plot
            ax1.plot(generations, best_fitness, 'b-', label='Best Fitness', linewidth=2)
            ax1.plot(generations, avg_fitness, 'r--', label='Average Fitness', linewidth=1)
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Genetic Optimization Progress')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Diversity plot
            ax2.plot(generations, diversity, 'g-', label='Population Diversity', linewidth=2)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Diversity (Std Dev)')
            ax2.set_title('Population Diversity Over Generations')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization plot saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Plotting failed: {e}")

# Example fitness function for trading strategy optimization
def example_trading_fitness(parameters: Dict[str, float]) -> Dict[str, float]:
    """
    Example fitness function for trading strategy optimization
    In real implementation, this would run backtesting
    """
    # Extract parameters
    rsi_period = int(parameters.get('rsi_period', 14))
    rsi_oversold = parameters.get('rsi_oversold', 30)
    rsi_overbought = parameters.get('rsi_overbought', 70)
    stop_loss = parameters.get('stop_loss', 0.02)
    take_profit = parameters.get('take_profit', 0.04)
    
    # Simulate trading performance (replace with actual backtesting)
    # This is a simplified example
    sharpe_ratio = random.uniform(0.5, 2.0)  # Simulated performance
    max_drawdown = -random.uniform(0.05, 0.20)  # Negative for minimization
    profit_factor = random.uniform(1.0, 3.0)
    win_rate = random.uniform(0.4, 0.7)
    
    # Penalize extreme parameters
    penalty = 0.0
    if rsi_period < 5 or rsi_period > 50:
        penalty += 0.5
    if stop_loss < 0.005 or stop_loss > 0.1:
        penalty += 0.5
    
    # Return fitness components
    return {
        'sharpe_ratio': max(0, sharpe_ratio - penalty),
        'max_drawdown': max_drawdown,
        'profit_factor': max(1.0, profit_factor - penalty),
        'win_rate': max(0.3, win_rate - penalty * 0.1)
    }

# Example usage and testing
def main():
    """Example usage of the AdvancedGeneticOptimizer"""
    
    # Define genes for optimization
    genes = [
        Gene('rsi_period', GeneType.INTEGER, 5, 50, mutation_rate=0.1),
        Gene('rsi_oversold', GeneType.INTEGER, 20, 40, mutation_rate=0.08),
        Gene('rsi_overbought', GeneType.INTEGER, 60, 80, mutation_rate=0.08),
        Gene('stop_loss', GeneType.CONTINUOUS, 0.005, 0.05, mutation_rate=0.15, mutation_strength=0.1),
        Gene('take_profit', GeneType.CONTINUOUS, 0.01, 0.1, mutation_rate=0.15, mutation_strength=0.1),
        Gene('trailing_stop', GeneType.CONTINUOUS, 0.005, 0.03, mutation_rate=0.1, mutation_strength=0.05),
        Gene('position_size', GeneType.CONTINUOUS, 0.01, 0.1, mutation_rate=0.12, mutation_strength=0.08)
    ]
    
    # Configuration
    config = GeneticOptimizerConfig(
        population_size=50,
        max_generations=100,
        elite_count=3,
        mutation_rate=0.15,
        crossover_rate=0.8,
        selection_method=SelectionMethod.TOURNAMENT,
        tournament_size=3,
        enable_parallel=True,
        verbose=True,
        early_stopping_patience=15
    )
    
    # Initialize optimizer
    optimizer = AdvancedGeneticOptimizer(genes, example_trading_fitness, config)
    
    # Run optimization
    print("=== Starting Genetic Optimization ===")
    best_solution = optimizer.optimize()
    
    # Display results
    print("\n=== Optimization Results ===")
    print(f"Best Fitness: {best_solution.fitness:.6f}")
    print("Best Parameters:")
    for param, value in best_solution.genes.items():
        print(f"  {param}: {value:.6f}")
    
    print("\nFitness Components:")
    for component, value in best_solution.fitness_components.items():
        print(f"  {component}: {value:.6f}")
    
    # Get optimization summary
    summary = optimizer.get_optimization_summary()
    print(f"\nOptimization Summary:")
    print(f"Generations: {summary['total_generations']}")
    print(f"Fitness Improvement: {summary['fitness_improvement']:.6f} ({summary['fitness_improvement_pct']:.2f}%)")
    print(f"Final Diversity: {summary['final_diversity']:.6f}")
    print(f"Converged: {summary['convergence_status']}")
    
    # Plot optimization history
    try:
        optimizer.plot_optimization_history("optimization_history.png")
    except Exception as e:
        print(f"Plotting not available: {e}")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
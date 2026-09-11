"""遗传算法权重优化."""
from __future__ import annotations

import numpy as np
from typing import Any, Callable

from cabin_eval.optimization.constraints import ConstraintHandler
from cabin_eval.optimization.objective import ObjectiveFunction


class GeneticOptimizer:
    """遗传算法优化器."""

    def __init__(self, config: dict[str, Any]):
        self._population_size = config.get("population_size", 50)
        self._generations = config.get("generations", 100)
        self._mutation_rate = config.get("mutation_rate", 0.10)
        self._random_seed = config.get("random_seed", 42)
        self._epsilon = config.get("epsilon", 0.10)
        self._lambda_l2 = config.get("lambda_l2", 0.01)
        self._lambda_rank = config.get("lambda_rank", 0.01)

        self._rng = np.random.RandomState(self._random_seed)

    def optimize(
        self,
        base_weights: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        sibling_constraints: list[list[int]],
    ) -> dict[str, Any]:
        """执行遗传算法优化."""
        n_weights = len(base_weights)
        constraint_handler = ConstraintHandler(
            sibling_constraints, base_weights, self._epsilon
        )

        population = self._initialize_population(base_weights, n_weights)

        best_fitness = float("inf")
        best_individual = None
        fitness_history = []

        for generation in range(self._generations):
            fitnesses = []
            for individual in population:
                adjusted = constraint_handler.project(individual)
                fitness = self._calculate_fitness(
                    adjusted, base_weights, X_train, y_train
                )
                fitnesses.append(fitness)

                if fitness < best_fitness:
                    best_fitness = fitness
                    best_individual = adjusted.copy()

            population = self._select(population, fitnesses)
            population = self._crossover(population, constraint_handler)
            population = self._mutate(population, constraint_handler)

            val_fitness = None
            if best_individual is not None:
                val_fitness = self._calculate_fitness(
                    constraint_handler.project(best_individual),
                    base_weights,
                    X_val,
                    y_val,
                )

            fitness_history.append({
                "generation": generation,
                "train_mae": best_fitness,
                "val_mae": val_fitness,
            })

        final_weights = constraint_handler.project(best_individual)

        return {
            "optimized_weights": final_weights,
            "best_train_mae": best_fitness,
            "best_val_mae": fitness_history[-1]["val_mae"] if fitness_history else None,
            "fitness_history": fitness_history,
        }

    def _initialize_population(
        self, base_weights: np.ndarray, n_weights: int
    ) -> list[np.ndarray]:
        """初始化种群."""
        population = []
        for _ in range(self._population_size):
            individual = base_weights + self._rng.uniform(
                -self._epsilon, self._epsilon, n_weights
            )
            population.append(individual)
        return population

    def _calculate_fitness(
        self,
        weights: np.ndarray,
        base_weights: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """计算适应度 (MAE + L2正则化)."""
        predictions = X @ weights
        mae = np.mean(np.abs(predictions - y))
        l2_penalty = self._lambda_l2 * np.sum((weights - base_weights) ** 2)
        return mae + l2_penalty

    def _select(
        self, population: list[np.ndarray], fitnesses: list[float]
    ) -> list[np.ndarray]:
        """选择操作 - 锦标赛选择."""
        selected = []
        for _ in range(len(population)):
            tournament_size = 3
            indices = self._rng.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitnesses[i] for i in indices]
            winner_idx = indices[np.argmin(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        return selected

    def _crossover(
        self, population: list[np.ndarray], constraint_handler: ConstraintHandler
    ) -> list[np.ndarray]:
        """交叉操作."""
        new_population = []
        for i in range(0, len(population), 2):
            if i + 1 >= len(population):
                new_population.append(population[i].copy())
                continue

            parent1 = population[i]
            parent2 = population[i + 1]

            crossover_point = self._rng.randint(1, len(parent1))
            child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])

            child1 = constraint_handler.project(child1)
            child2 = constraint_handler.project(child2)

            new_population.append(child1)
            new_population.append(child2)

        return new_population

    def _mutate(
        self, population: list[np.ndarray], constraint_handler: ConstraintHandler
    ) -> list[np.ndarray]:
        """变异操作."""
        for i in range(len(population)):
            if self._rng.random() < self._mutation_rate:
                mutation_idx = self._rng.randint(len(population[i]))
                population[i][mutation_idx] += self._rng.uniform(-0.05, 0.05)
                population[i] = constraint_handler.project(population[i])
        return population

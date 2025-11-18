"""
Utility functions for HMM operations.

This module provides helper functions for:
- Numerical stability (log-space operations)
- Matrix operations
- Sequence processing
- Validation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


def log_sum_exp(x: np.ndarray) -> float:
    """
    Compute log(sum(exp(x))) in a numerically stable way.
    
    This is important to avoid underflow when working with very small probabilities.
    
    Mathematical trick:
        log(Σ exp(xᵢ)) = log(exp(xₘₐₓ) * Σ exp(xᵢ - xₘₐₓ))
                       = xₘₐₓ + log(Σ exp(xᵢ - xₘₐₓ))
    
    Args:
        x: Array of log-probabilities
        
    Returns:
        log(sum(exp(x)))
    """
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))


def normalize_probabilities(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Normalize probabilities to sum to 1.
    
    This is useful for numerical stability during HMM operations.
    
    Args:
        p: Probability array
        axis: Axis along which to normalize
        
    Returns:
        Normalized probability array
    """
    p_sum = np.sum(p, axis=axis, keepdims=True)
    # Avoid division by zero
    p_sum = np.where(p_sum == 0, 1.0, p_sum)
    return p / p_sum


def validate_observation_sequence(observations: np.ndarray, n_observations: int):
    """
    Validate that an observation sequence is valid.
    
    Args:
        observations: Observation sequence (indices)
        n_observations: Number of possible observations
        
    Raises:
        ValueError: If observations are invalid
    """
    if not isinstance(observations, np.ndarray):
        observations = np.array(observations)
    
    if observations.ndim != 1:
        raise ValueError(f"Observation sequence must be 1D, got shape {observations.shape}")
    
    if len(observations) == 0:
        raise ValueError("Observation sequence cannot be empty")
    
    if not np.all((observations >= 0) & (observations < n_observations)):
        raise ValueError(
            f"All observations must be in range [0, {n_observations}), "
            f"got min={observations.min()}, max={observations.max()}"
        )


def validate_state_sequence(states: np.ndarray, n_states: int):
    """
    Validate that a state sequence is valid.
    
    Args:
        states: State sequence (indices)
        n_states: Number of possible states
        
    Raises:
        ValueError: If states are invalid
    """
    if not isinstance(states, np.ndarray):
        states = np.array(states)
    
    if states.ndim != 1:
        raise ValueError(f"State sequence must be 1D, got shape {states.shape}")
    
    if len(states) == 0:
        raise ValueError("State sequence cannot be empty")
    
    if not np.all((states >= 0) & (states < n_states)):
        raise ValueError(
            f"All states must be in range [0, {n_states}), "
            f"got min={states.min()}, max={states.max()}"
        )


def generate_random_sequence(hmm, length: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random observation and state sequence from an HMM.
    
    This is useful for:
    - Testing algorithms
    - Generating synthetic data
    - Understanding HMM behavior
    
    Algorithm:
    1. Sample initial state from π
    2. For each time step:
       a. Sample observation from B[current_state]
       b. Sample next state from A[current_state]
    
    Args:
        hmm: HMM object with initialized parameters
        length: Length of sequence to generate
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (observations, states) - both shape (length,)
    """
    if not hmm.is_initialized():
        raise ValueError("HMM must be initialized before generating sequences")
    
    if seed is not None:
        np.random.seed(seed)
    
    observations = np.zeros(length, dtype=int)
    states = np.zeros(length, dtype=int)
    
    # Sample initial state from π
    states[0] = np.random.choice(hmm.n_states, p=hmm.pi)
    
    # Sample initial observation from B[initial_state]
    observations[0] = np.random.choice(hmm.n_observations, p=hmm.B[states[0]])
    
    # Generate rest of sequence
    for t in range(1, length):
        # Sample next state from A[current_state]
        states[t] = np.random.choice(hmm.n_states, p=hmm.A[states[t-1]])
        
        # Sample observation from B[current_state]
        observations[t] = np.random.choice(hmm.n_observations, p=hmm.B[states[t]])
    
    return observations, states


def compute_sequence_probability_naive(hmm, observations: np.ndarray, states: np.ndarray) -> float:
    """
    Compute probability of observation and state sequence (naive method).
    
    This is the brute-force way to compute P(O, Q | λ).
    Not efficient, but useful for verification and learning.
    
    Formula:
        P(O, Q | λ) = π[q₁] * b[q₁](o₁) * Π(t=2 to T) a[qₜ₋₁][qₜ] * b[qₜ](oₜ)
    
    Args:
        hmm: HMM object
        observations: Observation sequence, shape (T,)
        states: State sequence, shape (T,)
        
    Returns:
        Joint probability P(O, Q | λ)
    """
    T = len(observations)
    
    # Initial probability: π[q₁] * b[q₁](o₁)
    prob = hmm.pi[states[0]] * hmm.B[states[0], observations[0]]
    
    # Multiply by transition and emission probabilities for rest of sequence
    for t in range(1, T):
        prob *= hmm.A[states[t-1], states[t]]  # Transition probability
        prob *= hmm.B[states[t], observations[t]]  # Emission probability
    
    return prob


def print_probability_matrix(matrix: np.ndarray, 
                             row_labels: List[str] = None,
                             col_labels: List[str] = None,
                             title: str = "Probability Matrix"):
    """
    Pretty print a probability matrix.
    
    Args:
        matrix: 2D probability matrix
        row_labels: Labels for rows
        col_labels: Labels for columns
        title: Title for the matrix
    """
    if matrix.ndim != 2:
        raise ValueError("Matrix must be 2D")
    
    rows, cols = matrix.shape
    
    # Create default labels if not provided
    if row_labels is None:
        row_labels = [f"R{i}" for i in range(rows)]
    if col_labels is None:
        col_labels = [f"C{i}" for i in range(cols)]
    
    print(f"\n{title}")
    print("=" * (15 + 10 * cols))
    
    # Print column headers
    print("     ", end="")
    for label in col_labels:
        print(f"{label:>10}", end="")
    print()
    
    # Print rows
    for i, row_label in enumerate(row_labels):
        print(f"{row_label:>5}", end="")
        for j in range(cols):
            print(f"{matrix[i, j]:>10.4f}", end="")
        print()
    print("=" * (15 + 10 * cols))


def compare_hmm_parameters(hmm1, hmm2, tolerance: float = 1e-6) -> Dict[str, bool]:
    """
    Compare two HMMs to see if they have similar parameters.
    
    Useful for:
    - Testing convergence in Baum-Welch
    - Verifying learned parameters
    
    Args:
        hmm1: First HMM
        hmm2: Second HMM
        tolerance: Maximum difference for "close" values
        
    Returns:
        Dictionary with comparison results
    """
    if not (hmm1.is_initialized() and hmm2.is_initialized()):
        raise ValueError("Both HMMs must be initialized")
    
    if hmm1.n_states != hmm2.n_states or hmm1.n_observations != hmm2.n_observations:
        raise ValueError("HMMs must have same dimensions")
    
    results = {
        "A_close": np.allclose(hmm1.A, hmm2.A, atol=tolerance),
        "B_close": np.allclose(hmm1.B, hmm2.B, atol=tolerance),
        "pi_close": np.allclose(hmm1.pi, hmm2.pi, atol=tolerance),
        "A_max_diff": np.max(np.abs(hmm1.A - hmm2.A)),
        "B_max_diff": np.max(np.abs(hmm1.B - hmm2.B)),
        "pi_max_diff": np.max(np.abs(hmm1.pi - hmm2.pi))
    }
    
    results["all_close"] = results["A_close"] and results["B_close"] and results["pi_close"]
    
    return results


# Example usage and testing
if __name__ == "__main__":
    print("Testing HMM Utility Functions\n")
    
    # Test log-sum-exp
    print("1. Testing log-sum-exp")
    x = np.array([-1000, -999, -998])  # Very small probabilities in log space
    result = log_sum_exp(x)
    print(f"   log_sum_exp({x}) = {result:.4f}")
    print(f"   Expected: approximately {x[-1]:.4f}")
    
    # Test normalization
    print("\n2. Testing normalization")
    p = np.array([1.0, 2.0, 3.0])
    p_norm = normalize_probabilities(p)
    print(f"   Original: {p}")
    print(f"   Normalized: {p_norm}")
    print(f"   Sum: {p_norm.sum():.6f}")
    
    # Test sequence generation
    print("\n3. Testing sequence generation")
    from hmm_base import HMM
    
    hmm = HMM(n_states=2, n_observations=3, 
              state_names=["S1", "S2"],
              observation_names=["O1", "O2", "O3"])
    hmm.initialize_random(seed=42)
    
    obs, states = generate_random_sequence(hmm, length=10, seed=42)
    print(f"   Generated observations: {obs}")
    print(f"   Generated states: {states}")
    
    # Test probability computation
    print("\n4. Testing naive probability computation")
    prob = compute_sequence_probability_naive(hmm, obs, states)
    print(f"   P(O, Q | λ) = {prob:.10f}")
    
    # Test matrix printing
    print("\n5. Testing matrix printing")
    print_probability_matrix(
        hmm.A,
        row_labels=hmm.state_names,
        col_labels=hmm.state_names,
        title="Transition Matrix (A)"
    )

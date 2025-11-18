"""
Forward Algorithm Implementation

The Forward algorithm solves the Evaluation Problem:
    Given: Model λ = (A, B, π) and observation sequence O
    Find: P(O | λ) - the probability of observing O given the model

This is one of the three fundamental HMM problems.

Mathematical Foundation:
    Define forward variable α_t(i):
        α_t(i) = P(o₁, o₂, ..., oₜ, qₜ = sᵢ | λ)
    
    Meaning: Probability of observing the first t observations
             AND being in state i at time t
    
    Recursive computation:
        1. Initialization: α₁(i) = πᵢ * bᵢ(o₁)
        2. Induction: α_{t+1}(j) = [Σᵢ αₜ(i) * aᵢⱼ] * bⱼ(o_{t+1})
        3. Termination: P(O | λ) = Σᵢ α_T(i)

Time Complexity: O(N² * T) where N = states, T = sequence length
Space Complexity: O(N * T) to store all α values

Reference: Rabiner (1989) "A Tutorial on Hidden Markov Models", Section III-A
"""

import numpy as np
from typing import Tuple, Optional, List
from .hmm_base import HMM
from .utils import validate_observation_sequence, normalize_probabilities


def forward_algorithm(
    hmm: HMM,
    observations: np.ndarray,
    return_alpha: bool = False,
    normalize: bool = True,
    verbose: bool = False
) -> float or Tuple[float, np.ndarray]:
    """
    Compute the probability of an observation sequence using the Forward algorithm.
    
    This is the efficient solution to the evaluation problem.
    Instead of summing over all possible state sequences (exponential),
    we use dynamic programming (polynomial).
    
    Args:
        hmm: Initialized HMM model
        observations: Observation sequence, shape (T,)
        return_alpha: If True, return alpha matrix along with probability
        normalize: If True, normalize alpha at each step (numerical stability)
        verbose: If True, print step-by-step computation
        
    Returns:
        If return_alpha=False: P(O | λ)
        If return_alpha=True: (P(O | λ), alpha matrix shape (T, N))
    """
    # ========================================================================
    # STEP 0: Validation
    # ========================================================================
    if not hmm.is_initialized():
        raise ValueError("HMM must be initialized before running forward algorithm")
    
    validate_observation_sequence(observations, hmm.n_observations)
    
    T = len(observations)  # Sequence length
    N = hmm.n_states       # Number of states
    
    if verbose:
        print("\n" + "="*70)
        print("FORWARD ALGORITHM - Step-by-Step Execution")
        print("="*70)
        print(f"Sequence length T = {T}")
        print(f"Number of states N = {N}")
        print(f"Observations: {observations}")
        print(f"Observation names: {[hmm.observation_names[o] for o in observations]}")
    
    # ========================================================================
    # STEP 1: Initialization
    # ========================================================================
    # Create alpha matrix to store forward probabilities
    # alpha[t, i] = α_t(i) = P(o₁, ..., oₜ, qₜ = i | λ)
    alpha = np.zeros((T, N))
    
    # Initialize for t=1: α₁(i) = πᵢ * bᵢ(o₁)
    # Meaning: Probability of starting in state i and observing o₁
    alpha[0, :] = hmm.pi * hmm.B[:, observations[0]]
    
    if normalize:
        # Normalize to prevent underflow
        alpha[0, :] = normalize_probabilities(alpha[0, :])
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STEP 1: Initialization (t=1)")
        print(f"{'='*70}")
        print(f"Observation at t=1: {hmm.observation_names[observations[0]]} (index {observations[0]})")
        print(f"\nFor each state i, compute: α₁(i) = πᵢ * bᵢ(o₁)")
        for i in range(N):
            print(f"  α₁({hmm.state_names[i]}) = π[{i}] * B[{i},{observations[0]}]")
            print(f"                         = {hmm.pi[i]:.4f} * {hmm.B[i, observations[0]]:.4f}")
            print(f"                         = {alpha[0, i]:.6f}")
    
    # Store scaling factors if normalizing (useful for computing actual probability)
    if normalize:
        scale_factors = [np.sum(hmm.pi * hmm.B[:, observations[0]])]
    
    # ========================================================================
    # STEP 2: Induction (Recursion)
    # ========================================================================
    # For each time step t = 2, ..., T:
    #   α_{t}(j) = [Σᵢ α_{t-1}(i) * aᵢⱼ] * bⱼ(oₜ)
    #
    # Intuition: To be in state j at time t with observations o₁...oₜ:
    #   1. Must have been in some state i at t-1 with observations o₁...o_{t-1}: α_{t-1}(i)
    #   2. Must transition from i to j: aᵢⱼ
    #   3. Must observe oₜ from state j: bⱼ(oₜ)
    
    for t in range(1, T):
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 2: Induction (t={t+1})")
            print(f"{'='*70}")
            print(f"Observation at t={t+1}: {hmm.observation_names[observations[t]]} (index {observations[t]})")
        
        for j in range(N):
            # Compute transition contribution: Σᵢ α_{t-1}(i) * aᵢⱼ
            # This is the probability of reaching state j at time t
            # by transitioning from any state i at time t-1
            transition_sum = 0.0
            
            if verbose:
                print(f"\nComputing α_{t+1}({hmm.state_names[j]}):")
                print(f"  Step 2a: Sum over all previous states")
            
            for i in range(N):
                contribution = alpha[t-1, i] * hmm.A[i, j]
                transition_sum += contribution
                
                if verbose:
                    print(f"    From {hmm.state_names[i]}: α_{t}({hmm.state_names[i]}) * A[{i},{j}] "
                          f"= {alpha[t-1, i]:.6f} * {hmm.A[i, j]:.4f} = {contribution:.6f}")
            
            if verbose:
                print(f"  Transition sum: {transition_sum:.6f}")
            
            # Multiply by emission probability: bⱼ(oₜ)
            alpha[t, j] = transition_sum * hmm.B[j, observations[t]]
            
            if verbose:
                print(f"  Step 2b: Multiply by emission probability")
                print(f"    B[{j},{observations[t]}] = {hmm.B[j, observations[t]]:.4f}")
                print(f"  Final: α_{t+1}({hmm.state_names[j]}) = {alpha[t, j]:.8f}")
        
        if normalize:
            # Normalize to prevent underflow
            scale_factor = np.sum(alpha[t, :])
            scale_factors.append(scale_factor)
            alpha[t, :] = normalize_probabilities(alpha[t, :])
            
            if verbose:
                print(f"\n  Normalization: scale factor = {scale_factor:.8e}")
    
    # ========================================================================
    # STEP 3: Termination
    # ========================================================================
    # P(O | λ) = Σᵢ α_T(i)
    # Sum over all possible final states
    
    if normalize:
        # If we normalized, we need to multiply by all scale factors
        # to get the actual probability
        log_prob = np.sum(np.log(scale_factors))
        probability = np.exp(log_prob)
    else:
        probability = np.sum(alpha[T-1, :])
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STEP 3: Termination")
        print(f"{'='*70}")
        print(f"Sum over all final states:")
        for i in range(N):
            print(f"  α_T({hmm.state_names[i]}) = {alpha[T-1, i]:.8f}")
        print(f"\nP(O | λ) = Σᵢ α_T(i) = {probability:.10e}")
        print("="*70 + "\n")
    
    if return_alpha:
        return probability, alpha
    else:
        return probability


def forward_step(
    alpha_prev: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    observation: int
) -> np.ndarray:
    """
    Perform a single forward step.
    
    This computes α_t from α_{t-1}.
    Useful for online/streaming applications.
    
    Formula:
        α_t(j) = [Σᵢ α_{t-1}(i) * aᵢⱼ] * bⱼ(oₜ)
    
    Args:
        alpha_prev: α_{t-1}, shape (N,)
        A: Transition matrix, shape (N, N)
        B: Emission matrix, shape (N, M)
        observation: Current observation index
        
    Returns:
        α_t, shape (N,)
    """
    # Matrix multiplication computes Σᵢ α_{t-1}(i) * aᵢⱼ for all j
    transition_probs = alpha_prev @ A
    
    # Element-wise multiplication with emission probabilities
    alpha_t = transition_probs * B[:, observation]
    
    return alpha_t


# Example usage and testing
if __name__ == "__main__":
    print("Testing Forward Algorithm\n")
    
    # ========================================================================
    # Example 1: Simple Weather Model
    # ========================================================================
    print("="*70)
    print("EXAMPLE 1: Weather Model")
    print("="*70)
    
    from hmm_base import HMM
    
    # Create weather HMM
    hmm = HMM(
        n_states=2,
        n_observations=3,
        state_names=["Sunny", "Rainy"],
        observation_names=["Walk", "Shop", "Clean"]
    )
    
    # Set parameters
    A = np.array([
        [0.7, 0.3],  # Sunny -> Sunny: 0.7, Sunny -> Rainy: 0.3
        [0.4, 0.6]   # Rainy -> Sunny: 0.4, Rainy -> Rainy: 0.6
    ])
    
    B = np.array([
        [0.6, 0.3, 0.1],  # Sunny: Walk 0.6, Shop 0.3, Clean 0.1
        [0.1, 0.4, 0.5]   # Rainy: Walk 0.1, Shop 0.4, Clean 0.5
    ])
    
    pi = np.array([0.6, 0.4])  # 60% sunny, 40% rainy initially
    
    hmm.set_transition_matrix(A)
    hmm.set_emission_matrix(B)
    hmm.set_initial_probabilities(pi)
    
    # Test observation sequence: Walk, Shop, Clean
    observations = np.array([0, 1, 2])
    
    print("\nModel Parameters:")
    hmm.print_parameters()
    
    print(f"\nObservation sequence: {[hmm.observation_names[o] for o in observations]}")
    
    # Run forward algorithm with verbose output
    print("\n" + "="*70)
    print("Running Forward Algorithm (verbose mode)")
    print("="*70)
    
    prob, alpha = forward_algorithm(
        hmm,
        observations,
        return_alpha=True,
        normalize=True,
        verbose=True
    )
    
    print(f"\nFinal Result: P(O | λ) = {prob:.10f}")
    print(f"             Log P(O | λ) = {np.log(prob):.6f}")
    
    # ========================================================================
    # Example 2: Test with random sequences
    # ========================================================================
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Random Sequences")
    print("="*70)
    
    from utils import generate_random_sequence
    
    # Generate random sequence
    obs, true_states = generate_random_sequence(hmm, length=5, seed=42)
    
    print(f"\nGenerated observations: {[hmm.observation_names[o] for o in obs]}")
    print(f"True hidden states: {[hmm.state_names[s] for s in true_states]}")
    
    # Compute probability
    prob = forward_algorithm(hmm, obs, verbose=False)
    print(f"\nP(O | λ) = {prob:.10e}")
    
    # ========================================================================
    # Example 3: Verify correctness
    # ========================================================================
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Verification")
    print("="*70)
    
    # For a short sequence, we can verify by brute force
    # Try all possible state sequences and sum their probabilities
    
    from utils import compute_sequence_probability_naive
    
    short_obs = np.array([0, 1])  # Walk, Shop
    
    print(f"\nObservations: {[hmm.observation_names[o] for o in short_obs]}")
    
    # Compute using forward algorithm
    prob_forward = forward_algorithm(hmm, short_obs)
    print(f"\nForward algorithm: P(O | λ) = {prob_forward:.10f}")
    
    # Compute by brute force: sum over all possible state sequences
    print("\nBrute force verification:")
    print("Summing over all possible state sequences:")
    
    total_prob = 0.0
    for s1 in range(2):
        for s2 in range(2):
            states = np.array([s1, s2])
            state_names = [hmm.state_names[s] for s in states]
            prob = compute_sequence_probability_naive(hmm, short_obs, states)
            total_prob += prob
            print(f"  States: {state_names}, P(O, Q | λ) = {prob:.10f}")
    
    print(f"\nSum of all paths: {total_prob:.10f}")
    print(f"Forward algorithm: {prob_forward:.10f}")
    print(f"Difference: {abs(total_prob - prob_forward):.2e}")
    
    if np.isclose(total_prob, prob_forward, rtol=1e-6):
        print("✓ Forward algorithm verified!")
    else:
        print("✗ Verification failed!")

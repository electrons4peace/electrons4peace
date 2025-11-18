"""
Baum-Welch Algorithm Implementation

The Baum-Welch algorithm solves the Learning Problem:
    Given: Observation sequence O (and possibly multiple sequences)
    Find: Model parameters λ = (A, B, π) that maximize P(O | λ)

This is an Expectation-Maximization (EM) algorithm for HMMs.

Mathematical Foundation:
    The algorithm iteratively improves the model parameters:
    1. E-step: Compute expected counts using current parameters
    2. M-step: Update parameters based on expected counts
    
    Key probability variables:
        γ_t(i) = P(qₜ = i | O, λ)
                 Probability of being in state i at time t
                 
        ξ_t(i,j) = P(qₜ = i, q_{t+1} = j | O, λ)
                   Probability of transitioning from i to j at time t
    
    Computing γ and ξ:
        γ_t(i) = [α_t(i) * β_t(i)] / P(O | λ)
        
        ξ_t(i,j) = [α_t(i) * a_{ij} * b_j(o_{t+1}) * β_{t+1}(j)] / P(O | λ)
    
    Parameter updates (M-step):
        π_i = γ_1(i)  (expected frequency in state i at t=1)
        
        a_{ij} = [Σ_t ξ_t(i,j)] / [Σ_t γ_t(i)]
                 (expected # transitions i→j) / (expected # times in i)
        
        b_i(k) = [Σ_t γ_t(i)·1_{o_t=k}] / [Σ_t γ_t(i)]
                 (expected # times in i observing k) / (expected # times in i)

Convergence:
    - Guaranteed to converge to a local maximum
    - May not find global maximum (depends on initialization)
    - Converges when log-likelihood improvement < threshold

Time Complexity: O(N² * T) per iteration
Space Complexity: O(N * T)

Reference: Rabiner (1989), Section V
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from .hmm_base import HMM
from .forward import forward_algorithm
from .backward import backward_algorithm
from .utils import validate_observation_sequence, normalize_probabilities


def compute_gamma(
    alpha: np.ndarray,
    beta: np.ndarray
) -> np.ndarray:
    """
    Compute γ_t(i) = P(qₜ = i | O, λ).
    
    This is the probability of being in state i at time t
    given the entire observation sequence.
    
    Formula:
        γ_t(i) = [α_t(i) * β_t(i)] / Σⱼ [α_t(j) * β_t(j)]
               = [α_t(i) * β_t(i)] / P(O | λ)
    
    Intuition:
        α_t(i): probability of past observations AND being in state i at t
        β_t(i): probability of future observations GIVEN state i at t
        Product gives joint probability, normalize to get posterior
    
    Args:
        alpha: Forward probabilities, shape (T, N)
        beta: Backward probabilities, shape (T, N)
        
    Returns:
        gamma: State occupation probabilities, shape (T, N)
    """
    # Compute γ_t(i) for all t and i
    gamma = alpha * beta
    
    # Normalize so that Σᵢ γ_t(i) = 1 for each t
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    
    return gamma


def compute_xi(
    alpha: np.ndarray,
    beta: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    observations: np.ndarray
) -> np.ndarray:
    """
    Compute ξ_t(i,j) = P(qₜ = i, q_{t+1} = j | O, λ).
    
    This is the probability of transitioning from state i to state j
    between times t and t+1, given the entire observation sequence.
    
    Formula:
        ξ_t(i,j) = [α_t(i) * a_{ij} * b_j(o_{t+1}) * β_{t+1}(j)] / P(O | λ)
    
    Intuition:
        α_t(i): probability of being in state i at t with past observations
        a_{ij}: probability of transitioning i→j
        b_j(o_{t+1}): probability of observing o_{t+1} from state j
        β_{t+1}(j): probability of future observations from state j
    
    Args:
        alpha: Forward probabilities, shape (T, N)
        beta: Backward probabilities, shape (T, N)
        A: Transition matrix, shape (N, N)
        B: Emission matrix, shape (N, M)
        observations: Observation sequence, shape (T,)
        
    Returns:
        xi: Transition probabilities, shape (T-1, N, N)
            xi[t, i, j] = ξ_t(i,j)
    """
    T, N = alpha.shape
    xi = np.zeros((T-1, N, N))
    
    # For each time step t = 1, ..., T-1
    for t in range(T-1):
        # For each pair of states (i, j)
        for i in range(N):
            for j in range(N):
                # ξ_t(i,j) = α_t(i) * a_{ij} * b_j(o_{t+1}) * β_{t+1}(j)
                xi[t, i, j] = (alpha[t, i] * 
                              A[i, j] * 
                              B[j, observations[t+1]] * 
                              beta[t+1, j])
        
        # Normalize: Σᵢ Σⱼ ξ_t(i,j) = 1
        xi[t] = xi[t] / xi[t].sum()
    
    return xi


def baum_welch_algorithm(
    hmm: HMM,
    observations: np.ndarray,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False
) -> Tuple[HMM, List[float], int]:
    """
    Learn HMM parameters using the Baum-Welch algorithm.
    
    This is an EM algorithm that iteratively improves model parameters
    to maximize the likelihood of the observed data.
    
    Algorithm:
        Repeat until convergence:
            E-step: Compute γ and ξ using current parameters
            M-step: Update A, B, π based on γ and ξ
    
    Args:
        hmm: HMM with initialized parameters (starting point)
        observations: Training observation sequence, shape (T,)
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold (change in log-likelihood)
        verbose: If True, print progress
        
    Returns:
        hmm: HMM with learned parameters
        log_likelihoods: Log-likelihood at each iteration
        num_iterations: Number of iterations until convergence
    """
    # ========================================================================
    # STEP 0: Validation and initialization
    # ========================================================================
    if not hmm.is_initialized():
        raise ValueError("HMM must be initialized before training")
    
    validate_observation_sequence(observations, hmm.n_observations)
    
    T = len(observations)
    N = hmm.n_states
    M = hmm.n_observations
    
    if verbose:
        print("\n" + "="*70)
        print("BAUM-WELCH ALGORITHM - Training HMM Parameters")
        print("="*70)
        print(f"Sequence length: T = {T}")
        print(f"Number of states: N = {N}")
        print(f"Number of observations: M = {M}")
        print(f"Max iterations: {max_iterations}")
        print(f"Convergence tolerance: {tolerance}")
        print("\nInitial parameters:")
        hmm.print_parameters()
    
    log_likelihoods = []
    prev_log_likelihood = -np.inf
    
    # ========================================================================
    # MAIN LOOP: Iterate until convergence
    # ========================================================================
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}")
            print(f"{'='*70}")
        
        # ====================================================================
        # E-STEP: Compute forward and backward probabilities
        # ====================================================================
        if verbose:
            print("\nE-STEP: Computing expectations")
            print("  Running forward algorithm...")
        
        prob_forward, alpha = forward_algorithm(
            hmm, observations, return_alpha=True, normalize=True
        )
        
        if verbose:
            print(f"    P(O | λ) = {prob_forward:.10e}")
            print("  Running backward algorithm...")
        
        prob_backward, beta = backward_algorithm(
            hmm, observations, return_beta=True, normalize=True
        )
        
        # Compute log-likelihood
        log_likelihood = np.log(prob_forward + 1e-10)
        log_likelihoods.append(log_likelihood)
        
        if verbose:
            print(f"    Log-likelihood: {log_likelihood:.6f}")
        
        # Compute γ_t(i) = P(qₜ = i | O, λ)
        if verbose:
            print("  Computing γ (state occupation probabilities)...")
        gamma = compute_gamma(alpha, beta)
        
        # Compute ξ_t(i,j) = P(qₜ = i, q_{t+1} = j | O, λ)
        if verbose:
            print("  Computing ξ (transition probabilities)...")
        xi = compute_xi(alpha, beta, hmm.A, hmm.B, observations)
        
        # ====================================================================
        # M-STEP: Update parameters
        # ====================================================================
        if verbose:
            print("\nM-STEP: Updating parameters")
        
        # Update initial probabilities: π_i = γ_1(i)
        if verbose:
            print("  Updating π (initial probabilities)...")
        pi_new = gamma[0, :]
        
        # Update transition probabilities
        # a_{ij} = [Σ_t ξ_t(i,j)] / [Σ_t γ_t(i)]
        if verbose:
            print("  Updating A (transition matrix)...")
        
        A_new = np.zeros((N, N))
        for i in range(N):
            denominator = gamma[:-1, i].sum()  # Σ_t γ_t(i), t=1 to T-1
            if denominator > 0:
                for j in range(N):
                    numerator = xi[:, i, j].sum()  # Σ_t ξ_t(i,j)
                    A_new[i, j] = numerator / denominator
            else:
                # If state i is never visited, keep uniform distribution
                A_new[i, :] = 1.0 / N
        
        # Normalize to ensure rows sum to 1
        A_new = A_new / A_new.sum(axis=1, keepdims=True)
        
        # Update emission probabilities
        # b_i(k) = [Σ_t γ_t(i)·1_{o_t=k}] / [Σ_t γ_t(i)]
        if verbose:
            print("  Updating B (emission matrix)...")
        
        B_new = np.zeros((N, M))
        for i in range(N):
            denominator = gamma[:, i].sum()  # Σ_t γ_t(i)
            if denominator > 0:
                for k in range(M):
                    # Sum γ_t(i) for all times where o_t = k
                    mask = (observations == k)
                    numerator = gamma[mask, i].sum()
                    B_new[i, k] = numerator / denominator
            else:
                # If state i is never visited, keep uniform distribution
                B_new[i, :] = 1.0 / M
        
        # Normalize to ensure rows sum to 1
        B_new = B_new / B_new.sum(axis=1, keepdims=True)
        
        # ====================================================================
        # Update HMM with new parameters
        # ====================================================================
        hmm.set_initial_probabilities(pi_new)
        hmm.set_transition_matrix(A_new)
        hmm.set_emission_matrix(B_new)
        
        if verbose:
            print(f"\nParameter changes:")
            print(f"  Δπ (max): {np.max(np.abs(pi_new - gamma[0, :])):.6f}")
            print(f"  ΔA (max): {np.max(np.abs(A_new - hmm.A)):.6f}")
            print(f"  ΔB (max): {np.max(np.abs(B_new - hmm.B)):.6f}")
        
        # ====================================================================
        # Check convergence
        # ====================================================================
        log_likelihood_change = log_likelihood - prev_log_likelihood
        
        if verbose:
            print(f"\nLog-likelihood: {log_likelihood:.6f}")
            if iteration > 0:
                print(f"Improvement: {log_likelihood_change:.6f}")
        
        # Check if converged
        if iteration > 0 and abs(log_likelihood_change) < tolerance:
            if verbose:
                print(f"\n{'='*70}")
                print(f"CONVERGED after {iteration + 1} iterations")
                print(f"Log-likelihood change < tolerance ({tolerance})")
                print(f"{'='*70}")
            break
        
        prev_log_likelihood = log_likelihood
    
    else:
        # Loop completed without convergence
        if verbose:
            print(f"\n{'='*70}")
            print(f"STOPPED after {max_iterations} iterations (max reached)")
            print(f"{'='*70}")
    
    if verbose:
        print("\nFinal learned parameters:")
        hmm.print_parameters()
    
    return hmm, log_likelihoods, iteration + 1


def baum_welch_multiple_sequences(
    hmm: HMM,
    observation_sequences: List[np.ndarray],
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    verbose: bool = False
) -> Tuple[HMM, List[float], int]:
    """
    Learn HMM parameters from multiple observation sequences.
    
    When you have multiple independent sequences, you can train on all of them
    by accumulating the expected counts across sequences.
    
    Args:
        hmm: HMM with initialized parameters
        observation_sequences: List of observation sequences
        max_iterations: Maximum iterations
        tolerance: Convergence threshold
        verbose: Print progress
        
    Returns:
        hmm: Trained HMM
        log_likelihoods: Total log-likelihood at each iteration
        num_iterations: Number of iterations
    """
    if not hmm.is_initialized():
        raise ValueError("HMM must be initialized")
    
    K = len(observation_sequences)  # Number of sequences
    N = hmm.n_states
    M = hmm.n_observations
    
    if verbose:
        print(f"\nTraining on {K} sequences")
        print(f"Sequence lengths: {[len(seq) for seq in observation_sequences]}")
    
    log_likelihoods = []
    prev_log_likelihood = -np.inf
    
    for iteration in range(max_iterations):
        if verbose and iteration % 10 == 0:
            print(f"\nIteration {iteration + 1}/{max_iterations}")
        
        # Accumulators for expected counts across all sequences
        pi_acc = np.zeros(N)
        A_numerator_acc = np.zeros((N, N))
        A_denominator_acc = np.zeros(N)
        B_numerator_acc = np.zeros((N, M))
        B_denominator_acc = np.zeros(N)
        
        total_log_likelihood = 0.0
        
        # Process each sequence
        for obs in observation_sequences:
            # E-step for this sequence
            prob, alpha = forward_algorithm(hmm, obs, return_alpha=True)
            _, beta = backward_algorithm(hmm, obs, return_beta=True)
            
            total_log_likelihood += np.log(prob + 1e-10)
            
            gamma = compute_gamma(alpha, beta)
            xi = compute_xi(alpha, beta, hmm.A, hmm.B, obs)
            
            # Accumulate expected counts
            pi_acc += gamma[0, :]
            
            for i in range(N):
                A_denominator_acc[i] += gamma[:-1, i].sum()
                B_denominator_acc[i] += gamma[:, i].sum()
                
                for j in range(N):
                    A_numerator_acc[i, j] += xi[:, i, j].sum()
                
                for k in range(M):
                    mask = (obs == k)
                    B_numerator_acc[i, k] += gamma[mask, i].sum()
        
        log_likelihoods.append(total_log_likelihood)
        
        # M-step: Update parameters using accumulated counts
        pi_new = pi_acc / K
        
        A_new = np.zeros((N, N))
        for i in range(N):
            if A_denominator_acc[i] > 0:
                A_new[i, :] = A_numerator_acc[i, :] / A_denominator_acc[i]
            else:
                A_new[i, :] = 1.0 / N
        A_new = A_new / A_new.sum(axis=1, keepdims=True)
        
        B_new = np.zeros((N, M))
        for i in range(N):
            if B_denominator_acc[i] > 0:
                B_new[i, :] = B_numerator_acc[i, :] / B_denominator_acc[i]
            else:
                B_new[i, :] = 1.0 / M
        B_new = B_new / B_new.sum(axis=1, keepdims=True)
        
        # Update HMM
        hmm.set_initial_probabilities(pi_new)
        hmm.set_transition_matrix(A_new)
        hmm.set_emission_matrix(B_new)
        
        # Check convergence
        if iteration > 0:
            improvement = total_log_likelihood - prev_log_likelihood
            if verbose and iteration % 10 == 0:
                print(f"  Log-likelihood: {total_log_likelihood:.4f} (Δ = {improvement:.6f})")
            
            if abs(improvement) < tolerance:
                if verbose:
                    print(f"\nConverged after {iteration + 1} iterations")
                break
        
        prev_log_likelihood = total_log_likelihood
    
    return hmm, log_likelihoods, iteration + 1


# Example usage and testing
if __name__ == "__main__":
    print("Testing Baum-Welch Algorithm\n")
    
    from hmm_base import HMM
    from utils import generate_random_sequence
    
    # ========================================================================
    # Example 1: Learn parameters from generated data
    # ========================================================================
    print("="*70)
    print("EXAMPLE 1: Learning from Synthetic Data")
    print("="*70)
    
    # Create true model
    true_hmm = HMM(
        n_states=2,
        n_observations=3,
        state_names=["Sunny", "Rainy"],
        observation_names=["Walk", "Shop", "Clean"]
    )
    
    # Set true parameters
    true_A = np.array([[0.7, 0.3], [0.4, 0.6]])
    true_B = np.array([[0.6, 0.3, 0.1], [0.1, 0.4, 0.5]])
    true_pi = np.array([0.6, 0.4])
    
    true_hmm.set_transition_matrix(true_A)
    true_hmm.set_emission_matrix(true_B)
    true_hmm.set_initial_probabilities(true_pi)
    
    print("\nTrue model:")
    true_hmm.print_parameters()
    
    # Generate training data
    print("\nGenerating training data...")
    observations, true_states = generate_random_sequence(true_hmm, length=100, seed=42)
    print(f"Generated sequence of length {len(observations)}")
    
    # Create learner with random initialization
    print("\nInitializing learner with random parameters...")
    learner_hmm = HMM(
        n_states=2,
        n_observations=3,
        state_names=["Sunny", "Rainy"],
        observation_names=["Walk", "Shop", "Clean"]
    )
    learner_hmm.initialize_random(seed=123)
    
    print("\nInitial (random) parameters:")
    learner_hmm.print_parameters()
    
    # Train using Baum-Welch
    print("\n" + "="*70)
    print("Training with Baum-Welch...")
    print("="*70)
    
    learned_hmm, log_liks, num_iters = baum_welch_algorithm(
        learner_hmm,
        observations,
        max_iterations=50,
        tolerance=1e-4,
        verbose=True
    )
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    print("\nTrue parameters:")
    true_hmm.print_parameters()
    
    print("\nLearned parameters:")
    learned_hmm.print_parameters()
    
    # Plot convergence
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(log_liks, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Log-Likelihood')
        plt.title('Baum-Welch Convergence')
        plt.grid(True)
        plt.savefig('/mnt/user-data/outputs/baum_welch_convergence.png', dpi=150)
        print("\nConvergence plot saved to baum_welch_convergence.png")
    except:
        print("\nNote: Matplotlib not available, skipping plot")

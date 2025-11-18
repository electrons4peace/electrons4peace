"""
Viterbi Algorithm Implementation

The Viterbi algorithm solves the Decoding Problem:
    Given: Model λ = (A, B, π) and observation sequence O
    Find: Most likely state sequence Q* = argmax P(Q | O, λ)

This finds the single best path through the state space.

Mathematical Foundation:
    Define Viterbi variable δ_t(i):
        δ_t(i) = max_{q1,...,qt-1} P(q₁, ..., q_{t-1}, qₜ = i, o₁, ..., oₜ | λ)
    
    Meaning: Probability of the best path ending in state i at time t
    
    Key difference from Forward:
        Forward: SUMS over all paths (total probability)
        Viterbi: Takes MAX over all paths (best path)
    
    Recursive computation:
        1. Initialization: δ₁(i) = πᵢ * bᵢ(o₁)
                          ψ₁(i) = 0 (no previous state)
        
        2. Induction: δₜ(j) = max_i [δ_{t-1}(i) * aᵢⱼ] * bⱼ(oₜ)
                      ψₜ(j) = argmax_i [δ_{t-1}(i) * aᵢⱼ] (backpointer)
        
        3. Termination: P* = max_i δ_T(i)
                       q*_T = argmax_i δ_T(i)
        
        4. Backtracking: q*_t = ψ_{t+1}(q*_{t+1}) for t = T-1, ..., 1

Time Complexity: O(N² * T)
Space Complexity: O(N * T) for storing δ and ψ

Intuition: Dynamic programming - at each step, we only remember the best
          path to each state, not all possible paths.

Reference: Rabiner (1989), Section IV
"""

import numpy as np
from typing import Tuple, List
from .hmm_base import HMM
from .utils import validate_observation_sequence


def viterbi_algorithm(
    hmm: HMM,
    observations: np.ndarray,
    return_delta: bool = False,
    verbose: bool = False
) -> Tuple[np.ndarray, float] or Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Find the most likely state sequence using the Viterbi algorithm.
    
    This uses dynamic programming to find the single best path through
    the state space that explains the observations.
    
    Args:
        hmm: Initialized HMM model
        observations: Observation sequence, shape (T,)
        return_delta: If True, return delta and psi matrices
        verbose: If True, print step-by-step computation
        
    Returns:
        best_path: Most likely state sequence, shape (T,)
        best_path_prob: Probability of the best path
        
        If return_delta=True, also returns:
            delta: Viterbi probabilities, shape (T, N)
            psi: Backpointers, shape (T, N)
    """
    # ========================================================================
    # STEP 0: Validation
    # ========================================================================
    if not hmm.is_initialized():
        raise ValueError("HMM must be initialized before running Viterbi algorithm")
    
    validate_observation_sequence(observations, hmm.n_observations)
    
    T = len(observations)  # Sequence length
    N = hmm.n_states       # Number of states
    
    if verbose:
        print("\n" + "="*70)
        print("VITERBI ALGORITHM - Step-by-Step Execution")
        print("="*70)
        print(f"Sequence length T = {T}")
        print(f"Number of states N = {N}")
        print(f"Observations: {observations}")
        print(f"Observation names: {[hmm.observation_names[o] for o in observations]}")
        print("\nGoal: Find the most likely state sequence")
    
    # ========================================================================
    # STEP 1: Initialization
    # ========================================================================
    # delta[t, i] = δ_t(i) = max probability of path ending in state i at time t
    delta = np.zeros((T, N))
    
    # psi[t, i] = ψ_t(i) = backpointer to previous state in best path
    # psi stores which state was chosen at each step
    psi = np.zeros((T, N), dtype=int)
    
    # Initialize for t=1: δ₁(i) = πᵢ * bᵢ(o₁)
    # Same as forward algorithm, but we'll take max instead of sum later
    delta[0, :] = hmm.pi * hmm.B[:, observations[0]]
    psi[0, :] = 0  # No previous state at t=1
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STEP 1: Initialization (t=1)")
        print(f"{'='*70}")
        print(f"Observation at t=1: {hmm.observation_names[observations[0]]} (index {observations[0]})")
        print(f"\nFor each state i, compute: δ₁(i) = πᵢ * bᵢ(o₁)")
        for i in range(N):
            print(f"  δ₁({hmm.state_names[i]}) = {hmm.pi[i]:.4f} * {hmm.B[i, observations[0]]:.4f}")
            print(f"                         = {delta[0, i]:.6f}")
    
    # ========================================================================
    # STEP 2: Recursion - Taking MAX instead of SUM
    # ========================================================================
    # For each time step t = 2, ..., T:
    #   δₜ(j) = max_i [δ_{t-1}(i) * aᵢⱼ] * bⱼ(oₜ)
    #   ψₜ(j) = argmax_i [δ_{t-1}(i) * aᵢⱼ]
    #
    # Key difference from Forward algorithm:
    #   Forward: α_{t}(j) = [Σᵢ α_{t-1}(i) * aᵢⱼ] * bⱼ(oₜ)  (SUM)
    #   Viterbi: δₜ(j) = [max_i δ_{t-1}(i) * aᵢⱼ] * bⱼ(oₜ)  (MAX)
    
    for t in range(1, T):
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 2: Recursion (t={t+1})")
            print(f"{'='*70}")
            print(f"Observation at t={t+1}: {hmm.observation_names[observations[t]]} (index {observations[t]})")
        
        for j in range(N):
            # For each possible next state j:
            # 1. Compute δ_{t-1}(i) * aᵢⱼ for all previous states i
            # 2. Take the MAXIMUM
            # 3. Multiply by emission probability bⱼ(oₜ)
            
            if verbose:
                print(f"\nComputing δ_{t+1}({hmm.state_names[j]}):")
                print(f"  Consider all possible previous states:")
            
            # Compute transition scores for all previous states
            transition_scores = delta[t-1, :] * hmm.A[:, j]
            
            if verbose:
                for i in range(N):
                    score = transition_scores[i]
                    print(f"    From {hmm.state_names[i]}: δ_{t}({hmm.state_names[i]}) * A[{i},{j}]")
                    print(f"                          = {delta[t-1, i]:.6f} * {hmm.A[i, j]:.4f}")
                    print(f"                          = {score:.8f}")
            
            # Take maximum and remember which state gave it (backpointer)
            best_prev_state = np.argmax(transition_scores)
            max_transition_score = transition_scores[best_prev_state]
            
            # Multiply by emission probability
            delta[t, j] = max_transition_score * hmm.B[j, observations[t]]
            psi[t, j] = best_prev_state
            
            if verbose:
                print(f"  Best previous state: {hmm.state_names[best_prev_state]}")
                print(f"  Max transition score: {max_transition_score:.8f}")
                print(f"  Emission B[{j},{observations[t]}]: {hmm.B[j, observations[t]]:.4f}")
                print(f"  δ_{t+1}({hmm.state_names[j]}) = {delta[t, j]:.10f}")
                print(f"  ψ_{t+1}({hmm.state_names[j]}) = {best_prev_state} ({hmm.state_names[best_prev_state]})")
    
    # ========================================================================
    # STEP 3: Termination
    # ========================================================================
    # Find the best final state
    # P* = max_i δ_T(i)
    # q*_T = argmax_i δ_T(i)
    
    best_last_state = np.argmax(delta[T-1, :])
    best_path_prob = delta[T-1, best_last_state]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STEP 3: Termination")
        print(f"{'='*70}")
        print(f"Find best final state:")
        for i in range(N):
            marker = " ← BEST" if i == best_last_state else ""
            print(f"  δ_T({hmm.state_names[i]}) = {delta[T-1, i]:.10f}{marker}")
        print(f"\nBest final state: {hmm.state_names[best_last_state]}")
        print(f"Best path probability: {best_path_prob:.10e}")
    
    # ========================================================================
    # STEP 4: Backtracking
    # ========================================================================
    # Reconstruct the best path by following backpointers
    # Start from best final state and work backwards
    
    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = best_last_state
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STEP 4: Backtracking")
        print(f"{'='*70}")
        print(f"Reconstruct best path by following backpointers:")
        print(f"  t={T}: q*_{T} = {hmm.state_names[best_last_state]}")
    
    # Follow backpointers from T-1 to 1
    for t in range(T-2, -1, -1):
        best_path[t] = psi[t+1, best_path[t+1]]
        
        if verbose:
            print(f"  t={t+1}: q*_{t+1} = ψ_{t+2}(q*_{t+2}) = ψ_{t+2}({hmm.state_names[best_path[t+1]]}) = {hmm.state_names[best_path[t]]}")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FINAL RESULT")
        print(f"{'='*70}")
        print(f"Best state sequence:")
        for t in range(T):
            print(f"  t={t+1}: {hmm.state_names[best_path[t]]:>10} (obs: {hmm.observation_names[observations[t]]})")
        print(f"\nBest path probability: {best_path_prob:.10e}")
        print(f"Log probability: {np.log(best_path_prob):.6f}")
        print("="*70 + "\n")
    
    if return_delta:
        return best_path, best_path_prob, delta, psi
    else:
        return best_path, best_path_prob


def viterbi_log_space(
    hmm: HMM,
    observations: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Viterbi algorithm in log space for numerical stability.
    
    When dealing with long sequences, probabilities can underflow.
    Working in log space prevents this:
        log(a * b) = log(a) + log(b)
        log(max(a, b)) = max(log(a), log(b))
    
    Args:
        hmm: Initialized HMM model
        observations: Observation sequence, shape (T,)
        
    Returns:
        best_path: Most likely state sequence, shape (T,)
        log_prob: Log probability of best path
    """
    validate_observation_sequence(observations, hmm.n_observations)
    
    T = len(observations)
    N = hmm.n_states
    
    # Work with log probabilities
    log_pi = np.log(hmm.pi + 1e-10)  # Add small constant to avoid log(0)
    log_A = np.log(hmm.A + 1e-10)
    log_B = np.log(hmm.B + 1e-10)
    
    # Delta in log space
    log_delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    
    # Initialization
    log_delta[0, :] = log_pi + log_B[:, observations[0]]
    
    # Recursion (in log space: max(a*b) becomes max(log(a) + log(b)))
    for t in range(1, T):
        for j in range(N):
            # Compute log(δ_{t-1}(i) * aᵢⱼ) = log(δ_{t-1}(i)) + log(aᵢⱼ)
            scores = log_delta[t-1, :] + log_A[:, j]
            psi[t, j] = np.argmax(scores)
            log_delta[t, j] = scores[psi[t, j]] + log_B[j, observations[t]]
    
    # Termination
    best_last_state = np.argmax(log_delta[T-1, :])
    log_prob = log_delta[T-1, best_last_state]
    
    # Backtracking
    best_path = np.zeros(T, dtype=int)
    best_path[T-1] = best_last_state
    for t in range(T-2, -1, -1):
        best_path[t] = psi[t+1, best_path[t+1]]
    
    return best_path, log_prob


# Example usage and testing
if __name__ == "__main__":
    print("Testing Viterbi Algorithm\n")
    
    from hmm_base import HMM
    from utils import generate_random_sequence, compute_sequence_probability_naive
    
    # ========================================================================
    # Example 1: Weather Model - Detailed walkthrough
    # ========================================================================
    print("="*70)
    print("EXAMPLE 1: Weather Model - Finding Hidden States")
    print("="*70)
    
    # Create weather HMM
    hmm = HMM(
        n_states=2,
        n_observations=3,
        state_names=["Sunny", "Rainy"],
        observation_names=["Walk", "Shop", "Clean"]
    )
    
    A = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    
    B = np.array([
        [0.6, 0.3, 0.1],
        [0.1, 0.4, 0.5]
    ])
    
    pi = np.array([0.6, 0.4])
    
    hmm.set_transition_matrix(A)
    hmm.set_emission_matrix(B)
    hmm.set_initial_probabilities(pi)
    
    observations = np.array([0, 1, 2])  # Walk, Shop, Clean
    
    print("\nScenario: You observe [Walk, Shop, Clean]")
    print("Question: What was the most likely weather sequence?")
    
    # Run Viterbi with verbose output
    best_path, prob = viterbi_algorithm(hmm, observations, verbose=True)
    
    print("\nInterpretation:")
    print("The Viterbi algorithm found the single most likely explanation")
    print("for the observed activities.")
    
    # ========================================================================
    # Example 2: Compare with all possible paths (brute force)
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 2: Verification - Compare with Brute Force")
    print("="*70)
    
    short_obs = np.array([0, 1])  # Walk, Shop
    
    print(f"\nObservations: {[hmm.observation_names[o] for o in short_obs]}")
    
    # Run Viterbi
    best_path, viterbi_prob = viterbi_algorithm(hmm, short_obs, verbose=False)
    
    print(f"\nViterbi result:")
    print(f"  Best path: {[hmm.state_names[s] for s in best_path]}")
    print(f"  Probability: {viterbi_prob:.10f}")
    
    # Enumerate all possible paths and find best
    print(f"\nBrute force - check all possible paths:")
    best_brute_path = None
    best_brute_prob = 0.0
    
    for s1 in range(2):
        for s2 in range(2):
            states = np.array([s1, s2])
            prob = compute_sequence_probability_naive(hmm, short_obs, states)
            state_names = [hmm.state_names[s] for s in states]
            marker = " ← BEST" if prob > best_brute_prob else ""
            print(f"  Path {state_names}: P = {prob:.10f}{marker}")
            
            if prob > best_brute_prob:
                best_brute_prob = prob
                best_brute_path = states
    
    print(f"\nComparison:")
    print(f"  Viterbi: {[hmm.state_names[s] for s in best_path]} (P = {viterbi_prob:.10f})")
    print(f"  Brute:   {[hmm.state_names[s] for s in best_brute_path]} (P = {best_brute_prob:.10f})")
    
    if np.array_equal(best_path, best_brute_path):
        print("  ✓ Viterbi found the correct best path!")
    else:
        print("  ✗ Mismatch!")
    
    # ========================================================================
    # Example 3: Test with known true states
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 3: Recovery of True Hidden States")
    print("="*70)
    
    # Generate a sequence where we know the true states
    obs, true_states = generate_random_sequence(hmm, length=10, seed=42)
    
    print(f"\nGenerated sequence:")
    print(f"  Observations: {[hmm.observation_names[o] for o in obs]}")
    print(f"  True states:  {[hmm.state_names[s] for s in true_states]}")
    
    # Run Viterbi
    pred_states, prob = viterbi_algorithm(hmm, obs, verbose=False)
    
    print(f"  Viterbi pred: {[hmm.state_names[s] for s in pred_states]}")
    
    # Compare
    accuracy = np.mean(pred_states == true_states)
    print(f"\nAccuracy: {accuracy*100:.1f}%")
    print(f"Best path probability: {prob:.10e}")
    
    if accuracy == 1.0:
        print("✓ Perfect recovery of hidden states!")
    else:
        print(f"Note: Viterbi finds MOST LIKELY path, not necessarily TRUE path")
        print(f"      (True path might have lower probability)")
    
    # ========================================================================
    # Example 4: Log-space Viterbi for long sequences
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 4: Log-space Viterbi (Numerical Stability)")
    print("="*70)
    
    long_obs, _ = generate_random_sequence(hmm, length=100, seed=123)
    
    print("\nTesting on long sequence (length=100):")
    
    # Regular Viterbi
    path_reg, prob_reg = viterbi_algorithm(hmm, long_obs, verbose=False)
    
    # Log-space Viterbi
    path_log, log_prob = viterbi_log_space(hmm, long_obs)
    
    print(f"  Regular Viterbi probability: {prob_reg:.2e}")
    print(f"  Log-space log-probability: {log_prob:.6f}")
    print(f"  Converted to probability: {np.exp(log_prob):.2e}")
    
    if np.array_equal(path_reg, path_log):
        print("  ✓ Both methods found the same path!")
    else:
        print("  Note: Different paths due to numerical precision")

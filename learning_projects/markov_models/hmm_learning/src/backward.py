"""
Backward Algorithm Implementation

The Backward algorithm complements the Forward algorithm.
It's used primarily as a component in the Baum-Welch learning algorithm.

Mathematical Foundation:
    Define backward variable β_t(i):
        β_t(i) = P(o_{t+1}, o_{t+2}, ..., o_T | qₜ = sᵢ, λ)
    
    Meaning: Probability of observing the remaining sequence (from t+1 to T)
             GIVEN that we are in state i at time t
    
    Recursive computation:
        1. Initialization: β_T(i) = 1 for all i
        2. Induction: β_t(i) = Σⱼ aᵢⱼ * bⱼ(o_{t+1}) * β_{t+1}(j)
        3. Probability: P(O | λ) = Σᵢ πᵢ * bᵢ(o₁) * β₁(i)

Key Insight:
    Forward looks ahead: "How did we get here?"
    Backward looks back: "Where can we go from here?"
    
    Together: α_t(i) * β_t(i) ∝ P(qₜ = i | O, λ)
              (probability of being in state i at time t given observations)

Time Complexity: O(N² * T)
Space Complexity: O(N * T)

Reference: Rabiner (1989), Section III-B
"""

import numpy as np
from typing import Tuple, Optional
from .hmm_base import HMM
from .utils import validate_observation_sequence, normalize_probabilities


def backward_algorithm(
    hmm: HMM,
    observations: np.ndarray,
    return_beta: bool = False,
    normalize: bool = True,
    verbose: bool = False
) -> float or Tuple[float, np.ndarray]:
    """
    Compute the probability of an observation sequence using the Backward algorithm.
    
    The backward algorithm works in reverse chronological order.
    It computes the probability of seeing the future observations
    given we're in a particular state at each time.
    
    Args:
        hmm: Initialized HMM model
        observations: Observation sequence, shape (T,)
        return_beta: If True, return beta matrix along with probability
        normalize: If True, normalize beta at each step (numerical stability)
        verbose: If True, print step-by-step computation
        
    Returns:
        If return_beta=False: P(O | λ)
        If return_beta=True: (P(O | λ), beta matrix shape (T, N))
    """
    # ========================================================================
    # STEP 0: Validation
    # ========================================================================
    if not hmm.is_initialized():
        raise ValueError("HMM must be initialized before running backward algorithm")
    
    validate_observation_sequence(observations, hmm.n_observations)
    
    T = len(observations)  # Sequence length
    N = hmm.n_states       # Number of states
    
    if verbose:
        print("\n" + "="*70)
        print("BACKWARD ALGORITHM - Step-by-Step Execution")
        print("="*70)
        print(f"Sequence length T = {T}")
        print(f"Number of states N = {N}")
        print(f"Observations: {observations}")
        print(f"Observation names: {[hmm.observation_names[o] for o in observations]}")
    
    # ========================================================================
    # STEP 1: Initialization
    # ========================================================================
    # Create beta matrix to store backward probabilities
    # beta[t, i] = β_t(i) = P(o_{t+1}, ..., o_T | qₜ = i, λ)
    beta = np.zeros((T, N))
    
    # Initialize for t=T: β_T(i) = 1 for all states
    # Intuition: If we're at the end, there are no future observations,
    # so the probability is 1 (certainty of seeing nothing)
    beta[T-1, :] = 1.0
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STEP 1: Initialization (t=T={T})")
        print(f"{'='*70}")
        print(f"Set β_T(i) = 1 for all states (no future observations)")
        for i in range(N):
            print(f"  β_{T}({hmm.state_names[i]}) = {beta[T-1, i]:.4f}")
    
    # ========================================================================
    # STEP 2: Induction (Recursion) - Working BACKWARDS
    # ========================================================================
    # For each time step t = T-1, T-2, ..., 1:
    #   β_t(i) = Σⱼ aᵢⱼ * bⱼ(o_{t+1}) * β_{t+1}(j)
    #
    # Intuition: The probability of future observations starting from state i
    # is the sum over all possible next states j of:
    #   - Probability of transitioning i→j: aᵢⱼ
    #   - Probability of observing o_{t+1} from state j: bⱼ(o_{t+1})
    #   - Probability of remaining observations from state j: β_{t+1}(j)
    
    for t in range(T-2, -1, -1):  # T-1 down to 0 (working backwards!)
        if verbose:
            print(f"\n{'='*70}")
            print(f"STEP 2: Induction (t={t+1}, computing from t={t+2})")
            print(f"{'='*70}")
            print(f"Next observation (t+1={t+2}): {hmm.observation_names[observations[t+1]]} (index {observations[t+1]})")
        
        for i in range(N):
            # Compute: β_t(i) = Σⱼ aᵢⱼ * bⱼ(o_{t+1}) * β_{t+1}(j)
            backward_sum = 0.0
            
            if verbose:
                print(f"\nComputing β_{t+1}({hmm.state_names[i]}):")
                print(f"  Sum over all next states j:")
            
            for j in range(N):
                # Contribution from transitioning to state j
                contribution = (hmm.A[i, j] *                    # Transition i→j
                              hmm.B[j, observations[t+1]] *      # Emit o_{t+1} from j
                              beta[t+1, j])                      # Future from j
                backward_sum += contribution
                
                if verbose:
                    print(f"    To {hmm.state_names[j]}: A[{i},{j}] * B[{j},{observations[t+1]}] * β_{t+2}({hmm.state_names[j]})")
                    print(f"                  = {hmm.A[i,j]:.4f} * {hmm.B[j,observations[t+1]]:.4f} * {beta[t+1,j]:.6f}")
                    print(f"                  = {contribution:.8f}")
            
            beta[t, i] = backward_sum
            
            if verbose:
                print(f"  β_{t+1}({hmm.state_names[i]}) = {beta[t, i]:.8f}")
        
        if normalize:
            # Normalize to prevent underflow (use same scale as forward)
            beta[t, :] = normalize_probabilities(beta[t, :])
            
            if verbose:
                scale_factor = np.sum(beta[t, :])
                print(f"\n  Normalization applied")
    
    # ========================================================================
    # STEP 3: Termination
    # ========================================================================
    # P(O | λ) = Σᵢ πᵢ * bᵢ(o₁) * β₁(i)
    # 
    # Intuition: Sum over all possible starting states i:
    #   - Probability of starting in state i: πᵢ
    #   - Probability of observing o₁ from state i: bᵢ(o₁)
    #   - Probability of remaining observations from state i: β₁(i)
    
    probability = np.sum(hmm.pi * hmm.B[:, observations[0]] * beta[0, :])
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STEP 3: Termination")
        print(f"{'='*70}")
        print(f"Sum over all initial states:")
        for i in range(N):
            term = hmm.pi[i] * hmm.B[i, observations[0]] * beta[0, i]
            print(f"  π[{i}] * B[{i},{observations[0]}] * β_1({hmm.state_names[i]})")
            print(f"    = {hmm.pi[i]:.4f} * {hmm.B[i,observations[0]]:.4f} * {beta[0,i]:.6f}")
            print(f"    = {term:.8f}")
        print(f"\nP(O | λ) = {probability:.10e}")
        print("="*70 + "\n")
    
    if return_beta:
        return probability, beta
    else:
        return probability


def backward_step(
    beta_next: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    observation_next: int
) -> np.ndarray:
    """
    Perform a single backward step.
    
    This computes β_t from β_{t+1}.
    
    Formula:
        β_t(i) = Σⱼ aᵢⱼ * bⱼ(o_{t+1}) * β_{t+1}(j)
    
    Args:
        beta_next: β_{t+1}, shape (N,)
        A: Transition matrix, shape (N, N)
        B: Emission matrix, shape (N, M)
        observation_next: Next observation index o_{t+1}
        
    Returns:
        β_t, shape (N,)
    """
    N = A.shape[0]
    beta_t = np.zeros(N)
    
    for i in range(N):
        # For each current state i, sum over all next states j
        beta_t[i] = np.sum(
            A[i, :] *                    # Transition probabilities from i
            B[:, observation_next] *     # Emission probabilities for next observation
            beta_next                    # Backward probabilities from next states
        )
    
    return beta_t


def verify_forward_backward_consistency(
    hmm: HMM,
    observations: np.ndarray,
    tolerance: float = 1e-6,
    verbose: bool = False
) -> bool:
    """
    Verify that forward and backward algorithms produce consistent probabilities.
    
    The forward and backward algorithms should compute the same P(O | λ).
    This is a good sanity check for implementation correctness.
    
    Args:
        hmm: Initialized HMM
        observations: Observation sequence
        tolerance: Acceptable difference between probabilities
        verbose: If True, print detailed comparison
        
    Returns:
        True if probabilities match within tolerance
    """
    from .forward import forward_algorithm
    
    # Compute probability using forward algorithm
    prob_forward = forward_algorithm(hmm, observations, normalize=True)
    
    # Compute probability using backward algorithm
    prob_backward = backward_algorithm(hmm, observations, normalize=True)
    
    # Check if they match
    match = np.isclose(prob_forward, prob_backward, rtol=tolerance)
    
    if verbose:
        print("\n" + "="*70)
        print("Forward-Backward Consistency Check")
        print("="*70)
        print(f"P(O | λ) from Forward:  {prob_forward:.10e}")
        print(f"P(O | λ) from Backward: {prob_backward:.10e}")
        print(f"Relative difference: {abs(prob_forward - prob_backward) / prob_forward:.2e}")
        print(f"Match (tolerance {tolerance}): {match}")
        print("="*70)
    
    return match


# Example usage and testing
if __name__ == "__main__":
    print("Testing Backward Algorithm\n")
    
    from hmm_base import HMM
    from forward import forward_algorithm
    
    # ========================================================================
    # Example 1: Weather Model - Detailed walkthrough
    # ========================================================================
    print("="*70)
    print("EXAMPLE 1: Weather Model")
    print("="*70)
    
    # Create weather HMM (same as forward algorithm example)
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
    
    print("\nModel Parameters:")
    hmm.print_parameters()
    
    print(f"\nObservation sequence: {[hmm.observation_names[o] for o in observations]}")
    
    # Run backward algorithm with verbose output
    print("\n" + "="*70)
    print("Running Backward Algorithm (verbose mode)")
    print("="*70)
    
    prob_backward, beta = backward_algorithm(
        hmm,
        observations,
        return_beta=True,
        normalize=True,
        verbose=True
    )
    
    # ========================================================================
    # Example 2: Compare Forward and Backward
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 2: Forward-Backward Consistency")
    print("="*70)
    
    prob_forward = forward_algorithm(hmm, observations)
    
    print(f"\nP(O | λ) from Forward:  {prob_forward:.10e}")
    print(f"P(O | λ) from Backward: {prob_backward:.10e}")
    print(f"Difference: {abs(prob_forward - prob_backward):.2e}")
    
    if np.isclose(prob_forward, prob_backward, rtol=1e-6):
        print("✓ Forward and Backward algorithms agree!")
    else:
        print("✗ Mismatch between Forward and Backward!")
    
    # ========================================================================
    # Example 3: Automatic verification
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 3: Automated Verification Test")
    print("="*70)
    
    from utils import generate_random_sequence
    
    # Test on multiple random sequences
    num_tests = 5
    all_passed = True
    
    for i in range(num_tests):
        obs, _ = generate_random_sequence(hmm, length=10, seed=i)
        match = verify_forward_backward_consistency(hmm, obs, verbose=False)
        
        if match:
            print(f"  Test {i+1}: ✓ PASSED")
        else:
            print(f"  Test {i+1}: ✗ FAILED")
            all_passed = False
    
    if all_passed:
        print("\n✓ All tests passed! Backward algorithm is correct.")
    else:
        print("\n✗ Some tests failed. Check implementation.")
    
    # ========================================================================
    # Example 4: Show beta matrix structure
    # ========================================================================
    print("\n" + "="*70)
    print("EXAMPLE 4: Beta Matrix Structure")
    print("="*70)
    
    short_obs = np.array([0, 1])
    _, beta = backward_algorithm(hmm, short_obs, return_beta=True, verbose=False)
    
    print(f"\nObservation sequence: {[hmm.observation_names[o] for o in short_obs]}")
    print("\nBeta matrix (rows=time, cols=states):")
    print(f"Time  ", end="")
    for state in hmm.state_names:
        print(f"{state:>12}", end="")
    print()
    print("-" * 30)
    
    for t in range(len(short_obs)):
        print(f"t={t+1:2d}  ", end="")
        for i in range(hmm.n_states):
            print(f"{beta[t, i]:>12.8f}", end="")
        print()
    
    print("\nInterpretation:")
    print(f"  β_1(Sunny) = {beta[0, 0]:.6f}: Probability of future obs from Sunny at t=1")
    print(f"  β_1(Rainy) = {beta[0, 1]:.6f}: Probability of future obs from Rainy at t=1")
    print(f"  β_2(Sunny) = {beta[1, 0]:.6f}: = 1 (no future observations)")
    print(f"  β_2(Rainy) = {beta[1, 1]:.6f}: = 1 (no future observations)")

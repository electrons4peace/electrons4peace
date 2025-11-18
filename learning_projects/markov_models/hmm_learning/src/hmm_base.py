"""
Base Hidden Markov Model Class

This module defines the fundamental HMM structure and parameters.
It serves as the foundation for all HMM algorithms (Forward, Backward, Viterbi, Baum-Welch).

Mathematical Notation:
    N = number of hidden states
    M = number of possible observations
    T = length of observation sequence
    
    States: S = {s₁, s₂, ..., sₙ}
    Observations: O = {o₁, o₂, ..., oₘ}
    
    Model parameters λ = (A, B, π):
        A[i][j] = P(state_t+1 = j | state_t = i)  - Transition probabilities
        B[i][k] = P(observation = k | state = i)  - Emission probabilities
        π[i] = P(state_1 = i)                      - Initial state probabilities
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class HMM:
    """
    Hidden Markov Model base class.
    
    This class stores the model parameters and provides methods for:
    - Model initialization
    - Parameter validation
    - Basic probability computations
    
    Attributes:
        n_states (int): Number of hidden states (N)
        n_observations (int): Number of possible observations (M)
        state_names (List[str]): Names of states (for visualization)
        observation_names (List[str]): Names of observations (for visualization)
        A (np.ndarray): Transition probability matrix, shape (N, N)
        B (np.ndarray): Emission probability matrix, shape (N, M)
        pi (np.ndarray): Initial state probabilities, shape (N,)
    """
    
    def __init__(
        self,
        n_states: int,
        n_observations: int,
        state_names: Optional[List[str]] = None,
        observation_names: Optional[List[str]] = None
    ):
        """
        Initialize HMM with specified dimensions.
        
        Args:
            n_states: Number of hidden states
            n_observations: Number of possible observations
            state_names: Optional names for states (for debugging/visualization)
            observation_names: Optional names for observations
        """
        self.n_states = n_states
        self.n_observations = n_observations
        
        # Create default names if not provided
        if state_names is None:
            self.state_names = [f"S{i}" for i in range(n_states)]
        else:
            assert len(state_names) == n_states, "Number of state names must match n_states"
            self.state_names = state_names
            
        if observation_names is None:
            self.observation_names = [f"O{i}" for i in range(n_observations)]
        else:
            assert len(observation_names) == n_observations, "Number of observation names must match n_observations"
            self.observation_names = observation_names
        
        # Initialize parameters (will be set by user or learned)
        # Using None to indicate uninitialized - must be set before use
        self.A = None  # Transition matrix
        self.B = None  # Emission matrix
        self.pi = None  # Initial probabilities
        
    def set_transition_matrix(self, A: np.ndarray):
        """
        Set the state transition probability matrix.
        
        Mathematical definition:
            A[i][j] = P(q_t+1 = j | q_t = i)
            
        Constraints:
            - A must be shape (N, N)
            - Each row must sum to 1 (probability distribution)
            - All values must be in [0, 1]
        
        Args:
            A: Transition matrix, shape (N, N)
        """
        A = np.array(A, dtype=np.float64)
        
        # Validate shape
        assert A.shape == (self.n_states, self.n_states), \
            f"Transition matrix must be shape ({self.n_states}, {self.n_states}), got {A.shape}"
        
        # Validate probability constraints
        assert np.allclose(A.sum(axis=1), 1.0), \
            "Each row of transition matrix must sum to 1"
        assert np.all(A >= 0) and np.all(A <= 1), \
            "All transition probabilities must be in [0, 1]"
        
        self.A = A
        
    def set_emission_matrix(self, B: np.ndarray):
        """
        Set the observation emission probability matrix.
        
        Mathematical definition:
            B[i][k] = P(observation = k | state = i)
            
        Constraints:
            - B must be shape (N, M)
            - Each row must sum to 1 (probability distribution)
            - All values must be in [0, 1]
        
        Args:
            B: Emission matrix, shape (N, M)
        """
        B = np.array(B, dtype=np.float64)
        
        # Validate shape
        assert B.shape == (self.n_states, self.n_observations), \
            f"Emission matrix must be shape ({self.n_states}, {self.n_observations}), got {B.shape}"
        
        # Validate probability constraints
        assert np.allclose(B.sum(axis=1), 1.0), \
            "Each row of emission matrix must sum to 1"
        assert np.all(B >= 0) and np.all(B <= 1), \
            "All emission probabilities must be in [0, 1]"
        
        self.B = B
        
    def set_initial_probabilities(self, pi: np.ndarray):
        """
        Set the initial state probability distribution.
        
        Mathematical definition:
            π[i] = P(q_1 = i)
            
        Constraints:
            - π must be shape (N,)
            - Must sum to 1 (probability distribution)
            - All values must be in [0, 1]
        
        Args:
            pi: Initial state probabilities, shape (N,)
        """
        pi = np.array(pi, dtype=np.float64)
        
        # Validate shape
        assert pi.shape == (self.n_states,), \
            f"Initial probabilities must be shape ({self.n_states},), got {pi.shape}"
        
        # Validate probability constraints
        assert np.allclose(pi.sum(), 1.0), \
            "Initial probabilities must sum to 1"
        assert np.all(pi >= 0) and np.all(pi <= 1), \
            "All initial probabilities must be in [0, 1]"
        
        self.pi = pi
        
    def initialize_random(self, seed: Optional[int] = None):
        """
        Initialize model parameters randomly.
        
        This is useful for:
        - Testing algorithms
        - Starting point for Baum-Welch training
        
        Note: Random initialization may not be optimal for learning.
        Consider using domain knowledge when possible.
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Random transition matrix (each row sums to 1)
        A = np.random.random((self.n_states, self.n_states))
        A = A / A.sum(axis=1, keepdims=True)
        self.set_transition_matrix(A)
        
        # Random emission matrix (each row sums to 1)
        B = np.random.random((self.n_states, self.n_observations))
        B = B / B.sum(axis=1, keepdims=True)
        self.set_emission_matrix(B)
        
        # Random initial probabilities (sums to 1)
        pi = np.random.random(self.n_states)
        pi = pi / pi.sum()
        self.set_initial_probabilities(pi)
        
    def initialize_uniform(self):
        """
        Initialize model parameters uniformly.
        
        This gives equal probability to all transitions and emissions.
        Useful as a neutral starting point for learning.
        """
        # Uniform transition probabilities
        A = np.ones((self.n_states, self.n_states)) / self.n_states
        self.set_transition_matrix(A)
        
        # Uniform emission probabilities
        B = np.ones((self.n_states, self.n_observations)) / self.n_observations
        self.set_emission_matrix(B)
        
        # Uniform initial probabilities
        pi = np.ones(self.n_states) / self.n_states
        self.set_initial_probabilities(pi)
        
    def is_initialized(self) -> bool:
        """
        Check if all model parameters have been set.
        
        Returns:
            True if A, B, and π are all initialized, False otherwise
        """
        return (self.A is not None and 
                self.B is not None and 
                self.pi is not None)
    
    def validate(self):
        """
        Validate that the model is properly initialized and satisfies all constraints.
        
        Raises:
            AssertionError: If any validation check fails
        """
        assert self.is_initialized(), "Model parameters not initialized"
        
        # Validate transition matrix
        assert np.allclose(self.A.sum(axis=1), 1.0), \
            "Transition matrix rows must sum to 1"
        
        # Validate emission matrix
        assert np.allclose(self.B.sum(axis=1), 1.0), \
            "Emission matrix rows must sum to 1"
        
        # Validate initial probabilities
        assert np.allclose(self.pi.sum(), 1.0), \
            "Initial probabilities must sum to 1"
        
    def observation_to_index(self, observation) -> int:
        """
        Convert an observation name to its index.
        
        Args:
            observation: Observation name or index
            
        Returns:
            Index of the observation
        """
        if isinstance(observation, int):
            return observation
        else:
            return self.observation_names.index(observation)
    
    def state_to_index(self, state) -> int:
        """
        Convert a state name to its index.
        
        Args:
            state: State name or index
            
        Returns:
            Index of the state
        """
        if isinstance(state, int):
            return state
        else:
            return self.state_names.index(state)
    
    def get_observation_sequence_indices(self, observations: List) -> np.ndarray:
        """
        Convert a sequence of observations to indices.
        
        Args:
            observations: List of observation names or indices
            
        Returns:
            Array of observation indices
        """
        return np.array([self.observation_to_index(obs) for obs in observations])
    
    def get_state_sequence_indices(self, states: List) -> np.ndarray:
        """
        Convert a sequence of states to indices.
        
        Args:
            states: List of state names or indices
            
        Returns:
            Array of state indices
        """
        return np.array([self.state_to_index(state) for state in states])
    
    def __repr__(self) -> str:
        """String representation of the HMM."""
        status = "initialized" if self.is_initialized() else "uninitialized"
        return (f"HMM(n_states={self.n_states}, "
                f"n_observations={self.n_observations}, "
                f"status={status})")
    
    def print_parameters(self):
        """
        Print model parameters in a readable format.
        Useful for debugging and understanding the model.
        """
        if not self.is_initialized():
            print("Model not initialized")
            return
        
        print("=" * 60)
        print("Hidden Markov Model Parameters")
        print("=" * 60)
        
        print(f"\nStates: {self.state_names}")
        print(f"Observations: {self.observation_names}")
        
        print("\nInitial Probabilities (π):")
        for i, state in enumerate(self.state_names):
            print(f"  P({state}) = {self.pi[i]:.4f}")
        
        print("\nTransition Probabilities (A):")
        print("  From \\ To  ", end="")
        for state in self.state_names:
            print(f"{state:>8}", end="")
        print()
        for i, from_state in enumerate(self.state_names):
            print(f"  {from_state:>10}", end="")
            for j in range(self.n_states):
                print(f"{self.A[i, j]:>8.4f}", end="")
            print()
        
        print("\nEmission Probabilities (B):")
        print("  State \\ Obs", end="")
        for obs in self.observation_names:
            print(f"{obs:>8}", end="")
        print()
        for i, state in enumerate(self.state_names):
            print(f"  {state:>10}", end="")
            for j in range(self.n_observations):
                print(f"{self.B[i, j]:>8.4f}", end="")
            print()
        print("=" * 60)


# Example usage and testing
if __name__ == "__main__":
    print("Testing HMM Base Class\n")
    
    # Create a simple weather HMM
    print("1. Creating Weather HMM")
    hmm = HMM(
        n_states=2,
        n_observations=3,
        state_names=["Sunny", "Rainy"],
        observation_names=["Walk", "Shop", "Clean"]
    )
    print(f"   {hmm}\n")
    
    # Set parameters manually
    print("2. Setting parameters manually")
    
    # Transition matrix: 
    # From Sunny: 70% stay sunny, 30% to rainy
    # From Rainy: 40% to sunny, 60% stay rainy
    A = np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])
    hmm.set_transition_matrix(A)
    
    # Emission matrix:
    # Sunny: 60% walk, 30% shop, 10% clean
    # Rainy: 10% walk, 40% shop, 50% clean
    B = np.array([
        [0.6, 0.3, 0.1],
        [0.1, 0.4, 0.5]
    ])
    hmm.set_emission_matrix(B)
    
    # Initial: 60% sunny, 40% rainy
    pi = np.array([0.6, 0.4])
    hmm.set_initial_probabilities(pi)
    
    print("   Parameters set!\n")
    
    # Validate
    print("3. Validating model")
    hmm.validate()
    print("   ✓ Model is valid\n")
    
    # Print parameters
    print("4. Printing parameters")
    hmm.print_parameters()
    
    # Test observation conversion
    print("\n5. Testing observation conversion")
    observations = ["Walk", "Shop", "Clean"]
    indices = hmm.get_observation_sequence_indices(observations)
    print(f"   Observations: {observations}")
    print(f"   Indices: {indices}")
    
    # Test random initialization
    print("\n6. Testing random initialization")
    hmm2 = HMM(n_states=3, n_observations=4)
    hmm2.initialize_random(seed=42)
    print("   Random HMM initialized:")
    hmm2.print_parameters()

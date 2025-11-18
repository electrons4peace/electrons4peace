# 01. Hidden Markov Models - Basics

## What is a Hidden Markov Model?

A Hidden Markov Model (HMM) is a statistical model that describes a system that transitions between **hidden states** while producing **observable outputs**. The key insight is that we can only see the outputs, not the states themselves.

## Real-World Analogy: Weather Forecasting

Imagine you're locked in a room with no windows. Your friend outside observes the weather each day and tells you what activities they did:

- **Hidden States**: Actual weather (Sunny, Rainy, Cloudy)
- **Observations**: Activities your friend mentions (Walk, Shop, Clean)

You never directly observe the weather, but you can infer it from the activities.

## Formal Definition

An HMM is defined by five components: **λ = (S, O, A, B, π)**

### 1. States (S)
The hidden states the system can be in.

```
S = {s₁, s₂, ..., sₙ}
```

**Example (Weather):**
```
S = {Sunny, Rainy}
```

### 2. Observations (O)
The observable outputs the system can produce.

```
O = {o₁, o₂, ..., oₘ}
```

**Example (Weather):**
```
O = {Walk, Shop, Clean}
```

### 3. Initial State Probabilities (π)
The probability of starting in each state.

```
π = [π₁, π₂, ..., πₙ]
where πᵢ = P(state₁ = sᵢ)
```

**Example (Weather):**
```
π = [0.6, 0.4]  # 60% chance of starting sunny, 40% rainy
```

### 4. Transition Probabilities (A)
The probability of transitioning from one state to another.

```
A = [aᵢⱼ]ₙₓₙ
where aᵢⱼ = P(stateₜ₊₁ = sⱼ | stateₜ = sᵢ)
```

Each row must sum to 1 (it's a probability distribution).

**Example (Weather):**
```
A = [[0.8, 0.2],   # From Sunny: 80% stay sunny, 20% to rainy
     [0.4, 0.6]]   # From Rainy: 40% to sunny, 60% stay rainy
```

### 5. Emission Probabilities (B)
The probability of observing each output from each state.

```
B = [bᵢ(oₖ)]ₙₓₘ
where bᵢ(oₖ) = P(observation = oₖ | state = sᵢ)
```

Each row must sum to 1.

**Example (Weather):**
```
B = [[0.6, 0.3, 0.1],   # Sunny: 60% walk, 30% shop, 10% clean
     [0.1, 0.4, 0.5]]   # Rainy: 10% walk, 40% shop, 50% clean
```

## Key Assumptions

### 1. Markov Assumption
The future state depends only on the current state, not on past states.

```
P(stateₜ₊₁ | state₁, state₂, ..., stateₜ) = P(stateₜ₊₁ | stateₜ)
```

### 2. Output Independence Assumption
The current observation depends only on the current state.

```
P(observationₜ | state₁, ..., stateₜ, observation₁, ..., observationₜ₋₁) 
    = P(observationₜ | stateₜ)
```

## The Three Fundamental Problems

Working with HMMs requires solving three key problems:

### Problem 1: Evaluation
**Given:** A model λ = (A, B, π) and an observation sequence O
**Find:** P(O|λ) - the probability of seeing this sequence

**Why it matters:** Model comparison, anomaly detection

**Algorithm:** Forward Algorithm (see `docs/02_forward_algorithm.md`)

### Problem 2: Decoding
**Given:** A model λ and an observation sequence O
**Find:** The most likely state sequence that generated O

**Why it matters:** Understanding what the system was doing

**Algorithm:** Viterbi Algorithm (see `docs/04_viterbi_algorithm.md`)

### Problem 3: Learning
**Given:** An observation sequence O
**Find:** Model parameters λ that maximize P(O|λ)

**Why it matters:** Learning from data, training models

**Algorithm:** Baum-Welch Algorithm (see `docs/05_baum_welch_algorithm.md`)

## Notation Reference

Throughout this documentation, we use:

- **T**: Length of observation sequence
- **N**: Number of hidden states
- **M**: Number of possible observations
- **t**: Time index (1 to T)
- **i, j**: State indices
- **k**: Observation index

### Sequences
- **O** = (o₁, o₂, ..., oₜ): Observation sequence
- **Q** = (q₁, q₂, ..., qₜ): State sequence

### Probabilities
- **αₜ(i)**: Forward probability at time t in state i
- **βₜ(i)**: Backward probability at time t in state i
- **γₜ(i)**: Probability of being in state i at time t
- **ξₜ(i,j)**: Probability of transitioning from state i to j at time t

## Mathematical Properties

### Probability Constraints

All probability distributions must be valid:

```
# Initial probabilities sum to 1
Σᵢ πᵢ = 1

# Each row of transition matrix sums to 1
∀i: Σⱼ aᵢⱼ = 1

# Each row of emission matrix sums to 1
∀i: Σₖ bᵢ(oₖ) = 1
```

### Total Probability

The sum of probabilities over all possible state sequences equals the observation probability:

```
P(O|λ) = Σ_Q P(O, Q|λ)
       = Σ_Q P(O|Q,λ) · P(Q|λ)
```

## Speech Recognition Example

HMMs are widely used in speech recognition. Here's the mapping:

- **Hidden States**: Phonemes (basic speech sounds)
- **Observations**: Acoustic features (spectral characteristics)
- **Goal**: Given audio features, determine what phonemes (and thus words) were spoken

## Why "Hidden"?

The states are "hidden" because:

1. We can't directly observe them
2. Multiple state sequences could produce the same observations
3. We must infer states probabilistically from observations

## Visualization

```
Time:     t=1      t=2      t=3      t=4
          
States:   [S] ---> [S] ---> [R] ---> [R]
          ↓        ↓        ↓        ↓
Observe:  Walk     Shop     Clean    Clean

Hidden: We don't see this (Sunny, Sunny, Rainy, Rainy)
Visible: We only see this (Walk, Shop, Clean, Clean)
```

## Next Steps

Now that you understand the basics, proceed to:

1. **Forward Algorithm** (`docs/02_forward_algorithm.md`): Computing P(O|λ)
2. **Run Simple Example**: `python examples/simple_examples/weather_model.py`
3. **Study Base Class**: `src/hmm_base.py`

## Key Takeaways

✅ HMMs model systems with hidden states and observable outputs
✅ Defined by five components: states, observations, π, A, B
✅ Based on Markov and independence assumptions
✅ Three fundamental problems: evaluation, decoding, learning
✅ Widely used in speech recognition, bioinformatics, finance

---

**Next:** [Forward Algorithm →](02_forward_algorithm.md)

# HMM Learning Project

A pedagogical implementation of Hidden Markov Models (HMMs) designed for learning through hands-on coding.

## 🎯 Learning Objectives

By working through this project, you will:

1. **Understand HMM fundamentals**: States, observations, transition probabilities
2. **Master core algorithms**:
   - **Forward Algorithm**: Compute probability of observation sequence
   - **Backward Algorithm**: Complement to forward for complete probabilities
   - **Viterbi Algorithm**: Find most likely state sequence
   - **Baum-Welch Algorithm**: Learn HMM parameters from data (EM algorithm)
3. **Visualize algorithm execution**: See how algorithms work step-by-step
4. **Apply to real problems**: Speech recognition toy examples

## 🚀 Quick Start

### Installation

```bash
# Clone or create the project directory
mkdir hmm_learning
cd hmm_learning

# Install dependencies
pip install -r requirements.txt
```

### Your First HMM

```python
# Run a simple example
python examples/simple_examples/weather_model.py

# Try speech recognition
python examples/speech_recognition/phoneme_recognition.py
```

## 📚 Learning Roadmap

### Week 1: Foundations
- [ ] Read `docs/01_hmm_basics.md`
- [ ] Run `examples/simple_examples/coin_flips.py`
- [ ] Study `src/hmm_base.py` - understand HMM structure
- [ ] Visualize: `python visualization/state_diagram.py`

### Week 2: Forward & Backward Algorithms
- [ ] Read `docs/02_forward_algorithm.md`
- [ ] Study `src/forward.py` implementation
- [ ] Run with visualization: See probability calculations step-by-step
- [ ] Read `docs/03_backward_algorithm.md`
- [ ] Study `src/backward.py` implementation
- [ ] Compare forward and backward probabilities

### Week 3: Viterbi Algorithm
- [ ] Read `docs/04_viterbi_algorithm.md`
- [ ] Study `src/viterbi.py` implementation
- [ ] Run `examples/simple_examples/casino_dice.py`
- [ ] Visualize the trellis diagram
- [ ] Compare Viterbi path with forward probabilities

### Week 4: Baum-Welch Algorithm
- [ ] Read `docs/05_baum_welch_algorithm.md`
- [ ] Understand the EM algorithm framework
- [ ] Study `src/baum_welch.py` implementation
- [ ] Watch convergence plots in action
- [ ] Run speech recognition training

### Week 5: Advanced Topics
- [ ] Read `docs/06_convergence_analysis.md`
- [ ] Experiment with different initialization strategies
- [ ] Try speech recognition examples
- [ ] Modify examples to test understanding

## 🔬 Project Structure

```
hmm_learning/
├── docs/              # Algorithm theory and mathematics
├── src/               # Core algorithm implementations
├── visualization/     # Tools to visualize algorithm behavior
├── examples/          # Practical applications
│   ├── simple_examples/      # Start here!
│   └── speech_recognition/   # Realistic application
└── tests/             # Validation and verification
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed layout.

## 🎨 Visualization Features

This project emphasizes visual learning:

- **State Transition Diagrams**: See HMM structure graphically
- **Probability Matrices**: View transition and emission probabilities as heatmaps
- **Step-by-Step Execution**: Watch algorithms compute values iteratively
- **Convergence Plots**: Track Baum-Welch parameter updates over iterations
- **Trellis Diagrams**: Visualize Viterbi algorithm dynamic programming

## 📖 Key Concepts

### Hidden Markov Model Components

1. **States (S)**: Hidden states the system can be in
2. **Observations (O)**: Visible outputs we can see
3. **Initial Probabilities (π)**: Probability of starting in each state
4. **Transition Probabilities (A)**: Probability of moving between states
5. **Emission Probabilities (B)**: Probability of observing output from each state

### The Three Fundamental Problems

1. **Evaluation**: Given model λ=(A,B,π) and observations O, compute P(O|λ)
   - Solved by: **Forward Algorithm**

2. **Decoding**: Given model λ and observations O, find most likely state sequence
   - Solved by: **Viterbi Algorithm**

3. **Learning**: Given observations O, find model λ that maximizes P(O|λ)
   - Solved by: **Baum-Welch Algorithm**

## 🗣️ Speech Recognition Example

The included speech recognition examples demonstrate:

- **Phoneme Recognition**: Identifying basic speech sounds
- **Word Recognition**: Recognizing complete words from phoneme sequences
- Shows how HMMs model temporal sequences
- Demonstrates real-world application of all algorithms

## 💡 Code Philosophy

This project prioritizes **learning over performance**:

- ✅ Extensive inline comments explaining each step
- ✅ Clear variable names matching mathematical notation
- ✅ Step-by-step algorithm execution with intermediate results
- ✅ Assertions to verify correctness at each stage
- ❌ No aggressive optimizations that obscure logic
- ❌ No production-grade error handling (simple assertions instead)

## 🧪 Testing Your Understanding

After studying each algorithm:

1. Run the corresponding test in `tests/`
2. Modify example parameters and predict results
3. Implement your own toy example
4. Explain the algorithm to someone else (or to yourself!)

## 📊 Example Output

When you run an example with visualization:

```
=== Running Baum-Welch Algorithm ===

Iteration 1:
  Log-likelihood: -15.234
  Transition matrix:
    [[0.7  0.3 ]
     [0.4  0.6 ]]
  
Iteration 2:
  Log-likelihood: -12.456
  Δ = 2.778 (improved!)
  
... converged after 15 iterations

Final Model:
  States: ['Fair', 'Loaded']
  Observations: [1, 2, 3, 4, 5, 6]
  
[Convergence plot saved to output/convergence.png]
[State diagram saved to output/final_model.png]
```

## 🤝 Contributing to Your Learning

Tips for getting the most out of this project:

1. **Read the theory first**: Don't skip `docs/` - it provides essential context
2. **Run before reading code**: See the algorithm in action, then study implementation
3. **Add print statements**: Insert debug prints to see intermediate values
4. **Break things intentionally**: Modify parameters to see what happens
5. **Visualize everything**: Use the visualization tools liberally
6. **Start simple**: Master coin flips before tackling speech recognition

## 🔗 References & Further Reading

- Rabiner, L. (1989). "A Tutorial on Hidden Markov Models"
- Durbin et al. (1998). "Biological Sequence Analysis"
- Jurafsky & Martin. "Speech and Language Processing"

## 📝 Notes

- This is a **learning project**, not a production library
- Focus is on **understanding**, not performance
- All algorithms implemented from scratch (no sklearn/hmmlearn dependencies)
- Code matches mathematical notation from documentation

## ❓ Getting Help

If something is unclear:

1. Check the corresponding `docs/` file for theory
2. Look at `tests/` for expected behavior
3. Run with visualization enabled to see what's happening
4. Add print statements to trace execution

---

**Happy Learning! 🎓**

*Remember: The goal is to deeply understand HMMs by implementing them yourself.*

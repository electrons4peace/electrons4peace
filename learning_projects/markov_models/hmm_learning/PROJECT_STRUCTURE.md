# HMM Learning Project Structure

A pedagogical implementation of Hidden Markov Models focusing on understanding through implementation.

## Project Layout

```
hmm_learning/
├── README.md                          # Project overview and learning roadmap
├── PROJECT_STRUCTURE.md               # This file
├── requirements.txt                   # Python dependencies
│
├── docs/                              # Algorithm theory and documentation
│   ├── 01_hmm_basics.md              # Introduction to HMMs
│   ├── 02_forward_algorithm.md       # Forward algorithm theory
│   ├── 03_backward_algorithm.md      # Backward algorithm theory
│   ├── 04_viterbi_algorithm.md       # Viterbi algorithm theory
│   ├── 05_baum_welch_algorithm.md    # Baum-Welch algorithm theory
│   └── 06_convergence_analysis.md    # Understanding convergence
│
├── src/                               # Core implementation
│   ├── __init__.py
│   ├── hmm_base.py                   # Base HMM class with parameters
│   ├── forward.py                    # Forward algorithm implementation
│   ├── backward.py                   # Backward algorithm implementation
│   ├── viterbi.py                    # Viterbi algorithm implementation
│   ├── baum_welch.py                 # Baum-Welch algorithm implementation
│   └── utils.py                      # Helper functions
│
├── visualization/                     # Visualization tools
│   ├── __init__.py
│   ├── matrix_viz.py                 # Visualize probability matrices
│   ├── state_diagram.py              # Draw state transition diagrams
│   ├── convergence_plots.py          # Plot convergence curves
│   └── step_by_step.py               # Step-by-step algorithm visualization
│
├── examples/                          # Example applications
│   ├── __init__.py
│   ├── speech_recognition/           # Speech recognition examples
│   │   ├── __init__.py
│   │   ├── phoneme_recognition.py   # Simple phoneme HMM
│   │   ├── word_recognition.py      # Word-level HMM
│   │   └── data/                    # Toy speech data
│   │       ├── phonemes.txt
│   │       └── words.txt
│   │
│   ├── simple_examples/              # Basic learning examples
│   │   ├── coin_flips.py            # Classic coin flip example
│   │   ├── weather_model.py         # Weather prediction
│   │   └── casino_dice.py           # Dishonest casino
│   │
│   └── run_all_examples.py          # Script to run all examples
│
├── tests/                             # Example-driven validation
│   ├── __init__.py
│   ├── test_forward.py               # Validate forward algorithm
│   ├── test_backward.py              # Validate backward algorithm
│   ├── test_viterbi.py               # Validate Viterbi algorithm
│   ├── test_baum_welch.py            # Validate Baum-Welch algorithm
│   └── test_data/                    # Known test cases
│       └── benchmark_results.json
│
└── notebooks/                         # Optional Jupyter notebooks
    ├── 01_exploring_forward.ipynb
    ├── 02_exploring_backward.ipynb
    ├── 03_exploring_viterbi.ipynb
    └── 04_exploring_baum_welch.ipynb
```

## Learning Path

1. Start with `docs/01_hmm_basics.md` to understand theory
2. Read and run `examples/simple_examples/` to see HMMs in action
3. Study `src/` implementations with detailed comments
4. Run `examples/speech_recognition/` for realistic applications
5. Use `visualization/` tools to understand algorithm behavior
6. Review `tests/` to verify understanding

## Key Design Principles

- **Clarity over performance**: Code prioritizes readability
- **Detailed comments**: Each algorithm step is explained inline
- **Visual learning**: Extensive visualization support
- **Progressive complexity**: Simple examples → complex applications
- **Theory-practice connection**: Documentation links to code

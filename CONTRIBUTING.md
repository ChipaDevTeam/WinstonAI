# Contributing to WinstonAI

First off, thank you for considering contributing to WinstonAI! It's people like you that make WinstonAI such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by the [WinstonAI Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

**Bug Report Template:**
- **Description:** Clear and concise description of the bug
- **Steps to Reproduce:** Detailed steps to reproduce the behavior
- **Expected Behavior:** What you expected to happen
- **Actual Behavior:** What actually happened
- **Environment:**
  - OS: [e.g., Ubuntu 22.04, Windows 11]
  - Python version: [e.g., 3.10.5]
  - PyTorch version: [e.g., 2.0.1]
  - GPU: [e.g., RTX 3060 Ti]
  - CUDA version: [e.g., 11.8]
- **Logs:** Relevant error messages or logs
- **Additional Context:** Any other context about the problem

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear Title:** Use a descriptive title
- **Detailed Description:** Provide a detailed description of the suggested enhancement
- **Use Case:** Explain why this enhancement would be useful
- **Possible Implementation:** If you have ideas on how to implement it

### Pull Requests

1. **Fork the Repository**
   ```bash
   git clone https://github.com/ChipaDevTeam/WinstonAI.git
   cd WinstonAI
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

3. **Make Your Changes**
   - Write clear, concise commit messages
   - Follow the existing code style
   - Add or update tests as needed
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   # Run tests if available
   pytest tests/
   
   # Test your changes manually
   python src/your_modified_script.py
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature" # or "fix: resolve issue with..."
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your fork and branch
   - Fill out the PR template
   - Submit the pull request

## Development Setup

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (for GPU features)
- Git

### Setup Steps

1. **Clone and create virtual environment:**
   ```bash
   git clone https://github.com/ChipaDevTeam/WinstonAI.git
   cd WinstonAI
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .  # Install in editable mode
   ```

3. **Install development dependencies:**
   ```bash
   pip install pytest pytest-asyncio black flake8 mypy
   ```

4. **Run tests to verify setup:**
   ```bash
   pytest tests/ -v
   ```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

- **Line Length:** Maximum 100 characters (not 79)
- **Indentation:** 4 spaces (no tabs)
- **Quotes:** Double quotes for strings
- **Naming Conventions:**
  - Classes: `PascalCase`
  - Functions/Variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

### Code Formatting

We use `black` for automatic code formatting:

```bash
# Format all Python files
black src/

# Check formatting without making changes
black --check src/
```

### Type Hints

Use type hints for function signatures:

```python
def calculate_indicators(df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
    """Calculate technical indicators."""
    pass
```

### Documentation

- **Docstrings:** Use Google-style docstrings
- **Comments:** Write self-documenting code; use comments sparingly for complex logic
- **README Updates:** Update README.md if you add new features

**Docstring Example:**
```python
def train_model(data: pd.DataFrame, epochs: int = 100) -> torch.nn.Module:
    """
    Train the WinstonAI model on historical data.
    
    Args:
        data: Historical price data as a pandas DataFrame
        epochs: Number of training epochs (default: 100)
        
    Returns:
        Trained PyTorch model
        
    Raises:
        ValueError: If data is empty or invalid
        
    Example:
        >>> df = load_historical_data()
        >>> model = train_model(df, epochs=200)
    """
    pass
```

## Testing

### Writing Tests

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names

**Test Example:**
```python
def test_technical_indicators_calculation():
    """Test that technical indicators are calculated correctly."""
    df = create_sample_dataframe()
    indicators = calculate_indicators(df)
    
    assert "rsi" in indicators
    assert 0 <= indicators["rsi"] <= 100
    assert "macd" in indicators
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_indicators.py

# Run with verbose output
pytest -v
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, no code change)
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

**Examples:**
```
feat: add support for multiple trading pairs
fix: resolve memory leak in training loop
docs: update installation instructions
perf: optimize GPU memory usage
```

## Project Structure

```
WinstonAI/
├── src/                    # Source code
│   ├── train_*.py         # Training scripts
│   ├── live_*.py          # Trading bots
│   ├── gpu_*.py           # GPU utilities
│   └── *.json             # Configuration files
├── tests/                 # Test files (mirror src/ structure)
├── docs/                  # Documentation
├── examples/              # Example scripts
├── requirements.txt       # Dependencies
├── setup.py              # Package setup
└── README.md             # Main documentation
```

## Getting Help

- **Documentation:** Check the [README](README.md) and [docs](docs/) folder
- **Issues:** Search existing issues or create a new one
- **Discussions:** Use GitHub Discussions for questions

## Recognition

Contributors will be recognized in:
- The project README (Contributors section)
- GitHub contributors page
- Release notes (for significant contributions)

## License

By contributing to WinstonAI, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to WinstonAI! 🚀

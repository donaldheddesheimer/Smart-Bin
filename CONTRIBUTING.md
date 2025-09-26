# Contributing to Smart Bin

Thank you for your interest in contributing to the Smart Bin waste classification system! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of conduct that promotes a respectful and inclusive environment. By participating, you agree to uphold this standard.

## Getting Started

### Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/smart-bin.git
   cd smart-bin
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8 pre-commit
   ```

5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Write comprehensive docstrings for all functions and classes
- Maximum line length: 88 characters (Black formatter)
- Use type hints where applicable

Example:
```python
def classify_waste_image(image_path: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Classify a waste image and return prediction results.
    
    Args:
        image_path: Path to the input image file
        confidence_threshold: Minimum confidence threshold for predictions
        
    Returns:
        Dictionary containing classification results and confidence scores
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If image format is unsupported
    """
```

### Testing

- Write unit tests for all new functionality
- Maintain test coverage above 80%
- Use meaningful test names that describe what is being tested
- Test both success and failure cases

```python
def test_classify_waste_image_with_valid_input():
    """Test waste classification with valid image input."""
    # Test implementation
    
def test_classify_waste_image_with_invalid_file():
    """Test error handling when image file doesn't exist."""
    # Test implementation
```

### Documentation

- Update README.md for significant changes
- Add docstrings to all public functions and classes
- Include examples in documentation
- Update CHANGELOG.md for user-facing changes

## Contribution Process

### 1. Create an Issue

Before starting work, create an issue to discuss:
- Bug reports with reproduction steps
- Feature requests with use cases
- Performance improvements with benchmarks
- Documentation improvements

### 2. Create a Branch

Create a descriptive branch name:
```bash
git checkout -b feature/model-performance-improvements
git checkout -b bugfix/inference-memory-leak
git checkout -b docs/api-documentation-update
```

### 3. Make Changes

- Write clean, well-documented code
- Follow the established patterns in the codebase
- Add tests for new functionality
- Update documentation as needed

### 4. Test Your Changes

Run the full test suite:
```bash
# Run all tests
python -m pytest tests/ -v

# Check code coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run linting
flake8 src/ tests/
black --check src/ tests/

# Type checking (if using mypy)
mypy src/
```

### 5. Submit a Pull Request

- Use a clear, descriptive title
- Reference related issues in the description
- Include screenshots for UI changes
- List breaking changes if any

Pull Request Template:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Changes don't break existing functionality
```

## Types of Contributions

### Bug Fixes
- Provide clear reproduction steps
- Include error messages and stack traces
- Test the fix thoroughly
- Consider edge cases

### New Features
- Discuss the feature in an issue first
- Consider backwards compatibility
- Add comprehensive tests
- Update documentation

### Performance Improvements
- Include benchmarks showing improvement
- Ensure accuracy isn't significantly impacted
- Test with various input sizes
- Profile memory usage

### Documentation
- Fix typos and grammatical errors
- Improve clarity and examples
- Add missing documentation
- Update outdated information

## Coding Standards

### Python Style Guide

```python
# Good
def preprocess_image(image_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Preprocess image for model inference."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    return np.array(image)

# Bad
def preprocess(img):
    # No docstring, unclear parameter names
    i = Image.open(img)
    return np.array(i.resize((224, 224)))
```

### Error Handling

- Use specific exception types
- Provide meaningful error messages
- Log errors appropriately
- Handle edge cases gracefully

```python
# Good
try:
    result = model.predict(processed_image)
except tf.errors.ResourceExhaustedError:
    logger.error("GPU memory exhausted during inference")
    raise RuntimeError("Insufficient GPU memory for inference")
except Exception as e:
    logger.error(f"Unexpected error during inference: {str(e)}")
    raise

# Bad
try:
    result = model.predict(processed_image)
except:
    print("Error occurred")
```

### Logging

Use appropriate log levels:
```python
import logging

logger = logging.getLogger(__name__)

# Debug: Detailed diagnostic information
logger.debug(f"Processing image with shape: {image.shape}")

# Info: General information about program execution
logger.info("Model loaded successfully")

# Warning: Something unexpected happened, but program continues
logger.warning("Low confidence prediction: {confidence:.2f}")

# Error: Serious problem occurred
logger.error(f"Failed to load model: {error_message}")
```

## Review Process

### What Reviewers Look For

1. **Correctness**: Does the code work as intended?
2. **Style**: Does it follow project conventions?
3. **Tests**: Are there adequate tests?
4. **Documentation**: Is it well-documented?
5. **Performance**: Are there any performance implications?
6. **Security**: Are there any security concerns?

### Responding to Feedback

- Address all reviewer comments
- Ask questions if feedback isn't clear
- Make requested changes promptly
- Thank reviewers for their time

## Release Process

### Version Numbers
We use semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

### Changelog
Update CHANGELOG.md with:
- New features
- Bug fixes
- Breaking changes
- Performance improvements

## Getting Help

- Join discussions in GitHub issues
- Ask questions in pull requests
- Contact maintainers for guidance
- Check existing documentation and code examples

## Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes for significant contributions
- Hall of fame for major contributors

Thank you for contributing to Smart Bin!
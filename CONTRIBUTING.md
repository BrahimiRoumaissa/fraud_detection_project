# Contributing to Fraud Detection System

Thank you for your interest in contributing to the Fraud Detection System! This document provides guidelines for contributing to this project.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs or suggest features
- Provide clear descriptions and steps to reproduce issues
- Include relevant system information (OS, Python version, etc.)

### Making Changes
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes with clear, descriptive commit messages
4. Add tests if applicable
5. Ensure all tests pass
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Include type hints where appropriate

## 📋 Development Setup

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

### Testing
```bash
# Run tests
pytest

# Run linting
flake8 src/
black --check src/

# Run type checking
mypy src/
```

## 📝 Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** your changes thoroughly
5. **Update** documentation if needed
6. **Submit** a pull request with a clear description

### Pull Request Guidelines
- Use clear, descriptive titles
- Provide detailed descriptions of changes
- Reference any related issues
- Ensure CI checks pass
- Request reviews from maintainers

## 🎯 Areas for Contribution

### High Priority
- Performance optimizations
- Additional ML algorithms
- Enhanced visualization features
- Better error handling

### Medium Priority
- Documentation improvements
- Additional test coverage
- Code refactoring
- UI/UX enhancements

### Low Priority
- Additional language support
- Extended deployment options
- Additional data sources

## 📞 Getting Help

If you need help or have questions:
- Open a GitHub issue
- Contact: brahimiroumaissa1@gmail.com
- Check existing issues and discussions

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to make fraud detection more accessible and effective!

# How to Contribute

Thanks for your interest in contributing to `NetHeal`! Here are a few general
guidelines on contributing and reporting bugs that we ask you to review.
Following these guidelines helps to communicate that you respect the time of the
contributors managing and developing this open source project. In return, they
should reciprocate that respect in addressing your issue, assessing changes, and
helping you finalize your pull requests. In that spirit of mutual respect, we
endeavor to review incoming issues and pull requests within 10 days, and will
close any lingering issues or pull requests after 60 days of inactivity.

Please note that all of your interactions in the project are subject to our
[Code of Conduct](/CODE_OF_CONDUCT.md). This includes creation of issues or pull
requests, commenting on issues or pull requests, and extends to all interactions
in any real-time space e.g., Slack, Discord, etc.

## Table of Contents

- [Reporting Issues](#reporting-issues)
- [Development Setup](#development-setup)
- [Sending Pull Requests](#sending-pull-requests)
- [Code Style](#code-style)
- [Testing](#testing)
- [Other Ways to Contribute](#other-ways-to-contribute)

## Reporting Issues

Before reporting a new issue, please ensure that the issue was not already
reported or fixed by searching through our issues list.

When creating a new issue, please be sure to include a **title and clear
description**, as much relevant information as possible, and, if possible, a
test case.

**If you discover a security bug, please do not report it through GitHub.
Instead, please see security procedures in [SECURITY.md](/SECURITY.md).**

## Development Setup

### Prerequisites

- Python 3.9+
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/cisco-open/netheal-ai-agent-benchmark
   cd netheal-ai-agent-benchmark
   ```

2. **Create and activate virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation by running tests:**

   ```bash
   pytest
   ```

### Project Structure

```
netheal-ai-agent-benchmark/
├── netheal/                # Main package
│   ├── environment/        # RL environment components (env, actions, observations, rewards)
│   ├── network/            # Network graph and topology generation
│   ├── faults/             # Fault injection system
│   ├── tools/              # Diagnostic tool simulation
│   ├── hints/              # Natural language hint providers
│   ├── aaa/                # AAA protocol integration (A2A server, MCP tools)
│   └── evaluation/         # Metrics and evaluation utilities
├── webapp/                 # Web demo (FastAPI backend + frontend)
├── examples/               # Usage examples
├── tests/                  # Unit and integration tests
├── docs/                   # Documentation
└── scenarios/              # AAA scenario definitions
```

## Sending Pull Requests

Before sending a new pull request, take a look at existing pull requests and
issues to see if the proposed change or fix has been discussed in the past, or
if the change was already implemented but not yet released.

We expect new pull requests to include tests for any affected behavior, and, as
we follow semantic versioning, we may reserve breaking changes until the next
major version release.

### Pull Request Process

1. **Fork the repository** and create your branch from `main`.

2. **Make your changes** following the existing code style.

3. **Add or update tests** for any new functionality or bug fixes.

4. **Run the test suite** to ensure all tests pass:

   ```bash
   pytest
   ```

5. **Update documentation** if your changes affect the public API or user-facing features.

6. **Commit your changes** with a clear and descriptive commit message:

   ```bash
   git commit -m 'Add support for custom topology generators'
   ```

7. **Push to your fork** and submit a pull request.

### What We Look For

- **Clear problem statement**: Explain what problem your PR solves.
- **Minimal, focused changes**: Keep PRs small and focused on a single concern.
- **Tests**: New features should include tests; bug fixes should include regression tests.
- **Documentation**: Update docs if the public API changes.
- **Backwards compatibility**: Avoid breaking changes unless absolutely necessary.

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines.
- Use type hints for function signatures where practical.
- Write docstrings for public functions, classes, and modules.
- Keep lines under 100 characters where reasonable.
- Use meaningful variable and function names.

## Testing

NetHeal uses `pytest` for testing. Tests are located in the `tests/` directory.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=netheal

# Run specific test file
pytest tests/test_environment.py -v

# Run tests matching a pattern
pytest -k "test_action" -v
```

### Writing Tests

- Place tests in the `tests/` directory with filenames starting with `test_`.
- Use descriptive test function names that explain what is being tested.
- Include both positive and negative test cases.
- Test edge cases and error conditions.

## Other Ways to Contribute

We welcome anyone that wants to contribute to `NetHeal` to triage and reply to
open issues to help troubleshoot and fix existing bugs. Here is what you can do:

- **Help with issue triage**: Ensure existing issues follow the recommendations
  from the _[Reporting Issues](#reporting-issues)_ section, providing feedback
  to the issue's author on what might be missing.
- **Improve documentation**: Help clarify existing docs, add examples, or fix
  typos.
- **Review pull requests**: Test patches against real use cases and provide
  feedback.
- **Write tests**: Add missing test cases to improve coverage.
- **Share your use case**: Let us know how you're using NetHeal in your research
  or applications.

## Questions?

If you have questions about contributing, feel free to open an issue for
discussion.

Thanks again for your interest in contributing to `NetHeal`!

:heart:

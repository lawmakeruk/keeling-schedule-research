# Keeling Schedule Service

A microservice that generates Keeling schedules by combining existing legislation with amending bills using both traditional algorithms and LLM approaches. This service is part of the larger Lawmaker ecosystem but is designed to operate as a standalone service.

## Overview

The Keeling Schedule Service processes legislative documents to create consolidated versions showing prospective amendments. It receives XML inputs from Lawmaker containing both the existing Act and amending Bill, processes these using a combination of traditional algorithms and LLM techniques, and returns an amended version of the legislation.

### User Journey
1. Existing Act (XML) is identified in Lawmaker
2. Amending Bill (XML) is selected in Lawmaker
3. Both documents are sent to the Keeling Service API
4. Processed amendments are returned to Lawmaker
5. PDF version can be generated via the PDF Service

## Technology Stack

- **Framework:** Flask
- **Language:** Python 3.12
- **Code Quality:**
  - Black (formatting)
  - Flake8 (linting)
  - Pytest (testing)

## Project Structure

- `app/` - Source code for the Flask application
- `tests/` - Test suite and test resources
- `scripts/` - Utility scripts (e.g., setup.sh)

## Getting Started

Quick Start:
```bash
git clone <repository URL>
cd keeling-schedule
chmod +x scripts/setup.sh
./scripts/setup.sh
```
The setup script will:
* Install Python dependencies
* Run Black for code formatting
* Run Flake8 for linting
* Run pytest for unit tests with code coverage
* Start the Flask development server

### Manual Setup
If you need to run steps individually:

1. Install dependencies:
    ```
    python -m pip install -r requirements.txt
    ```
2. Run formatting:
    ```
    python -m black app/ tests
    ```

3. Run linting:
    ```
    python -m flake8 app/ tests/
    ```

4. Run tests:
    ```
    python -m pytest tests/
    ```

5. Run tests with code coverage:
    ```
      python -m pytest --cov=app tests/ --cov-report=term-missing --cov-report=html
    ```

6. Start development server:
    ```
    export FLASK_APP="app"
    export FLASK_ENV=development
    python -m flask run
    ```

## Contributing

* All code must be formatted using Black
* Tests must pass (pytest)
* Linting must pass (flake8)

## Testing Approach

* Unit Tests: pytest
* Coverage Reports: pytest-cov 
* Coverage Goals: Aiming for 80% coverage (provisional goal)
* Test Categories:
  - Unit tests for individual functions
  - Integration tests for XML processing
  - API endpoint tests
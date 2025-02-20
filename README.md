# Open Deep Research Python Implementation

This is a Python implementation of the Open Deep Research project originally created by dzhng (https://github.com/dzhng/deep-research). This implementation aims to replicate the functionality while leveraging Python's strengths for research and development.

## Installation

To install the project dependencies, you'll need Poetry. Here's how to set up the project:

1. Install Poetry if you haven't already:
   ```bash
   curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
   ```

2. Clone this repository:
   ```bash
   git clone [your-repository-url]
   cd Open-Deep-Research-Py
   ```

3. Install the dependencies:
   ```bash
   poetry install
   ```

## Usage

After installing the dependencies, you can run the main script:

```bash
poetry run python src/main.py
```

The script will perform the research operations as implemented in the original project. You can modify the parameters and configuration to suit your specific needs.

## Features

- Python implementation of the Open Deep Research functionality
- Logging system for tracking operations
- Modular structure for easy maintenance
- Poetry-based dependency management

## Configuration

The project uses environment files for configuration. You can modify the `.env.local` file to adjust settings as needed.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a Pull Request

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is based on the original work by dzhng (https://github.com/dzhng/deep-research). Special thanks to the original author for creating the foundation of this research tool.

# DataCruiser - RAG-based Footfall Analysis API

A Flask-based API for analyzing footfall data using RAG (Retrieval-Augmented Generation) with TimesFM embeddings and FAISS similarity search.

## 🏗️ Project Structure

```
datacruiser-server/
├── src/                           # Source code
│   └── datacruiser/              # Main package
│       ├── __init__.py
│       ├── app_factory.py        # Application factory
│       ├── config/               # Configuration management
│       │   └── __init__.py
│       ├── models/               # Data models
│       │   ├── __init__.py
│       │   ├── data_models.py    # FootfallRecord, DataStats, QueryParams
│       │   └── embedding_models.py # TimesFMModel, FallbackEmbeddingModel
│       ├── services/              # Business logic
│       │   ├── __init__.py
│       │   ├── model_service.py  # Model management
│       │   ├── rag_service.py    # RAG operations
│       │   └── llm_service.py     # LLM operations
│       ├── routes/                # Flask routes
│       │   ├── __init__.py
│       │   ├── api_routes.py     # Main API endpoints
│       │   ├── health_routes.py # Health check endpoints
│       │   └── cache_routes.py  # Cache management endpoints
│       └── utils/                 # Utility functions
│           ├── __init__.py
│           ├── cache_utils.py    # Cache management
│           ├── file_utils.py     # File operations
│           └── device_utils.py   # Device detection
├── data/                          # Data files
│   └── data.json                 # Footfall data
├── cache/                         # Cache directory (auto-created)
├── logs/                          # Log files (auto-created)
├── tests/                         # Test files
├── run.py                         # Main entry point
├── setup.py                       # Package setup
├── pyproject.toml                 # Modern Python packaging
├── requirements.txt               # Dependencies
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd datacruiser-server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### 2. Configuration

```bash
# Set environment variables
export OPENAI_API_KEY="your-openai-api-key"

# Optional environment variables
export DEBUG=false
export HOST=0.0.0.0
export PORT=5080
export LOG_LEVEL=INFO
```

### 3. Run the Application

```bash
# Method 1: Using the run script
python run.py

# Method 2: Using the package directly
python -m datacruiser.app_factory

# Method 3: Using the console script (after pip install -e .)
datacruiser
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check and system status |
| `POST` | `/api/ask` | Ask questions about footfall data |
| `GET` | `/data/info` | Get data and cache information |
| `POST` | `/cache/clear` | Clear all cache files |
| `GET` | `/cache/status` | Get detailed cache status |

## 🔧 Usage Examples

### Ask a Question

```bash
curl -X POST http://localhost:5080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the busiest locations?"}'
```

### Check System Status

```bash
curl http://localhost:5080/
```

### Clear Cache

```bash
curl -X POST http://localhost:5080/cache/clear
```

## 🏗️ Architecture

### Key Components

1. **Config**: Centralized configuration management
2. **Models**: Data structures and embedding models
3. **Services**: Business logic (RAG, LLM, Model services)
4. **Routes**: HTTP request handling
5. **Utils**: Reusable utility functions

### Design Patterns

- **Factory Pattern**: Application factory for creating the Flask app
- **Service Layer**: Business logic separated from HTTP handling
- **Repository Pattern**: Data access abstraction
- **Dependency Injection**: Services injected where needed

## 🧪 Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=datacruiser
```

### Code Quality

```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
mypy src/
```

### Adding New Features

1. **New API endpoints**: Add to `src/datacruiser/routes/`
2. **New services**: Add to `src/datacruiser/services/`
3. **New models**: Add to `src/datacruiser/models/`
4. **New utilities**: Add to `src/datacruiser/utils/`

## 📁 Data Management

### Data Directory Structure

```
data/
└── data.json          # Main footfall data file
```

### Cache Directory Structure

```
cache/
├── timesfm_model.pkl      # Cached model (if applicable)
├── embeddings.npy         # Generated embeddings
├── faiss_index.pkl        # FAISS similarity index
├── processed_data.pkl     # Processed records
├── data_stats.pkl         # Data statistics
└── cache_metadata.json   # Cache metadata
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `DEBUG` | Enable debug mode | `false` |
| `HOST` | Server host | `0.0.0.0` |
| `PORT` | Server port | `5080` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Configuration Class

All configuration is managed through the `Config` class in `src/datacruiser/config/__init__.py`:

```python
from datacruiser.config import Config

# Access configuration
print(Config.HOST)
print(Config.PORT)
```

## 🚀 Deployment

### Production Deployment

```bash
# Install production dependencies
pip install -r requirements.txt

# Set production environment variables
export OPENAI_API_KEY="your-api-key"
export DEBUG=false
export LOG_LEVEL=INFO

# Run the application
python run.py
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

EXPOSE 5080
CMD ["python", "run.py"]
```

## 📈 Performance

### Caching Strategy

- **Aggressive caching**: All processed data is cached
- **Cache validation**: File hash-based cache invalidation
- **Memory management**: Proper cleanup and garbage collection
- **Batch processing**: Embeddings generated in batches

### Optimization

- **Device detection**: Automatic CPU/GPU optimization
- **Batch processing**: Efficient embedding generation
- **Memory cleanup**: Regular garbage collection
- **Fallback mechanisms**: Graceful degradation on failures

## 🔍 Monitoring

### Health Checks

- **System status**: `/` endpoint
- **Cache status**: `/cache/status` endpoint
- **Data information**: `/data/info` endpoint

### Logging

- **Structured logging**: JSON-formatted logs
- **Log levels**: DEBUG, INFO, WARNING, ERROR
- **Log rotation**: Automatic log file management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the documentation
2. Search existing issues
3. Create a new issue with detailed information
4. Contact the development team

---

**DataCruiser** - Making footfall data analysis intelligent and accessible! 🚀

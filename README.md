# ML Inference Pipeline

A production-ready machine learning inference system using TensorFlow Serving and Docker. This project demonstrates containerized ML model deployment with a REST API gateway, health monitoring, and metrics collection.

## 🎯 Project Overview

This system deploys a MobileNetV2 image classification model using modern MLOps practices:
- **TensorFlow Serving** for scalable model serving
- **FastAPI Gateway** for preprocessing and API management
- **Docker Compose** for orchestration
- **Health checks** and monitoring endpoints
- **Model versioning** support

## 🏗️ Architecture
```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   Client    │─────▶│   Gateway    │─────▶│  TF Serving     │
│             │      │  (FastAPI)   │      │  (MobileNetV2)  │
└─────────────┘      └──────────────┘      └─────────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │  Monitoring  │
                     │   Metrics    │
                     └──────────────┘
```

**Components:**
- **Gateway Service**: FastAPI application handling image preprocessing, validation, and response formatting
- **TensorFlow Serving**: High-performance model serving with REST API
- **Docker Network**: Isolated bridge network for service communication

## 📋 Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose)
- Python 3.12+ (for model export only)
- curl (for testing)

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd ml-inference-pipeline
```

### 2. Export Model (First Time Only)
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-export.txt

# Export MobileNetV2 model
python export_model.py

# Deactivate venv
deactivate
```

### 3. Start Services
```bash
docker-compose up --build
```

Wait for:
- ✅ `Successfully loaded servable version {name: mobilenet version: 1}`
- ✅ `Application startup complete`

### 4. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST -F "file=@test/cat.jpg" http://localhost:8000/predict

# View metrics
curl http://localhost:8000/metrics
```

## 📡 API Endpoints

### GET /

Returns API information and available endpoints

### GET /health

Health check endpoint - verifies gateway and TF Serving connectivity

**Response:**
```json
{
  "status": "healthy",
  "gateway": "operational",
  "tf_serving": "connected",
  "model": {...}
}
```

### POST /predict

Image classification endpoint

**Request:**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
```

**Response:**
```json
{
  "success": true,
  "filename": "cat.jpg",
  "original_size": "480x360",
  "inference_time_ms": 45.2,
  "predictions": [
    {
      "class_id": 285,
      "probability": 0.8523,
      "confidence_percent": 85.23
    }
  ],
  "model": {
    "name": "mobilenet",
    "version": "1"
  }
}
```

### GET /metrics

Service metrics and statistics

**Response:**
```json
{
  "service": "ml-inference-gateway",
  "model": "mobilenet",
  "model_version": "1",
  "requests_total": 42,
  "errors_total": 2,
  "success_rate": 95.24,
  "uptime_seconds": 3600.5,
  "uptime_hours": 1.0
}
```

### GET /model

TensorFlow Serving model metadata

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Model Serving | TensorFlow Serving 2.x | High-performance model inference |
| API Gateway | FastAPI 0.104+ | Request handling and preprocessing |
| ML Framework | TensorFlow 2.18+ | Model format and operations |
| Model Source | TensorFlow Hub | Pre-trained MobileNetV2 |
| Containerization | Docker & Docker Compose | Service orchestration |
| Image Processing | Pillow, NumPy | Preprocessing pipeline |

## 📁 Project Structure
```
ml-inference-pipeline/
├── docker-compose.yml          # Service orchestration
├── export_model.py             # Model export script
├── requirements-export.txt     # Python dependencies for export
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── models/
│   └── mobilenet/
│       └── 1/                  # Model version 1
│           ├── saved_model.pb
│           ├── variables/
│           └── assets/
│
├── gateway/
│   ├── Dockerfile              # Gateway container definition
│   ├── app.py                  # FastAPI application
│   └── requirements.txt        # Gateway dependencies
│
└── test/
    └── cat.jpg                 # Sample test image
```

## 🔧 Configuration

### Model Versioning

To add a new model version:

1. Export new model to `models/mobilenet/2/`
2. TensorFlow Serving automatically detects new versions
3. Update API calls to specify version if needed

### Environment Variables

**Gateway:**
- `TF_SERVING_URL`: TensorFlow Serving endpoint (default: `http://tf-serving:8501/v1/models/mobilenet:predict`)

**TensorFlow Serving:**
- `MODEL_NAME`: Model name (default: `mobilenet`)

### Port Configuration

- **8000**: Gateway REST API
- **8501**: TensorFlow Serving REST API
- **8500**: TensorFlow Serving gRPC API

## 🧪 Testing

### Unit Test Example
```bash
# Test with different images
curl -X POST -F "file=@test/dog.jpg" http://localhost:8000/predict
curl -X POST -F "file=@test/car.jpg" http://localhost:8000/predict
```

### Load Testing
```bash
# Simple load test with Apache Bench
ab -n 100 -c 10 http://localhost:8000/health
```

## 🚨 Troubleshooting

### Containers Won't Start
```bash
# Check logs
docker-compose logs

# Check specific service
docker-compose logs tf-serving
docker-compose logs gateway

# Rebuild from scratch
docker-compose down
docker-compose up --build
```

### Model Not Loading
```bash
# Verify model files exist
ls -la models/mobilenet/1/

# Check TensorFlow Serving logs
docker-compose logs tf-serving | grep "Successfully loaded"
```

### Gateway Can't Connect to TF Serving
```bash
# Test TF Serving directly
curl http://localhost:8501/v1/models/mobilenet

# Check Docker network
docker network inspect ml-inference-pipeline_ml-network
```

### Prediction Errors
```bash
# Check image format
file test/cat.jpg

# Verify image is valid
python -c "from PIL import Image; Image.open('test/cat.jpg').show()"

# Check gateway logs
docker-compose logs gateway
```

## 📊 Production Considerations

### Scaling
- Add load balancer (nginx/traefik) in front of gateway
- Run multiple gateway replicas
- Use Kubernetes for orchestration at scale

### Monitoring
- Integrate Prometheus for metrics collection
- Add Grafana dashboards for visualization
- Implement distributed tracing (Jaeger/Zipkin)

### Security
- Add authentication (API keys, OAuth)
- Enable HTTPS/TLS
- Implement rate limiting
- Add input validation and sanitization

### Performance
- Enable GPU support in TensorFlow Serving
- Implement request batching
- Add caching layer (Redis) for frequent predictions
- Use gRPC instead of REST for better performance

### Model Management
- Implement A/B testing between model versions
- Add model performance tracking
- Set up automated model retraining pipeline
- Version control models with DVC or MLflow

## 🔄 CI/CD Pipeline (Future)
```yaml
# Example GitHub Actions workflow
- Build and test Docker images
- Run integration tests
- Push to container registry
- Deploy to staging
- Run smoke tests
- Deploy to production
```

## 📝 Model Information

**MobileNetV2:**
- Input: 224x224 RGB images
- Output: 1001 ImageNet class probabilities
- Architecture: Efficient convolutional neural network
- Use case: General image classification

**ImageNet Classes:**
The model classifies images into 1000 object categories (plus background). Common classes include animals, vehicles, household objects, and more.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Your Name**
- GitHub: [@andrelsyoo](https://github.com/andrelsyoo)


## 🙏 Acknowledgments

- TensorFlow team for TensorFlow Serving
- Google for MobileNetV2 and TensorFlow Hub
- FastAPI team for the excellent web framework

---

**Built for test MLOps practices**
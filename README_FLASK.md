# GRPO Fine-Tuner Flask Server

Web-based interface for GRPO (Group Relative Policy Optimization) fine-tuning with Flask backend and real-time updates.

## Features

### Web Interface
- **Modern Browser-Based UI**: Access training from any device
- **Real-Time Updates**: Live training metrics via WebSocket
- **Multi-Session Support**: Train multiple models simultaneously
- **Responsive Design**: Works on desktop and mobile devices

### API Endpoints
- RESTful API for all training operations
- WebSocket support for real-time metrics
- Session management for concurrent users
- Configuration validation and storage

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM minimum

### Setup

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Create necessary directories**:
```bash
mkdir -p configs logs outputs checkpoints exports cache templates static
```

## Running the Server

### Development Mode
```bash
python flask_app.py
```
Access the web interface at `http://localhost:5000`

### Production Mode
```bash
# Using Gunicorn with Socket.IO support
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 flask_app:app
```

### Environment Variables
```bash
# Optional configuration
export FLASK_SECRET_KEY="your-secret-key-here"  # Change in production
export PORT=5000  # Server port
export FLASK_ENV=development  # or 'production'
```

## Usage

### Web Interface

1. **Open Browser**: Navigate to `http://localhost:5000`
2. **Configure Training**:
   - Select model from available options
   - Configure dataset source and parameters
   - Adjust training hyperparameters
3. **Start Training**: Click "Start Training" button
4. **Monitor Progress**: Real-time metrics in Monitoring tab
5. **Export Model**: Download trained model when complete

### API Usage

#### Start Training
```bash
curl -X POST http://localhost:5000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "unsloth/Qwen3-0.6B",
    "dataset_path": "tatsu-lab/alpaca",
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 0.0002
  }'
```

#### Check Status
```bash
curl http://localhost:5000/api/training/{session_id}/status
```

#### Get Metrics
```bash
curl http://localhost:5000/api/training/{session_id}/metrics
```

## API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/system/info` | GET | System information |
| `/api/models` | GET | Available models list |
| `/api/config/validate` | POST | Validate configuration |
| `/api/config/save` | POST | Save configuration |
| `/api/config/load/{filename}` | GET | Load configuration |
| `/api/training/start` | POST | Start training session |
| `/api/training/{id}/status` | GET | Get session status |
| `/api/training/{id}/stop` | POST | Stop training |
| `/api/training/{id}/metrics` | GET | Get training metrics |
| `/api/training/{id}/logs` | GET | Get training logs |
| `/api/training/sessions` | GET | List all sessions |
| `/api/export/{id}` | POST | Export trained model |

### WebSocket Events

#### Client → Server
- `connect`: Establish connection
- `join_session`: Join training session updates
- `leave_session`: Leave session updates
- `request_update`: Request status update

#### Server → Client
- `training_progress`: Training progress (0-1)
- `training_metrics`: Current metrics object
- `training_log`: Log message
- `training_complete`: Training finished
- `training_error`: Error occurred

## Configuration

### Example Configuration
```json
{
  "model_name": "unsloth/Qwen3-0.6B",
  "dataset_source": "huggingface",
  "dataset_path": "tatsu-lab/alpaca",
  "dataset_split": "train[:100]",
  "instruction_field": "instruction",
  "response_field": "output",
  "learning_rate": 0.0002,
  "batch_size": 4,
  "num_epochs": 3,
  "temperature": 0.7,
  "top_p": 0.95,
  "lora_rank": 16,
  "lora_alpha": 32,
  "use_flash_attention": false,
  "gradient_checkpointing": false,
  "mixed_precision": true
}
```

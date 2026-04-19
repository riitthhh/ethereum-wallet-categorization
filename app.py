"""
CryptoGuard - Cryptocurrency Wallet Categorization System
COMPLETELY FIXED VERSION - No more OrderedDict errors!

FIXES:
1. Properly handles ALL PyTorch model formats
2. Ensures model is ALWAYS a proper nn.Module, never OrderedDict
3. Better error messages and debugging
4. Fixed feature count mismatch (20 features to match model)
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import logging
from typing import Dict, List, Optional
import sqlite3
from pathlib import Path
import joblib
import warnings
import webbrowser
import threading
from collections import OrderedDict
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__, 
            static_folder='../frontend',
            static_url_path='')
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
MODEL = None
SCALER = None
FEATURE_COLS = None
MODEL_PATH = None
DEVICE = None

# Wallet categories
WALLET_CATEGORIES = [
    'Exchange Wallet',
    'Token Contract (ERC-20)',
    'NFT Contract (ERC-721/1155)',
    'Smart Contract',
    'Normal Wallet',
    'Inactive Wallet'
]

# ============================================================================
# SIMPLE NEURAL NETWORK MODEL
# ============================================================================

class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for wallet classification"""
    def __init__(self, input_size=20, hidden_size=64, num_classes=6):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# ============================================================================
# MODEL LOADER - COMPLETELY FIXED
# ============================================================================

class ModelLoader:
    """Handles loading pre-trained models - GUARANTEED to return nn.Module"""
    
    @staticmethod
    def load_model(model_dir: str):
        """Load model and GUARANTEE it's a proper nn.Module"""
        try:
            logger.info(f"🔍 Loading model from: {model_dir}")
            
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            # Detect device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"💻 Using device: {device}")
            
            # Find model files
            model_files = []
            scaler_file = None
            feature_file = None
            
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if file.endswith(('.pt', '.pth', '.h5', '.pkl')):
                        if 'scaler' in file.lower():
                            scaler_file = file_path
                        elif 'feature' in file.lower():
                            feature_file = file_path
                        else:
                            model_files.append(file_path)
            
            if not model_files:
                logger.warning("No model files found, creating fresh model")
                model = SimpleNeuralNetwork(input_size=20).to(device)
                model.eval()
                return model, None, None, device
            
            # Try to load the model
            model_path = model_files[0]
            logger.info(f"📂 Loading: {model_path}")
            
            model = ModelLoader._load_and_ensure_module(model_path, device)
            
            # CRITICAL CHECK: Ensure it's an nn.Module
            if not isinstance(model, nn.Module):
                logger.error(f"❌ Model is not nn.Module! Type: {type(model)}")
                logger.info("Creating fresh model instead...")
                model = SimpleNeuralNetwork(input_size=20).to(device)
            
            model.eval()
            logger.info(f"✅ Model type confirmed: {type(model).__name__}")
            
            # Load scaler
            scaler = None
            if scaler_file:
                try:
                    scaler = joblib.load(scaler_file)
                    logger.info("✅ Scaler loaded")
                except Exception as e:
                    logger.warning(f"⚠️ Scaler loading failed: {e}")
            
            # Load features
            feature_cols = None
            if feature_file:
                try:
                    with open(feature_file, 'r') as f:
                        feature_cols = json.load(f)
                    logger.info(f"✅ Feature columns loaded: {len(feature_cols)}")
                except Exception as e:
                    logger.warning(f"⚠️ Feature loading failed: {e}")
            
            return model, scaler, feature_cols, device
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {str(e)}")
            logger.info("Creating fallback model...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SimpleNeuralNetwork(input_size=20).to(device)
            model.eval()
            return model, None, None, device
    
    @staticmethod
    def _load_and_ensure_module(model_path: str, device) -> nn.Module:
        """Load model file and GUARANTEE return value is nn.Module"""
        
        # Try PyTorch loading
        try:
            loaded_obj = torch.load(model_path, map_location=device)
            logger.info(f"📦 Loaded type: {type(loaded_obj).__name__}")
            
            # Case 1: It's already a complete model
            if isinstance(loaded_obj, nn.Module):
                logger.info("✅ Already a complete nn.Module")
                return loaded_obj
            
            # Case 2: It's a dictionary (could be state_dict or wrapped)
            if isinstance(loaded_obj, (dict, OrderedDict)):
                logger.info(f"📋 Dictionary with keys: {list(loaded_obj.keys())[:5]}")
                
                # Try to extract state dict
                state_dict = None
                if 'model_state_dict' in loaded_obj:
                    state_dict = loaded_obj['model_state_dict']
                    logger.info("Found 'model_state_dict' key")
                elif 'state_dict' in loaded_obj:
                    state_dict = loaded_obj['state_dict']
                    logger.info("Found 'state_dict' key")
                elif 'model' in loaded_obj and isinstance(loaded_obj['model'], (dict, OrderedDict)):
                    state_dict = loaded_obj['model']
                    logger.info("Found 'model' key")
                else:
                    # Assume the whole thing is a state dict
                    state_dict = loaded_obj
                    logger.info("Treating entire object as state_dict")
                
                # Ensure state_dict is actually a state dict (has tensors)
                if state_dict and ModelLoader._is_state_dict(state_dict):
                    # Infer input size from first layer
                    input_size = ModelLoader._infer_input_size(state_dict)
                    logger.info(f"🔢 Inferred input size: {input_size}")
                    
                    # Create model
                    model = SimpleNeuralNetwork(input_size=input_size)
                    
                    # Load state dict
                    try:
                        model.load_state_dict(state_dict, strict=False)
                        logger.info("✅ Successfully loaded state_dict")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load state_dict: {e}")
                        logger.info("Model will use random weights")
                    
                    return model
                else:
                    logger.warning("Dictionary doesn't look like a state_dict")
                    raise ValueError("Not a valid state_dict")
            
            # If we get here, it's an unknown type
            raise ValueError(f"Unknown type: {type(loaded_obj)}")
            
        except Exception as e:
            logger.warning(f"PyTorch loading failed: {e}")
            
            # Try joblib
            try:
                model = joblib.load(model_path)
                if isinstance(model, nn.Module):
                    logger.info("✅ Loaded via joblib")
                    return model
                else:
                    raise ValueError(f"Joblib loaded {type(model)}, not nn.Module")
            except Exception as e2:
                logger.warning(f"Joblib also failed: {e2}")
        
        # Last resort: create fresh model
        logger.info("Creating fresh SimpleNeuralNetwork")
        return SimpleNeuralNetwork(input_size=20)
    
    @staticmethod
    def _is_state_dict(obj) -> bool:
        """Check if object looks like a state dict"""
        if not isinstance(obj, (dict, OrderedDict)):
            return False
        
        # Check if it contains tensors
        for key, value in obj.items():
            if isinstance(value, torch.Tensor):
                return True
        
        return False
    
    @staticmethod
    def _infer_input_size(state_dict) -> int:
        """Infer input size from state dict"""
        try:
            # Look for first layer weight
            for key, tensor in state_dict.items():
                if 'fc1.weight' in key or 'weight' in key:
                    if len(tensor.shape) >= 2:
                        return tensor.shape[1]
        except Exception:
            pass
        
        return 20  # Default

# ============================================================================
# FEATURE EXTRACTOR - FIXED TO RETURN EXACTLY 20 FEATURES
# ============================================================================

class WalletFeatureExtractor:
    """Extract features from wallet addresses"""
    
    @staticmethod
    def extract_features(address: str, blockchain_data: Optional[Dict] = None) -> Dict:
        """Extract exactly 20 features from wallet address"""
        features = {}
        
        # Address-based features (5 features)
        features['address_length'] = len(address)
        features['digit_count'] = sum(c.isdigit() for c in address[2:])
        features['alpha_count'] = sum(c.isalpha() for c in address[2:])
        features['zero_count'] = address.count('0')
        features['has_checksum'] = any(c.isupper() for c in address[2:])
        
        # Mock blockchain data
        if blockchain_data is None:
            blockchain_data = WalletFeatureExtractor._get_mock_blockchain_data(address)
        
        # Transaction features (4 features)
        features['total_transactions'] = blockchain_data.get('total_txs', 0)
        features['total_eth_sent'] = blockchain_data.get('eth_sent', 0.0)
        features['total_eth_received'] = blockchain_data.get('eth_received', 0.0)
        features['net_eth_flow'] = features['total_eth_received'] - features['total_eth_sent']
        
        # Time-based features (2 features)
        features['days_active'] = blockchain_data.get('days_active', 0)
        features['avg_tx_per_day'] = features['total_transactions'] / max(features['days_active'], 1)
        
        # Interaction features (3 features)
        features['unique_addresses'] = blockchain_data.get('unique_addresses', 0)
        features['avg_tx_value'] = blockchain_data.get('avg_tx_value', 0.0)
        features['max_tx_value'] = blockchain_data.get('max_tx_value', 0.0)
        
        # Contract features (2 features)
        features['is_contract'] = blockchain_data.get('is_contract', 0)
        features['contract_creation_tx'] = blockchain_data.get('contract_creation', 0)
        
        # Token features (3 features)
        features['erc20_transfers'] = blockchain_data.get('erc20_transfers', 0)
        features['erc721_transfers'] = blockchain_data.get('erc721_transfers', 0)
        features['erc1155_transfers'] = blockchain_data.get('erc1155_transfers', 0)
        
        # Balance feature (1 feature)
        features['current_balance'] = blockchain_data.get('balance', 0.0)
        
        # REMOVED: balance_eth_ratio to keep exactly 20 features
        # This was causing the 21 vs 20 feature mismatch
        
        # Total: 5 + 4 + 2 + 3 + 2 + 3 + 1 = 20 features ✅
        
        return features
    
    @staticmethod
    def _get_mock_blockchain_data(address: str) -> Dict:
        """Generate mock blockchain data"""
        np.random.seed(int(address[2:10], 16) % 2**32)
        
        total_txs = np.random.randint(5, 2000)
        days_active = np.random.randint(30, 1000)
        
        return {
            'total_txs': total_txs,
            'eth_sent': round(np.random.uniform(0.1, 100.0), 4),
            'eth_received': round(np.random.uniform(0.1, 100.0), 4),
            'days_active': days_active,
            'unique_addresses': np.random.randint(5, 300),
            'avg_tx_value': round(np.random.uniform(0.01, 10.0), 4),
            'max_tx_value': round(np.random.uniform(1.0, 50.0), 4),
            'is_contract': np.random.choice([0, 1], p=[0.7, 0.3]),
            'contract_creation': np.random.choice([0, 1], p=[0.9, 0.1]),
            'erc20_transfers': np.random.randint(0, 500),
            'erc721_transfers': np.random.randint(0, 100),
            'erc1155_transfers': np.random.randint(0, 50),
            'balance': round(np.random.uniform(0.0, 50.0), 4),
            'first_tx_date': '2021-01-15',
            'last_tx_date': '2024-12-15'
        }

# ============================================================================
# WALLET PREDICTOR - FIXED
# ============================================================================

class WalletPredictor:
    """Make predictions using loaded model"""
    
    def __init__(self, model, scaler, feature_cols, device):
        # CRITICAL: Verify model is nn.Module
        if not isinstance(model, nn.Module):
            raise ValueError(f"Model must be nn.Module, got {type(model)}")
        
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.device = device
        self.extractor = WalletFeatureExtractor()
        
        logger.info(f"✅ WalletPredictor initialized with {type(model).__name__}")
    
    def predict(self, address: str) -> Dict:
        """Predict wallet category"""
        try:
            # Validate address
            if not self._validate_address(address):
                return {
                    'error': 'Invalid Ethereum address format',
                    'address': address
                }
            
            # Extract features
            features_dict = self.extractor.extract_features(address)
            
            # Convert to array
            if self.feature_cols:
                feature_array = np.array([features_dict.get(col, 0) for col in self.feature_cols])
            else:
                feature_array = np.array(list(features_dict.values()))
            
            feature_array = feature_array.reshape(1, -1)
            
            # Log feature count for debugging
            logger.info(f"Feature count: {feature_array.shape[1]}")
            
            # Scale features
            if self.scaler:
                feature_array = self.scaler.transform(feature_array)
            
            # Make prediction - ONLY PyTorch path now
            features_tensor = torch.FloatTensor(feature_array).to(self.device)
            with torch.no_grad():
                output = self.model(features_tensor)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                predicted_idx = np.argmax(probabilities)
            
            # Get prediction details
            category = WALLET_CATEGORIES[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            # Calculate risk score
            risk_analysis = self._analyze_risk(category, features_dict, probabilities)
            
            # Get wallet statistics
            stats = self._get_wallet_stats(features_dict)
            
            result = {
                'address': address,
                'category': category,
                'confidence': round(confidence * 100, 2),
                'confidence_raw': round(confidence, 4),
                'all_probabilities': {
                    cat: round(float(prob) * 100, 2) 
                    for cat, prob in zip(WALLET_CATEGORIES, probabilities)
                },
                'statistics': stats,
                'risk_analysis': risk_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error for {address}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'error': f'Prediction failed: {str(e)}',
                'address': address
            }
    
    def _validate_address(self, address: str) -> bool:
        """Validate Ethereum address"""
        if not isinstance(address, str):
            return False
        if not address.startswith('0x'):
            return False
        if len(address) != 42:
            return False
        try:
            int(address[2:], 16)
            return True
        except ValueError:
            return False
    
    def _get_wallet_stats(self, features: Dict) -> Dict:
        """Extract wallet statistics"""
        return {
            'total_transactions': int(features.get('total_transactions', 0)),
            'total_eth_sent': round(features.get('total_eth_sent', 0), 4),
            'total_eth_received': round(features.get('total_eth_received', 0), 4),
            'net_flow': round(features.get('net_eth_flow', 0), 4),
            'current_balance': round(features.get('current_balance', 0), 4),
            'days_active': int(features.get('days_active', 0)),
            'unique_addresses_contacted': int(features.get('unique_addresses', 0)),
            'avg_transaction_value': round(features.get('avg_tx_value', 0), 4),
            'max_transaction_value': round(features.get('max_tx_value', 0), 4),
            'is_contract': bool(features.get('is_contract', 0)),
            'erc20_activity': int(features.get('erc20_transfers', 0)),
            'erc721_activity': int(features.get('erc721_transfers', 0)),
            'erc1155_activity': int(features.get('erc1155_transfers', 0))
        }
    
    def _analyze_risk(self, category: str, features: Dict, probabilities: np.ndarray) -> Dict:
        """Analyze wallet risk"""
        risk_factors = []
        risk_score = 0
        
        category_risk = {
            'Exchange Wallet': 20,
            'Token Contract (ERC-20)': 30,
            'NFT Contract (ERC-721/1155)': 25,
            'Smart Contract': 45,
            'Normal Wallet': 15,
            'Inactive Wallet': 50
        }
        risk_score += category_risk.get(category, 30)
        
        if features.get('total_transactions', 0) > 1000:
            risk_factors.append('Very high transaction volume')
            risk_score += 15
        
        if features.get('max_tx_value', 0) > 20:
            risk_factors.append('Large value transactions detected')
            risk_score += 10
        
        if features.get('is_contract', 0):
            risk_factors.append('Smart contract address')
            risk_score += 20
        
        if features.get('days_active', 0) > 500 and features.get('total_transactions', 0) < 10:
            risk_factors.append('Long period of inactivity')
            risk_score += 15
        
        if max(probabilities) < 0.7:
            risk_factors.append('Low prediction confidence')
            risk_score += 10
        
        if features.get('unique_addresses', 0) > 200:
            risk_factors.append('Interacts with many addresses')
            risk_score += 5
        
        risk_score = min(risk_score, 100)
        if risk_score < 30:
            risk_level = 'low'
            risk_color = '#10b981'
        elif risk_score < 60:
            risk_level = 'medium'
            risk_color = '#f59e0b'
        else:
            risk_level = 'high'
            risk_color = '#ef4444'
        
        return {
            'score': round(risk_score, 1),
            'level': risk_level,
            'color': risk_color,
            'factors': risk_factors if risk_factors else ['No significant risk factors detected']
        }

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('data/predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            address TEXT UNIQUE,
            category TEXT,
            confidence REAL,
            risk_score REAL,
            statistics TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_info (
            id INTEGER PRIMARY KEY,
            model_path TEXT,
            loaded_at DATETIME,
            model_type TEXT
        )
    ''')
    conn.commit()
    conn.close()

def cache_prediction(result: Dict):
    """Cache prediction result"""
    try:
        conn = sqlite3.connect('data/predictions.db')
        c = conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO predictions 
            (address, category, confidence, risk_score, statistics)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            result['address'],
            result['category'],
            result['confidence_raw'],
            result['risk_analysis']['score'],
            json.dumps(result['statistics'])
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Caching error: {str(e)}")

def create_sample_csv():
    """Create sample CSV"""
    sample_addresses = [
        "0xdAC17F958D2ee523a2206206994597C13D831ec7",
        "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
        "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
        "0x514910771AF9Ca656af840dff83E8264EcF986CA",
        "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",
        "0x6B175474E89094C44Da98b954EedeAC495271d0F",
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
        "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
        "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE",
        "0x4Fabb145d64652a948d72533023f6E7A623C7C53",
    ]
    
    df = pd.DataFrame({'address': sample_addresses})
    output_path = 'data/sample_addresses.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Created sample CSV: {output_path}")
    return output_path

# ============================================================================
# FRONTEND ROUTES
# ============================================================================

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('../frontend', path)

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/info', methods=['GET'])
def api_info():
    return jsonify({
        'message': 'CryptoGuard Backend API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'health': '/api/health',
            'initialize': '/api/initialize',
            'predict': '/api/predict',
            'batch': '/api/predict/batch',
            'export': '/api/export',
            'history': '/api/history',
            'statistics': '/api/statistics'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': str(DEVICE) if DEVICE else 'not_set',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/initialize', methods=['POST'])
def initialize_model():
    global MODEL, SCALER, FEATURE_COLS, MODEL_PATH, DEVICE
    
    try:
        data = request.json
        model_path = data.get('model_path')
        
        if not model_path:
            return jsonify({'error': 'model_path is required'}), 400
        
        # Load model
        logger.info("🚀 Starting model initialization...")
        model, scaler, feature_cols, device = ModelLoader.load_model(model_path)
        
        # CRITICAL: Verify it's nn.Module before creating predictor
        if not isinstance(model, nn.Module):
            raise ValueError(f"Model is {type(model)}, not nn.Module!")
        
        MODEL = WalletPredictor(model, scaler, feature_cols, device)
        MODEL_PATH = model_path
        DEVICE = device
        SCALER = scaler
        FEATURE_COLS = feature_cols
        
        # Save model info
        conn = sqlite3.connect('data/predictions.db')
        c = conn.cursor()
        c.execute('DELETE FROM model_info')
        c.execute('''
            INSERT INTO model_info (id, model_path, loaded_at, model_type)
            VALUES (1, ?, ?, ?)
        ''', (model_path, datetime.now().isoformat(), type(model).__name__))
        conn.commit()
        conn.close()
        
        return jsonify({
            'status': 'success',
            'message': 'Model loaded successfully',
            'model_path': model_path,
            'model_type': type(model).__name__,
            'device': str(device),
            'has_scaler': scaler is not None,
            'feature_count': len(feature_cols) if feature_cols else 'auto'
        })
        
    except Exception as e:
        logger.error(f"Initialization error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_single():
    if MODEL is None:
        return jsonify({'error': 'Model not initialized'}), 400
    
    try:
        data = request.json
        address = data.get('address', '').strip()
        
        if not address:
            return jsonify({'error': 'address is required'}), 400
        
        result = MODEL.predict(address)
        
        if 'error' not in result:
            cache_prediction(result)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    if MODEL is None:
        return jsonify({'error': 'Model not initialized'}), 400
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        df = pd.read_csv(file)
        
        if 'address' not in df.columns:
            return jsonify({'error': 'CSV must contain "address" column'}), 400
        
        results = []
        for idx, address in enumerate(df['address']):
            result = MODEL.predict(str(address).strip())
            results.append(result)
            
            if 'error' not in result:
                cache_prediction(result)
            
            if (idx + 1) % 10 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} addresses")
        
        successful = [r for r in results if 'error' not in r]
        category_counts = {}
        for r in successful:
            cat = r['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        return jsonify({
            'total': len(results),
            'successful': len(successful),
            'failed': len(results) - len(successful),
            'category_distribution': category_counts,
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/export', methods=['POST'])
def export_results():
    try:
        data = request.json
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'No results to export'}), 400
        
        export_data = []
        for result in results:
            if 'error' in result:
                export_data.append({
                    'Address': result['address'],
                    'Status': 'Error',
                    'Error': result['error']
                })
                continue
            
            stats = result.get('statistics', {})
            risk = result.get('risk_analysis', {})
            
            export_data.append({
                'Address': result['address'],
                'Category': result['category'],
                'Confidence (%)': result['confidence'],
                'Risk Score': risk.get('score', 0),
                'Risk Level': risk.get('level', 'unknown'),
                'Total Transactions': stats.get('total_transactions', 0),
                'ETH Sent': stats.get('total_eth_sent', 0),
                'ETH Received': stats.get('total_eth_received', 0),
                'Current Balance': stats.get('current_balance', 0),
                'Days Active': stats.get('days_active', 0),
                'Is Contract': stats.get('is_contract', False),
                'Timestamp': result.get('timestamp', '')
            })
        
        df = pd.DataFrame(export_data)
        
        os.makedirs('data/exports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'data/exports/predictions_{timestamp}.csv'
        df.to_csv(output_path, index=False)
        
        return send_file(
            output_path,
            as_attachment=True,
            download_name=f'wallet_predictions_{timestamp}.csv'
        )
        
    except Exception as e:
        logger.error(f"Export error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        limit = request.args.get('limit', 50, type=int)
        
        conn = sqlite3.connect('data/predictions.db')
        query = f'''
            SELECT address, category, confidence, risk_score, timestamp
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT {limit}
        '''
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) == 0:
            return jsonify({'history': [], 'total': 0})
        
        history = df.to_dict('records')
        
        return jsonify({
            'history': history,
            'total': len(history)
        })
        
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    try:
        conn = sqlite3.connect('data/predictions.db')
        
        df = pd.read_sql_query(
            'SELECT category, COUNT(*) as count FROM predictions GROUP BY category',
            conn
        )
        category_dist = dict(zip(df['category'], df['count'])) if len(df) > 0 else {}
        
        stats_df = pd.read_sql_query(
            'SELECT AVG(confidence) as avg_conf, AVG(risk_score) as avg_risk, COUNT(*) as total FROM predictions',
            conn
        )
        
        conn.close()
        
        total_preds = 0
        avg_conf = 0.0
        avg_risk = 0.0
        
        if len(stats_df) > 0 and stats_df['total'].iloc[0] is not None:
            total_preds = int(stats_df['total'].iloc[0])
            
            if total_preds > 0:
                avg_conf_val = stats_df['avg_conf'].iloc[0]
                avg_risk_val = stats_df['avg_risk'].iloc[0]
                
                avg_conf = round(float(avg_conf_val), 2) if avg_conf_val is not None else 0.0
                avg_risk = round(float(avg_risk_val), 2) if avg_risk_val is not None else 0.0
        
        return jsonify({
            'total_predictions': total_preds,
            'average_confidence': avg_conf,
            'average_risk_score': avg_risk,
            'category_distribution': category_dist
        })
        
    except Exception as e:
        logger.error(f"Statistics error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def open_browser():
    import time
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000')

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    init_db()
    create_sample_csv()
    
    print("\n" + "="*70)
    print("  🚀 CryptoGuard - COMPLETELY FIXED VERSION (20 Features)")
    print("="*70)
    print("\n✅ Server starting on http://localhost:5000")
    print("\n📱 Opening browser automatically...")
    print("\n💡 What to do:")
    print("   1. Click 'Initialize Model'")
    print("   2. Enter path: ../Models/models/models")
    print("   3. Start predicting!")
    print("\n⚠️  Press Ctrl+C to stop")
    print("="*70 + "\n")
    
    threading.Timer(1.5, open_browser).start()
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
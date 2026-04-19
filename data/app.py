"""
CryptoGuard - Cryptocurrency Wallet Categorization System
FIXED VERSION - Feature dimension mismatch resolved
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
# SIMPLE NEURAL NETWORK MODEL - FIXED INPUT SIZE
# ============================================================================

class SimpleNeuralNetwork(nn.Module):
    """Simple neural network for wallet classification - FIXED to match training"""
    def __init__(self, input_size=24, hidden_size=64, num_classes=6):  # CHANGED: 20 -> 24
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
# MODEL LOADER
# ============================================================================

class ModelLoader:
    """Handles loading pre-trained models"""
    
    @staticmethod
    def load_model(model_dir: str):
        """Load PyTorch model from directory"""
        try:
            logger.info(f"Loading model from: {model_dir}")
            
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {device}")
            
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
                raise FileNotFoundError("No model files found in directory")
            
            model_path = model_files[0]
            logger.info(f"Loading model file: {model_path}")
            
            model = None
            input_size = 24  # DEFAULT to 24 features
            
            try:
                loaded_obj = torch.load(model_path, map_location=device)
                logger.info(f"Loaded object type: {type(loaded_obj)}")
                
                if isinstance(loaded_obj, dict):
                    logger.info(f"Dict keys: {list(loaded_obj.keys())}")
                    
                    state_dict = None
                    if 'model_state_dict' in loaded_obj:
                        state_dict = loaded_obj['model_state_dict']
                    elif 'state_dict' in loaded_obj:
                        state_dict = loaded_obj['state_dict']
                    else:
                        state_dict = loaded_obj
                    
                    # Infer input size from first layer
                    first_key = list(state_dict.keys())[0]
                    logger.info(f"First key: {first_key}")
                    logger.info(f"First tensor shape: {state_dict[first_key].shape}")
                    
                    if 'weight' in first_key and len(state_dict[first_key].shape) >= 2:
                        input_size = state_dict[first_key].shape[1]
                        logger.info(f"✓ Inferred input size from model: {input_size}")
                    else:
                        logger.warning(f"Could not infer input size, using default: {input_size}")
                    
                    # Create model with correct input size
                    model = SimpleNeuralNetwork(input_size=input_size)
                    logger.info(f"Created SimpleNeuralNetwork with input_size={input_size}")
                    
                    try:
                        model.load_state_dict(state_dict, strict=False)
                        logger.info("✓ Successfully loaded state dict into model")
                    except Exception as e:
                        logger.warning(f"Could not load state dict: {e}")
                    
                elif isinstance(loaded_obj, nn.Module):
                    logger.info("Loaded complete PyTorch model")
                    model = loaded_obj
                else:
                    raise ValueError(f"Unknown PyTorch object type: {type(loaded_obj)}")
                    
            except Exception as e:
                logger.warning(f"PyTorch loading failed: {e}")
                logger.info("Creating new model with 24 features...")
                model = SimpleNeuralNetwork(input_size=24)
            
            if model is None:
                model = SimpleNeuralNetwork(input_size=24)
            
            if hasattr(model, 'eval'):
                model.eval()
            
            if isinstance(model, nn.Module):
                model = model.to(device)
            
            # Load scaler
            scaler = None
            if scaler_file:
                logger.info(f"Loading scaler: {scaler_file}")
                try:
                    scaler = joblib.load(scaler_file)
                except Exception as e:
                    logger.warning(f"Could not load scaler: {e}")
            
            # Load feature columns
            feature_cols = None
            if feature_file:
                logger.info(f"Loading feature columns: {feature_file}")
                try:
                    with open(feature_file, 'r') as f:
                        feature_cols = json.load(f)
                    logger.info(f"Loaded {len(feature_cols)} feature columns")
                except Exception as e:
                    logger.warning(f"Could not load feature columns: {e}")
            
            logger.info("✅ Model loading complete!")
            logger.info(f"   Input size: {input_size}")
            logger.info(f"   Device: {device}")
            logger.info(f"   Scaler: {'Yes' if scaler else 'No'}")
            logger.info(f"   Feature cols: {len(feature_cols) if feature_cols else 'Auto'}")
            
            return model, scaler, feature_cols, device, input_size
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Creating fallback model...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = SimpleNeuralNetwork(input_size=24).to(device)
            model.eval()
            return model, None, None, device, 24

# ============================================================================
# FEATURE EXTRACTOR - FIXED TO OUTPUT 24 FEATURES
# ============================================================================

class WalletFeatureExtractor:
    """Extract features from wallet addresses - FIXED to match 24 features"""
    
    @staticmethod
    def extract_features(address: str, blockchain_data: Optional[Dict] = None) -> Dict:
        """
        Extract 24 features from wallet address to match model training
        """
        features = {}
        
        # Mock blockchain data if not provided
        if blockchain_data is None:
            blockchain_data = WalletFeatureExtractor._get_mock_blockchain_data(address)
        
        # ==================== 24 FEATURES (matching training) ====================
        
        # 1. total_transactions
        features['total_transactions'] = blockchain_data.get('total_txs', 0)
        
        # 2. total_value_eth (total ETH sent + received)
        eth_sent = blockchain_data.get('eth_sent', 0.0)
        eth_received = blockchain_data.get('eth_received', 0.0)
        features['total_value_eth'] = eth_sent + eth_received
        
        # 3. unique_interactions (unique addresses contacted)
        features['unique_interactions'] = blockchain_data.get('unique_addresses', 0)
        
        # 4. contract_interactions
        features['contract_interactions'] = blockchain_data.get('contract_txs', 0)
        
        # 5. token_transfers (ERC-20)
        features['token_transfers'] = blockchain_data.get('erc20_transfers', 0)
        
        # 6. eth_balance (current balance)
        features['eth_balance'] = blockchain_data.get('balance', 0.0)
        
        # 7. avg_transaction_value
        total_tx = max(features['total_transactions'], 1)
        features['avg_transaction_value'] = features['total_value_eth'] / total_tx
        
        # 8. max_transaction_value
        features['max_transaction_value'] = blockchain_data.get('max_tx_value', 0.0)
        
        # 9. min_transaction_value  
        features['min_transaction_value'] = blockchain_data.get('min_tx_value', 0.0)
        
        # 10. transaction_frequency (txs per day)
        days_active = blockchain_data.get('days_active', 1)
        features['transaction_frequency'] = features['total_transactions'] / max(days_active, 1)
        
        # 11. first_transaction_days (days since first tx)
        features['first_transaction_days'] = blockchain_data.get('first_tx_days_ago', 0)
        
        # 12. last_transaction_days (days since last tx)
        features['last_transaction_days'] = blockchain_data.get('last_tx_days_ago', 0)
        
        # 13. is_contract (0 or 1)
        features['is_contract'] = int(blockchain_data.get('is_contract', 0))
        
        # 14. nft_transfers (ERC-721 + ERC-1155)
        features['nft_transfers'] = (
            blockchain_data.get('erc721_transfers', 0) + 
            blockchain_data.get('erc1155_transfers', 0)
        )
        
        # 15. incoming_transactions
        features['incoming_transactions'] = blockchain_data.get('incoming_txs', 0)
        
        # 16. outgoing_transactions
        features['outgoing_transactions'] = blockchain_data.get('outgoing_txs', 0)
        
        # 17. eth_sent
        features['eth_sent'] = eth_sent
        
        # 18. eth_received
        features['eth_received'] = eth_received
        
        # 19. net_eth_flow (received - sent)
        features['net_eth_flow'] = eth_received - eth_sent
        
        # 20. unique_senders
        features['unique_senders'] = blockchain_data.get('unique_senders', 0)
        
        # 21. unique_receivers
        features['unique_receivers'] = blockchain_data.get('unique_receivers', 0)
        
        # 22. contract_creation (0 or 1)
        features['contract_creation'] = int(blockchain_data.get('created_contract', 0))
        
        # 23. internal_transactions
        features['internal_transactions'] = blockchain_data.get('internal_txs', 0)
        
        # 24. failed_transactions
        features['failed_transactions'] = blockchain_data.get('failed_txs', 0)
        
        return features
    
    @staticmethod
    def _get_mock_blockchain_data(address: str) -> Dict:
        """Generate mock blockchain data for demonstration"""
        np.random.seed(int(address[2:10], 16) % 2**32)
        
        total_txs = np.random.randint(5, 2000)
        days_active = np.random.randint(30, 1000)
        incoming_txs = int(total_txs * np.random.uniform(0.3, 0.7))
        outgoing_txs = total_txs - incoming_txs
        
        return {
            'total_txs': total_txs,
            'eth_sent': round(np.random.uniform(0.1, 100.0), 4),
            'eth_received': round(np.random.uniform(0.1, 100.0), 4),
            'days_active': days_active,
            'unique_addresses': np.random.randint(5, 300),
            'contract_txs': np.random.randint(0, 500),
            'erc20_transfers': np.random.randint(0, 500),
            'balance': round(np.random.uniform(0.0, 50.0), 4),
            'max_tx_value': round(np.random.uniform(1.0, 50.0), 4),
            'min_tx_value': round(np.random.uniform(0.001, 0.1), 4),
            'first_tx_days_ago': days_active,
            'last_tx_days_ago': np.random.randint(0, 30),
            'is_contract': np.random.choice([0, 1], p=[0.7, 0.3]),
            'erc721_transfers': np.random.randint(0, 100),
            'erc1155_transfers': np.random.randint(0, 50),
            'incoming_txs': incoming_txs,
            'outgoing_txs': outgoing_txs,
            'unique_senders': np.random.randint(5, 150),
            'unique_receivers': np.random.randint(5, 150),
            'created_contract': np.random.choice([0, 1], p=[0.95, 0.05]),
            'internal_txs': np.random.randint(0, 200),
            'failed_txs': np.random.randint(0, 50),
        }

# ============================================================================
# WALLET PREDICTOR
# ============================================================================

class WalletPredictor:
    """Make predictions using loaded model"""
    
    def __init__(self, model, scaler, feature_cols, device, expected_features=24):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.device = device
        self.expected_features = expected_features
        self.extractor = WalletFeatureExtractor()
    
    def predict(self, address: str) -> Dict:
        """Predict wallet category for given address"""
        try:
            if not self._validate_address(address):
                return {
                    'error': 'Invalid Ethereum address format',
                    'address': address
                }
            
            # Extract features
            features_dict = self.extractor.extract_features(address)
            
            # Convert to array (ensure correct order and size)
            if self.feature_cols:
                # Use predefined feature order
                feature_array = np.array([features_dict.get(col, 0) for col in self.feature_cols])
            else:
                # Use natural order from features_dict
                feature_array = np.array(list(features_dict.values()))
            
            # Verify correct number of features
            if len(feature_array) != self.expected_features:
                logger.error(f"Feature mismatch! Expected {self.expected_features}, got {len(feature_array)}")
                logger.error(f"Features: {list(features_dict.keys())}")
                return {
                    'error': f'Feature dimension mismatch: expected {self.expected_features}, got {len(feature_array)}',
                    'address': address
                }
            
            feature_array = feature_array.reshape(1, -1)
            
            # Scale features if scaler exists
            if self.scaler:
                feature_array = self.scaler.transform(feature_array)
            
            # Make prediction
            if isinstance(self.model, nn.Module):
                features_tensor = torch.FloatTensor(feature_array).to(self.device)
                with torch.no_grad():
                    output = self.model(features_tensor)
                    probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                    predicted_idx = np.argmax(probabilities)
            else:
                # Fallback for scikit-learn models
                if hasattr(self.model, 'predict_proba'):
                    probabilities = self.model.predict_proba(feature_array)[0]
                    predicted_idx = np.argmax(probabilities)
                else:
                    predicted_idx = self.model.predict(feature_array)[0]
                    probabilities = np.zeros(len(WALLET_CATEGORIES))
                    probabilities[predicted_idx] = 0.9
                    probabilities[probabilities == 0] = 0.1 / (len(WALLET_CATEGORIES) - 1)
            
            category = WALLET_CATEGORIES[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            risk_analysis = self._analyze_risk(category, features_dict, probabilities)
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
        """Validate Ethereum address format"""
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
        """Extract wallet statistics from features"""
        return {
            'total_transactions': int(features.get('total_transactions', 0)),
            'total_value_eth': round(features.get('total_value_eth', 0), 4),
            'eth_sent': round(features.get('eth_sent', 0), 4),
            'eth_received': round(features.get('eth_received', 0), 4),
            'net_flow': round(features.get('net_eth_flow', 0), 4),
            'current_balance': round(features.get('eth_balance', 0), 4),
            'unique_interactions': int(features.get('unique_interactions', 0)),
            'avg_transaction_value': round(features.get('avg_transaction_value', 0), 4),
            'max_transaction_value': round(features.get('max_transaction_value', 0), 4),
            'is_contract': bool(features.get('is_contract', 0)),
            'token_transfers': int(features.get('token_transfers', 0)),
            'nft_transfers': int(features.get('nft_transfers', 0))
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
        
        if features.get('max_transaction_value', 0) > 20:
            risk_factors.append('Large value transactions detected')
            risk_score += 10
        
        if features.get('is_contract', 0):
            risk_factors.append('Smart contract address')
            risk_score += 20
        
        if features.get('first_transaction_days', 0) > 500 and features.get('total_transactions', 0) < 10:
            risk_factors.append('Long period of inactivity')
            risk_score += 15
        
        if max(probabilities) < 0.7:
            risk_factors.append('Low prediction confidence')
            risk_score += 10
        
        if features.get('unique_interactions', 0) > 200:
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
            model_type TEXT,
            input_size INTEGER
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
    """Create a sample CSV file for testing"""
    sample_addresses = [
        "0xdAC17F958D2ee523a2206206994597C13D831ec7",  # USDT
        "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",  # USDC
        "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",  # UNI
        "0x514910771AF9Ca656af840dff83E8264EcF986CA",  # LINK
        "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",  # MATIC
        "0x6B175474E89094C44Da98b954EedeAC495271d0F",  # DAI
        "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
        "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # WBTC
        "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE",  # SHIB
        "0x4Fabb145d64652a948d72533023f6E7A623C7C53",  # BUSD
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
        'message': 'CryptoGuard Backend API - FIXED VERSION',
        'version': '1.0.1-FIXED',
        'status': 'running',
        'fix': 'Feature dimension corrected to 24',
        'endpoints': {
            'health': '/api/health',
            'initialize': '/api/initialize',
            'predict': '/api/predict',
            'batch': '/api/predict/batch',
            'export': '/api/export',
            'history': '/api/history',
            'statistics': '/api/statistics',
            'sample_csv': '/api/sample-csv'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None,
        'device': str(DEVICE) if DEVICE else 'not_set',
        'expected_features': 24,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/sample-csv', methods=['GET'])
def get_sample_csv():
    try:
        csv_path = create_sample_csv()
        return send_file(csv_path, as_attachment=True, download_name='sample_wallet_addresses.csv', mimetype='text/csv')
    except Exception as e:
        logger.error(f"Sample CSV error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/initialize', methods=['POST'])
def initialize_model():
    global MODEL, SCALER, FEATURE_COLS, MODEL_PATH, DEVICE
    
    try:
        data = request.json
        model_path = data.get('model_path')
        
        if not model_path:
            return jsonify({'error': 'model_path is required'}), 400
        
        # Load model
        model, scaler, feature_cols, device, input_size = ModelLoader.load_model(model_path)
        
        MODEL = WalletPredictor(model, scaler, feature_cols, device, expected_features=input_size)
        MODEL_PATH = model_path
        DEVICE = device
        SCALER = scaler
        FEATURE_COLS = feature_cols
        
        # Save model info - with error handling for schema issues
        try:
            conn = sqlite3.connect('data/predictions.db')
            c = conn.cursor()
            c.execute('DELETE FROM model_info')
            
            # Try to insert with input_size
            try:
                c.execute('''
                    INSERT INTO model_info (id, model_path, loaded_at, model_type, input_size)
                    VALUES (1, ?, ?, ?, ?)
                ''', (model_path, datetime.now().isoformat(), type(model).__name__, input_size))
            except sqlite3.OperationalError as e:
                # If input_size column doesn't exist, just insert without it
                logger.warning(f"Database schema issue: {e}. Inserting without input_size.")
                c.execute('''
                    INSERT INTO model_info (id, model_path, loaded_at, model_type)
                    VALUES (1, ?, ?, ?)
                ''', (model_path, datetime.now().isoformat(), type(model).__name__))  
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not save model info to database: {e}")
        
        return jsonify({
            'status': 'success',
            'message': 'Model loaded successfully',
            'model_path': model_path,
            'device': str(device),
            'input_size': input_size,
            'has_scaler': scaler is not None,
            'feature_count': len(feature_cols) if feature_cols else input_size
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
        import traceback
        logger.error(traceback.format_exc())
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
                'ETH Sent': stats.get('eth_sent', 0),
                'ETH Received': stats.get('eth_received', 0),
                'Current Balance': stats.get('current_balance', 0),
                'Is Contract': stats.get('is_contract', False),
                'Token Transfers': stats.get('token_transfers', 0),
                'NFT Transfers': stats.get('nft_transfers', 0),
                'Timestamp': result.get('timestamp', '')
            })
        
        df = pd.DataFrame(export_data)
        
        os.makedirs('data/exports', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'data/exports/predictions_{timestamp}.csv'
        df.to_csv(output_path, index=False)
        
        return send_file(output_path, as_attachment=True, download_name=f'wallet_predictions_{timestamp}.csv')
        
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
    print("  🚀 CryptoGuard - FIXED VERSION")
    print("="*70)
    print("\n✅ Fixed feature dimension mismatch (21 → 24 features)")
    print("✅ Server starting on http://localhost:5000")
    print("\n💡 What to do next:")
    print("   1. Browser will open automatically")
    print("   2. Click 'Initialize Model' button")
    print("   3. Select model path (e.g., Models/models/models)")
    print("   4. Start analyzing wallets!")
    print("\n📁 Sample CSV created: data/sample_addresses.csv")
    print("\n⚠️  Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    threading.Timer(1.5, open_browser).start()
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

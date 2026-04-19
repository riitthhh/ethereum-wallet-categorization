// CryptoGuard - Frontend JavaScript
// Configuration
const API_BASE_URL = 'http://localhost:5000/api';

// State Management
const state = {
    modelLoaded: false,
    currentResults: [],
    batchResults: [],
    currentSection: 'init'
};

// Category Icons
const CATEGORY_ICONS = {
    'Exchange Wallet': '🏦',
    'Token Contract (ERC-20)': '🪙',
    'NFT Contract (ERC-721/1155)': '🎨',
    'Smart Contract': '📜',
    'Normal Wallet': '💼',
    'Inactive Wallet': '💤'
};

// DOM Elements
const elements = {
    // Initialization
    modelPath: document.getElementById('modelPath'),
    initModelBtn: document.getElementById('initModelBtn'),
    initStatus: document.getElementById('initStatus'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    
    // Analysis
    walletAddress: document.getElementById('walletAddress'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    resultsSection: document.getElementById('results-section'),
    
    // Results display
    resultAddress: document.getElementById('resultAddress'),
    categoryBadge: document.getElementById('categoryBadge'),
    categoryIcon: document.getElementById('categoryIcon'),
    categoryText: document.getElementById('categoryText'),
    confidenceValue: document.getElementById('confidenceValue'),
    confidenceFill: document.getElementById('confidenceFill'),
    riskScore: document.getElementById('riskScore'),
    riskLevel: document.getElementById('riskLevel'),
    riskProgressCircle: document.getElementById('riskProgressCircle'),
    riskFactors: document.getElementById('riskFactors'),
    statsGrid: document.getElementById('statsGrid'),
    
    // Batch processing
    csvFile: document.getElementById('csvFile'),
    uploadBtn: document.getElementById('uploadBtn'),
    uploadArea: document.getElementById('uploadArea'),
    batchProgress: document.getElementById('batchProgress'),
    progressText: document.getElementById('progressText'),
    progressPercent: document.getElementById('progressPercent'),
    progressFill: document.getElementById('progressFill'),
    batchResults: document.getElementById('batchResults'),
    batchTotal: document.getElementById('batchTotal'),
    batchSuccess: document.getElementById('batchSuccess'),
    batchFailed: document.getElementById('batchFailed'),
    exportBtn: document.getElementById('exportBtn'),
    
    // History
    historyBody: document.getElementById('historyBody'),
    
    // Stats
    totalPredictions: document.getElementById('totalPredictions'),
    avgConfidence: document.getElementById('avgConfidence')
};

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    checkHealth();
    loadStatistics();
    
    // Auto-hide init section and show analyze after a brief check
    setTimeout(() => {
        // If model path is pre-filled, suggest initialization
        if (elements.modelPath && elements.modelPath.value.trim()) {
            showToast('💡 Click "Initialize Model" to get started!', 'info');
        }
    }, 1000);
});

// Event Listeners
function initializeEventListeners() {
    // Model initialization
    elements.initModelBtn.addEventListener('click', initializeModel);
    
    // Analysis
    elements.analyzeBtn.addEventListener('click', analyzeSingleWallet);
    elements.walletAddress.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') analyzeSingleWallet();
    });
    
    // Batch processing
    elements.uploadBtn.addEventListener('click', () => elements.csvFile.click());
    elements.csvFile.addEventListener('change', handleFileUpload);
    elements.exportBtn.addEventListener('click', exportResults);
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    
    // Navigation
    const navLinks = document.querySelectorAll('.nav-link');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const section = link.getAttribute('data-section');
            switchSection(section);
        });
    });
}

// API Functions
async function apiRequest(endpoint, options = {}) {
    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Request failed');
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

async function checkHealth() {
    try {
        const data = await apiRequest('/health');
        updateStatus(data.model_loaded);
    } catch (error) {
        showToast('Cannot connect to backend server', 'error');
        updateStatus(false);
    }
}

// Model Initialization
async function initializeModel() {
    const modelPath = elements.modelPath.value.trim();
    
    if (!modelPath) {
        showToast('Please enter model path', 'error');
        return;
    }
    
    setLoading(elements.initModelBtn, true);
    elements.initStatus.className = 'status-message';
    
    try {
        const data = await apiRequest('/initialize', {
            method: 'POST',
            body: JSON.stringify({ model_path: modelPath })
        });
        
        elements.initStatus.className = 'status-message success';
        elements.initStatus.textContent = `✓ ${data.message}`;
        
        state.modelLoaded = true;
        updateStatus(true);
        showToast('Model loaded successfully!', 'success');
        
        // Show analysis section
        setTimeout(() => {
            switchSection('analyze');
        }, 1500);
        
    } catch (error) {
        elements.initStatus.className = 'status-message error';
        elements.initStatus.textContent = `✗ ${error.message}`;
        showToast(error.message, 'error');
    } finally {
        setLoading(elements.initModelBtn, false);
    }
}

// Single Wallet Analysis
async function analyzeSingleWallet() {
    const address = elements.walletAddress.value.trim();
    
    if (!address) {
        showToast('Please enter a wallet address', 'error');
        return;
    }
    
    if (!validateEthereumAddress(address)) {
        showToast('Invalid Ethereum address format', 'error');
        return;
    }
    
    setLoading(elements.analyzeBtn, true);
    
    try {
        const result = await apiRequest('/predict', {
            method: 'POST',
            body: JSON.stringify({ address })
        });
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        displayResults(result);
        state.currentResults.push(result);
        loadStatistics();
        showToast('Analysis complete!', 'success');
        
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        setLoading(elements.analyzeBtn, false);
    }
}

// Display Results
function displayResults(result) {
    // Show results section
    elements.resultsSection.classList.remove('section-hidden');
    
    // Scroll to results
    setTimeout(() => {
        elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 100);
    
    // Display address
    elements.resultAddress.textContent = result.address;
    
    // Display category
    const icon = CATEGORY_ICONS[result.category] || '📊';
    elements.categoryIcon.textContent = icon;
    elements.categoryText.textContent = result.category;
    
    // Display confidence
    elements.confidenceValue.textContent = `${result.confidence}%`;
    elements.confidenceFill.style.width = `${result.confidence}%`;
    
    // Color code confidence
    let confidenceColor;
    if (result.confidence >= 80) confidenceColor = '#10b981';
    else if (result.confidence >= 60) confidenceColor = '#f59e0b';
    else confidenceColor = '#ef4444';
    elements.confidenceFill.style.background = `linear-gradient(90deg, ${confidenceColor}, #7B61FF)`;
    
    // Display risk analysis
    displayRiskAnalysis(result.risk_analysis);
    
    // Display statistics
    displayStatistics(result.statistics);
    
    // Display probability chart
    displayProbabilityChart(result.all_probabilities);
}

// Risk Analysis Display
function displayRiskAnalysis(risk) {
    elements.riskScore.textContent = risk.score;
    elements.riskLevel.textContent = risk.level.toUpperCase();
    elements.riskLevel.style.color = risk.color;
    
    // Animate circle
    const circumference = 2 * Math.PI * 54; // radius = 54
    const offset = circumference - (risk.score / 100) * circumference;
    elements.riskProgressCircle.style.strokeDashoffset = offset;
    
    // Update circle color based on risk level
    const gradient = document.querySelector('#risk-gradient');
    if (risk.level === 'low') {
        gradient.innerHTML = `
            <stop offset="0%" style="stop-color:#10b981"/>
            <stop offset="100%" style="stop-color:#34d399"/>
        `;
    } else if (risk.level === 'medium') {
        gradient.innerHTML = `
            <stop offset="0%" style="stop-color:#f59e0b"/>
            <stop offset="100%" style="stop-color:#fbbf24"/>
        `;
    } else {
        gradient.innerHTML = `
            <stop offset="0%" style="stop-color:#ef4444"/>
            <stop offset="100%" style="stop-color:#f87171"/>
        `;
    }
    
    // Display risk factors
    elements.riskFactors.innerHTML = risk.factors.map(factor => `
        <div class="risk-factor-item">${factor}</div>
    `).join('');
}

// Statistics Display
function displayStatistics(stats) {
    const statsHTML = `
        <div class="stat-box">
            <div class="stat-box-label">Total Transactions</div>
            <div class="stat-box-value">${stats.total_transactions.toLocaleString()}</div>
        </div>
        <div class="stat-box">
            <div class="stat-box-label">ETH Sent</div>
            <div class="stat-box-value">${stats.total_eth_sent} ETH</div>
        </div>
        <div class="stat-box">
            <div class="stat-box-label">ETH Received</div>
            <div class="stat-box-value">${stats.total_eth_received} ETH</div>
        </div>
        <div class="stat-box">
            <div class="stat-box-label">Current Balance</div>
            <div class="stat-box-value">${stats.current_balance} ETH</div>
        </div>
        <div class="stat-box">
            <div class="stat-box-label">Days Active</div>
            <div class="stat-box-value">${stats.days_active}</div>
        </div>
        <div class="stat-box">
            <div class="stat-box-label">Unique Contacts</div>
            <div class="stat-box-value">${stats.unique_addresses_contacted}</div>
        </div>
        <div class="stat-box">
            <div class="stat-box-label">Avg TX Value</div>
            <div class="stat-box-value">${stats.avg_transaction_value} ETH</div>
        </div>
        <div class="stat-box">
            <div class="stat-box-label">Contract Status</div>
            <div class="stat-box-value">${stats.is_contract ? 'Yes' : 'No'}</div>
        </div>
    `;
    
    elements.statsGrid.innerHTML = statsHTML;
}

// Probability Chart - FIXED VERSION
function displayProbabilityChart(probabilities) {
    const ctx = document.getElementById('probabilityChart');
    
    // FIXED: Properly check if chart instance exists and has destroy method
    if (window.probabilityChart && typeof window.probabilityChart.destroy === 'function') {
        try {
            window.probabilityChart.destroy();
        } catch (error) {
            console.warn('Error destroying probability chart:', error);
        }
    }
    
    const labels = Object.keys(probabilities);
    const data = Object.values(probabilities);
    
    window.probabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Confidence (%)',
                data: data,
                backgroundColor: labels.map((_, idx) => {
                    const colors = [
                        'rgba(0, 245, 255, 0.7)',
                        'rgba(123, 97, 255, 0.7)',
                        'rgba(255, 107, 157, 0.7)',
                        'rgba(16, 185, 129, 0.7)',
                        'rgba(245, 158, 11, 0.7)',
                        'rgba(239, 68, 68, 0.7)'
                    ];
                    return colors[idx % colors.length];
                }),
                borderColor: labels.map((_, idx) => {
                    const colors = [
                        'rgba(0, 245, 255, 1)',
                        'rgba(123, 97, 255, 1)',
                        'rgba(255, 107, 157, 1)',
                        'rgba(16, 185, 129, 1)',
                        'rgba(245, 158, 11, 1)',
                        'rgba(239, 68, 68, 1)'
                    ];
                    return colors[idx % colors.length];
                }),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#94a3b8',
                        callback: function(value) {
                            return value + '%';
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#94a3b8',
                        font: {
                            size: 11
                        }
                    }
                }
            }
        }
    });
}

// Batch Processing
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        processBatchFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
    
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
        processBatchFile(file);
    } else {
        showToast('Please upload a CSV file', 'error');
    }
}

async function processBatchFile(file) {
    elements.batchProgress.classList.remove('section-hidden');
    elements.batchResults.classList.add('section-hidden');
    elements.progressFill.style.width = '0%';
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        // Show indeterminate progress
        elements.progressText.textContent = 'Processing batch...';
        elements.progressPercent.textContent = '...';
        elements.progressFill.style.width = '50%';
        
        const response = await fetch(`${API_BASE_URL}/predict/batch`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Batch processing failed');
        }
        
        const data = await response.json();
        
        // Complete progress
        elements.progressFill.style.width = '100%';
        elements.progressPercent.textContent = '100%';
        
        // Store results
        state.batchResults = data.results;
        
        // Display batch results
        setTimeout(() => {
            displayBatchResults(data);
            elements.batchProgress.classList.add('section-hidden');
        }, 500);
        
        showToast(`Batch processing complete! ${data.successful} successful predictions`, 'success');
        loadStatistics();
        
    } catch (error) {
        elements.batchProgress.classList.add('section-hidden');
        showToast(error.message, 'error');
    }
}

function displayBatchResults(data) {
    elements.batchResults.classList.remove('section-hidden');
    
    // Update summary
    elements.batchTotal.textContent = data.total;
    elements.batchSuccess.textContent = data.successful;
    elements.batchFailed.textContent = data.failed;
    
    // Display category distribution chart
    displayCategoryChart(data.category_distribution);
    
    // Scroll to results
    setTimeout(() => {
        elements.batchResults.scrollIntoView({ behavior: 'smooth' });
    }, 100);
}

// Category Chart - FIXED VERSION
function displayCategoryChart(distribution) {
    const ctx = document.getElementById('categoryChart');
    
    // FIXED: Properly check if chart instance exists and has destroy method
    if (window.categoryChart && typeof window.categoryChart.destroy === 'function') {
        try {
            window.categoryChart.destroy();
        } catch (error) {
            console.warn('Error destroying category chart:', error);
        }
    }
    
    const labels = Object.keys(distribution);
    const data = Object.values(distribution);
    
    window.categoryChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: [
                    'rgba(0, 245, 255, 0.8)',
                    'rgba(123, 97, 255, 0.8)',
                    'rgba(255, 107, 157, 0.8)',
                    'rgba(16, 185, 129, 0.8)',
                    'rgba(245, 158, 11, 0.8)',
                    'rgba(239, 68, 68, 0.8)'
                ],
                borderColor: [
                    'rgba(0, 245, 255, 1)',
                    'rgba(123, 97, 255, 1)',
                    'rgba(255, 107, 157, 1)',
                    'rgba(16, 185, 129, 1)',
                    'rgba(245, 158, 11, 1)',
                    'rgba(239, 68, 68, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        color: '#94a3b8',
                        padding: 15,
                        font: {
                            size: 12
                        }
                    }
                }
            }
        }
    });
}

// Export Results
async function exportResults() {
    if (state.batchResults.length === 0) {
        showToast('No results to export', 'error');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE_URL}/export`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ results: state.batchResults })
        });
        
        if (!response.ok) {
            throw new Error('Export failed');
        }
        
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `wallet_predictions_${Date.now()}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        showToast('Results exported successfully!', 'success');
        
    } catch (error) {
        showToast('Failed to export results', 'error');
    }
}

// History
async function loadHistory() {
    try {
        const data = await apiRequest('/history?limit=50');
        
        if (data.history.length === 0) {
            elements.historyBody.innerHTML = '<tr><td colspan="5" class="empty-state">No predictions yet</td></tr>';
            return;
        }
        
        const rows = data.history.map(item => `
            <tr>
                <td><code>${item.address.substring(0, 10)}...${item.address.substring(38)}</code></td>
                <td><strong>${item.category}</strong></td>
                <td>${(item.confidence * 100).toFixed(1)}%</td>
                <td>${item.risk_score.toFixed(1)}</td>
                <td>${new Date(item.timestamp).toLocaleString()}</td>
            </tr>
        `).join('');
        
        elements.historyBody.innerHTML = rows;
        
    } catch (error) {
        console.error('Failed to load history:', error);
    }
}

// Statistics
async function loadStatistics() {
    try {
        const data = await apiRequest('/statistics');
        
        if (elements.totalPredictions) {
            elements.totalPredictions.textContent = data.total_predictions.toLocaleString();
        }
        
        if (elements.avgConfidence) {
            elements.avgConfidence.textContent = `${data.average_confidence.toFixed(1)}%`;
        }
        
    } catch (error) {
        console.error('Failed to load statistics:', error);
    }
}

// Section Navigation
function switchSection(section) {
    // Update nav links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('data-section') === section) {
            link.classList.add('active');
        }
    });
    
    // Hide all sections
    document.getElementById('init-section').style.display = 'none';
    document.getElementById('analyze-section').classList.add('section-hidden');
    document.getElementById('batch-section').classList.add('section-hidden');
    document.getElementById('history-section').classList.add('section-hidden');
    
    // Show selected section
    if (section === 'init') {
        document.getElementById('init-section').style.display = 'block';
    } else if (section === 'analyze') {
        if (!state.modelLoaded) {
            showToast('Please initialize the model first', 'error');
            return;
        }
        document.getElementById('analyze-section').classList.remove('section-hidden');
    } else if (section === 'batch') {
        if (!state.modelLoaded) {
            showToast('Please initialize the model first', 'error');
            return;
        }
        document.getElementById('batch-section').classList.remove('section-hidden');
    } else if (section === 'history') {
        document.getElementById('history-section').classList.remove('section-hidden');
        loadHistory();
    }
    
    state.currentSection = section;
}

// Utility Functions
function updateStatus(loaded) {
    if (loaded) {
        elements.statusDot.classList.add('active');
        elements.statusText.textContent = 'Model Ready';
        state.modelLoaded = true;
    } else {
        elements.statusDot.classList.remove('active');
        elements.statusText.textContent = 'Not Initialized';
        state.modelLoaded = false;
    }
}

function validateEthereumAddress(address) {
    return /^0x[a-fA-F0-9]{40}$/.test(address);
}

function setLoading(button, loading) {
    if (loading) {
        button.classList.add('loading');
        button.disabled = true;
    } else {
        button.classList.remove('loading');
        button.disabled = false;
    }
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div style="display: flex; align-items: center; gap: 0.75rem;">
            <div style="font-weight: 600;">${message}</div>
        </div>
    `;
    
    const container = document.getElementById('toast-container');
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideIn 0.3s ease reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Auto-refresh statistics every 30 seconds
setInterval(() => {
    if (state.modelLoaded) {
        loadStatistics();
    }
}, 30000);
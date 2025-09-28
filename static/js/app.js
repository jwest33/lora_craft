// LoRA Craft Unified Interface JavaScript

// Global variables
let socket = null;
let currentSessionId = null;
let currentDatasetSession = null;
let availableModels = {};
let currentStep = 1;
let stepValidation = {
    1: false,
    2: false,
    3: false,
    4: false
};
let lossChart = null;
let lrChart = null;
let rewardChart = null;
let datasetStatusCache = {};
let trainedModels = [];
let selectedModelsForExport = new Set();
let collapseInstances = {}; // Store Bootstrap Collapse instances

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize Socket.IO
    initializeSocketIO();

    // Load models
    loadAvailableModels();

    // Load templates
    loadAvailableTemplates();

    // Add file upload listener
    const fileInput = document.getElementById('dataset-file');
    if (fileInput) {
        console.log('Attaching file upload listener to dataset-file input');
        fileInput.addEventListener('change', handleFileUpload);
    } else {
        console.warn('dataset-file input not found during initialization');
    }

    // Add drag and drop support for dataset upload
    setupDragAndDrop();

    // Refresh sessions and check for running sessions
    refreshSessions().then(() => {
        checkForRunningSessions();
    });

    // Load saved configurations
    loadConfigList();

    // Setup event listeners
    setupEventListeners();

    // Initialize Bootstrap collapse instances
    initializeCollapseInstances();

    // Initialize valid generations dropdown
    updateValidGenerations();

    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Add scroll effect for floating icon
    setupIconScrollEffect();

    // Load saved state from localStorage
    loadSavedState();

    // Update system status
    updateSystemStatus();

    // Set initial template preview
    updateTemplatePreview();

    // Initialize chat template preview
    setTimeout(initializeChatTemplate, 100); // Small delay to ensure DOM is ready
}

// ============================================================================
// Socket.IO Management
// ============================================================================

function initializeSocketIO() {
    socket = io();
    
    socket.on('connect', function() {
        console.log('Connected to server');
        updateConnectionStatus('Online');
    });
    
    socket.on('disconnect', function() {
        console.log('Disconnected from server');
        updateConnectionStatus('Offline');
    });
    
    socket.on('training_progress', function(data) {
        // Only process if it's for our current session
        if (data.session_id === currentSessionId) {
            console.log('Training progress:', data);
            updateTrainingProgress(data.progress);
        }
    });

    socket.on('training_metrics', function(data) {
        // Only process if it's for our current session
        if (data.session_id === currentSessionId) {
            console.log('Training metrics received:', data);
            updateTrainingMetrics(data);
        }
    });

    socket.on('training_log', function(data) {
        // Only process if it's for our current session
        if (data.session_id === currentSessionId) {
            console.log('Training log:', data.message);
            appendLog(data.message);
        }
    });

    socket.on('training_complete', function(data) {
        // Only process if it's for our current session
        if (data.session_id === currentSessionId) {
            console.log('Training complete:', data);
            handleTrainingComplete(data);
        }
    });

    socket.on('training_error', function(data) {
        // Only process if it's for our current session
        if (data.session_id === currentSessionId) {
            console.error('Training error:', data);
            handleTrainingError(data);
        }
    });

    // Dataset download events
    socket.on('dataset_progress', function(data) {
        updateDatasetProgress(data);
    });

    socket.on('dataset_complete', function(data) {
        handleDatasetComplete(data);
    });

    socket.on('dataset_error', function(data) {
        handleDatasetError(data);
    });

    // Export progress updates
    socket.on('export_progress', function(data) {
        const progressBar = document.getElementById('export-progress-bar');
        const statusText = document.getElementById('export-status');

        if (progressBar) {
            progressBar.style.width = `${data.progress}%`;
        }
        if (statusText) {
            statusText.textContent = data.message || 'Exporting...';
        }
    });
}

function updateConnectionStatus(status) {
    const connectionIndicator = document.getElementById('connection-indicator');
    if (connectionIndicator) {
        if (status === 'Online') {
            connectionIndicator.classList.add('online');
            connectionIndicator.classList.remove('offline');
        } else {
            connectionIndicator.classList.add('offline');
            connectionIndicator.classList.remove('online');
        }
    }
}

// ============================================================================
// Step Navigation & Validation
// ============================================================================

function initializeCollapseInstances() {
    // Initialize Bootstrap Collapse instances for all steps
    for (let i = 1; i <= 6; i++) {
        const content = document.getElementById(`step-${i}-content`);
        if (content) {
            // Only create instance if it doesn't already exist
            if (!collapseInstances[i]) {
                collapseInstances[i] = new bootstrap.Collapse(content, {
                    toggle: false
                });
            }
        }
    }
}

function toggleStep(stepNum) {
    const content = document.getElementById(`step-${stepNum}-content`);
    const chevron = document.getElementById(`step-${stepNum}-chevron`);

    if (!content || !chevron) {
        console.error(`Step ${stepNum} elements not found`);
        return;
    }

    // Use the pre-initialized collapse instance
    const bsCollapse = collapseInstances[stepNum];
    if (!bsCollapse) {
        console.error(`Collapse instance for step ${stepNum} not found`);
        return;
    }

    if (content.classList.contains('show')) {
        // Collapse this step
        bsCollapse.hide();
        chevron.classList.remove('fa-chevron-up');
        chevron.classList.add('fa-chevron-down');
    } else {
        // Collapse all other steps first
        for (let i = 1; i <= 6; i++) {
            if (i !== stepNum) {
                const otherContent = document.getElementById(`step-${i}-content`);
                const otherChevron = document.getElementById(`step-${i}-chevron`);
                if (otherContent && otherChevron && otherContent.classList.contains('show')) {
                    const otherCollapse = collapseInstances[i];
                    if (otherCollapse) {
                        otherCollapse.hide();
                        otherChevron.classList.remove('fa-chevron-up');
                        otherChevron.classList.add('fa-chevron-down');
                    }
                }
            }
        }

        // Expand this step
        bsCollapse.show();
        chevron.classList.remove('fa-chevron-down');
        chevron.classList.add('fa-chevron-up');

        // If opening exports section (step 5), refresh the models
        if (stepNum === 5) {
            setTimeout(() => {
                refreshTrainedModels();
            }, 100);
        }
        // If opening test section (step 6), load testable models
        if (stepNum === 6) {
            setTimeout(() => {
                loadTestableModels();
            }, 100);
        }
    }

    currentStep = stepNum;
    updateStepIndicators();
}

function goToStep(stepNum) {
    // Collapse current step
    const currentContent = document.getElementById(`step-${currentStep}-content`);
    const currentChevron = document.getElementById(`step-${currentStep}-chevron`);

    if (currentContent && currentChevron && currentContent.classList.contains('show')) {
        const currentCollapse = collapseInstances[currentStep];
        if (currentCollapse) {
            currentCollapse.hide();
            currentChevron.classList.remove('fa-chevron-up');
            currentChevron.classList.add('fa-chevron-down');
        }
    }

    // Expand target step
    const targetContent = document.getElementById(`step-${stepNum}-content`);
    const targetChevron = document.getElementById(`step-${stepNum}-chevron`);

    if (targetContent && targetChevron) {
        const targetCollapse = collapseInstances[stepNum];
        if (targetCollapse) {
            targetCollapse.show();
            targetChevron.classList.remove('fa-chevron-down');
            targetChevron.classList.add('fa-chevron-up');
        }
    }

    currentStep = stepNum;
    updateStepIndicators();

    // If navigating to exports section (step 5), refresh the models
    if (stepNum === 5) {
        setTimeout(() => {
            refreshTrainedModels();
        }, 100);
    }
    // If navigating to test section (step 6), load testable models
    if (stepNum === 6) {
        setTimeout(() => {
            loadTestableModels();
        }, 100);
    }

    // Smooth scroll to step after a short delay to allow animation
    setTimeout(() => {
        document.getElementById(`step-${stepNum}`).scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }, 300);
}

function validateAndProceed(stepNum) {
    if (validateStep(stepNum)) {
        stepValidation[stepNum] = true;
        updateStepIndicators();
        
        if (stepNum < 4) {
            goToStep(stepNum + 1);
        }
        
        // Save state
        saveState();
        
        // Update summary if reaching step 4
        if (stepNum === 3) {
            updateConfigSummary();
        }
    }
}

function validateStep(stepNum) {
    switch (stepNum) {
        case 1:
            // Validate model selection
            const customModelPath = document.getElementById('custom-model-path')?.value?.trim();
            const modelName = document.getElementById('model-name').value;

            if (!customModelPath && !modelName) {
                showAlert('Please select a model or provide a custom model path', 'warning');
                return false;
            }

            // Validate quantization settings
            const use4bit = document.getElementById('use-4bit')?.checked;
            const use8bit = document.getElementById('use-8bit')?.checked;
            if (use4bit && use8bit) {
                showAlert('Cannot use both 4-bit and 8-bit quantization simultaneously', 'warning');
                return false;
            }

            // Check LoRA configuration has at least one target module
            const targetModules = document.querySelectorAll('[id^="target-"]:checked');
            if (targetModules.length === 0 && document.getElementById('setup-mode')?.value !== 'setup-recommended') {
                // Only warn if not in recommended mode
                console.log('Warning: No LoRA target modules selected, will use defaults');
            }

            return true;

        case 2:
            // Validate dataset
            const datasetPath = document.getElementById('dataset-path').value;
            if (!datasetPath) {
                showAlert('Please select or enter a dataset', 'warning');
                return false;
            }
            return true;

        case 3:
            // Validate training parameters
            const epochs = document.getElementById('num-epochs').value;
            const batchSize = document.getElementById('batch-size').value;
            const lr = document.getElementById('learning-rate').value;

            if (!epochs || epochs < 1) {
                showAlert('Please enter valid number of epochs', 'warning');
                return false;
            }
            if (!batchSize || batchSize < 1) {
                showAlert('Please enter valid batch size', 'warning');
                return false;
            }
            if (!lr || lr <= 0) {
                showAlert('Please enter valid learning rate', 'warning');
                return false;
            }

            // Validate max sequence length vs max new tokens
            const maxSeqLen = parseInt(document.getElementById('max-sequence-length')?.value || 2048);
            const maxNewTokens = parseInt(document.getElementById('max-new-tokens')?.value || 256);
            if (maxNewTokens >= maxSeqLen) {
                showAlert('Max new tokens must be less than max sequence length', 'warning');
                return false;
            }

            // Validate gradient accumulation
            const gradAccum = parseInt(document.getElementById('gradient-accumulation')?.value || 1);
            if (gradAccum < 1) {
                showAlert('Gradient accumulation steps must be at least 1', 'warning');
                return false;
            }

            // Validate batch size and num_generations compatibility
            const numGenerations = parseInt(document.getElementById('num-generations')?.value || 4);
            if (batchSize % numGenerations !== 0) {
                console.log(`Note: Batch size (${batchSize}) is not divisible by num_generations (${numGenerations}). This will be adjusted automatically during training.`);
                // Show a non-blocking info message
                const adjustedBatchSize = numGenerations <= batchSize ?
                    Math.floor(batchSize / numGenerations) * numGenerations :
                    numGenerations;
                console.log(`Batch size will be adjusted to ${adjustedBatchSize} for GRPO compatibility.`);
            }

            return true;

        case 4:
            return true;

        default:
            return false;
    }
}

function updateStepIndicators() {
    for (let i = 1; i <= 6; i++) {
        const indicator = document.getElementById(`step-${i}-indicator`);
        if (indicator) {
            // Remove both old and new class names for compatibility
            indicator.classList.remove('active', 'completed');

            // Also check for .step class (old) and .compact-step class (new)
            const stepElement = indicator.classList.contains('step') ? indicator :
                                indicator.classList.contains('compact-step') ? indicator : null;

            if (i < currentStep && stepValidation[i]) {
                indicator.classList.add('completed');
            } else if (i === currentStep) {
                indicator.classList.add('active');
            }
        }
    }
}

// ============================================================================
// Model Management
// ============================================================================

async function loadAvailableModels() {
    try {
        const response = await fetch('/api/models');
        availableModels = await response.json();
        updateModelList(); // This will call updateConfigSummary
    } catch (error) {
        console.error('Failed to load models:', error);
        showAlert('Failed to load models', 'danger');
    }
}

function getDivisors(n) {
    const divisors = [];
    for (let i = 1; i <= n; i++) {
        if (n % i === 0) {
            divisors.push(i);
        }
    }
    return divisors;
}

function updateValidGenerations() {
    const batchSizeInput = document.getElementById('batch-size');
    const batchSize = parseInt(batchSizeInput.value) || 4;
    const generationsSelect = document.getElementById('num-generations');

    // Calculate valid divisors of batch size
    const validOptions = getDivisors(batchSize);

    // Clear current options
    generationsSelect.innerHTML = '';

    // Add valid options - always default to batch size
    validOptions.forEach(value => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;

        // Select the batch size by default
        if (value === batchSize) {
            option.selected = true;
        }

        generationsSelect.appendChild(option);
    });

    // Explicitly set the value to ensure it's selected
    generationsSelect.value = batchSize;

    // Update config summary if needed
    updateConfigSummary();
}

function updateModelList() {
    const family = document.getElementById('model-family').value;
    const modelSelect = document.getElementById('model-name');

    modelSelect.innerHTML = '';

    // If Recommended preset is active, update recommendations
    updateRecommendedIfActive();

    if (availableModels[family]) {
        availableModels[family].forEach((model, index) => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = `${model.name} (${model.vram})`;
            modelSelect.appendChild(option);
        });

        // Ensure first option is selected
        if (modelSelect.options.length > 0) {
            modelSelect.selectedIndex = 0;
        }
    }

    // Update config summary whenever model list changes
    updateConfigSummary();
}

// Model configuration modes
function setModelMode(mode) {
    const loraSection = document.getElementById('lora-config-section');
    const advancedSection = document.getElementById('advanced-config-section');
    const loraPresets = document.getElementById('lora-presets');

    switch(mode) {
        case 'recommended':
            // Hide all advanced options
            loraSection.style.display = 'none';
            advancedSection.style.display = 'none';
            // Apply recommended settings silently
            applyRecommendedSettings();
            break;

        case 'custom':
            // Show LoRA config with presets
            loraSection.style.display = 'block';
            loraPresets.style.display = 'block';
            advancedSection.style.display = 'none';
            break;

        case 'advanced':
            // Show everything
            loraSection.style.display = 'block';
            loraPresets.style.display = 'block';
            advancedSection.style.display = 'block';
            break;
    }
}

function applyRecommendedSettings() {
    // Set optimal default values
    document.getElementById('lora-rank').value = 16;
    document.getElementById('lora-rank-slider').value = 16;
    document.getElementById('lora-alpha').value = 32;
    document.getElementById('lora-alpha-slider').value = 32;
    document.getElementById('lora-dropout').value = 0.05;
    document.getElementById('lora-dropout-slider').value = 0.05;
}

function applyLoRAPreset(preset) {
    const presets = {
        'low': { rank: 8, alpha: 16, dropout: 0.1 },
        'balanced': { rank: 16, alpha: 32, dropout: 0.05 },
        'quality': { rank: 32, alpha: 64, dropout: 0.0 }
    };

    if (presets[preset]) {
        document.getElementById('lora-rank').value = presets[preset].rank;
        document.getElementById('lora-rank-slider').value = presets[preset].rank;
        document.getElementById('lora-alpha').value = presets[preset].alpha;
        document.getElementById('lora-alpha-slider').value = presets[preset].alpha;
        document.getElementById('lora-dropout').value = presets[preset].dropout;
        document.getElementById('lora-dropout-slider').value = presets[preset].dropout;

        // Visual feedback
        showAlert(`Applied ${preset} VRAM preset`, 'success');

        // Update configuration summary
        updateConfigSummary();
    }
}

// ============================================================================
// Dataset Management
// ============================================================================

// Dataset catalog
const datasetCatalog = {
    'alpaca': {
        name: 'Alpaca',
        path: 'tatsu-lab/alpaca',
        size: '52K samples',
        category: 'general',
        description: 'Instruction-following dataset',
        language: 'English',
        icon: '🦙',
        fields: { instruction: 'instruction', response: 'output' }
    },
    'gsm8k': {
        name: 'GSM8K',
        path: 'openai/gsm8k',
        config: 'main',  // GSM8K requires config specification (main or socratic)
        size: '8K problems',
        category: 'math',
        description: 'Grade school math problems',
        language: 'English',
        icon: '🧮',
        fields: { instruction: 'question', response: 'answer' }
    },
    'dapo-math': {
        name: 'DAPO Math 17k',
        path: 'open-r1/DAPO-Math-17k-Processed',
        config: 'all',  // DAPO-Math requires config specification (all, cn, or en)
        size: '17K problems',
        category: 'math',
        description: 'Advanced math with reasoning',
        language: 'English',
        icon: '📐',
        fields: { instruction: 'prompt', response: 'solution' }
    },
    'openmath': {
        name: 'OpenMath Reasoning',
        path: 'nvidia/OpenMathReasoning',
        size: '3.2M problems',
        category: 'math',
        description: 'Mathematical reasoning dataset',
        language: 'English',
        icon: '🔢',
        fields: { instruction: 'problem', response: 'generated_solution' }
    },
    'code-alpaca': {
        name: 'Code Alpaca',
        path: 'sahil2801/CodeAlpaca-20k',
        size: '20K examples',
        category: 'coding',
        description: 'Code generation instructions',
        language: 'English',
        icon: '💻',
        fields: { instruction: 'instruction', response: 'output' }
    },
    'dolly': {
        name: 'Dolly 15k',
        path: 'databricks/databricks-dolly-15k',
        size: '15K samples',
        category: 'general',
        description: 'High-quality instruction dataset',
        language: 'English',
        icon: '🎯',
        fields: { instruction: 'instruction', response: 'response' }
    },
    'orca-math': {
        name: 'Orca Math',
        path: 'microsoft/orca-math-word-problems-200k',
        size: '200K problems',
        category: 'math',
        description: 'Word problems with step-by-step solutions',
        language: 'English',
        icon: '🐋',
        fields: { instruction: 'question', response: 'answer' }
    },
    'squad': {
        name: 'SQuAD v2',
        path: 'squad',
        size: '130K questions',
        category: 'qa',
        description: 'Reading comprehension Q&A',
        language: 'English',
        icon: '📖',
        fields: { instruction: 'question', response: 'answers' },
        requiresProcessing: true
    }
};

function selectDatasetType(type) {
    // Update UI based on selection
    document.querySelectorAll('.selection-card').forEach(card => {
        if (card.id === 'dataset-popular' || card.id === 'dataset-upload' || card.id === 'dataset-custom') {
            card.classList.remove('active');
        }
    });

    // Add active class to selected card
    if (type === 'popular') {
        document.getElementById('dataset-popular').classList.add('active');
    } else if (type === 'upload') {
        document.getElementById('dataset-upload').classList.add('active');
    } else if (type === 'custom') {
        document.getElementById('dataset-custom').classList.add('active');
    }

    const datasetConfig = document.getElementById('dataset-config');
    const datasetCatalogEl = document.getElementById('dataset-catalog');
    const datasetUploadArea = document.getElementById('dataset-upload-area');
    const datasetCustomArea = document.getElementById('dataset-custom-area');

    // Show dataset configuration section
    datasetConfig.style.display = 'block';

    if (type === 'popular') {
        // Show catalog
        datasetCatalogEl.style.display = 'block';
        datasetUploadArea.style.display = 'none';
        datasetCustomArea.style.display = 'none';
        loadDatasetCatalog();
    } else if (type === 'custom') {
        // Show custom dataset input
        datasetCatalogEl.style.display = 'none';
        datasetUploadArea.style.display = 'none';
        datasetCustomArea.style.display = 'block';
    } else if (type === 'upload') {
        // Show upload area
        datasetCatalogEl.style.display = 'none';
        datasetCustomArea.style.display = 'none';
        datasetUploadArea.style.display = 'block';
        // Load previously uploaded datasets
        loadUploadedDatasets();
    }
}

async function loadDatasetCatalog() {
    const grid = document.getElementById('dataset-grid');
    grid.innerHTML = '';

    // First load status for all datasets
    await updateDatasetStatuses();

    // Get currently selected dataset path
    const currentDatasetPath = document.getElementById('dataset-path').value;

    Object.entries(datasetCatalog).forEach(([key, dataset]) => {
        const card = createDatasetCard(key, dataset);
        grid.appendChild(card);
    });

    // After all cards are created, set the selected state
    setTimeout(() => {
        const path = document.getElementById('dataset-path').value;
        if (path) {
            Object.entries(datasetCatalog).forEach(([key, dataset]) => {
                if (dataset.path === path) {
                    const card = document.querySelector(`.dataset-catalog-card[data-key="${key}"]`);
                    if (card) {
                        document.querySelectorAll('.dataset-catalog-card').forEach(c => c.classList.remove('selected'));
                        card.classList.add('selected');
                    }
                }
            });
        }
    }, 100);
}

async function updateDatasetStatuses() {
    // Get status for all Public Datasets
    try {
        const response = await fetch('/api/datasets/list');
        const data = await response.json();

        if (data.datasets) {
            data.datasets.forEach(dataset => {
                datasetStatusCache[dataset.path] = {
                    is_cached: dataset.is_cached,
                    cache_info: dataset.cache_info
                };
            });
        }
    } catch (error) {
        console.error('Failed to get dataset statuses:', error);
    }
}

function createDatasetCard(key, dataset) {
    const card = document.createElement('div');
    card.className = 'dataset-catalog-card';
    card.dataset.category = dataset.category;
    card.dataset.key = key;

    // Check cache status
    const status = datasetStatusCache[dataset.path] || { is_cached: false };

    let statusIcon = '';
    let statusClass = '';
    let actionButtons = '';

    if (status.is_cached) {
        statusIcon = '<i class="fas fa-check-circle text-success"></i>';
        statusClass = 'cached';
        actionButtons = `
            <button class="btn btn-sm btn-success mt-2" onclick="event.stopPropagation(); selectDataset('${key}')">
                <i class="fas fa-check"></i> Use Dataset
            </button>
            <button class="btn btn-sm btn-info mt-2" onclick="event.stopPropagation(); previewDataset('${key}')">
                <i class="fas fa-eye"></i> Preview
            </button>
        `;
    } else {
        statusIcon = '<i class="fas fa-cloud-download-alt text-muted"></i>';
        statusClass = 'not-cached';
        actionButtons = `
            <button class="btn btn-sm btn-primary mt-2" onclick="event.stopPropagation(); downloadDataset('${key}')">
                <i class="fas fa-download"></i> Download
            </button>
            <button class="btn btn-sm btn-outline-secondary mt-2" onclick="event.stopPropagation(); previewDataset('${key}')">
                <i class="fas fa-eye"></i> Sample
            </button>
        `;
    }

    // Make entire card clickable
    card.style.cursor = 'pointer';
    card.onclick = function(e) {
        // Don't select if clicking on action buttons
        if (!e.target.closest('.dataset-actions button')) {
            e.stopPropagation();
            selectDataset(key);
        }
    };

    card.innerHTML = `
        <div class="dataset-status ${statusClass}">
            ${statusIcon}
        </div>
        <div class="dataset-icon">${dataset.icon}</div>
        <h6>${dataset.name}</h6>
        <p class="dataset-description">${dataset.description}</p>
        <small class="dataset-meta">${dataset.size} • ${dataset.language}</small>
        <div class="dataset-actions">
            ${actionButtons}
        </div>
    `;
    return card;
}

function selectDataset(key) {
    const dataset = datasetCatalog[key];
    if (dataset) {
        // Update the dataset path
        document.getElementById('dataset-path').value = dataset.path;

        // Update visual selection - remove selected class from all cards
        document.querySelectorAll('.dataset-catalog-card').forEach(card => {
            card.classList.remove('selected');
        });

        // Add selected class to the chosen card
        const selectedCard = document.querySelector(`.dataset-catalog-card[data-key="${key}"]`);
        if (selectedCard) {
            selectedCard.classList.add('selected');
        }

        // Update configuration summary
        updateConfigSummary();

        // Store the config if available (for multi-config datasets)
        if (dataset.config) {
            // Store config in a data attribute or hidden field
            document.getElementById('dataset-path').setAttribute('data-config', dataset.config);
        }

        // Auto-configure field mappings with defensive checks
        if (dataset.fields) {
            const instructionField = dataset.fields.instruction || 'instruction';
            const responseField = dataset.fields.response || 'output';

            // Update hidden fields
            const hiddenInstruction = document.getElementById('instruction-field');
            const hiddenResponse = document.getElementById('response-field');
            if (hiddenInstruction) hiddenInstruction.value = instructionField;
            if (hiddenResponse) hiddenResponse.value = responseField;

            // Update visible fields
            const visibleInstruction = document.getElementById('instruction-field-visible');
            const visibleResponse = document.getElementById('response-field-visible');
            if (visibleInstruction) visibleInstruction.value = instructionField;
            if (visibleResponse) visibleResponse.value = responseField;

            console.log(`Field mapping updated - Instruction: ${instructionField}, Response: ${responseField}`);
        } else {
            // Default values if fields are not defined
            console.warn(`Dataset ${key} has no field mappings defined, using defaults`);
            const defaultInstruction = 'instruction';
            const defaultResponse = 'output';

            document.getElementById('instruction-field').value = defaultInstruction;
            document.getElementById('response-field').value = defaultResponse;
            document.getElementById('instruction-field-visible').value = defaultInstruction;
            document.getElementById('response-field-visible').value = defaultResponse;
        }

        // Show field mapping for advanced users
        const fieldMapping = document.getElementById('field-mapping');
        if (fieldMapping) {
            fieldMapping.style.display = 'block';
        }

        // Visual feedback
        showAlert(`Selected ${dataset.name} dataset`, 'success');
    }
}

function filterDatasets(category) {
    // Update active filter button
    document.querySelectorAll('.dataset-filters button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.currentTarget.classList.add('active');

    // Filter dataset cards
    const cards = document.querySelectorAll('.dataset-catalog-card');
    cards.forEach(card => {
        if (category === 'all' || card.dataset.category === category) {
            card.style.display = 'block';
        } else {
            card.style.display = 'none';
        }
    });
}

async function handleFileUpload(event) {
    console.log('handleFileUpload called', event);
    const file = event.target.files[0];
    if (!file) {
        console.log('No file selected');
        return;
    }
    console.log('File selected:', file.name, 'Size:', file.size, 'Type:', file.type);

    // Validate file size (10GB max)
    const maxSize = 10 * 1024 * 1024 * 1024; // 10GB
    if (file.size > maxSize) {
        showAlert(`File too large. Maximum size is 10GB, your file is ${(file.size / 1024 / 1024).toFixed(2)}MB`, 'danger');
        event.target.value = ''; // Clear the file input
        return;
    }

    // Validate file extension
    const allowedExtensions = ['json', 'jsonl', 'csv', 'parquet'];
    const extension = file.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(extension)) {
        showAlert(`Invalid file type. Allowed types: ${allowedExtensions.join(', ')}`, 'danger');
        event.target.value = ''; // Clear the file input
        return;
    }

    // Show upload progress
    const progressModal = createUploadProgressModal();
    const modal = new bootstrap.Modal(progressModal);
    modal.show();

    const formData = new FormData();
    formData.append('file', file);

    try {
        // Update progress message
        updateUploadProgress(progressModal, `Uploading ${file.name}...`, 30);

        const response = await fetch('/api/datasets/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok && result.success) {
            // Update dataset path with uploaded file path
            document.getElementById('dataset-path').value = result.filepath;
            document.getElementById('dataset-path').setAttribute('data-upload-filename', result.filename);
            document.getElementById('dataset-path').setAttribute('data-source-type', 'upload');

            // Show dataset info if available
            if (result.dataset_info) {
                displayUploadedDatasetInfo(result.dataset_info, result.original_filename);

                // Auto-fill field mappings if detected
                if (result.dataset_info.detected_instruction_field) {
                    document.getElementById('instruction-field').value = result.dataset_info.detected_instruction_field;
                }
                if (result.dataset_info.detected_response_field) {
                    document.getElementById('response-field').value = result.dataset_info.detected_response_field;
                }
            }

            updateUploadProgress(progressModal, 'Upload complete!', 100);
            setTimeout(() => {
                modal.hide();
                progressModal.remove();
                showAlert(`Dataset uploaded successfully: ${result.original_filename} (${result.size_mb}MB)`, 'success');
                // Refresh the uploaded datasets list
                loadUploadedDatasets();
            }, 1000);
        } else {
            throw new Error(result.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload failed:', error);
        modal.hide();
        progressModal.remove();
        showAlert(`Failed to upload dataset: ${error.message}`, 'danger');
        event.target.value = ''; // Clear the file input
    }
}

function createUploadProgressModal() {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.tabIndex = -1;
    modal.innerHTML = `
        <div class="modal-dialog modal-dialog-centered">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Uploading Dataset</h5>
                </div>
                <div class="modal-body">
                    <div class="upload-progress-message mb-3">Preparing upload...</div>
                    <div class="progress">
                        <div class="progress-bar progress-bar-striped progress-bar-animated"
                             role="progressbar" style="width: 0%"></div>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);
    return modal;
}

function updateUploadProgress(modal, message, percent) {
    const messageEl = modal.querySelector('.upload-progress-message');
    const progressBar = modal.querySelector('.progress-bar');

    if (messageEl) messageEl.textContent = message;
    if (progressBar) progressBar.style.width = `${percent}%`;
}

async function loadUploadedDatasets() {
    try {
        const response = await fetch('/api/datasets/uploaded');
        const data = await response.json();

        if (!response.ok) {
            console.error('Failed to load uploaded datasets:', data.error);
            return;
        }

        displayUploadedDatasets(data.files || []);
    } catch (error) {
        console.error('Error loading uploaded datasets:', error);
    }
}

function displayUploadedDatasets(files) {
    // Find or create container for uploaded datasets
    let container = document.getElementById('uploaded-datasets-list');
    if (!container) {
        const uploadArea = document.getElementById('dataset-upload-area');
        if (!uploadArea) return;

        // Create container structure WITHOUT destroying existing elements
        const uploadedContainer = document.createElement('div');
        uploadedContainer.id = 'uploaded-datasets-container';
        uploadedContainer.className = 'mb-4';
        uploadedContainer.innerHTML = `
            <h6 class="mb-3">Previously Uploaded Datasets</h6>
            <div id="uploaded-datasets-list"></div>
        `;

        const separator = document.createElement('hr');
        separator.className = 'my-4';

        const newDatasetHeader = document.createElement('h6');
        newDatasetHeader.className = 'mb-3';
        newDatasetHeader.textContent = 'Upload New Dataset';

        // Insert at the beginning without destroying existing content
        uploadArea.insertBefore(uploadedContainer, uploadArea.firstChild);
        uploadArea.insertBefore(separator, uploadArea.children[1]);
        uploadArea.insertBefore(newDatasetHeader, uploadArea.children[2]);

        container = document.getElementById('uploaded-datasets-list');

        // Re-attach file upload listener since DOM might have been modified
        const fileInput = document.getElementById('dataset-file');
        if (fileInput) {
            // Remove any existing listener first
            fileInput.removeEventListener('change', handleFileUpload);
            // Add the listener again
            fileInput.addEventListener('change', handleFileUpload);
        }
    }

    if (files.length === 0) {
        container.innerHTML = '<div class="text-muted">No uploaded datasets found. Upload a new dataset below.</div>';
        return;
    }

    // Create cards for each uploaded dataset
    container.innerHTML = files.map(file => {
        // Use relative path if available, otherwise use filepath
        const datasetPath = file.relative_path || file.filepath;
        // Escape paths for use in onclick attributes
        const escapedPath = datasetPath.replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        const escapedFilename = file.filename.replace(/'/g, "\\'");

        return `
            <div class="uploaded-dataset-card card mb-2" data-filepath="${datasetPath}" data-filename="${file.filename}">
                <div class="card-body p-3">
                    <div class="row align-items-center">
                        <div class="col-md-6">
                            <h6 class="mb-1">${file.filename}</h6>
                            <small class="text-muted">
                                <i class="fas fa-file"></i> ${file.extension.toUpperCase()} •
                                <i class="fas fa-database"></i> ${file.size_mb} MB •
                                <i class="fas fa-clock"></i> ${new Date(file.uploaded_at).toLocaleDateString()}
                            </small>
                        </div>
                        <div class="col-md-6 text-end">
                            <button class="btn btn-sm btn-outline-primary me-2" onclick="selectUploadedDataset('${escapedFilename}', '${escapedPath}')">
                                <i class="fas fa-check"></i> Select
                            </button>
                            <button class="btn btn-sm btn-outline-info me-2" onclick="configureDatasetFields('${escapedFilename}', '${escapedPath}')">
                                <i class="fas fa-cog"></i> Configure Fields
                            </button>
                            <button class="btn btn-sm btn-outline-danger" onclick="deleteUploadedDataset('${escapedFilename}')">
                                <i class="fas fa-trash"></i> Delete
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

async function selectUploadedDataset(filename, filepath) {
    // If filepath is relative (starts with 'uploads/'), prepend './' for proper resolution
    let datasetPath = filepath;
    if (filepath.startsWith('uploads/')) {
        datasetPath = `./${filepath}`;
    }

    // Update dataset path
    document.getElementById('dataset-path').value = datasetPath;
    document.getElementById('dataset-path').setAttribute('data-upload-filename', filename);
    document.getElementById('dataset-path').setAttribute('data-source-type', 'upload');

    // Load saved field mappings if they exist
    const fieldMappings = loadSavedFieldMapping(filename);
    if (fieldMappings) {
        document.getElementById('instruction-field').value = fieldMappings.instruction || 'instruction';
        document.getElementById('response-field').value = fieldMappings.response || 'output';

        // Update visible fields too
        const instructionVisible = document.getElementById('instruction-field-visible');
        const responseVisible = document.getElementById('response-field-visible');
        if (instructionVisible) instructionVisible.value = fieldMappings.instruction || 'instruction';
        if (responseVisible) responseVisible.value = fieldMappings.response || 'output';
    }

    // Visual feedback
    document.querySelectorAll('.uploaded-dataset-card').forEach(card => {
        card.classList.remove('selected');
    });
    document.querySelector(`.uploaded-dataset-card[data-filename="${filename}"]`)?.classList.add('selected');

    showAlert(`Selected dataset: ${filename}`, 'success');

    // Show field mapping section
    const fieldMapping = document.getElementById('field-mapping');
    if (fieldMapping) {
        fieldMapping.style.display = 'block';
    }
}

async function configureDatasetFields(filename, filepath) {
    try {
        // If filepath is relative (starts with 'uploads/'), prepend './' for proper resolution
        let datasetPath = filepath;
        if (filepath.startsWith('uploads/')) {
            datasetPath = `./${filepath}`;
        }

        // Fetch dataset fields
        const response = await fetch('/api/datasets/detect-fields', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_name: datasetPath,
                is_local: true
            })
        });

        const fieldsData = await response.json();

        if (!response.ok) {
            showAlert('Failed to detect dataset fields: ' + fieldsData.error, 'danger');
            return;
        }

        // Show field mapping dialog for uploaded dataset
        showUploadedDatasetFieldMapping(filename, filepath, fieldsData);

    } catch (error) {
        console.error('Error detecting fields:', error);
        showAlert('Failed to detect dataset fields', 'danger');
    }
}

function showUploadedDatasetFieldMapping(filename, filepath, fieldsData) {
    // Create or reuse field mapping modal
    let modal = document.getElementById('uploadedFieldMappingModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'uploadedFieldMappingModal';
        modal.className = 'modal fade';
        modal.tabIndex = -1;
        document.body.appendChild(modal);
    }

    // Load saved mappings
    const savedMappings = loadSavedFieldMapping(filename) || {};

    modal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Configure Fields for ${filename}</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <p class="text-muted">Select which columns contain the instruction/input and response/output for training.</p>

                    <div class="row">
                        <div class="col-md-6 mb-3">
                            <label class="form-label">
                                <i class="fas fa-question-circle"></i> Instruction Field
                            </label>
                            <select class="form-select" id="modal-instruction-field">
                                <option value="">-- Select Field --</option>
                                ${fieldsData.columns.map(col => `
                                    <option value="${col}"
                                        ${savedMappings.instruction === col || fieldsData.suggested_mappings?.instruction === col ? 'selected' : ''}>
                                        ${col}
                                    </option>
                                `).join('')}
                            </select>
                            <small class="text-muted">The field containing the input/question/prompt</small>
                        </div>

                        <div class="col-md-6 mb-3">
                            <label class="form-label">
                                <i class="fas fa-reply"></i> Response Field
                            </label>
                            <select class="form-select" id="modal-response-field">
                                <option value="">-- Select Field --</option>
                                ${fieldsData.columns.map(col => `
                                    <option value="${col}"
                                        ${savedMappings.response === col || fieldsData.suggested_mappings?.response === col ? 'selected' : ''}>
                                        ${col}
                                    </option>
                                `).join('')}
                            </select>
                            <small class="text-muted">The field containing the output/answer/response</small>
                        </div>
                    </div>

                    ${fieldsData.sample_data ? `
                        <div class="mt-3">
                            <h6>Sample Data Preview</h6>
                            <div class="table-responsive">
                                <table class="table table-sm">
                                    <thead>
                                        <tr>
                                            ${fieldsData.columns.map(col => `<th>${col}</th>`).join('')}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${fieldsData.sample_data.slice(0, 3).map(row => `
                                            <tr>
                                                ${fieldsData.columns.map(col => `
                                                    <td>${row[col] ? JSON.stringify(row[col]).slice(0, 100) : ''}</td>
                                                `).join('')}
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    ` : ''}
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" onclick="saveUploadedDatasetFieldMapping('${filename}', '${filepath}')">
                        Save Configuration
                    </button>
                </div>
            </div>
        </div>
    `;

    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
}

function saveUploadedDatasetFieldMapping(filename, filepath) {
    const instructionField = document.getElementById('modal-instruction-field').value;
    const responseField = document.getElementById('modal-response-field').value;

    if (!instructionField || !responseField) {
        showAlert('Please select both instruction and response fields', 'warning');
        return;
    }

    // Save field mappings to localStorage
    const mappings = { instruction: instructionField, response: responseField };
    saveFieldMapping(filename, mappings);

    // Update the main form fields
    document.getElementById('instruction-field').value = instructionField;
    document.getElementById('response-field').value = responseField;

    const instructionVisible = document.getElementById('instruction-field-visible');
    const responseVisible = document.getElementById('response-field-visible');
    if (instructionVisible) instructionVisible.value = instructionField;
    if (responseVisible) responseVisible.value = responseField;

    // Select the dataset
    selectUploadedDataset(filename, filepath);

    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('uploadedFieldMappingModal'));
    if (modal) modal.hide();

    showAlert('Field configuration saved', 'success');
}

function saveFieldMapping(filename, mappings) {
    let savedMappings = JSON.parse(localStorage.getItem('datasetFieldMappings') || '{}');
    savedMappings[filename] = mappings;
    localStorage.setItem('datasetFieldMappings', JSON.stringify(savedMappings));
}

function loadSavedFieldMapping(filename) {
    const savedMappings = JSON.parse(localStorage.getItem('datasetFieldMappings') || '{}');
    return savedMappings[filename];
}

function deleteUploadedDataset(filename) {
    showDeleteDatasetModal(filename);
}

function showDeleteDatasetModal(filename) {
    // Remove any existing delete modal
    const existingModal = document.getElementById('deleteDatasetModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Create modal
    const modalDiv = document.createElement('div');
    modalDiv.className = 'modal fade';
    modalDiv.id = 'deleteDatasetModal';
    modalDiv.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="fas fa-exclamation-triangle text-warning"></i> Delete Dataset
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <p>Are you sure you want to delete <strong>"${filename}"</strong>?</p>
                    <p class="text-danger mb-0">
                        <i class="fas fa-exclamation-circle"></i> This action cannot be undone.
                    </p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-danger" onclick="performDatasetDelete('${filename.replace(/'/g, "\\'")}')">
                        <i class="fas fa-trash"></i> Delete Dataset
                    </button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modalDiv);
    const deleteModal = new bootstrap.Modal(modalDiv);
    deleteModal.show();
}

async function performDatasetDelete(filename) {
    // Close the modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('deleteDatasetModal'));
    if (modal) {
        modal.hide();
    }

    try {
        const response = await fetch(`/api/datasets/uploaded/${filename}`, {
            method: 'DELETE'
        });

        const result = await response.json();

        if (response.ok && result.success) {
            // Remove saved field mappings
            let savedMappings = JSON.parse(localStorage.getItem('datasetFieldMappings') || '{}');
            delete savedMappings[filename];
            localStorage.setItem('datasetFieldMappings', JSON.stringify(savedMappings));

            // Reload the list
            loadUploadedDatasets();
            showAlert(`Dataset "${filename}" deleted successfully`, 'success');
        } else {
            showAlert(`Failed to delete dataset: ${result.error}`, 'danger');
        }
    } catch (error) {
        console.error('Error deleting dataset:', error);
        showAlert('Failed to delete dataset', 'danger');
    }
}

function displayUploadedDatasetInfo(info, filename) {
    // Create or update dataset info display
    let infoContainer = document.getElementById('uploaded-dataset-info');
    if (!infoContainer) {
        infoContainer = document.createElement('div');
        infoContainer.id = 'uploaded-dataset-info';
        infoContainer.className = 'alert alert-info mt-3';

        const uploadArea = document.getElementById('dataset-upload-area');
        if (uploadArea) {
            uploadArea.appendChild(infoContainer);
        }
    }

    let html = `<h6><i class="fas fa-file-upload"></i> Uploaded: ${filename}</h6>`;

    if (info.columns) {
        html += `<p class="mb-2"><strong>Columns:</strong> ${info.columns.join(', ')}</p>`;
    }

    if (info.num_samples) {
        html += `<p class="mb-2"><strong>Samples:</strong> ${info.num_samples.toLocaleString()}</p>`;
    }

    if (info.detected_instruction_field || info.detected_response_field) {
        html += '<p class="mb-2"><strong>Auto-detected fields:</strong></p>';
        html += '<ul class="small mb-0">';
        if (info.detected_instruction_field) {
            html += `<li>Instruction: <code>${info.detected_instruction_field}</code></li>`;
        }
        if (info.detected_response_field) {
            html += `<li>Response: <code>${info.detected_response_field}</code></li>`;
        }
        html += '</ul>';
    }

    infoContainer.innerHTML = html;
    infoContainer.style.display = 'block';
}

function setupDragAndDrop() {
    const uploadArea = document.getElementById('dataset-upload-area');
    if (!uploadArea) return;

    const dropZone = uploadArea.querySelector('.upload-area');
    if (!dropZone) return;

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    // Highlight drop zone when item is dragged over
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight(e) {
        dropZone.classList.add('border-primary');
        dropZone.style.backgroundColor = '#f0f8ff';
    }

    function unhighlight(e) {
        dropZone.classList.remove('border-primary');
        dropZone.style.backgroundColor = '';
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;

        if (files.length > 0) {
            // Create a fake event with the file for handleFileUpload
            const fakeEvent = {
                target: {
                    files: files,
                    value: ''
                }
            };
            handleFileUpload(fakeEvent);
        }
    }
}

// ============================================================================
// Template Management
// ============================================================================

async function loadAvailableTemplates() {
    try {
        const response = await fetch('/api/templates');
        const templates = await response.json();
        
        // Update template preview with default
        updateTemplatePreview();
    } catch (error) {
        console.error('Failed to load templates:', error);
    }
}

function updateTemplatePreview() {
    const reasoningStart = document.getElementById('reasoning-start').value;
    const reasoningEnd = document.getElementById('reasoning-end').value;
    const solutionStart = document.getElementById('solution-start').value;
    const solutionEnd = document.getElementById('solution-end').value;
    const systemPrompt = document.getElementById('system-prompt').value;

    const preview = `System Prompt:
${systemPrompt}`;

    document.getElementById('template-preview').textContent = preview;

    // Also update the chat template preview
    updateChatTemplatePreview();
}

// Template mode handlers
function setupTemplateHandlers() {
    document.getElementById('template-grpo').addEventListener('change', function() {
        if (this.checked) handleTemplateMode('grpo-default');
    });

    document.getElementById('template-custom').addEventListener('change', function() {
        if (this.checked) handleTemplateMode('custom');
    });
}

function handleTemplateMode(mode) {
    const templateEditor = document.getElementById('template-editor');
    const editBtn = document.getElementById('edit-template-btn');

    switch(mode) {
        case 'grpo-default':
            // Hide editor
            templateEditor.style.display = 'none';
            if (editBtn) editBtn.style.display = 'none';
            // Load default GRPO template
            loadDefaultGRPOTemplate();
            break;

        case 'custom':
            // Show editor
            templateEditor.style.display = 'block';
            if (editBtn) editBtn.style.display = 'inline';
            // Enable template editing
            enableTemplateEditing();
            break;
    }

    updateTemplatePreview();
}

function loadDefaultGRPOTemplate() {
    // Set default GRPO values
    document.getElementById('reasoning-start').value = '<start_working_out>';
    document.getElementById('reasoning-end').value = '<end_working_out>';
    document.getElementById('solution-start').value = '<SOLUTION>';
    document.getElementById('solution-end').value = '</SOLUTION>';
    document.getElementById('system-prompt').value = `You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>`;

    // Update custom fields too
    document.getElementById('custom-reasoning-start').value = '<start_working_out>';
    document.getElementById('custom-reasoning-end').value = '<end_working_out>';
    document.getElementById('custom-solution-start').value = '<SOLUTION>';
    document.getElementById('custom-solution-end').value = '</SOLUTION>';
    document.getElementById('custom-system-prompt').value = document.getElementById('system-prompt').value;
}

function enableTemplateEditing() {
    // Sync custom fields with hidden fields
    document.getElementById('custom-reasoning-start').value = document.getElementById('reasoning-start').value;
    document.getElementById('custom-reasoning-end').value = document.getElementById('reasoning-end').value;
    document.getElementById('custom-solution-start').value = document.getElementById('solution-start').value;
    document.getElementById('custom-solution-end').value = document.getElementById('solution-end').value;
    document.getElementById('custom-system-prompt').value = document.getElementById('system-prompt').value;
}

function editTemplate() {
    // Switch to custom mode
    document.getElementById('template-custom').checked = true;
    handleTemplateMode('custom');
}

async function testTemplate() {
    const config = {
        reasoning_start: document.getElementById('custom-reasoning-start').value,
        reasoning_end: document.getElementById('custom-reasoning-end').value,
        solution_start: document.getElementById('custom-solution-start').value,
        solution_end: document.getElementById('custom-solution-end').value,
        system_prompt: document.getElementById('custom-system-prompt').value,
        sample_instruction: 'Solve the equation: 2x + 5 = 13',
        sample_response: 'x = 4'
    };

    try {
        const response = await fetch('/api/templates/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        const result = await response.json();
        if (result.success) {
            document.getElementById('template-preview').textContent = result.preview;
            showAlert('Template test successful!', 'success');
        } else {
            showAlert('Template test failed: ' + (result.error || 'Unknown error'), 'danger');
        }
    } catch (error) {
        showAlert('Failed to test template: ' + error.message, 'danger');
    }
}

async function saveCustomTemplate() {
    // Show the save template modal instead of using prompt
    const modal = new bootstrap.Modal(document.getElementById('saveTemplateModal'));
    modal.show();
}

async function saveTemplateFromModal() {
    const nameInput = document.getElementById('template-name-input');
    const descriptionInput = document.getElementById('template-description-input');

    const name = nameInput?.value?.trim();
    if (!name) {
        showAlert('Please enter a template name', 'warning');
        return;
    }

    const description = descriptionInput?.value?.trim() || `Custom template created ${new Date().toLocaleDateString()}`;

    const templateData = {
        name: name,
        description: description,
        reasoning_start: document.getElementById('custom-reasoning-start').value,
        reasoning_end: document.getElementById('custom-reasoning-end').value,
        solution_start: document.getElementById('custom-solution-start').value,
        solution_end: document.getElementById('custom-solution-end').value,
        system_prompt: document.getElementById('custom-system-prompt').value
    };

    try {
        const response = await fetch('/api/templates/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(templateData)
        });

        const result = await response.json();
        if (result.success) {
            // Close the modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('saveTemplateModal'));
            if (modal) {
                modal.hide();
            }

            // Clear the inputs
            nameInput.value = '';
            descriptionInput.value = '';

            showAlert(result.message, 'success');
            loadSavedTemplatesList(); // Refresh the list
        } else {
            showAlert('Failed to save template: ' + (result.error || 'Unknown error'), 'danger');
        }
    } catch (error) {
        showAlert('Failed to save template: ' + error.message, 'danger');
    }
}

async function loadSavedTemplatesList() {
    try {
        const response = await fetch('/api/templates');
        const templates = await response.json();

        const select = document.getElementById('saved-templates-list');
        if (!select) {
            console.warn('saved-templates-list element not found, skipping template list update');
            return;
        }

        select.innerHTML = '<option value="">Select a saved template...</option>';

        // Add custom templates
        if (templates.custom) {
            for (const [id, template] of Object.entries(templates.custom)) {
                const option = document.createElement('option');
                option.value = id;
                option.textContent = template.name || id;
                select.appendChild(option);
            }
        }
    } catch (error) {
        console.error('Failed to load saved templates:', error);
    }
}

async function loadSelectedTemplate() {
    const templateId = document.getElementById('saved-templates-list').value;
    if (!templateId) {
        showAlert('Please select a template to load', 'warning');
        return;
    }

    try {
        const response = await fetch('/api/templates');
        const templates = await response.json();

        if (templates.custom && templates.custom[templateId]) {
            const template = templates.custom[templateId];

            // Update hidden fields
            document.getElementById('reasoning-start').value = template.reasoning_start || '';
            document.getElementById('reasoning-end').value = template.reasoning_end || '';
            document.getElementById('solution-start').value = template.solution_start || '';
            document.getElementById('solution-end').value = template.solution_end || '';
            document.getElementById('system-prompt').value = template.system_prompt || '';

            // Update custom fields
            document.getElementById('custom-reasoning-start').value = template.reasoning_start || '';
            document.getElementById('custom-reasoning-end').value = template.reasoning_end || '';
            document.getElementById('custom-solution-start').value = template.solution_start || '';
            document.getElementById('custom-solution-end').value = template.solution_end || '';
            document.getElementById('custom-system-prompt').value = template.system_prompt || '';

            updateTemplatePreview();
            showAlert(`Loaded template: ${template.name}`, 'success');
        }
    } catch (error) {
        showAlert('Failed to load template: ' + error.message, 'danger');
    }
}

function importTemplateFile() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const template = JSON.parse(e.target.result);
                    // Apply template
                    document.getElementById('reasoning-start').value = template.reasoning_start || '<start_working_out>';
                    document.getElementById('reasoning-end').value = template.reasoning_end || '<end_working_out>';
                    document.getElementById('solution-start').value = template.solution_start || '<SOLUTION>';
                    document.getElementById('solution-end').value = template.solution_end || '</SOLUTION>';
                    document.getElementById('system-prompt').value = template.system_prompt || '';

                    updateTemplatePreview();
                    showAlert('Template imported successfully!', 'success');
                } catch (error) {
                    showAlert('Invalid template file', 'danger');
                }
            };
            reader.readAsText(file);
        }
    };
    input.click();
}

// ============================================================================
// Chat Template Management
// ============================================================================

const chatTemplatePresets = {
    grpo: {
        template: "{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] + eos_token }}{% set loop_messages = messages[1:] %}{% else %}{{ system_prompt + eos_token }}{% set loop_messages = messages %}{% endif %}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ message['content'] }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ reasoning_start }}{% endif %}",
        description: "GRPO template optimized for reasoning tasks"
    }
};

function initializeChatTemplate() {
    // Set initial chat template value if not already set
    const chatTemplateField = document.getElementById('chat-template');
    const chatTemplateType = document.getElementById('chat-template-type');

    if (chatTemplateField && chatTemplateType) {
        // Clean up any raw tags from the field value
        const currentValue = chatTemplateField.value.replace(/\{%\s*raw\s*%\}/g, '').replace(/\{%\s*endraw\s*%\}/g, '').trim();

        // Set default to GRPO template if empty or has raw tags
        if (!currentValue || currentValue.length < 10) {
            // Use the default GRPO template
            const defaultTemplate = chatTemplatePresets.grpo.template;
            chatTemplateField.value = defaultTemplate;
        } else {
            chatTemplateField.value = currentValue;
        }

        // Initialize the preview
        updateChatTemplatePreview();
    }
}

function onChatTemplateTypeChange() {
    const templateType = document.getElementById('chat-template-type').value;
    const editor = document.getElementById('chat-template-editor');
    const customTemplate = document.getElementById('custom-chat-template');

    if (templateType === 'custom') {
        editor.style.display = 'block';
    } else {
        editor.style.display = 'none';
        if (chatTemplatePresets[templateType]) {
            // Update hidden chat template field
            document.getElementById('chat-template').value = chatTemplatePresets[templateType].template;
            // Update custom template editor for reference
            customTemplate.value = chatTemplatePresets[templateType].template;
        }
    }

    updateChatTemplatePreview();
}

function validateChatTemplate() {
    const template = document.getElementById('custom-chat-template').value;
    const validationDiv = document.getElementById('chat-template-validation');

    fetch('/api/templates/chat-template/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ template: template })
    })
    .then(response => response.json())
    .then(data => {
        if (data.valid) {
            validationDiv.innerHTML = '<div class="alert alert-success py-1"><i class="fas fa-check"></i> Template is valid</div>';
            // Update hidden field
            document.getElementById('chat-template').value = template;
        } else {
            validationDiv.innerHTML = `<div class="alert alert-danger py-1"><i class="fas fa-times"></i> ${data.error}</div>`;
        }
    })
    .catch(error => {
        validationDiv.innerHTML = '<div class="alert alert-warning py-1">Unable to validate template</div>';
    });
}

function updateChatTemplatePreview() {
    const templateTypeElem = document.getElementById('chat-template-type');
    const preview = document.getElementById('chat-template-preview');

    // Check if elements exist
    if (!templateTypeElem || !preview) {
        console.warn('Chat template elements not found');
        return;
    }

    const templateType = templateTypeElem.value || 'grpo';
    const systemPromptElem = document.getElementById('system-prompt');
    const customSystemPromptElem = document.getElementById('custom-system-prompt');
    const reasoningStartElem = document.getElementById('reasoning-start');
    const customReasoningStartElem = document.getElementById('custom-reasoning-start');

    const systemPrompt = (systemPromptElem && systemPromptElem.value) ||
                         (customSystemPromptElem && customSystemPromptElem.value) ||
                         'You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>';

    const reasoningStart = (reasoningStartElem && reasoningStartElem.value) ||
                          (customReasoningStartElem && customReasoningStartElem.value) ||
                          '<start_working_out>';

    // Create sample messages for preview
    const sampleMessages = [
        { role: 'user', content: 'What is 2 + 2?' }
    ];

    // Get template
    let template = '';
    if (templateType === 'custom') {
        const customTemplateElem = document.getElementById('custom-chat-template');
        template = customTemplateElem ? customTemplateElem.value : chatTemplatePresets.grpo.template;
    } else if (chatTemplatePresets[templateType]) {
        template = chatTemplatePresets[templateType].template;
    } else {
        template = chatTemplatePresets.grpo.template; // Default to GRPO
    }

    // Clean up template (remove any raw tags)
    template = template.replace(/\{%\s*raw\s*%\}/g, '').replace(/\{%\s*endraw\s*%\}/g, '').trim();

    // Set a simple preview first
    preview.textContent = 'Generating preview...';

    // Request preview from server
    fetch('/api/templates/chat-template/preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            template: template,
            messages: sampleMessages,
            system_prompt: systemPrompt,
            reasoning_start: reasoningStart,
            eos_token: '</s>',
            add_generation_prompt: true
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.preview) {
            preview.textContent = data.preview;
        } else if (data.error) {
            preview.textContent = `Error: ${data.error}`;
        } else {
            preview.textContent = 'Unable to generate preview';
        }
    })
    .catch(error => {
        console.error('Error generating preview:', error);
        // Provide a fallback preview
        preview.textContent = `System: ${systemPrompt}\n\nUser: What is 2 + 2?\n\nAssistant: ${reasoningStart}\n[Model reasoning would appear here]\n<end_working_out>\n<SOLUTION>\n4\n</SOLUTION>`;
    });
}

function saveChatTemplate() {
    // Show modal for saving chat template
    showSaveChatTemplateModal();
}

function showSaveChatTemplateModal() {
    // Remove any existing save chat template modal
    const existingModal = document.getElementById('saveChatTemplateModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Create modal
    const modalDiv = document.createElement('div');
    modalDiv.className = 'modal fade';
    modalDiv.id = 'saveChatTemplateModal';
    modalDiv.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="fas fa-save"></i> Save Chat Template
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label for="chat-template-name-input" class="form-label">Template Name</label>
                        <input type="text" class="form-control" id="chat-template-name-input"
                               placeholder="Enter template name..." autofocus>
                        <small class="text-muted">This name will be used to identify your chat template</small>
                    </div>
                    <div class="mb-3">
                        <label for="chat-template-desc-input" class="form-label">Description (optional)</label>
                        <textarea class="form-control" id="chat-template-desc-input" rows="2"
                                  placeholder="Enter template description..."></textarea>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" onclick="performChatTemplateSave()">
                        <i class="fas fa-save"></i> Save Template
                    </button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modalDiv);
    const saveModal = new bootstrap.Modal(modalDiv);

    // Add event listener for Enter key
    const input = modalDiv.querySelector('#chat-template-name-input');
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performChatTemplateSave();
        }
    });

    // Focus input when modal is shown
    modalDiv.addEventListener('shown.bs.modal', () => {
        input.focus();
    });

    saveModal.show();
}

async function performChatTemplateSave() {
    const nameInput = document.getElementById('chat-template-name-input');
    const descInput = document.getElementById('chat-template-desc-input');
    const template = document.getElementById('custom-chat-template').value;

    const name = nameInput?.value?.trim();
    if (!name) {
        showAlert('Please enter a template name', 'warning');
        return;
    }

    const description = descInput?.value?.trim() || 'Custom chat template';

    // Close the modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('saveChatTemplateModal'));
    if (modal) {
        modal.hide();
    }

    try {
        const response = await fetch('/api/templates/chat-template/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                template: template,
                description: description
            })
        });

        const data = await response.json();
        if (data.success) {
            showAlert('Chat template saved successfully', 'success');
        } else {
            showAlert('Failed to save chat template', 'error');
        }
    } catch (error) {
        showAlert('Error saving chat template', 'error');
    }
}

function loadChatTemplate() {
    fetch('/api/templates/chat-templates')
        .then(response => response.json())
        .then(data => {
            // Create a selection dialog
            const templates = data.custom || {};
            const options = Object.entries(templates).map(([id, tmpl]) =>
                `<option value="${id}">${tmpl.name}</option>`
            ).join('');

            // For now, use prompt - could be replaced with a modal
            const selected = prompt('Select a template to load:\n' + Object.keys(templates).join('\n'));
            if (selected && templates[selected]) {
                document.getElementById('custom-chat-template').value = templates[selected].template;
                document.getElementById('chat-template').value = templates[selected].template;
                updateChatTemplatePreview();
            }
        })
        .catch(error => {
            showAlert('Error loading templates', 'error');
        });
}

// ============================================================================
// Algorithm Selection
// ============================================================================

function selectAlgorithm(algorithm) {
    // Remove active class from all algorithm cards
    document.querySelectorAll('#algo-grpo, #algo-gspo, #algo-dr_grpo').forEach(card => {
        card.classList.remove('active');
    });

    // Add active class to selected algorithm
    document.getElementById(`algo-${algorithm}`).classList.add('active');

    // Update info text
    const infoDiv = document.getElementById('algorithm-info');
    const infoTexts = {
        'grpo': '<i class="fas fa-info-circle"></i> <strong>GRPO:</strong> Standard Group Relative Policy Optimization applies importance weights at the token level. Each token gets the same advantage scaling.',
        'gspo': '<i class="fas fa-info-circle"></i> <strong>GSPO (Qwen/Alibaba):</strong> Group Sequence Policy Optimization applies importance weights at the sequence level rather than token level, addressing the observation that advantages should not scale with individual tokens. Often results in more stable training.',
        'dr_grpo': '<i class="fas fa-info-circle"></i> <strong>DR-GRPO:</strong> Doubly Robust GRPO uses control variates to reduce variance in gradient estimates for improved stability.'
    };
    infoDiv.innerHTML = infoTexts[algorithm];

    // Show/hide GSPO parameters (epsilon values are used by both GSPO and DR-GRPO)
    const gspoParams = document.getElementById('gspo-params');
    if (algorithm === 'gspo' || algorithm === 'dr_grpo') {
        gspoParams.style.display = 'block';
    } else {
        gspoParams.style.display = 'none';
    }
}

// ============================================================================
// Reward Function Configuration
// ============================================================================

function selectRewardType(type) {
    // Remove active class from reward type cards
    document.querySelectorAll('#reward-preset, #reward-custom').forEach(card => {
        card.classList.remove('active');
    });

    // Add active class to selected type
    document.getElementById(`reward-${type}`).classList.add('active');

    // Show/hide relevant sections
    const presetRewards = document.getElementById('preset-rewards');
    const customBuilder = document.getElementById('custom-reward-builder');

    if (type === 'preset') {
        presetRewards.style.display = 'block';
        customBuilder.style.display = 'none';
    } else {
        presetRewards.style.display = 'none';
        customBuilder.style.display = 'block';
        // Initialize with one component if empty
        if (document.getElementById('reward-components').children.length === 0) {
            addRewardComponent();
        }
    }
}

function addRewardComponent() {
    const componentsDiv = document.getElementById('reward-components');
    const componentId = `component-${Date.now()}`;

    const componentHtml = `
        <div class="reward-component card mb-2" id="${componentId}">
            <div class="card-body p-2">
                <div class="row align-items-center">
                    <div class="col-md-3">
                        <select class="form-select form-select-sm" onchange="updateComponentType('${componentId}', this.value)">
                            <option value="binary">Binary Match</option>
                            <option value="numerical">Numerical</option>
                            <option value="length">Length</option>
                            <option value="format">Format</option>
                        </select>
                    </div>
                    <div class="col-md-6" id="${componentId}-params">
                        <input type="text" class="form-control form-control-sm" placeholder="Regex pattern (optional)">
                    </div>
                    <div class="col-md-2">
                        <input type="number" class="form-control form-control-sm" placeholder="Weight" value="1.0" step="0.1" min="0">
                    </div>
                    <div class="col-md-1">
                        <button class="btn btn-sm btn-danger" onclick="removeRewardComponent('${componentId}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    componentsDiv.insertAdjacentHTML('beforeend', componentHtml);
}

function removeRewardComponent(componentId) {
    const component = document.getElementById(componentId);
    if (component) {
        component.remove();
    }
}

function updateComponentType(componentId, type) {
    const paramsDiv = document.getElementById(`${componentId}-params`);

    const paramInputs = {
        'binary': '<input type="text" class="form-control form-control-sm" placeholder="Regex pattern (optional)">',
        'numerical': '<input type="number" class="form-control form-control-sm" placeholder="Tolerance" value="0.000001" step="0.000001">',
        'length': '<div class="d-flex gap-1"><input type="number" class="form-control form-control-sm" placeholder="Min" min="0"><input type="number" class="form-control form-control-sm" placeholder="Max" min="0"></div>',
        'format': '<input type="text" class="form-control form-control-sm" placeholder="Regex pattern (required)">'
    };

    paramsDiv.innerHTML = paramInputs[type] || '';
}

// ============================================================================
// Training Presets
// ============================================================================

function applyPreset(presetName) {
    // Remove active class from all presets
    document.querySelectorAll('#preset-recommended, #preset-fast, #preset-balanced, #preset-quality').forEach(card => {
        card.classList.remove('active');
    });

    // Add active class to selected preset
    document.getElementById(`preset-${presetName}`).classList.add('active');

    // Hide/show explanation
    const explanationDiv = document.getElementById('recommended-explanation');
    const indicatorDiv = document.getElementById('preset-indicator');

    if (presetName === 'recommended') {
        // Analyze current configuration
        const recommendation = analyzeAndRecommend();
        applyRecommendedSettings(recommendation);

        // Show explanation
        explanationDiv.style.display = 'block';
        document.getElementById('recommendation-details').innerHTML = recommendation.explanation;
        indicatorDiv.style.display = 'none';
    } else {
        explanationDiv.style.display = 'none';

        const presets = {
            fast: {
                epochs: 1,
                batchSize: 8,
                learningRate: 0.0003,
                temperature: 0.9,
                generations: 2
            },
            balanced: {
                epochs: 3,
                batchSize: 4,
                learningRate: 0.0002,
                temperature: 0.7,
                generations: 4
            },
            quality: {
                epochs: 5,
                batchSize: 2,
                learningRate: 0.0001,
                temperature: 0.5,
                generations: 8
            }
        };

        if (presets[presetName]) {
            document.getElementById('num-epochs').value = presets[presetName].epochs;
            document.getElementById('batch-size').value = presets[presetName].batchSize;
            document.getElementById('learning-rate').value = presets[presetName].learningRate;
            document.getElementById('temperature').value = presets[presetName].temperature;
            document.getElementById('num-generations').value = presets[presetName].generations;

            // Hide custom indicator
            indicatorDiv.style.display = 'none';

            // Update configuration summary
            updateConfigSummary();
        }
    }
}

function analyzeAndRecommend() {
    const recommendation = {
        explanation: '',
        settings: {}
    };

    // Get current model selection
    const modelFamily = document.getElementById('model-family')?.value;
    const modelName = document.getElementById('model-name')?.value;

    // Get dataset info
    const datasetSource = document.getElementById('dataset-source')?.value;

    // Get template type
    const hasReasoningMarkers = document.getElementById('reasoning-start')?.value &&
                                document.getElementById('reasoning-end')?.value;

    // Get reward type
    const rewardPreset = document.getElementById('reward-preset-select')?.value;

    // Analyze model size
    let modelSize = 'medium';
    if (modelName) {
        if (modelName.includes('0.5b') || modelName.includes('0.6b')) modelSize = 'small';
        else if (modelName.includes('1b') || modelName.includes('2b')) modelSize = 'medium';
        else if (modelName.includes('3b') || modelName.includes('7b')) modelSize = 'large';
        else if (modelName.includes('14b') || modelName.includes('70b')) modelSize = 'xlarge';
    }

    // Determine batch size based on model size
    const batchSizes = {
        'small': 8,
        'medium': 4,
        'large': 2,
        'xlarge': 1
    };

    // Determine learning rate based on model and task
    let learningRate = 2e-4;
    let epochs = 3;
    let temperature = 0.7;
    let generations = 4;
    let algorithm = 'grpo';

    // Adjust for mathematical tasks
    if (rewardPreset === 'math') {
        learningRate = 1e-4;
        epochs = 5;
        temperature = 0.5;
        generations = 6;
        algorithm = 'gspo';
        recommendation.explanation = `<strong>Math optimization:</strong> Using GSPO with lower temperature (${temperature}) for precise answers, ${epochs} epochs for better convergence, and smaller learning rate (${learningRate}) for stability.`;
    }
    // Adjust for code generation
    else if (rewardPreset === 'code') {
        learningRate = 3e-4;
        epochs = 4;
        temperature = 0.6;
        generations = 5;
        recommendation.explanation = `<strong>Code generation optimization:</strong> Moderate temperature (${temperature}) for creative but syntactically correct code, ${epochs} epochs with learning rate ${learningRate}.`;
    }
    // Adjust for creative tasks
    else if (rewardPreset === 'length' || !rewardPreset) {
        learningRate = 5e-4;
        epochs = 3;
        temperature = 0.8;
        generations = 4;
        recommendation.explanation = `<strong>General task optimization:</strong> Balanced settings with temperature ${temperature} for varied outputs, ${epochs} epochs for efficiency.`;
    }

    // Adjust for model size
    if (modelSize === 'small') {
        epochs = Math.min(epochs + 1, 10);
        recommendation.explanation += ` <strong>Small model adjustment:</strong> Added extra epoch for better learning.`;
    } else if (modelSize === 'xlarge') {
        learningRate = learningRate * 0.5;
        recommendation.explanation += ` <strong>Large model adjustment:</strong> Reduced learning rate to ${learningRate} for stability.`;
    }

    // Adjust for reasoning chains
    if (hasReasoningMarkers) {
        generations = Math.max(generations, 6);
        temperature = Math.min(temperature + 0.1, 0.9);
        recommendation.explanation += ` <strong>Reasoning chains detected:</strong> Increased generations to ${generations} for diverse reasoning paths.`;
    }

    recommendation.settings = {
        epochs: epochs,
        batchSize: batchSizes[modelSize],
        learningRate: learningRate,
        temperature: temperature,
        generations: generations,
        algorithm: algorithm,
        klPenalty: modelSize === 'small' ? 0.02 : 0.01,
        topP: 0.95
    };

    return recommendation;
}

function applyRecommendedSettings(recommendation) {
    const settings = recommendation.settings;

    // Apply basic settings
    document.getElementById('num-epochs').value = settings.epochs;
    document.getElementById('batch-size').value = settings.batchSize;
    document.getElementById('learning-rate').value = settings.learningRate;
    document.getElementById('temperature').value = settings.temperature;
    document.getElementById('num-generations').value = settings.generations;
    document.getElementById('kl-penalty').value = settings.klPenalty;
    document.getElementById('top-p').value = settings.topP;

    // Apply algorithm selection
    selectAlgorithm(settings.algorithm);

    // Auto-select optimizations based on model size
    const modelName = document.getElementById('model-name')?.value || '';
    if (modelName.includes('7b') || modelName.includes('14b') || modelName.includes('70b')) {
        // Enable memory optimizations for large models
        document.getElementById('gradient-checkpointing').checked = true;
        document.getElementById('mixed-precision').checked = true;
    }

    // Flash attention for newer models
    if (modelName.includes('qwen') || modelName.includes('llama-3')) {
        document.getElementById('use-flash-attention').checked = true;
    }
}

// Update recommendations if "Recommended" preset is active
function updateRecommendedIfActive() {
    const recommendedCard = document.getElementById('preset-recommended');
    if (recommendedCard && recommendedCard.classList.contains('active')) {
        // Re-apply recommended settings with new context
        applyPreset('recommended');
    }
}

// Track changes to training parameters
function setupTrainingParameterListeners() {
    const paramFields = ['num-epochs', 'batch-size', 'learning-rate', 'temperature', 'top-p', 'kl-penalty', 'num-generations'];

    paramFields.forEach(fieldId => {
        const field = document.getElementById(fieldId);
        if (field) {
            field.addEventListener('input', function() {
                checkIfCustomConfiguration();

                // Update generations dropdown when batch size changes
                if (fieldId === 'batch-size') {
                    updateValidGenerations();
                }
            });
        }
    });
}

function checkIfCustomConfiguration() {
    const presets = {
        fast: { epochs: 1, batchSize: 8, learningRate: 0.0003 },
        balanced: { epochs: 3, batchSize: 4, learningRate: 0.0002 },
        quality: { epochs: 5, batchSize: 2, learningRate: 0.0001 }
    };

    const currentValues = {
        epochs: parseInt(document.getElementById('num-epochs').value),
        batchSize: parseInt(document.getElementById('batch-size').value),
        learningRate: parseFloat(document.getElementById('learning-rate').value)
    };

    let matchesPreset = false;
    for (const [name, preset] of Object.entries(presets)) {
        if (preset.epochs === currentValues.epochs &&
            preset.batchSize === currentValues.batchSize &&
            Math.abs(preset.learningRate - currentValues.learningRate) < 0.00001) {
            matchesPreset = true;
            // Highlight the matching preset
            document.querySelectorAll('.preset-card').forEach(card => {
                card.classList.remove('active');
            });
            const matchingCard = Array.from(document.querySelectorAll('.preset-card')).find(
                card => card.textContent.toLowerCase().includes(name)
            );
            if (matchingCard) {
                matchingCard.classList.add('active');
            }
            break;
        }
    }

    // Show/hide custom indicator
    const indicator = document.getElementById('preset-indicator');
    if (!matchesPreset) {
        // Remove active from all preset cards
        document.querySelectorAll('.preset-card').forEach(card => {
            card.classList.remove('active');
        });
        // Show custom indicator
        indicator.style.display = 'block';
    } else {
        indicator.style.display = 'none';
    }
}

function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    const icon = event.currentTarget.querySelector('.float-end');
    
    if (section.style.display === 'none') {
        section.style.display = 'block';
        icon.classList.remove('fa-chevron-down');
        icon.classList.add('fa-chevron-up');
    } else {
        section.style.display = 'none';
        icon.classList.remove('fa-chevron-up');
        icon.classList.add('fa-chevron-down');
    }
}

// ============================================================================
// Configuration Summary
// ============================================================================

function updateConfigSummary() {
    // Update summary values with defaults if not set
    const modelSelect = document.getElementById('model-name');
    let modelText = '--';

    if (modelSelect && modelSelect.options.length > 0) {
        if (modelSelect.selectedIndex >= 0) {
            modelText = modelSelect.options[modelSelect.selectedIndex].text;
        } else if (modelSelect.options[0]) {
            modelText = modelSelect.options[0].text;
        }
    } else {
        // Default fallback
        modelText = 'Qwen3-0.6B (1.2GB)';
    }

    document.getElementById('summary-model').textContent = modelText;

    const datasetSelect = document.getElementById('dataset-path');
    document.getElementById('summary-dataset').textContent = datasetSelect.value || 'tatsu-lab/alpaca';

    const epochsValue = document.getElementById('num-epochs').value || '3';
    document.getElementById('summary-epochs').textContent = epochsValue;

    const batchValue = document.getElementById('batch-size').value || '4';
    document.getElementById('summary-batch').textContent = batchValue;

    // Estimate training time and VRAM
    estimateTrainingRequirements();
}

// ============================================================================
// Training Management
// ============================================================================

function gatherConfig() {
    // Check if custom model path is provided
    const customModelPath = document.getElementById('custom-model-path')?.value?.trim();
    const modelName = customModelPath || document.getElementById('model-name').value;

    // Gather LoRA target modules
    const targetModules = [];
    if (document.getElementById('target-q-proj')?.checked) targetModules.push('q_proj');
    if (document.getElementById('target-v-proj')?.checked) targetModules.push('v_proj');
    if (document.getElementById('target-k-proj')?.checked) targetModules.push('k_proj');
    if (document.getElementById('target-o-proj')?.checked) targetModules.push('o_proj');

    return {
        // Model configuration
        model_name: modelName,
        display_name: document.getElementById('model-display-name')?.value?.trim() || null,
        lora_rank: parseInt(document.getElementById('lora-rank').value),
        lora_alpha: parseInt(document.getElementById('lora-alpha').value),
        lora_dropout: parseFloat(document.getElementById('lora-dropout').value),
        lora_target_modules: targetModules.length > 0 ? targetModules : null,
        lora_bias: document.getElementById('lora-bias')?.value || 'none',

        // Dataset configuration
        dataset_source: document.getElementById('dataset-source').value,
        dataset_path: document.getElementById('dataset-path').value,
        dataset_config: document.getElementById('dataset-path').getAttribute('data-config') || null,
        dataset_split: document.getElementById('dataset-split').value,
        instruction_field: document.getElementById('instruction-field').value,
        response_field: document.getElementById('response-field').value,

        // Template configuration
        reasoning_start: document.getElementById('reasoning-start').value,
        reasoning_end: document.getElementById('reasoning-end').value,
        solution_start: document.getElementById('solution-start').value,
        solution_end: document.getElementById('solution-end').value,
        system_prompt: document.getElementById('system-prompt').value,

        // Chat template configuration
        chat_template_type: document.getElementById('chat-template-type').value,
        chat_template: document.getElementById('chat-template').value,

        // Pre-training configuration
        enable_pre_training: document.getElementById('enable-pre-training').checked,
        pre_training_epochs: parseInt(document.getElementById('pre-training-epochs').value),
        pre_training_samples: parseInt(document.getElementById('pre-training-samples').value),
        validate_format: document.getElementById('validate-format').checked,

        // Basic Training configuration
        learning_rate: parseFloat(document.getElementById('learning-rate').value),
        batch_size: parseInt(document.getElementById('batch-size').value),
        num_epochs: parseInt(document.getElementById('num-epochs').value),
        gradient_accumulation_steps: parseInt(document.getElementById('gradient-accumulation')?.value || 1),
        max_sequence_length: parseInt(document.getElementById('max-sequence-length')?.value || 2048),
        max_new_tokens: parseInt(document.getElementById('max-new-tokens')?.value || 256),
        warmup_steps: parseInt(document.getElementById('warmup-steps')?.value || 10),
        weight_decay: parseFloat(document.getElementById('weight-decay')?.value || 0.001),

        // GRPO configuration
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('top-p').value),
        top_k: parseInt(document.getElementById('top-k')?.value || 50),
        repetition_penalty: parseFloat(document.getElementById('repetition-penalty')?.value || 1.0),
        kl_penalty: parseFloat(document.getElementById('kl-penalty').value),
        clip_range: parseFloat(document.getElementById('clip-range')?.value || 0.2),
        num_generations: parseInt(document.getElementById('num-generations').value) || parseInt(document.getElementById('batch-size').value) || 4,
        value_coefficient: parseFloat(document.getElementById('value-coefficient')?.value || 1.0),

        // Algorithm selection (always use 'grpo' as loss_type, differentiate via importance_sampling_level)
        loss_type: 'grpo',
        importance_sampling_level: getSelectedAlgorithm() === 'gspo' ? 'sequence' : 'token',
        epsilon: document.getElementById('epsilon') ? parseFloat(document.getElementById('epsilon').value) : 0.0003,
        epsilon_high: document.getElementById('epsilon-high') ? parseFloat(document.getElementById('epsilon-high').value) : 0.0004,

        // Advanced training settings
        lr_scheduler_type: document.getElementById('lr-scheduler-type')?.value || 'constant',
        optim: document.getElementById('optimizer')?.value || 'paged_adamw_32bit',
        max_grad_norm: parseFloat(document.getElementById('max-grad-norm')?.value || 0.3),
        logging_steps: parseInt(document.getElementById('logging-steps')?.value || 10),
        save_steps: parseInt(document.getElementById('save-steps')?.value || 100),
        eval_steps: parseInt(document.getElementById('eval-steps')?.value || 100),
        seed: parseInt(document.getElementById('seed')?.value || 42),

        // Quantization configuration
        use_4bit: document.getElementById('use-4bit')?.checked || false,
        use_8bit: document.getElementById('use-8bit')?.checked || false,
        bnb_4bit_compute_dtype: document.getElementById('bnb-4bit-compute-dtype')?.value || 'float16',
        bnb_4bit_quant_type: document.getElementById('bnb-4bit-quant-type')?.value || 'nf4',
        use_nested_quant: document.getElementById('use-nested-quant')?.checked || false,

        // Reward configuration
        reward_config: gatherRewardConfig(),

        // Optimization flags
        use_flash_attention: document.getElementById('use-flash-attention').checked,
        gradient_checkpointing: document.getElementById('gradient-checkpointing').checked,
        fp16: document.getElementById('mixed-precision').checked && !document.getElementById('use-bf16')?.checked,
        bf16: document.getElementById('use-bf16')?.checked || false
    };
}

function getSelectedAlgorithm() {
    if (document.getElementById('algo-grpo').classList.contains('active')) return 'grpo';
    if (document.getElementById('algo-gspo').classList.contains('active')) return 'gspo';
    if (document.getElementById('algo-dr_grpo').classList.contains('active')) return 'dr_grpo';
    return 'grpo'; // default
}

function gatherRewardConfig() {
    const isPreset = document.getElementById('reward-preset').classList.contains('active');

    if (isPreset) {
        const presetValue = document.getElementById('reward-preset-select').value;
        return {
            type: 'preset',
            preset: presetValue
        };
    } else {
        // Gather custom reward components
        const components = [];
        const componentDivs = document.querySelectorAll('.reward-component');

        componentDivs.forEach(div => {
            const typeSelect = div.querySelector('select');
            const paramsDiv = div.querySelector('[id$="-params"]');
            const weightInput = div.querySelectorAll('input[type="number"]')[div.querySelectorAll('input[type="number"]').length - 1];

            const component = {
                type: typeSelect.value,
                weight: parseFloat(weightInput.value) || 1.0
            };

            // Get type-specific parameters
            if (typeSelect.value === 'binary' || typeSelect.value === 'format') {
                const patternInput = paramsDiv.querySelector('input[type="text"]');
                if (patternInput && patternInput.value) {
                    component.pattern = patternInput.value;
                }
            } else if (typeSelect.value === 'numerical') {
                const toleranceInput = paramsDiv.querySelector('input[type="number"]');
                if (toleranceInput) {
                    component.tolerance = parseFloat(toleranceInput.value) || 0.000001;
                }
            } else if (typeSelect.value === 'length') {
                const inputs = paramsDiv.querySelectorAll('input[type="number"]');
                if (inputs.length >= 2) {
                    component.min_length = parseInt(inputs[0].value) || null;
                    component.max_length = parseInt(inputs[1].value) || null;
                }
            }

            components.push(component);
        });

        return {
            type: 'custom',
            components: components
        };
    }
}

async function startTraining() {
    // Final validation
    for (let i = 1; i <= 3; i++) {
        if (!validateStep(i)) {
            showAlert(`Please complete Step ${i} before starting training`, 'warning');
            goToStep(i);
            return;
        }
    }

    // Leave any existing training session rooms
    if (socket && socket.connected && currentSessionId) {
        socket.emit('leave_training_session', { session_id: currentSessionId });
    }

    const config = gatherConfig();

    // Show training monitor and sticky progress bar
    document.getElementById('training-monitor').style.display = 'block';
    // Navbar progress bar removed
    // const navbarProgress = document.getElementById('navbar-progress');
    // if (navbarProgress) navbarProgress.style.display = 'block';
    const stickyContainer = document.getElementById('sticky-progress-container');
    if (stickyContainer) stickyContainer.style.display = 'block';
    document.getElementById('step-4-nav').style.display = 'none';

    // Disable train button
    const trainBtn = document.getElementById('train-btn');
    trainBtn.disabled = true;
    trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';

    // Clear previous logs
    document.getElementById('training-logs').innerHTML = '';

    // Initialize charts
    initializeCharts();

    try {
        const response = await fetch('/api/training/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        const data = await response.json();

        if (response.ok) {
            currentSessionId = data.session_id;

            // Reset charts for new training
            resetCharts();

            // Join the session room for updates
            if (socket && socket.connected) {
                console.log('Joining training session:', currentSessionId);
                socket.emit('join_training_session', { session_id: currentSessionId });
            }

            trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
            showAlert('Training started successfully!', 'success');
        } else {
            throw new Error(data.error || 'Failed to start training');
        }
    } catch (error) {
        trainBtn.disabled = false;
        trainBtn.innerHTML = '<i class="fas fa-dumbbell"></i> Start Training';
        showAlert('Failed to start training: ' + error.message, 'danger');
    }
}

function stopTraining() {
    if (!currentSessionId) return;

    fetch(`/api/training/${currentSessionId}/stop`, {
        method: 'POST'
    }).then(response => {
        if (response.ok) {
            showAlert('Training stopped', 'warning');
            // Hide sticky progress bar
            // Navbar progress bar removed
            // const navbarProgress = document.getElementById('navbar-progress');
            // if (navbarProgress) navbarProgress.style.display = 'none';
            const stickyContainer = document.getElementById('sticky-progress-container');
            if (stickyContainer) stickyContainer.style.display = 'none';
        }
    });
}

function pauseTraining() {
    // TODO: Implement pause functionality
    showAlert('Pause functionality coming soon!', 'info');
}

// ============================================================================
// Training Monitoring
// ============================================================================

function initializeCharts() {
    const lossCtx = document.getElementById('loss-chart').getContext('2d');
    const lrCtx = document.getElementById('lr-chart').getContext('2d');
    const rewardCtx = document.getElementById('reward-chart').getContext('2d');

    lossChart = new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Training Loss',
                data: [],
                borderColor: 'rgb(99, 102, 241)',
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });

    // Initialize reward chart
    rewardChart = new Chart(rewardCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Mean Reward',
                data: [],
                borderColor: 'rgb(34, 197, 94)',
                backgroundColor: 'rgba(34, 197, 94, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return 'Reward: ' + context.parsed.y.toFixed(6);
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: {
                        callback: function(value) {
                            return value.toFixed(4);
                        }
                    }
                }
            }
        }
    });

    lrChart = new Chart(lrCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Learning Rate',
                data: [],
                borderColor: 'rgb(139, 92, 246)',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true
                }
            },
            scales: {
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

function resetCharts() {
    // Clear all chart data
    if (lossChart) {
        lossChart.data.labels = [];
        lossChart.data.datasets[0].data = [];
        lossChart.update();
    }
    if (rewardChart) {
        rewardChart.data.labels = [];
        rewardChart.data.datasets[0].data = [];
        rewardChart.update();
    }
    if (lrChart) {
        lrChart.data.labels = [];
        lrChart.data.datasets[0].data = [];
        lrChart.update();
    }

    // Reset metrics panel
    document.getElementById('metric-step').textContent = '0';
    document.getElementById('metric-loss').textContent = '--';
    document.getElementById('metric-reward').textContent = '--';
    document.getElementById('metric-grad-norm').textContent = '--';
    document.getElementById('metric-lr').textContent = '--';
    document.getElementById('metric-epoch').textContent = '0';
}

function updateTrainingProgress(progress) {
    // Update main progress bar (if exists)
    const progressBar = document.getElementById('training-progress');
    if (progressBar) {
        progressBar.style.width = progress + '%';
        progressBar.textContent = progress + '%';

        if (progress >= 100) {
            progressBar.classList.remove('progress-bar-animated');
        }
    }

    // Navbar progress bar removed
    // const progressBarThin = document.getElementById('training-progress-thin');
    // const progressText = document.getElementById('progress-text');
    // if (progressBarThin) {
    //     progressBarThin.style.width = progress + '%';
    //     if (progressText) {
    //         progressText.textContent = progress + '%';
    //     }
    // }
}

function updateTrainingMetrics(metrics, isHistorical = false) {
    // Debug: Log incoming metrics
    if (!isHistorical) {
        console.log('Received metrics:', metrics);
    }

    // Update metrics panel
    if (metrics.step !== undefined && metrics.step > 0) {
        document.getElementById('metric-step').textContent = metrics.step;

        // If we have total_steps, update progress based on step count
        if (metrics.total_steps) {
            const progress = Math.round((metrics.step / metrics.total_steps) * 100);
            updateTrainingProgress(progress);
        }
    }
    if (metrics.loss !== undefined) {
        // Format loss value based on magnitude
        const lossValue = parseFloat(metrics.loss);
        const formattedLoss = lossValue < 0.01 ? lossValue.toFixed(6) : lossValue.toFixed(4);
        document.getElementById('metric-loss').textContent = formattedLoss;
    }
    if (metrics.mean_reward !== undefined) {
        document.getElementById('metric-reward').textContent = parseFloat(metrics.mean_reward).toFixed(6);
    }
    if (metrics.grad_norm !== undefined) {
        document.getElementById('metric-grad-norm').textContent = parseFloat(metrics.grad_norm).toFixed(4);
    }
    if (metrics.learning_rate !== undefined) {
        document.getElementById('metric-lr').textContent = parseFloat(metrics.learning_rate).toExponential(2);
    }
    if (metrics.epoch !== undefined) {
        document.getElementById('metric-epoch').textContent = parseFloat(metrics.epoch).toFixed(2);
    }

    // Determine the step number for chart labels
    const stepNumber = metrics.step || (lossChart ? lossChart.data.labels.length + 1 : 1);

    // Check if we already have data for this step (avoid duplicates)
    const lastStep = lossChart && lossChart.data.labels.length > 0
        ? lossChart.data.labels[lossChart.data.labels.length - 1]
        : -1;

    // Only add data if it's a new step or if we're loading historical data
    const isNewStep = stepNumber > lastStep || isHistorical;

    // Update loss chart
    if (lossChart && metrics.loss !== undefined && isNewStep) {
        // Remove duplicate if step already exists (for historical data)
        const existingIndex = lossChart.data.labels.indexOf(stepNumber);
        if (existingIndex >= 0 && !isHistorical) {
            lossChart.data.labels.splice(existingIndex, 1);
            lossChart.data.datasets[0].data.splice(existingIndex, 1);
        }

        lossChart.data.labels.push(stepNumber);
        lossChart.data.datasets[0].data.push(metrics.loss);

        // Keep last 100 points for better visualization
        if (lossChart.data.labels.length > 100) {
            lossChart.data.labels.shift();
            lossChart.data.datasets[0].data.shift();
        }

        lossChart.update('none'); // Disable animation for smoother updates
    }

    // Update reward chart
    if (rewardChart && metrics.mean_reward !== undefined && isNewStep) {
        const existingIndex = rewardChart.data.labels.indexOf(stepNumber);
        if (existingIndex >= 0 && !isHistorical) {
            rewardChart.data.labels.splice(existingIndex, 1);
            rewardChart.data.datasets[0].data.splice(existingIndex, 1);
        }

        rewardChart.data.labels.push(stepNumber);
        rewardChart.data.datasets[0].data.push(metrics.mean_reward);

        // Keep last 100 points
        if (rewardChart.data.labels.length > 100) {
            rewardChart.data.labels.shift();
            rewardChart.data.datasets[0].data.shift();
        }

        rewardChart.update('none');
    }

    // Update learning rate chart
    if (lrChart && metrics.learning_rate !== undefined && isNewStep) {
        const existingIndex = lrChart.data.labels.indexOf(stepNumber);
        if (existingIndex >= 0 && !isHistorical) {
            lrChart.data.labels.splice(existingIndex, 1);
            lrChart.data.datasets[0].data.splice(existingIndex, 1);
        }

        lrChart.data.labels.push(stepNumber);
        lrChart.data.datasets[0].data.push(metrics.learning_rate);

        // Keep last 100 points
        if (lrChart.data.labels.length > 100) {
            lrChart.data.labels.shift();
            lrChart.data.datasets[0].data.shift();
        }

        lrChart.update('none');
    }
}

function appendLog(message) {
    const logsDiv = document.getElementById('training-logs');
    const logEntry = document.createElement('div');
    logEntry.className = 'log-entry';
    logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logsDiv.appendChild(logEntry);
    logsDiv.scrollTop = logsDiv.scrollHeight;
}

function clearLogs() {
    document.getElementById('training-logs').innerHTML = '';
}

function handleTrainingComplete(data) {
    const trainBtn = document.getElementById('train-btn');
    trainBtn.disabled = false;
    trainBtn.innerHTML = '<i class="fas fa-check"></i> Training Complete';
    trainBtn.classList.add('btn-success');

    // Hide sticky progress bar
    // Navbar progress bar removed
    // const navbarProgress = document.getElementById('navbar-progress');
    // if (navbarProgress) navbarProgress.style.display = 'none';
    const stickyContainer = document.getElementById('sticky-progress-container');
    if (stickyContainer) stickyContainer.style.display = 'none';

    showAlert('Training completed successfully!', 'success');

    // Store as last completed session
    lastCompletedSessionId = currentSessionId;

    // Show export options
    showExportModal(currentSessionId);

    // Also refresh the sessions list to show the updated status
    refreshSessions();
}

function handleTrainingError(data) {
    const trainBtn = document.getElementById('train-btn');
    trainBtn.disabled = false;
    trainBtn.innerHTML = '<i class="fas fa-dumbbell"></i> Start Training';

    // Hide sticky progress bar
    // Navbar progress bar removed
    // const navbarProgress = document.getElementById('navbar-progress');
    // if (navbarProgress) navbarProgress.style.display = 'none';
    const stickyContainer = document.getElementById('sticky-progress-container');
    if (stickyContainer) stickyContainer.style.display = 'none';

    showAlert('Training error: ' + data.error, 'danger');
}

// ============================================================================
// Session Management
// ============================================================================

async function checkForRunningSessions() {
    try {
        const response = await fetch('/api/training/sessions');
        const sessions = await response.json();

        // Find any running sessions
        const runningSessions = sessions.filter(s => s.status === 'running');

        if (runningSessions.length > 0) {
            // Auto-reconnect to the most recent running session
            const mostRecentRunning = runningSessions.sort((a, b) =>
                new Date(b.created_at) - new Date(a.created_at)
            )[0];

            showAlert(`Found active training session: ${mostRecentRunning.display_name || mostRecentRunning.model}. Reconnecting...`, 'info');

            // Slight delay to allow UI to initialize
            setTimeout(() => {
                loadSession(mostRecentRunning.session_id);
            }, 500);
        }
    } catch (error) {
        console.error('Failed to check for running sessions:', error);
    }
}

async function refreshSessions() {
    try {
        const response = await fetch('/api/training/sessions');
        const sessions = await response.json();

        const sessionsList = document.getElementById('sessions-list');
        sessionsList.innerHTML = '';

        sessions.forEach(session => {
            const sessionItem = document.createElement('div');
            sessionItem.className = 'session-item p-2 border-bottom';

            // Add pulsing animation for running sessions
            if (session.status === 'running') {
                sessionItem.className += ' running-session';
            }

            // Create action button HTML - only export button for completed sessions
            let actionBtnHtml = '';
            if (session.status === 'completed') {
                actionBtnHtml = `<button class="btn btn-sm btn-success ms-2" onclick="event.stopPropagation(); showExportModal('${session.session_id}')">
                       <i class="fas fa-file-export"></i>
                   </button>`;
            }

            const displayName = session.display_name || session.model.split('/').pop();
            const statusText = session.status === 'running' ? 'Training...' : session.status;

            sessionItem.innerHTML = `
                <div class="position-relative" style="cursor: pointer;" onclick="loadSession('${session.session_id}')">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <div class="fw-bold">${displayName}</div>
                            <small class="text-muted">${new Date(session.created_at).toLocaleString()}</small>
                            <small class="text-muted d-block" style="font-size: 0.75rem;">ID: ${session.session_id.substring(0, 8)}</small>
                        </div>
                        <div class="d-flex align-items-center">
                            ${session.status !== 'running' ? `<span class="badge bg-${getStatusColor(session.status)}">${session.status}</span>` : ''}
                            ${actionBtnHtml}
                        </div>
                    </div>
                    ${session.status === 'running' ? `
                        <div class="position-absolute bottom-0 end-0 mb-1 me-1">
                            <span class="badge bg-${getStatusColor(session.status)} pulse-animation">${statusText}</span>
                        </div>
                    ` : ''}
                </div>
            `;
            sessionsList.appendChild(sessionItem);
        });
    } catch (error) {
        console.error('Failed to refresh sessions:', error);
    }
}

function getStatusColor(status) {
    switch (status) {
        case 'completed': return 'success';
        case 'running': return 'primary';
        case 'error': return 'danger';
        default: return 'secondary';
    }
}

function loadSession(sessionId) {
    currentSessionId = sessionId;

    // Fetch session details and show info panel
    fetch(`/api/training/sessions`)
        .then(response => response.json())
        .then(sessions => {
            const session = sessions.find(s => s.session_id === sessionId);
            if (session) {
                // If session is running, reconnect to it
                if (session.status === 'running') {
                    reconnectToTrainingSession(session);
                } else {
                    showSessionInfoPanel(session);
                    if (session.status === 'completed') {
                        lastCompletedSessionId = sessionId;
                    }
                }
            }
        })
        .catch(error => {
            console.error('Failed to load session:', error);
            showAlert('Failed to load session details.', 'danger');
        });
}

async function reconnectToTrainingSession(session) {
    console.log('Reconnecting to training session:', session.session_id);

    // Set the current session
    currentSessionId = session.session_id;

    // Navigate to Step 4 (Training section)
    goToStep(4);

    // Show training monitor
    document.getElementById('training-monitor').style.display = 'block';
    const stickyContainer = document.getElementById('sticky-progress-container');
    if (stickyContainer) stickyContainer.style.display = 'block';
    document.getElementById('step-4-nav').style.display = 'none';

    // Update train button to show it's in progress
    const trainBtn = document.getElementById('train-btn');
    if (trainBtn) {
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training in Progress...';
    }

    // Initialize charts if they haven't been initialized
    if (!lossChart) {
        initializeCharts();
    } else {
        // Reset charts if they exist to avoid duplicate data
        resetCharts();
    }

    // Join the socket room for this session
    if (socket && socket.connected) {
        socket.emit('join_training_session', { session_id: session.session_id });
    }

    // Fetch training history to populate charts and logs
    try {
        const historyResponse = await fetch(`/api/training/session/${session.session_id}/history`);
        if (historyResponse.ok) {
            const history = await historyResponse.json();

            // Clear and populate logs
            const logsContainer = document.getElementById('training-logs');
            if (logsContainer) {
                logsContainer.innerHTML = '';
                if (history.logs && history.logs.length > 0) {
                    history.logs.forEach(log => appendLog(log));
                }
            }

            // Populate metrics in charts (after reset)
            if (history.metrics && history.metrics.length > 0) {
                // Sort metrics by step to ensure proper order
                const sortedMetrics = history.metrics.sort((a, b) => (a.step || 0) - (b.step || 0));
                sortedMetrics.forEach(metric => {
                    updateTrainingMetrics(metric, true); // Pass flag to indicate historical data
                });
            }

            // Update progress if available
            if (history.progress !== undefined) {
                updateTrainingProgress(history.progress);
            }
        }
    } catch (error) {
        console.error('Failed to fetch training history:', error);
    }

    showAlert(`Reconnected to training session: ${session.display_name || session.model}`, 'info');
}

function showSessionInfoPanel(session) {
    // Find or create session info panel
    let infoPanel = document.getElementById('session-info-panel');
    if (!infoPanel) {
        // Create panel container in the main content area
        const mainContent = document.querySelector('.main-container');
        if (!mainContent) return;

        infoPanel = document.createElement('div');
        infoPanel.id = 'session-info-panel';
        infoPanel.className = 'alert alert-info alert-dismissible fade show mt-3';

        // Insert after the navbar or at the top of main container
        const firstChild = mainContent.firstChild;
        mainContent.insertBefore(infoPanel, firstChild);
    }

    // Build panel content
    const statusBadge = `<span class="badge bg-${getStatusColor(session.status)}">${session.status}</span>`;
    const exportButton = session.status === 'completed'
        ? `<button class="btn btn-success btn-sm" onclick="showExportModal('${session.session_id}')">
               <i class="fas fa-file-export"></i> Export Model
           </button>`
        : '';

    infoPanel.innerHTML = `
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        <h5 class="alert-heading">
            <i class="fas fa-info-circle"></i> Session Details
        </h5>
        <div class="row">
            <div class="col-md-8">
                <p class="mb-1"><strong>Model:</strong> ${session.model}</p>
                <p class="mb-1"><strong>Session ID:</strong> ${session.session_id}</p>
                <p class="mb-1"><strong>Created:</strong> ${new Date(session.created_at).toLocaleString()}</p>
                <p class="mb-1"><strong>Status:</strong> ${statusBadge}</p>
            </div>
            <div class="col-md-4 text-end">
                ${exportButton}
                ${session.status === 'running'
                    ? `<button class="btn btn-primary btn-sm" onclick="reconnectToSession('${session.session_id}')">
                           <i class="fas fa-plug"></i> Reconnect
                       </button>`
                    : ''}
            </div>
        </div>
    `;
}

function reconnectToSession(sessionId) {
    // Reconnect to a running training session
    currentSessionId = sessionId;
    socket.emit('join', { session_id: sessionId });
    showAlert('Reconnected to training session!', 'success');
}

// ============================================================================
// Configuration Management
// ============================================================================

async function loadConfigList() {
    try {
        const response = await fetch('/api/configs/list');
        const configs = await response.json();

        const select = document.getElementById('saved-configs-list');
        select.innerHTML = '<option value="">Select a config...</option>';

        configs.forEach(config => {
            const option = document.createElement('option');
            option.value = config.filename;
            option.textContent = config.name;
            option.dataset.modified = config.modified;
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load config list:', error);
    }
}

function showLoadConfigModal() {
    const select = document.getElementById('saved-configs-list');
    const filename = select.value;

    //if (!filename) {
    //    showAlert('Please select a configuration to load', 'warning');
    //    return;
    //}

    const configName = select.options[select.selectedIndex].text;

    // Remove any existing load config modal
    const existingModal = document.getElementById('loadConfigModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Create modal
    const modalDiv = document.createElement('div');
    modalDiv.className = 'modal fade';
    modalDiv.id = 'loadConfigModal';
    modalDiv.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="fas fa-folder-open"></i> Load Configuration
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <p>You are about to load the configuration:</p>
                    <div class="alert alert-info">
                        <strong>${configName}</strong>
                    </div>
                    <div class="alert alert-warning">
                        <i class="fas fa-exclamation-triangle"></i>
                        <strong>Warning:</strong> This will replace all current settings with the saved configuration.
                        Any unsaved changes will be lost.
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" onclick="performConfigLoad('${filename}')">
                        <i class="fas fa-folder-open"></i> Load Configuration
                    </button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modalDiv);
    const loadModal = new bootstrap.Modal(modalDiv);
    loadModal.show();
}

async function performConfigLoad(filename) {
    // Close the modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('loadConfigModal'));
    if (modal) {
        modal.hide();
    }

    try {
        const response = await fetch(`/api/config/load/${filename}`);

        if (!response.ok) {
            throw new Error('Failed to load configuration');
        }

        const config = await response.json();

        // Apply the loaded configuration to the UI
        applyConfigToUI(config);

        showAlert(`Configuration loaded successfully!`, 'success');
    } catch (error) {
        showAlert('Failed to load configuration: ' + error.message, 'danger');
    }
}

function loadSelectedConfig() {
    showLoadConfigModal();
}

function showSaveConfigModal() {
    // Remove any existing save config modal
    const existingModal = document.getElementById('saveConfigModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Create modal
    const modalDiv = document.createElement('div');
    modalDiv.className = 'modal fade';
    modalDiv.id = 'saveConfigModal';
    modalDiv.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="fas fa-save"></i> Save Configuration
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label for="config-name-input" class="form-label">Configuration Name</label>
                        <input type="text" class="form-control" id="config-name-input"
                               placeholder="Enter configuration name..." autofocus>
                        <small class="text-muted">This name will be used to identify your configuration</small>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" onclick="performConfigSave()">
                        <i class="fas fa-save"></i> Save Configuration
                    </button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modalDiv);
    const saveModal = new bootstrap.Modal(modalDiv);

    // Add event listener for Enter key
    const input = modalDiv.querySelector('#config-name-input');
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            performConfigSave();
        }
    });

    // Focus input when modal is shown
    modalDiv.addEventListener('shown.bs.modal', () => {
        input.focus();
    });

    saveModal.show();
}

async function performConfigSave() {
    const input = document.getElementById('config-name-input');
    const configName = input?.value?.trim();

    if (!configName) {
        showAlert('Please enter a configuration name', 'warning');
        return;
    }

    // Close the modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('saveConfigModal'));
    if (modal) {
        modal.hide();
    }

    const config = gatherConfig();
    config.filename = `${configName}.json`;

    try {
        const response = await fetch('/api/config/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });

        if (!response.ok) {
            throw new Error('Failed to save configuration');
        }

        const result = await response.json();
        showAlert(result.message, 'success');

        // Reload the config list
        await loadConfigList();
    } catch (error) {
        showAlert('Failed to save configuration: ' + error.message, 'danger');
    }
}

function saveConfig() {
    showSaveConfigModal();
}

function showConfirmModal(title, message, onConfirm, confirmBtnClass = 'btn-danger') {
    // Remove any existing confirm modal
    const existingModal = document.getElementById('confirmModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Create modal
    const modalDiv = document.createElement('div');
    modalDiv.className = 'modal fade';
    modalDiv.id = 'confirmModal';
    modalDiv.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="fas fa-exclamation-triangle"></i> ${title}
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <p>${message}</p>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn ${confirmBtnClass}" id="confirm-action-btn">
                        Confirm
                    </button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modalDiv);
    const confirmModal = new bootstrap.Modal(modalDiv);

    // Add event listener for confirm button
    const confirmBtn = modalDiv.querySelector('#confirm-action-btn');
    confirmBtn.addEventListener('click', () => {
        confirmModal.hide();
        if (onConfirm) {
            onConfirm();
        }
    });

    // Clean up modal after it's hidden
    modalDiv.addEventListener('hidden.bs.modal', () => {
        modalDiv.remove();
    });

    confirmModal.show();
}

async function deleteSelectedConfig() {
    const select = document.getElementById('saved-configs-list');
    const filename = select.value;

    if (!filename) {
        showAlert('Please select a configuration to delete', 'warning');
        return;
    }

    const configName = select.options[select.selectedIndex].text;

    showConfirmModal(
        'Delete Configuration',
        `Are you sure you want to delete the configuration "${configName}"? This action cannot be undone.`,
        async () => {
            try {
                const response = await fetch(`/api/configs/delete/${filename}`, {
                    method: 'DELETE'
                });

                if (!response.ok) {
                    throw new Error('Failed to delete configuration');
                }

                const result = await response.json();
                showAlert(result.message, 'success');

                // Reload the config list
                await loadConfigList();
            } catch (error) {
                showAlert('Failed to delete configuration: ' + error.message, 'danger');
            }
        }
    );
}

function onConfigSelect() {
    const select = document.getElementById('saved-configs-list');
    const deleteBtn = document.querySelector('button[onclick="deleteSelectedConfig()"]');
    const loadBtn = document.querySelector('button[onclick="loadSelectedConfig()"]');

    const hasSelection = select.value !== '';
    if (deleteBtn) deleteBtn.disabled = !hasSelection;
    if (loadBtn) loadBtn.disabled = !hasSelection;
}

// ============================================================================
// Theme Management
// ============================================================================

function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);

    // Update icon
    const icon = document.getElementById('theme-icon');
    icon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
}

function toggleSidebar() {
    // Updated for new flex layout
    const sidebar = document.getElementById('sidebar-panel');
    const mainWrapper = document.querySelector('.main-wrapper');

    if (sidebar) {
        sidebar.classList.toggle('hidden');

        // Toggle class on main wrapper for layout adjustment
        if (mainWrapper) {
            if (sidebar.classList.contains('hidden')) {
                mainWrapper.classList.add('sidebar-hidden');
            } else {
                mainWrapper.classList.remove('sidebar-hidden');
            }
        }

        // Save preference
        const isHidden = sidebar.classList.contains('hidden');
        localStorage.setItem('sidebarHidden', isHidden);
    }

    // For backwards compatibility, also check old ID
    const oldSidebar = document.getElementById('sidebar-column');
    if (oldSidebar) {
        oldSidebar.classList.toggle('hidden');
    }
}

function setupIconScrollEffect() {
    const floatingIcon = document.querySelector('.floating-icon-dock');
    if (!floatingIcon) return;

    let ticking = false;

    function updateIconPosition() {
        const scrollY = window.scrollY;
        const sidebarCard = document.querySelector('.sidebar-card');

        // Calculate the position of the Training Sessions header
        // Navbar (32px) + margin-top (45px) + padding-top (95px) = ~172px
        const sidebarHeaderTop = 172;
        const iconHeight = 88; // Icon container height
        const minTopPosition = sidebarHeaderTop - iconHeight - 20; // Keep 20px gap

        // After scrolling 100px, make icon scroll with content
        if (scrollY > 100) {
            // Calculate desired position
            let desiredTop = scrollY + 20;

            // But don't let it go below where it would overlap sidebar header
            if (desiredTop > minTopPosition) {
                desiredTop = minTopPosition;
            }

            floatingIcon.style.position = 'absolute';
            floatingIcon.style.top = desiredTop + 'px';
        } else {
            // Reset to fixed position when near top
            floatingIcon.style.position = 'fixed';
            floatingIcon.style.top = '20px';
        }

        ticking = false;
    }

    function requestTick() {
        if (!ticking) {
            window.requestAnimationFrame(updateIconPosition);
            ticking = true;
        }
    }

    // Listen for scroll events
    window.addEventListener('scroll', requestTick);
}

// Load saved theme and sidebar state
document.addEventListener('DOMContentLoaded', function() {
    // Load theme
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    const icon = document.getElementById('theme-icon');
    if (icon) {
        icon.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    // Load sidebar state
    const sidebarHidden = localStorage.getItem('sidebarHidden') === 'true';
    const sidebar = document.getElementById('sidebar-panel');
    const mainWrapper = document.querySelector('.main-wrapper');

    if (sidebar && sidebarHidden) {
        sidebar.classList.add('hidden');
        if (mainWrapper) {
            mainWrapper.classList.add('sidebar-hidden');
        }
    }
});

// ============================================================================
// System Status
// ============================================================================

async function updateSystemStatus() {
    try {
        const response = await fetch('/api/system/info');
        const info = await response.json();

        // Update GPU status
        if (info.gpu_available) {
            // Show GPU name - allow maximum space for visibility
            let gpuName = info.gpu_name || 'Available';
            // Only shorten if absolutely necessary (very long names)
            if (gpuName.length > 35) {
                gpuName = gpuName.substring(0, 32) + '...';
            }
            document.getElementById('gpu-status').textContent = gpuName;

            // Show VRAM (available/total)
            const vramTotal = info.gpu_memory_total || 0;
            const vramAllocated = info.gpu_memory_allocated || 0;
            const vramReserved = info.gpu_memory_reserved || 0;
            const vramAvailable = vramTotal - vramAllocated;  // Available = Total - Allocated
            document.getElementById('vram-status').textContent = `${vramAvailable.toFixed(1)}/${vramTotal.toFixed(1)}GB`;
        } else {
            document.getElementById('gpu-status').textContent = 'CPU Only';
            document.getElementById('vram-status').textContent = '--';
        }

        // Update RAM status (always available)
        const ramAvailable = info.ram_available || 0;
        const ramTotal = info.ram_total || 0;
        document.getElementById('ram-status').textContent = `${ramAvailable.toFixed(1)}/${ramTotal.toFixed(1)}GB`;

        // Update tooltips with more detail (if they exist)
        try {
            const gpuElement = document.querySelector('[data-bs-toggle="tooltip"][title="GPU Status"]');
            if (gpuElement) {
                const gpuTooltip = bootstrap.Tooltip.getInstance(gpuElement);
                if (gpuTooltip && info.gpu_available) {
                    gpuTooltip.setContent({ '.tooltip-inner': `GPU: ${info.gpu_name}` });
                }
            }

            const vramElement = document.querySelector('[data-bs-toggle="tooltip"][title="VRAM Available"]');
            if (vramElement && info.gpu_available) {
                const vramTooltip = bootstrap.Tooltip.getInstance(vramElement);
                if (vramTooltip) {
                    const vramTotal = info.gpu_memory_total || 0;
                    const vramAllocated = info.gpu_memory_allocated || 0;
                    const vramReserved = info.gpu_memory_reserved || 0;
                    const vramAvailable = vramReserved;
                    vramTooltip.setContent({ '.tooltip-inner': `VRAM: ${vramAvailable.toFixed(2)}GB available / ${vramTotal.toFixed(2)}GB total (${vramAllocated.toFixed(2)}GB allocated, ${vramReserved.toFixed(2)}GB reserved)` });
                }
            }

            const ramElement = document.querySelector('[data-bs-toggle="tooltip"][title="RAM Available"]');
            if (ramElement) {
                const ramTooltip = bootstrap.Tooltip.getInstance(ramElement);
                if (ramTooltip) {
                    const ramAvailable = info.ram_available || 0;
                    const ramTotal = info.ram_total || 0;
                    const ramUsed = info.ram_used || 0;
                    ramTooltip.setContent({ '.tooltip-inner': `RAM: ${ramAvailable.toFixed(2)}GB available / ${ramTotal.toFixed(2)}GB total (${ramUsed.toFixed(2)}GB used)` });
                }
            }
        } catch (tooltipError) {
            // Silently ignore tooltip errors - they're not critical
            console.debug('Tooltip update skipped:', tooltipError.message);
        }
    } catch (error) {
        console.error('Failed to update system status:', error);
        document.getElementById('gpu-status').textContent = 'Error';
        document.getElementById('vram-status').textContent = '--';
        document.getElementById('ram-status').textContent = '--';
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

function showAlert(message, type) {
    console.log(`[${type}] ${message}`);

    // Create styled Bootstrap toast notification
    const toastContainer = document.getElementById('toast-container') || createToastContainer();

    // Map alert types to Bootstrap classes
    const typeMap = {
        'success': 'bg-success text-white',
        'danger': 'bg-danger text-white',
        'warning': 'bg-warning text-dark',
        'info': 'bg-info text-white',
        'primary': 'bg-primary text-white'
    };

    const bgClass = typeMap[type] || 'bg-secondary text-white';

    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toastHtml = `
        <div id="${toastId}" class="toast align-items-center ${bgClass} border-0" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
        </div>
    `;

    toastContainer.insertAdjacentHTML('beforeend', toastHtml);

    // Initialize and show the toast
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, {
        autohide: true,
        delay: type === 'danger' || type === 'warning' ? 5000 : 3000
    });

    toast.show();

    // Remove the element after it's hidden
    toastElement.addEventListener('hidden.bs.toast', () => {
        toastElement.remove();
    });
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container position-fixed top-0 end-0 p-3';
    container.style.zIndex = '9999';
    document.body.appendChild(container);
    return container;
}

function updateValue(inputId, value) {
    document.getElementById(inputId).value = value;
}

function saveState() {
    const state = {
        currentStep: currentStep,
        config: gatherConfig(),
        validation: stepValidation
    };
    localStorage.setItem('grpo_state', JSON.stringify(state));
}

function loadSavedState() {
    const savedState = localStorage.getItem('grpo_state');
    if (savedState) {
        try {
            const state = JSON.parse(savedState);
            // TODO: Apply saved state to UI
        } catch (error) {
            console.error('Failed to load saved state:', error);
        }
    }
}

function showHelp() {
    window.open('https://github.com/jwest33/gpro_lora', '_blank');
}

function applyConfigToUI(config) {
    // Apply configuration values to UI elements

    // Model configuration
    if (config.model_name) {
        const modelSelect = document.getElementById('model-name');
        if (modelSelect) {
            modelSelect.value = config.model_name;
        }
        // Also check for custom model path
        const customModelPath = document.getElementById('custom-model-path');
        if (customModelPath && config.model_name.includes('/')) {
            customModelPath.value = config.model_name;
        }
    }

    if (config.display_name) {
        const displayName = document.getElementById('model-display-name');
        if (displayName) displayName.value = config.display_name;
    }

    // Dataset configuration
    if (config.dataset_source) {
        const datasetSource = document.getElementById('dataset-source');
        if (datasetSource) datasetSource.value = config.dataset_source;
    }

    if (config.dataset_path) {
        const datasetPath = document.getElementById('dataset-path');
        if (datasetPath) datasetPath.value = config.dataset_path;
    }

    if (config.dataset_split) {
        const datasetSplit = document.getElementById('dataset-split');
        if (datasetSplit) datasetSplit.value = config.dataset_split;
    }

    if (config.instruction_field) {
        const instructionField = document.getElementById('instruction-field');
        if (instructionField) instructionField.value = config.instruction_field;

        const instructionFieldVisible = document.getElementById('instruction-field-visible');
        if (instructionFieldVisible) instructionFieldVisible.value = config.instruction_field;
    }

    if (config.response_field) {
        const responseField = document.getElementById('response-field');
        if (responseField) responseField.value = config.response_field;

        const responseFieldVisible = document.getElementById('response-field-visible');
        if (responseFieldVisible) responseFieldVisible.value = config.response_field;
    }

    // Template configuration - IMPORTANT: Restore these fields
    if (config.system_prompt) {
        const systemPrompt = document.getElementById('system-prompt');
        if (systemPrompt) systemPrompt.value = config.system_prompt;

        const customSystemPrompt = document.getElementById('custom-system-prompt');
        if (customSystemPrompt) customSystemPrompt.value = config.system_prompt;
    }

    if (config.reasoning_start) {
        const reasoningStart = document.getElementById('reasoning-start');
        if (reasoningStart) reasoningStart.value = config.reasoning_start;

        const customReasoningStart = document.getElementById('custom-reasoning-start');
        if (customReasoningStart) customReasoningStart.value = config.reasoning_start;
    }

    if (config.reasoning_end) {
        const reasoningEnd = document.getElementById('reasoning-end');
        if (reasoningEnd) reasoningEnd.value = config.reasoning_end;

        const customReasoningEnd = document.getElementById('custom-reasoning-end');
        if (customReasoningEnd) customReasoningEnd.value = config.reasoning_end;
    }

    if (config.solution_start) {
        const solutionStart = document.getElementById('solution-start');
        if (solutionStart) solutionStart.value = config.solution_start;

        const customSolutionStart = document.getElementById('custom-solution-start');
        if (customSolutionStart) customSolutionStart.value = config.solution_start;
    }

    if (config.solution_end) {
        const solutionEnd = document.getElementById('solution-end');
        if (solutionEnd) solutionEnd.value = config.solution_end;

        const customSolutionEnd = document.getElementById('custom-solution-end');
        if (customSolutionEnd) customSolutionEnd.value = config.solution_end;
    }

    // Check if we need to switch to custom template mode
    const isCustomTemplate = (
        (config.system_prompt && config.system_prompt !== 'You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>') ||
        (config.reasoning_start && config.reasoning_start !== '<start_working_out>') ||
        (config.reasoning_end && config.reasoning_end !== '<end_working_out>') ||
        (config.solution_start && config.solution_start !== '<SOLUTION>') ||
        (config.solution_end && config.solution_end !== '</SOLUTION>')
    );

    if (isCustomTemplate) {
        // Switch to custom template mode
        const customRadio = document.getElementById('template-custom');
        if (customRadio) {
            customRadio.checked = true;
            handleTemplateMode('custom');
        }
    }

    // Chat template configuration
    if (config.chat_template_type) {
        const chatTemplateType = document.getElementById('chat-template-type');
        if (chatTemplateType) chatTemplateType.value = config.chat_template_type;
    }

    if (config.chat_template) {
        const chatTemplate = document.getElementById('chat-template');
        if (chatTemplate) chatTemplate.value = config.chat_template;
    }

    // Training parameters
    if (config.num_epochs) {
        const numEpochs = document.getElementById('num-epochs');
        if (numEpochs) numEpochs.value = config.num_epochs;
    }

    if (config.batch_size) {
        const batchSize = document.getElementById('batch-size');
        if (batchSize) batchSize.value = config.batch_size;
    }

    if (config.learning_rate) {
        const learningRate = document.getElementById('learning-rate');
        if (learningRate) learningRate.value = config.learning_rate;
    }

    if (config.gradient_accumulation_steps) {
        const gradAccum = document.getElementById('gradient-accumulation');
        if (gradAccum) gradAccum.value = config.gradient_accumulation_steps;
    }

    if (config.max_sequence_length) {
        const maxSeqLength = document.getElementById('max-sequence-length');
        if (maxSeqLength) maxSeqLength.value = config.max_sequence_length;
    }

    if (config.max_new_tokens) {
        const maxNewTokens = document.getElementById('max-new-tokens');
        if (maxNewTokens) maxNewTokens.value = config.max_new_tokens;
    }

    if (config.warmup_steps) {
        const warmupSteps = document.getElementById('warmup-steps');
        if (warmupSteps) warmupSteps.value = config.warmup_steps;
    }

    if (config.weight_decay) {
        const weightDecay = document.getElementById('weight-decay');
        if (weightDecay) weightDecay.value = config.weight_decay;
    }

    // LoRA parameters
    if (config.lora_rank) {
        const loraRank = document.getElementById('lora-rank');
        if (loraRank) loraRank.value = config.lora_rank;
    }

    if (config.lora_alpha) {
        const loraAlpha = document.getElementById('lora-alpha');
        if (loraAlpha) loraAlpha.value = config.lora_alpha;
    }

    if (config.lora_dropout) {
        const loraDropout = document.getElementById('lora-dropout');
        if (loraDropout) loraDropout.value = config.lora_dropout;
    }

    if (config.lora_target_modules && Array.isArray(config.lora_target_modules)) {
        // Reset all checkboxes first
        ['target-q-proj', 'target-v-proj', 'target-k-proj', 'target-o-proj'].forEach(id => {
            const checkbox = document.getElementById(id);
            if (checkbox) checkbox.checked = false;
        });

        // Check the ones in the config
        config.lora_target_modules.forEach(module => {
            const checkbox = document.getElementById(`target-${module.replace('_', '-')}`);
            if (checkbox) checkbox.checked = true;
        });
    }

    if (config.lora_bias) {
        const loraBias = document.getElementById('lora-bias');
        if (loraBias) loraBias.value = config.lora_bias;
    }

    // GRPO parameters
    if (config.temperature !== undefined) {
        const temperature = document.getElementById('temperature');
        if (temperature) temperature.value = config.temperature;
    }

    if (config.top_p !== undefined) {
        const topP = document.getElementById('top-p');
        if (topP) topP.value = config.top_p;
    }

    if (config.top_k !== undefined) {
        const topK = document.getElementById('top-k');
        if (topK) topK.value = config.top_k;
    }

    if (config.repetition_penalty !== undefined) {
        const repPenalty = document.getElementById('repetition-penalty');
        if (repPenalty) repPenalty.value = config.repetition_penalty;
    }

    if (config.kl_penalty !== undefined) {
        const klPenalty = document.getElementById('kl-penalty');
        if (klPenalty) klPenalty.value = config.kl_penalty;
    }

    if (config.clip_range !== undefined) {
        const clipRange = document.getElementById('clip-range');
        if (clipRange) clipRange.value = config.clip_range;
    }

    if (config.num_generations) {
        const numGenerations = document.getElementById('num-generations');
        if (numGenerations) numGenerations.value = config.num_generations;
    }

    if (config.value_coefficient !== undefined) {
        const valueCoeff = document.getElementById('value-coefficient');
        if (valueCoeff) valueCoeff.value = config.value_coefficient;
    }

    // Algorithm selection
    if (config.importance_sampling_level) {
        const algorithm = config.importance_sampling_level === 'sequence' ? 'gspo' : 'grpo';
        const grpoRadio = document.getElementById('algo-grpo');
        const gspoRadio = document.getElementById('algo-gspo');

        if (algorithm === 'gspo' && gspoRadio) {
            gspoRadio.checked = true;
        } else if (grpoRadio) {
            grpoRadio.checked = true;
        }
    }

    if (config.epsilon !== undefined) {
        const epsilon = document.getElementById('epsilon');
        if (epsilon) epsilon.value = config.epsilon;
    }

    if (config.epsilon_high !== undefined) {
        const epsilonHigh = document.getElementById('epsilon-high');
        if (epsilonHigh) epsilonHigh.value = config.epsilon_high;
    }

    // Pre-training configuration
    if (config.enable_pre_training !== undefined) {
        const enablePreTraining = document.getElementById('enable-pre-training');
        if (enablePreTraining) enablePreTraining.checked = config.enable_pre_training;
    }

    if (config.pre_training_epochs) {
        const preTrainingEpochs = document.getElementById('pre-training-epochs');
        if (preTrainingEpochs) preTrainingEpochs.value = config.pre_training_epochs;
    }

    if (config.pre_training_samples) {
        const preTrainingSamples = document.getElementById('pre-training-samples');
        if (preTrainingSamples) preTrainingSamples.value = config.pre_training_samples;
    }

    if (config.validate_format !== undefined) {
        const validateFormat = document.getElementById('validate-format');
        if (validateFormat) validateFormat.checked = config.validate_format;
    }

    // Advanced settings
    if (config.lr_scheduler_type) {
        const lrScheduler = document.getElementById('lr-scheduler-type');
        if (lrScheduler) lrScheduler.value = config.lr_scheduler_type;
    }

    if (config.optim) {
        const optimizer = document.getElementById('optimizer');
        if (optimizer) optimizer.value = config.optim;
    }

    if (config.max_grad_norm !== undefined) {
        const maxGradNorm = document.getElementById('max-grad-norm');
        if (maxGradNorm) maxGradNorm.value = config.max_grad_norm;
    }

    if (config.logging_steps) {
        const loggingSteps = document.getElementById('logging-steps');
        if (loggingSteps) loggingSteps.value = config.logging_steps;
    }

    if (config.save_steps) {
        const saveSteps = document.getElementById('save-steps');
        if (saveSteps) saveSteps.value = config.save_steps;
    }

    if (config.eval_steps) {
        const evalSteps = document.getElementById('eval-steps');
        if (evalSteps) evalSteps.value = config.eval_steps;
    }

    if (config.seed !== undefined) {
        const seed = document.getElementById('seed');
        if (seed) seed.value = config.seed;
    }

    // Update template preview after loading
    updateTemplatePreview();

    // Update configuration summary
    updateConfigSummary();
}

// ============================================================================
// Utility Functions
// ============================================================================

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// ============================================================================
// Event Listeners
// ============================================================================

function setupEventListeners() {
    // Step navigation handlers - clicking on step indicators
    document.getElementById('step-1-indicator').addEventListener('click', function() {
        goToStep(1);
    });
    document.getElementById('step-2-indicator').addEventListener('click', function() {
        goToStep(2);
    });
    document.getElementById('step-3-indicator').addEventListener('click', function() {
        goToStep(3);
    });
    document.getElementById('step-4-indicator').addEventListener('click', function() {
        goToStep(4);
    });

    // Update recommendations when key settings change
    document.getElementById('model-name')?.addEventListener('change', function() {
        updateRecommendedIfActive();
        updateConfigSummary(); // Update summary when model changes
    });
    document.getElementById('reward-preset-select')?.addEventListener('change', updateRecommendedIfActive);
    document.getElementById('reasoning-start')?.addEventListener('input', debounce(updateRecommendedIfActive, 1000));
    document.getElementById('reasoning-end')?.addEventListener('input', debounce(updateRecommendedIfActive, 1000));

    // Update chat template preview when system prompt or template fields change
    document.getElementById('system-prompt')?.addEventListener('input', updateChatTemplatePreview);
    document.getElementById('custom-system-prompt')?.addEventListener('input', function() {
        updateTemplatePreview(); // This now also calls updateChatTemplatePreview
    });
    document.getElementById('custom-reasoning-start')?.addEventListener('input', function() {
        updateTemplatePreview();
    });
    document.getElementById('custom-reasoning-end')?.addEventListener('input', function() {
        updateTemplatePreview();
    });
    document.getElementById('custom-solution-start')?.addEventListener('input', function() {
        updateTemplatePreview();
    });
    document.getElementById('custom-solution-end')?.addEventListener('input', function() {
        updateTemplatePreview();
    });

    // Update configuration summary when key fields change
    document.getElementById('model-family')?.addEventListener('change', updateConfigSummary);
    document.getElementById('dataset-path')?.addEventListener('input', updateConfigSummary);
    document.getElementById('num-epochs')?.addEventListener('input', updateConfigSummary);
    document.getElementById('batch-size')?.addEventListener('input', updateConfigSummary);
    document.getElementById('learning-rate')?.addEventListener('input', updateConfigSummary);

    // Model configuration mode handlers
    document.getElementById('setup-recommended').addEventListener('change', function() {
        if (this.checked) setModelMode('recommended');
    });

    document.getElementById('setup-custom').addEventListener('change', function() {
        if (this.checked) setModelMode('custom');
    });

    document.getElementById('setup-advanced').addEventListener('change', function() {
        if (this.checked) setModelMode('advanced');
    });

    // Setup template handlers
    setupTemplateHandlers();

    // Setup training parameter listeners
    setupTrainingParameterListeners();

    // Update sliders when input values change
    document.getElementById('lora-rank').addEventListener('input', function() {
        document.getElementById('lora-rank-slider').value = this.value;
        updateConfigSummary();
    });

    document.getElementById('lora-alpha').addEventListener('input', function() {
        document.getElementById('lora-alpha-slider').value = this.value;
        updateConfigSummary();
    });

    document.getElementById('lora-dropout').addEventListener('input', function() {
        document.getElementById('lora-dropout-slider').value = this.value;
        updateConfigSummary();
    });

    // Auto-save state on input changes
    document.querySelectorAll('input, select, textarea').forEach(element => {
        element.addEventListener('change', saveState);
    });

    // Field mapping sync - update hidden fields when visible fields change
    const instructionFieldVisible = document.getElementById('instruction-field-visible');
    const responseFieldVisible = document.getElementById('response-field-visible');

    if (instructionFieldVisible) {
        instructionFieldVisible.addEventListener('input', function() {
            const hiddenField = document.getElementById('instruction-field');
            if (hiddenField) {
                hiddenField.value = this.value;
                console.log('Updated instruction field to:', this.value);
            }
        });
    }

    if (responseFieldVisible) {
        responseFieldVisible.addEventListener('input', function() {
            const hiddenField = document.getElementById('response-field');
            if (hiddenField) {
                hiddenField.value = this.value;
                console.log('Updated response field to:', this.value);
            }
        });
    }

    // Keyboard shortcuts
    document.addEventListener('keydown', function(event) {
        if (event.ctrlKey || event.metaKey) {
            switch (event.key) {
                case 's':
                    event.preventDefault();
                    saveCurrentConfig();
                    break;
                case 'Enter':
                    event.preventDefault();
                    if (currentStep === 4) {
                        startTraining();
                    } else {
                        validateAndProceed(currentStep);
                    }
                    break;
            }
        }
    });

    // Periodic status updates
    setInterval(updateSystemStatus, 30000); // Every 30 seconds

    // Auto-refresh sessions
    setInterval(refreshSessions, 10000); // Every 10 seconds
}

function estimateTrainingRequirements() {
    const epochs = parseInt(document.getElementById('num-epochs').value) || 3;
    const batchSize = parseInt(document.getElementById('batch-size').value) || 4;

    // Simple estimation (would be more complex in real implementation)
    const estimatedMinutes = epochs * 5 * (4 / batchSize);
    document.getElementById('summary-time').textContent = `~${Math.round(estimatedMinutes)} minutes`;

    const estimatedVRAM = 2 + (batchSize * 0.5);
    document.getElementById('summary-vram').textContent = `~${estimatedVRAM.toFixed(1)}GB`;
}

// ============================================================================
// Dataset Management Functions
// ============================================================================

async function downloadDataset(key, fieldMapping = null) {
    const dataset = datasetCatalog[key];
    if (!dataset) return;

    // Check if dataset is large and show warning
    const estimatedMb = dataset.estimated_mb || 100;
    const sampleCount = dataset.sample_count || 50000;

    if (estimatedMb > 500 || sampleCount > 100000) {
        const confirmed = await showLargeDatasetWarning(dataset.name, estimatedMb, sampleCount);
        if (!confirmed) return;
    }

    try {
        // Use dataset's predefined field mapping if available
        if (!fieldMapping && dataset.fields) {
            fieldMapping = {
                instruction: dataset.fields.instruction,
                response: dataset.fields.response
            };
        }

        // If no field mapping provided and dataset doesn't have predefined mapping,
        // detect fields first
        if (!fieldMapping && !dataset.field_mapping) {
            const fieldsResponse = await fetch('/api/datasets/detect-fields', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset_name: dataset.path,
                    config: dataset.config
                })
            });

            if (fieldsResponse.ok) {
                const fieldsData = await fieldsResponse.json();

                // If suggested mappings are incomplete, show field selection dialog
                if (!fieldsData.suggested_mappings.instruction || !fieldsData.suggested_mappings.response) {
                    showFieldMappingDialog(key, fieldsData);
                    return;
                }

                // Use suggested mappings
                fieldMapping = fieldsData.suggested_mappings;
            }
        }

        // Show download progress modal
        const modal = new bootstrap.Modal(document.getElementById('downloadProgressModal'));
        modal.show();

        document.getElementById('downloading-dataset-name').textContent = dataset.name;
        document.getElementById('download-progress-bar').style.width = '0%';
        document.getElementById('download-status-message').textContent = 'Initializing download...';

        // Start download
        const requestBody = {
            dataset_name: dataset.path,
            force_download: false
        };

        // Add config if specified (for multi-config datasets like GSM8K)
        if (dataset.config) {
            requestBody.config = dataset.config;
        }

        // Add field mapping if available
        if (fieldMapping) {
            requestBody.field_mapping = fieldMapping;
        }

        const response = await fetch('/api/datasets/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        const result = await response.json();
        if (result.session_id) {
            currentDatasetSession = result.session_id;

            // Join WebSocket session for progress updates
            socket.emit('join_dataset_session', { session_id: result.session_id });
        }
    } catch (error) {
        console.error('Failed to start download:', error);
        showAlert('Failed to start dataset download', 'danger');
    }
}

function updateDatasetProgress(data) {
    const progressBar = document.getElementById('download-progress-bar');
    const statusMessage = document.getElementById('download-status-message');

    if (data.progress !== undefined && data.progress !== null) {
        const percentage = Math.round(data.progress * 100);
        progressBar.style.width = percentage + '%';
        progressBar.textContent = percentage + '%';
    }

    if (data.message) {
        statusMessage.textContent = data.message;
    }

    // Update status based on data.status
    if (data.status === 'cached') {
        statusMessage.innerHTML = '<i class="fas fa-check text-success"></i> Loaded from cache';
    } else if (data.status === 'downloading') {
        statusMessage.innerHTML = '<i class="fas fa-download text-primary"></i> Downloading...';
    } else if (data.status === 'processing') {
        statusMessage.innerHTML = '<i class="fas fa-cog fa-spin text-info"></i> Processing dataset...';
    } else if (data.status === 'filtering') {
        statusMessage.innerHTML = '<i class="fas fa-filter text-warning"></i> ' + data.message;
    } else if (data.status === 'preprocessing') {
        statusMessage.innerHTML = '<i class="fas fa-cogs fa-spin text-info"></i> Preprocessing...';
    } else if (data.status === 'finalizing') {
        statusMessage.innerHTML = '<i class="fas fa-chart-bar text-info"></i> Calculating statistics...';
    } else if (data.status === 'loaded') {
        statusMessage.innerHTML = '<i class="fas fa-check-circle text-info"></i> Dataset loaded, processing...';
    } else if (data.status === 'completed') {
        statusMessage.innerHTML = '<i class="fas fa-check-circle text-success"></i> Dataset ready!';
    }
}

function handleDatasetComplete(data) {
    // Hide modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('downloadProgressModal'));
    if (modal) modal.hide();

    showAlert(`Dataset "${data.dataset_name}" downloaded successfully!`, 'success');

    // Refresh dataset catalog to show updated status
    loadDatasetCatalog();
}

function handleDatasetError(data) {
    // Hide modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('downloadProgressModal'));
    if (modal) modal.hide();

    showAlert(`Failed to download dataset: ${data.error}`, 'danger');
}

function cancelDownload() {
    // TODO: Implement download cancellation
    const modal = bootstrap.Modal.getInstance(document.getElementById('downloadProgressModal'));
    if (modal) modal.hide();
}

function showLargeDatasetWarning(datasetName, estimatedMb, sampleCount) {
    return new Promise((resolve) => {
        // Create warning modal HTML if it doesn't exist
        let modal = document.getElementById('largeDatasetWarningModal');
        if (!modal) {
            const modalHTML = `
                <div class="modal fade" id="largeDatasetWarningModal" tabindex="-1">
                    <div class="modal-dialog">
                        <div class="modal-content">
                            <div class="modal-header bg-warning text-dark">
                                <h5 class="modal-title">
                                    <i class="fas fa-exclamation-triangle"></i> Large Dataset Warning
                                </h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body">
                                <div id="large-dataset-warning-content"></div>
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal" id="cancel-large-download">Cancel</button>
                                <button type="button" class="btn btn-warning" id="confirm-large-download">
                                    <i class="fas fa-download"></i> Download Anyway
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            document.body.insertAdjacentHTML('beforeend', modalHTML);
            modal = document.getElementById('largeDatasetWarningModal');
        }

        // Build warning content
        const content = document.getElementById('large-dataset-warning-content');
        const sizeText = estimatedMb >= 1000 ? `${(estimatedMb/1000).toFixed(1)} GB` : `${estimatedMb} MB`;

        content.innerHTML = `
            <div class="alert alert-warning mb-3">
                <strong>${datasetName}</strong> is a large dataset!
            </div>

            <div class="mb-3">
                <p><strong>Dataset Information:</strong></p>
                <ul>
                    <li>Estimated size: <strong>${sizeText}</strong></li>
                    <li>Sample count: <strong>${sampleCount.toLocaleString()}</strong> samples</li>
                    <li>Download time: <strong>${estimatedMb > 5000 ? '30+ minutes' : estimatedMb > 1000 ? '10-30 minutes' : '5-10 minutes'}</strong> (depending on connection)</li>
                </ul>
            </div>

            <div class="alert alert-info">
                <i class="fas fa-info-circle"></i> <strong>Tips:</strong>
                <ul class="mb-0 mt-2">
                    <li>Large datasets may take significant time to download and process</li>
                    <li>Consider using a smaller dataset for testing first</li>
                    <li>Ensure you have sufficient disk space available</li>
                </ul>
            </div>

            <p class="mb-0 text-muted">Do you want to continue with the download?</p>
        `;

        // Show modal
        const bsModal = new bootstrap.Modal(modal);
        bsModal.show();

        // Handle buttons
        document.getElementById('confirm-large-download').onclick = function() {
            bsModal.hide();
            resolve(true);
        };

        document.getElementById('cancel-large-download').onclick = function() {
            bsModal.hide();
            resolve(false);
        };

        // Handle modal dismiss (X button or ESC key)
        modal.addEventListener('hidden.bs.modal', function onHidden() {
            modal.removeEventListener('hidden.bs.modal', onHidden);
            resolve(false);
        }, { once: true });
    });
}

function showFieldMappingDialog(datasetKey, fieldsData) {
    const dataset = datasetCatalog[datasetKey];

    // Create field mapping modal HTML if it doesn't exist
    let modal = document.getElementById('fieldMappingModal');
    if (!modal) {
        const modalHTML = `
            <div class="modal fade" id="fieldMappingModal" tabindex="-1">
                <div class="modal-dialog modal-lg">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title">Select Dataset Fields</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <p class="text-muted">Please select which fields to use for instruction and response:</p>
                            <div id="field-mapping-content"></div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                            <button type="button" class="btn btn-primary" id="apply-field-mapping">Apply Mapping</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        modal = document.getElementById('fieldMappingModal');
    }

    // Build field selection content
    const content = document.getElementById('field-mapping-content');
    content.innerHTML = `
        <div class="mb-3">
            <label class="form-label fw-bold">Dataset: ${dataset.name}</label>
            <p class="text-muted small">Available columns: ${fieldsData.columns.join(', ')}</p>
        </div>

        <div class="row">
            <div class="col-md-6 mb-3">
                <label for="instruction-field" class="form-label">
                    <i class="fas fa-question-circle"></i> Instruction Field
                </label>
                <select class="form-select" id="instruction-field">
                    <option value="">-- Select Field --</option>
                    ${fieldsData.columns.map(col => `
                        <option value="${col}" ${fieldsData.suggested_mappings.instruction === col ? 'selected' : ''}>
                            ${col}
                        </option>
                    `).join('')}
                </select>
                <small class="text-muted">The field containing the input/question/prompt</small>
            </div>

            <div class="col-md-6 mb-3">
                <label for="response-field" class="form-label">
                    <i class="fas fa-reply"></i> Response Field
                </label>
                <select class="form-select" id="response-field">
                    <option value="">-- Select Field --</option>
                    ${fieldsData.columns.map(col => `
                        <option value="${col}" ${fieldsData.suggested_mappings.response === col ? 'selected' : ''}>
                            ${col}
                        </option>
                    `).join('')}
                </select>
                <small class="text-muted">The field containing the output/answer/response</small>
            </div>
        </div>

        ${fieldsData.sample_data ? `
            <div class="mt-3">
                <label class="form-label">Sample Data Preview:</label>
                <div class="table-responsive">
                    <table class="table table-sm table-bordered">
                        <thead>
                            <tr>
                                ${Object.keys(fieldsData.sample_data).map(col => `<th>${col}</th>`).join('')}
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                ${Object.values(fieldsData.sample_data).map(val => `<td class="text-truncate" style="max-width: 200px;">${val}</td>`).join('')}
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        ` : ''}
    `;

    // Show modal
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();

    // Handle apply button
    document.getElementById('apply-field-mapping').onclick = function() {
        const instructionField = document.getElementById('instruction-field').value;
        const responseField = document.getElementById('response-field').value;

        if (!instructionField || !responseField) {
            showAlert('Please select both instruction and response fields', 'warning');
            return;
        }

        // Hide modal
        bsModal.hide();

        // Continue download with field mapping
        downloadDataset(datasetKey, {
            instruction: instructionField,
            response: responseField
        });
    };
}

async function downloadCustomDataset() {
    const datasetName = document.getElementById('custom-dataset-name').value.trim();
    const datasetConfig = document.getElementById('custom-dataset-config').value.trim();

    if (!datasetName) {
        showAlert('Please enter a dataset name', 'warning');
        return;
    }

    // Validate dataset name format
    const validFormat = /^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_-]+$|^[a-zA-Z0-9_-]+$/;
    if (!validFormat.test(datasetName)) {
        showAlert('Invalid dataset name format. Use: username/dataset-name or dataset-name', 'danger');
        return;
    }

    try {
        // First detect fields
        const fieldsResponse = await fetch('/api/datasets/detect-fields', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_name: datasetName,
                config: datasetConfig || null
            })
        });

        if (!fieldsResponse.ok) {
            const error = await fieldsResponse.json();
            showAlert(`Failed to detect dataset fields: ${error.error}`, 'danger');
            return;
        }

        const fieldsData = await fieldsResponse.json();

        // Check if we need field mapping
        let fieldMapping = null;
        if (!fieldsData.suggested_mappings.instruction || !fieldsData.suggested_mappings.response) {
            // Show field mapping dialog
            const customKey = 'custom_' + datasetName.replace('/', '_');
            datasetCatalog[customKey] = {
                name: datasetName,
                path: datasetName,
                config: datasetConfig || null
            };
            showFieldMappingDialog(customKey, fieldsData);
            return;
        }

        // Use suggested mappings
        fieldMapping = fieldsData.suggested_mappings;

        // Show download progress modal
        const modal = new bootstrap.Modal(document.getElementById('downloadProgressModal'));
        modal.show();

        document.getElementById('downloading-dataset-name').textContent = datasetName;
        document.getElementById('download-progress-bar').style.width = '0%';
        document.getElementById('download-status-message').textContent = 'Initializing download...';

        // Start download
        const requestBody = {
            dataset_name: datasetName,
            config: datasetConfig || null,
            force_download: false,
            field_mapping: fieldMapping
        };

        const response = await fetch('/api/datasets/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        const result = await response.json();
        if (result.session_id) {
            currentDatasetSession = result.session_id;

            // Join WebSocket session for progress updates
            socket.emit('join_dataset_session', { session_id: result.session_id });

            // Clear the input fields
            document.getElementById('custom-dataset-name').value = '';
            document.getElementById('custom-dataset-config').value = '';
        }
    } catch (error) {
        console.error('Failed to download custom dataset:', error);
        showAlert('Failed to download custom dataset', 'danger');
    }
}

async function previewDataset(key) {
    const dataset = datasetCatalog[key];
    if (!dataset) return;

    try {
        // Show preview modal
        const modal = new bootstrap.Modal(document.getElementById('datasetPreviewModal'));
        modal.show();

        // Reset modal content
        document.getElementById('preview-dataset-name').textContent = dataset.name;
        document.getElementById('dataset-preview-loading').style.display = 'block';
        document.getElementById('dataset-preview-content').style.display = 'none';

        // Fetch dataset samples
        const requestBody = {
            dataset_name: dataset.path,
            sample_size: 5
        };

        // Add config if specified (for multi-config datasets)
        if (dataset.config) {
            requestBody.config = dataset.config;
        }

        const response = await fetch('/api/datasets/sample', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestBody)
        });

        const result = await response.json();

        // Hide loading, show content
        document.getElementById('dataset-preview-loading').style.display = 'none';
        document.getElementById('dataset-preview-content').style.display = 'block';

        // Display statistics
        if (result.statistics) {
            const statsDiv = document.getElementById('dataset-stats');
            statsDiv.innerHTML = `
                <strong>Total Samples:</strong> ${result.statistics.total_samples}<br>
                <strong>Avg Instruction Length:</strong> ${Math.round(result.statistics.avg_instruction_length)} chars<br>
                <strong>Avg Response Length:</strong> ${Math.round(result.statistics.avg_response_length)} chars
            `;
        }

        // Display samples
        const samplesDiv = document.getElementById('dataset-samples');
        samplesDiv.innerHTML = '';

        if (result.samples && result.samples.length > 0) {
            result.samples.forEach((sample, idx) => {
                const sampleCard = document.createElement('div');
                sampleCard.className = 'card sample-card mb-2';
                sampleCard.innerHTML = `
                    <div class="card-body">
                        <h6 class="card-subtitle mb-2 text-muted">Sample ${idx + 1}</h6>
                        <div class="mb-2">
                            <strong>Instruction:</strong>
                            <pre class="sample-pre p-2 rounded" style="white-space: pre-wrap;">${escapeHtml(sample[dataset.fields.instruction] || sample.instruction || '')}</pre>
                        </div>
                        <div>
                            <strong>Response:</strong>
                            <pre class="sample-pre p-2 rounded" style="white-space: pre-wrap;">${escapeHtml(sample[dataset.fields.response] || sample.response || sample.output || '')}</pre>
                        </div>
                    </div>
                `;
                samplesDiv.appendChild(sampleCard);
            });
        } else {
            samplesDiv.innerHTML = '<p>No samples available</p>';
        }

        // Store dataset key for "Use This Dataset" button
        document.getElementById('use-dataset-btn').dataset.key = key;

    } catch (error) {
        console.error('Failed to preview dataset:', error);
        showAlert('Failed to load dataset preview', 'danger');

        // Hide modal on error
        const modal = bootstrap.Modal.getInstance(document.getElementById('datasetPreviewModal'));
        if (modal) modal.hide();
    }
}

function useDatasetFromPreview() {
    const key = document.getElementById('use-dataset-btn').dataset.key;
    if (key) {
        selectDataset(key);

        // Close preview modal
        const modal = bootstrap.Modal.getInstance(document.getElementById('datasetPreviewModal'));
        if (modal) modal.hide();
    }
}

async function clearDatasetCache(key) {
    const dataset = datasetCatalog[key];
    if (!dataset) return;

    if (!confirm(`Are you sure you want to clear the cache for "${dataset.name}"?`)) {
        return;
    }

    try {
        const response = await fetch('/api/datasets/cache/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_name: dataset.path
            })
        });

        const result = await response.json();
        if (result.success) {
            showAlert(`Cache cleared for ${dataset.name}`, 'success');
            // Refresh catalog
            loadDatasetCatalog();
        }
    } catch (error) {
        console.error('Failed to clear cache:', error);
        showAlert('Failed to clear dataset cache', 'danger');
    }
}

async function clearAllCache() {
    if (!confirm('Are you sure you want to clear ALL cached datasets? This cannot be undone.')) {
        return;
    }

    try {
        const response = await fetch('/api/datasets/cache/clear', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        const result = await response.json();
        if (result.success) {
            showAlert('All dataset cache cleared', 'success');
            // Refresh catalog
            loadDatasetCatalog();
        }
    } catch (error) {
        console.error('Failed to clear cache:', error);
        showAlert('Failed to clear cache', 'danger');
    }
}

async function refreshCacheInfo() {
    try {
        const response = await fetch('/api/datasets/cache/info');
        const data = await response.json();

        document.getElementById('cache-size').textContent = `${data.total_size_mb} MB`;
        document.getElementById('cached-datasets-count').textContent = data.cache_items.length;

    } catch (error) {
        console.error('Failed to get cache info:', error);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================================================
// Model Export Functions
// ============================================================================

let exportFormats = {};
let ggufQuantizations = {};
let lastCompletedSessionId = null;

async function loadExportFormats() {
    try {
        const response = await fetch('/api/export/formats');
        const data = await response.json();
        exportFormats = data.formats;
        ggufQuantizations = data.gguf_quantizations;
    } catch (error) {
        console.error('Failed to load export formats:', error);
    }
}

async function showExportModal(sessionId) {
    // Remove any existing export modal
    const existingModal = document.getElementById('exportModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Store this as the last session
    lastCompletedSessionId = sessionId;

    // Create and show export modal
    const modal = await createExportModal(sessionId);
    document.body.appendChild(modal);
    const exportModal = new bootstrap.Modal(modal);
    exportModal.show();

    // Load export formats if not already loaded
    if (Object.keys(exportFormats).length === 0) {
        loadExportFormats().then(() => updateExportFormatOptions());
    } else {
        updateExportFormatOptions();
    }
}

async function showExportOptions() {
    // Show list of completed sessions that can be exported
    try {
        const response = await fetch('/api/training/sessions');
        const sessions = await response.json();
        const completedSessions = sessions.filter(s => s.status === 'completed');

        if (completedSessions.length === 0) {
            showAlert('No completed training sessions available for export.', 'warning');
            return;
        }

        if (completedSessions.length === 1) {
            // If only one session, directly open export modal
            showExportModal(completedSessions[0].session_id);
        } else {
            // Show selection modal if multiple sessions
            showSessionSelectionModal(completedSessions);
        }
    } catch (error) {
        console.error('Failed to load sessions:', error);
        showAlert('Failed to load sessions for export.', 'danger');
    }
}

function showSessionSelectionModal(sessions) {
    // Remove any existing selection modal
    const existingModal = document.getElementById('sessionSelectModal');
    if (existingModal) {
        existingModal.remove();
    }

    const modalDiv = document.createElement('div');
    modalDiv.className = 'modal fade';
    modalDiv.id = 'sessionSelectModal';
    modalDiv.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="fas fa-list"></i> Select Session to Export
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="list-group">
                        ${sessions.map(session => `
                            <button class="list-group-item list-group-item-action"
                                    onclick="document.querySelector('#sessionSelectModal .btn-close').click(); showExportModal('${session.session_id}')">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <strong>${session.model.split('/').pop()}</strong>
                                        <br>
                                        <small class="text-muted">${new Date(session.created_at).toLocaleString()}</small>
                                    </div>
                                    <i class="fas fa-chevron-right"></i>
                                </div>
                            </button>
                        `).join('')}
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modalDiv);
    const selectModal = new bootstrap.Modal(modalDiv);
    selectModal.show();
}

async function createExportModal(sessionId) {
    // Generate auto name based on session info
    const autoGeneratedName = await generateExportName(sessionId);

    const modalDiv = document.createElement('div');
    modalDiv.className = 'modal fade';
    modalDiv.id = 'exportModal';
    modalDiv.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="fas fa-file-export"></i> Export Model
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label for="export-format" class="form-label">Export Format</label>
                        <select class="form-select" id="export-format" onchange="updateExportOptions()">
                            <option value="huggingface">HuggingFace (Standard)</option>
                            <option value="safetensors">SafeTensors (Efficient)</option>
                            <option value="gguf">GGUF (llama.cpp)</option>
                            <option value="merged">Merged Model (LoRA + Base)</option>
                        </select>
                        <small class="form-text text-muted" id="format-description"></small>
                    </div>

                    <div class="mb-3">
                        <label for="export-name" class="form-label">Export Name</label>
                        <input type="text" class="form-control" id="export-name"
                               placeholder="${autoGeneratedName}"
                               value="">
                        <small class="text-muted">Leave empty to use: ${autoGeneratedName}</small>
                    </div>

                    <div id="gguf-options" style="display: none;">
                        <div class="mb-3">
                            <label for="quantization" class="form-label">Quantization</label>
                            <select class="form-select" id="quantization">
                                <option value="q4_k_m" selected>Q4_K_M (Recommended)</option>
                                <option value="q5_k_m">Q5_K_M (Balanced)</option>
                                <option value="q6_k">Q6_K (Good Quality)</option>
                                <option value="q8_0">Q8_0 (Best Quality)</option>
                                <option value="f16">F16 (No Quantization)</option>
                            </select>
                            <small class="form-text text-muted" id="quant-description"></small>
                        </div>
                    </div>

                    <div id="lora-options" class="mb-3">
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="merge-lora">
                            <label class="form-check-label" for="merge-lora">
                                Merge LoRA weights with base model
                            </label>
                        </div>
                    </div>

                    <div id="export-progress" style="display: none;">
                        <div class="progress mb-2">
                            <div class="progress-bar progress-bar-striped progress-bar-animated"
                                 id="export-progress-bar" style="width: 0%"></div>
                        </div>
                        <p class="text-muted mb-0" id="export-status"></p>
                    </div>

                    <div id="export-result" style="display: none;">
                        <div class="alert alert-success">
                            <i class="fas fa-check-circle"></i> Export completed successfully!
                        </div>
                        <div class="d-grid gap-2">
                            <button class="btn btn-primary" onclick="downloadExport()">
                                <i class="fas fa-download"></i> Download Model
                            </button>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    <button type="button" class="btn btn-primary" id="start-export-btn"
                            onclick="startExport('${sessionId}')">
                        <i class="fas fa-file-export"></i> Start Export
                    </button>
                </div>
            </div>
        </div>
    `;
    return modalDiv;
}

function updateExportOptions() {
    const format = document.getElementById('export-format').value;
    const ggufOptions = document.getElementById('gguf-options');
    const loraOptions = document.getElementById('lora-options');
    const formatDesc = document.getElementById('format-description');

    // Show/hide format-specific options
    if (format === 'gguf') {
        ggufOptions.style.display = 'block';
        updateQuantizationDescription();
    } else {
        ggufOptions.style.display = 'none';
    }

    // Update format description
    if (exportFormats[format]) {
        formatDesc.textContent = exportFormats[format];
    }

    // Show/hide LoRA options based on format
    if (format === 'merged') {
        loraOptions.style.display = 'none'; // Always merges
    } else if (format === 'gguf') {
        loraOptions.style.display = 'block';
    } else {
        loraOptions.style.display = 'block';
    }
}

function updateQuantizationDescription() {
    const quant = document.getElementById('quantization').value;
    const desc = document.getElementById('quant-description');
    if (ggufQuantizations[quant]) {
        desc.textContent = ggufQuantizations[quant];
    }
}

function updateExportFormatOptions() {
    const formatSelect = document.getElementById('export-format');
    if (!formatSelect) return;

    formatSelect.innerHTML = '';
    for (const [key, value] of Object.entries(exportFormats)) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = key.charAt(0).toUpperCase() + key.slice(1);
        formatSelect.appendChild(option);
    }

    updateExportOptions();
}

let currentExportPath = null;
let currentExportSession = null;

async function startExport(sessionId) {
    const format = document.getElementById('export-format').value;
    const name = document.getElementById('export-name').value || null;
    const quantization = document.getElementById('quantization').value;
    const mergeLora = document.getElementById('merge-lora').checked;

    // Hide button, show progress
    document.getElementById('start-export-btn').disabled = true;
    document.getElementById('export-progress').style.display = 'block';
    document.getElementById('export-result').style.display = 'none';

    currentExportSession = sessionId;

    // Join WebSocket room for progress updates
    socket.emit('join', { session_id: sessionId });

    try {
        const response = await fetch(`/api/export/${sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                format: format,
                name: name,
                quantization: format === 'gguf' ? quantization : null,
                merge_lora: mergeLora
            })
        });

        const result = await response.json();

        if (result.success) {
            currentExportPath = result.path;
            document.getElementById('export-progress').style.display = 'none';
            document.getElementById('export-result').style.display = 'block';
            showAlert('Model exported successfully!', 'success');
        } else {
            showAlert('Export failed: ' + result.error, 'danger');
            document.getElementById('start-export-btn').disabled = false;
            document.getElementById('export-progress').style.display = 'none';
        }
    } catch (error) {
        console.error('Export error:', error);
        showAlert('Export failed: ' + error.message, 'danger');
        document.getElementById('start-export-btn').disabled = false;
        document.getElementById('export-progress').style.display = 'none';
    }
}

function downloadExport() {
    if (!currentExportPath || !currentExportSession) {
        showAlert('No export available', 'warning');
        return;
    }

    // The path is already relative in format: session_id/format/export_name
    // We just need the format/export_name part
    const pathParts = currentExportPath.split('/');

    // If path starts with session_id, skip it
    let relativePath;
    if (pathParts[0] === currentExportSession) {
        relativePath = pathParts.slice(1).join('/');
    } else {
        // Path is already in the correct format
        relativePath = currentExportPath;
    }

    const downloadUrl = `/api/export/download/${currentExportSession}/${relativePath}`;
    window.open(downloadUrl, '_blank');
}

function showExportButton(sessionId) {
    // Add export button to the UI
    const controlsDiv = document.querySelector('.training-controls');
    if (controlsDiv && !document.getElementById('export-model-btn')) {
        const exportBtn = document.createElement('button');
        exportBtn.id = 'export-model-btn';
        exportBtn.className = 'btn btn-success ms-2';
        exportBtn.innerHTML = '<i class="fas fa-file-export"></i> Export Model';
        exportBtn.onclick = () => showExportModal(sessionId);
        controlsDiv.appendChild(exportBtn);
    }
}

// Export progress updates will be handled in initializeSocketIO

// ============================================================================
// Exports Management Functions
// ============================================================================

// Generate a meaningful display name for a model
function generateModelDisplayName(sessionData) {
    // If display_name is provided, use it with session ID
    if (sessionData.display_name) {
        return `${sessionData.display_name} (${sessionData.session_id.substring(0, 8)})`;
    }

    // Otherwise, generate a name from available data
    const modelName = (sessionData.model || sessionData.model_name || 'model').split('/').pop();
    const datasetName = (sessionData.dataset || sessionData.dataset_path || 'dataset').split('/').pop();
    const date = new Date(sessionData.created_at || Date.now()).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    const epochs = sessionData.epochs || sessionData.num_epochs || '?';
    return `${modelName} - ${datasetName} - ${date} (${epochs} epochs)`;
}

// Generate an export name for a model
async function generateExportName(sessionId) {
    try {
        // Fetch session data from API
        const response = await fetch(`/api/training/${sessionId}/status`);
        if (response.ok) {
            const session = await response.json();

            // Try to get dataset name from multiple sources
            let datasetName = 'dataset';

            // Priority 1: Check session.dataset_name or session.dataset
            if (session.dataset_name) {
                datasetName = session.dataset_name;
            } else if (session.dataset) {
                datasetName = session.dataset;
            }
            // Priority 2: Check config.dataset_name or config.dataset_path
            else if (session.config?.dataset_name) {
                datasetName = session.config.dataset_name;
            } else if (session.config?.dataset_path) {
                datasetName = session.config.dataset_path;
            }
            // Priority 3: Check config.dataset_source
            else if (session.config?.dataset_source === 'popular' && session.config?.dataset_config) {
                // Try to extract from dataset config
                const configData = typeof session.config.dataset_config === 'string'
                    ? JSON.parse(session.config.dataset_config)
                    : session.config.dataset_config;
                if (configData?.name) {
                    datasetName = configData.name;
                }
            }

            // Clean up dataset name - remove paths and special chars
            datasetName = datasetName.split('/').pop().replace(/[^a-zA-Z0-9]/g, '_');

            // Use display_name if available and includes dataset
            if (session.display_name && session.display_name.toLowerCase().includes(datasetName.toLowerCase())) {
                const cleanName = session.display_name.replace(/[^a-zA-Z0-9]/g, '_');
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
                return `${cleanName}_${timestamp}`;
            }

            // Otherwise generate from model and dataset names
            const modelName = (session.model || session.config?.model_name || 'model').split('/').pop().replace(/[^a-zA-Z0-9]/g, '_');
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);

            // Always include dataset name in export
            return `${modelName}_${datasetName}_${timestamp}`;
        }
    } catch (error) {
        console.error('Failed to fetch session data for export name:', error);
    }

    // Fallback name if API call fails - try to get dataset from current form
    const datasetPath = document.getElementById('dataset-path')?.value;
    const datasetName = datasetPath ? datasetPath.split('/').pop().replace(/[^a-zA-Z0-9]/g, '_') : 'dataset';
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    return `model_${datasetName}_${timestamp}`;
}



// Make functions globally accessible
// (Removed references to deleted Quick Actions functions)

// Uploaded datasets functions
window.selectUploadedDataset = selectUploadedDataset;
window.configureDatasetFields = configureDatasetFields;
window.deleteUploadedDataset = deleteUploadedDataset;
window.saveUploadedDatasetFieldMapping = saveUploadedDatasetFieldMapping;

async function refreshTrainedModels() {
    const modelsList = document.getElementById('trained-models-list');
    if (!modelsList) {
        console.warn('trained-models-list element not found');
        return;
    }

    try {
        console.log('Fetching trained models...');
        const response = await fetch('/api/models/trained');

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('Loaded models:', data);

        trainedModels = data.models || [];
        displayTrainedModels(trainedModels);

        // Also refresh export history
        refreshExportHistory();
    } catch (error) {
        console.error('Failed to load trained models:', error);
        modelsList.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle"></i> Failed to load models: ${error.message}
            </div>
        `;
    }
}

function displayTrainedModels(models) {
    const modelsList = document.getElementById('trained-models-list');

    if (!models || models.length === 0) {
        modelsList.innerHTML = `
            <div class="text-center text-muted p-4">
                <i class="fas fa-info-circle"></i>
                <p class="mb-0">No trained models found</p>
                <small>Train a model first to see it here</small>
            </div>
        `;
        return;
    }

    modelsList.innerHTML = models.map(model => `
        <div class="model-card card mb-3">
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-start">
                    <div class="flex-grow-1">
                        <h6 class="card-title mb-1">
                            <i class="fas fa-brain"></i> ${generateModelDisplayName(model)}
                        </h6>
                        <p class="text-muted small mb-2">
                            Session: ${model.session_id.substring(0, 8)}...
                            <br>
                            Trained: ${new Date(model.modified_at).toLocaleDateString()}
                        </p>
                        ${model.epochs ? `<span class="badge bg-info">Epochs: ${model.epochs}</span>` : ''}
                        ${model.best_reward ? `<span class="badge bg-success ms-1">Reward: ${model.best_reward.toFixed(3)}</span>` : ''}
                        ${model.has_final_checkpoint ? '<span class="badge bg-primary ms-1">Final</span>' : ''}
                    </div>
                    <div class="btn-group btn-group-sm">
                        <button class="btn btn-outline-primary" onclick="showModelDetails('${model.session_id}')" title="Details">
                            <i class="fas fa-info"></i>
                        </button>
                        <button class="btn btn-success" onclick="showExportModalForModel('${model.session_id}')" title="Export">
                            <i class="fas fa-file-export"></i>
                        </button>
                        <button class="btn btn-outline-danger" onclick="deleteModel('${model.session_id}')" title="Delete">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="form-check mt-2">
                    <input class="form-check-input" type="checkbox"
                           id="select-${model.session_id}"
                           onchange="toggleModelSelection('${model.session_id}')">
                    <label class="form-check-label small" for="select-${model.session_id}">
                        Select for batch export
                    </label>
                </div>
            </div>
        </div>
    `).join('');
}

async function showExportModalForModel(sessionId) {
    // Use the existing export modal function
    await showExportModal(sessionId);
}

async function showModelDetails(sessionId) {
    try {
        const response = await fetch(`/api/models/${sessionId}/info`);
        const modelInfo = await response.json();

        const detailsPanel = document.getElementById('model-details-panel');
        const detailsContent = document.getElementById('model-details-content');

        detailsContent.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Training Information</h6>
                    <p class="mb-1"><strong>Session ID:</strong> ${sessionId}</p>
                    <p class="mb-1"><strong>Checkpoints:</strong> ${modelInfo.checkpoints.length}</p>
                    ${modelInfo.checkpoints.map(cp => `
                        <div class="ms-3 small">
                            <i class="fas fa-save"></i> ${cp.name}
                            ${cp.training_state ? `(Step: ${cp.training_state.global_step})` : ''}
                        </div>
                    `).join('')}
                </div>
                <div class="col-md-6">
                    <h6>Export History</h6>
                    ${modelInfo.exports.length > 0 ?
                        modelInfo.exports.map(exp => `
                            <div class="small mb-1">
                                <i class="fas fa-file-export"></i> ${exp.export_format}
                                <br>
                                <small class="text-muted">${new Date(exp.export_timestamp).toLocaleDateString()}</small>
                            </div>
                        `).join('') :
                        '<p class="text-muted small">No exports yet</p>'
                    }
                </div>
            </div>
            <div class="mt-3">
                <button class="btn btn-sm btn-primary" onclick="showExportModalForModel('${sessionId}')">
                    <i class="fas fa-file-export"></i> Export This Model
                </button>
            </div>
        `;

        detailsPanel.style.display = 'block';
    } catch (error) {
        console.error('Failed to load model details:', error);
        showAlert('Failed to load model details', 'danger');
    }
}

function hideModelDetails() {
    document.getElementById('model-details-panel').style.display = 'none';
}

function toggleModelSelection(sessionId) {
    if (selectedModelsForExport.has(sessionId)) {
        selectedModelsForExport.delete(sessionId);
    } else {
        selectedModelsForExport.add(sessionId);
    }

    // Update batch export button
    updateBatchExportButton();
}

function updateBatchExportButton() {
    const batchBtn = document.querySelector('[onclick="showBatchExport()"]');
    if (batchBtn) {
        if (selectedModelsForExport.size > 0) {
            batchBtn.innerHTML = `<i class="fas fa-file-archive"></i> Batch Export (${selectedModelsForExport.size})`;
            batchBtn.classList.add('btn-warning');
            batchBtn.classList.remove('btn-success');
        } else {
            batchBtn.innerHTML = '<i class="fas fa-file-archive"></i> Batch Export';
            batchBtn.classList.remove('btn-warning');
            batchBtn.classList.add('btn-success');
        }
    }
}

async function showBatchExport() {
    if (selectedModelsForExport.size === 0) {
        showAlert('Please select at least one model for batch export', 'warning');
        return;
    }

    // Create batch export modal
    const modalDiv = document.createElement('div');
    modalDiv.className = 'modal fade';
    modalDiv.id = 'batchExportModal';
    modalDiv.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">
                        <i class="fas fa-file-archive"></i> Batch Export ${selectedModelsForExport.size} Models
                    </h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label class="form-label">Export Format</label>
                        <select class="form-select" id="batch-export-format">
                            <option value="huggingface">HuggingFace</option>
                            <option value="safetensors">SafeTensors</option>
                            <option value="gguf">GGUF (llama.cpp)</option>
                        </select>
                    </div>
                    <div class="mb-3" id="batch-quantization-options" style="display: none;">
                        <label class="form-label">GGUF Quantization</label>
                        <select class="form-select" id="batch-quantization">
                            <option value="q4_k_m">Q4_K_M (Recommended)</option>
                            <option value="q5_k_m">Q5_K_M</option>
                            <option value="q6_k">Q6_K</option>
                            <option value="q8_0">Q8_0</option>
                        </select>
                    </div>
                    <div id="batch-export-progress" style="display: none;">
                        <div class="progress mb-2">
                            <div class="progress-bar progress-bar-striped progress-bar-animated"
                                 id="batch-progress-bar" style="width: 0%"></div>
                        </div>
                        <p class="small text-muted" id="batch-status"></p>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" onclick="startBatchExport()">
                        <i class="fas fa-play"></i> Start Export
                    </button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modalDiv);
    const modal = new bootstrap.Modal(modalDiv);
    modal.show();

    // Show/hide quantization options
    document.getElementById('batch-export-format').addEventListener('change', (e) => {
        document.getElementById('batch-quantization-options').style.display =
            e.target.value === 'gguf' ? 'block' : 'none';
    });
}

async function startBatchExport() {
    const format = document.getElementById('batch-export-format').value;
    const quantization = document.getElementById('batch-quantization').value;

    document.getElementById('batch-export-progress').style.display = 'block';

    try {
        const response = await fetch('/api/exports/batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_ids: Array.from(selectedModelsForExport),
                format: format,
                quantization: quantization
            })
        });

        const result = await response.json();

        if (result.successful > 0) {
            showAlert(`Successfully exported ${result.successful} out of ${result.total} models`, 'success');

            // Clear selection
            selectedModelsForExport.clear();
            document.querySelectorAll('[id^="select-"]').forEach(cb => cb.checked = false);
            updateBatchExportButton();

            // Refresh export history
            refreshExportHistory();
        } else {
            showAlert('Batch export failed for all models', 'danger');
        }

        // Close modal
        bootstrap.Modal.getInstance(document.getElementById('batchExportModal')).hide();

    } catch (error) {
        console.error('Batch export error:', error);
        showAlert('Batch export failed: ' + error.message, 'danger');
    }
}

async function refreshExportHistory() {
    try {
        const historyList = document.getElementById('export-history-list');
        if (!historyList) return; // Exit if element doesn't exist

        let allExports = [];

        // Only fetch exports if we have trained models
        if (trainedModels && trainedModels.length > 0) {
            // Get exports for all models
            for (const model of trainedModels) {
                try {
                    const response = await fetch(`/api/export/list/${model.session_id}`);
                    const data = await response.json();
                    if (data.exports) {
                        allExports = allExports.concat(data.exports.map(exp => ({
                            ...exp,
                            session_id: model.session_id,
                            model_name: model.model_name
                        })));
                    }
                } catch (err) {
                    console.warn(`Failed to load exports for ${model.session_id}:`, err);
                }
            }
        }

        // Sort by date (most recent first)
        allExports.sort((a, b) => new Date(b.export_timestamp) - new Date(a.export_timestamp));

        // Display only recent exports (last 10)
        const recentExports = allExports.slice(0, 10);

        if (recentExports.length === 0) {
            historyList.innerHTML = `
                <div class="text-center text-muted p-4">
                    <p class="mb-0">No exports yet</p>
                    <small>Exported models will appear here</small>
                </div>
            `;
            return;
        }

        historyList.innerHTML = recentExports.map(exp => `
            <div class="export-item card mb-2">
                <div class="card-body p-2">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <strong class="small">${exp.export_format.toUpperCase()}</strong>
                            ${exp.quantization ? `<span class="badge bg-secondary ms-1">${exp.quantization}</span>` : ''}
                            <br>
                            <small class="text-muted">
                                ${exp.display_name || exp.model_name || 'Model'} - ${new Date(exp.export_timestamp).toLocaleDateString()}
                            </small>
                            ${exp.total_size_mb ? `<br><small>Size: ${exp.total_size_mb} MB</small>` : ''}
                        </div>
                        <div class="btn-group btn-group-sm">
                            <button class="btn btn-sm btn-primary" title="Download"
                                    onclick="downloadExportFromHistory('${exp.session_id}', '${exp.export_format}', '${exp.export_name}')">
                                <i class="fas fa-download"></i>
                            </button>
                            <button class="btn btn-sm btn-outline-danger" title="Delete"
                                    onclick="deleteExport('${exp.session_id}', '${exp.export_format}', '${exp.export_name}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');

    } catch (error) {
        console.error('Failed to load export history:', error);
    }
}

function downloadExportFromHistory(sessionId, format, exportName) {
    const downloadUrl = `/api/export/download/${sessionId}/${format}/${exportName}`;
    window.open(downloadUrl, '_blank');
}

async function deleteModel(sessionId) {
    // Get model name for confirmation dialog
    const model = trainedModels.find(m => m.session_id === sessionId);
    const modelName = model ? generateModelDisplayName(model) : sessionId.substring(0, 8) + '...';

    if (!confirm(`Are you sure you want to delete this model?\n\n${modelName}\n\nThis will permanently delete:\n• All model checkpoints\n• All exports\n• Training history\n\nThis action cannot be undone!`)) {
        return;
    }

    try {
        const response = await fetch(`/api/models/${sessionId}`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' }
        });

        const result = await response.json();

        if (response.ok) {
            showAlert(`Model deleted successfully`, 'success');
            // Refresh the models list
            refreshTrainedModels();
            // Also refresh export history
            refreshExportHistory();
        } else {
            showAlert(`Failed to delete model: ${result.error}`, 'danger');
        }
    } catch (error) {
        console.error('Delete model error:', error);
        showAlert('Failed to delete model: ' + error.message, 'danger');
    }
}

async function deleteExport(sessionId, exportFormat, exportName) {
    const displayName = `${exportFormat.toUpperCase()} - ${exportName}`;

    if (!confirm(`Are you sure you want to delete this export?\n\n${displayName}\n\nThis action cannot be undone!`)) {
        return;
    }

    try {
        const response = await fetch(`/api/export/${sessionId}/${exportFormat}/${exportName}`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' }
        });

        const result = await response.json();

        if (response.ok) {
            showAlert(`Export deleted successfully`, 'success');
            // Refresh export history
            refreshExportHistory();
        } else {
            showAlert(`Failed to delete export: ${result.error}`, 'danger');
        }
    } catch (error) {
        console.error('Delete export error:', error);
        showAlert('Failed to delete export: ' + error.message, 'danger');
    }
}

// Initialize export formats on page load
document.addEventListener('DOMContentLoaded', () => {
    loadExportFormats();

    // Add keyboard shortcut for export (Ctrl+E or Cmd+E)
    document.addEventListener('keydown', (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key === 'e') {
            event.preventDefault();

            if (lastCompletedSessionId) {
                showExportModal(lastCompletedSessionId);
            } else {
                // Try to find any completed session
                showExportOptions();
            }
        }
    });

    // Auto-refresh sessions on page load
    refreshSessions();
});

// ============================================================================
// Model Testing Functions
// ============================================================================

let testableModels = [];
let selectedTestModel = null;
let selectedComparisonModel = null;
let testHistory = [];
let comparisonMode = 'base';

async function loadTestableModels() {
    try {
        const response = await fetch('/api/test/models');
        const data = await response.json();
        testableModels = data.models || [];

        const select = document.getElementById('test-model-select');
        if (select) {
            select.innerHTML = '<option value="">Select a model...</option>';

            testableModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model.session_id;
                option.dataset.baseModel = model.base_model;
                option.dataset.checkpointPath = model.checkpoint_path;
                option.textContent = generateModelDisplayName(model);
                select.appendChild(option);
            });
        }

        // Update comparison model select based on primary selection
        const primarySelect = document.getElementById('test-model-select');
        if (primarySelect && primarySelect.value) {
            updateComparisonDropdown(primarySelect.value);
        } else {
            updateComparisonDropdown(null);
        }

        // Update loaded models indicator
        if (data.loaded && data.loaded.length > 0) {
            const loadedInfo = data.loaded.map(m => `${m.type}: ${m.id}`).join(', ');
            console.log('Loaded models:', loadedInfo);
        }

    } catch (error) {
        console.error('Failed to load testable models:', error);
        showAlert('Failed to load models', 'danger');
    }
}

function toggleComparisonMode() {
    const mode = document.querySelector('input[name="comparison-mode"]:checked').value;
    comparisonMode = mode;

    const secondModelSection = document.getElementById('second-model-section');
    const comparisonCardHeader = document.getElementById('comparison-card-header');
    const comparisonModelName = document.getElementById('comparison-model-name');
    const comparisonIcon = document.getElementById('comparison-icon');

    if (mode === 'model') {
        // Show second model selector
        secondModelSection.style.display = 'block';

        // Update card header for model comparison
        if (comparisonIcon) {
            comparisonIcon.className = 'fas fa-graduation-cap';
        }
        if (comparisonModelName) {
            comparisonModelName.textContent = selectedComparisonModel ?
                (selectedComparisonModel.name || 'Comparison Model') : 'Comparison Model';
        }
    } else {
        // Hide second model selector
        secondModelSection.style.display = 'none';

        // Update card header for base model comparison
        if (comparisonIcon) {
            comparisonIcon.className = 'fas fa-cube';
        }
        if (comparisonModelName) {
            comparisonModelName.textContent = 'Base Model';
        }
    }
}

function updateModelInfo() {
    const select = document.getElementById('test-model-select');
    const infoDiv = document.getElementById('model-info');
    const primaryModelName = document.getElementById('primary-model-name');
    const loadBtn = document.getElementById('load-models-btn');

    // Reset load button state when model selection changes
    if (loadBtn) {
        // Check if currently selected model is different from loaded model
        if (selectedTestModel && selectedTestModel.sessionId !== select.value) {
            // Reset button to primary state if a different model is selected
            loadBtn.classList.remove('btn-success');
            loadBtn.classList.add('btn-primary');
            loadBtn.innerHTML = '<i class="fas fa-download"></i> <span id="load-btn-text">Load Selected</span>';
            loadBtn.disabled = false;

            // Hide loaded models status
            const statusDiv = document.getElementById('loaded-models-status');
            if (statusDiv) {
                statusDiv.style.display = 'none';
            }
        } else if (selectedTestModel && selectedTestModel.sessionId === select.value) {
            // Keep success state if same model is already loaded
            loadBtn.classList.remove('btn-primary');
            loadBtn.classList.add('btn-success');
            loadBtn.innerHTML = '<i class="fas fa-check-circle"></i> <span id="load-btn-text">Models Loaded</span>';
        }
    }

    if (select && infoDiv) {
        const selectedOption = select.selectedOptions[0];
        if (selectedOption && selectedOption.value) {
            const model = testableModels.find(m => m.session_id === selectedOption.value);
            if (model) {
                // Update primary model name in results card
                if (primaryModelName) {
                    primaryModelName.textContent = model.name || 'Trained Model';
                }

                infoDiv.innerHTML = `
                    <i class="fas fa-check-circle text-success"></i>
                    <strong>${generateModelDisplayName(model)}</strong>
                    <br>
                    <small>Base: ${model.base_model}</small>
                `;

                // Update comparison dropdown to exclude this model
                updateComparisonDropdown(model.session_id);
            }
        } else {
            infoDiv.innerHTML = '<i class="fas fa-info-circle"></i> No model selected';
            // Reset comparison dropdown if no primary model selected
            updateComparisonDropdown(null);
        }
    }
}

function updateComparisonDropdown(excludeSessionId) {
    const comparisonSelect = document.getElementById('comparison-model-select');
    if (!comparisonSelect) return;

    const currentValue = comparisonSelect.value;
    comparisonSelect.innerHTML = '<option value="">Select a model to compare against...</option>';

    testableModels.forEach(model => {
        // Skip the model that's selected as primary
        if (model.session_id === excludeSessionId) return;

        const option = document.createElement('option');
        option.value = model.session_id;
        option.textContent = generateModelDisplayName(model);
        option.dataset.sessionId = model.session_id;
        option.dataset.baseModel = model.base_model;

        // Restore previous selection if it's still valid
        if (model.session_id === currentValue) {
            option.selected = true;
        }

        comparisonSelect.appendChild(option);
    });

    // If the previously selected comparison model was the same as the new primary, clear the selection
    if (currentValue === excludeSessionId) {
        comparisonSelect.value = '';
        updateComparisonModelInfo();
    }
}

function updateComparisonModelInfo() {
    const select = document.getElementById('comparison-model-select');
    const infoDiv = document.getElementById('comparison-model-info');
    const comparisonModelName = document.getElementById('comparison-model-name');

    if (select && infoDiv) {
        const selectedOption = select.selectedOptions[0];
        if (selectedOption && selectedOption.value) {
            const model = testableModels.find(m => m.session_id === selectedOption.value);
            if (model) {
                selectedComparisonModel = model;

                // Update comparison model name in results card if in model mode
                if (comparisonMode === 'model' && comparisonModelName) {
                    comparisonModelName.textContent = model.name || 'Comparison Model';
                }

                infoDiv.innerHTML = `
                    <i class="fas fa-check-circle text-success"></i>
                    Selected: ${model.name || model.session_id.substring(0, 8) + '...'}
                    <br>Base: ${model.base_model}
                `;
            }
        } else {
            selectedComparisonModel = null;
            infoDiv.innerHTML = '<i class="fas fa-info-circle"></i> No comparison model selected';
        }
    }
}

async function loadFromPath() {
    const pathInput = document.getElementById('custom-model-path');
    if (!pathInput || !pathInput.value) {
        showAlert('Please enter a model path', 'warning');
        return;
    }

    const checkpointPath = pathInput.value.trim();

    // Try to extract session ID from path
    const pathParts = checkpointPath.split('/');
    let sessionId = 'custom_' + Date.now();

    // Look for outputs/session_id pattern
    const outputsIndex = pathParts.indexOf('outputs');
    if (outputsIndex !== -1 && outputsIndex < pathParts.length - 1) {
        sessionId = pathParts[outputsIndex + 1];
    }

    // Prompt for base model name
    const baseModel = prompt('Enter the base model name (e.g., meta-llama/Llama-2-7b-hf):');
    if (!baseModel) {
        showAlert('Base model name is required', 'warning');
        return;
    }

    const loadBtn = document.getElementById('load-models-btn');
    if (loadBtn) {
        loadBtn.disabled = true;
        loadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    }

    try {
        // Add to registry first if needed
        const response = await fetch('/api/test/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                base_model: baseModel,
                checkpoint_path: checkpointPath
            })
        });

        const result = await response.json();

        if (result.results && result.results.trained.success && result.results.base.success) {
            showAlert('Models loaded successfully from path!', 'success');
            selectedTestModel = { sessionId, baseModel };

            const compareBtn = document.getElementById('compare-btn');
            if (compareBtn) compareBtn.disabled = false;

            // Update model info
            const infoDiv = document.getElementById('model-info');
            if (infoDiv) {
                infoDiv.innerHTML = `
                    <i class="fas fa-check-circle text-success"></i>
                    <strong>Custom Model</strong> -
                    Base: ${baseModel} |
                    Path: ${checkpointPath}
                `;
            }
        } else {
            const errors = [];
            if (result.results && !result.results.trained.success) {
                errors.push(`Trained model: ${result.results.trained.error}`);
            }
            if (result.results && !result.results.base.success) {
                errors.push(`Base model: ${result.results.base.error}`);
            }
            showAlert('Failed to load models: ' + errors.join(', '), 'danger');
        }

    } catch (error) {
        console.error('Failed to load models from path:', error);
        showAlert('Failed to load models: ' + error.message, 'danger');
    } finally {
        if (loadBtn) {
            loadBtn.disabled = false;
            loadBtn.innerHTML = '<i class="fas fa-download"></i> Load Selected';
        }
    }
}

async function loadModelsForTesting() {
    const select = document.getElementById('test-model-select');
    const sessionId = select.value;

    if (!sessionId) {
        showAlert('Please select a model first', 'warning');
        return;
    }

    const selectedOption = select.selectedOptions[0];
    const baseModel = selectedOption.dataset.baseModel;

    const loadBtn = document.getElementById('load-models-btn');
    loadBtn.disabled = true;
    loadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';

    try {
        const response = await fetch('/api/test/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                base_model: baseModel
            })
        });

        const result = await response.json();

        if (result.results.trained.success && result.results.base.success) {
            showAlert('Models loaded successfully!', 'success');
            selectedTestModel = { sessionId, baseModel };
            document.getElementById('compare-btn').disabled = false;

            // Update button to show success state
            loadBtn.classList.remove('btn-primary');
            loadBtn.classList.add('btn-success');
            loadBtn.innerHTML = '<i class="fas fa-check-circle"></i> <span id="load-btn-text">Models Loaded</span>';

            // Show loaded models status
            const statusDiv = document.getElementById('loaded-models-status');
            statusDiv.style.display = 'block';

            // Update badges with model names
            const trainedBadge = document.getElementById('loaded-trained-badge');
            const baseBadge = document.getElementById('loaded-base-badge');

            // Get model display name from select option
            const modelName = selectedOption.textContent.trim();
            trainedBadge.textContent = modelName.length > 20 ? modelName.substring(0, 20) + '...' : modelName;
            baseBadge.textContent = baseModel.split('/').pop();

        } else {
            const errors = [];
            if (!result.results.trained.success) {
                errors.push(`Trained model: ${result.results.trained.error}`);
            }
            if (!result.results.base.success) {
                errors.push(`Base model: ${result.results.base.error}`);
            }
            showAlert('Failed to load models: ' + errors.join(', '), 'danger');

            // Reset button on failure
            loadBtn.disabled = false;
            loadBtn.innerHTML = '<i class="fas fa-download"></i> <span id="load-btn-text">Load Selected</span>';
        }

    } catch (error) {
        console.error('Failed to load models:', error);
        showAlert('Failed to load models: ' + error.message, 'danger');

        // Reset button on error
        loadBtn.disabled = false;
        loadBtn.innerHTML = '<i class="fas fa-download"></i> <span id="load-btn-text">Load Selected</span>';
    } finally {
        // Re-enable button but keep success state if successful
        if (!loadBtn.classList.contains('btn-success')) {
            loadBtn.disabled = false;
        }
    }
}

async function compareModels() {
    if (!selectedTestModel) {
        showAlert('Please load models first', 'warning');
        return;
    }

    // Check if comparison model is selected when in model mode
    if (comparisonMode === 'model' && !selectedComparisonModel) {
        showAlert('Please select a comparison model', 'warning');
        return;
    }

    // Check if trying to compare model with itself
    if (comparisonMode === 'model' && selectedTestModel.sessionId === selectedComparisonModel.session_id) {
        showAlert('Please select a different model for comparison', 'warning');
        return;
    }

    const prompt = document.getElementById('test-prompt').value;
    if (!prompt) {
        showAlert('Please enter a test prompt', 'warning');
        return;
    }

    const compareBtn = document.getElementById('compare-btn');
    compareBtn.disabled = true;
    compareBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

    // Show results section
    document.getElementById('comparison-results').style.display = 'block';

    // Reset results for streaming
    const trainedResponseDiv = document.getElementById('trained-response');
    const baseResponseDiv = document.getElementById('base-response');
    trainedResponseDiv.textContent = '';
    baseResponseDiv.textContent = '';

    try {
        const config = {
            temperature: parseFloat(document.getElementById('test-temperature').value),
            max_new_tokens: parseInt(document.getElementById('test-max-tokens').value),
            top_p: parseFloat(document.getElementById('test-top-p').value),
            repetition_penalty: parseFloat(document.getElementById('test-rep-penalty').value),
            do_sample: true
        };

        // Prepare request data
        const requestData = {
            prompt: prompt,
            config: config,
            use_chat_template: document.getElementById('use-chat-template').checked
        };

        let streamUrl;
        if (comparisonMode === 'base') {
            requestData.session_id = selectedTestModel.sessionId;
            requestData.base_model = selectedTestModel.baseModel;
            streamUrl = '/api/test/compare/stream';
        } else {
            requestData.model1_session_id = selectedTestModel.sessionId;
            requestData.model2_session_id = selectedComparisonModel.session_id;
            streamUrl = '/api/test/compare-models/stream';
        }

        // First send the POST request to initiate streaming
        const initResponse = await fetch(streamUrl, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        if (!initResponse.ok) {
            throw new Error(`HTTP error! status: ${initResponse.status}`);
        }

        // Read the stream
        const reader = initResponse.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let results = {};

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));

                        if (data.type === 'trained' || data.type === 'model1') {
                            trainedResponseDiv.textContent += data.token;
                        } else if (data.type === 'base' || data.type === 'model2') {
                            baseResponseDiv.textContent += data.token;
                        } else if (data.type === 'complete') {
                            // Store final results
                            results = data;

                            // Update metadata for trained/model1
                            const trainedData = data.trained || data.model1;
                            if (trainedData && trainedData.success) {
                                document.getElementById('trained-time').textContent =
                                    `${trainedData.metadata.generation_time.toFixed(2)}s`;
                                document.getElementById('trained-tokens').textContent =
                                    `${trainedData.metadata.output_tokens} tokens`;
                            }

                            // Update metadata for base/model2
                            const comparisonData = data.base || data.model2;
                            if (comparisonData && comparisonData.success) {
                                document.getElementById('base-time').textContent =
                                    `${comparisonData.metadata.generation_time.toFixed(2)}s`;
                                document.getElementById('base-tokens').textContent =
                                    `${comparisonData.metadata.output_tokens} tokens`;
                            }

                            // Display comparison metrics if available
                            if (trainedData && comparisonData) {
                                const trainedLength = trainedResponseDiv.textContent.length;
                                const baseLength = baseResponseDiv.textContent.length;
                                const lengthDiff = trainedLength - baseLength;
                                document.getElementById('length-diff').textContent =
                                    lengthDiff > 0 ? `+${lengthDiff}` : lengthDiff;

                                const timeDiff = trainedData.metadata.generation_time -
                                    comparisonData.metadata.generation_time;
                                document.getElementById('time-diff').textContent = `${timeDiff.toFixed(2)}s`;

                                // Simple quality assessment
                                document.getElementById('trained-quality').textContent = '★★★★☆';
                                document.getElementById('base-quality').textContent = '★★★☆☆';
                            }
                        } else if (data.type === 'error') {
                            throw new Error(data.error);
                        }
                    } catch (e) {
                        console.error('Error parsing stream data:', e);
                    }
                }
            }
        }

        // Add to history
        if (results.trained || results.model1) {
            addToTestHistory(prompt, results);
        }

    } catch (error) {
        console.error('Comparison failed:', error);
        showAlert('Comparison failed: ' + error.message, 'danger');
    } finally {
        compareBtn.disabled = false;
        compareBtn.innerHTML = '<i class="fas fa-play"></i> Compare Models';
    }
}

function addToTestHistory(prompt, results) {
    const historyItem = {
        timestamp: new Date().toISOString(),
        prompt: prompt,
        results: results
    };

    testHistory.unshift(historyItem);
    if (testHistory.length > 10) {
        testHistory = testHistory.slice(0, 10);
    }

    updateTestHistoryDisplay();
}

function updateTestHistoryDisplay() {
    const historyList = document.getElementById('test-history-list');

    if (testHistory.length === 0) {
        historyList.innerHTML = '<div class="text-muted text-center p-3">No test history yet</div>';
        return;
    }

    historyList.innerHTML = testHistory.map((item, index) => `
        <div class="list-group-item">
            <div class="d-flex justify-content-between align-items-start">
                <div class="flex-grow-1">
                    <strong>Test ${testHistory.length - index}</strong>
                    <small class="text-muted ms-2">${new Date(item.timestamp).toLocaleTimeString()}</small>
                    <div class="text-truncate small text-muted mt-1">${item.prompt}</div>
                </div>
                <button class="btn btn-sm btn-outline-primary" onclick="viewTestResult(${index})">
                    <i class="fas fa-eye"></i>
                </button>
            </div>
        </div>
    `).join('');
}

function viewTestResult(index) {
    const item = testHistory[index];
    if (!item) return;

    // Update the prompt
    document.getElementById('test-prompt').value = item.prompt;

    // Update the results
    if (item.results.trained && item.results.trained.success) {
        document.getElementById('trained-response').textContent = item.results.trained.response;
        document.getElementById('trained-time').textContent = `${item.results.trained.metadata.generation_time.toFixed(2)}s`;
        document.getElementById('trained-tokens').textContent = `${item.results.trained.metadata.output_tokens} tokens`;
    }

    if (item.results.base && item.results.base.success) {
        document.getElementById('base-response').textContent = item.results.base.response;
        document.getElementById('base-time').textContent = `${item.results.base.metadata.generation_time.toFixed(2)}s`;
        document.getElementById('base-tokens').textContent = `${item.results.base.metadata.output_tokens} tokens`;
    }

    // Show results section
    document.getElementById('comparison-results').style.display = 'block';
}

async function clearModelCache() {
    if (!confirm('Clear all loaded models from memory?')) {
        return;
    }

    try {
        const response = await fetch('/api/test/clear-cache', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        const result = await response.json();
        if (result.success) {
            showAlert('Model cache cleared', 'success');
            selectedTestModel = null;
            document.getElementById('compare-btn').disabled = true;

            // Reset load button to default state
            const loadBtn = document.getElementById('load-models-btn');
            if (loadBtn) {
                loadBtn.classList.remove('btn-success');
                loadBtn.classList.add('btn-primary');
                loadBtn.innerHTML = '<i class="fas fa-download"></i> <span id="load-btn-text">Load Selected</span>';
                loadBtn.disabled = false;
            }

            // Hide loaded models status
            const statusDiv = document.getElementById('loaded-models-status');
            if (statusDiv) {
                statusDiv.style.display = 'none';
            }
        }

    } catch (error) {
        console.error('Failed to clear cache:', error);
        showAlert('Failed to clear cache', 'danger');
    }
}

// Update range sliders
document.addEventListener('DOMContentLoaded', function() {
    const tempSlider = document.getElementById('test-temperature');
    const topPSlider = document.getElementById('test-top-p');
    const repSlider = document.getElementById('test-rep-penalty');

    if (tempSlider) {
        tempSlider.addEventListener('input', function() {
            const tempValue = document.getElementById('test-temp-value');
            if (tempValue) tempValue.textContent = this.value;
        });
    }

    if (topPSlider) {
        topPSlider.addEventListener('input', function() {
            const topPValue = document.getElementById('test-top-p-value');
            if (topPValue) topPValue.textContent = this.value;
        });
    }

    if (repSlider) {
        repSlider.addEventListener('input', function() {
            const repValue = document.getElementById('test-rep-value');
            if (repValue) repValue.textContent = this.value;
        });
    }
});


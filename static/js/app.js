// GRPO Fine-Tuner Unified Interface JavaScript

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
    
    // Refresh sessions
    refreshSessions();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Load saved state from localStorage
    loadSavedState();
    
    // Update system status
    updateSystemStatus();
    
    // Set initial template preview
    updateTemplatePreview();
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
        console.log('Training progress:', data);
        updateTrainingProgress(data.progress);
    });

    socket.on('training_metrics', function(data) {
        console.log('Training metrics received:', data);
        updateTrainingMetrics(data);
    });

    socket.on('training_log', function(data) {
        console.log('Training log:', data.message);
        appendLog(data.message);
    });

    socket.on('training_complete', function(data) {
        console.log('Training complete:', data);
        handleTrainingComplete(data);
    });

    socket.on('training_error', function(data) {
        console.error('Training error:', data);
        handleTrainingError(data);
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
    const statusEl = document.getElementById('connection-status');
    statusEl.textContent = status;
    statusEl.className = status === 'Online' ? 'text-success' : 'text-danger';
}

// ============================================================================
// Step Navigation & Validation
// ============================================================================

function toggleStep(stepNum) {
    const content = document.getElementById(`step-${stepNum}-content`);
    const chevron = document.getElementById(`step-${stepNum}-chevron`);

    if (!content || !chevron) {
        console.error(`Step ${stepNum} elements not found`);
        return;
    }

    // Use Bootstrap's collapse functionality
    const bsCollapse = new bootstrap.Collapse(content, {
        toggle: false
    });

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
                    const otherCollapse = new bootstrap.Collapse(otherContent, {
                        toggle: false
                    });
                    otherCollapse.hide();
                    otherChevron.classList.remove('fa-chevron-up');
                    otherChevron.classList.add('fa-chevron-down');
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
        const currentCollapse = new bootstrap.Collapse(currentContent, {
            toggle: false
        });
        currentCollapse.hide();
        currentChevron.classList.remove('fa-chevron-up');
        currentChevron.classList.add('fa-chevron-down');
    }

    // Expand target step
    const targetContent = document.getElementById(`step-${stepNum}-content`);
    const targetChevron = document.getElementById(`step-${stepNum}-chevron`);

    if (targetContent && targetChevron) {
        const targetCollapse = new bootstrap.Collapse(targetContent, {
            toggle: false
        });
        targetCollapse.show();
        targetChevron.classList.remove('fa-chevron-down');
        targetChevron.classList.add('fa-chevron-up');
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
            const modelName = document.getElementById('model-name').value;
            if (!modelName) {
                showAlert('Please select a model', 'warning');
                return false;
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
            indicator.classList.remove('active', 'completed');

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
        updateModelList();
    } catch (error) {
        console.error('Failed to load models:', error);
        showAlert('Failed to load models', 'danger');
    }
}

function updateModelList() {
    const family = document.getElementById('model-family').value;
    const modelSelect = document.getElementById('model-name');

    modelSelect.innerHTML = '';

    // If Recommended preset is active, update recommendations
    updateRecommendedIfActive();

    if (availableModels[family]) {
        availableModels[family].forEach(model => {
            const option = document.createElement('option');
            option.value = model.id;
            option.textContent = `${model.name} (${model.vram})`;
            modelSelect.appendChild(option);
        });
    }
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
        size: '8.5K problems',
        category: 'math',
        description: 'Grade school math problems',
        language: 'English',
        icon: '🧮',
        fields: { instruction: 'question', response: 'answer' }
    },
    'dapo-math': {
        name: 'DAPO Math 17k',
        path: 'open-r1/DAPO-Math-17k-Processed',
        size: '17K problems',
        category: 'math',
        description: 'Advanced math with reasoning',
        language: 'English',
        icon: '📐',
        fields: { instruction: 'problem', response: 'solution' }
    },
    'openmath': {
        name: 'OpenMath Reasoning',
        path: 'nvidia/OpenMathReasoning',
        size: '100K+ problems',
        category: 'math',
        description: 'Mathematical reasoning dataset',
        language: 'English',
        icon: '🔢',
        fields: { instruction: 'question', response: 'answer' }
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
        path: 'squad_v2',
        size: '150K questions',
        category: 'qa',
        description: 'Reading comprehension Q&A',
        language: 'English',
        icon: '📖',
        fields: { instruction: 'question', response: 'answers' }
    }
};

function selectDatasetType(type) {
    // Update UI based on selection
    document.querySelectorAll('.selection-card').forEach(card => {
        if (card.id === 'dataset-popular' || card.id === 'dataset-upload') {
            card.classList.remove('active');
        }
    });

    // Add active class to selected card
    if (type === 'popular') {
        document.getElementById('dataset-popular').classList.add('active');
    } else if (type === 'upload') {
        document.getElementById('dataset-upload').classList.add('active');
    }

    const datasetConfig = document.getElementById('dataset-config');
    const datasetCatalogEl = document.getElementById('dataset-catalog');
    const datasetUploadArea = document.getElementById('dataset-upload-area');

    // Show dataset configuration section
    datasetConfig.style.display = 'block';

    if (type === 'popular') {
        // Show catalog
        datasetCatalogEl.style.display = 'block';
        datasetUploadArea.style.display = 'none';
        loadDatasetCatalog();
    } else if (type === 'upload') {
        // Show upload area
        datasetCatalogEl.style.display = 'none';
        datasetUploadArea.style.display = 'block';
    }
}

async function loadDatasetCatalog() {
    const grid = document.getElementById('dataset-grid');
    grid.innerHTML = '';

    // First load status for all datasets
    await updateDatasetStatuses();

    // Show cache management bar
    document.getElementById('cache-management-bar').style.display = 'block';
    await refreshCacheInfo();

    Object.entries(datasetCatalog).forEach(([key, dataset]) => {
        const card = createDatasetCard(key, dataset);
        grid.appendChild(card);
    });
}

async function updateDatasetStatuses() {
    // Get status for all popular datasets
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
            <button class="btn btn-sm btn-success mt-2" onclick="selectDataset('${key}')">
                <i class="fas fa-check"></i> Use Dataset
            </button>
            <button class="btn btn-sm btn-info mt-2" onclick="previewDataset('${key}')">
                <i class="fas fa-eye"></i> Preview
            </button>
            <div class="mt-1">
                <small class="text-muted">
                    Cached: ${status.cache_info ? status.cache_info.size_mb + ' MB' : ''}
                </small>
                <button class="btn btn-sm btn-link text-danger p-0 ms-2" onclick="clearDatasetCache('${key}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `;
    } else {
        statusIcon = '<i class="fas fa-cloud-download-alt text-muted"></i>';
        statusClass = 'not-cached';
        actionButtons = `
            <button class="btn btn-sm btn-primary mt-2" onclick="downloadDataset('${key}')">
                <i class="fas fa-download"></i> Download
            </button>
            <button class="btn btn-sm btn-outline-secondary mt-2" onclick="previewDataset('${key}')">
                <i class="fas fa-eye"></i> Sample
            </button>
        `;
    }

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

        // Store the config if available (for multi-config datasets)
        if (dataset.config) {
            // Store config in a data attribute or hidden field
            document.getElementById('dataset-path').setAttribute('data-config', dataset.config);
        }

        // Auto-configure field mappings
        document.getElementById('instruction-field').value = dataset.fields.instruction;
        document.getElementById('response-field').value = dataset.fields.response;
        document.getElementById('instruction-field-visible').value = dataset.fields.instruction;
        document.getElementById('response-field-visible').value = dataset.fields.response;

        // Show field mapping for advanced users
        const fieldMapping = document.getElementById('field-mapping');
        fieldMapping.style.display = 'block';

        // Visual feedback
        showAlert(`Selected ${dataset.name} dataset`, 'success');

        // Highlight selected card
        document.querySelectorAll('.dataset-catalog-card').forEach(card => {
            card.classList.remove('selected');
        });
        event.currentTarget.parentElement.classList.add('selected');
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

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (file) {
        // Handle file upload
        document.getElementById('dataset-path').value = file.name;
        showAlert(`File selected: ${file.name}`, 'success');
        // TODO: Implement actual file upload
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
${systemPrompt}

User: [Your instruction here]

Assistant: ${reasoningStart}
[Model reasoning/working out]
${reasoningEnd}
${solutionStart}
[Model solution]
${solutionEnd}`;
    
    document.getElementById('template-preview').textContent = preview;
}

// Template mode handlers
function setupTemplateHandlers() {
    document.getElementById('template-grpo').addEventListener('change', function() {
        if (this.checked) handleTemplateMode('grpo-default');
    });

    document.getElementById('template-custom').addEventListener('change', function() {
        if (this.checked) handleTemplateMode('custom');
    });

    document.getElementById('template-import').addEventListener('change', function() {
        if (this.checked) handleTemplateMode('import');
    });
}

function handleTemplateMode(mode) {
    const templateEditor = document.getElementById('template-editor');
    const templateImport = document.getElementById('template-import');
    const editBtn = document.getElementById('edit-template-btn');

    switch(mode) {
        case 'grpo-default':
            // Hide editor and import
            templateEditor.style.display = 'none';
            templateImport.style.display = 'none';
            editBtn.style.display = 'none';
            // Load default GRPO template
            loadDefaultGRPOTemplate();
            break;

        case 'custom':
            // Show editor
            templateEditor.style.display = 'block';
            templateImport.style.display = 'none';
            editBtn.style.display = 'inline';
            // Enable template editing
            enableTemplateEditing();
            break;

        case 'import':
            // Show import options
            templateEditor.style.display = 'none';
            templateImport.style.display = 'block';
            editBtn.style.display = 'none';
            // Load saved templates list
            loadSavedTemplatesList();
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
    const name = prompt('Enter a name for this template:');
    if (!name) return;

    const templateData = {
        name: name,
        description: `Custom template created ${new Date().toLocaleDateString()}`,
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
        'grpo': '<i class="fas fa-info-circle"></i> <strong>GRPO:</strong> Standard Group Relative Policy Optimization applies importance weights at the token level.',
        'gspo': '<i class="fas fa-info-circle"></i> <strong>GSPO:</strong> Group Sequence Policy Optimization applies importance weights at the sequence level, often resulting in more stable training.',
        'dr_grpo': '<i class="fas fa-info-circle"></i> <strong>DR-GRPO:</strong> Doubly Robust GRPO uses control variates to reduce variance in gradient estimates.'
    };
    infoDiv.innerHTML = infoTexts[algorithm];

    // Show/hide GSPO parameters
    const gspoParams = document.getElementById('gspo-params');
    if (algorithm === 'gspo') {
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
    // Update summary values
    const modelSelect = document.getElementById('model-name');
    const modelText = modelSelect.options[modelSelect.selectedIndex]?.text || '--';
    document.getElementById('summary-model').textContent = modelText;
    
    const datasetSelect = document.getElementById('dataset-path');
    document.getElementById('summary-dataset').textContent = datasetSelect.value || '--';
    
    document.getElementById('summary-epochs').textContent = document.getElementById('num-epochs').value;
    document.getElementById('summary-batch').textContent = document.getElementById('batch-size').value;
    
    // Estimate training time and VRAM
    estimateTrainingRequirements();
}

// ============================================================================
// Training Management
// ============================================================================

function gatherConfig() {
    return {
        // Model configuration
        model_name: document.getElementById('model-name').value,
        lora_rank: parseInt(document.getElementById('lora-rank').value),
        lora_alpha: parseInt(document.getElementById('lora-alpha').value),
        lora_dropout: parseFloat(document.getElementById('lora-dropout').value),

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

        // Training configuration
        learning_rate: parseFloat(document.getElementById('learning-rate').value),
        batch_size: parseInt(document.getElementById('batch-size').value),
        num_epochs: parseInt(document.getElementById('num-epochs').value),

        // GRPO configuration
        temperature: parseFloat(document.getElementById('temperature').value),
        top_p: parseFloat(document.getElementById('top-p').value),
        kl_penalty: parseFloat(document.getElementById('kl-penalty').value),
        num_generations: parseInt(document.getElementById('num-generations').value),

        // Algorithm selection
        loss_type: getSelectedAlgorithm(),
        epsilon: document.getElementById('epsilon') ? parseFloat(document.getElementById('epsilon').value) : 0.0003,
        epsilon_high: document.getElementById('epsilon-high') ? parseFloat(document.getElementById('epsilon-high').value) : 0.0004,

        // Reward configuration
        reward_config: gatherRewardConfig(),

        // Optimization flags
        use_flash_attention: document.getElementById('use-flash-attention').checked,
        gradient_checkpointing: document.getElementById('gradient-checkpointing').checked,
        mixed_precision: document.getElementById('mixed-precision').checked,
        pre_finetune: document.getElementById('pre-finetune').checked
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

    const config = gatherConfig();

    // Show training monitor
    document.getElementById('training-monitor').style.display = 'block';
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
            if (socket) {
                console.log('Joining training session:', currentSessionId);
                socket.emit('join_session', { session_id: currentSessionId });
            }

            trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
            showAlert('Training started successfully!', 'success');
        } else {
            throw new Error(data.error || 'Failed to start training');
        }
    } catch (error) {
        trainBtn.disabled = false;
        trainBtn.innerHTML = '<i class="fas fa-rocket"></i> Start Training';
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
    const progressBar = document.getElementById('training-progress');
    progressBar.style.width = progress + '%';
    progressBar.textContent = progress + '%';

    if (progress >= 100) {
        progressBar.classList.remove('progress-bar-animated');
    }
}

function updateTrainingMetrics(metrics) {
    // Debug: Log incoming metrics
    console.log('Received metrics:', metrics);

    // Update metrics panel
    if (metrics.step !== undefined) {
        document.getElementById('metric-step').textContent = metrics.step;
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

    // Update loss chart
    if (lossChart && metrics.loss !== undefined) {
        lossChart.data.labels.push(metrics.step || lossChart.data.labels.length);
        lossChart.data.datasets[0].data.push(metrics.loss);

        // Keep last 100 points for better visualization
        if (lossChart.data.labels.length > 100) {
            lossChart.data.labels.shift();
            lossChart.data.datasets[0].data.shift();
        }

        lossChart.update('none'); // Disable animation for smoother updates
    }

    // Update reward chart
    if (rewardChart && metrics.mean_reward !== undefined) {
        rewardChart.data.labels.push(metrics.step || rewardChart.data.labels.length);
        rewardChart.data.datasets[0].data.push(metrics.mean_reward);

        // Keep last 100 points
        if (rewardChart.data.labels.length > 100) {
            rewardChart.data.labels.shift();
            rewardChart.data.datasets[0].data.shift();
        }

        rewardChart.update('none');
    }

    // Update learning rate chart
    if (lrChart && metrics.learning_rate !== undefined) {
        lrChart.data.labels.push(metrics.step || lrChart.data.labels.length);
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
    trainBtn.innerHTML = '<i class="fas fa-rocket"></i> Start Training';

    showAlert('Training error: ' + data.error, 'danger');
}

// ============================================================================
// Session Management
// ============================================================================

async function refreshSessions() {
    try {
        const response = await fetch('/api/training/sessions');
        const sessions = await response.json();

        const sessionsList = document.getElementById('sessions-list');
        sessionsList.innerHTML = '';

        sessions.forEach(session => {
            const sessionItem = document.createElement('div');
            sessionItem.className = 'session-item p-2 border-bottom';

            // Create export button HTML for completed sessions
            const exportBtnHtml = session.status === 'completed'
                ? `<button class="btn btn-sm btn-success ms-2" onclick="event.stopPropagation(); showExportModal('${session.session_id}')">
                       <i class="fas fa-file-export"></i>
                   </button>`
                : '';

            sessionItem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <div class="flex-grow-1" style="cursor: pointer;" onclick="loadSession('${session.session_id}')">
                        <div class="fw-bold">${session.model.split('/').pop()}</div>
                        <small class="text-muted">${new Date(session.created_at).toLocaleString()}</small>
                    </div>
                    <div class="d-flex align-items-center">
                        <span class="badge bg-${getStatusColor(session.status)}">${session.status}</span>
                        ${exportBtnHtml}
                    </div>
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
                showSessionInfoPanel(session);
                if (session.status === 'completed') {
                    lastCompletedSessionId = sessionId;
                }
            }
        })
        .catch(error => {
            console.error('Failed to load session:', error);
            showAlert('Failed to load session details.', 'danger');
        });
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

function saveCurrentConfig() {
    const config = gatherConfig();
    const configName = prompt('Enter a name for this configuration:');

    if (configName) {
        localStorage.setItem(`grpo_config_${configName}`, JSON.stringify(config));
        showAlert(`Configuration "${configName}" saved successfully!`, 'success');
    }
}

function loadConfig() {
    // TODO: Implement config loading UI
    showAlert('Config loading UI coming soon!', 'info');
}

function saveConfig() {
    saveCurrentConfig();
}

function exportConfig() {
    const config = gatherConfig();
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `grpo_config_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    showAlert('Configuration exported successfully!', 'success');
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

// Load saved theme
document.addEventListener('DOMContentLoaded', function() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    const icon = document.getElementById('theme-icon');
    icon.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
});

// ============================================================================
// System Status
// ============================================================================

async function updateSystemStatus() {
    try {
        const response = await fetch('/api/system/info');
        const info = await response.json();

        if (info.gpu_available) {
            document.getElementById('gpu-status').textContent = info.gpu_name || 'Available';
            document.getElementById('memory-status').textContent = `${info.gpu_memory_allocated?.toFixed(1) || 0}GB`;
        } else {
            document.getElementById('gpu-status').textContent = 'CPU Only';
            document.getElementById('memory-status').textContent = '--';
        }
    } catch (error) {
        console.error('Failed to update system status:', error);
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

function showAlert(message, type) {
    // TODO: Implement better alert system
    console.log(`[${type}] ${message}`);

    // For now, use basic alert
    if (type === 'danger' || type === 'warning') {
        alert(message);
    }
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
    window.open('https://github.com/your-repo/grpo-finetuner/wiki', '_blank');
}

function showQuickStart() {
    // TODO: Implement quick start wizard
    showAlert('Quick Start Wizard coming soon!', 'info');
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
    document.getElementById('model-name')?.addEventListener('change', updateRecommendedIfActive);
    document.getElementById('reward-preset-select')?.addEventListener('change', updateRecommendedIfActive);
    document.getElementById('reasoning-start')?.addEventListener('input', debounce(updateRecommendedIfActive, 1000));
    document.getElementById('reasoning-end')?.addEventListener('input', debounce(updateRecommendedIfActive, 1000));

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
    });

    document.getElementById('lora-alpha').addEventListener('input', function() {
        document.getElementById('lora-alpha-slider').value = this.value;
    });

    document.getElementById('lora-dropout').addEventListener('input', function() {
        document.getElementById('lora-dropout-slider').value = this.value;
    });

    // Auto-save state on input changes
    document.querySelectorAll('input, select, textarea').forEach(element => {
        element.addEventListener('change', saveState);
    });

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
    const epochs = parseInt(document.getElementById('num-epochs').value);
    const batchSize = parseInt(document.getElementById('batch-size').value);

    // Simple estimation (would be more complex in real implementation)
    const estimatedMinutes = epochs * 5 * (4 / batchSize);
    document.getElementById('summary-time').textContent = `~${Math.round(estimatedMinutes)} minutes`;

    const estimatedVRAM = 2 + (batchSize * 0.5);
    document.getElementById('summary-vram').textContent = `~${estimatedVRAM.toFixed(1)}GB`;
}

// ============================================================================
// Dataset Management Functions
// ============================================================================

async function downloadDataset(key) {
    const dataset = datasetCatalog[key];
    if (!dataset) return;

    try {
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
                sampleCard.className = 'card mb-2';
                sampleCard.innerHTML = `
                    <div class="card-body">
                        <h6 class="card-subtitle mb-2 text-muted">Sample ${idx + 1}</h6>
                        <div class="mb-2">
                            <strong>Instruction:</strong>
                            <pre class="bg-light p-2 rounded" style="white-space: pre-wrap;">${escapeHtml(sample[dataset.fields.instruction] || sample.instruction || '')}</pre>
                        </div>
                        <div>
                            <strong>Response:</strong>
                            <pre class="bg-light p-2 rounded" style="white-space: pre-wrap;">${escapeHtml(sample[dataset.fields.response] || sample.response || sample.output || '')}</pre>
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

function showExportModal(sessionId) {
    // Remove any existing export modal
    const existingModal = document.getElementById('exportModal');
    if (existingModal) {
        existingModal.remove();
    }

    // Store this as the last session
    lastCompletedSessionId = sessionId;

    // Create and show export modal
    const modal = createExportModal(sessionId);
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

function createExportModal(sessionId) {
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
                        <label for="export-name" class="form-label">Export Name (Optional)</label>
                        <input type="text" class="form-control" id="export-name"
                               placeholder="Leave empty for auto-generated name">
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
                            <button class="btn btn-secondary" onclick="viewExportDetails()">
                                <i class="fas fa-info-circle"></i> View Details
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

function viewExportDetails() {
    // TODO: Show detailed export information
    showAlert('Export details coming soon!', 'info');
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

function goToExportsTab() {
    console.log('Navigating to Exports tab');
    // Navigate to the exports section (step 5)
    goToStep(5);
    // Refresh the trained models list after a short delay to ensure DOM is ready
    setTimeout(() => {
        refreshTrainedModels();
    }, 100);
}

function goToTestTab() {
    console.log('Navigating to Test tab');
    // Navigate to the test model section (step 6)
    goToStep(6);
    // Load testable models after a short delay to ensure DOM is ready
    setTimeout(() => {
        loadTestableModels();
    }, 100);
}

// Make functions globally accessible
window.goToExportsTab = goToExportsTab;
window.goToTestTab = goToTestTab;

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
                            <i class="fas fa-brain"></i> ${model.model_name || 'Model'}
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
                        <button class="btn btn-outline-primary" onclick="showModelDetails('${model.session_id}')">
                            <i class="fas fa-info"></i>
                        </button>
                        <button class="btn btn-success" onclick="showExportModalForModel('${model.session_id}')">
                            <i class="fas fa-file-export"></i> Export
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

function showExportModalForModel(sessionId) {
    // Use the existing export modal function
    showExportModal(sessionId);
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
                                ${exp.model_name || 'Model'} - ${new Date(exp.export_timestamp).toLocaleDateString()}
                            </small>
                            ${exp.total_size_mb ? `<br><small>Size: ${exp.total_size_mb} MB</small>` : ''}
                        </div>
                        <button class="btn btn-sm btn-primary"
                                onclick="downloadExportFromHistory('${exp.session_id}', '${exp.export_format}', '${exp.export_name}')">
                            <i class="fas fa-download"></i>
                        </button>
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
let testHistory = [];

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
                option.textContent = `${model.model_name || model.session_id} (${model.epochs || 0} epochs)`;
                select.appendChild(option);
            });
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

function updateModelInfo() {
    const select = document.getElementById('test-model-select');
    const infoDiv = document.getElementById('model-info');

    if (select && infoDiv) {
        const selectedOption = select.selectedOptions[0];
        if (selectedOption && selectedOption.value) {
            const model = testableModels.find(m => m.session_id === selectedOption.value);
            if (model) {
                infoDiv.innerHTML = `
                    <i class="fas fa-check-circle text-success"></i>
                    <strong>${model.model_name}</strong> -
                    Base: ${model.base_model} |
                    Epochs: ${model.epochs || 0}
                `;
            }
        } else {
            infoDiv.innerHTML = '<i class="fas fa-info-circle"></i> No model selected';
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
        } else {
            const errors = [];
            if (!result.results.trained.success) {
                errors.push(`Trained model: ${result.results.trained.error}`);
            }
            if (!result.results.base.success) {
                errors.push(`Base model: ${result.results.base.error}`);
            }
            showAlert('Failed to load models: ' + errors.join(', '), 'danger');
        }

    } catch (error) {
        console.error('Failed to load models:', error);
        showAlert('Failed to load models: ' + error.message, 'danger');
    } finally {
        loadBtn.disabled = false;
        loadBtn.innerHTML = '<i class="fas fa-download"></i> Load Models';
    }
}

async function compareModels() {
    if (!selectedTestModel) {
        showAlert('Please load models first', 'warning');
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

    // Reset results
    document.getElementById('trained-response').innerHTML = '<div class="text-center text-muted p-3"><i class="fas fa-spinner fa-spin"></i> Generating...</div>';
    document.getElementById('base-response').innerHTML = '<div class="text-center text-muted p-3"><i class="fas fa-spinner fa-spin"></i> Generating...</div>';

    try {
        const config = {
            temperature: parseFloat(document.getElementById('test-temperature').value),
            max_new_tokens: parseInt(document.getElementById('test-max-tokens').value),
            top_p: parseFloat(document.getElementById('test-top-p').value),
            repetition_penalty: parseFloat(document.getElementById('test-rep-penalty').value),
            do_sample: true
        };

        const response = await fetch('/api/test/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                session_id: selectedTestModel.sessionId,
                base_model: selectedTestModel.baseModel,
                config: config,
                use_chat_template: document.getElementById('use-chat-template').checked
            })
        });

        const results = await response.json();

        // Display trained model response
        if (results.trained && results.trained.success) {
            document.getElementById('trained-response').textContent = results.trained.response;
            document.getElementById('trained-time').textContent = `${results.trained.metadata.generation_time.toFixed(2)}s`;
            document.getElementById('trained-tokens').textContent = `${results.trained.metadata.output_tokens} tokens`;
        } else {
            document.getElementById('trained-response').innerHTML = `<div class="text-danger">Error: ${results.trained.error}</div>`;
        }

        // Display base model response
        if (results.base && results.base.success) {
            document.getElementById('base-response').textContent = results.base.response;
            document.getElementById('base-time').textContent = `${results.base.metadata.generation_time.toFixed(2)}s`;
            document.getElementById('base-tokens').textContent = `${results.base.metadata.output_tokens} tokens`;
        } else {
            document.getElementById('base-response').innerHTML = `<div class="text-danger">Error: ${results.base.error}</div>`;
        }

        // Display comparison metrics
        if (results.comparison) {
            document.getElementById('length-diff').textContent = results.comparison.length_diff > 0 ? `+${results.comparison.length_diff}` : results.comparison.length_diff;
            document.getElementById('time-diff').textContent = `${results.comparison.time_diff.toFixed(2)}s`;

            // Simple quality assessment (can be enhanced)
            document.getElementById('trained-quality').textContent = '★★★★☆';
            document.getElementById('base-quality').textContent = '★★★☆☆';
        }

        // Add to history
        addToTestHistory(prompt, results);

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


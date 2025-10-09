// ============================================================================
// Enhanced Reward System Functions
// ============================================================================

let rewardPresets = {};
let rewardTemplates = {};
// Make selectedRewardConfig global for integration with app.js
// Default to first quick-start template (math_problem_solving)
window.selectedRewardConfig = { type: 'preset', preset_name: 'math' };
let currentlySelectedCard = null;
let selectedRewardName = null;
let selectedRewardType = null;
let isRestoringSession = false; // Track if we're restoring a saved session

// Compatibility function for notifications
function showNotification(message, type = 'info') {
    // Use showAlert if available, otherwise console.log
    if (typeof showAlert === 'function') {
        showAlert(message, type === 'error' ? 'danger' : type);
    } else {
        console.log(`[${type}] ${message}`);
    }
}

// View reward details function
function viewRewardDetails() {
    if (!selectedRewardName || !selectedRewardType) {
        alert("Please select a reward first");
        return;
    }

    // For custom rewards, show a different modal or message
    if (selectedRewardType === 'custom' || selectedRewardName.includes('Custom')) {
        // Custom rewards don't have preset details, show component info instead
        if (window.selectedRewardConfig && window.selectedRewardConfig.components) {
            // Could create a custom modal here, but for now just show notification
            showNotification('Custom reward details are shown in the component display below', 'info');
        } else {
            showNotification('No details available for custom reward', 'warning');
        }
        return;
    }

    // Find the preset key
    const presetKey = Object.keys(rewardPresets).find(key =>
        rewardPresets[key].name === selectedRewardName
    );

    if (presetKey) {
        // Call the same function as "Components" button
        showPresetDetails(presetKey);
    } else {
        showNotification('Could not find reward details', 'warning');
    }
}

// Show reward details in a modal
function showRewardDetailsModal(reward) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('rewardDetailsModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'rewardDetailsModal';
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Reward Details</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body" id="rewardDetailsBody">
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    }

    // Populate modal body
    const body = document.getElementById('rewardDetailsBody');
    body.innerHTML = `
        <h5>${reward.name}</h5>
        <p class="text-muted">${reward.description}</p>

        <div class="mb-3">
            <strong>Category:</strong> ${reward.category || 'N/A'}
        </div>

        <div class="mb-3">
            <strong>Difficulty:</strong>
            <span class="badge bg-${reward.difficulty === 'beginner' ? 'success' : reward.difficulty === 'intermediate' ? 'warning' : 'danger'}">
                ${reward.difficulty || 'N/A'}
            </span>
        </div>

        ${reward.tags ? `
        <div class="mb-3">
            <strong>Tags:</strong>
            ${reward.tags.map(tag => `<span class="badge bg-secondary me-1">${tag}</span>`).join('')}
        </div>
        ` : ''}

        ${reward.example_input ? `
        <div class="mb-3">
            <strong>Example Input:</strong>
            <pre class="bg-light p-2 rounded" style="white-space: pre-wrap;">${reward.example_input}</pre>
        </div>
        ` : ''}

        ${reward.example_output ? `
        <div class="mb-3">
            <strong>Example Output:</strong>
            <pre class="bg-light p-2 rounded" style="white-space: pre-wrap;">${reward.example_output}</pre>
        </div>
        ` : ''}

        ${reward.reward_preset ? `
        <div class="mb-3">
            <strong>Uses Preset:</strong> ${reward.reward_preset}
        </div>
        ` : ''}

        ${reward.recommended_settings ? `
        <div class="mb-3">
            <strong>Recommended Settings:</strong>
            <ul>
                ${Object.entries(reward.recommended_settings).map(([key, value]) =>
                    `<li>${key}: ${value}</li>`
                ).join('')}
            </ul>
        </div>
        ` : ''}
    `;

    // Show modal
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
}

// Initialize reward system
async function initializeRewardSystem() {
    try {
        // Load presets
        const presetsResponse = await fetch('/api/rewards/presets');
        const presetsData = await presetsResponse.json();
        rewardPresets = presetsData.presets || {};

        // Load custom presets
        const customResponse = await fetch('/api/rewards/custom-presets');
        const customData = await customResponse.json();
        const customPresets = customData.presets || {};

        // Merge custom presets into main presets
        rewardPresets = {...rewardPresets, ...customPresets};

        // Populate UI if elements exist
        if (document.getElementById('preset-categories')) {
            populatePresetCategories();
        }

        // Initialize first category (default to "all")
        if (Object.keys(rewardPresets).length > 0) {
            filterPresetsByCategory('all');
        }
    } catch (error) {
        console.error('Failed to initialize reward system:', error);
        // Fallback to existing system
    }
}

function populatePresetCategories() {
    const categoriesDiv = document.getElementById('preset-categories');
    if (!categoriesDiv) return;

    const categories = [...new Set(Object.values(rewardPresets).map(p => p.category))];
    categoriesDiv.innerHTML = '';

    // Add "All" category
    categoriesDiv.innerHTML += `
        <a href="#" class="list-group-item list-group-item-action active"
           onclick="filterPresetsByCategory('all'); return false;">
            <i class="fas fa-th"></i> All Presets
        </a>
    `;

    categories.forEach(category => {
        categoriesDiv.innerHTML += `
            <a href="#" class="list-group-item list-group-item-action"
               onclick="filterPresetsByCategory('${category}'); return false;">
                <i class="fas fa-folder"></i> ${category}
            </a>
        `;
    });
}

function filterPresetsByCategory(category) {
    const presetListDiv = document.getElementById('preset-list');
    if (!presetListDiv) return;

    // Update active category
    document.querySelectorAll('#preset-categories .list-group-item').forEach(item => {
        item.classList.remove('active');
        if (category === 'all' && item.textContent.includes('All Presets')) {
            item.classList.add('active');
        } else if (item.textContent.includes(category)) {
            item.classList.add('active');
        }
    });

    // Filter and display presets
    presetListDiv.innerHTML = '';

    // Add "Custom Reward" card at the beginning if showing all or Custom category
    if (category === 'all' || category === 'Custom') {
        presetListDiv.innerHTML += `
            <div class="col-md-6 mb-3">
                <div class="card preset-card h-100 border-primary" onclick="openCustomRewardBuilder()" style="cursor: pointer;">
                    <div class="card-body">
                        <h6 class="card-title">
                            <i class="fas fa-tools text-primary"></i> Custom Reward
                        </h6>
                        <p class="card-text small text-muted">Build your own custom reward by combining components tailored to your specific task</p>
                        <div class="mt-2">
                            <span class="badge bg-primary">Build Your Own</span>
                        </div>
                        <div class="mt-2">
                            <button class="btn btn-sm btn-primary" onclick="event.stopPropagation(); openCustomRewardBuilder()">
                                <i class="fas fa-plus"></i> Create Custom
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    Object.entries(rewardPresets).forEach(([name, preset]) => {
        if (category === 'all' || preset.category === category) {
            const difficultyColor = {
                'beginner': 'success',
                'intermediate': 'warning',
                'advanced': 'danger'
            }[preset.difficulty] || 'secondary';

            const isCustomPreset = preset.category === 'Custom';

            presetListDiv.innerHTML += `
                <div class="col-md-6 mb-3">
                    <div class="card preset-card h-100" onclick="selectPresetByName('${name}')">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start">
                                <h6 class="card-title">${preset.name}</h6>
                                ${isCustomPreset ? `
                                <button class="btn btn-sm btn-outline-danger"
                                        onclick="event.stopPropagation(); deleteCustomPreset('${name}')"
                                        title="Delete custom preset">
                                    <i class="fas fa-trash"></i>
                                </button>
                                ` : ''}
                            </div>
                            <p class="card-text small text-muted">${preset.description}</p>
                            <div class="mt-2">
                                <span class="badge bg-${difficultyColor}">${preset.difficulty}</span>
                                ${preset.tags.map(tag => `<span class="badge bg-light text-dark ms-1">${tag}</span>`).join('')}
                            </div>
                            <div class="mt-2 d-flex gap-2">
                                <button class="btn btn-sm btn-outline-primary"
                                        onclick="event.stopPropagation(); showPresetDetails('${name}')">
                                    <i class="fas fa-puzzle-piece"></i> Components
                                </button>
                                <button class="btn btn-sm btn-outline-info"
                                        onclick="event.stopPropagation(); showPresetExample('${name}')">
                                    <i class="fas fa-eye"></i> Example
                                </button>
                                <button class="btn btn-sm btn-outline-success"
                                        onclick="event.stopPropagation(); showFieldMappingForPreset('${name}');">
                                    <i class="fas fa-link"></i> Map Fields
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }
    });
}

async function selectPresetByName(presetName, silent = false) {
    const preset = rewardPresets[presetName];
    if (!preset) return;

    // Update internal state
    window.selectedRewardConfig = {
        type: 'preset',
        preset_name: presetName
    };
    selectedRewardName = preset.name;
    selectedRewardType = 'preset';

    // Sync with AppState for config save/load
    if (window.AppState && AppState.setConfigValue) {
        AppState.setConfigValue('rewardConfig', window.selectedRewardConfig);
    }

    // Clear previous selection
    document.querySelectorAll('.preset-card, .template-card').forEach(card => {
        card.classList.remove('selected');
    });

    // Mark new selection
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('selected');
        currentlySelectedCard = event.currentTarget;
    } else {
        // Find and mark the card if not triggered by event
        const cards = document.querySelectorAll('.preset-card');
        cards.forEach(card => {
            if (card.textContent.includes(preset.name)) {
                card.classList.add('selected');
                currentlySelectedCard = card;
            }
        });
    }

    // Update display panel
    const nameElement = document.getElementById('selected-reward-name');
    const descElement = document.getElementById('selected-reward-description');
    if (nameElement) nameElement.textContent = preset.name;
    if (descElement) descElement.textContent = preset.description;

    // Fetch and display component details
    try {
        const response = await fetch(`/api/rewards/preset-details/${presetName}`);
        if (response.ok) {
            const details = await response.json();
            displayPresetComponents(details);
        }
    } catch (error) {
        console.error('Failed to fetch preset details:', error);
    }

    // Visual feedback (suppress for auto-restored selections)
    if (!silent) {
        showNotification(`Selected: ${preset.name}`, 'success');
    }

    // Animate the selection
    if (currentlySelectedCard) {
        currentlySelectedCard.classList.add('selecting');
        setTimeout(() => {
            if (currentlySelectedCard) {
                currentlySelectedCard.classList.remove('selecting');
            }
        }, 500);
    }

    // Store selection in localStorage for persistence
    localStorage.setItem('selectedReward', JSON.stringify(selectedRewardConfig));
}

function displayPresetComponents(details) {
    // Find or create component display area
    let componentDisplay = document.getElementById('preset-component-display');
    if (!componentDisplay) {
        // Create display area if it doesn't exist
        const selectedRewardDisplay = document.getElementById('selected-reward-display');
        if (!selectedRewardDisplay) return;

        componentDisplay = document.createElement('div');
        componentDisplay.id = 'preset-component-display';
        componentDisplay.className = 'mt-3';
        selectedRewardDisplay.appendChild(componentDisplay);
    }

    // Generate component HTML
    const componentsHtml = details.components.map(comp => {
        const weightPercentage = comp.weight_percentage.toFixed(1);
        const barWidth = Math.max(5, comp.weight_percentage); // Minimum 5% for visibility

        // Choose icon based on component type
        const iconMap = {
            'binary': 'fas fa-toggle-on',
            'continuous': 'fas fa-sliders-h',
            'format': 'fas fa-code',
            'numerical': 'fas fa-calculator',
            'length': 'fas fa-ruler',
            'template': 'fas fa-file-alt',
            'choice': 'fas fa-list-ul',
            'content': 'fas fa-paragraph',
            'pattern': 'fas fa-search'
        };

        const icon = iconMap[comp.type.toLowerCase()] || 'fas fa-cog';

        // Format parameters for display
        let paramHtml = '';
        if (comp.parameters && Object.keys(comp.parameters).length > 0) {
            paramHtml = '<div class="component-params mt-2 small text-muted">';
            for (const [key, value] of Object.entries(comp.parameters)) {
                if (value !== null && value !== undefined) {
                    let displayValue = value;
                    if (Array.isArray(value)) {
                        displayValue = value.length > 0 ? value.join(', ') : 'None';
                    } else if (typeof value === 'boolean') {
                        displayValue = value ? 'Yes' : 'No';
                    } else if (typeof value === 'number') {
                        // Check for NaN
                        displayValue = isNaN(value) ? 'Not set' : value;
                    } else if (value === '') {
                        displayValue = 'Not set';
                    }
                    paramHtml += `<div><strong>${key.replace(/_/g, ' ')}:</strong> ${displayValue}</div>`;
                }
            }
            paramHtml += '</div>';
        }

        return `
            <div class="component-item mb-3 p-3 border rounded">
                <div class="d-flex align-items-center justify-content-between mb-2">
                    <div class="d-flex align-items-center">
                        <i class="${icon} me-2 text-primary"></i>
                        <strong>${comp.name}</strong>
                    </div>
                    <span class="badge bg-secondary">${weightPercentage}%</span>
                </div>
                <div class="component-description small text-muted mb-2">
                    ${comp.description}
                </div>
                <div class="progress mb-2" style="height: 8px;">
                    <div class="progress-bar bg-primary" role="progressbar"
                         style="width: ${barWidth}%"
                         aria-valuenow="${comp.weight_percentage}"
                         aria-valuemin="0"
                         aria-valuemax="100">
                    </div>
                </div>
                ${paramHtml}
            </div>
        `;
    }).join('');

    // Add weight validation indicator
    const weightStatus = details.weight_valid
        ? '<span class="badge bg-success"><i class="fas fa-check"></i> Weight Valid (1.0)</span>'
        : `<span class="badge bg-danger"><i class="fas fa-exclamation-triangle"></i> Weight Invalid (${details.total_weight.toFixed(2)})</span>`;

    componentDisplay.innerHTML = `
        <div class="card">
            <div class="card-header bg-primary text-white">
                <div class="d-flex justify-content-between align-items-center">
                    <span><i class="fas fa-puzzle-piece"></i> Reward Components</span>
                    <div>
                        ${weightStatus}
                        <button class="btn btn-sm btn-light ms-2" onclick='makePresetEditable(${JSON.stringify(details).replace(/'/g, "&#39;")})'>
                            <i class="fas fa-edit"></i> Customize
                        </button>
                    </div>
                </div>
            </div>
            <div class="card-body">
                <div class="components-list" id="components-list">
                    ${componentsHtml}
                </div>
            </div>
        </div>
    `;
}

// Store original preset details for reset functionality
let originalPresetDetails = null;

function makePresetEditable(details) {
    // Store original for reset
    originalPresetDetails = JSON.parse(JSON.stringify(details));

    const componentsList = document.getElementById('components-list');
    if (!componentsList) return;

    // Generate editable component HTML
    const editableComponentsHtml = details.components.map((comp, index) => {
        const icon = {
            'binary': 'fas fa-toggle-on',
            'continuous': 'fas fa-sliders-h',
            'format': 'fas fa-code',
            'numerical': 'fas fa-calculator',
            'length': 'fas fa-ruler',
            'template': 'fas fa-file-alt',
            'choice': 'fas fa-list-ul',
            'content': 'fas fa-paragraph',
            'pattern': 'fas fa-search'
        }[comp.type.toLowerCase()] || 'fas fa-cog';

        // Generate editable parameter inputs with improved layout
        let paramInputsHtml = '';
        let paramCount = 0;

        if (comp.parameters && Object.keys(comp.parameters).length > 0) {
            const paramFields = [];

            for (const [key, value] of Object.entries(comp.parameters)) {
                if (value !== null && value !== undefined) {
                    paramCount++;
                    const inputId = `param-${index}-${key}`;
                    let inputHtml = '';

                    if (typeof value === 'boolean') {
                        inputHtml = `
                            <div class="param-field">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="${inputId}" ${value ? 'checked' : ''}>
                                    <label class="form-check-label" for="${inputId}">${key.replace(/_/g, ' ')}</label>
                                </div>
                            </div>
                        `;
                    } else if (typeof value === 'number') {
                        inputHtml = `
                            <div class="param-field">
                                <label class="form-label small fw-bold">${key.replace(/_/g, ' ')}</label>
                                <input type="number" class="form-control" id="${inputId}" value="${value}" step="any">
                            </div>
                        `;
                    } else if (Array.isArray(value)) {
                        inputHtml = `
                            <div class="param-field">
                                <label class="form-label small fw-bold">${key.replace(/_/g, ' ')}</label>
                                <input type="text" class="form-control" id="${inputId}" value="${value.join(', ')}" placeholder="Comma-separated values">
                            </div>
                        `;
                    } else {
                        inputHtml = `
                            <div class="param-field">
                                <label class="form-label small fw-bold">${key.replace(/_/g, ' ')}</label>
                                <input type="text" class="form-control" id="${inputId}" value="${value}">
                            </div>
                        `;
                    }
                    paramFields.push(inputHtml);
                }
            }

            if (paramCount > 0) {
                paramInputsHtml = `
                    <div class="params-grid mt-3">
                        ${paramFields.join('')}
                    </div>
                `;
            }
        }

        return `
            <div class="component-item mb-3 p-3 border rounded" data-index="${index}">
                <div class="component-content">
                    <!-- Component Name -->
                    <div class="d-flex align-items-center mb-2">
                        <i class="${icon} me-2 text-primary"></i>
                        <strong>${comp.name}</strong>
                    </div>

                    <!-- Description -->
                    <div class="component-description small text-muted mb-3">
                        ${comp.description}
                    </div>

                    <!-- Parameters -->
                    ${paramInputsHtml}
                </div>

                <!-- Weight Input -->
                <div class="component-weight">
                    <label class="form-label small fw-bold">Weight:</label>
                    <input type="number" class="form-control weight-input"
                           value="${comp.weight}"
                           step="0.1"
                           min="0"
                           data-index="${index}"
                           onchange="updateWeightTotal()">
                </div>
            </div>
        `;
    }).join('');

    componentsList.innerHTML = editableComponentsHtml;

    // Update header with action buttons
    const cardHeader = componentsList.closest('.card').querySelector('.card-header');
    cardHeader.innerHTML = `
        <div class="d-flex justify-content-between align-items-center">
            <span><i class="fas fa-edit"></i> Customize Reward</span>
            <div>
                <span id="weight-total-display" class="badge bg-light text-dark me-2"></span>
                <button class="btn btn-sm btn-secondary me-1" onclick="resetPresetToOriginal()">
                    <i class="fas fa-undo"></i> Reset
                </button>
                <button class="btn btn-sm btn-success" onclick="saveCustomizedPreset()">
                    <i class="fas fa-check"></i> Apply Changes
                </button>
            </div>
        </div>
    `;

    // Initial weight total update
    updateWeightTotal();
}

function updateWeightTotal() {
    const weightInputs = document.querySelectorAll('.weight-input');
    let total = 0;
    weightInputs.forEach(input => {
        total += parseFloat(input.value) || 0;
    });

    const display = document.getElementById('weight-total-display');
    if (display) {
        const isValid = Math.abs(total - 1.0) < 0.001;
        display.className = `badge ${isValid ? 'bg-success' : 'bg-warning'} text-white me-2`;
        display.innerHTML = `Total: ${total.toFixed(3)} ${isValid ? '✓' : ''}`;
    }
}

function resetPresetToOriginal() {
    if (originalPresetDetails) {
        displayPresetComponents(originalPresetDetails);
        originalPresetDetails = null;
    }
}

function saveCustomizedPreset() {
    const components = [];
    const componentItems = document.querySelectorAll('.component-item');

    componentItems.forEach((item, index) => {
        const weightInput = item.querySelector('.weight-input');
        const originalComp = originalPresetDetails.components[index];

        // Gather parameters
        const parameters = {};
        if (originalComp.parameters) {
            for (const key of Object.keys(originalComp.parameters)) {
                const inputId = `param-${index}-${key}`;
                const input = document.getElementById(inputId);

                if (input) {
                    if (input.type === 'checkbox') {
                        parameters[key] = input.checked;
                    } else if (input.type === 'number') {
                        parameters[key] = parseFloat(input.value);
                    } else if (originalComp.parameters[key] !== null && Array.isArray(originalComp.parameters[key])) {
                        parameters[key] = input.value.split(',').map(v => v.trim()).filter(v => v);
                    } else {
                        parameters[key] = input.value;
                    }
                }
            }
        }

        components.push({
            type: originalComp.type,
            name: originalComp.name,
            weight: parseFloat(weightInput.value) || 0,
            parameters: parameters
        });
    });

    // Check weight total
    const totalWeight = components.reduce((sum, c) => sum + c.weight, 0);
    if (Math.abs(totalWeight - 1.0) > 0.001) {
        if (!confirm(`Weight total is ${totalWeight.toFixed(3)}, not 1.0. Continue anyway?`)) {
            return;
        }
    }

    // Update global reward config
    window.selectedRewardConfig = {
        type: 'custom',
        components: components
    };

    selectedRewardName = `Customized ${originalPresetDetails.name}`;
    selectedRewardType = 'custom';

    // Update display
    const nameElement = document.getElementById('selected-reward-name');
    const descElement = document.getElementById('selected-reward-description');
    if (nameElement) nameElement.textContent = selectedRewardName;
    if (descElement) descElement.textContent = 'Custom configuration based on ' + originalPresetDetails.name;

    // Show success notification
    showNotification('Customized reward applied!', 'success');

    // Clear original details
    originalPresetDetails = null;

    // Optionally display as read-only again
    displayCustomComponents(components);
}

function displayCustomComponents(components) {
    // Find or create component display area (similar to displayPresetComponents)
    let componentDisplay = document.getElementById('preset-component-display');
    if (!componentDisplay) {
        // Create display area if it doesn't exist
        const selectedRewardDisplay = document.getElementById('selected-reward-display');
        if (!selectedRewardDisplay) {
            console.error('selected-reward-display element not found');
            return;
        }

        componentDisplay = document.createElement('div');
        componentDisplay.id = 'preset-component-display';
        componentDisplay.className = 'mt-3';
        selectedRewardDisplay.appendChild(componentDisplay);
    }

    const totalWeight = components.reduce((sum, c) => sum + c.weight, 0);
    const weightPercentages = components.map(c => ({
        ...c,
        weight_percentage: (c.weight / totalWeight) * 100
    }));

    const componentsHtml = weightPercentages.map(comp => {
        const icon = {
            'binary': 'fas fa-toggle-on',
            'continuous': 'fas fa-sliders-h',
            'format': 'fas fa-code',
            'numerical': 'fas fa-calculator',
            'length': 'fas fa-ruler',
            'template': 'fas fa-file-alt',
            'choice': 'fas fa-list-ul',
            'content': 'fas fa-paragraph',
            'pattern': 'fas fa-search'
        }[comp.type.toLowerCase()] || 'fas fa-cog';

        let paramHtml = '';
        if (comp.parameters && Object.keys(comp.parameters).length > 0) {
            paramHtml = '<div class="component-params mt-2 small text-muted">';
            for (const [key, value] of Object.entries(comp.parameters)) {
                if (value !== null && value !== undefined) {
                    let displayValue = value;
                    if (Array.isArray(value)) {
                        displayValue = value.length > 0 ? value.join(', ') : 'None';
                    } else if (typeof value === 'boolean') {
                        displayValue = value ? 'Yes' : 'No';
                    } else if (typeof value === 'number') {
                        // Check for NaN
                        displayValue = isNaN(value) ? 'Not set' : value;
                    } else if (value === '') {
                        displayValue = 'Not set';
                    }
                    paramHtml += `<div><strong>${key.replace(/_/g, ' ')}:</strong> ${displayValue}</div>`;
                }
            }
            paramHtml += '</div>';
        }

        const weightPercentage = comp.weight_percentage.toFixed(1);
        const barWidth = Math.max(5, comp.weight_percentage);

        return `
            <div class="component-item mb-3 p-3 border rounded">
                <div class="d-flex align-items-center justify-content-between mb-2">
                    <div class="d-flex align-items-center">
                        <i class="${icon} me-2 text-primary"></i>
                        <strong>${comp.name}</strong>
                    </div>
                    <span class="badge bg-secondary">${weightPercentage}% (${comp.weight.toFixed(2)})</span>
                </div>
                <div class="progress mb-2" style="height: 8px;">
                    <div class="progress-bar bg-primary" role="progressbar"
                         style="width: ${barWidth}%"
                         aria-valuenow="${comp.weight_percentage}"
                         aria-valuemin="0"
                         aria-valuemax="100">
                    </div>
                </div>
                ${paramHtml}
            </div>
        `;
    }).join('');

    // Create details object for re-editing
    const details = {
        name: 'Custom Reward',
        components: components
    };

    // Check if weights are valid
    const isValid = Math.abs(totalWeight - 1.0) < 0.001;
    const weightStatus = isValid
        ? '<span class="badge bg-success"><i class="fas fa-check"></i> Weight Valid (1.0)</span>'
        : `<span class="badge bg-warning text-dark"><i class="fas fa-exclamation-triangle"></i> Total: ${totalWeight.toFixed(3)}</span>`;

    // Create the full card structure with header and body
    componentDisplay.innerHTML = `
        <div class="card">
            <div class="card-header bg-primary text-white">
                <div class="d-flex justify-content-between align-items-center">
                    <span><i class="fas fa-puzzle-piece"></i> Custom Reward Components</span>
                    <div>
                        ${weightStatus}
                        <button class="btn btn-sm btn-light ms-2" onclick='makePresetEditable(${JSON.stringify(details).replace(/'/g, "&#39;")})'>
                            <i class="fas fa-edit"></i> Customize
                        </button>
                    </div>
                </div>
            </div>
            <div class="card-body">
                <div class="components-list" id="components-list">
                    ${componentsHtml}
                </div>
            </div>
        </div>
    `;
}

async function showPresetDetails(presetName) {
    try {
        const response = await fetch(`/api/rewards/preset-details/${presetName}`);
        if (!response.ok) throw new Error('Failed to fetch preset details');

        const details = await response.json();

        // Generate component breakdown HTML
        const componentsHtml = details.components.map(comp => {
            const weightPercentage = comp.weight_percentage.toFixed(1);
            const barWidth = Math.max(5, comp.weight_percentage);

            const iconMap = {
                'binary': 'fas fa-toggle-on',
                'continuous': 'fas fa-sliders-h',
                'format': 'fas fa-code',
                'numerical': 'fas fa-calculator',
                'length': 'fas fa-ruler',
                'template': 'fas fa-file-alt',
                'choice': 'fas fa-list-ul',
                'content': 'fas fa-paragraph',
                'pattern': 'fas fa-search'
            };

            const icon = iconMap[comp.type.toLowerCase()] || 'fas fa-cog';

            return `
                <div class="component-detail-item mb-3">
                    <div class="d-flex align-items-center justify-content-between mb-2">
                        <div>
                            <i class="${icon} text-primary me-2"></i>
                            <strong>${comp.name}</strong>
                            <span class="badge bg-secondary ms-2">${weightPercentage}%</span>
                        </div>
                    </div>
                    <div class="small text-muted mb-2">${comp.description}</div>
                    <div class="progress" style="height: 10px;">
                        <div class="progress-bar" role="progressbar"
                             style="width: ${barWidth}%"
                             aria-valuenow="${comp.weight_percentage}"
                             aria-valuemin="0" aria-valuemax="100"></div>
                    </div>
                </div>
            `;
        }).join('');

        const difficultyColor = {
            'beginner': 'success',
            'intermediate': 'warning',
            'advanced': 'danger'
        }[details.difficulty] || 'secondary';

        const modalHtml = `
            <div class="modal fade" id="detailsModal" tabindex="-1">
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header bg-primary text-white">
                            <h5 class="modal-title">
                                <i class="fas fa-puzzle-piece"></i> ${details.name} - Component Breakdown
                            </h5>
                            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="mb-3">
                                <span class="badge bg-${difficultyColor}">${details.difficulty}</span>
                                ${details.tags.map(tag => `<span class="badge bg-light text-dark ms-1">${tag}</span>`).join('')}
                            </div>

                            <p class="mb-4">${details.description}</p>

                            <div class="row">
                                <div class="col-lg-7">
                                    <h6 class="text-primary mb-3">
                                        <i class="fas fa-puzzle-piece"></i> Components (${details.components.length})
                                    </h6>
                                    <div class="components-breakdown">
                                        ${componentsHtml}
                                    </div>

                                    <div class="mt-3 text-center">
                                        <strong>Total Weight:</strong>
                                        ${details.weight_valid
                                            ? '<span class="badge bg-success">1.0 ✓</span>'
                                            : `<span class="badge bg-danger">${details.total_weight.toFixed(2)} ✗</span>`}
                                    </div>
                                </div>

                                <div class="col-lg-5">
                                    <h6 class="text-primary mb-3">
                                        <i class="fas fa-lightbulb"></i> Example
                                    </h6>
                                    <div class="example-box">
                                        <div class="mb-3">
                                            <label class="small text-muted fw-bold">Input:</label>
                                            <pre class="bg-light p-2 rounded small">${details.example_input}</pre>
                                        </div>
                                        <div>
                                            <label class="small text-muted fw-bold">Expected Output (High Score):</label>
                                            <pre class="bg-light p-2 rounded small">${details.example_output}</pre>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-primary" onclick="selectPresetByName('${presetName}'); bootstrap.Modal.getInstance(document.getElementById('detailsModal')).hide();">
                                <i class="fas fa-check"></i> Select This Preset
                            </button>
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Remove existing modal if present
        const existingModal = document.getElementById('detailsModal');
        if (existingModal) {
            existingModal.remove();
        }

        // Add modal to page
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('detailsModal'));
        modal.show();

        // Clean up on hide
        document.getElementById('detailsModal').addEventListener('hidden.bs.modal', function() {
            this.remove();
        });

    } catch (error) {
        console.error('Error showing preset details:', error);
        showNotification('Failed to load preset details', 'error');
    }
}

function showPresetExample(presetName) {
    const preset = rewardPresets[presetName];
    if (!preset) return;

    const difficultyColor = {
        'beginner': 'success',
        'intermediate': 'warning',
        'advanced': 'danger'
    }[preset.difficulty] || 'secondary';

    const modalHtml = `
        <div class="modal fade" id="exampleModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-trophy"></i>
                            ${preset.name}
                        </h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body" style="min-height: 400px;">
                        <div class="row">
                            <div class="col-lg-6">
                                <h6 class="text-primary mb-3">Example Input:</h6>
                                <pre class="example-pre">${preset.example_input}</pre>
                            </div>
                            <div class="col-lg-6">
                                <h6 class="text-success mb-3">Expected Output (High Reward):</h6>
                                <pre class="example-pre">${preset.example_output}</pre>
                            </div>
                        </div>
                        <hr class="my-4">
                        <div class="preset-details">
                            <div class="row">
                                <div class="col-md-12">
                                    <p><strong>Description:</strong> ${preset.description}</p>
                                </div>
                                <div class="col-md-6">
                                    <p><strong>Difficulty:</strong>
                                        <span class="badge bg-${difficultyColor}">${preset.difficulty}</span>
                                    </p>
                                </div>
                                <div class="col-md-6">
                                    <p><strong>Category:</strong>
                                        <span class="badge bg-info">${preset.category}</span>
                                    </p>
                                </div>
                                <div class="col-md-12">
                                    <p><strong>Tags:</strong>
                                        ${preset.tags.map(tag =>
                                            `<span class="badge bg-secondary me-1">${tag}</span>`
                                        ).join('')}
                                    </p>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                            <i class="fas fa-times"></i> Cancel
                        </button>
                        <button type="button" class="btn btn-primary"
                                onclick="confirmPresetSelection('${presetName}')">
                            <i class="fas fa-check"></i> Select This Preset
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove any existing modal
    const existingModal = document.getElementById('exampleModal');
    if (existingModal) existingModal.remove();

    // Add new modal to body
    document.body.insertAdjacentHTML('beforeend', modalHtml);

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('exampleModal'));
    modal.show();
}

function populateTemplates() {
    const templatesGrid = document.getElementById('reward-templates-grid');
    if (!templatesGrid || !rewardTemplates) return;

    templatesGrid.innerHTML = '';

    Object.entries(rewardTemplates).forEach(([key, template]) => {
        const iconMap = {
            'Math Problem Solving': 'fa-calculator text-primary',
            'Code Generation': 'fa-code text-success',
            'Question Answering': 'fa-comments text-info'
        };

        const icon = iconMap[template.name] || 'fa-star text-warning';

        templatesGrid.innerHTML += `
            <div class="col-md-4 mb-3">
                <div class="card template-card h-100" onclick="selectTemplate('${key}')">
                    <div class="card-body">
                        <h6 class="card-title">
                            <i class="fas ${icon}"></i> ${template.name}
                        </h6>
                        <p class="card-text small">${template.description}</p>
                        <div class="mt-3">
                            ${template.tips.slice(0, 2).map(tip =>
                                `<small class="d-block text-muted mb-1">• ${tip}</small>`
                            ).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
}

function selectTemplate(templateKey) {
    const template = rewardTemplates[templateKey];
    if (!template) return;

    // Clear previous selection
    document.querySelectorAll('.preset-card, .template-card').forEach(card => {
        card.classList.remove('selected');
    });

    // Mark template card as selected
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('selected');
        currentlySelectedCard = event.currentTarget;
    }

    // Set the reward configuration
    window.selectedRewardConfig = {
        type: 'preset',
        preset_name: template.reward_preset
    };
    selectedRewardName = template.name;
    selectedRewardType = 'template';

    // Update display panel
    const nameElement = document.getElementById('selected-reward-name');
    const descElement = document.getElementById('selected-reward-description');
    if (nameElement) nameElement.textContent = template.name + ' Template';
    if (descElement) descElement.textContent = template.description;

    // Animate selection
    if (currentlySelectedCard) {
        currentlySelectedCard.classList.add('selecting');
        setTimeout(() => {
            if (currentlySelectedCard) {
                currentlySelectedCard.classList.remove('selecting');
            }
        }, 500);
    }

    // Apply recommended settings if user confirms
    if (confirm('Apply recommended settings for this template?')) {
        Object.entries(template.recommended_settings).forEach(([key, value]) => {
            const elementId = key.replace(/_/g, '-');
            const element = document.getElementById(elementId);
            if (element) element.value = value;
        });
    }

    // Show tips
    showTemplateTips(template);

    // Store selection
    localStorage.setItem('selectedReward', JSON.stringify(selectedRewardConfig));
}

function showTemplateTips(template) {
    const tipsHtml = template.tips.map(tip => `<li>${tip}</li>`).join('');
    showNotification(
        `<strong>${template.name} Tips:</strong><ul class="mb-0 mt-2">${tipsHtml}</ul>`,
        'info',
        10000  // Show for 10 seconds
    );
}

// Advanced component builder
function addAdvancedComponent() {
    const componentsDiv = document.getElementById('reward-components-advanced');
    if (!componentsDiv) return;

    const componentId = `adv-component-${Date.now()}`;
    const componentHtml = `
        <div class="card mb-3" id="${componentId}">
            <div class="card-body">
                <div class="row">
                    <div class="col-md-3">
                        <label class="form-label">Component Type</label>
                        <select class="form-select" onchange="updateAdvancedComponentType('${componentId}', this.value)">
                            <option value="binary">Binary Match</option>
                            <option value="numerical">Numerical</option>
                            <option value="length">Length</option>
                            <option value="format">Format</option>
                            <option value="template">Template Validation</option>
                            <option value="multi_choice">Multi-Choice</option>
                            <option value="section_content">Section Content</option>
                            <option value="sequential">Sequential Pattern</option>
                        </select>
                    </div>
                    <div class="col-md-6" id="${componentId}-params">
                        <label class="form-label">Pattern/Parameters</label>
                        <input type="text" class="form-control" placeholder="Enter regex pattern">
                    </div>
                    <div class="col-md-2">
                        <label class="form-label">Weight</label>
                        <input type="number" class="form-control" value="1.0" step="0.1" min="0">
                    </div>
                    <div class="col-md-1">
                        <label class="form-label">&nbsp;</label>
                        <button class="btn btn-danger btn-sm form-control" onclick="removeComponent('${componentId}')">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                </div>
                <div class="row mt-2">
                    <div class="col-12" id="${componentId}-help">
                        <small class="text-muted">Select a component type to see configuration options</small>
                    </div>
                </div>
            </div>
        </div>
    `;
    componentsDiv.insertAdjacentHTML('beforeend', componentHtml);
}

function updateAdvancedComponentType(componentId, type) {
    const paramsDiv = document.getElementById(`${componentId}-params`);
    const helpDiv = document.getElementById(`${componentId}-help`);
    if (!paramsDiv) return;

    let paramsHtml = '';
    let helpText = '';

    switch (type) {
        case 'binary':
            paramsHtml = `
                <label class="form-label">Regex Pattern</label>
                <input type="text" class="form-control" placeholder="e.g., \\\\boxed\\\\{.*\\\\}">
            `;
            helpText = 'Binary reward: Returns 1 if pattern matches, 0 otherwise';
            break;

        case 'format':
            paramsHtml = `
                <label class="form-label">Regex Pattern</label>
                <input type="text" class="form-control" placeholder="e.g., \\\\boxed\\\\{.*\\\\}">
            `;
            helpText = 'Format reward: Validates specific formatting patterns';
            break;

        case 'numerical':
            paramsHtml = `
                <label class="form-label">Tolerance</label>
                <input type="number" class="form-control" value="0.000001" step="0.000001">
            `;
            helpText = 'Numerical reward: Compares numeric values within tolerance';
            break;

        case 'length':
            paramsHtml = `
                <div class="row">
                    <div class="col-4">
                        <label class="form-label">Min</label>
                        <input type="number" class="form-control" placeholder="10">
                    </div>
                    <div class="col-4">
                        <label class="form-label">Max</label>
                        <input type="number" class="form-control" placeholder="200">
                    </div>
                    <div class="col-4">
                        <label class="form-label">Optimal</label>
                        <input type="number" class="form-control" placeholder="50">
                    </div>
                </div>
            `;
            helpText = 'Length reward: Scores based on response length (in words)';
            break;

        case 'template':
            paramsHtml = `
                <div class="mb-2">
                    <label class="form-label">All Sections to Check (comma-separated)</label>
                    <input type="text" class="form-control" placeholder="e.g., analysis,signal,confidence" data-param="section_tags">
                    <small class="form-text text-muted">All possible XML tags to validate (e.g., &lt;analysis&gt;, &lt;signal&gt;)</small>
                </div>
                <div class="mb-2">
                    <label class="form-label">Mandatory Sections (comma-separated)</label>
                    <input type="text" class="form-control" placeholder="e.g., signal" data-param="required_sections">
                    <small class="form-text text-muted">Subset of sections above that must be present (leave empty if all are optional)</small>
                </div>
                <div class="form-check">
                    <input type="checkbox" class="form-check-input" id="${componentId}-order" data-param="order_matters">
                    <label class="form-check-label" for="${componentId}-order">Order matters</label>
                </div>
            `;
            helpText = 'Template validation: Ensures output follows a specific template structure with required sections';
            break;

        case 'multi_choice':
            paramsHtml = `
                <div class="mb-2">
                    <label class="form-label">Valid Choices (comma-separated)</label>
                    <input type="text" class="form-control" placeholder="e.g., STRONG_BUY,WEAK_BUY,HOLD,WEAK_SELL,STRONG_SELL" data-param="valid_choices">
                </div>
                <div class="form-check">
                    <input type="checkbox" class="form-check-input" id="${componentId}-case" checked data-param="case_sensitive">
                    <label class="form-check-label" for="${componentId}-case">Case sensitive</label>
                </div>
                <div class="form-check">
                    <input type="checkbox" class="form-check-input" id="${componentId}-exact" checked data-param="exact_match">
                    <label class="form-check-label" for="${componentId}-exact">Exact match (word boundaries)</label>
                </div>
            `;
            helpText = 'Multi-choice validation: Ensures output contains exactly one of the valid choices';
            break;

        case 'section_content':
            paramsHtml = `
                <div class="mb-2">
                    <label class="form-label">Section Tag</label>
                    <input type="text" class="form-control" placeholder="e.g., analysis" data-param="section_tag">
                </div>
                <div class="row mb-2">
                    <div class="col-6">
                        <label class="form-label">Min Words</label>
                        <input type="number" class="form-control" placeholder="20" data-param="min_words">
                    </div>
                    <div class="col-6">
                        <label class="form-label">Max Words</label>
                        <input type="number" class="form-control" placeholder="200" data-param="max_words">
                    </div>
                </div>
                <div>
                    <label class="form-label">Required Keywords (comma-separated)</label>
                    <input type="text" class="form-control" placeholder="e.g., RSI,MACD,trend" data-param="required_keywords">
                </div>
            `;
            helpText = 'Section content: Validates content within a specific section tag';
            break;

        case 'sequential':
            paramsHtml = `
                <div class="mb-2">
                    <label class="form-label">Patterns (one per line, regex supported)</label>
                    <textarea class="form-control" rows="3" placeholder="First pattern\nSecond pattern\nThird pattern" data-param="patterns"></textarea>
                </div>
                <div class="form-check">
                    <input type="checkbox" class="form-check-input" id="${componentId}-strict" checked data-param="strict_order">
                    <label class="form-check-label" for="${componentId}-strict">Strict order required</label>
                </div>
            `;
            helpText = 'Sequential pattern: Validates that patterns appear in a specific order';
            break;
    }

    paramsDiv.innerHTML = paramsHtml;
    if (helpDiv) {
        helpDiv.innerHTML = `<small class="text-muted">${helpText}</small>`;
    }
}

function removeComponent(componentId) {
    const component = document.getElementById(componentId);
    if (component) component.remove();
}

// Test reward functionality
async function testReward() {
    const instruction = document.getElementById('test-instruction').value;
    const generated = document.getElementById('test-generated').value;
    const reference = document.getElementById('test-reference').value;

    if (!instruction || !generated) {
        showNotification('Please provide instruction and generated response', 'warning');
        return;
    }

    // Get current reward config
    const rewardConfig = gatherRewardConfig();

    const testData = {
        reward_config: rewardConfig,
        test_cases: [{
            instruction: instruction,
            generated: generated,
            reference: reference || null
        }]
    };

    try {
        const response = await fetch('/api/rewards/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(testData)
        });

        const result = await response.json();

        if (result.error) {
            showNotification(`Error: ${result.error}`, 'error');
            return;
        }

        // Display results
        displayTestResults(result);
    } catch (error) {
        console.error('Test error:', error);
        showNotification('Failed to test reward', 'error');
    }
}

function displayTestResults(result) {
    const resultsDiv = document.getElementById('test-results');
    const vizDiv = document.getElementById('reward-visualization');

    if (!resultsDiv) return;

    // Handle both formats: direct result or nested in results array
    let testResult;
    if (result.results && result.results.length > 0) {
        // Format from /api/rewards/test (has results array)
        testResult = result.results[0];
    } else if (result.total_reward !== undefined) {
        // Direct format from /api/rewards/test-with-model
        testResult = result;
    } else {
        resultsDiv.innerHTML = '<p class="text-danger">Invalid result format</p>';
        console.error('Invalid result format:', result);
        return;
    }

    const totalReward = testResult.total_reward.toFixed(3);
    const components = testResult.components;

    // Build results HTML
    let html = `
        <h5>Total Reward: <span class="badge bg-primary fs-6">${totalReward}</span></h5>
        <hr>
        <h6>Component Breakdown:</h6>
        <div class="component-scores">
    `;

    // Add component scores
    Object.entries(components).forEach(([name, value]) => {
        const percentage = (value * 100).toFixed(1);
        const barClass = value > 0.7 ? 'bg-success' : value > 0.4 ? 'bg-warning' : 'bg-danger';
        html += `
            <div class="mb-2">
                <div class="d-flex justify-content-between">
                    <small>${name}</small>
                    <small>${value.toFixed(3)}</small>
                </div>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar ${barClass}" style="width: ${percentage}%">${percentage}%</div>
                </div>
            </div>
        `;
    });

    html += '</div>';
    resultsDiv.innerHTML = html;

    // Create visualization if Chart.js is available
    if (vizDiv && typeof Chart !== 'undefined') {
        createRewardChart(vizDiv, components);
    }
}

function createRewardChart(container, components) {
    // Clear existing chart
    container.innerHTML = '<canvas id="rewardChart"></canvas>';
    const ctx = document.getElementById('rewardChart').getContext('2d');

    new Chart(ctx, {
        type: 'radar',
        data: {
            labels: Object.keys(components),
            datasets: [{
                label: 'Component Scores',
                data: Object.values(components),
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgb(54, 162, 235)',
                pointBackgroundColor: 'rgb(54, 162, 235)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(54, 162, 235)'
            }]
        },
        options: {
            elements: {
                line: {
                    borderWidth: 3
                }
            },
            scales: {
                r: {
                    angleLines: {
                        display: false
                    },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            }
        }
    });
}

function showRewardHelp() {
    const helpModal = `
        <div class="modal fade" id="rewardHelpModal" tabindex="-1">
            <div class="modal-dialog modal-lg">
                <div class="modal-content">
                    <div class="modal-header">
                        <h5 class="modal-title">Understanding Reward Functions</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <h6>What are Reward Functions?</h6>
                        <p>Reward functions evaluate how well a model's output matches your expectations. They assign scores (0-1) to generated responses, guiding the training process.</p>

                        <h6>Choosing the Right Reward:</h6>
                        <ul>
                            <li><strong>Quick Start:</strong> Use templates for common tasks with pre-configured settings</li>
                            <li><strong>Preset Library:</strong> Browse categorized presets for specific use cases</li>
                            <li><strong>Custom Builder:</strong> Combine components for unique requirements</li>
                        </ul>

                        <h6>Testing Your Reward:</h6>
                        <p>Always test your reward function before training to ensure it scores outputs correctly. Use the Test tab to verify behavior with sample inputs.</p>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    // Remove existing modal if present
    const existing = document.getElementById('rewardHelpModal');
    if (existing) existing.remove();

    document.body.insertAdjacentHTML('beforeend', helpModal);
    const modal = new bootstrap.Modal(document.getElementById('rewardHelpModal'));
    modal.show();
}

function gatherCustomRewardConfig() {
    // Gather components from the advanced builder UI
    const components = [];
    const componentCards = document.querySelectorAll('#reward-components-advanced .card');

    componentCards.forEach(card => {
        const typeSelect = card.querySelector('select');
        const weightInput = card.querySelector('input[type="number"][step="0.1"]');
        const paramsDiv = card.querySelector('[id$="-params"]');

        if (!typeSelect) return;

        const component = {
            type: typeSelect.value,
            weight: parseFloat(weightInput?.value) || 1.0
        };

        // Gather parameters based on component type
        const inputs = paramsDiv?.querySelectorAll('input, textarea');
        if (inputs) {
            inputs.forEach(input => {
                const param = input.dataset.param;
                if (param) {
                    if (input.type === 'checkbox') {
                        component[param] = input.checked;
                    } else if (input.type === 'number') {
                        component[param] = parseFloat(input.value) || 0;
                    } else {
                        component[param] = input.value;
                    }
                }
            });
        }

        components.push(component);
    });

    return {
        type: 'custom',
        components: components
    };
}

function updateCustomRewardConfig() {
    // Update the global config when using custom builder
    window.selectedRewardConfig = gatherCustomRewardConfig();
    selectedRewardName = 'Custom Configuration';
    selectedRewardType = 'custom';
}

function saveCustomReward() {
    const name = prompt('Enter a name for this reward configuration:');
    if (!name) return;

    const config = gatherCustomRewardConfig();

    // Update global config
    window.selectedRewardConfig = config;

    fetch('/api/rewards/save', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, config: config })
    })
    .then(response => response.json())
    .then(result => {
        if (result.error) {
            showNotification(`Error: ${result.error}`, 'error');
        } else {
            showNotification(`Reward configuration saved as ${name}`, 'success');
        }
    })
    .catch(error => {
        console.error('Save error:', error);
        showNotification('Failed to save reward configuration', 'error');
    });
}

// Add confirmation function for preset selection from modal
function confirmPresetSelection(presetName) {
    // Close the modal first
    const modal = bootstrap.Modal.getInstance(document.getElementById('exampleModal'));
    if (modal) modal.hide();

    // Select the preset
    selectPresetByName(presetName);

    // Find and highlight the card in the grid
    const cards = document.querySelectorAll('.preset-card');
    cards.forEach(card => {
        if (card.querySelector('.card-title')?.textContent.includes(rewardPresets[presetName].name)) {
            card.classList.add('selected');
            currentlySelectedCard = card;
            // Scroll to the selected card
            card.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    });
}

// Function to test the selected reward
function testSelectedReward() {
    // Switch to test tab
    const testTab = document.querySelector('[data-bs-target="#reward-test-tab"]');
    if (testTab) {
        const tab = new bootstrap.Tab(testTab);
        tab.show();
    }
}

// Load saved selection on page load
function loadSavedSelection() {
    const saved = localStorage.getItem('selectedReward');
    // Check if we're loading an existing configuration (indicated by saved config ID or active session)
    const savedConfigId = localStorage.getItem('selectedConfigId');
    const hasActiveSession = document.getElementById('sessions-list')?.children?.length > 0;

    // Only restore saved selection if we're resuming work on an existing config
    // For new configs, let the default quick-start selection take effect
    if (saved && (savedConfigId || isRestoringSession)) {
        try {
            const config = JSON.parse(saved);
            if (config.preset_name) {
                // Wait for presets to load, then select (silently restore without notification)
                setTimeout(() => {
                    selectPresetByName(config.preset_name, true);  // Silent restore
                }, 500);
            }
        } catch (e) {
            console.error('Failed to load saved reward selection:', e);
        }
    } else {
        // For new configurations, select the first quick-start template by default
        setTimeout(() => {
            // Try to select math template, fallback to first available preset
            if (rewardPresets['math']) {
                selectPresetByName('math', true);
            } else {
                const firstPreset = Object.keys(rewardPresets)[0];
                if (firstPreset) {
                    selectPresetByName(firstPreset, true);
                }
            }
        }, 500);
    }
}

// ============================================================================
// Legacy Reward Functions (for backward compatibility)
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

// Setup tab change listeners
function setupTabListeners() {
    // Listen for tab changes to update config accordingly
    const customBuilderTab = document.querySelector('[data-bs-target="#reward-custom-tab"]');
    if (customBuilderTab) {
        customBuilderTab.addEventListener('shown.bs.tab', () => {
            updateCustomRewardConfig();
        });
    }

    // Also listen for changes to components
    const observer = new MutationObserver(() => {
        const activeTab = document.querySelector('#reward-custom-tab.active');
        if (activeTab) {
            updateCustomRewardConfig();
        }
    });

    const componentsDiv = document.getElementById('reward-components-advanced');
    if (componentsDiv) {
        observer.observe(componentsDiv, { childList: true, subtree: true });
    }
}

// ============================================================================
// Main Configuration Gathering Function
// ============================================================================

function gatherRewardConfig() {
    // Check if using the new enhanced reward system
    if (window.selectedRewardConfig) {
        return window.selectedRewardConfig;
    }

    // Fallback to old system for backward compatibility
    const rewardPresetElement = document.getElementById('reward-preset');
    if (!rewardPresetElement) {
        // If old elements don't exist, return default config
        return {
            type: 'preset',
            preset: 'math'
        };
    }

    const isPreset = rewardPresetElement.classList.contains('active');

    if (isPreset) {
        const presetSelectElement = document.getElementById('reward-preset-select');
        const presetValue = presetSelectElement ? presetSelectElement.value : 'math';
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

            if (!typeSelect) return;

            const component = {
                type: typeSelect.value,
                weight: parseFloat(weightInput?.value) || 1.0
            };

            // Get type-specific parameters
            if (typeSelect.value === 'binary' || typeSelect.value === 'format') {
                const patternInput = paramsDiv?.querySelector('input[type="text"]');
                if (patternInput && patternInput.value) {
                    component.pattern = patternInput.value;
                }
            } else if (typeSelect.value === 'numerical') {
                const toleranceInput = paramsDiv?.querySelector('input[type="number"]');
                if (toleranceInput) {
                    component.tolerance = parseFloat(toleranceInput.value) || 0.000001;
                }
            } else if (typeSelect.value === 'length') {
                const inputs = paramsDiv?.querySelectorAll('input[type="number"]');
                if (inputs && inputs.length >= 2) {
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

// ============================================================================
// Model-Based Testing Functions
// ============================================================================

// Store original generated text for restoration
let originalGeneratedText = '';
let currentTestResult = null;
let scoreUpdateDebounceTimer = null;

// Toggle between manual and model-based testing
function toggleTestMode(useModel) {
    const generationPanel = document.getElementById('generation-settings-panel');
    const testBtn = document.getElementById('test-reward-btn');
    const generateBtn = document.getElementById('generate-and-test-btn');
    const editBadge = document.getElementById('edit-mode-badge');

    if (useModel) {
        if (generationPanel) generationPanel.style.display = 'block';
        if (testBtn) testBtn.style.display = 'none';
        if (generateBtn) generateBtn.style.display = 'block';
        if (editBadge) editBadge.style.display = 'inline-block';

        // Verify base model is selected
        const baseModel = getSelectedBaseModel();
        if (!baseModel || baseModel === 'unsloth/Qwen2.5-1.5B-Instruct') {
            showNotification('Using default base model. Select a model in Step 1 to use a different one.', 'info');
        }
    } else {
        if (generationPanel) generationPanel.style.display = 'none';
        if (testBtn) testBtn.style.display = 'block';
        if (generateBtn) generateBtn.style.display = 'none';
        if (editBadge) editBadge.style.display = 'none';
    }
}

// Get the selected base model from Step 1
function getSelectedBaseModel() {
    // Try multiple possible element IDs where base model might be stored
    const modelName = document.getElementById('model-name')?.value ||
                     document.getElementById('base-model')?.value ||
                     document.getElementById('selected-model')?.value;

    if (!modelName) {
        console.warn('No base model found, using default');
        return 'unsloth/Qwen2.5-1.5B-Instruct'; // Default fallback
    }

    return modelName;
}

// Load available trained models (kept for backward compatibility but not used in test panel)
async function loadAvailableModels() {
    try {
        const response = await fetch('/api/sessions/list');
        const data = await response.json();

        const modelSelect = document.getElementById('test-model-select');
        if (!modelSelect) return; // Element might have been removed

        modelSelect.innerHTML = '<option value="">-- Select a trained model --</option>';

        if (data.sessions && data.sessions.length > 0) {
            data.sessions.forEach(session => {
                if (session.status === 'completed') {
                    const option = document.createElement('option');
                    option.value = session.session_id;
                    option.textContent = `${session.session_id} (${session.model_name})`;
                    modelSelect.appendChild(option);
                }
            });
        }
    } catch (error) {
        console.error('Failed to load models:', error);
        showNotification('Failed to load available models', 'error');
    }
}

// Test reward with model generation
async function testRewardWithModel() {
    const instruction = document.getElementById('test-instruction').value;
    const reference = document.getElementById('test-reference').value;

    if (!instruction) {
        showNotification('Please provide an instruction', 'warning');
        return;
    }

    // Get the base model from Step 1
    const baseModel = getSelectedBaseModel();

    if (!baseModel) {
        showNotification('Please select a base model in Step 1', 'warning');
        return;
    }

    // Get generation config
    const generationConfig = {
        temperature: parseFloat(document.getElementById('gen-temperature').value),
        max_new_tokens: parseInt(document.getElementById('gen-max-tokens').value),
        top_p: parseFloat(document.getElementById('gen-top-p').value),
        top_k: parseInt(document.getElementById('gen-top-k').value)
    };

    // Get current reward config
    const rewardConfig = gatherRewardConfig();

    // Get system prompt from dataset section (if configured)
    const systemPrompt = document.getElementById('custom-system-prompt')?.value ||
                        document.getElementById('system-prompt')?.value || '';

    const requestData = {
        reward_config: rewardConfig,
        instruction: instruction,
        reference: reference || null,
        model_type: 'base',  // Use base model instead of trained
        model_key: baseModel,  // Base model name from Step 1
        generation_config: generationConfig,
        system_prompt: systemPrompt  // Include configured system prompt
    };

    try {
        // Show loading state
        const generateBtn = document.getElementById('generate-and-test-btn');
        const originalText = generateBtn.innerHTML;
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

        const response = await fetch('/api/rewards/test-with-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });

        const result = await response.json();

        if (result.error) {
            showNotification(`Error: ${result.error}`, 'error');
            generateBtn.disabled = false;
            generateBtn.innerHTML = originalText;
            return;
        }

        // Store original generated text
        originalGeneratedText = result.generated;

        // Populate the generated textarea
        document.getElementById('test-generated').value = result.generated;

        // Display results
        currentTestResult = result;
        displayTestResults(result);

        // Show restore button
        document.getElementById('generated-actions').style.display = 'block';
        document.getElementById('edit-status').textContent = 'Original model output';

        showNotification('Model generated response and scored successfully', 'success');

        // Restore button
        generateBtn.disabled = false;
        generateBtn.innerHTML = originalText;

    } catch (error) {
        console.error('Test error:', error);
        showNotification('Failed to test with model', 'error');
        const generateBtn = document.getElementById('generate-and-test-btn');
        generateBtn.disabled = false;
        generateBtn.innerHTML = '<i class="fas fa-robot"></i> Generate & Score';
    }
}

// Test reward with or without model (based on mode)
function testRewardWithOrWithoutModel() {
    const useModel = document.getElementById('use-model-generation').checked;
    if (useModel) {
        testRewardWithModel();
    } else {
        testReward();
    }
}

// Handle edits to generated text
function onGeneratedTextEdit() {
    const currentText = document.getElementById('test-generated').value;

    // Only proceed if we have an original text to compare against
    if (!originalGeneratedText) {
        return;
    }

    // Update edit status
    const editStatus = document.getElementById('edit-status');
    if (currentText === originalGeneratedText) {
        editStatus.textContent = 'Original model output';
        editStatus.className = 'text-muted ms-2';
    } else {
        editStatus.textContent = 'Modified (scores updating...)';
        editStatus.className = 'text-warning ms-2';
    }

    // Debounce the score update
    clearTimeout(scoreUpdateDebounceTimer);
    scoreUpdateDebounceTimer = setTimeout(() => {
        updateScoreForEditedText(currentText);
    }, 500); // Wait 500ms after user stops typing
}

// Update score for edited text
async function updateScoreForEditedText(editedText) {
    const instruction = document.getElementById('test-instruction').value;
    const reference = document.getElementById('test-reference').value;
    const rewardConfig = gatherRewardConfig();

    const testData = {
        reward_config: rewardConfig,
        test_cases: [{
            instruction: instruction,
            generated: editedText,
            reference: reference || null
        }]
    };

    try {
        const response = await fetch('/api/rewards/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(testData)
        });

        const result = await response.json();

        if (result.error) {
            console.error('Score update error:', result.error);
            return;
        }

        // Update the display with new scores
        const testResult = result.results[0];
        displayTestResults(testResult);

        // Update edit status
        const editStatus = document.getElementById('edit-status');
        if (editedText === originalGeneratedText) {
            editStatus.textContent = 'Original model output';
            editStatus.className = 'text-muted ms-2';
        } else {
            // Compare scores
            const scoreDiff = testResult.total_reward - (currentTestResult?.total_reward || 0);
            if (scoreDiff > 0) {
                editStatus.textContent = `Improved score: +${scoreDiff.toFixed(3)}`;
                editStatus.className = 'text-success ms-2';
            } else if (scoreDiff < 0) {
                editStatus.textContent = `Lower score: ${scoreDiff.toFixed(3)}`;
                editStatus.className = 'text-danger ms-2';
            } else {
                editStatus.textContent = 'Same score';
                editStatus.className = 'text-muted ms-2';
            }
        }

    } catch (error) {
        console.error('Failed to update score:', error);
    }
}

// Restore original generated text
function restoreOriginalGenerated() {
    if (originalGeneratedText) {
        document.getElementById('test-generated').value = originalGeneratedText;
        onGeneratedTextEdit(); // Trigger score update
        showNotification('Restored original model output', 'info');
    }
}

// ============================================================================
// Export Functions Globally for App.js Integration
// ============================================================================

// Restore custom reward configuration from saved config
function restoreCustomReward(rewardConfig) {
    console.log('Restoring custom reward:', rewardConfig);

    if (!rewardConfig || !rewardConfig.components || rewardConfig.components.length === 0) {
        console.warn('No custom reward components to restore');
        return;
    }

    // Switch to custom tab
    const customTab = document.querySelector('[data-bs-target="#reward-custom-tab"]');
    if (customTab) {
        const tab = new bootstrap.Tab(customTab);
        tab.show();
    }

    // Clear existing components
    const componentsDiv = document.getElementById('reward-components-advanced');
    if (!componentsDiv) {
        console.error('reward-components-advanced container not found');
        return;
    }
    componentsDiv.innerHTML = '';

    // Rebuild each component
    rewardConfig.components.forEach((comp, index) => {
        const componentId = `adv-component-restored-${index}`;

        // Create component card
        const componentHtml = `
            <div class="card mb-3" id="${componentId}">
                <div class="card-body">
                    <div class="row">
                        <div class="col-md-3">
                            <label class="form-label">Component Type</label>
                            <select class="form-select" onchange="updateAdvancedComponentType('${componentId}', this.value)">
                                <option value="binary" ${comp.type === 'binary' ? 'selected' : ''}>Binary Match</option>
                                <option value="numerical" ${comp.type === 'numerical' ? 'selected' : ''}>Numerical</option>
                                <option value="length" ${comp.type === 'length' ? 'selected' : ''}>Length</option>
                                <option value="format" ${comp.type === 'format' ? 'selected' : ''}>Format</option>
                                <option value="template" ${comp.type === 'template' ? 'selected' : ''}>Template Validation</option>
                                <option value="multi_choice" ${comp.type === 'multi_choice' ? 'selected' : ''}>Multi-Choice</option>
                                <option value="section_content" ${comp.type === 'section_content' ? 'selected' : ''}>Section Content</option>
                                <option value="sequential" ${comp.type === 'sequential' ? 'selected' : ''}>Sequential Pattern</option>
                            </select>
                        </div>
                        <div class="col-md-6" id="${componentId}-params">
                            <!-- Parameters will be populated by updateAdvancedComponentType -->
                        </div>
                        <div class="col-md-2">
                            <label class="form-label">Weight</label>
                            <input type="number" class="form-control" value="${comp.weight || 1.0}" step="0.1" min="0">
                        </div>
                        <div class="col-md-1">
                            <label class="form-label">&nbsp;</label>
                            <button class="btn btn-danger btn-sm form-control" onclick="removeComponent('${componentId}')">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-12" id="${componentId}-help">
                            <small class="text-muted">Component restored from configuration</small>
                        </div>
                    </div>
                </div>
            </div>
        `;

        componentsDiv.insertAdjacentHTML('beforeend', componentHtml);

        // Trigger updateAdvancedComponentType to create proper parameter inputs
        updateAdvancedComponentType(componentId, comp.type);

        // Now populate the parameters with saved values
        setTimeout(() => {
            const paramsDiv = document.getElementById(`${componentId}-params`);
            if (paramsDiv) {
                const inputs = paramsDiv.querySelectorAll('input, textarea');
                inputs.forEach(input => {
                    const param = input.dataset.param;
                    // Parameters are stored directly on the component object, not in a nested 'parameters' object
                    if (param && comp[param] !== undefined) {
                        if (input.type === 'checkbox') {
                            input.checked = comp[param];
                        } else {
                            input.value = comp[param];
                        }
                    }
                });
            }
        }, 50);
    });

    // Update the global config
    window.selectedRewardConfig = rewardConfig;
    selectedRewardName = 'Custom Configuration';
    selectedRewardType = 'custom';

    console.log('Custom reward restored successfully');
}

// Show custom reward as selected (summary view, not full builder)
function showCustomRewardAsSelected(rewardConfig) {
    console.log('Showing custom reward as selected:', rewardConfig);

    if (!rewardConfig || !rewardConfig.components || rewardConfig.components.length === 0) {
        console.warn('No custom reward components to display');
        return;
    }

    // Update the global config
    window.selectedRewardConfig = rewardConfig;
    selectedRewardName = 'Custom Configuration';
    selectedRewardType = 'custom';

    // Sync with AppState
    if (window.AppState && AppState.setConfigValue) {
        AppState.setConfigValue('rewardConfig', rewardConfig);
    }

    // Clear any preset selections
    document.querySelectorAll('.preset-card, .template-card').forEach(card => {
        card.classList.remove('selected');
    });

    // Update the reward display panel to show summary
    const nameElement = document.getElementById('selected-reward-name');
    const descElement = document.getElementById('selected-reward-description');

    if (nameElement) {
        nameElement.textContent = 'Custom Reward Configuration';
    }

    if (descElement) {
        // Create component summary
        const componentsSummary = rewardConfig.components.map((comp, i) => {
            const typeName = comp.type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase());
            return `${i + 1}. ${comp.name || typeName} (weight: ${comp.weight || 1.0})`;
        }).join('<br>');

        descElement.innerHTML = `
            <p class="mb-2"><strong>Components:</strong></p>
            <div class="small text-muted">${componentsSummary}</div>
            <div class="mt-3">
                <button class="btn btn-sm btn-primary" onclick="restoreCustomReward(window.selectedRewardConfig)">
                    <i class="fas fa-edit"></i> Edit Components
                </button>
                <button class="btn btn-sm btn-outline-secondary ms-2" onclick="testSelectedReward()">
                    <i class="fas fa-flask"></i> Test Reward
                </button>
            </div>
        `;
    }

    // Display detailed component view with all parameters
    displayCustomComponents(rewardConfig.components);

    console.log('Custom reward displayed as selected (summary mode)');
}

// ============================================================================
// Custom Reward Builder Functions
// ============================================================================

let customComponentCounter = 0;

function openCustomRewardBuilder() {
    // Reset counter and clear components
    customComponentCounter = 0;
    const componentsDiv = document.getElementById('custom-reward-components');
    if (componentsDiv) {
        componentsDiv.innerHTML = '';
    }

    // Show modal
    const modal = new bootstrap.Modal(document.getElementById('customRewardBuilderModal'));
    modal.show();

    // Add one default component
    addCustomComponent();
}

function addCustomComponent() {
    const componentsDiv = document.getElementById('custom-reward-components');
    if (!componentsDiv) return;

    const componentId = `custom-comp-${customComponentCounter++}`;

    const componentHtml = `
        <div class="card mb-3 custom-component" id="${componentId}">
            <div class="card-header d-flex justify-content-between align-items-center">
                <span>Component ${customComponentCounter}</span>
                <button class="btn btn-sm btn-danger" onclick="removeCustomComponent('${componentId}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6 mb-3">
                        <label class="form-label">Component Name</label>
                        <input type="text" class="form-control component-name" placeholder="e.g., accuracy_check">
                    </div>
                    <div class="col-md-3 mb-3">
                        <label class="form-label">Type</label>
                        <select class="form-select component-type" onchange="updateCustomComponentFields('${componentId}', this.value)">
                            <option value="binary">Binary (Pass/Fail)</option>
                            <option value="continuous">Continuous (0-1 Score)</option>
                            <option value="numerical">Numerical Comparison</option>
                            <option value="format">Format/Pattern Match</option>
                            <option value="length">Length Validation</option>
                            <option value="template">Template Structure</option>
                            <option value="multi_choice">Multi-Choice</option>
                            <option value="section_content">Section Content</option>
                            <option value="sequential">Sequential Pattern</option>
                        </select>
                    </div>
                    <div class="col-md-3 mb-3">
                        <label class="form-label">Weight</label>
                        <input type="number" class="form-control component-weight" value="1.0" step="0.1" min="0" onchange="updateCustomWeightTotal()">
                    </div>
                </div>
                <div class="component-parameters" id="${componentId}-params">
                    <!-- Parameters will be added dynamically based on type -->
                </div>
            </div>
        </div>
    `;

    componentsDiv.insertAdjacentHTML('beforeend', componentHtml);
    updateCustomComponentFields(componentId, 'binary');
    updateCustomWeightTotal();
}

function removeCustomComponent(componentId) {
    const component = document.getElementById(componentId);
    if (component) {
        component.remove();
        updateCustomWeightTotal();
    }
}

function updateCustomComponentFields(componentId, type) {
    const paramsDiv = document.getElementById(`${componentId}-params`);
    if (!paramsDiv) return;

    let fieldsHtml = '';

    switch(type) {
        case 'binary':
            fieldsHtml = `
                <div class="mb-2">
                    <label class="form-label small">Regex Pattern (optional)</label>
                    <input type="text" class="form-control form-control-sm" data-param="regex_pattern" placeholder="e.g., ^[A-Z]+$">
                </div>
            `;
            break;
        case 'numerical':
            fieldsHtml = `
                <div class="row">
                    <div class="col-md-6 mb-2">
                        <label class="form-label small">Tolerance</label>
                        <input type="number" class="form-control form-control-sm" data-param="tolerance" value="0.000001" step="0.000001">
                    </div>
                    <div class="col-md-6 mb-2">
                        <label class="form-label small">Relative Tolerance</label>
                        <select class="form-select form-select-sm" data-param="relative">
                            <option value="false">No</option>
                            <option value="true">Yes</option>
                        </select>
                    </div>
                </div>
            `;
            break;
        case 'format':
            fieldsHtml = `
                <div class="mb-2">
                    <label class="form-label small">Regex Pattern</label>
                    <input type="text" class="form-control form-control-sm" data-param="pattern" placeholder="e.g., \\\\boxed\\{.*?\\}">
                </div>
            `;
            break;
        case 'length':
            fieldsHtml = `
                <div class="row">
                    <div class="col-md-4 mb-2">
                        <label class="form-label small">Min Length</label>
                        <input type="number" class="form-control form-control-sm" data-param="min_length" placeholder="Optional">
                    </div>
                    <div class="col-md-4 mb-2">
                        <label class="form-label small">Max Length</label>
                        <input type="number" class="form-control form-control-sm" data-param="max_length" placeholder="Optional">
                    </div>
                    <div class="col-md-4 mb-2">
                        <label class="form-label small">Optimal Length</label>
                        <input type="number" class="form-control form-control-sm" data-param="optimal_length" placeholder="Optional">
                    </div>
                </div>
            `;
            break;
        case 'template':
            fieldsHtml = `
                <div class="mb-2">
                    <label class="form-label small">Section Tags (comma-separated)</label>
                    <input type="text" class="form-control form-control-sm" data-param="section_tags" placeholder="e.g., section1,section2">
                </div>
                <div class="mb-2">
                    <label class="form-label small">Required Sections (comma-separated)</label>
                    <input type="text" class="form-control form-control-sm" data-param="required_sections" placeholder="e.g., section1">
                </div>
                <div class="mb-2">
                    <label class="form-label small">Order Matters</label>
                    <select class="form-select form-select-sm" data-param="order_matters">
                        <option value="false">No</option>
                        <option value="true">Yes</option>
                    </select>
                </div>
            `;
            break;
        case 'multi_choice':
            fieldsHtml = `
                <div class="mb-2">
                    <label class="form-label small">Valid Choices (comma-separated)</label>
                    <input type="text" class="form-control form-control-sm" data-param="valid_choices" placeholder="e.g., A,B,C,D">
                </div>
                <div class="row">
                    <div class="col-md-6 mb-2">
                        <label class="form-label small">Case Sensitive</label>
                        <select class="form-select form-select-sm" data-param="case_sensitive">
                            <option value="false">No</option>
                            <option value="true">Yes</option>
                        </select>
                    </div>
                    <div class="col-md-6 mb-2">
                        <label class="form-label small">Exact Match</label>
                        <select class="form-select form-select-sm" data-param="exact_match">
                            <option value="true">Yes</option>
                            <option value="false">No</option>
                        </select>
                    </div>
                </div>
            `;
            break;
        case 'section_content':
            fieldsHtml = `
                <div class="mb-2">
                    <label class="form-label small">Section Tag</label>
                    <input type="text" class="form-control form-control-sm" data-param="section_tag" placeholder="e.g., analysis">
                </div>
                <div class="row">
                    <div class="col-md-6 mb-2">
                        <label class="form-label small">Min Words</label>
                        <input type="number" class="form-control form-control-sm" data-param="min_words" placeholder="Optional">
                    </div>
                    <div class="col-md-6 mb-2">
                        <label class="form-label small">Max Words</label>
                        <input type="number" class="form-control form-control-sm" data-param="max_words" placeholder="Optional">
                    </div>
                </div>
                <div class="mb-2">
                    <label class="form-label small">Required Keywords (comma-separated)</label>
                    <input type="text" class="form-control form-control-sm" data-param="required_keywords" placeholder="Optional">
                </div>
            `;
            break;
        case 'sequential':
            fieldsHtml = `
                <div class="mb-2">
                    <label class="form-label small">Patterns (comma-separated regex)</label>
                    <input type="text" class="form-control form-control-sm" data-param="patterns" placeholder="e.g., Step 1,Step 2,Step 3">
                </div>
                <div class="mb-2">
                    <label class="form-label small">Strict Order</label>
                    <select class="form-select form-select-sm" data-param="strict_order">
                        <option value="true">Yes</option>
                        <option value="false">No</option>
                    </select>
                </div>
            `;
            break;
    }

    paramsDiv.innerHTML = fieldsHtml;
}

function updateCustomWeightTotal() {
    const weights = document.querySelectorAll('.component-weight');
    let total = 0;
    weights.forEach(input => {
        total += parseFloat(input.value) || 0;
    });

    const display = document.getElementById('custom-weight-total');
    if (display) {
        const isValid = Math.abs(total - 1.0) < 0.001;
        display.className = `badge ${isValid ? 'bg-success' : 'bg-warning'}`;
        display.textContent = `Total Weight: ${total.toFixed(3)} ${isValid ? '✓' : '⚠'}`;
    }
}

function gatherCustomComponents() {
    const components = [];
    const componentCards = document.querySelectorAll('.custom-component');

    componentCards.forEach(card => {
        const name = card.querySelector('.component-name').value.trim();
        const type = card.querySelector('.component-type').value;
        const weight = parseFloat(card.querySelector('.component-weight').value) || 1.0;

        if (!name) return; // Skip components without names

        const component = {
            name: name,
            type: type,
            weight: weight,
            parameters: {}
        };

        // Gather parameters
        const paramInputs = card.querySelectorAll('[data-param]');
        paramInputs.forEach(input => {
            const paramName = input.getAttribute('data-param');
            let value = input.value;

            // Convert to appropriate type
            if (value === 'true') value = true;
            else if (value === 'false') value = false;
            else if (value && value.includes(',')) {
                // Array value (comma-separated)
                value = value.split(',').map(v => v.trim()).filter(v => v);
            } else if (input.type === 'number' && value) {
                value = parseFloat(value);
            }

            if (value !== '' && value !== null && value !== undefined) {
                component.parameters[paramName] = value;
            }
        });

        components.push(component);
    });

    return components;
}

function applyCustomReward() {
    const components = gatherCustomComponents();

    if (components.length === 0) {
        alert('Please add at least one component');
        return;
    }

    // Check weight total
    const totalWeight = components.reduce((sum, c) => sum + c.weight, 0);
    if (Math.abs(totalWeight - 1.0) > 0.001) {
        if (!confirm(`Weight total is ${totalWeight.toFixed(3)}, not 1.0. Continue anyway?`)) {
            return;
        }
    }

    // Update global reward config
    window.selectedRewardConfig = {
        type: 'custom',
        components: components
    };

    selectedRewardName = 'Custom Reward Configuration';
    selectedRewardType = 'custom';

    // Update display
    const nameElement = document.getElementById('selected-reward-name');
    const descElement = document.getElementById('selected-reward-description');
    if (nameElement) nameElement.textContent = selectedRewardName;
    if (descElement) descElement.textContent = `Custom reward with ${components.length} component(s)`;

    // Show success notification
    showNotification('Custom reward applied!', 'success');

    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('customRewardBuilderModal'));
    if (modal) modal.hide();

    // Display components
    displayCustomComponents(components);
}

async function saveCustomRewardAsPreset() {
    const components = gatherCustomComponents();

    if (components.length === 0) {
        alert('Please add at least one component');
        return;
    }

    const name = prompt('Enter a name for this custom preset:');
    if (!name) return;

    const description = prompt('Enter a description (optional):', 'Custom reward configuration') || 'Custom reward configuration';

    try {
        const response = await fetch('/api/rewards/custom-preset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name: name,
                description: description,
                components: components,
                difficulty: 'intermediate',
                tags: ['custom']
            })
        });

        const result = await response.json();

        if (response.ok && result.success) {
            showNotification(`Custom preset "${name}" saved successfully!`, 'success');

            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('customRewardBuilderModal'));
            if (modal) modal.hide();

            // Reload presets
            await initializeRewardSystem();

            // Select the new preset
            filterPresetsByCategory('Custom');
        } else {
            showNotification(`Error: ${result.error || 'Failed to save preset'}`, 'error');
        }
    } catch (error) {
        console.error('Failed to save custom preset:', error);
        showNotification('Failed to save custom preset', 'error');
    }
}

async function deleteCustomPreset(presetName) {
    if (!confirm(`Are you sure you want to delete the custom preset "${presetName}"?`)) {
        return;
    }

    try {
        const response = await fetch(`/api/rewards/custom-preset/${encodeURIComponent(presetName)}`, {
            method: 'DELETE'
        });

        const result = await response.json();

        if (response.ok && result.success) {
            showNotification(`Custom preset "${presetName}" deleted successfully!`, 'success');

            // Reload presets
            await initializeRewardSystem();

            // Refresh display
            filterPresetsByCategory('Custom');
        } else {
            showNotification(`Error: ${result.error || 'Failed to delete preset'}`, 'error');
        }
    } catch (error) {
        console.error('Failed to delete custom preset:', error);
        showNotification('Failed to delete custom preset', 'error');
    }
}

// Export all reward functions to window object for global access
window.gatherRewardConfig = gatherRewardConfig;
window.restoreCustomReward = restoreCustomReward;
window.showCustomRewardAsSelected = showCustomRewardAsSelected;
window.selectRewardType = selectRewardType;
window.addRewardComponent = addRewardComponent;
window.removeRewardComponent = removeRewardComponent;
window.updateComponentType = updateComponentType;
window.selectPresetByName = selectPresetByName;
window.selectTemplate = selectTemplate;
window.addAdvancedComponent = addAdvancedComponent;
window.updateAdvancedComponentType = updateAdvancedComponentType;
window.removeComponent = removeComponent;
window.testSelectedReward = testSelectedReward;
window.saveCustomReward = saveCustomReward;
window.viewRewardDetails = viewRewardDetails;
window.confirmPresetSelection = confirmPresetSelection;
window.loadSavedSelection = loadSavedSelection;
window.filterPresetsByCategory = filterPresetsByCategory;

// Export new model-based testing functions
window.toggleTestMode = toggleTestMode;
window.loadAvailableModels = loadAvailableModels;
window.testRewardWithModel = testRewardWithModel;
window.testRewardWithOrWithoutModel = testRewardWithOrWithoutModel;
window.onGeneratedTextEdit = onGeneratedTextEdit;
window.restoreOriginalGenerated = restoreOriginalGenerated;

// Export preset customization functions
window.makePresetEditable = makePresetEditable;
window.updateWeightTotal = updateWeightTotal;
window.resetPresetToOriginal = resetPresetToOriginal;
window.saveCustomizedPreset = saveCustomizedPreset;
window.toggleParameters = toggleParameters;

// Export custom reward builder functions
window.openCustomRewardBuilder = openCustomRewardBuilder;
window.addCustomComponent = addCustomComponent;
window.removeCustomComponent = removeCustomComponent;
window.updateCustomComponentFields = updateCustomComponentFields;
window.updateCustomWeightTotal = updateCustomWeightTotal;
window.applyCustomReward = applyCustomReward;
window.saveCustomRewardAsPreset = saveCustomRewardAsPreset;
window.deleteCustomPreset = deleteCustomPreset;

// Toggle function for expandable parameters
function toggleParameters(index) {
    const content = document.getElementById(`params-${index}`);
    const icon = document.getElementById(`param-icon-${index}`);

    if (content.style.display === 'none') {
        content.style.display = 'block';
        icon.classList.remove('fa-chevron-down');
        icon.classList.add('fa-chevron-up');
    } else {
        content.style.display = 'none';
        icon.classList.remove('fa-chevron-up');
        icon.classList.add('fa-chevron-down');
    }
}

// Toggle reward panel collapse
function toggleRewardPanel() {
    const content = document.getElementById('selected-reward-display');
    const chevron = document.getElementById('reward-panel-chevron');

    if (content && chevron) {
        if (content.classList.contains('show')) {
            content.classList.remove('show');
            chevron.classList.remove('fa-chevron-up');
            chevron.classList.add('fa-chevron-down');
        } else {
            content.classList.add('show');
            chevron.classList.remove('fa-chevron-down');
            chevron.classList.add('fa-chevron-up');
        }
    }
}

window.toggleRewardPanel = toggleRewardPanel;

// ============================================================================
// Field Mapping Functions
// ============================================================================

// Store current dataset columns globally
window.currentDatasetColumns = null;
window.currentFieldMapping = {};

// Show field mapping modal
async function showFieldMappingModal(presetName, datasetColumns) {
    if (!presetName || !datasetColumns || datasetColumns.length === 0) {
        showNotification('Missing preset or dataset information', 'warning');
        return;
    }

    try {
        // Get current field mappings from dataset step
        const currentDatasetMapping = {
            instruction: document.getElementById('instruction-field')?.value || '',
            response: document.getElementById('response-field')?.value || '',
            reasoning: document.getElementById('reasoning-field')?.value || ''
        };

        // Merge with any existing window.currentFieldMapping
        const currentMapping = { ...window.currentFieldMapping, ...currentDatasetMapping };

        // Call validation API to get suggestions
        const response = await fetch('/api/rewards/validate-fields', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                reward_preset: presetName,
                dataset_columns: datasetColumns,
                current_mapping: currentMapping
            })
        });

        const result = await response.json();

        if (result.error) {
            showNotification(`Error: ${result.error}`, 'error');
            return;
        }

        // Build modal HTML
        const modalHtml = createFieldMappingModalHTML(presetName, datasetColumns, result);

        // Remove existing modal if present
        const existingModal = document.getElementById('fieldMappingModal');
        if (existingModal) existingModal.remove();

        // Add modal to page
        document.body.insertAdjacentHTML('beforeend', modalHtml);

        // Initialize dropdowns with suggestions
        initializeFieldMappingDropdowns(result.suggestions, datasetColumns);

        // Show modal
        const modal = new bootstrap.Modal(document.getElementById('fieldMappingModal'));
        modal.show();

        // Clean up on hide
        document.getElementById('fieldMappingModal').addEventListener('hidden.bs.modal', function() {
            this.remove();
        });

    } catch (error) {
        console.error('Error showing field mapping modal:', error);
        showNotification('Failed to load field mapping', 'error');
    }
}

// Create field mapping modal HTML
function createFieldMappingModalHTML(presetName, datasetColumns, validationResult) {
    const preset = rewardPresets[presetName];
    const presetDisplayName = preset ? preset.name : presetName;

    // Generate compatibility score HTML
    const compatScore = validationResult.compatibility_score || 0;
    const compatBarWidth = Math.min(compatScore, 100);
    const compatClass = compatScore >= 80 ? 'success' : compatScore >= 50 ? 'warning' : 'danger';

    // Generate field mapping rows
    const fieldRows = Object.entries(validationResult.expected_fields).map(([fieldName, fieldDesc]) => {
        const isOptional = validationResult.optional_fields.includes(fieldName);
        const suggestion = validationResult.suggestions[fieldName] || '';
        const example = validationResult.field_examples[fieldName] || '';
        const confidence = validationResult.confidence_scores[fieldName] || 0;

        return `
            <div class="field-mapping-row">
                <div class="field-box dataset-field">
                    <div class="field-box-label">Dataset Column</div>
                    <select class="field-select" data-field="${fieldName}" id="map-${fieldName}">
                        <option value="">-- Select Column --</option>
                        ${datasetColumns.map(col =>
                            `<option value="${col}" ${col === suggestion ? 'selected' : ''}>${col}</option>`
                        ).join('')}
                    </select>
                    ${confidence > 0 ? `<small class="text-muted mt-1 d-block">Confidence: ${Math.round(confidence * 100)}%</small>` : ''}
                </div>

                <div class="field-arrow">
                    <i class="fas fa-arrow-right"></i>
                </div>

                <div class="field-box reward-field">
                    <div class="field-box-label">
                        Reward Field ${isOptional ? '<span class="badge field-status-badge optional ms-1">Optional</span>' : '<span class="badge field-status-badge mapped ms-1">Required</span>'}
                    </div>
                    <div class="field-box-content">${fieldName}</div>
                    <small class="text-muted d-block mt-1">${fieldDesc}</small>
                    ${example ? `<div class="field-box-example">Example: <code>${example}</code></div>` : ''}
                </div>
            </div>
        `;
    }).join('');

    // Generate warnings HTML
    const warningsHtml = validationResult.warnings.length > 0 ? `
        <div class="warnings-section mb-3">
            ${validationResult.warnings.map(warning => {
                const iconClass = warning.type === 'missing_required' ? 'fa-exclamation-circle' : 'fa-info-circle';
                const alertClass = warning.type === 'missing_required' ? 'field-warning' : 'field-info';
                return `
                    <div class="${alertClass}">
                        <i class="fas ${iconClass}"></i>
                        ${warning.message}
                    </div>
                `;
            }).join('')}
        </div>
    ` : '';

    return `
        <div class="modal fade" id="fieldMappingModal" tabindex="-1">
            <div class="modal-dialog modal-xl">
                <div class="modal-content field-mapping-modal">
                    <div class="modal-header">
                        <h5 class="modal-title">
                            <i class="fas fa-link"></i> Field Mapping: ${presetDisplayName}
                        </h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <!-- Compatibility Score -->
                        <div class="compatibility-score-container">
                            <div class="compatibility-label">
                                <i class="fas fa-chart-line"></i> Dataset Compatibility
                            </div>
                            <div class="compatibility-bar">
                                <div class="compatibility-fill bg-${compatClass}" style="width: ${compatBarWidth}%">
                                    <span class="compatibility-text">${compatScore}%</span>
                                </div>
                            </div>
                            <div class="compatibility-details mt-2">
                                ${validationResult.valid ?
                                    '<span class="field-status-badge mapped"><i class="fas fa-check"></i> All required fields mapped</span>' :
                                    '<span class="field-status-badge missing"><i class="fas fa-times"></i> Missing required fields</span>'
                                }
                            </div>
                        </div>

                        ${warningsHtml}

                        <h6 class="mb-3"><i class="fas fa-exchange-alt"></i> Map Your Dataset Fields</h6>

                        ${fieldRows}

                        <!-- Preview Section -->
                        <div class="mapping-preview-section mt-4">
                            <h6><i class="fas fa-eye"></i> Preview</h6>
                            <div id="mapping-preview-content">
                                <p class="text-muted">Select fields to see a preview of how your data will be mapped</p>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                            <i class="fas fa-times"></i> Cancel
                        </button>
                        <button type="button" class="btn btn-auto-map" onclick="autoMapFields('${presetName}')">
                            <i class="fas fa-magic"></i> Auto-Map
                        </button>
                        <button type="button" class="btn btn-primary" onclick="confirmFieldMapping('${presetName}')">
                            <i class="fas fa-check"></i> Confirm Mapping
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Initialize field mapping dropdowns
function initializeFieldMappingDropdowns(suggestions, datasetColumns) {
    // Set up change listeners on all dropdowns
    document.querySelectorAll('.field-select').forEach(select => {
        select.addEventListener('change', function() {
            updateMappingPreview();
            validateCurrentMapping();
        });
    });

    // Initial preview update
    updateMappingPreview();
}

// Update mapping preview
function updateMappingPreview() {
    const previewContent = document.getElementById('mapping-preview-content');
    if (!previewContent) return;

    // Gather current mapping
    const mapping = {};
    document.querySelectorAll('.field-select').forEach(select => {
        const field = select.dataset.field;
        const value = select.value;
        if (value) {
            mapping[field] = value;
        }
    });

    // Show preview if we have a dataset loaded
    if (window.currentDatasetPreview && window.currentDatasetPreview.length > 0) {
        const sampleData = window.currentDatasetPreview[0]; // First row

        let previewHtml = '<div class="mapping-preview-card">';
        previewHtml += '<div class="mapping-preview-header"><i class="fas fa-table"></i> Sample Data Mapping</div>';

        Object.entries(mapping).forEach(([rewardField, datasetCol]) => {
            const value = sampleData[datasetCol] || 'N/A';
            const truncated = value.length > 100 ? value.substring(0, 100) + '...' : value;
            previewHtml += `
                <div class="mapping-example mb-2">
                    <strong>${rewardField}</strong> ← <em>${datasetCol}</em><br>
                    <code>${truncated}</code>
                </div>
            `;
        });

        previewHtml += '</div>';
        previewContent.innerHTML = previewHtml;
    } else {
        previewContent.innerHTML = `
            <div class="mapping-preview-card">
                <div class="mapping-preview-header"><i class="fas fa-info-circle"></i> Mapping Summary</div>
                ${Object.entries(mapping).map(([rewardField, datasetCol]) => `
                    <div class="mapping-example">
                        <strong>${rewardField}</strong> → <em>${datasetCol}</em>
                    </div>
                `).join('')}
            </div>
        `;
    }
}

// Validate current mapping
function validateCurrentMapping() {
    // Get required fields from modal data
    const requiredFields = [];
    document.querySelectorAll('.field-status-badge.mapped').forEach(badge => {
        const fieldBox = badge.closest('.field-box');
        if (fieldBox) {  // Add null check to prevent error
            const fieldNameElement = fieldBox.querySelector('.field-box-content');
            if (fieldNameElement) {
                const fieldName = fieldNameElement.textContent.trim();
                requiredFields.push(fieldName);
            }
        }
    });

    // Check if all required fields are mapped
    const mapping = {};
    document.querySelectorAll('.field-select').forEach(select => {
        const field = select.dataset.field;
        const value = select.value;
        if (value) {
            mapping[field] = value;
        }
    });

    const allMapped = requiredFields.every(field => mapping[field]);

    // Enable/disable confirm button
    const confirmBtn = document.querySelector('#fieldMappingModal .btn-primary');
    if (confirmBtn) {
        confirmBtn.disabled = !allMapped;
        if (!allMapped) {
            confirmBtn.title = 'Please map all required fields';
        } else {
            confirmBtn.title = '';
        }
    }
}

// Auto-map fields
async function autoMapFields(presetName) {
    const datasetColumns = window.currentDatasetColumns;
    if (!datasetColumns) return;

    try {
        const response = await fetch('/api/rewards/validate-fields', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                reward_preset: presetName,
                dataset_columns: datasetColumns
            })
        });

        const result = await response.json();

        // Apply suggestions to dropdowns
        Object.entries(result.suggestions).forEach(([field, column]) => {
            const select = document.getElementById(`map-${field}`);
            if (select) {
                select.value = column;
            }
        });

        updateMappingPreview();
        validateCurrentMapping();
        showNotification('Fields auto-mapped successfully', 'success');

    } catch (error) {
        console.error('Error auto-mapping fields:', error);
        showNotification('Failed to auto-map fields', 'error');
    }
}

// Confirm field mapping
function confirmFieldMapping(presetName) {
    // Gather final mapping
    const mapping = {};
    document.querySelectorAll('.field-select').forEach(select => {
        const field = select.dataset.field;
        const value = select.value;
        if (value) {
            mapping[field] = value;
        }
    });

    // Store mapping globally
    window.currentFieldMapping = mapping;

    // Update the configuration
    if (window.selectedRewardConfig) {
        window.selectedRewardConfig.field_mapping = mapping;
    }

    // Store in localStorage
    localStorage.setItem('fieldMapping', JSON.stringify(mapping));

    // Close modal
    const modal = bootstrap.Modal.getInstance(document.getElementById('fieldMappingModal'));
    if (modal) modal.hide();

    // Show success message
    showNotification(`Field mapping confirmed for ${rewardPresets[presetName]?.name || presetName}`, 'success');

    // Update reward display to show mapping status
    updateRewardMappingStatus(presetName, mapping);
}

// Update reward display with mapping status
function updateRewardMappingStatus(presetName, mapping) {
    const componentDisplay = document.getElementById('preset-component-display');
    if (!componentDisplay) return;

    // Add mapping status indicator
    const mappingStatusHtml = `
        <div class="card mt-3">
            <div class="card-header bg-success text-white">
                <i class="fas fa-check-circle"></i> Field Mapping Configured
            </div>
            <div class="card-body">
                <div class="row">
                    ${Object.entries(mapping).map(([field, column]) => `
                        <div class="col-md-6 mb-2">
                            <span class="field-status-badge mapped">
                                <i class="fas fa-link"></i> ${field} → ${column}
                            </span>
                        </div>
                    `).join('')}
                </div>
            </div>
        </div>
    `;

    // Check if mapping status already exists
    const existingStatus = componentDisplay.querySelector('.field-mapping-status');
    if (existingStatus) {
        existingStatus.remove();
    }

    componentDisplay.insertAdjacentHTML('beforeend', `<div class="field-mapping-status">${mappingStatusHtml}</div>`);
}

// Function to show field mapping for a specific preset (without selecting it first)
function showFieldMappingForPreset(presetName) {
    if (!window.currentDatasetColumns || window.currentDatasetColumns.length === 0) {
        showNotification('Please load a dataset first to map fields', 'warning');
        return;
    }

    if (!rewardPresets[presetName]) {
        showNotification('Invalid reward preset', 'error');
        return;
    }

    showFieldMappingModal(presetName, window.currentDatasetColumns);
}

// Show field mapping for current reward and dataset
function showFieldMappingForCurrentReward() {
    if (!selectedRewardName || !window.currentDatasetColumns) {
        showNotification('Please select both a dataset and a reward function first', 'warning');
        return;
    }

    // Check if this is a custom reward
    if (selectedRewardType === 'custom' || selectedRewardName.includes('Custom')) {
        showNotification('Field mapping is only available for preset rewards. Custom rewards use the general dataset field mappings.', 'info');
        return;
    }

    const presetName = Object.keys(rewardPresets).find(key =>
        rewardPresets[key].name === selectedRewardName
    );

    if (presetName) {
        showFieldMappingModal(presetName, window.currentDatasetColumns);
    } else {
        showNotification('Could not find reward preset for field mapping', 'warning');
    }
}

// Export field mapping functions
window.showFieldMappingModal = showFieldMappingModal;
window.showFieldMappingForPreset = showFieldMappingForPreset;
window.showFieldMappingForCurrentReward = showFieldMappingForCurrentReward;
window.confirmFieldMapping = confirmFieldMapping;
window.autoMapFields = autoMapFields;

// ============================================================================
// Initialize reward system when DOM is ready
// ============================================================================

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        initializeRewardSystem();
        loadSavedSelection();
        setupTabListeners();
    });
} else {
    initializeRewardSystem();
    loadSavedSelection();
    setupTabListeners();
}

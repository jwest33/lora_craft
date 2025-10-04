// ============================================================================
// Model Management Module
// ============================================================================

(function(window) {
    'use strict';

    const ModelsModule = {
        // Model definitions by family
        modelsByFamily: {
            qwen: [
                { id: 'unsloth/Qwen3-0.6B', name: 'Qwen3 0.6B', size: '600M', vram: '~1.2GB' },
                { id: 'unsloth/Qwen3-1.7B', name: 'Qwen3 1.7B', size: '1.7B', vram: '~3.4GB' },
                { id: 'unsloth/Qwen3-4B', name: 'Qwen3 4B', size: '4B', vram: '~8GB' },
                { id: 'unsloth/Qwen3-8B', name: 'Qwen3 8B', size: '8B', vram: '~16GB' }
            ],
            llama: [
                { id: 'unsloth/Llama-3.2-1B-Instruct', name: 'LLaMA 3.2 1B', size: '1B', vram: '~2GB' },
                { id: 'unsloth/Llama-3.2-3B-Instruct', name: 'LLaMA 3.2 3B', size: '3B', vram: '~6GB' }
            ],
            phi: [
                { id: 'unsloth/phi-4-reasoning', name: 'Phi-4 Reasoning', size: '15B', vram: '~30GB' }
            ]
        },

        // Initialize the module
        init() {
            this.setupEventListeners();
            this.loadAvailableModels();
            this.updateModelList(); // Initialize model list based on default family
        },

        // Setup model-related event listeners
        setupEventListeners() {
            // Model family change
            const modelFamilySelect = document.getElementById('model-family');
            if (modelFamilySelect) {
                modelFamilySelect.addEventListener('change', () => this.updateModelList());
            }

            // Model selection change
            const modelSelect = document.getElementById('model-name');
            if (modelSelect) {
                modelSelect.addEventListener('change', () => this.onModelChange());
            }

            // LoRA rank change
            const loraRankInput = document.getElementById('lora-rank');
            if (loraRankInput) {
                loraRankInput.addEventListener('change', () => this.updateLoraAlpha());
            }

            // Model search
            const modelSearchInput = document.getElementById('model-search');
            if (modelSearchInput) {
                modelSearchInput.addEventListener('input', CoreModule.debounce(() => {
                    this.searchModels(modelSearchInput.value);
                }, 300));
            }

            // Setup mode radio buttons
            const setupModeRadios = document.querySelectorAll('input[name="setup-mode"]');
            setupModeRadios.forEach(radio => {
                radio.addEventListener('change', () => this.handleSetupModeChange());
            });
        },

        // Handle setup mode changes (Recommended/Custom/Advanced)
        handleSetupModeChange() {
            const loraConfigSection = document.getElementById('lora-config-section');
            const loraPresetsSection = document.getElementById('lora-presets');
            const advancedConfigSection = document.getElementById('advanced-config-section');

            // Get selected mode
            const selectedMode = document.querySelector('input[name="setup-mode"]:checked');
            if (!selectedMode) return;

            const mode = selectedMode.id;

            // Handle visibility based on mode
            switch (mode) {
                case 'setup-recommended':
                    // Hide all advanced options
                    if (loraConfigSection) loraConfigSection.style.display = 'none';
                    if (advancedConfigSection) advancedConfigSection.style.display = 'none';
                    break;

                case 'setup-custom':
                    // Show LoRA config with presets, hide advanced
                    if (loraConfigSection) loraConfigSection.style.display = 'block';
                    if (loraPresetsSection) loraPresetsSection.style.display = 'block';
                    if (advancedConfigSection) advancedConfigSection.style.display = 'none';
                    break;

                case 'setup-advanced':
                    // Show both LoRA and advanced config
                    if (loraConfigSection) loraConfigSection.style.display = 'block';
                    if (loraPresetsSection) loraPresetsSection.style.display = 'block';
                    if (advancedConfigSection) advancedConfigSection.style.display = 'block';
                    break;
            }

            // Save the selected mode to state
            if (window.AppState && AppState.setConfigValue) {
                AppState.setConfigValue('setupMode', mode);
            }
        },

        // Update model list based on selected family
        updateModelList() {
            const familySelect = document.getElementById('model-family');
            const modelSelect = document.getElementById('model-name');

            if (!familySelect || !modelSelect) return;

            const selectedFamily = familySelect.value;
            const models = this.modelsByFamily[selectedFamily] || [];

            // Clear existing options
            modelSelect.innerHTML = '<option value="">Select a model...</option>';

            // Add options for the selected family
            models.forEach(model => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = `${model.name} (${model.size}, ${model.vram})`;
                modelSelect.appendChild(option);
            });

            // Trigger validation
            if (window.NavigationModule && NavigationModule.validateStep) {
                NavigationModule.validateStep(1);
            }
        },

        // Load available models from server
        loadAvailableModels() {
            fetch('/api/models')
                .then(response => response.json())
                .then(data => {
                    const models = data.models || [];
                    // Store server models for reference, but don't populate the dropdown
                    // The dropdown is populated by updateModelList() based on family selection
                    if (models.length > 0) {
                        AppState.availableModels = models;
                    }
                })
                .catch(error => {
                    console.error('Failed to load models:', error);
                    // Don't show alert on initialization, just log
                });
        },

        // Populate model select dropdown
        populateModelSelect(models) {
            const modelSelect = document.getElementById('model-name');
            if (!modelSelect) return;

            // Clear existing options
            modelSelect.innerHTML = '<option value="">Select a model...</option>';

            // Group models by category if categories exist
            const categories = {};
            let hasCategories = false;

            models.forEach(model => {
                if (model.category) {
                    hasCategories = true;
                    if (!categories[model.category]) {
                        categories[model.category] = [];
                    }
                    categories[model.category].push(model);
                }
            });

            if (hasCategories) {
                // Add models grouped by category
                Object.keys(categories).sort().forEach(category => {
                    const optgroup = document.createElement('optgroup');
                    optgroup.label = category.toUpperCase();

                    categories[category].forEach(model => {
                        const option = document.createElement('option');
                        option.value = model.id || model.name;
                        option.textContent = `${model.name} (${model.size}, ${model.vram})`;

                        if (model.description) {
                            option.setAttribute('data-description', model.description);
                        }

                        optgroup.appendChild(option);
                    });

                    modelSelect.appendChild(optgroup);
                });
            } else {
                // Add models without grouping (fallback)
                models.forEach(model => {
                    const option = document.createElement('option');
                    option.value = model.id || model.name;
                    option.textContent = model.name;

                    if (model.description) {
                        option.setAttribute('data-description', model.description);
                    }

                    modelSelect.appendChild(option);
                });
            }

            // Restore saved selection if exists
            const savedModel = AppState.getConfigValue('modelName');
            if (savedModel && models.some(m => m.id === savedModel || m.name === savedModel)) {
                modelSelect.value = savedModel;
                this.onModelChange();
            }
        },

        // Handle model selection change
        onModelChange() {
            const modelSelect = document.getElementById('model-name');
            if (!modelSelect) return;

            const selectedModel = modelSelect.value;
            AppState.setConfigValue('modelName', selectedModel);

            // Update model info display
            this.updateModelInfo(selectedModel);

            // Update configuration summary
            if (typeof window.updateConfigSummary === 'function') {
                window.updateConfigSummary();
            }

            // Validate step
            NavigationModule.validateStep(1);

            // Save state
            CoreModule.saveState();
        },

        // Update model information display
        updateModelInfo(modelId) {
            const modelInfo = document.getElementById('model-info');
            if (!modelInfo) return;

            if (!modelId) {
                modelInfo.innerHTML = '<p class="text-muted">Select a model to see details</p>';
                return;
            }

            // Handle both array and object formats for availableModels
            let model = null;

            if (Array.isArray(AppState.availableModels)) {
                // Direct array format
                model = AppState.availableModels.find(m => m.id === modelId || m.name === modelId);
            } else if (AppState.availableModels?.models && Array.isArray(AppState.availableModels.models)) {
                // Object with models property
                model = AppState.availableModels.models.find(m => m.id === modelId || m.name === modelId);
            }

            if (model) {
                modelInfo.innerHTML = `
                    <div class="model-details">
                        <h6>${CoreModule.escapeHtml(model.name)}</h6>
                        ${model.description ? `<p class="text-muted small">${CoreModule.escapeHtml(model.description)}</p>` : ''}
                        ${model.parameters ? `<p><strong>Parameters:</strong> ${CoreModule.escapeHtml(model.parameters)}</p>` : ''}
                        ${model.context_length ? `<p><strong>Context Length:</strong> ${CoreModule.escapeHtml(model.context_length)}</p>` : ''}
                    </div>
                `;
            }
        },

        // Update LoRA alpha based on rank
        updateLoraAlpha() {
            const loraRankInput = document.getElementById('lora-rank');
            const loraAlphaInput = document.getElementById('lora-alpha');

            if (loraRankInput && loraAlphaInput) {
                const rank = parseInt(loraRankInput.value) || 8;
                // Set alpha to 2x rank by default (common practice)
                loraAlphaInput.value = rank * 2;

                AppState.setConfigValue('loraRank', rank);
                AppState.setConfigValue('loraAlpha', rank * 2);
            }
        },

        // Search models
        searchModels(query) {
            const modelSelect = document.getElementById('model-name');
            if (!modelSelect) return;

            const options = modelSelect.querySelectorAll('option');
            let hasVisibleOption = false;

            options.forEach(option => {
                if (option.value === '') return; // Skip placeholder

                const text = option.textContent.toLowerCase();
                const description = option.getAttribute('data-description')?.toLowerCase() || '';
                const matches = text.includes(query.toLowerCase()) || description.includes(query.toLowerCase());

                option.style.display = matches ? '' : 'none';
                if (matches) hasVisibleOption = true;
            });

            // Show no results message if needed
            if (!hasVisibleOption && query) {
                CoreModule.showAlert('No models found matching your search', 'warning');
            }
        },

        // Get recommended LoRA settings for model
        getRecommendedSettings(modelName) {
            // Default recommendations based on model size
            const recommendations = {
                '7b': { rank: 8, alpha: 16, learningRate: 2e-4 },
                '13b': { rank: 16, alpha: 32, learningRate: 1e-4 },
                '30b': { rank: 32, alpha: 64, learningRate: 5e-5 },
                '70b': { rank: 64, alpha: 128, learningRate: 2e-5 }
            };

            // Try to determine model size from name
            const modelLower = modelName.toLowerCase();
            for (const [size, settings] of Object.entries(recommendations)) {
                if (modelLower.includes(size)) {
                    return settings;
                }
            }

            // Return default if size not detected
            return recommendations['7b'];
        },

        // Apply recommended settings
        applyRecommendedSettings() {
            const modelSelect = document.getElementById('model-name');
            if (!modelSelect || !modelSelect.value) {
                CoreModule.showAlert('Please select a model first', 'warning');
                return;
            }

            const settings = this.getRecommendedSettings(modelSelect.value);

            // Apply settings to form
            if (settings.rank) {
                document.getElementById('lora-rank').value = settings.rank;
                document.getElementById('lora-alpha').value = settings.alpha;
            }

            if (settings.learningRate) {
                const lrInput = document.getElementById('learning-rate');
                if (lrInput) lrInput.value = settings.learningRate;
            }

            CoreModule.showAlert('Recommended settings applied', 'success');
            NavigationModule.validateStep(1);
        },

        // Validate model configuration
        validateModelConfig() {
            const modelName = document.getElementById('model-name')?.value;
            const loraRank = document.getElementById('lora-rank')?.value;
            const loraAlpha = document.getElementById('lora-alpha')?.value;

            // Check if at least one target module checkbox is selected
            const hasTargetModule =
                document.getElementById('target-q-proj')?.checked ||
                document.getElementById('target-v-proj')?.checked ||
                document.getElementById('target-k-proj')?.checked ||
                document.getElementById('target-o-proj')?.checked;

            const errors = [];

            if (!modelName) {
                errors.push('Model selection is required');
            }

            if (!loraRank || loraRank < 1 || loraRank > 256) {
                errors.push('LoRA rank must be between 1 and 256');
            }

            if (!loraAlpha || loraAlpha < 1) {
                errors.push('LoRA alpha must be positive');
            }

            if (!hasTargetModule) {
                errors.push('At least one target module must be selected');
            }

            if (errors.length > 0) {
                errors.forEach(error => CoreModule.showAlert(error, 'warning'));
                return false;
            }

            return true;
        },

        // Export model configuration
        exportModelConfig() {
            const config = {
                modelName: document.getElementById('model-name')?.value,
                customModelPath: document.getElementById('custom-model-path')?.value || '',
                quantization: document.getElementById('quantization')?.value || 'q8_0'
            };

            return config;
        },

        // Import model configuration
        importModelConfig(config) {
            console.log('ModelsModule.importModelConfig called with:', config);

            if (config.modelName) {
                const modelSelect = document.getElementById('model-name');
                if (modelSelect) {
                    modelSelect.value = config.modelName;
                    this.onModelChange();
                } else {
                    console.warn('model-name element not found');
                }
            }

            if (config.customModelPath) {
                const customPathInput = document.getElementById('custom-model-path');
                if (customPathInput) {
                    customPathInput.value = config.customModelPath;
                }
            }

            if (config.loraRank) {
                const loraRankInput = document.getElementById('lora-rank');
                const loraRankSlider = document.getElementById('lora-rank-slider');
                if (loraRankInput) loraRankInput.value = config.loraRank;
                if (loraRankSlider) loraRankSlider.value = config.loraRank;
            }

            if (config.loraAlpha) {
                const loraAlphaInput = document.getElementById('lora-alpha');
                const loraAlphaSlider = document.getElementById('lora-alpha-slider');
                if (loraAlphaInput) loraAlphaInput.value = config.loraAlpha;
                if (loraAlphaSlider) loraAlphaSlider.value = config.loraAlpha;
            }

            // Restore individual target module checkboxes
            if (config.targetModulesArray && Array.isArray(config.targetModulesArray)) {
                // First uncheck all (including MLP modules)
                const targetCheckboxes = [
                    'target-q-proj', 'target-v-proj', 'target-k-proj', 'target-o-proj',
                    'target-gate-proj', 'target-up-proj', 'target-down-proj'
                ];
                targetCheckboxes.forEach(id => {
                    const checkbox = document.getElementById(id);
                    if (checkbox) checkbox.checked = false;
                });

                // Then check the ones in the array
                config.targetModulesArray.forEach(module => {
                    const checkboxId = `target-${module.replace('_', '-')}`;
                    const checkbox = document.getElementById(checkboxId);
                    if (checkbox) checkbox.checked = true;
                });
            }

            if (config.loraDropout !== undefined) {
                const loraDropoutInput = document.getElementById('lora-dropout');
                const loraDropoutSlider = document.getElementById('lora-dropout-slider');
                if (loraDropoutInput) loraDropoutInput.value = config.loraDropout;
                if (loraDropoutSlider) loraDropoutSlider.value = config.loraDropout;
            }

            if (config.loraBias) {
                const loraBiasSelect = document.getElementById('lora-bias');
                if (loraBiasSelect) {
                    loraBiasSelect.value = config.loraBias;
                }
            }

            if (config.quantization) {
                const quantizationSelect = document.getElementById('quantization');
                if (quantizationSelect) {
                    quantizationSelect.value = config.quantization;
                }
            }
        },

        // Apply LoRA configuration preset
        applyLoRAPreset(preset) {
            const loraRankInput = document.getElementById('lora-rank');
            const loraAlphaInput = document.getElementById('lora-alpha');
            const loraDropoutInput = document.getElementById('lora-dropout');

            if (!loraRankInput || !loraAlphaInput) {
                CoreModule.showAlert('LoRA configuration fields not found', 'danger');
                return;
            }

            let rank, alpha, dropout;

            // Define preset configurations
            switch (preset) {
                case 'rl':  // RL/GRPO
                    rank = 1;
                    alpha = 32;
                    dropout = 0.0;
                    break;
                case 'sft':  // SFT
                    rank = 256;
                    alpha = 32;
                    dropout = 0.0;
                    break;
                case 'low':
                    rank = 8;
                    alpha = 16;
                    dropout = 0.0;
                    break;
                case 'balanced':
                    rank = 16;
                    alpha = 32;
                    dropout = 0.0;
                    break;
                case 'quality':
                    rank = 64;
                    alpha = 128;
                    dropout = 0.0;
                    break;
                default:
                    CoreModule.showAlert('Invalid preset selected', 'warning');
                    return;
            }

            // Apply values to form
            loraRankInput.value = rank;
            loraAlphaInput.value = alpha;
            if (loraDropoutInput) loraDropoutInput.value = dropout;

            // Update sliders if present
            const loraRankSlider = document.getElementById('lora-rank-slider');
            const loraAlphaSlider = document.getElementById('lora-alpha-slider');
            const loraDropoutSlider = document.getElementById('lora-dropout-slider');
            if (loraRankSlider) loraRankSlider.value = rank;
            if (loraAlphaSlider) loraAlphaSlider.value = alpha;
            if (loraDropoutSlider) loraDropoutSlider.value = dropout;

            // Save to AppState
            AppState.setConfigValue('loraRank', rank);
            AppState.setConfigValue('loraAlpha', alpha);
            AppState.setConfigValue('loraDropout', dropout);

            // Trigger validation
            if (window.NavigationModule && NavigationModule.validateStep) {
                NavigationModule.validateStep(1);
            }

            // Show feedback based on preset
            const presetDescriptions = {
                'rl': 'RL/GRPO preset (Rank: 1) - Recommended for reinforcement learning',
                'sft': 'SFT preset (Rank: 256) - Recommended for large-scale supervised fine-tuning',
                'low': 'Low VRAM preset (Rank: 8) - Best for limited GPU memory',
                'balanced': 'Balanced preset (Rank: 16) - Good balance of quality and speed',
                'quality': 'High quality preset (Rank: 64) - Best results with more VRAM'
            };

            CoreModule.showAlert(presetDescriptions[preset] || 'Preset applied', 'success');
        },

        // Select all-linear modules (attention + MLP)
        selectAllLinearModules() {
            const allModules = [
                'target-q-proj', 'target-v-proj', 'target-k-proj', 'target-o-proj',
                'target-gate-proj', 'target-up-proj', 'target-down-proj'
            ];

            allModules.forEach(id => {
                const checkbox = document.getElementById(id);
                if (checkbox) checkbox.checked = true;
            });

            CoreModule.showAlert('All-linear configuration applied', 'success');
        },

        // Load model from custom path
        loadFromPath() {
            const pathInput = document.getElementById('custom-model-path');

            if (!pathInput) {
                CoreModule.showAlert('Custom path input not found', 'danger');
                return;
            }

            const modelPath = pathInput.value.trim();

            if (!modelPath) {
                CoreModule.showAlert('Please enter a model path or HuggingFace ID', 'warning');
                return;
            }

            // Show loading state
            const loadBtn = document.querySelector('[onclick="loadFromPath()"]');
            if (loadBtn) {
                loadBtn.disabled = true;
                loadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
            }

            // Validate and load model from path
            fetch('/api/validate_model_path', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: modelPath })
            })
            .then(response => response.json())
            .then(data => {
                if (data.valid) {
                    // Add model to available models
                    const customModel = {
                        id: modelPath,
                        name: data.name || modelPath,
                        size: data.size || 'Unknown',
                        vram: data.vram || 'Unknown',
                        description: data.description || 'Custom model path',
                        category: 'custom'
                    };

                    // Add to AppState if not already present
                    if (!Array.isArray(AppState.availableModels)) {
                        AppState.availableModels = [];
                    }

                    const existingIndex = AppState.availableModels.findIndex(m => m.id === modelPath);
                    if (existingIndex === -1) {
                        AppState.availableModels.push(customModel);
                    }

                    // Add to model select dropdown
                    const modelSelect = document.getElementById('model-name');
                    if (modelSelect) {
                        // Create custom optgroup if it doesn't exist
                        let customOptgroup = modelSelect.querySelector('optgroup[label="Custom"]');
                        if (!customOptgroup) {
                            customOptgroup = document.createElement('optgroup');
                            customOptgroup.label = 'Custom';
                            modelSelect.appendChild(customOptgroup);
                        }

                        // Add option if not exists
                        const existingOption = modelSelect.querySelector(`option[value="${modelPath}"]`);
                        if (!existingOption) {
                            const option = document.createElement('option');
                            option.value = modelPath;
                            option.textContent = `${customModel.name} (Custom)`;
                            customOptgroup.appendChild(option);
                        }

                        // Select the model
                        modelSelect.value = modelPath;
                        this.onModelChange();
                    }

                    // Save to AppState
                    AppState.setConfigValue('modelName', modelPath);
                    AppState.setConfigValue('customModelPath', modelPath);

                    CoreModule.showAlert(`Model loaded successfully: ${data.name || modelPath}`, 'success');

                    // Clear input
                    pathInput.value = '';
                } else {
                    throw new Error(data.error || 'Invalid model path');
                }
            })
            .catch(error => {
                console.error('Failed to load model from path:', error);
                CoreModule.showAlert(`Failed to load model: ${error.message}`, 'danger');
            })
            .finally(() => {
                // Reset button state
                if (loadBtn) {
                    loadBtn.disabled = false;
                    loadBtn.innerHTML = '<i class="fas fa-folder-open"></i> Load';
                }
            });
        },

        // Clear model cache
        clearModelCache() {
            CoreModule.showConfirmModal(
                'Clear Model Cache',
                'This will clear all cached model data and force a refresh from the server. Continue?',
                () => {
                    // Show loading state
                    const clearBtn = document.querySelector('[onclick="clearModelCache()"]');
                    if (clearBtn) {
                        clearBtn.disabled = true;
                        clearBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Clearing...';
                    }

                    // Call API to clear cache
                    fetch('/api/clear_model_cache', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            // Clear local cache
                            AppState.availableModels = [];
                            AppState.configCache = {};

                            // Clear localStorage cache
                            localStorage.removeItem('modelCache');
                            localStorage.removeItem('modelCacheTimestamp');

                            // Reload models from server
                            this.loadAvailableModels();

                            CoreModule.showAlert('Model cache cleared successfully', 'success');
                        } else {
                            throw new Error(data.error || 'Failed to clear cache');
                        }
                    })
                    .catch(error => {
                        console.error('Failed to clear model cache:', error);
                        CoreModule.showAlert(`Failed to clear cache: ${error.message}`, 'danger');
                    })
                    .finally(() => {
                        // Reset button state
                        if (clearBtn) {
                            clearBtn.disabled = false;
                            clearBtn.innerHTML = '<i class="fas fa-trash-alt"></i> Clear Cache';
                        }
                    });
                },
                'btn-warning'
            );
        }
    };

    // Export to window
    window.ModelsModule = ModelsModule;

    // Export functions for onclick handlers
    window.applyRecommendedSettings = () => ModelsModule.applyRecommendedSettings();
    window.updateModelList = () => ModelsModule.updateModelList();

    // Update configuration summary display
    window.updateConfigSummary = () => {
        // Get current configuration values from form fields
        const modelNameEl = document.getElementById('model-name');
        const datasetPathEl = document.getElementById('dataset-path');
        const numEpochsEl = document.getElementById('num-epochs');
        const batchSizeEl = document.getElementById('batch-size');

        // Get the selected model name
        let modelName = '--';
        if (modelNameEl && modelNameEl.value) {
            // Get the selected option's text (which shows the model name)
            const selectedOption = modelNameEl.options[modelNameEl.selectedIndex];
            modelName = selectedOption ? selectedOption.text : modelNameEl.value;
        }

        // Get dataset name (extract filename only)
        let datasetName = '--';
        if (datasetPathEl && datasetPathEl.value) {
            const fullPath = datasetPathEl.value;
            // Extract just the filename
            datasetName = fullPath.split('/').pop().split('\\').pop();
            // Truncate if too long
            if (datasetName.length > 40) {
                datasetName = datasetName.substring(0, 37) + '...';
            }
        }

        // Get training parameters
        const epochs = numEpochsEl?.value || '--';
        const batchSize = batchSizeEl?.value || '--';

        // Calculate estimated time and VRAM based on config
        let estimatedTime = '~15 minutes';
        let estimatedVRAM = '~4GB';

        // Simple estimation based on epochs and batch size
        if (epochs !== '--' && batchSize !== '--') {
            const timePerEpoch = 5; // baseline minutes per epoch
            const totalMinutes = parseInt(epochs) * timePerEpoch;
            estimatedTime = totalMinutes < 60
                ? `~${totalMinutes} minutes`
                : `~${Math.round(totalMinutes / 60)} hours`;

            // VRAM scales with batch size
            const baseVRAM = 3;
            const vramGB = baseVRAM + (parseInt(batchSize) - 1) * 0.5;
            estimatedVRAM = `~${Math.ceil(vramGB)}GB`;
        }

        // Update summary fields
        const summaryModel = document.getElementById('summary-model');
        const summaryDataset = document.getElementById('summary-dataset');
        const summaryEpochs = document.getElementById('summary-epochs');
        const summaryBatch = document.getElementById('summary-batch');
        const summaryTime = document.getElementById('summary-time');
        const summaryVRAM = document.getElementById('summary-vram');

        if (summaryModel) summaryModel.textContent = modelName;
        if (summaryDataset) summaryDataset.textContent = datasetName;
        if (summaryEpochs) summaryEpochs.textContent = epochs;
        if (summaryBatch) summaryBatch.textContent = batchSize;
        if (summaryTime) summaryTime.textContent = estimatedTime;
        if (summaryVRAM) summaryVRAM.textContent = estimatedVRAM;
    };

})(window);

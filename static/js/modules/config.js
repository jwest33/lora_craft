// ============================================================================
// Configuration Management Module
// ============================================================================

(function(window) {
    'use strict';

    const ConfigModule = {
        // Configuration state
        savedConfigs: [],
        currentConfig: null,

        // Initialize the module
        init() {
            this.setupEventListeners();
            this.loadConfigList();
        },

        // Setup configuration-related event listeners
        setupEventListeners() {
            // Save configuration button
            const saveConfigBtn = document.getElementById('save-config-btn');
            if (saveConfigBtn) {
                saveConfigBtn.addEventListener('click', () => this.saveCurrentConfig());
            }

            // Load configuration button
            const loadConfigBtn = document.getElementById('load-config-btn');
            if (loadConfigBtn) {
                loadConfigBtn.addEventListener('click', () => this.showLoadConfigDialog());
            }

            // Export configuration button
            const exportConfigBtn = document.getElementById('export-config-btn');
            if (exportConfigBtn) {
                exportConfigBtn.addEventListener('click', () => this.exportConfig());
            }

            // Import configuration button
            const importConfigBtn = document.getElementById('import-config-btn');
            if (importConfigBtn) {
                importConfigBtn.addEventListener('click', () => this.showImportDialog());
            }
        },

        // Load saved configurations list
        loadConfigList() {
            fetch('/api/configurations')
                .then(response => response.json())
                .then(data => {
                    this.savedConfigs = data.configurations || [];
                    this.updateConfigDropdown();
                })
                .catch(error => {
                    console.error('Failed to load configurations:', error);
                });
        },

        // Update configuration dropdown
        updateConfigDropdown() {
            // Try both possible IDs for compatibility
            const configSelect = document.getElementById('saved-configs-list') || document.getElementById('saved-configs');
            if (!configSelect) return;

            configSelect.innerHTML = '<option value="">Select a configuration...</option>';

            this.savedConfigs.forEach(config => {
                const option = document.createElement('option');
                option.value = config.id;
                option.textContent = config.name;
                option.setAttribute('data-description', config.description || '');
                configSelect.appendChild(option);
            });
        },

        // Save current configuration
        saveCurrentConfig() {
            const config = this.gatherCurrentConfig();

            // Show save dialog
            this.showSaveConfigDialog(config);
        },

        // Show save configuration dialog
        showSaveConfigDialog(config) {
            const modalId = 'saveConfigModal';
            let modal = document.getElementById(modalId);

            if (!modal) {
                const modalHtml = `
                    <div class="modal fade" id="${modalId}" tabindex="-1">
                        <div class="modal-dialog">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">Save Configuration</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body">
                                    <div class="mb-3">
                                        <label for="config-name" class="form-label">Configuration Name</label>
                                        <input type="text" class="form-control" id="config-name" placeholder="My Training Config">
                                    </div>
                                    <div class="mb-3">
                                        <label for="config-description" class="form-label">Description</label>
                                        <textarea class="form-control" id="config-description" rows="3" placeholder="Description of this configuration..."></textarea>
                                    </div>
                                </div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                                    <button type="button" class="btn btn-primary" onclick="ConfigModule.confirmSaveConfig()">Save</button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
                modal = document.getElementById(modalId);
            }

            // Store config for saving
            this.pendingConfig = config;

            // Show modal
            new bootstrap.Modal(modal).show();
        },

        // Confirm save configuration
        confirmSaveConfig() {
            const name = document.getElementById('config-name')?.value;
            const description = document.getElementById('config-description')?.value;

            if (!name) {
                CoreModule.showAlert('Please enter a configuration name', 'warning');
                return;
            }

            const configData = {
                name: name,
                description: description,
                config: this.pendingConfig,
                timestamp: Date.now()
            };

            // Save to server
            fetch('/api/save_configuration', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(configData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    CoreModule.showAlert('Configuration saved successfully', 'success');
                    this.loadConfigList();

                    // Close modal
                    const modal = bootstrap.Modal.getInstance(document.getElementById('saveConfigModal'));
                    if (modal) modal.hide();
                } else {
                    throw new Error(data.error || 'Failed to save configuration');
                }
            })
            .catch(error => {
                console.error('Failed to save configuration:', error);
                CoreModule.showAlert(`Failed to save configuration: ${error.message}`, 'danger');
            });
        },

        // Show load configuration dialog
        showLoadConfigDialog() {
            const modalId = 'loadConfigModal';
            let modal = document.getElementById(modalId);

            if (!modal) {
                const modalHtml = `
                    <div class="modal fade" id="${modalId}" tabindex="-1">
                        <div class="modal-dialog modal-lg">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">Load Configuration</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body">
                                    <div class="list-group" id="config-list"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
                modal = document.getElementById(modalId);
            }

            // Populate configuration list
            const configList = modal.querySelector('#config-list');
            configList.innerHTML = '';

            if (this.savedConfigs.length === 0) {
                configList.innerHTML = '<p class="text-muted">No saved configurations found</p>';
            } else {
                this.savedConfigs.forEach(config => {
                    const item = document.createElement('button');
                    item.className = 'list-group-item list-group-item-action';
                    item.innerHTML = `
                        <div class="d-flex justify-content-between align-items-start">
                            <div>
                                <h6 class="mb-1">${CoreModule.escapeHtml(config.name)}</h6>
                                ${config.description ? `<p class="mb-1 small text-muted">${CoreModule.escapeHtml(config.description)}</p>` : ''}
                                <small class="text-muted">Saved: ${new Date(config.timestamp).toLocaleString()}</small>
                            </div>
                            <div class="btn-group">
                                <button class="btn btn-sm btn-primary" onclick="ConfigModule.loadConfig('${config.id}'); return false;">
                                    <i class="fas fa-upload"></i> Load
                                </button>
                                <button class="btn btn-sm btn-danger" onclick="ConfigModule.deleteConfig('${config.id}'); return false;">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </div>
                    `;
                    configList.appendChild(item);
                });
            }

            // Show modal
            new bootstrap.Modal(modal).show();
        },

        // Load configuration
        loadConfig(configId) {
            fetch(`/api/configuration/${configId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        this.applyConfiguration(data.configuration);

                        // Close modal
                        const modal = bootstrap.Modal.getInstance(document.getElementById('loadConfigModal'));
                        if (modal) modal.hide();

                        CoreModule.showAlert('Configuration loaded successfully', 'success');
                    } else {
                        throw new Error(data.error || 'Failed to load configuration');
                    }
                })
                .catch(error => {
                    console.error('Failed to load configuration:', error);
                    CoreModule.showAlert(`Failed to load configuration: ${error.message}`, 'danger');
                });
        },

        // Delete configuration
        deleteConfig(configId) {
            CoreModule.showConfirmModal(
                'Delete Configuration',
                'Are you sure you want to delete this configuration?',
                () => {
                    fetch(`/api/configuration/${configId}`, {
                        method: 'DELETE'
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            CoreModule.showAlert('Configuration deleted', 'success');
                            this.loadConfigList();
                            // Refresh the list in modal if open
                            this.showLoadConfigDialog();
                        } else {
                            throw new Error(data.error || 'Failed to delete configuration');
                        }
                    })
                    .catch(error => {
                        console.error('Failed to delete configuration:', error);
                        CoreModule.showAlert(`Failed to delete configuration: ${error.message}`, 'danger');
                    });
                }
            );
        },

        // Gather current configuration
        gatherCurrentConfig() {
            return {
                model: ModelsModule.exportModelConfig(),
                dataset: {
                    type: document.getElementById('dataset-type')?.value,
                    path: document.getElementById('dataset-path')?.value,
                    sample_size: parseInt(document.getElementById('sample-size')?.value) || 0,
                    train_split: parseInt(document.getElementById('train-split')?.value) || 80
                },
                template: TemplatesModule.exportTemplateConfig(),
                training: {
                    num_epochs: parseInt(document.getElementById('num-epochs')?.value) || 1,
                    batch_size: parseInt(document.getElementById('batch-size')?.value) || 4,
                    gradient_accumulation: parseInt(document.getElementById('gradient-accumulation')?.value) || 1,
                    learning_rate: parseFloat(document.getElementById('learning-rate')?.value) || 2e-4,
                    lr_schedule: document.getElementById('lr-schedule')?.value || 'cosine',
                    warmup_ratio: parseFloat(document.getElementById('warmup-ratio')?.value) || 0.1,
                    weight_decay: parseFloat(document.getElementById('weight-decay')?.value) || 0.01,
                    max_grad_norm: parseFloat(document.getElementById('max-grad-norm')?.value) || 1.0,
                    seed: parseInt(document.getElementById('seed')?.value) || 42
                },
                grpo: {
                    num_generations: parseInt(document.getElementById('num-generations')?.value) || 2,
                    kl_weight: parseFloat(document.getElementById('kl-weight')?.value) || 0.1,
                    clip_range: parseFloat(document.getElementById('clip-range')?.value) || 0.2
                },
                output: {
                    name: document.getElementById('output-name')?.value || '',
                    save_steps: parseInt(document.getElementById('save-steps')?.value) || 100,
                    eval_steps: parseInt(document.getElementById('eval-steps')?.value) || 100
                }
            };
        },

        // Apply configuration
        applyConfiguration(config) {
            // Apply model configuration
            if (config.model) {
                ModelsModule.importModelConfig(config.model);
            }

            // Apply dataset configuration
            if (config.dataset) {
                if (config.dataset.type) {
                    document.getElementById('dataset-type').value = config.dataset.type;
                    DatasetModule.onDatasetTypeChange();
                }
                if (config.dataset.path) {
                    document.getElementById('dataset-path').value = config.dataset.path;
                }
                if (config.dataset.sample_size !== undefined) {
                    document.getElementById('sample-size').value = config.dataset.sample_size;
                }
                if (config.dataset.train_split !== undefined) {
                    document.getElementById('train-split').value = config.dataset.train_split;
                    DatasetModule.updateSplitDisplay();
                }
            }

            // Apply template configuration
            if (config.template) {
                TemplatesModule.importTemplateConfig(config.template);
            }

            // Apply training configuration
            if (config.training) {
                const trainingFields = [
                    'num-epochs', 'batch-size', 'gradient-accumulation',
                    'learning-rate', 'lr-schedule', 'warmup-ratio',
                    'weight-decay', 'max-grad-norm', 'seed'
                ];

                trainingFields.forEach(field => {
                    const key = field.replace(/-/g, '_');
                    if (config.training[key] !== undefined) {
                        const element = document.getElementById(field);
                        if (element) element.value = config.training[key];
                    }
                });
            }

            // Apply GRPO configuration
            if (config.grpo) {
                if (config.grpo.num_generations !== undefined) {
                    document.getElementById('num-generations').value = config.grpo.num_generations;
                }
                if (config.grpo.kl_weight !== undefined) {
                    document.getElementById('kl-weight').value = config.grpo.kl_weight;
                }
                if (config.grpo.clip_range !== undefined) {
                    document.getElementById('clip-range').value = config.grpo.clip_range;
                }
            }

            // Apply output configuration
            if (config.output) {
                if (config.output.name) {
                    document.getElementById('output-name').value = config.output.name;
                }
                if (config.output.save_steps !== undefined) {
                    document.getElementById('save-steps').value = config.output.save_steps;
                }
                if (config.output.eval_steps !== undefined) {
                    document.getElementById('eval-steps').value = config.output.eval_steps;
                }
            }

            // Validate all steps
            for (let i = 1; i <= 3; i++) {
                NavigationModule.validateStep(i);
            }
        },

        // Export configuration to file
        exportConfig() {
            const config = this.gatherCurrentConfig();
            const configData = {
                name: 'Exported Configuration',
                timestamp: Date.now(),
                version: '1.0.0',
                config: config
            };

            const dataStr = JSON.stringify(configData, null, 2);
            const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);

            const exportLink = document.createElement('a');
            exportLink.setAttribute('href', dataUri);
            exportLink.setAttribute('download', `config_${Date.now()}.json`);
            document.body.appendChild(exportLink);
            exportLink.click();
            document.body.removeChild(exportLink);

            CoreModule.showAlert('Configuration exported successfully', 'success');
        },

        // Show import dialog
        showImportDialog() {
            const modalId = 'importConfigModal';
            let modal = document.getElementById(modalId);

            if (!modal) {
                const modalHtml = `
                    <div class="modal fade" id="${modalId}" tabindex="-1">
                        <div class="modal-dialog">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">Import Configuration</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body">
                                    <div class="mb-3">
                                        <label for="import-file" class="form-label">Select Configuration File</label>
                                        <input type="file" class="form-control" id="import-file" accept=".json">
                                    </div>
                                    <div id="import-preview"></div>
                                </div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                                    <button type="button" class="btn btn-primary" onclick="ConfigModule.importConfigFile()">Import</button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
                modal = document.getElementById(modalId);

                // Setup file input listener
                document.getElementById('import-file').addEventListener('change', (e) => {
                    this.previewImportFile(e.target.files[0]);
                });
            }

            // Show modal
            new bootstrap.Modal(modal).show();
        },

        // Preview import file
        previewImportFile(file) {
            if (!file) return;

            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const data = JSON.parse(e.target.result);
                    this.pendingImport = data;

                    const preview = document.getElementById('import-preview');
                    if (preview) {
                        preview.innerHTML = `
                            <div class="alert alert-info">
                                <strong>Configuration Name:</strong> ${CoreModule.escapeHtml(data.name || 'Unnamed')}<br>
                                <strong>Created:</strong> ${data.timestamp ? new Date(data.timestamp).toLocaleString() : 'Unknown'}<br>
                                <strong>Version:</strong> ${CoreModule.escapeHtml(data.version || 'Unknown')}
                            </div>
                        `;
                    }
                } catch (error) {
                    CoreModule.showAlert('Invalid configuration file', 'danger');
                }
            };
            reader.readAsText(file);
        },

        // Import configuration file
        importConfigFile() {
            if (!this.pendingImport) {
                CoreModule.showAlert('Please select a file first', 'warning');
                return;
            }

            this.applyConfiguration(this.pendingImport.config);

            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('importConfigModal'));
            if (modal) modal.hide();

            CoreModule.showAlert('Configuration imported successfully', 'success');
        },

        // Apply preset configuration
        applyPreset(preset) {
            // Remove active class from all preset cards
            const presetCards = document.querySelectorAll('.training-presets .selection-card');
            presetCards.forEach(card => card.classList.remove('active'));

            // Add active class to selected preset
            const selectedCard = document.getElementById(`preset-${preset}`);
            if (selectedCard) {
                selectedCard.classList.add('active');
            }

            // Define preset configurations
            const presets = {
                recommended: {
                    num_epochs: 3,
                    batch_size: 4,
                    gradient_accumulation: 1,
                    learning_rate: 0.0002,
                    warmup_steps: 10,
                    weight_decay: 0.001,
                    max_grad_norm: 1.0,
                    kl_weight: 0.1,
                    clip_range: 0.2,
                    message: 'Auto-optimized settings based on your configuration'
                },
                fast: {
                    num_epochs: 1,
                    batch_size: 8,
                    gradient_accumulation: 1,
                    learning_rate: 0.0005,
                    warmup_steps: 5,
                    weight_decay: 0.0001,
                    max_grad_norm: 1.0,
                    kl_weight: 0.05,
                    clip_range: 0.2,
                    message: 'Fast training for quick testing and experimentation'
                },
                balanced: {
                    num_epochs: 3,
                    batch_size: 4,
                    gradient_accumulation: 2,
                    learning_rate: 0.0002,
                    warmup_steps: 10,
                    weight_decay: 0.001,
                    max_grad_norm: 1.0,
                    kl_weight: 0.1,
                    clip_range: 0.2,
                    message: 'Balanced settings for general-purpose training'
                },
                quality: {
                    num_epochs: 5,
                    batch_size: 2,
                    gradient_accumulation: 4,
                    learning_rate: 0.0001,
                    warmup_steps: 20,
                    weight_decay: 0.01,
                    max_grad_norm: 0.5,
                    kl_weight: 0.15,
                    clip_range: 0.15,
                    message: 'High-quality training for best results (slower)'
                }
            };

            const config = presets[preset];
            if (!config) {
                CoreModule.showAlert('Unknown preset', 'warning');
                return;
            }

            // Apply settings to form
            if (document.getElementById('num-epochs')) {
                document.getElementById('num-epochs').value = config.num_epochs;
            }
            if (document.getElementById('batch-size')) {
                document.getElementById('batch-size').value = config.batch_size;
            }
            if (document.getElementById('gradient-accumulation')) {
                document.getElementById('gradient-accumulation').value = config.gradient_accumulation;
            }
            if (document.getElementById('learning-rate')) {
                document.getElementById('learning-rate').value = config.learning_rate;
            }
            if (document.getElementById('warmup-steps')) {
                document.getElementById('warmup-steps').value = config.warmup_steps;
            }
            if (document.getElementById('weight-decay')) {
                document.getElementById('weight-decay').value = config.weight_decay;
            }
            if (document.getElementById('max-grad-norm')) {
                document.getElementById('max-grad-norm').value = config.max_grad_norm;
            }
            if (document.getElementById('kl-weight')) {
                document.getElementById('kl-weight').value = config.kl_weight;
            }
            if (document.getElementById('clip-range')) {
                document.getElementById('clip-range').value = config.clip_range;
            }

            // Show preset message
            const presetIndicator = document.getElementById('preset-indicator');
            const presetMessage = document.getElementById('preset-message');
            if (presetIndicator && presetMessage) {
                presetMessage.textContent = config.message;
                presetIndicator.style.display = 'block';
            }

            // Store preset selection
            AppState.setConfigValue('selectedPreset', preset);

            CoreModule.showAlert(`${preset.charAt(0).toUpperCase() + preset.slice(1)} preset applied`, 'success');
        },

        // Select training algorithm
        selectAlgorithm(algorithm) {
            // Remove active class from all algorithm cards
            const algoCards = document.querySelectorAll('.algorithm-selection .selection-card');
            algoCards.forEach(card => card.classList.remove('active'));

            // Add active class to selected algorithm
            const selectedCard = document.getElementById(`algo-${algorithm}`);
            if (selectedCard) {
                selectedCard.classList.add('active');
            }

            // Define algorithm information
            const algorithmInfo = {
                grpo: {
                    title: 'GRPO',
                    description: 'Standard Group Relative Policy Optimization applies importance weights at the token level.',
                    showGspoParams: false
                },
                gspo: {
                    title: 'GSPO',
                    description: 'Group Sequence Policy Optimization applies importance weights at the sequence level, with epsilon bounds for stability.',
                    showGspoParams: true
                },
                dr_grpo: {
                    title: 'DR-GRPO',
                    description: 'Doubly Robust GRPO combines token-level optimization with variance reduction techniques for more stable training.',
                    showGspoParams: false
                }
            };

            const info = algorithmInfo[algorithm];
            if (!info) {
                CoreModule.showAlert('Unknown algorithm', 'warning');
                return;
            }

            // Update algorithm info display
            const algoInfoElement = document.getElementById('algorithm-info');
            if (algoInfoElement) {
                algoInfoElement.innerHTML = `
                    <i class="fas fa-info-circle"></i>
                    <strong>${info.title}:</strong> ${info.description}
                `;
            }

            // Show/hide GSPO-specific parameters
            const gspoParams = document.getElementById('gspo-params');
            if (gspoParams) {
                gspoParams.style.display = info.showGspoParams ? 'block' : 'none';
            }

            // Store algorithm selection
            AppState.setConfigValue('selectedAlgorithm', algorithm);

            CoreModule.showAlert(`${info.title} algorithm selected`, 'success');
        },

        // Handle config selection from dropdown
        onConfigSelect() {
            const configSelect = document.getElementById('saved-configs-list');
            if (!configSelect) return;

            const selectedConfigId = configSelect.value;
            if (selectedConfigId) {
                AppState.setConfigValue('selectedConfigId', selectedConfigId);
            }
        },

        // Save current configuration (sidebar button)
        saveConfig() {
            this.saveCurrentConfig();
        },

        // Load selected configuration (sidebar button)
        loadSelectedConfig() {
            const configSelect = document.getElementById('saved-configs-list');
            if (!configSelect) {
                CoreModule.showAlert('Configuration selector not found', 'danger');
                return;
            }

            const selectedConfigId = configSelect.value;
            if (!selectedConfigId) {
                CoreModule.showAlert('Please select a configuration first', 'warning');
                return;
            }

            this.loadConfig(selectedConfigId);
        },

        // Delete selected configuration (sidebar button)
        deleteSelectedConfig() {
            const configSelect = document.getElementById('saved-configs-list');
            if (!configSelect) {
                CoreModule.showAlert('Configuration selector not found', 'danger');
                return;
            }

            const selectedConfigId = configSelect.value;
            if (!selectedConfigId) {
                CoreModule.showAlert('Please select a configuration first', 'warning');
                return;
            }

            CoreModule.showConfirmModal(
                'Delete Configuration',
                'Are you sure you want to delete this configuration?',
                () => {
                    fetch(`/api/configuration/${selectedConfigId}`, {
                        method: 'DELETE'
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            CoreModule.showAlert('Configuration deleted', 'success');
                            this.loadConfigList();
                            // Reset dropdown
                            configSelect.value = '';
                        } else {
                            throw new Error(data.error || 'Failed to delete configuration');
                        }
                    })
                    .catch(error => {
                        console.error('Failed to delete configuration:', error);
                        CoreModule.showAlert(`Failed to delete configuration: ${error.message}`, 'danger');
                    });
                }
            );
        },

        // Show reward help modal
        showRewardHelp() {
            const modalId = 'rewardHelpModal';
            let modal = document.getElementById(modalId);

            if (!modal) {
                const modalHtml = `
                    <div class="modal fade" id="${modalId}" tabindex="-1">
                        <div class="modal-dialog modal-lg">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">
                                        <i class="fas fa-trophy text-warning"></i> Reward Function Help
                                    </h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body">
                                    <h6><i class="fas fa-info-circle"></i> What is a Reward Function?</h6>
                                    <p>A reward function evaluates the quality of generated responses during training. It assigns scores that guide the model to produce better outputs.</p>

                                    <h6 class="mt-3"><i class="fas fa-rocket"></i> Quick Start</h6>
                                    <p>Choose from pre-built templates optimized for common tasks like math problems, code generation, or question answering.</p>

                                    <h6 class="mt-3"><i class="fas fa-list-alt"></i> Preset Library</h6>
                                    <p>Browse a collection of reward functions organized by category. Each preset is tuned for specific use cases.</p>

                                    <h6 class="mt-3"><i class="fas fa-tools"></i> Custom Builder</h6>
                                    <p>Build your own reward function by combining multiple components. Each component can evaluate different aspects like accuracy, format, length, etc.</p>

                                    <h6 class="mt-3"><i class="fas fa-flask"></i> Test Reward</h6>
                                    <p>Test your reward function with sample inputs to see how it scores different responses before training.</p>

                                    <div class="alert alert-info mt-3">
                                        <i class="fas fa-lightbulb"></i> <strong>Tip:</strong> Start with a template and customize it if needed. Most tasks work well with the default templates.
                                    </div>
                                </div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-primary" data-bs-dismiss="modal">Got it!</button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
                modal = document.getElementById(modalId);
            }

            // Show modal
            new bootstrap.Modal(modal).show();
        },

        // Select reward template
        selectTemplate(template) {
            // Define template configurations
            const templates = {
                math_problem_solving: {
                    name: 'Mathematical Problem Solving',
                    description: 'Rewards accurate numerical answers with proper formatting (e.g., \\boxed{} notation)',
                    config: {
                        type: 'template',
                        template: 'math_problem_solving',
                        components: [
                            { type: 'accuracy', weight: 0.6 },
                            { type: 'format', weight: 0.3, pattern: '\\\\boxed' },
                            { type: 'completeness', weight: 0.1 }
                        ]
                    }
                },
                code_generation: {
                    name: 'Code Generation',
                    description: 'Generate clean, well-documented code with proper formatting',
                    config: {
                        type: 'template',
                        template: 'code_generation',
                        components: [
                            { type: 'syntax', weight: 0.4 },
                            { type: 'functionality', weight: 0.4 },
                            { type: 'style', weight: 0.2 }
                        ]
                    }
                },
                question_answering: {
                    name: 'Question Answering',
                    description: 'Train models to provide accurate, concise answers to questions',
                    config: {
                        type: 'template',
                        template: 'question_answering',
                        components: [
                            { type: 'relevance', weight: 0.5 },
                            { type: 'accuracy', weight: 0.3 },
                            { type: 'conciseness', weight: 0.2 }
                        ]
                    }
                }
            };

            const selectedTemplate = templates[template];
            if (!selectedTemplate) {
                CoreModule.showAlert('Unknown template', 'warning');
                return;
            }

            // Update selected reward display
            const rewardNameEl = document.getElementById('selected-reward-name');
            const rewardDescEl = document.getElementById('selected-reward-description');

            if (rewardNameEl) {
                rewardNameEl.textContent = selectedTemplate.name;
            }
            if (rewardDescEl) {
                rewardDescEl.textContent = selectedTemplate.description;
            }

            // Highlight selected template card
            const templateCards = document.querySelectorAll('.template-card');
            templateCards.forEach(card => card.classList.remove('border-primary'));

            const selectedCard = document.querySelector(`[onclick*="selectTemplate('${template}')"]`);
            if (selectedCard) {
                selectedCard.classList.add('border-primary', 'border-2');
            }

            // Store template selection
            AppState.setConfigValue('selectedRewardTemplate', template);
            AppState.setConfigValue('rewardConfig', selectedTemplate.config);

            CoreModule.showAlert(`Template "${selectedTemplate.name}" selected`, 'success');
        },

        // Add advanced component to reward function
        addAdvancedComponent() {
            const componentsContainer = document.getElementById('reward-components-advanced');
            if (!componentsContainer) {
                CoreModule.showAlert('Components container not found', 'danger');
                return;
            }

            const componentId = 'component-' + Date.now();
            const componentHtml = `
                <div class="reward-component-card mb-3" id="${componentId}">
                    <div class="card">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-center mb-3">
                                <h6 class="mb-0">Reward Component</h6>
                                <button class="btn btn-sm btn-outline-danger" onclick="document.getElementById('${componentId}').remove()">
                                    <i class="fas fa-times"></i>
                                </button>
                            </div>
                            <div class="row">
                                <div class="col-md-6">
                                    <label class="form-label">Component Type</label>
                                    <select class="form-select component-type">
                                        <option value="accuracy">Accuracy</option>
                                        <option value="format">Format Matching</option>
                                        <option value="length">Length Control</option>
                                        <option value="completeness">Completeness</option>
                                        <option value="relevance">Relevance</option>
                                        <option value="style">Code Style</option>
                                        <option value="syntax">Syntax Validation</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Weight (0-1)</label>
                                    <input type="number" class="form-control component-weight" value="0.5" min="0" max="1" step="0.1">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            componentsContainer.insertAdjacentHTML('beforeend', componentHtml);
            CoreModule.showAlert('Component added', 'success');
        },

        // Save custom reward function
        saveCustomReward() {
            const componentsContainer = document.getElementById('reward-components-advanced');
            if (!componentsContainer) {
                CoreModule.showAlert('Components container not found', 'danger');
                return;
            }

            const components = [];
            const componentCards = componentsContainer.querySelectorAll('.reward-component-card');

            componentCards.forEach(card => {
                const type = card.querySelector('.component-type')?.value;
                const weight = parseFloat(card.querySelector('.component-weight')?.value);

                if (type && !isNaN(weight)) {
                    components.push({ type, weight });
                }
            });

            if (components.length === 0) {
                CoreModule.showAlert('Please add at least one component', 'warning');
                return;
            }

            // Validate weights sum
            const totalWeight = components.reduce((sum, comp) => sum + comp.weight, 0);
            if (Math.abs(totalWeight - 1.0) > 0.01) {
                CoreModule.showAlert(`Warning: Component weights sum to ${totalWeight.toFixed(2)}, consider normalizing to 1.0`, 'warning');
            }

            const rewardConfig = {
                type: 'custom',
                components: components
            };

            // Store custom reward configuration
            AppState.setConfigValue('rewardConfig', rewardConfig);

            // Update selected reward display
            const rewardNameEl = document.getElementById('selected-reward-name');
            const rewardDescEl = document.getElementById('selected-reward-description');

            if (rewardNameEl) {
                rewardNameEl.textContent = 'Custom Reward Function';
            }
            if (rewardDescEl) {
                rewardDescEl.textContent = `Custom configuration with ${components.length} component(s)`;
            }

            CoreModule.showAlert('Custom reward function saved', 'success');
        },

        // Test reward function
        testReward() {
            const instruction = document.getElementById('test-instruction')?.value;
            const generated = document.getElementById('test-generated')?.value;
            const reference = document.getElementById('test-reference')?.value;

            if (!instruction || !generated) {
                CoreModule.showAlert('Please provide instruction and generated response', 'warning');
                return;
            }

            const rewardConfig = AppState.getConfigValue('rewardConfig');
            if (!rewardConfig) {
                CoreModule.showAlert('Please select a reward function first', 'warning');
                return;
            }

            // Send test request to server
            fetch('/api/test_reward', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    instruction: instruction,
                    generated: generated,
                    reference: reference,
                    reward_config: rewardConfig
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.displayTestResults(data.results);
                } else {
                    throw new Error(data.error || 'Failed to test reward');
                }
            })
            .catch(error => {
                console.error('Failed to test reward:', error);
                CoreModule.showAlert(`Failed to test reward: ${error.message}`, 'danger');
            });
        },

        // Display test results
        displayTestResults(results) {
            const resultsContainer = document.getElementById('test-results');
            if (!resultsContainer) return;

            let html = '<div class="test-results-content">';
            html += `<div class="mb-3"><strong>Total Score:</strong> <span class="badge bg-primary">${results.total_score?.toFixed(4) || 'N/A'}</span></div>`;

            if (results.components && results.components.length > 0) {
                html += '<h6>Component Scores:</h6><ul class="list-group">';
                results.components.forEach(comp => {
                    html += `
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            ${comp.name || comp.type}
                            <span class="badge bg-info">${comp.score?.toFixed(4) || 'N/A'}</span>
                        </li>
                    `;
                });
                html += '</ul>';
            }

            html += '</div>';
            resultsContainer.innerHTML = html;

            CoreModule.showAlert('Reward test completed', 'success');
        },

        // View reward details
        viewRewardDetails() {
            const rewardConfig = AppState.getConfigValue('rewardConfig');
            if (!rewardConfig) {
                CoreModule.showAlert('No reward function selected', 'warning');
                return;
            }

            const modalId = 'rewardDetailsModal';
            let modal = document.getElementById(modalId);

            if (!modal) {
                const modalHtml = `
                    <div class="modal fade" id="${modalId}" tabindex="-1">
                        <div class="modal-dialog modal-lg">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">
                                        <i class="fas fa-info-circle"></i> Reward Function Details
                                    </h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body" id="reward-details-content">
                                    <!-- Content will be populated dynamically -->
                                </div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-primary" data-bs-dismiss="modal">Close</button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
                modal = document.getElementById(modalId);
            }

            // Populate content
            const content = document.getElementById('reward-details-content');
            if (content) {
                let html = `<h6>Configuration Type: ${rewardConfig.type || 'Unknown'}</h6>`;

                if (rewardConfig.template) {
                    html += `<p><strong>Template:</strong> ${rewardConfig.template}</p>`;
                }

                if (rewardConfig.components && rewardConfig.components.length > 0) {
                    html += '<h6 class="mt-3">Components:</h6><ul class="list-group">';
                    rewardConfig.components.forEach(comp => {
                        html += `
                            <li class="list-group-item">
                                <strong>${comp.type}</strong> - Weight: ${comp.weight}
                                ${comp.pattern ? `<br><small>Pattern: ${CoreModule.escapeHtml(comp.pattern)}</small>` : ''}
                            </li>
                        `;
                    });
                    html += '</ul>';
                }

                html += '<div class="mt-3"><pre class="bg-light p-3 rounded"><code>' +
                        CoreModule.escapeHtml(JSON.stringify(rewardConfig, null, 2)) +
                        '</code></pre></div>';

                content.innerHTML = html;
            }

            // Show modal
            new bootstrap.Modal(modal).show();
        },

        // Test selected reward
        testSelectedReward() {
            const rewardConfig = AppState.getConfigValue('rewardConfig');
            if (!rewardConfig) {
                CoreModule.showAlert('No reward function selected', 'warning');
                return;
            }

            // Switch to test tab
            const testTab = document.querySelector('[data-bs-target="#reward-test-tab"]');
            if (testTab) {
                testTab.click();
            }

            CoreModule.showAlert('Switched to test tab. Enter sample data to test the reward function.', 'info');
        }
    };

    // Export to window
    window.ConfigModule = ConfigModule;

    // Export functions for compatibility layer
    window.loadConfigListLegacy = () => ConfigModule.loadConfigList();

})(window);
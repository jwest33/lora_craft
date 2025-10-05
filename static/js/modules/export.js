// ============================================================================
// Export Module - Model Export and Conversion
// ============================================================================

(function(window) {
    'use strict';

    const ExportModule = {
        // Export state
        availableFormats: ['safetensors', 'gguf', 'pytorch', 'onnx'],
        exportInProgress: false,

        // Initialize the module
        init() {
            this.setupEventListeners();
            this.loadExportableModels();
            this.displayTrainedModelCards();
        },

        // Setup export-related event listeners
        setupEventListeners() {
            // Export format change
            const formatSelect = document.getElementById('export-format');
            if (formatSelect) {
                formatSelect.addEventListener('change', () => this.onFormatChange());
            }

            // Quantization toggle
            const quantizeCheckbox = document.getElementById('enable-quantization');
            if (quantizeCheckbox) {
                quantizeCheckbox.addEventListener('change', () => this.toggleQuantizationOptions());
            }

            // Model selection for export
            const modelSelect = document.getElementById('export-model-select');
            if (modelSelect) {
                modelSelect.addEventListener('change', () => this.onModelSelectChange());
            }
        },

        // Load exportable models
        loadExportableModels() {
            const modelSelect = document.getElementById('export-model-select');
            if (!modelSelect) return;

            // Clear existing options
            modelSelect.innerHTML = '<option value="">Select a trained model...</option>';

            // Add trained models
            AppState.trainedModels.forEach((model, index) => {
                const option = document.createElement('option');
                option.value = index;
                const modelName = model.name || model.model_name || 'Unknown Model';
                const timestamp = model.timestamp || model.modified_at || model.created_at;
                option.textContent = `${modelName} (${new Date(timestamp).toLocaleString()})`;
                modelSelect.appendChild(option);
            });

            // Check for models from server
            fetch('/api/trained_models')
                .then(response => response.json())
                .then(data => {
                    if (data.models && data.models.length > 0) {
                        const optgroup = document.createElement('optgroup');
                        optgroup.label = 'Server Models';

                        data.models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = `server:${model.path}`;
                            // Use model_name or session_id as fallback
                            const modelName = model.model_name || model.session_id || 'Unknown';
                            const timestamp = model.modified_at || model.created_at;
                            option.textContent = `${modelName} (${new Date(timestamp).toLocaleString()})`;
                            optgroup.appendChild(option);
                        });

                        modelSelect.appendChild(optgroup);
                    }
                })
                .catch(error => {
                    console.error('Failed to load server models:', error);
                });
        },

        // Display trained model cards in the export view
        displayTrainedModelCards() {
            const modelsListContainer = document.getElementById('trained-models-list');
            if (!modelsListContainer) return;

            // Show loading state
            modelsListContainer.innerHTML = '<div class="text-center text-muted p-4"><i class="fas fa-spinner fa-spin"></i> Loading models...</div>';

            // Fetch trained models from server
            fetch('/api/trained_models')
                .then(response => response.json())
                .then(data => {
                    if (data.models && data.models.length > 0) {
                        // Update AppState
                        AppState.trainedModels = data.models;

                        // Generate model cards HTML
                        const cardsHtml = data.models.map(model => {
                            // Get display name (user-entered name) as primary title
                            let displayName = model.display_name || model.session_id || 'Unnamed Model';

                            // Get base model name for meta details
                            let baseModelName = model.model_name;
                            if (!baseModelName || baseModelName === 'Unknown') {
                                baseModelName = model.training_config?.model?.modelName || null;
                            }

                            const createdDate = new Date(model.created_at || Date.now());
                            const completedDate = new Date(model.modified_at || Date.now());
                            const bestReward = model.best_reward !== null && model.best_reward !== undefined
                                ? model.best_reward.toFixed(4)
                                : 'N/A';

                            // Get epochs from various sources
                            const epochs = model.epochs ||
                                          model.epochs_trained ||
                                          model.training_config?.training?.num_epochs ||
                                          0;

                            // Extract dataset name if available
                            const datasetPath = model.training_config?.dataset_path || model.training_config?.dataset?.path;
                            let datasetName = '';
                            if (datasetPath) {
                                const pathParts = datasetPath.split('/');
                                datasetName = pathParts[pathParts.length - 1];
                                if (datasetName.length > 30) {
                                    datasetName = datasetName.substring(0, 27) + '...';
                                }
                            }

                            return `
                                <div class="card model-card mb-3">
                                    <div class="card-body">
                                        <div class="d-flex justify-content-between align-items-start mb-2">
                                            <div class="flex-grow-1">
                                                <h6 class="card-title mb-0">
                                                    <i class="fas fa-robot text-primary"></i>
                                                    ${CoreModule.escapeHtml(displayName)}
                                                </h6>
                                                ${datasetName ? `<small class="text-muted"><i class="fas fa-database"></i> ${CoreModule.escapeHtml(datasetName)}</small>` : ''}
                                            </div>
                                            <span class="badge bg-success">Completed</span>
                                        </div>

                                        <div class="model-meta text-muted small mb-3">
                                            ${baseModelName ? `<div><i class="fas fa-cube"></i> Base: ${CoreModule.escapeHtml(baseModelName)}</div>` : ''}
                                            <div><i class="fas fa-calendar"></i> Completed: ${completedDate.toLocaleString()}</div>
                                            <div><i class="fas fa-layer-group"></i> Epochs: ${epochs}</div>
                                            ${model.has_final_checkpoint ? '<div><i class="fas fa-check-circle text-success"></i> Final checkpoint saved</div>' : ''}
                                        </div>

                                        <div class="d-flex gap-2">
                                            <button class="btn btn-sm btn-primary flex-grow-1"
                                                    onclick="ExportModule.showExportDialog('${CoreModule.escapeHtml(model.session_id)}', '${CoreModule.escapeHtml(model.path)}')">
                                                <i class="fas fa-file-export"></i> Export
                                            </button>
                                            <button class="btn btn-sm btn-outline-secondary"
                                                    onclick="ExportModule.showModelDetails('${CoreModule.escapeHtml(model.session_id)}')">
                                                <i class="fas fa-info-circle"></i> Details
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            `;
                        }).join('');

                        modelsListContainer.innerHTML = cardsHtml;
                    } else {
                        // No models found
                        modelsListContainer.innerHTML = `
                            <div class="text-center text-muted p-4">
                                <i class="fas fa-inbox fa-3x mb-3 opacity-25"></i>
                                <p class="mb-0">No trained models available</p>
                                <small>Complete a training session to see models here</small>
                            </div>
                        `;
                    }
                })
                .catch(error => {
                    console.error('Failed to load trained models:', error);
                    modelsListContainer.innerHTML = `
                        <div class="text-center text-danger p-4">
                            <i class="fas fa-exclamation-triangle mb-2"></i>
                            <p class="mb-0">Failed to load models</p>
                            <small>${CoreModule.escapeHtml(error.message)}</small>
                            <div class="mt-2">
                                <button class="btn btn-sm btn-outline-primary" onclick="ExportModule.displayTrainedModelCards()">
                                    <i class="fas fa-redo"></i> Retry
                                </button>
                            </div>
                        </div>
                    `;
                });
        },

        // Handle format change
        onFormatChange() {
            const format = document.getElementById('export-format')?.value;
            this.updateFormatOptions(format);
        },

        // Update format-specific options
        updateFormatOptions(format) {
            const optionsContainer = document.getElementById('format-options');
            if (!optionsContainer) return;

            let optionsHtml = '';

            switch (format) {
                case 'gguf':
                    optionsHtml = `
                        <div class="mb-3">
                            <label class="form-label">GGUF Options</label>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="gguf-f16" checked>
                                <label class="form-check-label" for="gguf-f16">
                                    Use F16 precision
                                </label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="gguf-metadata">
                                <label class="form-check-label" for="gguf-metadata">
                                    Include training metadata
                                </label>
                            </div>
                        </div>
                    `;
                    break;

                case 'onnx':
                    optionsHtml = `
                        <div class="mb-3">
                            <label class="form-label">ONNX Options</label>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="onnx-optimize" checked>
                                <label class="form-check-label" for="onnx-optimize">
                                    Optimize for inference
                                </label>
                            </div>
                            <select class="form-select mt-2" id="onnx-opset">
                                <option value="14">ONNX OpSet 14</option>
                                <option value="15" selected>ONNX OpSet 15</option>
                                <option value="16">ONNX OpSet 16</option>
                            </select>
                        </div>
                    `;
                    break;

                case 'safetensors':
                    optionsHtml = `
                        <div class="mb-3">
                            <label class="form-label">SafeTensors Options</label>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="st-metadata" checked>
                                <label class="form-check-label" for="st-metadata">
                                    Include metadata
                                </label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="st-compress">
                                <label class="form-check-label" for="st-compress">
                                    Compress output
                                </label>
                            </div>
                        </div>
                    `;
                    break;

                case 'pytorch':
                    optionsHtml = `
                        <div class="mb-3">
                            <label class="form-label">PyTorch Options</label>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="pt-state-dict" checked>
                                <label class="form-check-label" for="pt-state-dict">
                                    Export state dict only
                                </label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="pt-trace">
                                <label class="form-check-label" for="pt-trace">
                                    Include traced model
                                </label>
                            </div>
                        </div>
                    `;
                    break;
            }

            optionsContainer.innerHTML = optionsHtml;
        },

        // Toggle quantization options
        toggleQuantizationOptions() {
            const enabled = document.getElementById('enable-quantization')?.checked;
            const quantOptions = document.getElementById('quantization-options');

            if (quantOptions) {
                quantOptions.style.display = enabled ? 'block' : 'none';
            }
        },

        // Handle model selection change
        onModelSelectChange() {
            const modelSelect = document.getElementById('export-model-select');
            if (!modelSelect) return;

            const selectedValue = modelSelect.value;
            if (!selectedValue) return;

            // Load model info
            if (selectedValue.startsWith('server:')) {
                const modelPath = selectedValue.substring(7);
                this.loadServerModelInfo(modelPath);
            } else {
                const modelIndex = parseInt(selectedValue);
                const model = AppState.trainedModels[modelIndex];
                if (model) {
                    this.displayModelInfo(model);
                }
            }
        },

        // Load server model information
        loadServerModelInfo(modelPath) {
            fetch('/api/model_info', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: modelPath })
            })
            .then(response => response.json())
            .then(data => {
                this.displayModelInfo(data);
            })
            .catch(error => {
                console.error('Failed to load model info:', error);
            });
        },

        // Display model information
        displayModelInfo(model) {
            const infoContainer = document.getElementById('export-model-info');
            if (!infoContainer) return;

            // Handle both old and new model structures
            const modelName = model.name || model.model_name || model.session_id || 'Unknown Model';
            const timestamp = model.timestamp || model.modified_at || model.created_at;
            const size = model.size || 0;

            // Build info HTML with available data
            let infoHtml = `
                <div class="card" style="background-color: var(--bg-card); border-color: var(--border-color);">
                    <div class="card-body">
                        <h6 class="card-title" style="color: var(--text-primary);">Model Information</h6>
                        <dl class="row mb-0" style="color: var(--text-primary);">
                            <dt class="col-sm-4">Name:</dt>
                            <dd class="col-sm-8">${CoreModule.escapeHtml(modelName)}</dd>

                            ${model.session_id ? `
                                <dt class="col-sm-4">Session ID:</dt>
                                <dd class="col-sm-8"><small>${CoreModule.escapeHtml(model.session_id)}</small></dd>
                            ` : ''}

                            ${size > 0 ? `
                                <dt class="col-sm-4">Size:</dt>
                                <dd class="col-sm-8">${this.formatFileSize(size)}</dd>
                            ` : ''}

                            ${timestamp ? `
                                <dt class="col-sm-4">Date:</dt>
                                <dd class="col-sm-8">${new Date(timestamp).toLocaleString()}</dd>
                            ` : ''}

                            ${model.epochs !== undefined && model.epochs !== null ? `
                                <dt class="col-sm-4">Epochs:</dt>
                                <dd class="col-sm-8">${model.epochs}</dd>
                            ` : ''}

                            ${model.metrics?.final_loss !== undefined ? `
                                <dt class="col-sm-4">Final Loss:</dt>
                                <dd class="col-sm-8">${model.metrics.final_loss.toFixed(4)}</dd>
                            ` : ''}

                            ${model.path ? `
                                <dt class="col-sm-4">Path:</dt>
                                <dd class="col-sm-8"><small class="text-muted">${CoreModule.escapeHtml(model.path)}</small></dd>
                            ` : ''}
                        </dl>
                    </div>
                </div>
            `;

            infoContainer.innerHTML = infoHtml;
            infoContainer.style.display = 'block';
        },

        // Start export process
        startExport() {
            if (this.exportInProgress) {
                CoreModule.showAlert('Export already in progress', 'warning');
                return;
            }

            const modelSelect = document.getElementById('export-model-select');
            const format = document.getElementById('export-format')?.value;
            const outputName = document.getElementById('export-output-name')?.value;

            if (!modelSelect?.value) {
                CoreModule.showAlert('Please select a model to export', 'warning');
                return;
            }

            if (!format) {
                CoreModule.showAlert('Please select an export format', 'warning');
                return;
            }

            if (!outputName) {
                CoreModule.showAlert('Please enter an output name', 'warning');
                return;
            }

            const exportConfig = this.gatherExportConfig();

            CoreModule.showConfirmModal(
                'Export Model',
                `Export model to ${format.toUpperCase()} format?`,
                () => this.executeExport(exportConfig),
                'btn-primary'
            );
        },

        // Gather export configuration
        gatherExportConfig() {
            const modelSelect = document.getElementById('export-model-select');
            const selectedValue = modelSelect.value;

            let modelPath;
            let sessionId;

            if (selectedValue.startsWith('server:')) {
                modelPath = selectedValue.substring(7);
                // Extract session ID from path (e.g., "./outputs/session-id")
                const pathParts = modelPath.split('/');
                sessionId = pathParts[pathParts.length - 1];
            } else {
                const modelIndex = parseInt(selectedValue);
                const model = AppState.trainedModels[modelIndex];
                modelPath = model?.path;
                sessionId = model?.session_id;
            }

            const format = document.getElementById('export-format')?.value;
            const formatOptions = this.gatherFormatOptions();

            // Build config for backend
            const config = {
                format: format,
                name: document.getElementById('export-output-name')?.value,
                merge_lora: false // Could add UI option for this later
            };

            // For GGUF format, add quantization from format options
            if (format === 'gguf' && formatOptions.quantization) {
                config.quantization = formatOptions.quantization;
            }

            // Add session ID and model path for reference
            config.session_id = sessionId;
            config.model_path = modelPath;

            return config;
        },

        // Gather format-specific options
        gatherFormatOptions() {
            const format = document.getElementById('export-format')?.value;
            const options = {};

            switch (format) {
                case 'gguf':
                    options.quantization = document.getElementById('gguf-quantization')?.value || 'q4_k_m';
                    options.include_metadata = document.getElementById('gguf-metadata')?.checked;
                    break;

                case 'onnx':
                    options.optimize = document.getElementById('onnx-optimize')?.checked;
                    options.opset_version = parseInt(document.getElementById('onnx-opset')?.value) || 15;
                    break;

                case 'safetensors':
                    options.include_metadata = document.getElementById('st-metadata')?.checked;
                    options.compress = document.getElementById('st-compress')?.checked;
                    break;

                case 'pytorch':
                    options.state_dict_only = document.getElementById('pt-state-dict')?.checked;
                    options.include_traced = document.getElementById('pt-trace')?.checked;
                    break;
            }

            return options;
        },

        // Execute export
        executeExport(config) {
            if (!config.session_id) {
                CoreModule.showAlert('Cannot determine session ID for export', 'danger');
                return;
            }

            this.exportInProgress = true;
            this.updateExportUI(true);

            // Use the correct API endpoint with session ID
            fetch(`/api/export/${config.session_id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    CoreModule.showAlert('Model exported successfully!', 'success');
                    this.onExportComplete(data);
                } else {
                    throw new Error(data.error || 'Export failed');
                }
            })
            .catch(error => {
                console.error('Export error:', error);
                CoreModule.showAlert(`Export failed: ${error.message}`, 'danger');
            })
            .finally(() => {
                this.exportInProgress = false;
                this.updateExportUI(false);
            });
        },

        // Handle export completion
        onExportComplete(data) {
            const resultsContainer = document.getElementById('export-results');
            if (!resultsContainer) return;

            resultsContainer.innerHTML = `
                <div class="alert alert-success">
                    <h5 class="alert-heading">Export Complete!</h5>
                    <p>Model exported successfully to ${data.format.toUpperCase()} format.</p>
                    <hr>
                    <div class="mb-2">
                        <strong>Output Path:</strong> ${CoreModule.escapeHtml(data.output_path)}
                    </div>
                    <div class="mb-2">
                        <strong>File Size:</strong> ${this.formatFileSize(data.file_size)}
                    </div>
                    ${data.download_url ? `
                        <div class="mt-3">
                            <a href="${data.download_url}" class="btn btn-primary" download>
                                <i class="fas fa-download me-2"></i>Download Model
                            </a>
                        </div>
                    ` : ''}
                </div>
            `;

            // Update export history
            this.addToExportHistory(data);
        },

        // Update export UI
        updateExportUI(inProgress) {
            const exportBtn = document.getElementById('start-export-btn');
            const progressContainer = document.getElementById('export-progress');

            if (exportBtn) {
                exportBtn.disabled = inProgress;
                exportBtn.innerHTML = inProgress ?
                    '<i class="fas fa-spinner fa-spin me-2"></i>Exporting...' :
                    '<i class="fas fa-file-export me-2"></i>Export Model';
            }

            if (progressContainer) {
                progressContainer.style.display = inProgress ? 'block' : 'none';
            }
        },

        // Add to export history
        addToExportHistory(exportData) {
            const history = JSON.parse(localStorage.getItem('exportHistory') || '[]');
            history.unshift({
                timestamp: Date.now(),
                format: exportData.format,
                name: exportData.output_name,
                path: exportData.output_path,
                size: exportData.file_size
            });

            // Keep only last 20 exports
            if (history.length > 20) {
                history.length = 20;
            }

            localStorage.setItem('exportHistory', JSON.stringify(history));
            this.displayExportHistory();
        },

        // Display export history
        displayExportHistory() {
            const historyContainer = document.getElementById('export-history');
            if (!historyContainer) return;

            const history = JSON.parse(localStorage.getItem('exportHistory') || '[]');

            if (history.length === 0) {
                historyContainer.innerHTML = '<p class="text-muted">No export history</p>';
                return;
            }

            historyContainer.innerHTML = `
                <div class="list-group">
                    ${history.map(item => `
                        <div class="list-group-item">
                            <div class="d-flex justify-content-between align-items-start">
                                <div>
                                    <h6 class="mb-1">${CoreModule.escapeHtml(item.name)}</h6>
                                    <p class="mb-0 small text-muted">
                                        ${item.format.toUpperCase()} | ${this.formatFileSize(item.size)} |
                                        ${new Date(item.timestamp).toLocaleString()}
                                    </p>
                                </div>
                                <button class="btn btn-sm btn-outline-secondary" onclick="ExportModule.copyPath('${CoreModule.escapeHtml(item.path)}')">
                                    <i class="fas fa-copy"></i>
                                </button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        },

        // Copy path to clipboard
        copyPath(path) {
            navigator.clipboard.writeText(path)
                .then(() => {
                    CoreModule.showAlert('Path copied to clipboard', 'success');
                })
                .catch(err => {
                    console.error('Failed to copy path:', err);
                    CoreModule.showAlert('Failed to copy path', 'danger');
                });
        },

        // Format file size
        formatFileSize(bytes) {
            if (!bytes || bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
        },

        // Refresh list of trained models
        refreshTrainedModels() {
            const refreshBtn = document.querySelector('[onclick="refreshTrainedModels()"]');

            // Show loading state
            if (refreshBtn) {
                refreshBtn.disabled = true;
                refreshBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
            }

            // Fetch updated model list from server
            fetch('/api/trained_models')
                .then(response => response.json())
                .then(data => {
                    if (data.models) {
                        // Update AppState with server models (keep full structure)
                        AppState.trainedModels = data.models;

                        // Reload exportable models dropdown
                        this.loadExportableModels();

                        // Reload model cards display
                        this.displayTrainedModelCards();

                        CoreModule.showAlert('Model list refreshed successfully', 'success');
                    } else {
                        throw new Error('Invalid response format');
                    }
                })
                .catch(error => {
                    console.error('Failed to refresh trained models:', error);
                    CoreModule.showAlert(`Failed to refresh models: ${error.message}`, 'danger');
                })
                .finally(() => {
                    // Reset button state
                    if (refreshBtn) {
                        refreshBtn.disabled = false;
                        refreshBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh List';
                    }
                });
        },

        // Show export dialog for a specific model
        showExportDialog(sessionId, modelPath) {
            // Load available models first
            this.loadExportableModels();

            // Show export form
            const exportSection = document.querySelector('.export-config-section');
            if (exportSection) {
                exportSection.style.display = 'block';

                // Wait for models to load then select the model
                setTimeout(() => {
                    const modelSelect = document.getElementById('export-model-select');
                    if (modelSelect) {
                        modelSelect.value = `server:${modelPath}`;
                        this.onModelSelectChange();
                    }
                }, 100);

                // Scroll to export form
                exportSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        },

        // Show detailed information about a model
        showModelDetails(sessionId) {
            const detailsPanel = document.getElementById('model-details-panel');
            const detailsContent = document.getElementById('model-details-content');

            if (!detailsPanel || !detailsContent) {
                console.warn('Model details panel not found');
                return;
            }

            // Find model in AppState
            const model = AppState.trainedModels.find(m => m.session_id === sessionId);

            if (!model) {
                detailsContent.innerHTML = '<div class="alert alert-warning">Model not found</div>';
                detailsPanel.style.display = 'block';
                return;
            }

            // Show panel
            detailsPanel.style.display = 'block';

            // Extract model name
            let modelName = model.model_name;
            if (!modelName || modelName === 'Unknown') {
                modelName = model.training_config?.model?.modelName ||
                           model.display_name ||
                           model.session_id ||
                           'Unknown Model';
            }

            // Build details HTML
            let detailsHtml = `
                <div class="mb-3">
                    <h6 class="fw-bold mb-2" style="color: var(--text-primary);">${CoreModule.escapeHtml(modelName)}</h6>
                    <div class="small text-muted">
                        <div class="mb-1"><strong>Session ID:</strong> ${CoreModule.escapeHtml(model.session_id)}</div>
                        <div class="mb-1"><strong>Path:</strong> ${CoreModule.escapeHtml(model.path)}</div>
                        <div class="mb-1"><strong>Created:</strong> ${new Date(model.created_at).toLocaleString()}</div>
                        <div class="mb-1"><strong>Completed:</strong> ${new Date(model.modified_at).toLocaleString()}</div>
                        <div class="mb-1"><strong>Epochs:</strong> ${model.epochs || 0}</div>
                    </div>
                </div>
            `;

            // Add checkpoints if available
            if (model.checkpoints && model.checkpoints.length > 0) {
                const checkpointsList = model.checkpoints.map(cp => `
                    <li class="list-group-item" style="background-color: var(--bg-card); color: var(--text-primary); border-color: var(--border-color);">
                        <div>
                            <strong>${CoreModule.escapeHtml(cp.name)}</strong>
                            <br>
                            <small class="text-muted">${CoreModule.escapeHtml(cp.path)}</small>
                        </div>
                    </li>
                `).join('');

                detailsHtml += `
                    <div class="card" style="background-color: var(--bg-card); border-color: var(--border-color);">
                        <div class="card-header" style="background-color: var(--bg-secondary); border-color: var(--border-color);">
                            <h6 class="mb-0" style="color: var(--text-primary);">Model Checkpoints</h6>
                        </div>
                        <div class="card-body p-0">
                            <ul class="list-group list-group-flush">
                                ${checkpointsList}
                            </ul>
                        </div>
                    </div>
                `;
            } else {
                detailsHtml += '<div class="alert alert-info mb-0">No checkpoints available</div>';
            }

            detailsContent.innerHTML = detailsHtml;
        },

        // Hide model details panel
        hideModelDetails() {
            const detailsPanel = document.getElementById('model-details-panel');
            if (!detailsPanel) {
                console.warn('Model details panel not found');
                return;
            }

            // Hide with animation
            detailsPanel.style.opacity = '0';
            detailsPanel.style.transform = 'translateY(-10px)';

            setTimeout(() => {
                detailsPanel.style.display = 'none';
                detailsPanel.style.opacity = '1';
                detailsPanel.style.transform = 'translateY(0)';
            }, 300);

            // Clear content
            const detailsContent = document.getElementById('model-details-content');
            if (detailsContent) {
                detailsContent.innerHTML = '';
            }
        }
    };

    // Export to window
    window.ExportModule = ExportModule;

    // Export functions for onclick handlers
    window.startExport = () => ExportModule.startExport();

})(window);

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
                option.textContent = `${model.name} (${new Date(model.timestamp).toLocaleString()})`;
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
                            option.textContent = model.name;
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
                            // Extract model name from various sources
                            let modelName = model.model_name;
                            if (!modelName || modelName === 'Unknown') {
                                // Try to get from training_config
                                modelName = model.training_config?.model?.modelName ||
                                           model.display_name ||
                                           model.session_id ||
                                           'Unknown Model';
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
                                                    ${CoreModule.escapeHtml(modelName)}
                                                </h6>
                                                ${datasetName ? `<small class="text-muted"><i class="fas fa-database"></i> ${CoreModule.escapeHtml(datasetName)}</small>` : ''}
                                            </div>
                                            <span class="badge bg-success">Completed</span>
                                        </div>

                                        <div class="model-meta text-muted small mb-3">
                                            <div><i class="fas fa-calendar"></i> Completed: ${completedDate.toLocaleString()}</div>
                                            <div><i class="fas fa-layer-group"></i> Epochs: ${epochs}</div>
                                            <div><i class="fas fa-award"></i> Best Reward: ${bestReward}</div>
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

            infoContainer.innerHTML = `
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">Model Information</h6>
                        <dl class="row mb-0">
                            <dt class="col-sm-4">Name:</dt>
                            <dd class="col-sm-8">${CoreModule.escapeHtml(model.name)}</dd>

                            <dt class="col-sm-4">Size:</dt>
                            <dd class="col-sm-8">${this.formatFileSize(model.size || 0)}</dd>

                            <dt class="col-sm-4">Created:</dt>
                            <dd class="col-sm-8">${new Date(model.timestamp || Date.now()).toLocaleString()}</dd>

                            ${model.metrics ? `
                                <dt class="col-sm-4">Final Loss:</dt>
                                <dd class="col-sm-8">${model.metrics.final_loss?.toFixed(4) || 'N/A'}</dd>
                            ` : ''}
                        </dl>
                    </div>
                </div>
            `;
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
            if (selectedValue.startsWith('server:')) {
                modelPath = selectedValue.substring(7);
            } else {
                const modelIndex = parseInt(selectedValue);
                modelPath = AppState.trainedModels[modelIndex]?.path;
            }

            return {
                model_path: modelPath,
                format: document.getElementById('export-format')?.value,
                output_name: document.getElementById('export-output-name')?.value,
                quantization: {
                    enabled: document.getElementById('enable-quantization')?.checked,
                    type: document.getElementById('quantization-type')?.value,
                    bits: parseInt(document.getElementById('quantization-bits')?.value) || 8
                },
                format_options: this.gatherFormatOptions()
            };
        },

        // Gather format-specific options
        gatherFormatOptions() {
            const format = document.getElementById('export-format')?.value;
            const options = {};

            switch (format) {
                case 'gguf':
                    options.use_f16 = document.getElementById('gguf-f16')?.checked;
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
            this.exportInProgress = true;
            this.updateExportUI(true);

            fetch('/api/export_model', {
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
                        // Update AppState with server models
                        AppState.trainedModels = data.models.map(model => ({
                            name: model.name,
                            path: model.path,
                            timestamp: model.timestamp || Date.now(),
                            size: model.size || 0,
                            metrics: model.metrics || {}
                        }));

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

        // Show batch export modal
        showBatchExport() {
            const modalId = 'batchExportModal';
            let modalElement = document.getElementById(modalId);

            // Create modal if it doesn't exist
            if (!modalElement) {
                const modalHtml = `
                    <div class="modal fade" id="${modalId}" tabindex="-1">
                        <div class="modal-dialog modal-lg">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">
                                        <i class="fas fa-boxes"></i> Batch Export Models
                                    </h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body">
                                    <div class="mb-3">
                                        <label class="form-label">Select Models to Export</label>
                                        <div id="batch-export-model-list" class="list-group" style="max-height: 300px; overflow-y: auto;">
                                            <!-- Model list will be populated here -->
                                        </div>
                                    </div>
                                    <div class="mb-3">
                                        <label class="form-label">Export Format</label>
                                        <select class="form-select" id="batch-export-format">
                                            <option value="safetensors">SafeTensors</option>
                                            <option value="gguf">GGUF</option>
                                            <option value="pytorch">PyTorch</option>
                                            <option value="onnx">ONNX</option>
                                        </select>
                                    </div>
                                    <div class="form-check mb-3">
                                        <input class="form-check-input" type="checkbox" id="batch-export-quantize">
                                        <label class="form-check-label" for="batch-export-quantize">
                                            Enable quantization
                                        </label>
                                    </div>
                                    <div id="batch-export-progress" style="display: none;">
                                        <div class="progress mb-2">
                                            <div class="progress-bar" role="progressbar" id="batch-export-progress-bar" style="width: 0%"></div>
                                        </div>
                                        <p class="text-muted small mb-0" id="batch-export-status">Preparing export...</p>
                                    </div>
                                </div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                                    <button type="button" class="btn btn-primary" id="batch-export-start-btn" onclick="ExportModule.executeBatchExport()">
                                        <i class="fas fa-file-export me-2"></i>Export Selected
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
                modalElement = document.getElementById(modalId);
            }

            // Populate model list
            this.populateBatchExportList();

            // Show modal
            const modal = new bootstrap.Modal(modalElement);
            modal.show();
        },

        // Populate batch export model list
        populateBatchExportList() {
            const listContainer = document.getElementById('batch-export-model-list');
            if (!listContainer) return;

            if (AppState.trainedModels.length === 0) {
                listContainer.innerHTML = '<div class="text-center text-muted p-3">No trained models available</div>';
                return;
            }

            listContainer.innerHTML = AppState.trainedModels.map((model, index) => `
                <div class="list-group-item">
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="batch-model-${index}" value="${index}">
                        <label class="form-check-label" for="batch-model-${index}">
                            <strong>${CoreModule.escapeHtml(model.name)}</strong>
                            <br>
                            <small class="text-muted">
                                ${this.formatFileSize(model.size)} | ${new Date(model.timestamp).toLocaleString()}
                            </small>
                        </label>
                    </div>
                </div>
            `).join('');
        },

        // Execute batch export
        executeBatchExport() {
            const selectedModels = [];
            const checkboxes = document.querySelectorAll('#batch-export-model-list input[type="checkbox"]:checked');

            checkboxes.forEach(cb => {
                const index = parseInt(cb.value);
                if (AppState.trainedModels[index]) {
                    selectedModels.push(AppState.trainedModels[index]);
                }
            });

            if (selectedModels.length === 0) {
                CoreModule.showAlert('Please select at least one model to export', 'warning');
                return;
            }

            const format = document.getElementById('batch-export-format')?.value;
            const quantize = document.getElementById('batch-export-quantize')?.checked;

            // Show progress
            const progressContainer = document.getElementById('batch-export-progress');
            const startBtn = document.getElementById('batch-export-start-btn');

            if (progressContainer) progressContainer.style.display = 'block';
            if (startBtn) startBtn.disabled = true;

            // Export models sequentially
            this.exportModelsSequentially(selectedModels, format, quantize, 0);
        },

        // Export models sequentially
        exportModelsSequentially(models, format, quantize, index) {
            if (index >= models.length) {
                // All exports complete
                CoreModule.showAlert(`Successfully exported ${models.length} models`, 'success');

                const progressContainer = document.getElementById('batch-export-progress');
                const startBtn = document.getElementById('batch-export-start-btn');

                if (progressContainer) progressContainer.style.display = 'none';
                if (startBtn) startBtn.disabled = false;

                // Close modal after a delay
                setTimeout(() => {
                    const modal = bootstrap.Modal.getInstance(document.getElementById('batchExportModal'));
                    if (modal) modal.hide();
                }, 2000);
                return;
            }

            const model = models[index];
            const progressBar = document.getElementById('batch-export-progress-bar');
            const statusText = document.getElementById('batch-export-status');

            // Update progress
            const progress = ((index + 1) / models.length) * 100;
            if (progressBar) progressBar.style.width = progress + '%';
            if (statusText) statusText.textContent = `Exporting ${model.name} (${index + 1}/${models.length})...`;

            // Export current model
            fetch('/api/export_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_path: model.path,
                    format: format,
                    output_name: `${model.name}_${format}`,
                    quantization: { enabled: quantize }
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Continue to next model
                    this.exportModelsSequentially(models, format, quantize, index + 1);
                } else {
                    throw new Error(data.error || 'Export failed');
                }
            })
            .catch(error => {
                console.error(`Failed to export ${model.name}:`, error);
                CoreModule.showAlert(`Failed to export ${model.name}: ${error.message}`, 'danger');
                // Continue to next model despite error
                this.exportModelsSequentially(models, format, quantize, index + 1);
            });
        },

        // Show export dialog for a specific model
        showExportDialog(sessionId, modelPath) {
            // Populate export form with the selected model
            const modelSelect = document.getElementById('export-model-select');
            if (modelSelect) {
                modelSelect.value = `server:${modelPath}`;
                this.onModelSelectChange();
            }

            // Scroll to export form or show it
            const exportSection = document.querySelector('.export-config-section');
            if (exportSection) {
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

            // Show loading state
            detailsContent.innerHTML = '<div class="text-center p-3"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';
            detailsPanel.style.display = 'block';

            // Fetch model details
            fetch(`/api/export/checkpoints/${sessionId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.checkpoints) {
                        const checkpointsList = data.checkpoints.map(cp => `
                            <li class="list-group-item">
                                <div class="d-flex justify-content-between align-items-center">
                                    <div>
                                        <strong>${CoreModule.escapeHtml(cp.name)}</strong>
                                        <br>
                                        <small class="text-muted">${cp.path}</small>
                                    </div>
                                </div>
                            </li>
                        `).join('');

                        detailsContent.innerHTML = `
                            <div class="card">
                                <div class="card-header">
                                    <h6 class="mb-0">Model Checkpoints</h6>
                                </div>
                                <div class="card-body">
                                    <ul class="list-group list-group-flush">
                                        ${checkpointsList}
                                    </ul>
                                </div>
                            </div>
                        `;
                    } else {
                        detailsContent.innerHTML = '<div class="alert alert-info">No checkpoint details available</div>';
                    }
                })
                .catch(error => {
                    console.error('Failed to load model details:', error);
                    detailsContent.innerHTML = `
                        <div class="alert alert-danger">
                            Failed to load model details: ${CoreModule.escapeHtml(error.message)}
                        </div>
                    `;
                });
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

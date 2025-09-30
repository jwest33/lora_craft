// ============================================================================
// Dataset Management Module
// ============================================================================

(function(window) {
    'use strict';

    const DatasetModule = {
        // GRPO prompt templates
        promptTemplates: {
            'grpo-default': {
                name: 'GRPO Default',
                reasoning_start: '<start_working_out>',
                reasoning_end: '<end_working_out>',
                solution_start: '<SOLUTION>',
                solution_end: '</SOLUTION>',
                system_prompt: 'You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>'
            },
            'qwen': {
                name: 'Qwen GRPO',
                reasoning_start: '<think>',
                reasoning_end: '</think>',
                solution_start: '<answer>',
                solution_end: '</answer>',
                system_prompt: 'You are Qwen, a helpful AI assistant. For each problem:\n1. Show your reasoning in <think></think> tags\n2. Provide your final answer in <answer></answer> tags'
            },
            'llama': {
                name: 'LLaMA GRPO',
                reasoning_start: '[REASONING]',
                reasoning_end: '[/REASONING]',
                solution_start: '[ANSWER]',
                solution_end: '[/ANSWER]',
                system_prompt: 'You are a helpful assistant. Approach each problem by:\n1. Explaining your thought process in [REASONING][/REASONING] tags\n2. Providing your final answer in [ANSWER][/ANSWER] tags'
            },
            'phi': {
                name: 'Phi GRPO',
                reasoning_start: '<reasoning>',
                reasoning_end: '</reasoning>',
                solution_start: '<output>',
                solution_end: '</output>',
                system_prompt: 'You are Phi, an AI assistant trained to think step-by-step.\n- Use <reasoning></reasoning> tags for your thought process\n- Use <output></output> tags for your final answer'
            }
        },

        // Initialize the module
        init() {
            this.setupEventListeners();
            this.initializeDatasetUpload();
            this.loadPopularDatasets();
            this.updateSavedTemplatesList();
        },

        // Setup dataset-related event listeners
        setupEventListeners() {
            // Dataset type change
            const datasetTypeSelect = document.getElementById('dataset-type');
            if (datasetTypeSelect) {
                datasetTypeSelect.addEventListener('change', () => this.onDatasetTypeChange());
            }

            // Sample size change
            const sampleSizeInput = document.getElementById('sample-size');
            if (sampleSizeInput) {
                sampleSizeInput.addEventListener('change', () => this.updateDatasetStats());
            }

            // Train/eval split change
            const trainSplitInput = document.getElementById('train-split');
            if (trainSplitInput) {
                trainSplitInput.addEventListener('input', () => this.updateSplitDisplay());
            }

            // GRPO prompt template change
            const promptTemplateSelect = document.getElementById('prompt-template-select');
            if (promptTemplateSelect) {
                promptTemplateSelect.addEventListener('change', () => this.onPromptTemplateChange());
            }
        },

        // Load popular datasets from API
        loadPopularDatasets() {
            const grid = document.getElementById('dataset-grid');
            if (!grid) return;

            // Show loading state
            grid.innerHTML = '<div class="text-center p-4"><i class="fas fa-spinner fa-spin fa-2x"></i><p class="mt-2">Loading datasets...</p></div>';

            // Fetch datasets from API
            fetch('/api/datasets/list')
                .then(response => response.json())
                .then(data => {
                    if (data.datasets && data.datasets.length > 0) {
                        this.renderDatasetGrid(data.datasets);
                    } else {
                        grid.innerHTML = '<div class="alert alert-info">No datasets available</div>';
                    }
                })
                .catch(error => {
                    console.error('Failed to load datasets:', error);
                    grid.innerHTML = '<div class="alert alert-danger">Failed to load datasets. Please try again.</div>';
                });
        },

        // Render dataset grid
        renderDatasetGrid(datasets) {
            const grid = document.getElementById('dataset-grid');
            if (!grid) return;

            grid.innerHTML = datasets.map(dataset => `
                <div class="dataset-item ${dataset.category}" data-dataset-path="${dataset.path}">
                    <div class="dataset-card" onclick="DatasetsModule.selectDataset('${dataset.path}', '${CoreModule.escapeHtml(dataset.name)}')">
                        <div class="dataset-header">
                            <h6 class="dataset-name">${CoreModule.escapeHtml(dataset.name)}</h6>
                            ${dataset.is_cached ? '<span class="badge bg-success"><i class="fas fa-check"></i> Cached</span>' : '<span class="badge bg-secondary"><i class="fas fa-download"></i> Download</span>'}
                        </div>
                        <div class="dataset-meta">
                            <span class="badge bg-info">${dataset.category}</span>
                            <span class="text-muted small">${dataset.size}</span>
                        </div>
                    </div>
                </div>
            `).join('');
        },

        // Select a dataset from the catalog
        selectDataset(datasetPath, datasetName) {
            console.log('Dataset selected:', datasetPath);

            // Store selection
            AppState.setConfigValue('datasetPath', datasetPath);
            AppState.setConfigValue('datasetName', datasetName);

            // Set datasetType to 'popular' so it maps to 'huggingface' source_type
            // This is crucial for the backend to know to download from HuggingFace
            AppState.setConfigValue('datasetType', 'popular');

            // Update UI to show selection
            const datasetPathInput = document.getElementById('dataset-path');
            if (datasetPathInput) {
                datasetPathInput.value = datasetPath;
            }

            // Update configuration summary
            if (typeof window.updateConfigSummary === 'function') {
                window.updateConfigSummary();
            }

            // Show success message
            CoreModule.showAlert(`Selected: ${datasetName}`, 'success');

            // Mark step as complete
            if (window.NavigationModule && NavigationModule.completeStep) {
                NavigationModule.completeStep(2);
            }
        },

        // Initialize dataset upload functionality
        initializeDatasetUpload() {
            const uploadZone = document.getElementById('dataset-upload-zone');
            const fileInput = document.getElementById('dataset-file-input');

            if (uploadZone && fileInput) {
                // NOTE: Removed whole-zone click listener to prevent auto-trigger
                // The "Browse Files" button handles clicks via inline onclick

                // Drag and drop
                uploadZone.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    uploadZone.classList.add('dragover');
                });

                uploadZone.addEventListener('dragleave', () => {
                    uploadZone.classList.remove('dragover');
                });

                uploadZone.addEventListener('drop', (e) => {
                    e.preventDefault();
                    uploadZone.classList.remove('dragover');

                    const files = e.dataTransfer.files;
                    if (files.length > 0) {
                        this.handleDatasetFile(files[0]);
                    }
                });

                // File input change
                fileInput.addEventListener('change', (e) => {
                    if (e.target.files.length > 0) {
                        this.handleDatasetFile(e.target.files[0]);
                        // Reset the input so same file can be selected again if needed
                        e.target.value = '';
                    }
                });
            }
        },

        // Handle dataset type change
        onDatasetTypeChange() {
            const datasetType = document.getElementById('dataset-type')?.value;

            // Show/hide relevant sections based on type
            const uploadSection = document.getElementById('dataset-upload-section');
            const huggingfaceSection = document.getElementById('dataset-huggingface-section');
            const localSection = document.getElementById('dataset-local-section');

            if (uploadSection) uploadSection.style.display = datasetType === 'upload' ? 'block' : 'none';
            if (huggingfaceSection) huggingfaceSection.style.display = datasetType === 'huggingface' ? 'block' : 'none';
            if (localSection) localSection.style.display = datasetType === 'local' ? 'block' : 'none';

            AppState.setConfigValue('datasetType', datasetType);
        },

        // Handle dataset file upload
        handleDatasetFile(file) {
            // Safety check - ensure file exists
            if (!file) {
                console.warn('No file provided to handleDatasetFile');
                return;
            }

            // Validate file type
            const validTypes = ['.json', '.jsonl', '.csv', '.txt', '.parquet'];
            const fileExt = '.' + file.name.split('.').pop().toLowerCase();

            if (!validTypes.includes(fileExt)) {
                CoreModule.showAlert(`Invalid file type. Supported types: ${validTypes.join(', ')}`, 'danger');
                return;
            }

            // Update UI
            const uploadStatus = document.getElementById('dataset-upload-status');
            if (uploadStatus) {
                uploadStatus.innerHTML = `
                    <div class="alert alert-info">
                        <i class="fas fa-spinner fa-spin me-2"></i>
                        Uploading ${CoreModule.escapeHtml(file.name)}...
                    </div>
                `;
            }

            // Create form data
            const formData = new FormData();
            formData.append('file', file);  // Backend expects 'file' field name

            // Upload file
            fetch('/api/upload_dataset', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.onDatasetUploaded(data);
                } else {
                    throw new Error(data.error || 'Upload failed');
                }
            })
            .catch(error => {
                console.error('Dataset upload error:', error);
                CoreModule.showAlert(`Failed to upload dataset: ${error.message}`, 'danger');
                if (uploadStatus) {
                    uploadStatus.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle me-2"></i>
                            Upload failed: ${CoreModule.escapeHtml(error.message)}
                        </div>
                    `;
                }
            });
        },

        // Handle successful dataset upload
        onDatasetUploaded(data) {
            const uploadStatus = document.getElementById('dataset-upload-status');
            if (uploadStatus) {
                uploadStatus.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle me-2"></i>
                        Dataset uploaded successfully!
                        <div class="mt-2">
                            <strong>File:</strong> ${CoreModule.escapeHtml(data.filename)}<br>
                            <strong>Samples:</strong> ${data.sample_count || 'Unknown'}<br>
                            <strong>Size:</strong> ${this.formatFileSize(data.file_size)}
                        </div>
                    </div>
                `;
            }

            // Update dataset path
            const datasetPath = document.getElementById('dataset-path');
            if (datasetPath) {
                datasetPath.value = data.path;
            }

            // Update stats
            AppState.setConfigValue('datasetPath', data.path);
            AppState.setConfigValue('datasetSamples', data.sample_count);
            AppState.setConfigValue('datasetType', 'upload');  // Explicitly set type to upload
            this.updateDatasetStats();

            // Update configuration summary
            if (typeof window.updateConfigSummary === 'function') {
                window.updateConfigSummary();
            }

            // Validate step
            NavigationModule.validateStep(2);

            // Reload uploaded files list
            this.loadUploadedDatasets();
        },

        // Load previously uploaded datasets
        async loadUploadedDatasets() {
            try {
                const response = await fetch('/api/datasets/uploaded');
                const data = await response.json();

                if (data.files && data.files.length > 0) {
                    this.renderUploadedDatasets(data.files);
                } else {
                    // Hide container if no files
                    const container = document.getElementById('uploaded-datasets-container');
                    if (container) container.style.display = 'none';
                }
            } catch (error) {
                console.error('Failed to load uploaded datasets:', error);
            }
        },

        // Render uploaded datasets as cards
        renderUploadedDatasets(files) {
            const container = document.getElementById('uploaded-datasets-container');
            const grid = document.getElementById('uploaded-datasets-grid');

            if (!container || !grid) return;

            container.style.display = 'block';
            grid.innerHTML = files.map(file => `
                <div class="col-md-6">
                    <div class="card uploaded-dataset-card">
                        <div class="card-body">
                            <h6 class="card-title text-truncate" title="${CoreModule.escapeHtml(file.filename)}">
                                ${CoreModule.escapeHtml(file.filename)}
                            </h6>
                            <p class="card-text small text-muted mb-2">
                                <i class="fas fa-database"></i> ${file.size_mb} MB<br>
                                <i class="fas fa-clock"></i> ${new Date(file.uploaded_at).toLocaleString()}
                            </p>
                            <div class="d-flex gap-2">
                                <button class="btn btn-sm btn-primary flex-grow-1"
                                        onclick="DatasetsModule.selectUploadedDataset('${file.relative_path}', '${CoreModule.escapeHtml(file.filename)}')">
                                    <i class="fas fa-check"></i> Select
                                </button>
                                <button class="btn btn-sm btn-outline-danger"
                                        onclick="DatasetsModule.deleteUploadedDataset('${CoreModule.escapeHtml(file.filename)}')">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `).join('');
        },

        // Select an uploaded dataset
        selectUploadedDataset(path, filename) {
            // Update dataset path input
            const datasetPathInput = document.getElementById('dataset-path');
            if (datasetPathInput) {
                datasetPathInput.value = path;
            }

            // Store in app state
            AppState.setConfigValue('datasetPath', path);
            AppState.setConfigValue('datasetName', filename);
            AppState.setConfigValue('datasetType', 'upload');  // Explicitly set type to upload

            // Update configuration summary
            if (typeof window.updateConfigSummary === 'function') {
                window.updateConfigSummary();
            }

            // Show success message
            CoreModule.showAlert(`Selected: ${filename}`, 'success');

            // Mark step as complete
            if (NavigationModule && NavigationModule.validateStep) {
                NavigationModule.validateStep(2);
            }
        },

        // Delete an uploaded dataset
        async deleteUploadedDataset(filename) {
            if (!confirm(`Delete ${filename}?\n\nThis action cannot be undone.`)) {
                return;
            }

            try {
                const response = await fetch(`/api/datasets/uploaded/${encodeURIComponent(filename)}`, {
                    method: 'DELETE'
                });
                const data = await response.json();

                if (data.success) {
                    CoreModule.showAlert('File deleted successfully', 'success');
                    this.loadUploadedDatasets();  // Refresh list
                } else {
                    throw new Error(data.error || 'Delete failed');
                }
            } catch (error) {
                console.error('Failed to delete dataset:', error);
                CoreModule.showAlert(`Delete failed: ${error.message}`, 'danger');
            }
        },

        // Browse for local dataset
        browseDataset() {
            fetch('/api/list_datasets')
                .then(response => response.json())
                .then(data => {
                    this.showDatasetBrowser(data.datasets);
                })
                .catch(error => {
                    console.error('Failed to list datasets:', error);
                    CoreModule.showAlert('Failed to load dataset list', 'danger');
                });
        },

        // Show dataset browser modal
        showDatasetBrowser(datasets) {
            const modalId = 'datasetBrowserModal';
            let modal = document.getElementById(modalId);

            if (!modal) {
                const modalHtml = `
                    <div class="modal fade" id="${modalId}" tabindex="-1">
                        <div class="modal-dialog modal-lg">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">Select Dataset</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body">
                                    <div class="list-group" id="dataset-list"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
                modal = document.getElementById(modalId);
            }

            // Populate dataset list
            const datasetList = modal.querySelector('#dataset-list');
            datasetList.innerHTML = '';

            if (datasets.length === 0) {
                datasetList.innerHTML = '<p class="text-muted">No datasets found</p>';
            } else {
                datasets.forEach(dataset => {
                    const item = document.createElement('button');
                    item.className = 'list-group-item list-group-item-action';
                    item.innerHTML = `
                        <div class="d-flex justify-content-between align-items-center">
                            <div>
                                <h6 class="mb-1">${CoreModule.escapeHtml(dataset.name)}</h6>
                                <p class="mb-0 small text-muted">
                                    ${dataset.samples || 'Unknown'} samples |
                                    ${this.formatFileSize(dataset.size)}
                                </p>
                            </div>
                            <i class="fas fa-chevron-right"></i>
                        </div>
                    `;
                    item.onclick = () => {
                        this.selectDataset(dataset.path);
                        bootstrap.Modal.getInstance(modal).hide();
                    };
                    datasetList.appendChild(item);
                });
            }

            // Show modal
            new bootstrap.Modal(modal).show();
        },

        // Select a dataset
        selectDataset(path) {
            const datasetPath = document.getElementById('dataset-path');
            if (datasetPath) {
                datasetPath.value = path;
            }

            AppState.setConfigValue('datasetPath', path);
            this.loadDatasetInfo(path);
            NavigationModule.validateStep(2);
        },

        // Load dataset information
        loadDatasetInfo(path) {
            fetch('/api/dataset_info', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path })
            })
            .then(response => response.json())
            .then(data => {
                this.displayDatasetInfo(data);
            })
            .catch(error => {
                console.error('Failed to load dataset info:', error);
            });
        },

        // Display dataset information
        displayDatasetInfo(info) {
            const infoContainer = document.getElementById('dataset-info');
            if (!infoContainer) return;

            infoContainer.innerHTML = `
                <div class="card">
                    <div class="card-body">
                        <h6 class="card-title">Dataset Information</h6>
                        <dl class="row mb-0">
                            <dt class="col-sm-4">Total Samples:</dt>
                            <dd class="col-sm-8">${info.total_samples || 'Unknown'}</dd>

                            <dt class="col-sm-4">Format:</dt>
                            <dd class="col-sm-8">${info.format || 'Unknown'}</dd>

                            <dt class="col-sm-4">Size:</dt>
                            <dd class="col-sm-8">${this.formatFileSize(info.size)}</dd>

                            ${info.columns ? `
                                <dt class="col-sm-4">Columns:</dt>
                                <dd class="col-sm-8">${info.columns.join(', ')}</dd>
                            ` : ''}
                        </dl>
                    </div>
                </div>
            `;

            AppState.setConfigValue('datasetSamples', info.total_samples);
            this.updateDatasetStats();
        },

        // Update dataset statistics display
        updateDatasetStats() {
            const sampleSize = parseInt(document.getElementById('sample-size')?.value) || 0;
            const trainSplit = parseInt(document.getElementById('train-split')?.value) || 80;
            const totalSamples = AppState.getConfigValue('datasetSamples') || 0;

            const actualSamples = sampleSize > 0 ? Math.min(sampleSize, totalSamples) : totalSamples;
            const trainSamples = Math.floor(actualSamples * (trainSplit / 100));
            const evalSamples = actualSamples - trainSamples;

            // Update display
            const statsDisplay = document.getElementById('dataset-stats');
            if (statsDisplay) {
                statsDisplay.innerHTML = `
                    <div class="row">
                        <div class="col-md-4">
                            <small class="text-muted">Total Samples</small>
                            <div class="h5">${actualSamples.toLocaleString()}</div>
                        </div>
                        <div class="col-md-4">
                            <small class="text-muted">Training</small>
                            <div class="h5 text-primary">${trainSamples.toLocaleString()}</div>
                        </div>
                        <div class="col-md-4">
                            <small class="text-muted">Evaluation</small>
                            <div class="h5 text-info">${evalSamples.toLocaleString()}</div>
                        </div>
                    </div>
                `;
            }
        },

        // Update train/eval split display
        updateSplitDisplay() {
            const trainSplit = document.getElementById('train-split');
            const splitDisplay = document.getElementById('split-display');

            if (trainSplit && splitDisplay) {
                const value = trainSplit.value;
                splitDisplay.textContent = `${value}% / ${100 - value}%`;
                this.updateDatasetStats();
            }
        },

        // Preview dataset samples
        previewDataset() {
            const datasetPath = document.getElementById('dataset-path')?.value;
            if (!datasetPath) {
                CoreModule.showAlert('Please select a dataset first', 'warning');
                return;
            }

            fetch('/api/preview_dataset', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: datasetPath, samples: 5 })
            })
            .then(response => response.json())
            .then(data => {
                this.showDatasetPreview(data.samples);
            })
            .catch(error => {
                console.error('Failed to preview dataset:', error);
                CoreModule.showAlert('Failed to load dataset preview', 'danger');
            });
        },

        // Show dataset preview modal
        showDatasetPreview(samples) {
            const modalId = 'datasetPreviewModal';
            let modal = document.getElementById(modalId);

            if (!modal) {
                const modalHtml = `
                    <div class="modal fade" id="${modalId}" tabindex="-1">
                        <div class="modal-dialog modal-lg">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">Dataset Preview</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body">
                                    <div id="preview-samples"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
                modal = document.getElementById(modalId);
            }

            // Populate samples
            const previewContainer = modal.querySelector('#preview-samples');
            previewContainer.innerHTML = '';

            samples.forEach((sample, index) => {
                const sampleHtml = `
                    <div class="card mb-3">
                        <div class="card-header">
                            Sample ${index + 1}
                        </div>
                        <div class="card-body">
                            <pre class="mb-0" style="white-space: pre-wrap;">${CoreModule.escapeHtml(JSON.stringify(sample, null, 2))}</pre>
                        </div>
                    </div>
                `;
                previewContainer.insertAdjacentHTML('beforeend', sampleHtml);
            });

            // Show modal
            new bootstrap.Modal(modal).show();
        },

        // Format file size
        formatFileSize(bytes) {
            if (!bytes || bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
        },

        // Validate dataset configuration
        validateDatasetConfig() {
            const datasetPath = document.getElementById('dataset-path')?.value;
            const sampleSize = document.getElementById('sample-size')?.value;

            if (!datasetPath) {
                CoreModule.showAlert('Please select a dataset', 'warning');
                return false;
            }

            if (sampleSize && parseInt(sampleSize) < 1) {
                CoreModule.showAlert('Sample size must be at least 1', 'warning');
                return false;
            }

            return true;
        },

        // Handle GRPO prompt template change
        onPromptTemplateChange() {
            const templateSelect = document.getElementById('prompt-template-select');
            const templateEditor = document.getElementById('template-editor');

            if (!templateSelect) return;

            const selectedTemplate = templateSelect.value;

            // Show/hide custom template editor
            if (templateEditor) {
                templateEditor.style.display = selectedTemplate === 'custom' ? 'block' : 'none';
            }

            // Load predefined template if not custom
            if (selectedTemplate !== 'custom' && this.promptTemplates[selectedTemplate]) {
                this.loadPromptTemplate(selectedTemplate);
            }

            // Save selection
            AppState.setConfigValue('promptTemplate', selectedTemplate);
        },

        // Load a prompt template
        loadPromptTemplate(templateId) {
            const template = this.promptTemplates[templateId];
            if (!template) return;

            // Update custom template fields (so they're ready if user switches to custom)
            const reasoningStart = document.getElementById('custom-reasoning-start');
            const reasoningEnd = document.getElementById('custom-reasoning-end');
            const solutionStart = document.getElementById('custom-solution-start');
            const solutionEnd = document.getElementById('custom-solution-end');
            const systemPrompt = document.getElementById('custom-system-prompt');
            const hiddenSystemPrompt = document.getElementById('system-prompt');

            if (reasoningStart) reasoningStart.value = template.reasoning_start;
            if (reasoningEnd) reasoningEnd.value = template.reasoning_end;
            if (solutionStart) solutionStart.value = template.solution_start;
            if (solutionEnd) solutionEnd.value = template.solution_end;
            if (systemPrompt) systemPrompt.value = template.system_prompt;

            // Update hidden field that backend uses
            if (hiddenSystemPrompt) hiddenSystemPrompt.value = template.system_prompt;

            // Update preview display
            this.updateTemplatePreview(template);
        },

        // Update template preview display
        updateTemplatePreview(template) {
            const previewElement = document.getElementById('template-preview');
            if (!previewElement) return;

            // Create a sample formatted output showing the template structure
            const samplePreview = `System Prompt:
            ${template.system_prompt}

            ────────────────────────────────────

            User: Solve the equation: 2x + 5 = 13
            Assistant: ${template.reasoning_start}Let me think about this. 13 - 5 = 8. 8 / 2 = 4. x = 4.${template.reasoning_end}
            ${template.solution_start}4${template.solution_end}`

            previewElement.textContent = samplePreview;
        },

        // Test template with sample data
        testTemplate() {
            const reasoningStart = document.getElementById('custom-reasoning-start')?.value || '';
            const reasoningEnd = document.getElementById('custom-reasoning-end')?.value || '';
            const solutionStart = document.getElementById('custom-solution-start')?.value || '';
            const solutionEnd = document.getElementById('custom-solution-end')?.value || '';
            const systemPrompt = document.getElementById('custom-system-prompt')?.value || '';

            const sampleOutput = `${systemPrompt}

User: What is 2 + 2?
Assistant: ${reasoningStart}Let me think about this. 2 + 2 equals 4.${reasoningEnd}
${solutionStart}4${solutionEnd}`;

            CoreModule.showAlert(sampleOutput, "info");
        },

        // Save custom template
        saveCustomTemplate() {
            CoreModule.showInputModal(
                'Save Template',
                'Enter a name for this template:',
                'Template name',
                (templateName) => {
                    const customTemplate = {
                        name: templateName,
                        reasoning_start: document.getElementById('custom-reasoning-start')?.value || '',
                        reasoning_end: document.getElementById('custom-reasoning-end')?.value || '',
                        solution_start: document.getElementById('custom-solution-start')?.value || '',
                        solution_end: document.getElementById('custom-solution-end')?.value || '',
                        system_prompt: document.getElementById('custom-system-prompt')?.value || ''
                    };

                    // Save to localStorage
                    const savedTemplates = JSON.parse(localStorage.getItem('customPromptTemplates') || '{}');
                    savedTemplates[templateName] = customTemplate;
                    localStorage.setItem('customPromptTemplates', JSON.stringify(savedTemplates));

                    CoreModule.showAlert('Template saved successfully', 'success');
                    this.updateSavedTemplatesList();
                },
                'btn-success'
            );
        },

        // Load selected saved template
        loadSelectedTemplate() {
            const savedTemplatesList = document.getElementById('saved-templates-list');
            if (!savedTemplatesList || !savedTemplatesList.value) return;

            const savedTemplates = JSON.parse(localStorage.getItem('customPromptTemplates') || '{}');
            const template = savedTemplates[savedTemplatesList.value];

            if (template) {
                document.getElementById('custom-reasoning-start').value = template.reasoning_start;
                document.getElementById('custom-reasoning-end').value = template.reasoning_end;
                document.getElementById('custom-solution-start').value = template.solution_start;
                document.getElementById('custom-solution-end').value = template.solution_end;
                document.getElementById('custom-system-prompt').value = template.system_prompt;

                // Update template preview to reflect the loaded custom system prompt
                if (window.updateTemplatePreview) {
                    window.updateTemplatePreview();
                }

                CoreModule.showAlert('Template loaded', 'success');
            }
        },

        // Update saved templates list
        updateSavedTemplatesList() {
            const savedTemplatesList = document.getElementById('saved-templates-list');
            if (!savedTemplatesList) {
                console.warn('saved-templates-list element not found');
                return;
            }

            try {
                const savedTemplates = JSON.parse(localStorage.getItem('customPromptTemplates') || '{}');
                const templateCount = Object.keys(savedTemplates).length;

                console.log(`Loading ${templateCount} saved templates:`, Object.keys(savedTemplates));

                savedTemplatesList.innerHTML = '<option value="">Select a saved template...</option>';
                Object.keys(savedTemplates).forEach(name => {
                    const option = document.createElement('option');
                    option.value = name;
                    option.textContent = name;
                    savedTemplatesList.appendChild(option);
                    console.log(`Added template option: ${name}`);
                });

                if (templateCount === 0) {
                    console.log('No saved templates found in localStorage');
                }
            } catch (error) {
                console.error('Error updating saved templates list:', error);
            }
        },

        // Handle dataset type selection UI (popular / custom / upload)
        selectDatasetType(type) {
            console.log('Dataset type selected:', type);

            // Get the actual section elements
            const datasetConfig = document.getElementById('dataset-config');
            const catalogSection = document.getElementById('dataset-catalog');
            const customSection = document.getElementById('dataset-custom-area');
            const uploadSection = document.getElementById('dataset-upload-zone');

            // Remove active class from all cards
            document.querySelectorAll('.selection-card').forEach(card => {
                card.classList.remove('active');
            });

            // Add active class to selected card
            const selectedCard = document.getElementById(`dataset-${type}`);
            if (selectedCard) {
                selectedCard.classList.add('active');
            }

            // Show dataset config container
            if (datasetConfig) {
                datasetConfig.style.display = 'block';
            }

            // Hide all sections first
            if (catalogSection) catalogSection.style.display = 'none';
            if (customSection) customSection.style.display = 'none';
            if (uploadSection) uploadSection.style.display = 'none';

            // Show the selected section
            switch (type) {
                case 'popular':
                    if (catalogSection) catalogSection.style.display = 'block';
                    break;
                case 'custom':
                    if (customSection) customSection.style.display = 'block';
                    break;
                case 'upload':
                    if (uploadSection) uploadSection.style.display = 'block';
                    this.loadUploadedDatasets();  // Load cached uploads
                    break;
            }

            // Store selection in AppState
            AppState.setConfigValue('datasetType', type);
        },

        // Filter datasets list
        filterDatasets(filter) {
            console.log('Filtering datasets:', filter);
            const datasetItems = document.querySelectorAll('.dataset-item');
            const searchTerm = filter.toLowerCase();

            datasetItems.forEach(item => {
                const text = item.textContent.toLowerCase();
                item.style.display = text.includes(searchTerm) ? '' : 'none';
            });
        },

        // Download custom dataset from HuggingFace
        downloadCustomDataset() {
            const datasetInput = document.getElementById('hf-dataset-name');
            if (!datasetInput || !datasetInput.value.trim()) {
                CoreModule.showAlert('Please enter a dataset name', 'warning');
                return;
            }

            const datasetName = datasetInput.value.trim();
            console.log('Downloading dataset:', datasetName);

            // Show loading state
            const downloadBtn = document.getElementById('download-dataset-btn');
            if (downloadBtn) {
                downloadBtn.disabled = true;
                downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Downloading...';
            }

            // Make API call to download dataset
            fetch('/api/dataset/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_name: datasetName })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    CoreModule.showAlert('Dataset downloaded successfully', 'success');
                    // Refresh dataset list
                    this.loadAvailableDatasets();
                } else {
                    CoreModule.showAlert(data.error || 'Failed to download dataset', 'error');
                }
            })
            .catch(error => {
                console.error('Error downloading dataset:', error);
                CoreModule.showAlert('Error downloading dataset', 'error');
            })
            .finally(() => {
                // Reset button state
                if (downloadBtn) {
                    downloadBtn.disabled = false;
                    downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download';
                }
            });
        },

        // Cancel download
        cancelDownload() {
            console.log('Download cancelled');
            CoreModule.showAlert('Download cancelled', 'info');
        },

        // Use dataset from preview modal
        useDatasetFromPreview() {
            const modal = bootstrap.Modal.getInstance(document.getElementById('datasetPreviewModal'));
            if (modal) {
                modal.hide();
            }
            CoreModule.showAlert('Dataset selected', 'success');
        },

        // Load available datasets
        loadAvailableDatasets() {
            fetch('/api/datasets/list')
            .then(response => response.json())
            .then(data => {
                if (data.datasets) {
                    this.updateDatasetsList(data.datasets);
                }
            })
            .catch(error => {
                console.error('Error loading datasets:', error);
            });
        },

        // Update datasets list in UI
        updateDatasetsList(datasets) {
            const datasetsList = document.getElementById('datasets-list');
            if (!datasetsList) return;

            datasetsList.innerHTML = datasets.map(dataset => `
                <div class="dataset-item" onclick="DatasetsModule.selectDataset('${CoreModule.escapeHtml(dataset.name)}')">
                    <i class="fas fa-database"></i>
                    <span>${CoreModule.escapeHtml(dataset.name)}</span>
                </div>
            `).join('');
        },

        // Select a dataset
        selectDataset(datasetName) {
            console.log('Dataset selected:', datasetName);
            AppState.setConfigValue('selectedDataset', datasetName);
            CoreModule.showAlert(`Dataset "${datasetName}" selected`, 'success');
        }
    };

    // Export to window
    window.DatasetModule = DatasetModule;
    window.DatasetsModule = DatasetModule; // Alias for consistency

    // Export functions for onclick handlers
    window.browseDataset = () => DatasetModule.browseDataset();
    window.previewDataset = () => DatasetModule.previewDataset();
    window.onPromptTemplateChange = () => DatasetModule.onPromptTemplateChange();
    window.testTemplate = () => DatasetModule.testTemplate();
    window.saveCustomTemplate = () => DatasetModule.saveCustomTemplate();
    window.loadSelectedTemplate = () => DatasetModule.loadSelectedTemplate();

})(window);

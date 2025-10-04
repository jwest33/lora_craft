// ============================================================================
// Testing Module - Model Testing and Evaluation
// ============================================================================

(function(window) {
    'use strict';

    const TestingModule = {
        // Testing state
        currentTest: null,
        currentEvaluation: null,
        testHistory: [],
        batchTests: [],
        activeBatchTest: null,

        // Initialize the module
        init() {
            this.setupEventListeners();
            this.loadTestableModels();
            this.checkForActiveBatchTests();
            this.loadTestHistory();
        },

        // Debounce utility - delays function execution until after wait time has elapsed
        debounce(func, wait) {
            let timeout;
            return function executedFunction(...args) {
                const later = () => {
                    clearTimeout(timeout);
                    func(...args);
                };
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        },

        // Setup testing-related event listeners
        setupEventListeners() {
            // Test type change
            const testTypeSelect = document.getElementById('test-type');
            if (testTypeSelect) {
                testTypeSelect.addEventListener('change', () => this.onTestTypeChange());
            }

            // Model selection for testing
            const modelSelect = document.getElementById('test-model-select');
            if (modelSelect) {
                modelSelect.addEventListener('change', () => this.onTestModelChange());
            }

            // Comparison model selection
            const comparisonSelect = document.getElementById('comparison-model-select');
            if (comparisonSelect) {
                comparisonSelect.addEventListener('change', () => this.updateComparisonModelInfo());
            }

            // Comparison mode radio buttons
            const compareBaseRadio = document.getElementById('compare-base');
            const compareModelRadio = document.getElementById('compare-model');
            if (compareBaseRadio) {
                compareBaseRadio.addEventListener('change', () => this.toggleComparisonMode());
            }
            if (compareModelRadio) {
                compareModelRadio.addEventListener('change', () => this.toggleComparisonMode());
            }

            // Temperature slider
            const tempSlider = document.getElementById('test-temperature');
            if (tempSlider) {
                tempSlider.addEventListener('input', () => this.updateTemperatureDisplay());
            }

            // Top-p slider
            const topPSlider = document.getElementById('test-top-p');
            if (topPSlider) {
                topPSlider.addEventListener('input', () => this.updateTopPDisplay());
            }

            // Test prompt input - update counter and preview
            const promptInput = document.getElementById('test-prompt');
            if (promptInput) {
                // Create debounced preview update function
                const debouncedPreviewUpdate = this.debounce(() => {
                    this.updateTestPromptPreview();
                }, 500); // 500ms delay after user stops typing

                promptInput.addEventListener('input', () => {
                    this.updatePromptCounter();
                    debouncedPreviewUpdate(); // Auto-update preview
                });
            }

            // Batch comparison mode radio buttons
            const batchCompareBaseRadio = document.getElementById('batch-compare-base');
            const batchCompareModelRadio = document.getElementById('batch-compare-model');
            if (batchCompareBaseRadio) {
                batchCompareBaseRadio.addEventListener('change', () => this.updateBatchModelSelection());
            }
            if (batchCompareModelRadio) {
                batchCompareModelRadio.addEventListener('change', () => this.updateBatchModelSelection());
            }
        },

        // Check for active batch tests
        checkForActiveBatchTests() {
            fetch('/api/batch_tests/active')
                .then(response => response.json())
                .then(data => {
                    if (data.active_test) {
                        this.activeBatchTest = data.active_test;
                        this.updateBatchTestStatus();
                        this.monitorBatchTest();
                    }
                })
                .catch(error => {
                    console.error('Failed to check for active batch tests:', error);
                });
        },

        // Load test history
        loadTestHistory() {
            fetch('/api/test_history')
                .then(response => response.json())
                .then(data => {
                    this.testHistory = data.tests || [];
                    this.displayTestHistory();
                })
                .catch(error => {
                    console.error('Failed to load test history:', error);
                });
        },

        // Handle test type change
        onTestTypeChange() {
            const testType = document.getElementById('test-type')?.value;

            // Show/hide relevant sections
            const singleTestSection = document.getElementById('single-test-section');
            const batchTestSection = document.getElementById('batch-test-section');
            const benchmarkSection = document.getElementById('benchmark-section');

            if (singleTestSection) singleTestSection.style.display = testType === 'single' ? 'block' : 'none';
            if (batchTestSection) batchTestSection.style.display = testType === 'batch' ? 'block' : 'none';
            if (benchmarkSection) benchmarkSection.style.display = testType === 'benchmark' ? 'block' : 'none';
        },

        // Handle test model change
        onTestModelChange() {
            const modelSelect = document.getElementById('test-model-select');
            if (!modelSelect) return;

            const selectedValue = modelSelect.value;
            if (!selectedValue) return;

            // Load model capabilities
            this.loadModelCapabilities(selectedValue);

            // Update model info display
            this.updateModelInfo();

            // Update prompt preview with training chat template
            this.updateTestPromptPreview();
        },

        // Toggle between comparison modes
        toggleComparisonMode() {
            const mode = document.querySelector('input[name="comparison-mode"]:checked')?.value;
            const secondModelSection = document.getElementById('second-model-section');

            if (mode === 'model') {
                // Show second model selector for model-to-model comparison
                if (secondModelSection) {
                    secondModelSection.style.display = 'block';
                }
                // Populate comparison model dropdown if not already done
                this.populateComparisonModelSelect();
            } else {
                // Hide second model selector for base model comparison
                if (secondModelSection) {
                    secondModelSection.style.display = 'none';
                }
            }
        },

        // Update primary model info display
        updateModelInfo() {
            const modelSelect = document.getElementById('test-model-select');
            const modelInfo = document.getElementById('model-info');

            if (!modelSelect || !modelInfo) return;

            const selectedOption = modelSelect.options[modelSelect.selectedIndex];
            if (!selectedOption || !selectedOption.value) {
                modelInfo.innerHTML = '<i class="fas fa-info-circle"></i> No model selected';
                return;
            }

            const baseModel = selectedOption.dataset.baseModel || 'Unknown';
            const epochs = selectedOption.dataset.epochs || 'N/A';

            modelInfo.innerHTML = `
                <i class="fas fa-check-circle text-success"></i>
                <strong>${CoreModule.escapeHtml(selectedOption.text)}</strong><br>
                <small>Base: ${CoreModule.escapeHtml(baseModel)} | Epochs: ${CoreModule.escapeHtml(epochs)}</small>
            `;
        },

        // Update comparison model info display
        updateComparisonModelInfo() {
            const comparisonSelect = document.getElementById('comparison-model-select');
            const comparisonInfo = document.getElementById('comparison-model-info');

            if (!comparisonSelect || !comparisonInfo) return;

            const selectedOption = comparisonSelect.options[comparisonSelect.selectedIndex];
            if (!selectedOption || !selectedOption.value) {
                comparisonInfo.innerHTML = '<i class="fas fa-info-circle"></i> No comparison model selected';
                return;
            }

            const baseModel = selectedOption.dataset.baseModel || 'Unknown';
            const epochs = selectedOption.dataset.epochs || 'N/A';

            comparisonInfo.innerHTML = `
                <i class="fas fa-check-circle text-success"></i>
                <strong>${CoreModule.escapeHtml(selectedOption.text)}</strong><br>
                <small>Base: ${CoreModule.escapeHtml(baseModel)} | Epochs: ${CoreModule.escapeHtml(epochs)}</small>
            `;
        },

        // Populate comparison model select dropdown
        populateComparisonModelSelect() {
            const comparisonSelect = document.getElementById('comparison-model-select');
            if (!comparisonSelect) return;

            // If already populated, skip
            if (comparisonSelect.options.length > 1) return;

            fetch('/api/models/trained')
                .then(response => response.json())
                .then(data => {
                    if (data.models && data.models.length > 0) {
                        comparisonSelect.innerHTML = '<option value="">Select a model to compare against...</option>';
                        data.models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model.session_id;
                            option.textContent = `${model.name} (${model.epochs} epochs)`;
                            option.dataset.baseModel = model.base_model;
                            option.dataset.epochs = model.epochs;
                            comparisonSelect.appendChild(option);
                        });
                    }
                })
                .catch(error => {
                    console.error('Failed to load comparison models:', error);
                });
        },

        // Load model capabilities
        loadModelCapabilities(modelValue) {
            // Update UI based on model capabilities
            const contextInfo = document.getElementById('model-context-info');
            if (contextInfo) {
                // Mock data - would be fetched from server
                contextInfo.innerHTML = `
                    <div class="alert alert-info">
                        <strong>Model Context:</strong> 4096 tokens<br>
                        <strong>Supports:</strong> Text generation, Chat completion
                    </div>
                `;
            }
        },

        // Update temperature display
        updateTemperatureDisplay() {
            const slider = document.getElementById('test-temperature');
            const display = document.getElementById('temperature-display');
            if (slider && display) {
                display.textContent = slider.value;
            }
        },

        // Update top-p display
        updateTopPDisplay() {
            const slider = document.getElementById('test-top-p');
            const display = document.getElementById('top-p-display');
            if (slider && display) {
                display.textContent = slider.value;
            }
        },

        // Update prompt counter
        updatePromptCounter() {
            const promptInput = document.getElementById('test-prompt');
            const counter = document.getElementById('prompt-counter');
            if (promptInput && counter) {
                const length = promptInput.value.length;
                counter.textContent = `${length} characters`;
            }
        },

        // Run single test
        runSingleTest() {
            const modelSelect = document.getElementById('test-model-select');
            const prompt = document.getElementById('test-prompt')?.value;

            if (!modelSelect?.value) {
                CoreModule.showAlert('Please select a model to test', 'warning');
                return;
            }

            if (!prompt) {
                CoreModule.showAlert('Please enter a test prompt', 'warning');
                return;
            }

            const testConfig = {
                model: modelSelect.value,
                prompt: prompt,
                temperature: parseFloat(document.getElementById('test-temperature')?.value) || 0.7,
                top_p: parseFloat(document.getElementById('test-top-p')?.value) || 0.9,
                max_tokens: parseInt(document.getElementById('test-max-tokens')?.value) || 256,
                repetition_penalty: parseFloat(document.getElementById('test-rep-penalty')?.value) || 1.1
            };

            this.executeSingleTest(testConfig);
        },

        // Execute single test
        executeSingleTest(config) {
            // Update UI
            const resultContainer = document.getElementById('test-result');
            if (resultContainer) {
                resultContainer.innerHTML = `
                    <div class="alert alert-info">
                        <i class="fas fa-spinner fa-spin me-2"></i>
                        Generating response...
                    </div>
                `;
            }

            // Start test
            fetch('/api/test_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.displayTestResult(data);
                } else {
                    throw new Error(data.error || 'Test failed');
                }
            })
            .catch(error => {
                console.error('Test error:', error);
                if (resultContainer) {
                    resultContainer.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle me-2"></i>
                            ${CoreModule.escapeHtml(error.message)}
                        </div>
                    `;
                }
            });
        },

        // Display test result
        displayTestResult(data) {
            const resultContainer = document.getElementById('test-result');
            if (!resultContainer) return;

            const elapsed = data.generation_time ? `${data.generation_time.toFixed(2)}s` : 'N/A';
            const tokensPerSec = data.tokens_per_second ? `${data.tokens_per_second.toFixed(1)} tok/s` : 'N/A';

            resultContainer.innerHTML = `
                <div class="card">
                    <div class="card-header">
                        <div class="d-flex justify-content-between">
                            <span>Generation Result</span>
                            <span class="text-muted">
                                ${elapsed} | ${tokensPerSec}
                            </span>
                        </div>
                    </div>
                    <div class="card-body">
                        <pre class="mb-0" style="white-space: pre-wrap;">${CoreModule.escapeHtml(data.response)}</pre>
                    </div>
                    <div class="card-footer">
                        <div class="row small text-muted">
                            <div class="col-md-3">
                                <strong>Tokens:</strong> ${data.token_count || 'N/A'}
                            </div>
                            <div class="col-md-3">
                                <strong>Model:</strong> ${CoreModule.escapeHtml(data.model || 'Unknown')}
                            </div>
                            <div class="col-md-6 text-end">
                                <button class="btn btn-sm btn-outline-secondary" onclick="TestingModule.copyResponse()">
                                    <i class="fas fa-copy me-1"></i>Copy
                                </button>
                                <button class="btn btn-sm btn-outline-primary ms-2" onclick="TestingModule.saveTest()">
                                    <i class="fas fa-save me-1"></i>Save
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Store current result
            this.currentTest = {
                prompt: document.getElementById('test-prompt')?.value,
                response: data.response,
                config: data.config,
                metrics: {
                    generation_time: data.generation_time,
                    tokens_per_second: data.tokens_per_second,
                    token_count: data.token_count
                }
            };
        },

        // Copy response to clipboard
        copyResponse() {
            if (!this.currentTest) return;

            navigator.clipboard.writeText(this.currentTest.response)
                .then(() => {
                    CoreModule.showAlert('Response copied to clipboard', 'success');
                })
                .catch(err => {
                    console.error('Failed to copy:', err);
                    CoreModule.showAlert('Failed to copy response', 'danger');
                });
        },

        // Save test result
        saveTest() {
            if (!this.currentTest) return;

            const testData = {
                ...this.currentTest,
                timestamp: Date.now()
            };

            fetch('/api/save_test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(testData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    CoreModule.showAlert('Test saved successfully', 'success');
                    this.testHistory.unshift(testData);
                    this.displayTestHistory();
                }
            })
            .catch(error => {
                console.error('Failed to save test:', error);
                CoreModule.showAlert('Failed to save test', 'danger');
            });
        },

        // Start batch test
        startBatchTest() {
            const modelSelect = document.getElementById('test-model-select');
            const testFile = document.getElementById('batch-test-file')?.files[0];

            if (!modelSelect?.value) {
                CoreModule.showAlert('Please select a model to test', 'warning');
                return;
            }

            if (!testFile) {
                CoreModule.showAlert('Please select a test file', 'warning');
                return;
            }

            // Create form data
            const formData = new FormData();
            formData.append('model', modelSelect.value);
            formData.append('test_file', testFile);
            formData.append('temperature', document.getElementById('test-temperature')?.value || 0.7);
            formData.append('top_p', document.getElementById('test-top-p')?.value || 0.9);

            // Start batch test
            this.executeBatchTest(formData);
        },

        // Execute batch test
        executeBatchTest(formData) {
            // Update UI
            const progressContainer = document.getElementById('batch-test-progress');
            if (progressContainer) {
                progressContainer.style.display = 'block';
                progressContainer.innerHTML = `
                    <div class="alert alert-info">
                        <i class="fas fa-spinner fa-spin me-2"></i>
                        Starting batch test...
                    </div>
                `;
            }

            fetch('/api/start_batch_test', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    this.activeBatchTest = {
                        id: data.test_id,
                        total: data.total_prompts,
                        completed: 0
                    };
                    this.monitorBatchTest();
                } else {
                    throw new Error(data.error || 'Failed to start batch test');
                }
            })
            .catch(error => {
                console.error('Batch test error:', error);
                CoreModule.showAlert(`Batch test failed: ${error.message}`, 'danger');
                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }
            });
        },

        // Monitor batch test progress
        monitorBatchTest() {
            if (!this.activeBatchTest) return;

            const checkProgress = () => {
                fetch(`/api/batch_test_status/${this.activeBatchTest.id}`)
                    .then(response => response.json())
                    .then(data => {
                        this.updateBatchTestProgress(data);

                        if (data.status === 'completed') {
                            this.onBatchTestComplete(data);
                        } else if (data.status === 'failed') {
                            this.onBatchTestFailed(data);
                        } else {
                            // Continue monitoring
                            setTimeout(checkProgress, 2000);
                        }
                    })
                    .catch(error => {
                        console.error('Failed to check batch test status:', error);
                    });
            };

            checkProgress();
        },

        // Update batch test progress
        updateBatchTestProgress(data) {
            const progressContainer = document.getElementById('batch-test-progress');
            if (!progressContainer) return;

            const percentage = (data.completed / data.total) * 100;

            progressContainer.innerHTML = `
                <div class="progress mb-2">
                    <div class="progress-bar progress-bar-striped progress-bar-animated"
                         style="width: ${percentage}%">
                        ${Math.round(percentage)}%
                    </div>
                </div>
                <div class="text-muted">
                    Processing: ${data.completed} / ${data.total} prompts
                </div>
            `;
        },

        // Handle batch test completion
        onBatchTestComplete(data) {
            this.activeBatchTest = null;

            const resultsContainer = document.getElementById('batch-test-results');
            if (resultsContainer) {
                resultsContainer.innerHTML = `
                    <div class="alert alert-success">
                        <h5 class="alert-heading">Batch Test Complete!</h5>
                        <hr>
                        <div class="row">
                            <div class="col-md-3">
                                <strong>Total Prompts:</strong> ${data.total}
                            </div>
                            <div class="col-md-3">
                                <strong>Successful:</strong> ${data.successful}
                            </div>
                            <div class="col-md-3">
                                <strong>Failed:</strong> ${data.failed}
                            </div>
                            <div class="col-md-3">
                                <strong>Avg Time:</strong> ${data.avg_time?.toFixed(2)}s
                            </div>
                        </div>
                        <div class="mt-3">
                            <button class="btn btn-primary" onclick="TestingModule.downloadBatchResults('${data.results_file}')">
                                <i class="fas fa-download me-2"></i>Download Results
                            </button>
                            <button class="btn btn-secondary ms-2" onclick="TestingModule.viewBatchResults('${data.test_id}')">
                                <i class="fas fa-eye me-2"></i>View Details
                            </button>
                        </div>
                    </div>
                `;
            }

            // Hide progress
            const progressContainer = document.getElementById('batch-test-progress');
            if (progressContainer) {
                progressContainer.style.display = 'none';
            }
        },

        // Handle batch test failure
        onBatchTestFailed(data) {
            this.activeBatchTest = null;

            CoreModule.showAlert(`Batch test failed: ${data.error || 'Unknown error'}`, 'danger');

            const progressContainer = document.getElementById('batch-test-progress');
            if (progressContainer) {
                progressContainer.style.display = 'none';
            }
        },

        // Download batch test results
        downloadBatchResults(resultsFile) {
            window.location.href = `/api/download_batch_results/${resultsFile}`;
        },

        // View batch test results
        viewBatchResults(testId) {
            fetch(`/api/batch_test_results/${testId}`)
                .then(response => response.json())
                .then(data => {
                    this.showBatchResultsModal(data);
                })
                .catch(error => {
                    console.error('Failed to load batch test results:', error);
                    CoreModule.showAlert('Failed to load results', 'danger');
                });
        },

        // Show batch results modal
        showBatchResultsModal(data) {
            const modalId = 'batchResultsModal';
            let modal = document.getElementById(modalId);

            if (!modal) {
                const modalHtml = `
                    <div class="modal fade" id="${modalId}" tabindex="-1">
                        <div class="modal-dialog modal-xl">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">Batch Test Results</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body">
                                    <div id="batch-results-content"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
                modal = document.getElementById(modalId);
            }

            // Populate results
            const content = modal.querySelector('#batch-results-content');
            content.innerHTML = this.formatBatchResults(data);

            // Show modal
            new bootstrap.Modal(modal).show();
        },

        // Format batch results
        formatBatchResults(data) {
            return `
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Prompt</th>
                                <th>Response</th>
                                <th>Time (s)</th>
                                <th>Tokens</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${data.results.map((result, idx) => `
                                <tr>
                                    <td>${idx + 1}</td>
                                    <td class="text-truncate" style="max-width: 200px;" title="${CoreModule.escapeHtml(result.prompt)}">
                                        ${CoreModule.escapeHtml(result.prompt)}
                                    </td>
                                    <td class="text-truncate" style="max-width: 300px;" title="${CoreModule.escapeHtml(result.response)}">
                                        ${CoreModule.escapeHtml(result.response)}
                                    </td>
                                    <td>${result.generation_time?.toFixed(2) || 'N/A'}</td>
                                    <td>${result.token_count || 'N/A'}</td>
                                    <td>
                                        <span class="badge bg-${result.success ? 'success' : 'danger'}">
                                            ${result.success ? 'Success' : 'Failed'}
                                        </span>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            `;
        },

        // Display test history
        displayTestHistory() {
            const historyContainer = document.getElementById('test-history');
            if (!historyContainer) return;

            if (this.testHistory.length === 0) {
                historyContainer.innerHTML = '<p class="text-muted">No test history</p>';
                return;
            }

            historyContainer.innerHTML = `
                <div class="list-group">
                    ${this.testHistory.slice(0, 10).map(test => `
                        <div class="list-group-item">
                            <div class="d-flex justify-content-between">
                                <div>
                                    <h6 class="mb-1 text-truncate" style="max-width: 400px;">
                                        ${CoreModule.escapeHtml(test.prompt)}
                                    </h6>
                                    <small class="text-muted">
                                        ${new Date(test.timestamp).toLocaleString()}
                                    </small>
                                </div>
                                <button class="btn btn-sm btn-outline-secondary"
                                        onclick="TestingModule.loadHistoricalTest(${test.timestamp})">
                                    <i class="fas fa-redo"></i>
                                </button>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
        },

        // Load historical test
        loadHistoricalTest(timestamp) {
            const test = this.testHistory.find(t => t.timestamp === timestamp);
            if (!test) return;

            // Load test configuration
            document.getElementById('test-prompt').value = test.prompt;
            if (test.config) {
                if (test.config.temperature !== undefined) {
                    document.getElementById('test-temperature').value = test.config.temperature;
                    this.updateTemperatureDisplay();
                }
                if (test.config.top_p !== undefined) {
                    document.getElementById('test-top-p').value = test.config.top_p;
                    this.updateTopPDisplay();
                }
            }

            CoreModule.showAlert('Test configuration loaded', 'success');
        },

        // Update batch test status
        updateBatchTestStatus() {
            const statusContainer = document.getElementById('batch-test-status');
            if (!statusContainer) return;

            if (this.activeBatchTest) {
                statusContainer.innerHTML = `
                    <div class="alert alert-warning">
                        <i class="fas fa-spinner fa-spin me-2"></i>
                        Batch test in progress: ${this.activeBatchTest.completed} / ${this.activeBatchTest.total}
                    </div>
                `;
            } else {
                statusContainer.innerHTML = '';
            }
        },

        // Load list of trained models available for testing
        loadTestableModels() {
            const modelSelect = document.getElementById('test-model-select');
            const batchModelSelect = document.getElementById('batch-test-model-select');

            // Show loading state
            if (modelSelect) {
                modelSelect.innerHTML = '<option value="">Loading models...</option>';
            }
            if (batchModelSelect) {
                batchModelSelect.innerHTML = '<option value="">Loading models...</option>';
            }

            fetch('/api/test/models')
                .then(response => response.json())
                .then(data => {
                    if (data.models && data.models.length > 0) {
                        // Populate single test model select
                        if (modelSelect) {
                            modelSelect.innerHTML = '<option value="">Select a trained model...</option>';
                            data.models.forEach(model => {
                                const option = this.createModelOption(model);
                                modelSelect.appendChild(option);
                            });
                        }

                        // Populate batch test model select
                        if (batchModelSelect) {
                            batchModelSelect.innerHTML = '<option value="">Select a trained model...</option>';
                            data.models.forEach(model => {
                                const option = this.createModelOption(model);
                                batchModelSelect.appendChild(option);
                            });
                        }
                    } else {
                        if (modelSelect) {
                            modelSelect.innerHTML = '<option value="">No trained models available</option>';
                        }
                        if (batchModelSelect) {
                            batchModelSelect.innerHTML = '<option value="">No trained models available</option>';
                        }
                    }
                })
                .catch(error => {
                    console.error('Failed to load testable models:', error);
                    if (modelSelect) {
                        modelSelect.innerHTML = '<option value="">Error loading models</option>';
                    }
                    if (batchModelSelect) {
                        batchModelSelect.innerHTML = '<option value="">Error loading models</option>';
                    }
                    CoreModule.showAlert('Failed to load trained models', 'danger');
                });
        },

        // Create a model option element
        createModelOption(model) {
            const option = document.createElement('option');
            option.value = model.session_id;
            option.dataset.baseModel = model.base_model;
            option.dataset.checkpointPath = model.checkpoint_path;

            // Get display name with fallbacks
            let displayName = model.display_name || model.model_name;
            if (!displayName || displayName === 'Unknown') {
                displayName = model.base_model || model.session_id;
            }

            // Get epochs with fallback
            const epochs = model.epochs || model.num_epochs || 0;
            option.dataset.epochs = epochs;

            option.textContent = `${displayName} (${epochs} epochs)`;
            return option;
        },

        // Compare two models side-by-side
        compareModels() {
            const modelSelect = document.getElementById('test-model-select');
            const prompt = document.getElementById('test-prompt')?.value;
            const comparisonMode = document.querySelector('input[name="comparison-mode"]:checked')?.value;

            if (!modelSelect?.value) {
                CoreModule.showAlert('Please select a model to compare', 'warning');
                return;
            }

            if (!prompt) {
                CoreModule.showAlert('Please enter a test prompt', 'warning');
                return;
            }

            const sessionId = modelSelect.value;
            const selectedOption = modelSelect.options[modelSelect.selectedIndex];
            const baseModel = selectedOption.dataset.baseModel;

            // Show loading state
            const resultsContainer = document.getElementById('comparison-results');
            if (resultsContainer) {
                resultsContainer.style.display = 'block';
                resultsContainer.innerHTML = `
                    <div class="alert alert-info">
                        <i class="fas fa-spinner fa-spin me-2"></i>
                        Comparing models...
                    </div>
                `;
            }

            // Prepare generation config
            const genConfig = {
                temperature: parseFloat(document.getElementById('test-temperature')?.value) || 0.7,
                top_p: parseFloat(document.getElementById('test-top-p')?.value) || 0.95,
                max_new_tokens: parseInt(document.getElementById('test-max-tokens')?.value) || 512,
                repetition_penalty: parseFloat(document.getElementById('test-rep-penalty')?.value) || 1.0,
                do_sample: true
            };

            const useChatTemplate = document.getElementById('use-chat-template')?.checked ?? true;

            let endpoint, config;

            if (comparisonMode === 'model') {
                // Model-to-model comparison
                const comparisonSelect = document.getElementById('comparison-model-select');
                if (!comparisonSelect?.value) {
                    CoreModule.showAlert('Please select a comparison model', 'warning');
                    if (resultsContainer) {
                        resultsContainer.style.display = 'none';
                    }
                    return;
                }

                endpoint = '/api/test/compare-models';
                config = {
                    prompt: prompt,
                    model1_session_id: sessionId,
                    model2_session_id: comparisonSelect.value,
                    config: genConfig,
                    use_chat_template: useChatTemplate
                };
            } else {
                // Base model comparison
                if (!baseModel) {
                    CoreModule.showAlert('Base model information not available', 'danger');
                    if (resultsContainer) {
                        resultsContainer.style.display = 'none';
                    }
                    return;
                }

                endpoint = '/api/test/compare';
                config = {
                    prompt: prompt,
                    session_id: sessionId,
                    base_model: baseModel,
                    config: genConfig,
                    use_chat_template: useChatTemplate
                };
            }

            // Execute comparison with streaming
            this.compareWithStreaming(endpoint + '/stream', config, resultsContainer);
        },

        // Compare models with streaming token-by-token display
        async compareWithStreaming(endpoint, config, resultsContainer) {
            try {
                // Initialize streaming display
                this.initStreamingDisplay(resultsContainer);

                // Debug logging
                console.log('Streaming comparison request:', { endpoint, config });

                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });

                if (!response.ok) {
                    // Try to get detailed error message from response
                    let errorMessage = `HTTP error! status: ${response.status}`;
                    try {
                        const errorData = await response.json();
                        errorMessage = errorData.error || errorData.message || errorMessage;
                    } catch (e) {
                        // Response might not be JSON
                        try {
                            const textError = await response.text();
                            if (textError) {
                                errorMessage = textError;
                            }
                        } catch (e2) {
                            // Keep generic message
                        }
                    }
                    console.error('Streaming fetch failed:', errorMessage);
                    throw new Error(errorMessage);
                }

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();

                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop(); // Keep incomplete line in buffer

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = JSON.parse(line.slice(6));
                            this.handleStreamEvent(data);
                        }
                    }
                }

            } catch (error) {
                console.error('Streaming comparison error:', error);
                if (resultsContainer) {
                    resultsContainer.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle me-2"></i>
                            ${CoreModule.escapeHtml(error.message)}
                        </div>
                    `;
                }
                CoreModule.showAlert(`Comparison failed: ${error.message}`, 'danger');
            }
        },

        // Initialize streaming display UI
        initStreamingDisplay(resultsContainer) {
            resultsContainer.style.display = 'block';
            resultsContainer.innerHTML = `
                <div class="row">
                    <!-- Trained Model Column -->
                    <div class="col-md-6">
                        <div class="card h-100">
                            <div class="card-header bg-primary text-white">
                                <div class="d-flex justify-content-between align-items-center">
                                    <h6 class="mb-0"><i class="fas fa-robot me-2"></i>Trained Model</h6>
                                    <span class="badge bg-light text-primary" id="trained-status">
                                        <i class="fas fa-spinner fa-spin me-1"></i>Generating...
                                    </span>
                                </div>
                            </div>
                            <div class="card-body">
                                <div id="trained-response" class="streaming-text mb-3" style="white-space: pre-wrap; min-height: 200px; max-height: 400px; overflow-y: auto;"></div>
                                <div id="trained-stats" class="small text-muted" style="display: none;"></div>
                            </div>
                        </div>
                    </div>

                    <!-- Base/Comparison Model Column -->
                    <div class="col-md-6">
                        <div class="card h-100">
                            <div class="card-header bg-secondary text-white">
                                <div class="d-flex justify-content-between align-items-center">
                                    <h6 class="mb-0"><i class="fas fa-cube me-2"></i>Base Model</h6>
                                    <span class="badge bg-light text-dark" id="base-status">
                                        <i class="fas fa-spinner fa-spin me-1"></i>Generating...
                                    </span>
                                </div>
                            </div>
                            <div class="card-body">
                                <div id="base-response" class="streaming-text mb-3" style="white-space: pre-wrap; min-height: 200px; max-height: 400px; overflow-y: auto;"></div>
                                <div id="base-stats" class="small text-muted" style="display: none;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        },

        // Handle streaming events
        handleStreamEvent(data) {
            const trainedResponse = document.getElementById('trained-response');
            const baseResponse = document.getElementById('base-response');
            const trainedStatus = document.getElementById('trained-status');
            const baseStatus = document.getElementById('base-status');
            const trainedStats = document.getElementById('trained-stats');
            const baseStats = document.getElementById('base-stats');

            switch (data.type) {
                case 'trained':
                    // Append token to trained model response
                    if (trainedResponse) {
                        trainedResponse.textContent += data.token;
                        trainedResponse.scrollTop = trainedResponse.scrollHeight;
                    }
                    break;

                case 'base':
                    // Append token to base model response
                    if (baseResponse) {
                        baseResponse.textContent += data.token;
                        baseResponse.scrollTop = baseResponse.scrollHeight;
                    }
                    break;

                case 'trained_complete':
                    // Mark trained model as complete
                    if (trainedStatus) {
                        trainedStatus.innerHTML = '<i class="fas fa-check-circle me-1"></i>Complete';
                        trainedStatus.classList.remove('bg-light', 'text-primary');
                        trainedStatus.classList.add('bg-success');
                    }
                    break;

                case 'base_complete':
                    // Mark base model as complete
                    if (baseStatus) {
                        baseStatus.innerHTML = '<i class="fas fa-check-circle me-1"></i>Complete';
                        baseStatus.classList.remove('bg-light', 'text-dark');
                        baseStatus.classList.add('bg-success');
                    }
                    break;

                case 'complete':
                    // Show final statistics
                    const trainedResult = data.trained || {};
                    const baseResult = data.base || {};

                    if (trainedStats && trainedResult.success) {
                        trainedStats.style.display = 'block';
                        trainedStats.innerHTML = `
                            <strong>Time:</strong> ${trainedResult.metadata?.generation_time?.toFixed(2) || 'N/A'}s |
                            <strong>Tokens:</strong> ${trainedResult.metadata?.token_count || 'N/A'} |
                            <strong>Speed:</strong> ${trainedResult.metadata?.tokens_per_second?.toFixed(1) || 'N/A'} tok/s
                        `;
                    }

                    if (baseStats && baseResult.success) {
                        baseStats.style.display = 'block';
                        baseStats.innerHTML = `
                            <strong>Time:</strong> ${baseResult.metadata?.generation_time?.toFixed(2) || 'N/A'}s |
                            <strong>Tokens:</strong> ${baseResult.metadata?.token_count || 'N/A'} |
                            <strong>Speed:</strong> ${baseResult.metadata?.tokens_per_second?.toFixed(1) || 'N/A'} tok/s
                        `;
                    }

                    // Check for expected answer and show match status
                    const expectedAnswer = document.getElementById('test-expected-answer')?.value?.trim();
                    if (expectedAnswer) {
                        this.displayMatchStatus(trainedResult, baseResult, expectedAnswer);
                    }
                    break;

                case 'error':
                    // Show error
                    const resultsContainer = document.getElementById('comparison-results');
                    if (resultsContainer) {
                        resultsContainer.innerHTML = `
                            <div class="alert alert-danger">
                                <i class="fas fa-exclamation-circle me-2"></i>
                                ${CoreModule.escapeHtml(data.error)}
                            </div>
                        `;
                    }
                    CoreModule.showAlert(`Error: ${data.error}`, 'danger');
                    break;
            }
        },

        // Display match status for expected answer
        displayMatchStatus(trainedResult, baseResult, expectedAnswer) {
            const trainedMatchContainer = document.getElementById('trained-match-status');
            const baseMatchContainer = document.getElementById('base-match-status');
            const trainedMatchAlert = document.getElementById('trained-match-alert');
            const baseMatchAlert = document.getElementById('base-match-alert');
            const trainedMatchText = document.getElementById('trained-match-text');
            const baseMatchText = document.getElementById('base-match-text');

            // Case-insensitive exact match comparison
            const normalizedExpected = expectedAnswer.toLowerCase().trim();

            if (trainedResult.success && trainedResult.response) {
                const normalizedTrained = trainedResult.response.toLowerCase().trim();
                const trainedMatches = normalizedTrained === normalizedExpected;

                if (trainedMatchContainer && trainedMatchAlert && trainedMatchText) {
                    trainedMatchContainer.style.display = 'block';
                    if (trainedMatches) {
                        trainedMatchAlert.className = 'alert alert-sm alert-success mb-0';
                        trainedMatchText.innerHTML = '<i class="fas fa-check-circle me-1"></i> Matches expected answer';
                    } else {
                        trainedMatchAlert.className = 'alert alert-sm alert-warning mb-0';
                        trainedMatchText.innerHTML = '<i class="fas fa-times-circle me-1"></i> Does not match expected answer';
                    }
                }
            }

            if (baseResult.success && baseResult.response) {
                const normalizedBase = baseResult.response.toLowerCase().trim();
                const baseMatches = normalizedBase === normalizedExpected;

                if (baseMatchContainer && baseMatchAlert && baseMatchText) {
                    baseMatchContainer.style.display = 'block';
                    if (baseMatches) {
                        baseMatchAlert.className = 'alert alert-sm alert-success mb-0';
                        baseMatchText.innerHTML = '<i class="fas fa-check-circle me-1"></i> Matches expected answer';
                    } else {
                        baseMatchAlert.className = 'alert alert-sm alert-warning mb-0';
                        baseMatchText.innerHTML = '<i class="fas fa-times-circle me-1"></i> Does not match expected answer';
                    }
                }
            }
        },

        // Display comparison results
        displayComparisonResults(data) {
            const resultsContainer = document.getElementById('comparison-results');
            if (!resultsContainer) return;

            const results = data.results || data;
            const trainedResult = results.trained || {};
            const baseResult = results.base || {};

            resultsContainer.style.display = 'block';
            resultsContainer.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-primary text-white">
                                <h6 class="mb-0"><i class="fas fa-robot me-2"></i>Trained Model</h6>
                            </div>
                            <div class="card-body">
                                <pre class="mb-2" style="white-space: pre-wrap; max-height: 300px; overflow-y: auto;">${CoreModule.escapeHtml(trainedResult.response || 'No response')}</pre>
                                <div class="small text-muted">
                                    <strong>Time:</strong> ${trainedResult.metadata?.generation_time?.toFixed(2) || 'N/A'}s |
                                    <strong>Tokens:</strong> ${trainedResult.metadata?.token_count || 'N/A'} |
                                    <strong>Speed:</strong> ${trainedResult.metadata?.tokens_per_second?.toFixed(1) || 'N/A'} tok/s
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="card">
                            <div class="card-header bg-secondary text-white">
                                <h6 class="mb-0"><i class="fas fa-cube me-2"></i>Base Model</h6>
                            </div>
                            <div class="card-body">
                                <pre class="mb-2" style="white-space: pre-wrap; max-height: 300px; overflow-y: auto;">${CoreModule.escapeHtml(baseResult.response || 'No response')}</pre>
                                <div class="small text-muted">
                                    <strong>Time:</strong> ${baseResult.metadata?.generation_time?.toFixed(2) || 'N/A'}s |
                                    <strong>Tokens:</strong> ${baseResult.metadata?.token_count || 'N/A'} |
                                    <strong>Speed:</strong> ${baseResult.metadata?.tokens_per_second?.toFixed(1) || 'N/A'} tok/s
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        },

        // Handle batch test file upload
        handleBatchTestFileUpload() {
            const fileInput = document.getElementById('batch-test-file');
            const file = fileInput?.files[0];

            if (!file) {
                return;
            }

            // Show uploading state
            const fileInfoDiv = document.getElementById('batch-file-info');
            const fileDetailsSpan = document.getElementById('batch-file-details');

            if (fileInfoDiv && fileDetailsSpan) {
                fileInfoDiv.style.display = 'block';
                fileDetailsSpan.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Uploading and analyzing file...';
            }

            // Create form data
            const formData = new FormData();
            formData.append('file', file);

            // Upload file
            fetch('/api/test/upload-file', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Update file info display
                    if (fileDetailsSpan) {
                        fileDetailsSpan.innerHTML = `
                            <i class="fas fa-file-csv me-2"></i>
                            <strong>${CoreModule.escapeHtml(data.filename)}</strong> -
                            ${data.sample_count} rows, ${data.columns.length} columns
                        `;
                    }

                    // Populate column dropdowns
                    this.populateColumnSelects(data.columns, data.sample_count);

                    // Show column configuration section
                    const columnConfig = document.getElementById('batch-column-config');
                    if (columnConfig) {
                        columnConfig.style.display = 'block';
                    }

                    // Show test options section
                    const testOptions = document.getElementById('batch-test-options');
                    if (testOptions) {
                        testOptions.style.display = 'block';
                    }

                    // Enable the run button
                    const runButton = document.getElementById('batch-compare-btn');
                    if (runButton) {
                        runButton.disabled = false;
                    }

                    CoreModule.showAlert('File uploaded and analyzed successfully', 'success');
                } else {
                    throw new Error(data.error || 'Failed to upload file');
                }
            })
            .catch(error => {
                console.error('File upload error:', error);
                CoreModule.showAlert(`Failed to upload file: ${error.message}`, 'danger');

                // Reset file input
                if (fileInput) {
                    fileInput.value = '';
                }

                // Hide file info
                if (fileInfoDiv) {
                    fileInfoDiv.style.display = 'none';
                }
            });
        },

        // Populate column selection dropdowns
        populateColumnSelects(columns, sampleCount) {
            const instructionSelect = document.getElementById('batch-instruction-column');
            const responseSelect = document.getElementById('batch-response-column');
            const totalSamplesSpan = document.getElementById('batch-total-samples');

            // Update total samples display
            if (totalSamplesSpan) {
                totalSamplesSpan.textContent = `of ${sampleCount}`;
            }

            // Populate instruction column dropdown
            if (instructionSelect) {
                instructionSelect.innerHTML = '<option value="">Select column...</option>';
                columns.forEach(column => {
                    const option = document.createElement('option');
                    option.value = column;
                    option.textContent = column;
                    // Auto-select common instruction column names
                    if (['instruction', 'prompt', 'input', 'question'].includes(column.toLowerCase())) {
                        option.selected = true;
                    }
                    instructionSelect.appendChild(option);
                });
            }

            // Populate response column dropdown
            if (responseSelect) {
                responseSelect.innerHTML = '<option value="">None - Just compare models</option>';
                columns.forEach(column => {
                    const option = document.createElement('option');
                    option.value = column;
                    option.textContent = column;
                    // Auto-select common response column names
                    if (['response', 'output', 'answer', 'expected'].includes(column.toLowerCase())) {
                        option.selected = true;
                    }
                    responseSelect.appendChild(option);
                });
            }
        },

        // Update batch model selection
        updateBatchModelSelection() {
            const modelSelect = document.getElementById('batch-test-model-select');
            const comparisonMode = document.querySelector('input[name="batch-compare-mode"]:checked')?.value;
            const baseModelGroup = document.getElementById('batch-base-model-group');
            const comparisonModelGroup = document.getElementById('batch-comparison-model-group');
            const baseModelInput = document.getElementById('batch-base-model-input');

            if (!modelSelect || !modelSelect.value) {
                return;
            }

            const selectedOption = modelSelect.options[modelSelect.selectedIndex];
            const baseModel = selectedOption.dataset.baseModel;

            // Update base model display
            if (baseModelInput && baseModel) {
                baseModelInput.value = baseModel;
            }

            // Toggle comparison mode sections
            if (comparisonMode === 'base') {
                if (baseModelGroup) baseModelGroup.style.display = 'block';
                if (comparisonModelGroup) comparisonModelGroup.style.display = 'none';
            } else {
                if (baseModelGroup) baseModelGroup.style.display = 'none';
                if (comparisonModelGroup) comparisonModelGroup.style.display = 'block';

                // Populate comparison model dropdown if not already done
                this.populateBatchComparisonModels(modelSelect.value);
            }
        },

        // Populate batch comparison model dropdown
        populateBatchComparisonModels(excludeSessionId) {
            const comparisonSelect = document.getElementById('batch-comparison-model-select');
            if (!comparisonSelect) return;

            // Only populate once
            if (comparisonSelect.options.length > 1) return;

            fetch('/api/test/models')
                .then(response => response.json())
                .then(data => {
                    if (data.models && data.models.length > 0) {
                        comparisonSelect.innerHTML = '<option value="">Select another trained model...</option>';
                        data.models.forEach(model => {
                            // Exclude the primary model from comparison options
                            if (model.session_id !== excludeSessionId) {
                                const option = document.createElement('option');
                                option.value = model.session_id;
                                option.dataset.baseModel = model.base_model;

                                let displayName = model.display_name || model.model_name || model.base_model;
                                const epochs = model.epochs || model.num_epochs || 0;
                                option.textContent = `${displayName} (${epochs} epochs)`;

                                comparisonSelect.appendChild(option);
                            }
                        });
                    }
                })
                .catch(error => {
                    console.error('Failed to load comparison models:', error);
                });
        },

        // Clear uploaded batch test file
        clearBatchTestFile() {
            const fileInput = document.getElementById('batch-test-file');
            if (fileInput) {
                fileInput.value = '';
            }

            const fileNameDisplay = document.getElementById('batch-file-name');
            if (fileNameDisplay) {
                fileNameDisplay.textContent = 'No file selected';
            }

            // Clear server-side session data
            fetch('/api/test/clear-cache', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ clear_batch_file: true })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    CoreModule.showAlert('Batch test file cleared', 'success');
                } else {
                    console.warn('Failed to clear batch file from server');
                }
            })
            .catch(error => {
                console.error('Error clearing batch file:', error);
            });
        },

        // Run batch comparison test
        runBatchComparison() {
            const modelSelect = document.getElementById('batch-test-model-select');
            const instructionColumn = document.getElementById('batch-instruction-column')?.value;
            const responseColumn = document.getElementById('batch-response-column')?.value;
            const sampleSize = document.getElementById('batch-sample-size')?.value;
            const comparisonMode = document.querySelector('input[name="batch-compare-mode"]:checked')?.value || 'base';

            if (!modelSelect?.value) {
                CoreModule.showAlert('Please select a trained model', 'warning');
                return;
            }

            if (!instructionColumn) {
                CoreModule.showAlert('Please specify the instruction column', 'warning');
                return;
            }

            const sessionId = modelSelect.value;
            const selectedOption = modelSelect.options[modelSelect.selectedIndex];
            const baseModel = selectedOption.dataset.baseModel;

            // Prepare batch comparison config
            const config = {
                session_id: sessionId,
                instruction_column: instructionColumn,
                response_column: responseColumn || null,
                sample_size: sampleSize ? parseInt(sampleSize) : null,
                use_chat_template: document.getElementById('batch-use-chat-template')?.checked !== false,
                compare_mode: comparisonMode,
                config: {
                    temperature: parseFloat(document.getElementById('test-temperature')?.value) || 0.7,
                    top_p: parseFloat(document.getElementById('test-top-p')?.value) || 0.95,
                    max_new_tokens: parseInt(document.getElementById('test-max-tokens')?.value) || 512,
                    repetition_penalty: parseFloat(document.getElementById('test-rep-penalty')?.value) || 1.0,
                    do_sample: true
                }
            };

            // Add appropriate comparison model info based on mode
            if (comparisonMode === 'base') {
                if (!baseModel) {
                    CoreModule.showAlert('Base model information not available', 'danger');
                    return;
                }
                config.base_model = baseModel;
            } else {
                // Model-to-model comparison
                const comparisonModelSelect = document.getElementById('batch-comparison-model-select');
                if (!comparisonModelSelect?.value) {
                    CoreModule.showAlert('Please select a comparison model', 'warning');
                    return;
                }
                config.comparison_session_id = comparisonModelSelect.value;
            }

            // Show loading state
            const progressContainer = document.getElementById('batch-progress');
            const resultsContainer = document.getElementById('batch-results');

            if (resultsContainer) {
                resultsContainer.style.display = 'block';
            }

            if (progressContainer) {
                progressContainer.style.display = 'block';
                progressContainer.innerHTML = `
                    <div class="d-flex align-items-center justify-content-between mb-2">
                        <span class="text-muted">Processing:</span>
                        <span id="batch-step-counter" class="badge bg-info">0 / 0</span>
                    </div>
                    <div class="progress mb-2">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%"></div>
                    </div>
                    <div id="batch-current-item" class="text-center text-muted small mb-3">
                        <i class="fas fa-spinner fa-spin"></i> <span>Starting batch comparison...</span>
                    </div>
                `;
            }

            // Hide previous summary and results
            const summaryContainer = document.getElementById('batch-summary');
            if (summaryContainer) {
                summaryContainer.style.display = 'none';
            }

            const resultsBody = document.getElementById('batch-results-body');
            if (resultsBody) {
                resultsBody.innerHTML = '';
            }

            // Show cancel button, hide run button
            const runButton = document.getElementById('batch-compare-btn');
            const cancelButton = document.getElementById('batch-cancel-btn');
            if (runButton) runButton.style.display = 'none';
            if (cancelButton) cancelButton.style.display = 'block';

            // Start batch comparison
            fetch('/api/test/batch-compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success && data.batch_id) {
                    this.activeBatchTest = {
                        id: data.batch_id,
                        total: 0,
                        completed: 0
                    };
                    CoreModule.showAlert('Batch comparison started', 'success');
                    this.monitorBatchComparison(data.batch_id);
                } else {
                    throw new Error(data.error || 'Failed to start batch comparison');
                }
            })
            .catch(error => {
                console.error('Batch comparison error:', error);
                CoreModule.showAlert(`Failed to start batch comparison: ${error.message}`, 'danger');

                if (progressContainer) {
                    progressContainer.style.display = 'none';
                }

                // Restore buttons
                if (runButton) runButton.style.display = 'block';
                if (cancelButton) cancelButton.style.display = 'none';
            });
        },

        // Monitor batch comparison progress
        monitorBatchComparison(batchId) {
            const checkProgress = () => {
                fetch(`/api/test/batch-status/${batchId}`)
                    .then(response => response.json())
                    .then(data => {
                        this.updateBatchComparisonProgress(data);

                        if (data.status === 'completed') {
                            this.onBatchComparisonComplete(batchId);
                        } else if (data.status === 'error' || data.status === 'failed') {
                            this.onBatchComparisonFailed(data);
                        } else if (data.status === 'cancelled') {
                            CoreModule.showAlert('Batch comparison was cancelled', 'warning');
                            this.activeBatchTest = null;
                        } else {
                            // Continue monitoring
                            setTimeout(checkProgress, 2000);
                        }
                    })
                    .catch(error => {
                        console.error('Failed to check batch status:', error);
                        setTimeout(checkProgress, 3000); // Retry with longer delay
                    });
            };

            checkProgress();
        },

        // Update batch comparison progress
        updateBatchComparisonProgress(data) {
            const progressContainer = document.getElementById('batch-test-progress');
            if (!progressContainer) return;

            const percentage = data.total > 0 ? (data.progress / data.total) * 100 : 0;

            progressContainer.style.display = 'block';
            progressContainer.innerHTML = `
                <div class="progress mb-2">
                    <div class="progress-bar progress-bar-striped progress-bar-animated"
                         style="width: ${percentage}%">
                        ${Math.round(percentage)}%
                    </div>
                </div>
                <div class="text-muted">
                    Processing: ${data.progress || 0} / ${data.total || 0} prompts
                    ${data.current_instruction ? `<br><small>Current: ${CoreModule.escapeHtml(data.current_instruction.substring(0, 80))}...</small>` : ''}
                </div>
                <button class="btn btn-sm btn-danger mt-2" onclick="TestingModule.cancelBatchTest()">
                    <i class="fas fa-stop me-1"></i>Cancel Test
                </button>
            `;
        },

        // Handle batch comparison completion
        onBatchComparisonComplete(batchId) {
            this.activeBatchTest = null;

            fetch(`/api/test/batch-results/${batchId}`)
                .then(response => response.json())
                .then(data => {
                    // Hide progress
                    const progressContainer = document.getElementById('batch-progress');
                    if (progressContainer) {
                        progressContainer.style.display = 'none';
                    }

                    // Restore buttons
                    const runButton = document.getElementById('batch-compare-btn');
                    const cancelButton = document.getElementById('batch-cancel-btn');
                    if (runButton) runButton.style.display = 'block';
                    if (cancelButton) cancelButton.style.display = 'none';

                    // Display summary metrics
                    this.displayBatchSummary(data);

                    // Display results table
                    this.displayBatchResultsTable(data);

                    CoreModule.showAlert('Batch comparison completed successfully', 'success');
                })
                .catch(error => {
                    console.error('Failed to load batch results:', error);
                    CoreModule.showAlert('Batch comparison completed but failed to load results', 'warning');
                });
        },

        // Display batch test summary
        displayBatchSummary(data) {
            const summaryContainer = document.getElementById('batch-summary');
            if (!summaryContainer) return;

            const summary = data.summary || {};
            const hasExpected = data.results && data.results.length > 0 && 'expected' in data.results[0];

            summaryContainer.style.display = 'flex';

            // Update summary metrics
            if (hasExpected && summary.trained_accuracy !== undefined) {
                document.getElementById('batch-accuracy').textContent = `${summary.trained_accuracy}%`;
            } else {
                document.getElementById('batch-accuracy').textContent = 'N/A';
            }

            document.getElementById('batch-avg-length').textContent =
                summary.avg_trained_length ? `${Math.round(summary.avg_trained_length)} chars` : 'N/A';

            document.getElementById('batch-avg-time').textContent =
                summary.avg_trained_time ? `${summary.avg_trained_time}s` : 'N/A';

            document.getElementById('batch-total-samples').textContent = summary.total_samples || 0;
        },

        // Display batch results table
        displayBatchResultsTable(data) {
            const resultsBody = document.getElementById('batch-results-body');
            const expectedHeader = document.getElementById('expected-header');
            const matchHeader = document.getElementById('match-header');

            if (!resultsBody) return;

            const results = data.results || [];
            const hasExpected = results.length > 0 && 'expected' in results[0];

            // Show/hide expected answer columns
            if (expectedHeader) {
                expectedHeader.style.display = hasExpected ? 'table-cell' : 'none';
            }
            if (matchHeader) {
                matchHeader.style.display = hasExpected ? 'table-cell' : 'none';
            }

            // Clear previous results
            resultsBody.innerHTML = '';

            // Populate results table
            results.forEach((result, idx) => {
                const row = document.createElement('tr');

                const trainedMatch = result.trained_match === true;
                const comparisonMatch = result.comparison_match === true;

                let matchCell = '';
                if (hasExpected) {
                    matchCell = `
                        <td style="display: ${hasExpected ? 'table-cell' : 'none'};">
                            ${CoreModule.escapeHtml(result.expected || 'N/A')}
                        </td>
                        <td style="display: ${hasExpected ? 'table-cell' : 'none'};">
                            <div class="d-flex gap-2">
                                <span class="badge bg-${trainedMatch ? 'success' : 'warning'}">
                                    <i class="fas fa-${trainedMatch ? 'check' : 'times'}"></i> Trained
                                </span>
                                <span class="badge bg-${comparisonMatch ? 'success' : 'warning'}">
                                    <i class="fas fa-${comparisonMatch ? 'check' : 'times'}"></i> Comparison
                                </span>
                            </div>
                        </td>
                    `;
                } else {
                    matchCell = `
                        <td style="display: none;"></td>
                        <td style="display: none;"></td>
                    `;
                }

                row.innerHTML = `
                    <td>${idx + 1}</td>
                    <td class="text-truncate" style="max-width: 200px;" title="${CoreModule.escapeHtml(result.instruction || '')}">
                        ${CoreModule.escapeHtml((result.instruction || '').substring(0, 100))}${result.instruction && result.instruction.length > 100 ? '...' : ''}
                    </td>
                    <td class="text-truncate" style="max-width: 250px;" title="${CoreModule.escapeHtml(result.trained_response || '')}">
                        ${CoreModule.escapeHtml((result.trained_response || '').substring(0, 100))}${result.trained_response && result.trained_response.length > 100 ? '...' : ''}
                    </td>
                    <td class="text-truncate" style="max-width: 250px;" title="${CoreModule.escapeHtml(result.comparison_response || '')}">
                        ${CoreModule.escapeHtml((result.comparison_response || '').substring(0, 100))}${result.comparison_response && result.comparison_response.length > 100 ? '...' : ''}
                    </td>
                    ${matchCell}
                `;

                resultsBody.appendChild(row);
            });

            // Store results for export
            this.currentBatchResults = data;
        },

        // Handle batch comparison failure
        onBatchComparisonFailed(data) {
            this.activeBatchTest = null;

            CoreModule.showAlert(`Batch comparison failed: ${data.error || 'Unknown error'}`, 'danger');

            const progressContainer = document.getElementById('batch-test-progress');
            if (progressContainer) {
                progressContainer.style.display = 'none';
            }

            const resultsContainer = document.getElementById('batch-test-results');
            if (resultsContainer) {
                resultsContainer.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Batch comparison failed: ${CoreModule.escapeHtml(data.error || 'Unknown error')}
                    </div>
                `;
            }
        },

        // Cancel running batch test
        cancelBatchTest() {
            if (!this.activeBatchTest || !this.activeBatchTest.id) {
                CoreModule.showAlert('No active batch test to cancel', 'warning');
                return;
            }

            const batchId = this.activeBatchTest.id;

            CoreModule.showConfirmModal(
                'Cancel Batch Test',
                'Are you sure you want to cancel this batch test? Progress will be lost.',
                () => {
                    fetch(`/api/test/batch-cancel/${batchId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' }
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            CoreModule.showAlert('Batch test cancelled', 'success');
                            this.activeBatchTest = null;

                            const progressContainer = document.getElementById('batch-test-progress');
                            if (progressContainer) {
                                progressContainer.style.display = 'none';
                            }
                        } else {
                            throw new Error(data.error || 'Failed to cancel batch test');
                        }
                    })
                    .catch(error => {
                        console.error('Failed to cancel batch test:', error);
                        CoreModule.showAlert(`Failed to cancel: ${error.message}`, 'danger');
                    });
                },
                'btn-danger'
            );
        },

        // Export batch test results to file
        exportBatchResults(batchId) {
            // Use stored results if available, otherwise fetch
            if (this.currentBatchResults) {
                this.doExportBatchResults(this.currentBatchResults, batchId);
            } else if (batchId) {
                // Show loading state
                CoreModule.showAlert('Preparing export...', 'info');

                fetch(`/api/test/batch-results/${batchId}`)
                    .then(response => response.json())
                    .then(data => {
                        this.doExportBatchResults(data, batchId);
                    })
                    .catch(error => {
                        console.error('Export failed:', error);
                        CoreModule.showAlert(`Failed to export results: ${error.message}`, 'danger');
                    });
            } else {
                CoreModule.showAlert('No batch results to export', 'warning');
            }
        },

        // Actually perform the CSV export
        doExportBatchResults(data, batchId) {
            if (!data.results) {
                CoreModule.showAlert('No results data available', 'warning');
                return;
            }

            const results = data.results;
            const hasExpected = results.length > 0 && 'expected' in results[0];

            // Build headers based on what data we have
            const headers = ['ID', 'Instruction', 'Trained Response', 'Comparison Response', 'Trained Time (s)', 'Comparison Time (s)'];
            if (hasExpected) {
                headers.push('Expected Answer', 'Trained Match', 'Comparison Match');
            }

            const csvRows = [headers.join(',')];

            // Build data rows
            results.forEach((result, idx) => {
                const row = [
                    idx + 1,
                    `"${(result.instruction || '').replace(/"/g, '""')}"`,
                    `"${(result.trained_response || '').replace(/"/g, '""')}"`,
                    `"${(result.comparison_response || '').replace(/"/g, '""')}"`,
                    result.trained_time?.toFixed(2) || 'N/A',
                    result.comparison_time?.toFixed(2) || 'N/A'
                ];

                if (hasExpected) {
                    row.push(
                        `"${(result.expected || '').replace(/"/g, '""')}"`,
                        result.trained_match ? 'Yes' : 'No',
                        result.comparison_match ? 'Yes' : 'No'
                    );
                }

                csvRows.push(row.join(','));
            });

            // Add summary section
            if (data.summary) {
                const summary = data.summary;
                csvRows.push('');
                csvRows.push('Summary');
                csvRows.push(`Total Samples,${summary.total_samples || 0}`);
                csvRows.push(`Avg Trained Time (s),${summary.avg_trained_time || 'N/A'}`);
                csvRows.push(`Avg Comparison Time (s),${summary.avg_comparison_time || 'N/A'}`);
                csvRows.push(`Avg Trained Length,${summary.avg_trained_length || 'N/A'}`);
                csvRows.push(`Avg Comparison Length,${summary.avg_comparison_length || 'N/A'}`);

                if (hasExpected) {
                    csvRows.push(`Trained Accuracy (%),${summary.trained_accuracy || 'N/A'}`);
                    csvRows.push(`Comparison Accuracy (%),${summary.comparison_accuracy || 'N/A'}`);
                }
            }

            // Create and download file
            const csvContent = csvRows.join('\n');
            const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
            const url = window.URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = `batch_comparison_${batchId || Date.now()}.csv`;
            link.click();
            window.URL.revokeObjectURL(url);

            CoreModule.showAlert('Batch results exported successfully', 'success');
        },

        // Run evaluation on test dataset
        runEvaluation() {
            const modelSelect = document.getElementById('test-model-select');
            const testCasesInput = document.getElementById('eval-test-cases')?.value;
            const promptTemplate = document.getElementById('eval-prompt-template')?.value || '{input}';

            if (!modelSelect?.value) {
                CoreModule.showAlert('Please select a model to evaluate', 'warning');
                return;
            }

            if (!testCasesInput) {
                CoreModule.showAlert('Please provide test cases (JSON format)', 'warning');
                return;
            }

            let testCases;
            try {
                testCases = JSON.parse(testCasesInput);
                if (!Array.isArray(testCases)) {
                    throw new Error('Test cases must be an array');
                }
            } catch (error) {
                CoreModule.showAlert('Invalid test cases format. Expected JSON array.', 'danger');
                return;
            }

            // Show loading state
            const resultsContainer = document.getElementById('eval-results');
            if (resultsContainer) {
                resultsContainer.style.display = 'block';
                resultsContainer.innerHTML = `
                    <div class="card-body">
                        <div class="alert alert-info">
                            <i class="fas fa-spinner fa-spin me-2"></i>
                            Running evaluation on ${testCases.length} test cases...
                        </div>
                    </div>
                `;
            }

            const config = {
                session_id: modelSelect.value,
                test_cases: testCases,
                prompt_template: promptTemplate,
                config: {
                    temperature: parseFloat(document.getElementById('test-temperature')?.value) || 0.1,
                    top_p: parseFloat(document.getElementById('test-top-p')?.value) || 0.95,
                    max_new_tokens: parseInt(document.getElementById('test-max-tokens')?.value) || 256
                }
            };

            // Execute evaluation
            fetch('/api/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.accuracy !== undefined) {
                    this.displayEvaluationResults(data);
                    CoreModule.showAlert('Evaluation completed successfully', 'success');
                } else {
                    throw new Error(data.error || 'Evaluation failed');
                }
            })
            .catch(error => {
                console.error('Evaluation error:', error);
                if (resultsContainer) {
                    resultsContainer.innerHTML = `
                        <div class="card-body">
                            <div class="alert alert-danger">
                                <i class="fas fa-exclamation-circle me-2"></i>
                                ${CoreModule.escapeHtml(error.message)}
                            </div>
                        </div>
                    `;
                }
                CoreModule.showAlert(`Evaluation failed: ${error.message}`, 'danger');
            });
        },

        // Display evaluation results
        displayEvaluationResults(data) {
            const resultsContainer = document.getElementById('eval-results');
            if (!resultsContainer) return;

            resultsContainer.style.display = 'block';
            resultsContainer.innerHTML = `
                <div class="card-body">
                    <div class="alert alert-success">
                        <h5 class="alert-heading">
                            <i class="fas fa-chart-bar me-2"></i>Evaluation Results
                        </h5>
                        <hr>
                        <div class="row">
                            <div class="col-md-3">
                                <strong>Accuracy:</strong> ${(data.accuracy * 100).toFixed(2)}%
                            </div>
                            <div class="col-md-3">
                                <strong>Total:</strong> ${data.total || 0}
                            </div>
                            <div class="col-md-3">
                                <strong>Correct:</strong> ${data.correct || 0}
                            </div>
                            <div class="col-md-3">
                                <strong>Avg Time:</strong> ${data.average_time?.toFixed(2) || 'N/A'}s
                            </div>
                        </div>
                    </div>
                    <div class="mt-3">
                        <h6>Detailed Results:</h6>
                        <div class="table-responsive">
                            <table class="table table-sm table-striped">
                                <thead>
                                    <tr>
                                        <th>Input</th>
                                        <th>Expected</th>
                                        <th>Generated</th>
                                        <th>Match</th>
                                        <th>Time (s)</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${(data.results || []).map(result => `
                                        <tr>
                                            <td class="text-truncate" style="max-width: 200px;" title="${CoreModule.escapeHtml(result.input)}">
                                                ${CoreModule.escapeHtml(result.input)}
                                            </td>
                                            <td class="text-truncate" style="max-width: 150px;" title="${CoreModule.escapeHtml(result.expected)}">
                                                ${CoreModule.escapeHtml(result.expected)}
                                            </td>
                                            <td class="text-truncate" style="max-width: 150px;" title="${CoreModule.escapeHtml(result.output)}">
                                                ${CoreModule.escapeHtml(result.output)}
                                            </td>
                                            <td>
                                                <span class="badge bg-${result.correct ? 'success' : 'danger'}">
                                                    ${result.correct ? 'Match' : 'No Match'}
                                                </span>
                                            </td>
                                            <td>${result.generation_time?.toFixed(2) || 'N/A'}</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                        <button class="btn btn-primary mt-2" onclick="TestingModule.exportEvalResults()">
                            <i class="fas fa-download me-2"></i>Export Results
                        </button>
                    </div>
                </div>
            `;

            // Store current evaluation for export
            this.currentEvaluation = data;
        },

        // Export evaluation results
        exportEvalResults() {
            if (!this.currentEvaluation) {
                CoreModule.showAlert('No evaluation results to export', 'warning');
                return;
            }

            try {
                const data = this.currentEvaluation;

                // Create CSV content
                const headers = ['Input', 'Expected', 'Generated', 'Correct', 'Time (s)'];
                const csvRows = [headers.join(',')];

                (data.results || []).forEach(result => {
                    const row = [
                        `"${(result.input || '').replace(/"/g, '""')}"`,
                        `"${(result.expected || '').replace(/"/g, '""')}"`,
                        `"${(result.output || '').replace(/"/g, '""')}"`,
                        result.correct ? 'Yes' : 'No',
                        result.generation_time?.toFixed(2) || 'N/A'
                    ];
                    csvRows.push(row.join(','));
                });

                // Add summary row
                csvRows.push('');
                csvRows.push(`Summary:,Total: ${data.total},Correct: ${data.correct},Accuracy: ${(data.accuracy * 100).toFixed(2)}%,Avg Time: ${data.average_time?.toFixed(2)}s`);

                const csvContent = csvRows.join('\n');
                const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                const url = window.URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = `evaluation_results_${Date.now()}.csv`;
                link.click();
                window.URL.revokeObjectURL(url);

                CoreModule.showAlert('Evaluation results exported successfully', 'success');
            } catch (error) {
                console.error('Export failed:', error);
                CoreModule.showAlert(`Failed to export results: ${error.message}`, 'danger');
            }
        },

        // Update test prompt preview with actual training chat template
        updateTestPromptPreview() {
            const modelSelect = document.getElementById('test-model-select');
            const promptInput = document.getElementById('test-prompt');
            const formattedPromptPreview = document.getElementById('formatted-prompt-preview');
            const useChatTemplate = document.getElementById('use-chat-template')?.checked ?? true;

            if (!modelSelect?.value) {
                if (formattedPromptPreview) {
                    formattedPromptPreview.textContent = 'Please select a model first';
                }
                return;
            }

            if (!promptInput?.value) {
                if (formattedPromptPreview) {
                    formattedPromptPreview.textContent = 'Please enter a test prompt first';
                }
                return;
            }

            const sessionId = modelSelect.value;
            const prompt = promptInput.value;

            // Show loading state
            if (formattedPromptPreview) {
                formattedPromptPreview.textContent = 'Loading formatted prompt...';
            }

            // Fetch the prompt preview from backend
            fetch('/api/test/prompt-preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    prompt: prompt,
                    use_chat_template: useChatTemplate
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    throw new Error(data.error);
                }

                // Update formatted prompt preview with the complete formatted prompt
                if (formattedPromptPreview) {
                    if (data.formatted_prompt) {
                        formattedPromptPreview.textContent = data.formatted_prompt;
                    } else {
                        formattedPromptPreview.textContent = 'Error: No formatted prompt received from server';
                        console.error('Response data:', data);
                    }
                }
            })
            .catch(error => {
                console.error('Failed to load prompt preview:', error);
                if (formattedPromptPreview) {
                    formattedPromptPreview.textContent = `Error loading preview: ${error.message}`;
                }
            });
        }
    };

    // Export to window
    window.TestingModule = TestingModule;

    // Export functions for onclick handlers
    window.runSingleTest = () => TestingModule.runSingleTest();
    window.startBatchTest = () => TestingModule.startBatchTest();
    window.checkForActiveBatchTestsLegacy = () => TestingModule.checkForActiveBatchTests();

})(window);

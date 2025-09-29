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
            this.checkForActiveBatchTests();
            this.loadTestHistory();
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

            // Test prompt input
            const promptInput = document.getElementById('test-prompt');
            if (promptInput) {
                promptInput.addEventListener('input', () => this.updatePromptCounter());
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
            if (!modelSelect) return;

            // Show loading state
            modelSelect.innerHTML = '<option value="">Loading models...</option>';

            fetch('/api/test/models')
                .then(response => response.json())
                .then(data => {
                    if (data.models && data.models.length > 0) {
                        modelSelect.innerHTML = '<option value="">Select a trained model...</option>';

                        data.models.forEach(model => {
                            const option = document.createElement('option');
                            option.value = model.session_id;
                            option.dataset.baseModel = model.base_model;
                            option.dataset.checkpointPath = model.checkpoint_path;
                            option.textContent = `${model.display_name || model.model_name} (${model.epochs} epochs)`;
                            modelSelect.appendChild(option);
                        });
                    } else {
                        modelSelect.innerHTML = '<option value="">No trained models available</option>';
                    }
                })
                .catch(error => {
                    console.error('Failed to load testable models:', error);
                    modelSelect.innerHTML = '<option value="">Error loading models</option>';
                    CoreModule.showAlert('Failed to load trained models', 'danger');
                });
        },

        // Compare two models side-by-side
        compareModels() {
            const modelSelect = document.getElementById('test-model-select');
            const prompt = document.getElementById('test-prompt')?.value;

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

            if (!baseModel) {
                CoreModule.showAlert('Base model information not available', 'danger');
                return;
            }

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

            // Prepare comparison config
            const config = {
                session_id: sessionId,
                base_model: baseModel,
                prompt: prompt,
                temperature: parseFloat(document.getElementById('test-temperature')?.value) || 0.7,
                top_p: parseFloat(document.getElementById('test-top-p')?.value) || 0.9,
                max_tokens: parseInt(document.getElementById('test-max-tokens')?.value) || 256,
                repetition_penalty: parseFloat(document.getElementById('test-rep-penalty')?.value) || 1.1
            };

            // Execute comparison
            fetch('/api/test/compare-models', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success || data.results) {
                    this.displayComparisonResults(data);
                } else {
                    throw new Error(data.error || 'Comparison failed');
                }
            })
            .catch(error => {
                console.error('Comparison error:', error);
                if (resultsContainer) {
                    resultsContainer.innerHTML = `
                        <div class="alert alert-danger">
                            <i class="fas fa-exclamation-circle me-2"></i>
                            ${CoreModule.escapeHtml(error.message)}
                        </div>
                    `;
                }
                CoreModule.showAlert(`Comparison failed: ${error.message}`, 'danger');
            });
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
                                    <strong>Time:</strong> ${trainedResult.generation_time?.toFixed(2) || 'N/A'}s |
                                    <strong>Tokens:</strong> ${trainedResult.token_count || 'N/A'} |
                                    <strong>Speed:</strong> ${trainedResult.tokens_per_second?.toFixed(1) || 'N/A'} tok/s
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
                                    <strong>Time:</strong> ${baseResult.generation_time?.toFixed(2) || 'N/A'}s |
                                    <strong>Tokens:</strong> ${baseResult.token_count || 'N/A'} |
                                    <strong>Speed:</strong> ${baseResult.tokens_per_second?.toFixed(1) || 'N/A'} tok/s
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
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
            const instructionColumn = document.getElementById('instruction-column')?.value;
            const responseColumn = document.getElementById('response-column')?.value;
            const sampleSize = document.getElementById('batch-sample-size')?.value;

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

            if (!baseModel) {
                CoreModule.showAlert('Base model information not available', 'danger');
                return;
            }

            // Show loading state
            const progressContainer = document.getElementById('batch-test-progress');
            if (progressContainer) {
                progressContainer.style.display = 'block';
                progressContainer.innerHTML = `
                    <div class="alert alert-info">
                        <i class="fas fa-spinner fa-spin me-2"></i>
                        Starting batch comparison...
                    </div>
                `;
            }

            // Prepare batch comparison config
            const config = {
                session_id: sessionId,
                base_model: baseModel,
                instruction_column: instructionColumn,
                response_column: responseColumn || null,
                sample_size: sampleSize ? parseInt(sampleSize) : null,
                use_chat_template: document.getElementById('use-chat-template')?.checked !== false,
                compare_mode: 'base',
                config: {
                    temperature: parseFloat(document.getElementById('test-temperature')?.value) || 0.7,
                    top_p: parseFloat(document.getElementById('test-top-p')?.value) || 0.9,
                    max_tokens: parseInt(document.getElementById('test-max-tokens')?.value) || 256
                }
            };

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
                    const resultsContainer = document.getElementById('batch-test-results');
                    if (resultsContainer) {
                        const metrics = data.metrics || {};
                        resultsContainer.innerHTML = `
                            <div class="alert alert-success">
                                <h5 class="alert-heading">
                                    <i class="fas fa-check-circle me-2"></i>Batch Comparison Complete!
                                </h5>
                                <hr>
                                <div class="row">
                                    <div class="col-md-3">
                                        <strong>Total Tests:</strong> ${metrics.total || 0}
                                    </div>
                                    <div class="col-md-3">
                                        <strong>Successful:</strong> ${metrics.successful || 0}
                                    </div>
                                    <div class="col-md-3">
                                        <strong>Failed:</strong> ${metrics.failed || 0}
                                    </div>
                                    <div class="col-md-3">
                                        <strong>Avg Time:</strong> ${metrics.avg_time?.toFixed(2) || 'N/A'}s
                                    </div>
                                </div>
                                <div class="mt-3">
                                    <button class="btn btn-primary" onclick="TestingModule.exportBatchResults('${batchId}')">
                                        <i class="fas fa-download me-2"></i>Export Results
                                    </button>
                                    <button class="btn btn-secondary ms-2" onclick="TestingModule.viewBatchResults('${batchId}')">
                                        <i class="fas fa-eye me-2"></i>View Details
                                    </button>
                                </div>
                            </div>
                        `;
                    }

                    const progressContainer = document.getElementById('batch-test-progress');
                    if (progressContainer) {
                        progressContainer.style.display = 'none';
                    }

                    CoreModule.showAlert('Batch comparison completed successfully', 'success');
                })
                .catch(error => {
                    console.error('Failed to load batch results:', error);
                    CoreModule.showAlert('Batch comparison completed but failed to load results', 'warning');
                });
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
            if (!batchId) {
                CoreModule.showAlert('No batch results to export', 'warning');
                return;
            }

            // Show loading state
            CoreModule.showAlert('Preparing export...', 'info');

            fetch(`/api/test/batch-results/${batchId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.results) {
                        // Create CSV content
                        const results = data.results;
                        const headers = ['ID', 'Instruction', 'Trained Response', 'Base Response', 'Trained Time (s)', 'Base Time (s)', 'Status'];
                        const csvRows = [headers.join(',')];

                        results.forEach((result, idx) => {
                            const row = [
                                idx + 1,
                                `"${(result.instruction || '').replace(/"/g, '""')}"`,
                                `"${(result.trained_response || '').replace(/"/g, '""')}"`,
                                `"${(result.base_response || '').replace(/"/g, '""')}"`,
                                result.trained_time?.toFixed(2) || 'N/A',
                                result.base_time?.toFixed(2) || 'N/A',
                                result.success ? 'Success' : 'Failed'
                            ];
                            csvRows.push(row.join(','));
                        });

                        const csvContent = csvRows.join('\n');
                        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
                        const url = window.URL.createObjectURL(blob);
                        const link = document.createElement('a');
                        link.href = url;
                        link.download = `batch_comparison_${batchId}_${Date.now()}.csv`;
                        link.click();
                        window.URL.revokeObjectURL(url);

                        CoreModule.showAlert('Batch results exported successfully', 'success');
                    } else {
                        throw new Error('No results data available');
                    }
                })
                .catch(error => {
                    console.error('Export failed:', error);
                    CoreModule.showAlert(`Failed to export results: ${error.message}`, 'danger');
                });
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
        }
    };

    // Export to window
    window.TestingModule = TestingModule;

    // Export functions for onclick handlers
    window.runSingleTest = () => TestingModule.runSingleTest();
    window.startBatchTest = () => TestingModule.startBatchTest();
    window.checkForActiveBatchTestsLegacy = () => TestingModule.checkForActiveBatchTests();

})(window);
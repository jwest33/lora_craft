// ============================================================================
// Training Management Module
// ============================================================================

(function(window) {
    'use strict';

    const TrainingModule = {
        // Training state
        isTraining: false,
        trainingStartTime: null,
        trainingProgress: {},

        // Initialize the module
        init() {
            this.setupEventListeners();
            this.initializeCharts();
            this.setupTrainingMonitoring();
        },

        // Setup training-related event listeners
        setupEventListeners() {
            // Learning rate schedule change
            const lrSchedule = document.getElementById('lr-schedule');
            if (lrSchedule) {
                lrSchedule.addEventListener('change', () => this.onLRScheduleChange());
            }

            // Batch size change
            const batchSize = document.getElementById('batch-size');
            if (batchSize) {
                batchSize.addEventListener('change', () => {
                    NavigationModule.updateValidGenerations();
                    this.updateMemoryEstimate();
                });
            }

            // Gradient accumulation change
            const gradAccum = document.getElementById('gradient-accumulation');
            if (gradAccum) {
                gradAccum.addEventListener('change', () => this.updateEffectiveBatchSize());
            }

            // Number of epochs change
            const numEpochs = document.getElementById('num-epochs');
            if (numEpochs) {
                numEpochs.addEventListener('change', () => this.updateTrainingTimeEstimate());
            }
        },

        // Initialize training charts
        initializeCharts() {
            // Check if Chart.js is loaded
            if (typeof Chart === 'undefined') {
                console.error('Chart.js is not loaded! Training charts will not work.');
                return;
            }

            // Loss chart
            const lossCtx = document.getElementById('loss-chart');
            if (lossCtx) {
                AppState.charts.loss = new Chart(lossCtx.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Training Loss',
                            data: [],
                            borderColor: 'rgb(147, 51, 234)',
                            backgroundColor: 'rgba(147, 51, 234, 0.1)',
                            tension: 0.1
                        }, {
                            label: 'Validation Loss',
                            data: [],
                            borderColor: 'rgb(59, 130, 246)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'top' },
                            title: { display: true, text: 'Training Loss' }
                        },
                        scales: {
                            x: { title: { display: true, text: 'Step' } },
                            y: { title: { display: true, text: 'Loss' } }
                        }
                    }
                });
            }

            // Learning rate chart
            const lrCtx = document.getElementById('lr-chart');
            if (lrCtx) {
                AppState.charts.lr = new Chart(lrCtx.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Learning Rate',
                            data: [],
                            borderColor: 'rgb(34, 197, 94)',
                            backgroundColor: 'rgba(34, 197, 94, 0.1)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'top' },
                            title: { display: true, text: 'Learning Rate Schedule' }
                        },
                        scales: {
                            x: { title: { display: true, text: 'Step' } },
                            y: {
                                title: { display: true, text: 'Learning Rate' },
                                type: 'logarithmic'
                            }
                        }
                    }
                });
            }

            // Reward chart (for GRPO)
            const rewardCtx = document.getElementById('reward-chart');
            if (rewardCtx) {
                AppState.charts.reward = new Chart(rewardCtx.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Average Reward',
                            data: [],
                            borderColor: 'rgb(251, 146, 60)',
                            backgroundColor: 'rgba(251, 146, 60, 0.1)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { position: 'top' },
                            title: { display: true, text: 'Average Reward' }
                        },
                        scales: {
                            x: { title: { display: true, text: 'Step' } },
                            y: { title: { display: true, text: 'Reward' } }
                        }
                    }
                });
            }
        },

        // Setup training monitoring
        setupTrainingMonitoring() {
            // Listen for training updates via socket
            const socket = AppState.socket;
            if (socket) {
                // Check if socket is connected
                if (!socket.connected) {
                    console.warn('Socket not connected yet, training monitoring may have delayed updates');
                }

                // Store listener references for cleanup
                this.trainingProgressListener = (data) => this.handleTrainingProgress(data);
                this.trainingCompleteListener = (data) => this.handleTrainingComplete(data);
                this.trainingErrorListener = (data) => this.handleTrainingError(data);

                socket.on('training_progress', this.trainingProgressListener);
                socket.on('training_complete', this.trainingCompleteListener);
                socket.on('training_error', this.trainingErrorListener);
            } else {
                console.error('Socket not available, training updates will not work');
            }
        },

        // Cleanup socket listeners
        cleanupTrainingMonitoring() {
            const socket = AppState.socket;
            if (socket) {
                if (this.trainingProgressListener) {
                    socket.off('training_progress', this.trainingProgressListener);
                }
                if (this.trainingCompleteListener) {
                    socket.off('training_complete', this.trainingCompleteListener);
                }
                if (this.trainingErrorListener) {
                    socket.off('training_error', this.trainingErrorListener);
                }
            }
        },

        // Start training
        startTraining() {
            if (this.isTraining) {
                CoreModule.showAlert('Training is already in progress', 'warning');
                return;
            }

            // Validate all steps
            let canStart = true;
            for (let i = 1; i <= 3; i++) {
                if (!NavigationModule.validateStep(i)) {
                    canStart = false;
                    CoreModule.showAlert(`Please complete Step ${i} before starting training`, 'warning');
                    break;
                }
            }

            if (!canStart) return;

            // Gather configuration
            const config = this.gatherTrainingConfig();

            // Show confirmation modal
            CoreModule.showConfirmModal(
                'Start Training',
                `Are you ready to start training with the current configuration?\n\nEstimated time: ${this.estimateTrainingTime(config)}`,
                () => this.executeTraining(config),
                'btn-primary'
            );
        },

        // Gather training configuration
        gatherTrainingConfig() {
            return {
                // Model config
                model: (window.ModelsModule && ModelsModule.exportModelConfig)
                    ? ModelsModule.exportModelConfig()
                    : {},

                // Dataset config
                dataset: {
                    path: document.getElementById('dataset-path')?.value,
                    sample_size: parseInt(document.getElementById('sample-size')?.value) || 0,
                    train_split: parseInt(document.getElementById('train-split')?.value) || 80
                },

                // Template config
                template: (window.TemplatesModule && TemplatesModule.exportTemplateConfig)
                    ? TemplatesModule.exportTemplateConfig()
                    : {},

                // Training config
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

                // GRPO specific
                grpo: {
                    num_generations: parseInt(document.getElementById('num-generations')?.value) || 2,
                    kl_weight: parseFloat(document.getElementById('kl-weight')?.value) || 0.1,
                    clip_range: parseFloat(document.getElementById('clip-range')?.value) || 0.2
                },

                // Pre-training configuration
                pre_training: {
                    enabled: document.getElementById('enable-pre-training')?.checked ?? true,
                    epochs: parseInt(document.getElementById('pre-training-epochs')?.value) || 2,
                    max_samples: parseInt(document.getElementById('pre-training-samples')?.value) || null,
                    filter_by_length: document.getElementById('pre-training-filter-length')?.checked ?? false,
                    validate_format: document.getElementById('validate-format')?.checked ?? true
                },

                // Output
                output: {
                    name: document.getElementById('output-name')?.value || `lora_${Date.now()}`,
                    save_steps: parseInt(document.getElementById('save-steps')?.value) || 100,
                    eval_steps: parseInt(document.getElementById('eval-steps')?.value) || 100
                }
            };
        },

        // Execute training
        executeTraining(config) {
            this.isTraining = true;
            this.trainingStartTime = Date.now();

            // Update UI
            this.updateTrainingUI(true);

            // Clear previous charts
            this.clearCharts();

            // Start training via API
            fetch('/api/start_training', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    AppState.currentSessionId = data.session_id;
                    CoreModule.showAlert('Training started successfully!', 'success');
                } else {
                    throw new Error(data.error || 'Failed to start training');
                }
            })
            .catch(error => {
                console.error('Failed to start training:', error);
                CoreModule.showAlert(`Failed to start training: ${error.message}`, 'danger');
                this.isTraining = false;
                this.updateTrainingUI(false);
            });
        },

        // Stop training
        stopTraining() {
            if (!this.isTraining) return;

            CoreModule.showConfirmModal(
                'Stop Training',
                'Are you sure you want to stop the training? Progress will be saved.',
                () => {
                    fetch('/api/stop_training', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ session_id: AppState.currentSessionId })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            CoreModule.showAlert('Training stopped', 'info');
                            this.isTraining = false;
                            this.updateTrainingUI(false);
                        }
                    })
                    .catch(error => {
                        console.error('Failed to stop training:', error);
                        CoreModule.showAlert('Failed to stop training', 'danger');
                    });
                }
            );
        },

        // Handle training progress updates
        handleTrainingProgress(data) {
            this.trainingProgress = data;

            // Update progress bar
            const progressBar = document.getElementById('training-progress-bar');
            if (progressBar) {
                const percentage = (data.current_step / data.total_steps) * 100;
                progressBar.style.width = `${percentage}%`;
                progressBar.textContent = `${Math.round(percentage)}%`;
            }

            // Update stats
            this.updateTrainingStats(data);

            // Update charts
            if (data.metrics) {
                this.updateCharts(data.metrics);
            }

            // Update time estimate
            this.updateTimeRemaining(data);
        },

        // Handle training completion
        handleTrainingComplete(data) {
            this.isTraining = false;
            this.updateTrainingUI(false);

            CoreModule.showAlert('Training completed successfully!', 'success');

            // Add to trained models
            if (data.model_path) {
                AppState.trainedModels.push({
                    name: data.model_name,
                    path: data.model_path,
                    timestamp: Date.now(),
                    metrics: data.final_metrics
                });
            }

            // Navigate to export step
            NavigationModule.goToStep(5);
        },

        // Handle training error
        handleTrainingError(data) {
            this.isTraining = false;
            this.updateTrainingUI(false);

            CoreModule.showAlert(`Training error: ${data.error}`, 'danger');
            console.error('Training error:', data);
        },

        // Update training UI
        updateTrainingUI(isTraining) {
            // Toggle start/stop buttons
            const startBtn = document.getElementById('start-training-btn');
            const stopBtn = document.getElementById('stop-training-btn');

            if (startBtn) startBtn.style.display = isTraining ? 'none' : 'inline-block';
            if (stopBtn) stopBtn.style.display = isTraining ? 'inline-block' : 'none';

            // Show/hide progress section
            const progressSection = document.getElementById('training-progress-section');
            if (progressSection) {
                progressSection.style.display = isTraining ? 'block' : 'none';
            }

            // Disable form inputs during training
            const inputs = document.querySelectorAll('#step-3 input, #step-3 select');
            inputs.forEach(input => {
                input.disabled = isTraining;
            });
        },

        // Update training statistics display
        updateTrainingStats(data) {
            const statsContainer = document.getElementById('training-stats');
            if (!statsContainer) return;

            statsContainer.innerHTML = `
                <div class="row">
                    <div class="col-md-3">
                        <small class="text-muted">Epoch</small>
                        <div class="h5">${data.current_epoch || 0} / ${data.total_epochs || 0}</div>
                    </div>
                    <div class="col-md-3">
                        <small class="text-muted">Step</small>
                        <div class="h5">${data.current_step || 0} / ${data.total_steps || 0}</div>
                    </div>
                    <div class="col-md-3">
                        <small class="text-muted">Loss</small>
                        <div class="h5">${(data.loss || 0).toFixed(4)}</div>
                    </div>
                    <div class="col-md-3">
                        <small class="text-muted">Learning Rate</small>
                        <div class="h5">${(data.learning_rate || 0).toExponential(2)}</div>
                    </div>
                </div>
            `;
        },

        // Update charts with new data
        updateCharts(metrics) {
            // Update loss chart
            if (AppState.charts.loss && metrics.loss !== undefined) {
                AppState.charts.loss.data.labels.push(metrics.step);
                AppState.charts.loss.data.datasets[0].data.push(metrics.loss);

                if (metrics.val_loss !== undefined) {
                    AppState.charts.loss.data.datasets[1].data.push(metrics.val_loss);
                }

                AppState.charts.loss.update('none');
            }

            // Update learning rate chart
            if (AppState.charts.lr && metrics.learning_rate !== undefined) {
                AppState.charts.lr.data.labels.push(metrics.step);
                AppState.charts.lr.data.datasets[0].data.push(metrics.learning_rate);
                AppState.charts.lr.update('none');
            }

            // Update reward chart
            if (AppState.charts.reward && metrics.reward !== undefined) {
                AppState.charts.reward.data.labels.push(metrics.step);
                AppState.charts.reward.data.datasets[0].data.push(metrics.reward);
                AppState.charts.reward.update('none');
            }
        },

        // Clear all charts
        clearCharts() {
            Object.values(AppState.charts).forEach(chart => {
                if (chart) {
                    chart.data.labels = [];
                    chart.data.datasets.forEach(dataset => {
                        dataset.data = [];
                    });
                    chart.update();
                }
            });
        },

        // Update time remaining
        updateTimeRemaining(data) {
            if (!this.trainingStartTime || !data.current_step || !data.total_steps) return;

            const elapsed = Date.now() - this.trainingStartTime;
            const stepsComplete = data.current_step;
            const stepsRemaining = data.total_steps - stepsComplete;
            const timePerStep = elapsed / stepsComplete;
            const timeRemaining = timePerStep * stepsRemaining;

            const timeDisplay = document.getElementById('time-remaining');
            if (timeDisplay) {
                timeDisplay.textContent = this.formatTime(timeRemaining);
            }
        },

        // Format time duration
        formatTime(milliseconds) {
            const seconds = Math.floor(milliseconds / 1000);
            const minutes = Math.floor(seconds / 60);
            const hours = Math.floor(minutes / 60);

            if (hours > 0) {
                return `${hours}h ${minutes % 60}m`;
            } else if (minutes > 0) {
                return `${minutes}m ${seconds % 60}s`;
            } else {
                return `${seconds}s`;
            }
        },

        // Estimate training time
        estimateTrainingTime(config) {
            // Rough estimate based on configuration
            const samples = config.dataset.sample_size || 1000;
            const epochs = config.training.num_epochs;
            const batchSize = config.training.batch_size;
            const stepsPerEpoch = Math.ceil(samples / batchSize);
            const totalSteps = stepsPerEpoch * epochs;

            // Assume ~1 second per step (very rough estimate)
            const estimatedSeconds = totalSteps;
            return this.formatTime(estimatedSeconds * 1000);
        },

        // Handle LR schedule change
        onLRScheduleChange() {
            const schedule = document.getElementById('lr-schedule')?.value;
            const warmupRatio = document.getElementById('warmup-ratio');

            // Enable/disable warmup ratio based on schedule
            if (warmupRatio) {
                warmupRatio.disabled = schedule === 'constant';
            }
        },

        // Update memory estimate
        updateMemoryEstimate() {
            const batchSize = parseInt(document.getElementById('batch-size')?.value) || 4;
            const modelName = document.getElementById('model-name')?.value;

            // Rough memory estimates
            let baseMemory = 4; // GB
            if (modelName?.includes('7b')) baseMemory = 8;
            if (modelName?.includes('13b')) baseMemory = 16;
            if (modelName?.includes('30b')) baseMemory = 32;
            if (modelName?.includes('70b')) baseMemory = 64;

            const estimatedMemory = baseMemory + (batchSize * 0.5);

            const memoryDisplay = document.getElementById('memory-estimate');
            if (memoryDisplay) {
                memoryDisplay.textContent = `~${estimatedMemory.toFixed(1)} GB VRAM`;
            }
        },

        // Update effective batch size
        updateEffectiveBatchSize() {
            const batchSize = parseInt(document.getElementById('batch-size')?.value) || 4;
            const gradAccum = parseInt(document.getElementById('gradient-accumulation')?.value) || 1;
            const effectiveSize = batchSize * gradAccum;

            const display = document.getElementById('effective-batch-size');
            if (display) {
                display.textContent = `Effective batch size: ${effectiveSize}`;
            }
        },

        // Pause training
        pauseTraining() {
            if (!this.isTraining) {
                CoreModule.showAlert('No training in progress', 'warning');
                return;
            }

            fetch('/api/training/pause', {
                method: 'POST'
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    CoreModule.showAlert('Training paused', 'info');
                } else {
                    CoreModule.showAlert(data.error || 'Failed to pause training', 'error');
                }
            })
            .catch(error => {
                console.error('Error pausing training:', error);
                CoreModule.showAlert('Error pausing training', 'error');
            });
        },

        // Clear training logs
        clearLogs() {
            const logsContainer = document.getElementById('training-logs');
            if (logsContainer) {
                logsContainer.innerHTML = '';
                CoreModule.showAlert('Logs cleared', 'success');
            }
        },

        // Update prompt preview
        updatePromptPreview() {
            const systemPrompt = document.getElementById('system-prompt')?.value || '';
            const previewElement = document.getElementById('system-prompt-preview');

            if (previewElement) {
                previewElement.textContent = systemPrompt || 'No system prompt configured';
            }
        }
    };

    // Export to window
    window.TrainingModule = TrainingModule;

    // Export functions for onclick handlers
    window.startTraining = () => TrainingModule.startTraining();
    window.stopTraining = () => TrainingModule.stopTraining();

})(window);
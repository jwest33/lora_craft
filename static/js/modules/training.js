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

            // Common chart options
            const commonOptions = {
                responsive: true,
                maintainAspectRatio: false,  // Allow charts to fill container height
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 15
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        padding: 12,
                        titleFont: { size: 14, weight: 'bold' },
                        bodyFont: { size: 12 },
                        displayColors: true
                    }
                }
            };

            // Reward chart with std deviation bands
            const rewardCtx = document.getElementById('reward-chart');
            if (rewardCtx) {
                AppState.charts.reward = new Chart(rewardCtx.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Mean Reward',
                            data: [],
                            borderColor: 'rgb(251, 146, 60)',
                            backgroundColor: 'rgba(251, 146, 60, 0.1)',
                            borderWidth: 3,
                            tension: 0.3,
                            fill: false
                        }, {
                            label: 'Reward Std',
                            data: [],
                            borderColor: 'rgb(251, 191, 36)',
                            backgroundColor: 'rgba(251, 191, 36, 0.05)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                            tension: 0.3,
                            fill: false
                        }]
                    },
                    options: {
                        ...commonOptions,
                        plugins: {
                            ...commonOptions.plugins,
                            title: { display: true, text: 'Reward Metrics', font: { size: 16, weight: 'bold' } }
                        },
                        scales: {
                            x: {
                                title: { display: true, text: 'Training Step', font: { size: 12 } },
                                grid: { color: 'rgba(128, 128, 128, 0.1)' }
                            },
                            y: {
                                title: { display: true, text: 'Reward Value', font: { size: 12 } },
                                grid: { color: 'rgba(128, 128, 128, 0.1)' }
                            }
                        }
                    }
                });
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
                            borderWidth: 3,
                            tension: 0.3
                        }, {
                            label: 'Validation Loss',
                            data: [],
                            borderColor: 'rgb(59, 130, 246)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            tension: 0.3
                        }]
                    },
                    options: {
                        ...commonOptions,
                        plugins: {
                            ...commonOptions.plugins,
                            title: { display: true, text: 'Training Loss', font: { size: 16, weight: 'bold' } }
                        },
                        scales: {
                            x: {
                                title: { display: true, text: 'Training Step', font: { size: 12 } },
                                grid: { color: 'rgba(128, 128, 128, 0.1)' }
                            },
                            y: {
                                title: { display: true, text: 'Loss', font: { size: 12 } },
                                grid: { color: 'rgba(128, 128, 128, 0.1)' }
                            }
                        }
                    }
                });
            }

            // KL Divergence chart
            const klEntropyCtx = document.getElementById('kl-entropy-chart');
            if (klEntropyCtx) {
                AppState.charts.klEntropy = new Chart(klEntropyCtx.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'KL Divergence',
                            data: [],
                            borderColor: 'rgb(239, 68, 68)',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            borderWidth: 2,
                            tension: 0.3
                        }]
                    },
                    options: {
                        ...commonOptions,
                        plugins: {
                            ...commonOptions.plugins,
                            title: { display: true, text: 'KL Divergence', font: { size: 16, weight: 'bold' } }
                        },
                        scales: {
                            x: {
                                title: { display: true, text: 'Training Step', font: { size: 12 } },
                                grid: { color: 'rgba(128, 128, 128, 0.1)' }
                            },
                            y: {
                                type: 'linear',
                                display: true,
                                position: 'left',
                                title: { display: true, text: 'KL Divergence', font: { size: 12 } },
                                grid: { color: 'rgba(128, 128, 128, 0.1)' }
                            }
                        }
                    }
                });
            }

            // Completion Statistics chart
            const completionStatsCtx = document.getElementById('completion-stats-chart');
            if (completionStatsCtx) {
                AppState.charts.completionStats = new Chart(completionStatsCtx.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Mean Length',
                            data: [],
                            borderColor: 'rgb(59, 130, 246)',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            borderWidth: 3,
                            tension: 0.3
                        }, {
                            label: 'Min Length',
                            data: [],
                            borderColor: 'rgb(168, 85, 247)',
                            backgroundColor: 'rgba(168, 85, 247, 0.05)',
                            borderWidth: 1,
                            borderDash: [3, 3],
                            tension: 0.3
                        }, {
                            label: 'Max Length',
                            data: [],
                            borderColor: 'rgb(236, 72, 153)',
                            backgroundColor: 'rgba(236, 72, 153, 0.05)',
                            borderWidth: 1,
                            borderDash: [3, 3],
                            tension: 0.3
                        }]
                    },
                    options: {
                        ...commonOptions,
                        plugins: {
                            ...commonOptions.plugins,
                            title: { display: true, text: 'Completion Length Statistics', font: { size: 16, weight: 'bold' } }
                        },
                        scales: {
                            x: {
                                title: { display: true, text: 'Training Step', font: { size: 12 } },
                                grid: { color: 'rgba(128, 128, 128, 0.1)' }
                            },
                            y: {
                                title: { display: true, text: 'Token Count', font: { size: 12 } },
                                grid: { color: 'rgba(128, 128, 128, 0.1)' },
                                beginAtZero: true
                            }
                        }
                    }
                });
            }

            // Clip Ratio chart
            const clipRatioCtx = document.getElementById('clip-ratio-chart');
            if (clipRatioCtx) {
                AppState.charts.clipRatio = new Chart(clipRatioCtx.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Region Mean',
                            data: [],
                            borderColor: 'rgb(99, 102, 241)',
                            backgroundColor: 'rgba(99, 102, 241, 0.1)',
                            borderWidth: 3,
                            tension: 0.3
                        }, {
                            label: 'Low Mean',
                            data: [],
                            borderColor: 'rgb(239, 68, 68)',
                            backgroundColor: 'rgba(239, 68, 68, 0.1)',
                            borderWidth: 2,
                            tension: 0.3
                        }, {
                            label: 'High Mean',
                            data: [],
                            borderColor: 'rgb(34, 197, 94)',
                            backgroundColor: 'rgba(34, 197, 94, 0.1)',
                            borderWidth: 2,
                            tension: 0.3
                        }]
                    },
                    options: {
                        ...commonOptions,
                        plugins: {
                            ...commonOptions.plugins,
                            title: { display: true, text: 'Policy Clip Ratios', font: { size: 16, weight: 'bold' } }
                        },
                        scales: {
                            x: {
                                title: { display: true, text: 'Training Step', font: { size: 12 } },
                                grid: { color: 'rgba(128, 128, 128, 0.1)' }
                            },
                            y: {
                                title: { display: true, text: 'Clip Ratio', font: { size: 12 } },
                                grid: { color: 'rgba(128, 128, 128, 0.1)' },
                                beginAtZero: true,
                                max: 1.0
                            }
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
                            borderWidth: 3,
                            tension: 0.3,
                            fill: true
                        }]
                    },
                    options: {
                        ...commonOptions,
                        plugins: {
                            ...commonOptions.plugins,
                            title: { display: true, text: 'Learning Rate Schedule', font: { size: 16, weight: 'bold' } }
                        },
                        scales: {
                            x: {
                                title: { display: true, text: 'Training Step', font: { size: 12 } },
                                grid: { color: 'rgba(128, 128, 128, 0.1)' }
                            },
                            y: {
                                title: { display: true, text: 'Learning Rate', font: { size: 12 } },
                                type: 'logarithmic',
                                grid: { color: 'rgba(128, 128, 128, 0.1)' }
                            }
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

        // Connect to active training session after page reload
        connectToActiveSession(sessionId) {
            // Set training state
            this.isTraining = true;
            AppState.currentSessionId = sessionId;

            // Show training UI
            this.updateTrainingUI(true);

            // Fetch current session status and metrics
            fetch(`/api/training/${sessionId}/status`)
                .then(response => response.json())
                .then(data => {
                    if (data.success && data.status) {
                        // Restore progress
                        if (data.status.progress) {
                            this.trainingProgress = data.status.progress;
                            this.handleTrainingProgress(data.status.progress);
                        }

                        // Restore metrics history
                        if (data.status.metrics_history && data.status.metrics_history.length > 0) {
                            // Clear and repopulate charts
                            this.clearCharts();
                            data.status.metrics_history.forEach(metric => {
                                this.updateCharts(metric);
                            });
                        }

                        // Restore logs if available
                        if (data.status.logs) {
                            const logsContainer = document.getElementById('training-logs');
                            if (logsContainer) {
                                logsContainer.textContent = data.status.logs;
                                // Scroll to bottom
                                logsContainer.scrollTop = logsContainer.scrollHeight;
                            }
                        }

                        // Set training start time
                        if (data.status.start_time) {
                            this.trainingStartTime = new Date(data.status.start_time);
                        }

                        CoreModule.showAlert('Reconnected to active training session', 'success');
                    } else {
                        throw new Error(data.error || 'Failed to get session status');
                    }
                })
                .catch(error => {
                    console.error('Failed to connect to active session:', error);
                    CoreModule.showAlert(`Failed to reconnect: ${error.message}`, 'danger');
                    this.isTraining = false;
                    this.updateTrainingUI(false);
                });

            // Ensure socket listeners are active (they should be from init)
            if (!this.trainingProgressListener) {
                this.setupTrainingMonitoring();
            }
        },

        // Clear all chart data
        clearCharts() {
            const charts = ['reward', 'loss', 'kl', 'lr'];
            charts.forEach(chartName => {
                const chart = AppState.charts[chartName];
                if (chart) {
                    chart.data.labels = [];
                    chart.data.datasets.forEach(dataset => {
                        dataset.data = [];
                    });
                    chart.update('none'); // Update without animation
                }
            });
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
                    sample_size: parseInt(document.getElementById('max-samples')?.value) || null,
                    train_split: parseInt(document.getElementById('train-split')?.value) || 80,
                    instruction_field: document.getElementById('instruction-field')?.value || 'instruction',
                    response_field: document.getElementById('response-field')?.value || 'output'
                },
                // Map frontend datasetType to backend source_type
                dataset_source: (() => {
                    const datasetType = AppState.getConfigValue('datasetType') || 'upload';
                    if (datasetType === 'upload') return 'local';
                    if (datasetType === 'popular' || datasetType === 'custom') return 'huggingface';
                    return 'huggingface'; // default
                })(),
                dataset_path: document.getElementById('dataset-path')?.value,

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
                    max_sequence_length: parseInt(document.getElementById('max-sequence-length')?.value) || 2048,
                    max_new_tokens: parseInt(document.getElementById('max-new-tokens')?.value) || 512,
                    seed: parseInt(document.getElementById('seed')?.value) || 42
                },

                // GRPO specific
                grpo: {
                    num_generations: parseInt(document.getElementById('num-generations')?.value) || 2,
                    temperature: parseFloat(document.getElementById('temperature')?.value) || 0.7,
                    top_p: parseFloat(document.getElementById('top-p')?.value) || 0.95,
                    top_k: parseInt(document.getElementById('top-k')?.value) || 50,
                    repetition_penalty: parseFloat(document.getElementById('repetition-penalty')?.value) || 1.0,
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

        // Socket.js compatibility methods (aliases)
        updateProgress(data) {
            this.handleTrainingProgress(data);
        },

        updateMetrics(data) {
            // Data is already flat, not nested under 'metrics'
            console.log('updateMetrics called with keys:', Object.keys(data));
            console.log('updateMetrics data sample - kl:', data.kl, 'epoch:', data.epoch, 'step:', data.step);
            this.updateCharts(data);
            this.updateMetricsPanel(data);
        },

        handleComplete(data) {
            this.handleTrainingComplete(data);
        },

        handleError(data) {
            this.handleTrainingError(data);
        },

        // Update training UI
        updateTrainingUI(isTraining) {
            // Toggle start/stop buttons (match actual HTML IDs)
            const startBtn = document.getElementById('train-btn');
            const stopBtn = document.getElementById('stop-btn');

            if (startBtn) startBtn.style.display = isTraining ? 'none' : 'inline-block';
            if (stopBtn) stopBtn.style.display = isTraining ? 'inline-block' : 'none';

            // Show/hide training monitor (match actual HTML ID)
            const trainingMonitor = document.getElementById('training-monitor');
            if (trainingMonitor) {
                trainingMonitor.style.display = isTraining ? 'block' : 'none';
            }

            // Disable form inputs during training
            const inputs = document.querySelectorAll('#step-3 input, #step-3 select, #step-4 input, #step-4 select');
            inputs.forEach(input => {
                input.disabled = isTraining;
            });
        },

        // Update metrics panel display
        updateMetricsPanel(data) {
            // Helper function to safely update metric
            const updateMetric = (id, value, formatter = null) => {
                const element = document.getElementById(id);
                if (element && value !== undefined && value !== null) {
                    element.textContent = formatter ? formatter(value) : value;
                }
            };

            // Primary metrics
            updateMetric('metric-step', data.step || 0);
            // Display "Pre-training" for negative epochs, otherwise show 1-based epoch number
            const epochValue = data.epoch !== undefined && data.epoch < 0
                ? 'Pre-training'
                : Math.floor(data.epoch || 0) + 1;  // Add 1 to convert 0-based to 1-based display
            updateMetric('metric-epoch', epochValue);
            updateMetric('metric-loss', data.loss, v => v.toFixed(4));

            // Use mean_reward with fallback to reward or rewards/reward_wrapper/mean
            const rewardValue = data.mean_reward ?? data.reward ?? data['rewards/reward_wrapper/mean'];
            updateMetric('metric-reward', rewardValue, v => v.toFixed(4));

            // Use reward_std with fallback to rewards/reward_wrapper/std
            const rewardStdValue = data.reward_std ?? data['rewards/reward_wrapper/std'];
            updateMetric('metric-reward-std', rewardStdValue, v => v.toFixed(4));

            updateMetric('metric-lr', data.learning_rate, v => v.toExponential(2));

            // GRPO-specific metrics
            console.log('Updating GRPO metrics - kl:', data.kl);
            updateMetric('metric-kl', data.kl, v => v.toFixed(6));
            updateMetric('metric-grad-norm', data.grad_norm, v => v.toFixed(4));

            // Completion length (mean)
            const meanLength = data['completions/mean_length'];
            console.log('Updating completion metrics - meanLength:', meanLength);
            updateMetric('metric-comp-length', meanLength, v => v.toFixed(1));

            // Clipped ratio - try both possible field names
            const clippedRatio = data['completions/clipped_ratio'] ?? data['clip_ratio/clipped_ratio'];
            console.log('Updating clipped ratio:', clippedRatio);
            updateMetric('metric-clipped-ratio', clippedRatio, v => (v * 100).toFixed(1) + '%');

            // Clip region mean - try multiple possible field names
            const clipRegion = data['clip_ratio/region_mean'] ?? data['completions/clipped_ratio'];
            console.log('Updating clip region:', clipRegion);
            updateMetric('metric-clip-region', clipRegion, v => (v * 100).toFixed(1) + '%');
        },

        // Update charts with new data
        updateCharts(metrics) {
            const step = metrics.step;
            if (step === undefined || step === null) return;

            // Update reward chart (mean + std) with fallbacks
            const rewardValue = metrics.mean_reward ?? metrics.reward ?? metrics['rewards/reward_wrapper/mean'];
            if (AppState.charts.reward && rewardValue !== undefined) {
                AppState.charts.reward.data.labels.push(step);
                AppState.charts.reward.data.datasets[0].data.push(rewardValue);

                // Add reward std with fallback
                const rewardStdValue = metrics.reward_std ?? metrics['rewards/reward_wrapper/std'];
                if (rewardStdValue !== undefined && AppState.charts.reward.data.datasets[1]) {
                    AppState.charts.reward.data.datasets[1].data.push(rewardStdValue);
                }

                AppState.charts.reward.update('none');
            }

            // Update loss chart
            if (AppState.charts.loss && metrics.loss !== undefined) {
                AppState.charts.loss.data.labels.push(step);
                AppState.charts.loss.data.datasets[0].data.push(metrics.loss);

                if (metrics.val_loss !== undefined) {
                    AppState.charts.loss.data.datasets[1].data.push(metrics.val_loss);
                }

                AppState.charts.loss.update('none');
            }

            // Update KL Divergence chart
            if (AppState.charts.klEntropy && metrics.kl !== undefined) {
                AppState.charts.klEntropy.data.labels.push(step);
                AppState.charts.klEntropy.data.datasets[0].data.push(metrics.kl);
                AppState.charts.klEntropy.update('none');
                console.log(`KL chart updated: step=${step}, kl=${metrics.kl}`);
            }

            // Update completion statistics chart
            if (AppState.charts.completionStats) {
                const meanLen = metrics['completions/mean_length'];
                const minLen = metrics['completions/min_length'];
                const maxLen = metrics['completions/max_length'];

                if (meanLen !== undefined || minLen !== undefined || maxLen !== undefined) {
                    AppState.charts.completionStats.data.labels.push(step);

                    // Always push to all datasets to keep them synchronized (use null for missing)
                    AppState.charts.completionStats.data.datasets[0].data.push(meanLen ?? null);
                    AppState.charts.completionStats.data.datasets[1].data.push(minLen ?? null);
                    AppState.charts.completionStats.data.datasets[2].data.push(maxLen ?? null);

                    AppState.charts.completionStats.update('none');
                    console.log(`Completion stats chart updated: step=${step}, mean=${meanLen}, min=${minLen}, max=${maxLen}`);
                }
            }

            // Update clip ratio chart
            if (AppState.charts.clipRatio) {
                // TRL GRPO provides completions/clipped_ratio (overall clipping rate)
                // Use it for the main "Region Mean" dataset, others remain null
                const clippedRatio = metrics['completions/clipped_ratio'];
                const regionMean = metrics['clip_ratio/region_mean']; // May not exist in TRL
                const lowMean = metrics['clip_ratio/low_mean']; // May not exist in TRL
                const highMean = metrics['clip_ratio/high_mean']; // May not exist in TRL

                // Use clipped_ratio as fallback for region_mean if region_mean doesn't exist
                const effectiveRegionMean = regionMean ?? clippedRatio;

                if (effectiveRegionMean !== undefined || lowMean !== undefined || highMean !== undefined) {
                    AppState.charts.clipRatio.data.labels.push(step);

                    // Push data to all datasets to keep them synchronized
                    AppState.charts.clipRatio.data.datasets[0].data.push(effectiveRegionMean ?? null);
                    AppState.charts.clipRatio.data.datasets[1].data.push(lowMean ?? null);
                    AppState.charts.clipRatio.data.datasets[2].data.push(highMean ?? null);

                    AppState.charts.clipRatio.update('none');
                    console.log(`Clip ratio chart updated: step=${step}, region=${effectiveRegionMean}, low=${lowMean}, high=${highMean}`);
                }
            }

            // Update learning rate chart
            if (AppState.charts.lr && metrics.learning_rate !== undefined) {
                AppState.charts.lr.data.labels.push(step);
                AppState.charts.lr.data.datasets[0].data.push(metrics.learning_rate);
                AppState.charts.lr.update('none');
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

        // Estimate training time
        estimateTrainingTime(config) {
            // GRPO-specific calculation
            const samples = config.dataset.sample_size || 1000;
            const epochs = config.training.num_epochs || 1;
            const batchSize = config.training.batch_size || 4;
            const numGenerations = config.grpo.num_generations || 2;

            // GRPO step calculation: each step processes batch_size/num_generations prompts
            // because we generate num_generations completions per prompt
            const effectivePromptsPerStep = Math.max(1, Math.floor(batchSize / numGenerations));
            const stepsPerEpoch = Math.ceil(samples / effectivePromptsPerStep);
            const grpoSteps = stepsPerEpoch * epochs;

            // Add pre-training steps if enabled
            let preTrainingSteps = 0;
            if (config.pre_training && config.pre_training.enabled) {
                const preTrainingSamples = config.pre_training.max_samples || Math.min(samples, 100);
                const preTrainingEpochs = config.pre_training.epochs || 2;
                preTrainingSteps = Math.ceil(preTrainingSamples / batchSize) * preTrainingEpochs;
            }

            const totalSteps = preTrainingSteps + grpoSteps;

            // Realistic time per step for GRPO:
            // - Generation: 5-15s (depends on model size, sequence length, num_generations)
            // - Reward computation: 1-2s
            // - Optimization: 1-2s
            // Base estimate: 10-20 seconds per step

            // Adjust based on model size (rough heuristic)
            let timePerStepMin = 10;  // seconds
            let timePerStepMax = 20;  // seconds

            // For pre-training steps, use faster estimate (no reward computation, simpler)
            const preTrainingTime = preTrainingSteps * 3; // ~3s per step for SFT

            // GRPO training time
            const minGrpoTime = grpoSteps * timePerStepMin;
            const maxGrpoTime = grpoSteps * timePerStepMax;

            const minSeconds = preTrainingTime + minGrpoTime;
            const maxSeconds = preTrainingTime + maxGrpoTime;

            // Return as range for honesty about uncertainty
            if (minSeconds === maxSeconds) {
                return this.formatTime(minSeconds * 1000);
            } else {
                return `${this.formatTime(minSeconds * 1000)} - ${this.formatTime(maxSeconds * 1000)}`;
            }
        },

        // Format time in milliseconds to human-readable string
        formatTime(milliseconds) {
            const totalSeconds = Math.floor(milliseconds / 1000);

            if (totalSeconds < 60) {
                return `${totalSeconds}s`;
            }

            const hours = Math.floor(totalSeconds / 3600);
            const minutes = Math.floor((totalSeconds % 3600) / 60);
            const seconds = totalSeconds % 60;

            if (hours > 0) {
                if (minutes > 0) {
                    return `${hours}h ${minutes}m`;
                }
                return `${hours}h`;
            }

            if (seconds > 0) {
                return `${minutes}m ${seconds}s`;
            }
            return `${minutes}m`;
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

        // Append log message
        appendLog(message) {
            const logsContainer = document.getElementById('training-logs');
            if (!logsContainer) return;

            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.textContent = message;
            logsContainer.appendChild(logEntry);

            // Auto-scroll to bottom
            logsContainer.scrollTop = logsContainer.scrollHeight;
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

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
                maintainAspectRatio: true,
                aspectRatio: 2,
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

            // KL Divergence & Entropy chart
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
                            tension: 0.3,
                            yAxisID: 'y'
                        }, {
                            label: 'Entropy',
                            data: [],
                            borderColor: 'rgb(34, 197, 94)',
                            backgroundColor: 'rgba(34, 197, 94, 0.1)',
                            borderWidth: 2,
                            tension: 0.3,
                            yAxisID: 'y1'
                        }]
                    },
                    options: {
                        ...commonOptions,
                        plugins: {
                            ...commonOptions.plugins,
                            title: { display: true, text: 'KL Divergence & Entropy', font: { size: 16, weight: 'bold' } }
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
                                title: { display: true, text: 'KL Divergence', font: { size: 12 }, color: 'rgb(239, 68, 68)' },
                                grid: { color: 'rgba(128, 128, 128, 0.1)' }
                            },
                            y1: {
                                type: 'linear',
                                display: true,
                                position: 'right',
                                title: { display: true, text: 'Entropy', font: { size: 12 }, color: 'rgb(34, 197, 94)' },
                                grid: { drawOnChartArea: false }
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
                    train_split: parseInt(document.getElementById('train-split')?.value) || 80
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

        // Socket.js compatibility methods (aliases)
        updateProgress(data) {
            this.handleTrainingProgress(data);
        },

        updateMetrics(data) {
            // Data is already flat, not nested under 'metrics'
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
            // Display "Pre-training" for negative epochs, otherwise show epoch number
            const epochValue = data.epoch !== undefined && data.epoch < 0
                ? 'Pre-training'
                : Math.floor(data.epoch || 0);
            updateMetric('metric-epoch', epochValue);
            updateMetric('metric-loss', data.loss, v => v.toFixed(4));
            updateMetric('metric-reward', data.mean_reward, v => v.toFixed(4));
            updateMetric('metric-reward-std', data.reward_std, v => v.toFixed(4));
            updateMetric('metric-lr', data.learning_rate, v => v.toExponential(2));

            // GRPO-specific metrics
            updateMetric('metric-kl', data.kl, v => v.toFixed(6));
            updateMetric('metric-entropy', data.entropy, v => v.toFixed(4));
            updateMetric('metric-grad-norm', data.grad_norm, v => v.toFixed(4));

            // Completion length (mean)
            const meanLength = data['completions/mean_length'];
            updateMetric('metric-comp-length', meanLength, v => v.toFixed(1));

            // Clipped ratio (from completions)
            const clippedRatio = data['completions/clipped_ratio'];
            updateMetric('metric-clipped-ratio', clippedRatio, v => (v * 100).toFixed(1) + '%');

            // Clip region mean
            const clipRegion = data['clip_ratio/region_mean'];
            updateMetric('metric-clip-region', clipRegion, v => (v * 100).toFixed(1) + '%');
        },

        // Update charts with new data
        updateCharts(metrics) {
            const step = metrics.step;
            if (!step) return;

            // Update reward chart (mean + std)
            if (AppState.charts.reward && metrics.mean_reward !== undefined) {
                AppState.charts.reward.data.labels.push(step);
                AppState.charts.reward.data.datasets[0].data.push(metrics.mean_reward);

                if (metrics.reward_std !== undefined) {
                    AppState.charts.reward.data.datasets[1].data.push(metrics.reward_std);
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

            // Update KL & Entropy chart
            if (AppState.charts.klEntropy) {
                const hasKL = metrics.kl !== undefined;
                const hasEntropy = metrics.entropy !== undefined;

                if (hasKL || hasEntropy) {
                    AppState.charts.klEntropy.data.labels.push(step);

                    if (hasKL) {
                        AppState.charts.klEntropy.data.datasets[0].data.push(metrics.kl);
                    }
                    if (hasEntropy) {
                        AppState.charts.klEntropy.data.datasets[1].data.push(metrics.entropy);
                    }

                    AppState.charts.klEntropy.update('none');
                }
            }

            // Update completion statistics chart
            if (AppState.charts.completionStats) {
                const meanLen = metrics['completions/mean_length'];
                const minLen = metrics['completions/min_length'];
                const maxLen = metrics['completions/max_length'];

                if (meanLen !== undefined || minLen !== undefined || maxLen !== undefined) {
                    AppState.charts.completionStats.data.labels.push(step);

                    if (meanLen !== undefined) {
                        AppState.charts.completionStats.data.datasets[0].data.push(meanLen);
                    }
                    if (minLen !== undefined) {
                        AppState.charts.completionStats.data.datasets[1].data.push(minLen);
                    }
                    if (maxLen !== undefined) {
                        AppState.charts.completionStats.data.datasets[2].data.push(maxLen);
                    }

                    AppState.charts.completionStats.update('none');
                }
            }

            // Update clip ratio chart
            if (AppState.charts.clipRatio) {
                const regionMean = metrics['clip_ratio/region_mean'];
                const lowMean = metrics['clip_ratio/low_mean'];
                const highMean = metrics['clip_ratio/high_mean'];

                if (regionMean !== undefined || lowMean !== undefined || highMean !== undefined) {
                    AppState.charts.clipRatio.data.labels.push(step);

                    if (regionMean !== undefined) {
                        AppState.charts.clipRatio.data.datasets[0].data.push(regionMean);
                    }
                    if (lowMean !== undefined) {
                        AppState.charts.clipRatio.data.datasets[1].data.push(lowMean);
                    }
                    if (highMean !== undefined) {
                        AppState.charts.clipRatio.data.datasets[2].data.push(highMean);
                    }

                    AppState.charts.clipRatio.update('none');
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

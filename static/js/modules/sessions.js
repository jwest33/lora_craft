// ============================================================================
// Sessions Management Module
// ============================================================================

(function(window) {
    'use strict';

    const SessionsModule = {
        // Session state
        activeSessions: [],
        sessionHistory: [],

        // Initialize the module
        init() {
            this.setupEventListeners();
            this.checkForRunningSessions();  // Will call updateSidebarSessions() after fetch
            this.refreshSessions();           // Will call updateSidebarSessions() after fetch
            // Don't call updateSidebarSessions() here - it runs before async fetches complete
        },

        // Setup session-related event listeners
        setupEventListeners() {
            // Session list refresh button
            const refreshBtn = document.getElementById('refresh-sessions-btn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => this.refreshSessions());
            }

            // Session filter
            const filterInput = document.getElementById('session-filter');
            if (filterInput) {
                filterInput.addEventListener('input', CoreModule.debounce(() => {
                    this.filterSessions(filterInput.value);
                }, 300));
            }
        },

        // Check for running sessions
        checkForRunningSessions() {
            fetch('/api/active_sessions')
                .then(response => response.json())
                .then(data => {
                    this.activeSessions = data.sessions || [];
                    this.updateActiveSessionsDisplay();
                    this.updateSidebarSessions();

                    if (this.activeSessions.length > 0) {
                        CoreModule.showAlert(`${this.activeSessions.length} active session(s) found`, 'info');
                    }
                })
                .catch(error => {
                    console.error('Failed to check for running sessions:', error);
                });
        },

        // Refresh session list
        refreshSessions() {
            const sessionList = document.getElementById('session-list');
            if (!sessionList) return;

            // Show loading state
            sessionList.innerHTML = '<div class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading sessions...</div>';

            fetch('/api/sessions')
                .then(response => response.json())
                .then(data => {
                    this.sessionHistory = data.sessions || [];
                    this.displaySessions();
                    this.updateSidebarSessions();
                })
                .catch(error => {
                    console.error('Failed to load sessions:', error);
                    sessionList.innerHTML = '<div class="alert alert-danger">Failed to load sessions</div>';
                });

            return Promise.resolve();
        },

        // Display sessions
        displaySessions() {
            const sessionList = document.getElementById('session-list');
            if (!sessionList) return;

            if (this.sessionHistory.length === 0) {
                sessionList.innerHTML = '<div class="text-muted">No sessions found</div>';
                return;
            }

            sessionList.innerHTML = '';

            this.sessionHistory.forEach(session => {
                const sessionCard = this.createSessionCard(session);
                sessionList.appendChild(sessionCard);
            });
        },

        // Create session card element
        createSessionCard(session) {
            const card = document.createElement('div');
            card.className = 'card mb-3';
            card.dataset.sessionId = session.id;

            const isActive = this.activeSessions.some(s => s.id === session.id);
            const statusClass = isActive ? 'success' : (session.status === 'completed' ? 'primary' : 'secondary');
            const statusText = isActive ? 'Active' : (session.status || 'Unknown');

            card.innerHTML = `
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h5 class="card-title">${CoreModule.escapeHtml(session.name || session.id)}</h5>
                            <p class="card-text">
                                <small class="text-muted">
                                    Started: ${new Date(session.start_time).toLocaleString()}
                                    ${session.end_time ? `<br>Ended: ${new Date(session.end_time).toLocaleString()}` : ''}
                                </small>
                            </p>
                            <span class="badge bg-${statusClass}">${statusText}</span>
                            ${session.model ? `<span class="badge bg-info ms-2">${CoreModule.escapeHtml(session.model)}</span>` : ''}
                        </div>
                        <div class="btn-group">
                            ${isActive ? `
                                <button class="btn btn-sm btn-success" onclick="SessionsModule.resumeSession('${session.id}')" title="Monitor active training">
                                    <i class="fas fa-chart-line me-1"></i>Monitor
                                </button>
                            ` : `
                                <button class="btn btn-sm btn-outline-primary" onclick="SessionsModule.viewSession('${session.id}')" title="View session details">
                                    <i class="fas fa-eye"></i>
                                </button>
                            `}
                            <button class="btn btn-sm btn-outline-danger" onclick="SessionsModule.deleteSession('${session.id}')" title="Delete session">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                    ${session.metrics ? this.createMetricsSummary(session.metrics) : ''}
                </div>
            `;

            return card;
        },

        // Create metrics summary
        createMetricsSummary(metrics) {
            return `
                <div class="mt-3 pt-3 border-top">
                    <div class="row small">
                        ${metrics.final_loss ? `
                            <div class="col-md-3">
                                <strong>Final Loss:</strong> ${metrics.final_loss.toFixed(4)}
                            </div>
                        ` : ''}
                        ${metrics.best_loss ? `
                            <div class="col-md-3">
                                <strong>Best Loss:</strong> ${metrics.best_loss.toFixed(4)}
                            </div>
                        ` : ''}
                        ${metrics.total_steps ? `
                            <div class="col-md-3">
                                <strong>Total Steps:</strong> ${metrics.total_steps}
                            </div>
                        ` : ''}
                        ${metrics.training_time ? `
                            <div class="col-md-3">
                                <strong>Duration:</strong> ${this.formatDuration(metrics.training_time)}
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        },

        // View session details
        viewSession(sessionId) {
            fetch(`/api/session/${sessionId}`)
                .then(response => response.json())
                .then(data => {
                    this.showSessionDetails(data);
                })
                .catch(error => {
                    console.error('Failed to load session details:', error);
                    CoreModule.showAlert('Failed to load session details', 'danger');
                });
        },

        // Show session details modal
        showSessionDetails(session) {
            const modalId = 'sessionDetailsModal';
            let modal = document.getElementById(modalId);

            if (!modal) {
                const modalHtml = `
                    <div class="modal fade" id="${modalId}" tabindex="-1">
                        <div class="modal-dialog modal-xl">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">Session Details</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body" id="session-details-content"></div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                                    <button type="button" class="btn btn-primary" onclick="SessionsModule.exportSessionData()">
                                        <i class="fas fa-download me-2"></i>Export Data
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
                modal = document.getElementById(modalId);
            }

            // Populate session details
            const content = modal.querySelector('#session-details-content');
            content.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Configuration</h6>
                        <pre class="bg-light p-3 rounded">${JSON.stringify(session.config, null, 2)}</pre>
                    </div>
                    <div class="col-md-6">
                        <h6>Training Metrics</h6>
                        ${this.createMetricsCharts(session.metrics_history)}
                    </div>
                </div>
                ${session.logs ? `
                    <div class="mt-4">
                        <h6>Training Logs</h6>
                        <pre class="bg-dark text-light p-3 rounded" style="max-height: 300px; overflow-y: auto;">${CoreModule.escapeHtml(session.logs)}</pre>
                    </div>
                ` : ''}
            `;

            // Store current session for export
            this.currentViewedSession = session;

            // Show modal
            new bootstrap.Modal(modal).show();
        },

        // Create metrics charts for session details
        createMetricsCharts(metricsHistory) {
            if (!metricsHistory || metricsHistory.length === 0) {
                return '<p class="text-muted">No metrics history available</p>';
            }

            // Create canvas elements for mini charts
            return `
                <div class="metrics-charts">
                    <canvas id="session-loss-chart" height="150"></canvas>
                    <canvas id="session-lr-chart" height="150" class="mt-3"></canvas>
                </div>
                <script>
                    // These charts would be initialized after modal is shown
                    setTimeout(() => SessionsModule.initializeSessionCharts(${JSON.stringify(metricsHistory)}), 100);
                </script>
            `;
        },

        // Initialize session detail charts
        initializeSessionCharts(metricsHistory) {
            // Loss chart
            const lossCanvas = document.getElementById('session-loss-chart');
            if (lossCanvas) {
                new Chart(lossCanvas.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: metricsHistory.map(m => m.step),
                        datasets: [{
                            label: 'Loss',
                            data: metricsHistory.map(m => m.loss),
                            borderColor: 'rgb(147, 51, 234)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false }
                        }
                    }
                });
            }

            // Learning rate chart
            const lrCanvas = document.getElementById('session-lr-chart');
            if (lrCanvas) {
                new Chart(lrCanvas.getContext('2d'), {
                    type: 'line',
                    data: {
                        labels: metricsHistory.map(m => m.step),
                        datasets: [{
                            label: 'Learning Rate',
                            data: metricsHistory.map(m => m.learning_rate),
                            borderColor: 'rgb(34, 197, 94)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: false }
                        }
                    }
                });
            }
        },

        // Resume session
        resumeSession(sessionId) {
            // Navigate to training step
            NavigationModule.goToStep(4);

            // Connect to active session to restore progress
            if (window.TrainingModule && TrainingModule.connectToActiveSession) {
                TrainingModule.connectToActiveSession(sessionId);
            } else {
                // Fallback if TrainingModule not available
                AppState.currentSessionId = sessionId;
                CoreModule.showAlert('Session resumed (monitoring unavailable)', 'info');
            }
        },

        // Delete session
        deleteSession(sessionId) {
            CoreModule.showConfirmModal(
                'Delete Session',
                'Are you sure you want to delete this session? This action cannot be undone.',
                () => {
                    fetch(`/api/session/${sessionId}`, {
                        method: 'DELETE'
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            CoreModule.showAlert('Session deleted', 'success');
                            this.refreshSessions();
                        } else {
                            throw new Error(data.error || 'Failed to delete session');
                        }
                    })
                    .catch(error => {
                        console.error('Failed to delete session:', error);
                        CoreModule.showAlert(`Failed to delete session: ${error.message}`, 'danger');
                    });
                }
            );
        },

        // Export session data
        exportSessionData() {
            if (!this.currentViewedSession) return;

            const dataStr = JSON.stringify(this.currentViewedSession, null, 2);
            const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);

            const exportLink = document.createElement('a');
            exportLink.setAttribute('href', dataUri);
            exportLink.setAttribute('download', `session_${this.currentViewedSession.id}.json`);
            document.body.appendChild(exportLink);
            exportLink.click();
            document.body.removeChild(exportLink);

            CoreModule.showAlert('Session data exported', 'success');
        },

        // Filter sessions
        filterSessions(query) {
            const sessionCards = document.querySelectorAll('#session-list .card');

            sessionCards.forEach(card => {
                const text = card.textContent.toLowerCase();
                const matches = text.includes(query.toLowerCase());
                card.style.display = matches ? '' : 'none';
            });
        },

        // Update active sessions display
        updateActiveSessionsDisplay() {
            const indicator = document.getElementById('active-sessions-indicator');
            if (indicator) {
                if (this.activeSessions.length > 0) {
                    indicator.innerHTML = `
                        <span class="badge bg-success">
                            <i class="fas fa-circle-notch fa-spin me-1"></i>
                            ${this.activeSessions.length} Active
                        </span>
                    `;
                } else {
                    indicator.innerHTML = '';
                }
            }
        },

        // Format duration
        formatDuration(seconds) {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);

            if (hours > 0) {
                return `${hours}h ${minutes}m ${secs}s`;
            } else if (minutes > 0) {
                return `${minutes}m ${secs}s`;
            } else {
                return `${secs}s`;
            }
        },

        // Update sidebar sessions list
        updateSidebarSessions() {
            const sessionsList = document.getElementById('sessions-list');
            if (!sessionsList) {
                console.warn('Sessions list element not found');
                return;
            }

            console.log(`Updating sidebar sessions: ${this.activeSessions.length} active, ${this.sessionHistory.length} history`);

            // Combine active and recent sessions
            const allSessions = [];

            // Add active sessions first
            if (this.activeSessions && this.activeSessions.length > 0) {
                console.log('Adding active sessions:', this.activeSessions);
                this.activeSessions.forEach(session => {
                    allSessions.push({
                        ...session,
                        isActive: true
                    });
                });
            }

            // Add recent completed sessions (max 5 total)
            const remainingSlots = 5 - allSessions.length;
            if (remainingSlots > 0 && this.sessionHistory && this.sessionHistory.length > 0) {
                const recentSessions = this.sessionHistory
                    .filter(s => !allSessions.some(as => as.id === s.id))
                    .slice(0, remainingSlots);

                console.log(`Adding ${recentSessions.length} recent sessions from history`);
                recentSessions.forEach(session => {
                    allSessions.push({
                        ...session,
                        isActive: false
                    });
                });
            }

            // Render sessions list
            if (allSessions.length === 0) {
                console.log('No sessions to display');
                sessionsList.innerHTML = '<p class="text-muted small mb-0">No sessions found</p>';
                return;
            }

            console.log(`Rendering ${allSessions.length} sessions to sidebar`);

            sessionsList.innerHTML = '';

            allSessions.forEach(session => {
                const sessionItem = document.createElement('div');
                sessionItem.className = 'session-item';

                const statusBadge = session.isActive
                    ? '<span class="badge bg-success">Active</span>'
                    : '<span class="badge bg-secondary">Completed</span>';

                const actionButton = session.isActive
                    ? `<button class="btn btn-sm btn-success" onclick="SessionsModule.resumeSession('${session.id}')" title="Monitor">
                           <i class="fas fa-chart-line"></i>
                       </button>`
                    : `<button class="btn btn-sm btn-outline-primary" onclick="SessionsModule.viewSession('${session.id}')" title="View">
                           <i class="fas fa-eye"></i>
                       </button>`;

                sessionItem.innerHTML = `
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <div class="flex-grow-1">
                            <div class="fw-bold small">${CoreModule.escapeHtml(session.name || session.id)}</div>
                            <div class="text-muted" style="font-size: 0.75rem;">
                                ${new Date(session.start_time).toLocaleString('en-US', {
                                    month: 'short',
                                    day: 'numeric',
                                    hour: '2-digit',
                                    minute: '2-digit'
                                })}
                            </div>
                            ${statusBadge}
                        </div>
                        <div>
                            ${actionButton}
                        </div>
                    </div>
                `;

                sessionsList.appendChild(sessionItem);
            });
        }
    };

    // Export to window
    window.SessionsModule = SessionsModule;

    // Export functions for compatibility layer
    window.checkForRunningSessionsLegacy = () => SessionsModule.checkForRunningSessions();
    window.refreshSessionsLegacy = () => SessionsModule.refreshSessions();

})(window);

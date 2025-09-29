// ============================================================================
// Socket.IO Management Module
// ============================================================================

(function(window) {
    'use strict';

    const SocketModule = {
        // Initialize Socket.IO connection
        init() {
            const socket = io();
            AppState.setSocket(socket);

            this.setupEventHandlers(socket);
            return socket;
        },

        // Setup all socket event handlers
        setupEventHandlers(socket) {
            // Connection events
            socket.on('connect', () => {
                console.log('Connected to server');
                this.updateConnectionStatus('Online');
            });

            socket.on('disconnect', () => {
                console.log('Disconnected from server');
                this.updateConnectionStatus('Offline');
            });

            // Training events
            socket.on('training_progress', (data) => {
                if (data.session_id === AppState.currentSessionId) {
                    console.log('Training progress:', data);
                    if (window.TrainingModule) {
                        window.TrainingModule.updateProgress(data.progress);
                    }
                }
            });

            socket.on('training_metrics', (data) => {
                if (data.session_id === AppState.currentSessionId) {
                    console.log('Training metrics received:', data);
                    if (window.TrainingModule) {
                        window.TrainingModule.updateMetrics(data);
                    }
                }
            });

            socket.on('training_log', (data) => {
                if (data.session_id === AppState.currentSessionId) {
                    console.log('Training log:', data.message);
                    if (window.TrainingModule) {
                        window.TrainingModule.appendLog(data.message);
                    }
                }
            });

            socket.on('training_complete', (data) => {
                if (data.session_id === AppState.currentSessionId) {
                    console.log('Training complete:', data);
                    if (window.TrainingModule) {
                        window.TrainingModule.handleComplete(data);
                    }
                }
            });

            socket.on('training_error', (data) => {
                if (data.session_id === AppState.currentSessionId) {
                    console.error('Training error:', data);
                    if (window.TrainingModule) {
                        window.TrainingModule.handleError(data);
                    }
                }
            });

            // Dataset events
            socket.on('dataset_progress', (data) => {
                if (window.DatasetModule) {
                    window.DatasetModule.updateProgress(data);
                }
            });

            socket.on('dataset_complete', (data) => {
                if (window.DatasetModule) {
                    window.DatasetModule.handleComplete(data);
                }
            });

            socket.on('dataset_error', (data) => {
                if (window.DatasetModule) {
                    window.DatasetModule.handleError(data);
                }
            });

            // Export events
            socket.on('export_progress', (data) => {
                const statusText = document.getElementById('export-status');
                if (statusText) {
                    statusText.textContent = data.message || 'Exporting...';
                }
            });

            // Batch test events
            socket.on('batch_test_progress', (data) => {
                if (window.TestingModule) {
                    window.TestingModule.updateBatchProgress(data);
                }
            });

            socket.on('batch_test_complete', (data) => {
                if (window.TestingModule) {
                    window.TestingModule.handleBatchComplete(data);
                }
            });

            socket.on('batch_test_error', (data) => {
                if (window.TestingModule) {
                    window.TestingModule.handleBatchError(data);
                }
            });
        },

        // Update connection status in UI
        updateConnectionStatus(status) {
            const indicator = document.getElementById('connection-indicator');
            if (indicator) {
                if (status === 'Online') {
                    indicator.classList.remove('offline');
                    indicator.classList.add('online');
                } else {
                    indicator.classList.remove('online');
                    indicator.classList.add('offline');
                }
            }

            // Update any other UI elements that show connection status
            const statusElements = document.querySelectorAll('.connection-status');
            statusElements.forEach(el => {
                el.textContent = status;
                el.className = 'connection-status ' + status.toLowerCase();
            });
        },

        // Emit socket events with error handling
        emit(event, data, callback) {
            const socket = AppState.socket;
            if (socket && socket.connected) {
                if (callback) {
                    socket.emit(event, data, callback);
                } else {
                    socket.emit(event, data);
                }
                return true;
            } else {
                console.warn('Socket not connected. Cannot emit:', event);
                return false;
            }
        },

        // Check if socket is connected
        isConnected() {
            return AppState.socket && AppState.socket.connected;
        },

        // Reconnect to socket
        reconnect() {
            if (AppState.socket) {
                AppState.socket.connect();
            }
        },

        // Disconnect from socket
        disconnect() {
            if (AppState.socket) {
                AppState.socket.disconnect();
            }
        }
    };

    // Export to window
    window.SocketModule = SocketModule;

})(window);
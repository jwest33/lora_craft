// ============================================================================
// Core Application Module - Utilities and Base Functions
// ============================================================================

(function(window) {
    'use strict';

    const CoreModule = {
        // Interval ID for system status updates
        systemStatusInterval: null,

        // Initialize the core module
        init() {
            this.setupEventListeners();
            this.initializeTooltips();
            this.setupIconScrollEffect();
            this.loadSavedState();
            this.updateSystemStatus();
            this.startSystemStatusUpdates();
        },

        // Setup global event listeners
        setupEventListeners() {
            // Theme toggle - find by icon
            const themeToggle = document.querySelector('#theme-icon')?.closest('button');
            if (themeToggle) {
                themeToggle.onclick = () => this.toggleTheme();
            }

            // Sidebar toggle - find by bars icon
            const sidebarToggle = document.querySelector('.fa-bars')?.closest('button');
            if (sidebarToggle) {
                sidebarToggle.onclick = () => this.toggleSidebar();
            }

            // Help button - find by question icon
            const helpButton = document.querySelector('.fa-question-circle')?.closest('button');
            if (helpButton) {
                helpButton.onclick = () => this.showHelp();
            }

            // Save state on important changes
            window.addEventListener('beforeunload', () => {
                this.saveState();
            });
        },

        // Initialize Bootstrap tooltips
        initializeTooltips() {
            const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
            tooltipTriggerList.map(function(tooltipTriggerEl) {
                return new bootstrap.Tooltip(tooltipTriggerEl);
            });
        },

        // Setup icon scroll effect
        setupIconScrollEffect() {
            let lastScrollTop = 0;
            const iconDock = document.querySelector('.floating-icon-dock');

            if (iconDock) {
                window.addEventListener('scroll', () => {
                    const scrollTop = window.pageYOffset || document.documentElement.scrollTop;

                    if (scrollTop > lastScrollTop && scrollTop > 100) {
                        iconDock.style.transform = 'translateX(-50%) translateY(-100px)';
                    } else {
                        iconDock.style.transform = 'translateX(-50%) translateY(0)';
                    }

                    lastScrollTop = scrollTop;
                });
            }
        },

        // Toggle theme between light and dark
        toggleTheme() {
            const html = document.documentElement;
            const currentTheme = html.getAttribute('data-theme');
            const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);

            // Update theme icon
            const themeIcon = document.getElementById('theme-icon');
            if (themeIcon) {
                themeIcon.className = newTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
            }
        },

        // Toggle sidebar visibility
        toggleSidebar() {
            const sidebar = document.getElementById('sidebar-panel');
            const mainContent = document.querySelector('.content-area');

            if (sidebar) {
                sidebar.classList.toggle('hidden');

                // Adjust main content width
                if (mainContent) {
                    if (sidebar.classList.contains('hidden')) {
                        mainContent.style.marginLeft = '0';
                        mainContent.style.maxWidth = '100%';
                    } else {
                        mainContent.style.marginLeft = '';
                        mainContent.style.maxWidth = '';
                    }
                }

                // Save sidebar state
                localStorage.setItem('sidebarHidden', sidebar.classList.contains('hidden'));
            }
        },

        // Show help modal
        showHelp() {
            this.showAlert('Help documentation coming soon!', 'info');
        },

        // Show app info modal
        showAppInfo() {
            const modal = new bootstrap.Modal(document.getElementById('appInfoModal') || this.createInfoModal());
            modal.show();
        },

        // Create info modal if it doesn't exist
        createInfoModal() {
            const modalHtml = `
                <div class="modal fade" id="appInfoModal" tabindex="-1">
                    <div class="modal-dialog">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title">LoRA Craft</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body">
                                <p><strong>Version:</strong> 1.0.0</p>
                                <p><strong>Description:</strong> Professional LoRA training interface</p>
                                <p>Click the icon anytime to return to the dashboard.</p>
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-primary" data-bs-dismiss="modal">Close</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            document.body.insertAdjacentHTML('beforeend', modalHtml);
            return document.getElementById('appInfoModal');
        },

        // Show alert/toast notification
        showAlert(message, type = 'info') {
            // Create toast container if it doesn't exist
            let toastContainer = document.getElementById('toast-container');
            if (!toastContainer) {
                toastContainer = this.createToastContainer();
            }

            const toastId = 'toast-' + Date.now();
            const toastHtml = `
                <div id="${toastId}" class="toast align-items-center text-white bg-${type} border-0" role="alert">
                    <div class="d-flex">
                        <div class="toast-body">
                            ${this.escapeHtml(message)}
                        </div>
                        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                    </div>
                </div>
            `;

            toastContainer.insertAdjacentHTML('beforeend', toastHtml);

            const toastElement = document.getElementById(toastId);
            const toast = new bootstrap.Toast(toastElement, {
                autohide: true,
                delay: 5000
            });

            toast.show();

            // Remove toast element after it's hidden
            toastElement.addEventListener('hidden.bs.toast', () => {
                toastElement.remove();
            });
        },

        // Create toast container
        createToastContainer() {
            const container = document.createElement('div');
            container.id = 'toast-container';
            container.className = 'toast-container position-fixed top-0 end-0 p-3';
            container.style.zIndex = '1080';
            container.style.marginTop = '100px';
            document.body.appendChild(container);
            return container;
        },

        // Show confirmation modal
        showConfirmModal(title, message, onConfirm, confirmBtnClass = 'btn-danger') {
            const modalId = 'confirmModal-' + Date.now();
            const isDarkMode = document.documentElement.getAttribute('data-theme') === 'dark';

            // Synthwave dark mode styles
            const darkStyles = {
                modalContent: 'border: 2px solid rgba(168, 85, 247, 0.5); box-shadow: 0 10px 40px rgba(0,0,0,0.6), 0 0 30px rgba(168, 85, 247, 0.4); background: #1e293b;',
                modalHeader: 'background: linear-gradient(135deg, #7e22ce, #9333ea); color: #f1f5f9; border-bottom: 1px solid rgba(168, 85, 247, 0.3);',
                modalBody: 'padding: 2rem; background: #1e293b; border: 1px solid rgba(168, 85, 247, 0.2); border-left: none; border-right: none; font-size: 1.1rem;',
                messageColor: '#cbd5e1',
                modalFooter: 'padding: 1.5rem; background: #1e293b; border-top: 1px solid rgba(168, 85, 247, 0.3); gap: 1rem;'
            };

            // Light mode styles (original)
            const lightStyles = {
                modalContent: 'border: 2px solid #dc3545; box-shadow: 0 10px 40px rgba(0,0,0,0.3);',
                modalHeader: 'background: linear-gradient(135deg, #dc3545 0%, #c82333 100%); color: white; border-bottom: none;',
                modalBody: 'padding: 2rem; background-color: #fff3cd; font-size: 1.1rem;',
                messageColor: '#856404',
                modalFooter: 'padding: 1.5rem; border-top: none; gap: 1rem;'
            };

            const styles = isDarkMode ? darkStyles : lightStyles;

            const modalHtml = `
                <div class="modal fade" id="${modalId}" tabindex="-1">
                    <div class="modal-dialog modal-dialog-centered">
                        <div class="modal-content" style="${styles.modalContent}">
                            <div class="modal-header" style="${styles.modalHeader}">
                                <h5 class="modal-title" style="font-size: 1.5rem; font-weight: 600;">
                                    <i class="fas fa-exclamation-triangle me-2"></i>${this.escapeHtml(title)}
                                </h5>
                                <button type="button" class="btn-close ${isDarkMode ? 'btn-close-white' : ''}" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body" style="${styles.modalBody}">
                                <p class="mb-0" style="color: ${styles.messageColor};">${this.escapeHtml(message)}</p>
                            </div>
                            <div class="modal-footer" style="${styles.modalFooter}">
                                <button type="button" class="btn btn-secondary btn-lg" data-bs-dismiss="modal" style="min-width: 100px;">Cancel</button>
                                <button type="button" class="btn ${confirmBtnClass} btn-lg" id="${modalId}-confirm" style="min-width: 100px;">Confirm</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            document.body.insertAdjacentHTML('beforeend', modalHtml);

            const modalElement = document.getElementById(modalId);
            const modal = new bootstrap.Modal(modalElement);

            document.getElementById(`${modalId}-confirm`).onclick = () => {
                onConfirm();
                modal.hide();
            };

            modalElement.addEventListener('hidden.bs.modal', () => {
                modalElement.remove();
            });

            modal.show();
        },

        // Show input modal
        showInputModal(title, message, placeholder, onSubmit, submitBtnClass = 'btn-primary') {
            const modalId = 'inputModal-' + Date.now();
            const inputId = `${modalId}-input`;
            const modalHtml = `
                <div class="modal fade" id="${modalId}" tabindex="-1">
                    <div class="modal-dialog">
                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title">${this.escapeHtml(title)}</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                            </div>
                            <div class="modal-body">
                                <p>${this.escapeHtml(message)}</p>
                                <input type="text" class="form-control" id="${inputId}" placeholder="${this.escapeHtml(placeholder || '')}" />
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                                <button type="button" class="btn ${submitBtnClass}" id="${modalId}-submit">Submit</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            document.body.insertAdjacentHTML('beforeend', modalHtml);

            const modalElement = document.getElementById(modalId);
            const modal = new bootstrap.Modal(modalElement);
            const inputElement = document.getElementById(inputId);
            const submitBtn = document.getElementById(`${modalId}-submit`);

            const handleSubmit = () => {
                const value = inputElement.value.trim();
                if (value) {
                    onSubmit(value);
                    modal.hide();
                }
            };

            submitBtn.onclick = handleSubmit;

            // Allow Enter key to submit
            inputElement.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    handleSubmit();
                }
            });

            modalElement.addEventListener('hidden.bs.modal', () => {
                modalElement.remove();
            });

            modalElement.addEventListener('shown.bs.modal', () => {
                inputElement.focus();
            });

            modal.show();
        },

        // Start periodic system status updates
        startSystemStatusUpdates() {
            // Update system status every 3 seconds
            this.systemStatusInterval = setInterval(() => {
                this.updateSystemStatus();
            }, 3000);
        },

        // Stop periodic system status updates
        stopSystemStatusUpdates() {
            if (this.systemStatusInterval) {
                clearInterval(this.systemStatusInterval);
                this.systemStatusInterval = null;
            }
        },

        // Update system status indicators
        updateSystemStatus() {
            fetch('/api/system_status')
                .then(response => response.json())
                .then(data => {
                    // Update GPU status
                    const gpuStatus = document.getElementById('gpu-status');
                    if (gpuStatus) {
                        gpuStatus.textContent = data.gpu || 'No GPU';
                    }

                    // Update VRAM status with dynamic coloring
                    const vramStatus = document.getElementById('vram-status');
                    if (vramStatus) {
                        vramStatus.textContent = data.vram || 'N/A';
                        this.applyUsageClass(vramStatus, data.vram_percent || 0);
                    }

                    // Update RAM status with dynamic coloring
                    const ramStatus = document.getElementById('ram-status');
                    if (ramStatus) {
                        ramStatus.textContent = data.ram || 'N/A';
                        this.applyUsageClass(ramStatus, data.ram_percent || 0);
                    }
                })
                .catch(error => {
                    console.error('Failed to fetch system status:', error);
                });
        },

        // Apply usage-based CSS class to element
        applyUsageClass(element, percent) {
            // Remove all existing usage classes
            element.classList.remove('usage-low', 'usage-medium', 'usage-high', 'usage-critical');

            // Apply appropriate class based on usage percentage
            if (percent >= 90) {
                element.classList.add('usage-critical');
            } else if (percent >= 75) {
                element.classList.add('usage-high');
            } else if (percent >= 50) {
                element.classList.add('usage-medium');
            } else {
                element.classList.add('usage-low');
            }
        },

        // Save application state to localStorage
        saveState() {
            const state = {
                theme: document.documentElement.getAttribute('data-theme'),
                sidebarHidden: document.getElementById('sidebar-panel')?.classList.contains('hidden'),
                currentStep: AppState.currentStep,
                stepValidation: AppState.stepValidation
            };

            localStorage.setItem('appState', JSON.stringify(state));
            AppState.saveToLocalStorage();
        },

        // Load saved state from localStorage
        loadSavedState() {
            // Load theme
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme) {
                document.documentElement.setAttribute('data-theme', savedTheme);
                const themeIcon = document.getElementById('theme-icon');
                if (themeIcon) {
                    themeIcon.className = savedTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
                }
            }

            // Load sidebar state
            const sidebarHidden = localStorage.getItem('sidebarHidden');
            if (sidebarHidden === 'true') {
                const sidebar = document.getElementById('sidebar-panel');
                if (sidebar) {
                    sidebar.classList.add('hidden');
                }
            }

            // Load app state
            AppState.loadFromLocalStorage();
        },

        // Utility function to escape HTML
        escapeHtml(text) {
            const map = {
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#039;'
            };
            return text.replace(/[&<>"']/g, m => map[m]);
        },

        // Debounce utility function
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

        // Update form input value
        updateValue(inputId, value) {
            const element = document.getElementById(inputId);
            if (element) {
                element.value = value;
            }
        }
    };

    // Export to window
    window.CoreModule = CoreModule;

})(window);

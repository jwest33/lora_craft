// ============================================================================
// Application Initialization
// ============================================================================

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    try {
        // Initialize core modules that need to run immediately
        SocketModule.init();
        CoreModule.init();
        NavigationModule.init();

        // Wait a moment for Bootstrap to fully initialize other modules
        setTimeout(() => {

        // Initialize other modules if they exist
        if (window.ModelsModule) {
            ModelsModule.init();
        }

        if (window.DatasetModule) {
            DatasetModule.init();
        }

        if (window.TemplatesModule) {
            TemplatesModule.init();
        }

        if (window.TrainingModule) {
            TrainingModule.init();
        }

        if (window.SessionsModule) {
            SessionsModule.init();
        }

        if (window.ConfigModule) {
            ConfigModule.init();
        }

        if (window.ExportModule) {
            ExportModule.init();
        }

        if (window.TestingModule) {
            TestingModule.init();
        }

        if (window.RewardAnalysisModule) {
            RewardAnalysisModule.init();
        }

        // Check for any running sessions
        checkForRunningSessions();

        // Refresh session list
        if (window.refreshSessions) {
            refreshSessions();
        }

        // Load configuration list
        if (window.loadConfigList) {
            loadConfigList();
        }

        // Set initial template preview
        if (window.updateTemplatePreview) {
            updateTemplatePreview();
        }

        // Initialize chart template after a small delay
        setTimeout(() => {
            if (window.initializeChatTemplate) {
                initializeChatTemplate();
            }
        }, 100);

        // Check for active batch tests
        if (window.checkForActiveBatchTests) {
            checkForActiveBatchTests();
        }

        // Application initialization complete
        }, 100); // End of setTimeout

    } catch (error) {
        console.error('Error during application initialization:', error);

        // Show error to user
        if (window.CoreModule && CoreModule.showAlert) {
            CoreModule.showAlert('Error initializing application. Please refresh the page.', 'danger');
        } else {
            alert('Error initializing application. Please refresh the page.');
        }
    }
});

// Temporary compatibility layer for functions still in app.js
// These will be removed once all modules are created

function checkForRunningSessions() {
    if (window.SessionsModule && SessionsModule.checkForRunningSessions) {
        SessionsModule.checkForRunningSessions();
    } else if (window.checkForRunningSessionsLegacy) {
        // Fall back to legacy function in app.js
        window.checkForRunningSessionsLegacy();
    }
}

function refreshSessions() {
    if (window.SessionsModule && SessionsModule.refreshSessions) {
        return SessionsModule.refreshSessions();
    } else if (window.refreshSessionsLegacy) {
        // Fall back to legacy function in app.js
        return window.refreshSessionsLegacy();
    }
    return Promise.resolve();
}

function loadConfigList() {
    if (window.ConfigModule && ConfigModule.loadConfigList) {
        ConfigModule.loadConfigList();
    } else if (window.loadConfigListLegacy) {
        // Fall back to legacy function in app.js
        window.loadConfigListLegacy();
    }
}

function updateTemplatePreview() {
    if (window.TemplatesModule && TemplatesModule.updateTemplatePreview) {
        TemplatesModule.updateTemplatePreview();
    } else if (window.updateTemplatePreviewLegacy) {
        // Fall back to legacy function in app.js
        window.updateTemplatePreviewLegacy();
    }
}

function initializeChatTemplate() {
    if (window.TemplatesModule && TemplatesModule.initializeChatTemplate) {
        TemplatesModule.initializeChatTemplate();
    } else if (window.initializeChatTemplateLegacy) {
        // Fall back to legacy function in app.js
        window.initializeChatTemplateLegacy();
    }
}

function checkForActiveBatchTests() {
    if (window.TestingModule && TestingModule.checkForActiveBatchTests) {
        TestingModule.checkForActiveBatchTests();
    } else if (window.checkForActiveBatchTestsLegacy) {
        // Fall back to legacy function in app.js
        window.checkForActiveBatchTestsLegacy();
    }
}

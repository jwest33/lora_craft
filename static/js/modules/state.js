// ============================================================================
// Global State Management Module
// ============================================================================

const AppState = {
    // Core application state
    socket: null,
    currentSessionId: null,
    currentDatasetSession: null,
    currentStep: 1,
    activeBatchTestId: null,

    // Models and datasets
    availableModels: {},
    datasetStatusCache: {},
    trainedModels: [],
    selectedModelsForExport: new Set(),

    // Step validation
    stepValidation: {
        1: false,
        2: false,
        3: false,
        4: false
    },

    // Chart instances
    charts: {
        loss: null,
        lr: null,
        reward: null
    },

    // Bootstrap collapse instances
    collapseInstances: {},

    // Test history
    testHistory: [],

    // Configuration cache
    configCache: {},
    config: {},

    // Methods to update state
    setSocket(socket) {
        this.socket = socket;
    },

    setCurrentSession(sessionId) {
        this.currentSessionId = sessionId;
    },

    setCurrentDatasetSession(session) {
        this.currentDatasetSession = session;
    },

    setCurrentStep(step) {
        this.currentStep = step;
    },

    validateStep(stepNum, isValid) {
        this.stepValidation[stepNum] = isValid;
    },

    isStepValid(stepNum) {
        return this.stepValidation[stepNum];
    },

    addTrainedModel(model) {
        this.trainedModels.push(model);
    },

    toggleModelSelection(sessionId) {
        if (this.selectedModelsForExport.has(sessionId)) {
            this.selectedModelsForExport.delete(sessionId);
        } else {
            this.selectedModelsForExport.add(sessionId);
        }
    },

    clearModelSelection() {
        this.selectedModelsForExport.clear();
    },

    setChart(type, chart) {
        this.charts[type] = chart;
    },

    getChart(type) {
        return this.charts[type];
    },

    resetCharts() {
        Object.keys(this.charts).forEach(key => {
            if (this.charts[key]) {
                this.charts[key].destroy();
                this.charts[key] = null;
            }
        });
    },

    // Configuration methods
    getConfigValue(key) {
        return this.config[key];
    },

    setConfigValue(key, value) {
        this.config[key] = value;
        this.saveToLocalStorage();
    },

    // Save state to localStorage
    saveToLocalStorage() {
        const stateToSave = {
            currentStep: this.currentStep,
            stepValidation: this.stepValidation,
            configCache: this.configCache,
            config: this.config
        };
        localStorage.setItem('appState', JSON.stringify(stateToSave));
    },

    // Load state from localStorage
    loadFromLocalStorage() {
        const savedState = localStorage.getItem('appState');
        if (savedState) {
            const state = JSON.parse(savedState);
            this.currentStep = state.currentStep || 1;
            this.stepValidation = state.stepValidation || {1: false, 2: false, 3: false, 4: false};
            this.configCache = state.configCache || {};
            this.config = state.config || {};
        }
    }
};

// Make AppState globally available
window.AppState = AppState;
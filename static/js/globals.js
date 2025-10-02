// ============================================================================
// Global Function Exports
// This file exports all functions that are called from HTML onclick handlers
// ============================================================================

(function(window) {
    'use strict';

    // ========================================
    // Core Module Functions
    // ========================================

    window.showAppInfo = function() {
        if (window.CoreModule && CoreModule.showAppInfo) {
            CoreModule.showAppInfo();
        } else {
            console.error('CoreModule.showAppInfo not available');
        }
    };

    window.updateValue = function(inputId, value) {
        if (window.CoreModule && CoreModule.updateValue) {
            CoreModule.updateValue(inputId, value);
        } else {
            const input = document.getElementById(inputId);
            if (input) input.value = value;
        }
    };

    // ========================================
    // Navigation Functions
    // ========================================

    window.goToStep = function(step) {
        if (window.NavigationModule && NavigationModule.goToStep) {
            NavigationModule.goToStep(step);
        } else {
            console.error('NavigationModule.goToStep not available');
        }
    };

    // ========================================
    // Models Module Functions
    // ========================================

    window.applyLoRAPreset = function(preset) {
        if (window.ModelsModule && ModelsModule.applyLoRAPreset) {
            ModelsModule.applyLoRAPreset(preset);
        } else {
            console.error('ModelsModule.applyLoRAPreset not available');
        }
    };

    window.loadFromPath = function() {
        if (window.ModelsModule && ModelsModule.loadFromPath) {
            ModelsModule.loadFromPath();
        } else {
            console.error('ModelsModule.loadFromPath not available');
        }
    };

    window.clearModelCache = function() {
        if (window.ModelsModule && ModelsModule.clearModelCache) {
            ModelsModule.clearModelCache();
        } else {
            console.error('ModelsModule.clearModelCache not available');
        }
    };

    // ========================================
    // Dataset Module Functions
    // ========================================

    window.selectDatasetType = function(type) {
        if (window.DatasetsModule && DatasetsModule.selectDatasetType) {
            DatasetsModule.selectDatasetType(type);
        } else {
            console.error('DatasetsModule.selectDatasetType not available');
        }
    };

    window.filterDatasets = function(filter) {
        if (window.DatasetsModule && DatasetsModule.filterDatasets) {
            DatasetsModule.filterDatasets(filter);
        } else {
            console.error('DatasetsModule.filterDatasets not available');
        }
    };

    window.downloadCustomDataset = function() {
        if (window.DatasetsModule && DatasetsModule.downloadCustomDataset) {
            DatasetsModule.downloadCustomDataset();
        } else {
            console.error('DatasetsModule.downloadCustomDataset not available');
        }
    };

    window.cancelDownload = function() {
        if (window.DatasetsModule && DatasetsModule.cancelDownload) {
            DatasetsModule.cancelDownload();
        } else {
            console.error('DatasetsModule.cancelDownload not available');
        }
    };

    window.useDatasetFromPreview = function() {
        if (window.DatasetsModule && DatasetsModule.useDatasetFromPreview) {
            DatasetsModule.useDatasetFromPreview();
        } else {
            console.error('DatasetsModule.useDatasetFromPreview not available');
        }
    };

    // ========================================
    // Templates Module Functions
    // ========================================

    window.editTemplate = function() {
        if (window.TemplatesModule && TemplatesModule.editTemplate) {
            TemplatesModule.editTemplate();
        } else {
            console.error('TemplatesModule.editTemplate not available');
        }
    };

    window.validateChatTemplate = function() {
        if (window.TemplatesModule && TemplatesModule.validateChatTemplate) {
            TemplatesModule.validateChatTemplate();
        } else {
            console.error('TemplatesModule.validateChatTemplate not available');
        }
    };

    window.saveChatTemplate = function() {
        if (window.TemplatesModule && TemplatesModule.saveChatTemplate) {
            TemplatesModule.saveChatTemplate();
        } else {
            console.error('TemplatesModule.saveChatTemplate not available');
        }
    };

    window.loadChatTemplate = function() {
        if (window.TemplatesModule && TemplatesModule.loadChatTemplate) {
            TemplatesModule.loadChatTemplate();
        } else {
            console.error('TemplatesModule.loadChatTemplate not available');
        }
    };

    window.saveTemplateFromModal = function() {
        if (window.TemplatesModule && TemplatesModule.saveTemplateFromModal) {
            TemplatesModule.saveTemplateFromModal();
        } else {
            console.error('TemplatesModule.saveTemplateFromModal not available');
        }
    };

    // ========================================
    // Config Module Functions
    // ========================================

    window.applyPreset = function(preset) {
        if (window.ConfigModule && ConfigModule.applyPreset) {
            ConfigModule.applyPreset(preset);
        } else {
            console.error('ConfigModule.applyPreset not available');
        }
    };

    window.selectAlgorithm = function(algorithm) {
        if (window.ConfigModule && ConfigModule.selectAlgorithm) {
            ConfigModule.selectAlgorithm(algorithm);
        } else {
            console.error('ConfigModule.selectAlgorithm not available');
        }
    };

    window.onConfigSelect = function() {
        if (window.ConfigModule && ConfigModule.onConfigSelect) {
            ConfigModule.onConfigSelect();
        } else {
            console.error('ConfigModule.onConfigSelect not available');
        }
    };

    window.saveConfig = function() {
        if (window.ConfigModule && ConfigModule.saveConfig) {
            ConfigModule.saveConfig();
        } else {
            console.error('ConfigModule.saveConfig not available');
        }
    };

    window.loadSelectedConfig = function() {
        if (window.ConfigModule && ConfigModule.loadSelectedConfig) {
            ConfigModule.loadSelectedConfig();
        } else {
            console.error('ConfigModule.loadSelectedConfig not available');
        }
    };

    window.deleteSelectedConfig = function() {
        if (window.ConfigModule && ConfigModule.deleteSelectedConfig) {
            ConfigModule.deleteSelectedConfig();
        } else {
            console.error('ConfigModule.deleteSelectedConfig not available');
        }
    };

    // ========================================
    // Reward Functions
    // ========================================

    window.showRewardHelp = function() {
        if (window.ConfigModule && ConfigModule.showRewardHelp) {
            ConfigModule.showRewardHelp();
        } else {
            console.error('ConfigModule.showRewardHelp not available');
        }
    };

    window.selectTemplate = function(template) {
        if (window.ConfigModule && ConfigModule.selectTemplate) {
            ConfigModule.selectTemplate(template);
        } else {
            console.error('ConfigModule.selectTemplate not available');
        }
    };

    window.addAdvancedComponent = function() {
        if (window.ConfigModule && ConfigModule.addAdvancedComponent) {
            ConfigModule.addAdvancedComponent();
        } else {
            console.error('ConfigModule.addAdvancedComponent not available');
        }
    };

    window.saveCustomReward = function() {
        if (window.ConfigModule && ConfigModule.saveCustomReward) {
            ConfigModule.saveCustomReward();
        } else {
            console.error('ConfigModule.saveCustomReward not available');
        }
    };

    window.testReward = function() {
        if (window.ConfigModule && ConfigModule.testReward) {
            ConfigModule.testReward();
        } else {
            console.error('ConfigModule.testReward not available');
        }
    };

    window.viewRewardDetails = function() {
        if (window.ConfigModule && ConfigModule.viewRewardDetails) {
            ConfigModule.viewRewardDetails();
        } else {
            console.error('ConfigModule.viewRewardDetails not available');
        }
    };

    window.testSelectedReward = function() {
        if (window.ConfigModule && ConfigModule.testSelectedReward) {
            ConfigModule.testSelectedReward();
        } else {
            console.error('ConfigModule.testSelectedReward not available');
        }
    };

    // ========================================
    // Training Module Functions
    // ========================================

    window.clearLogs = function() {
        if (window.TrainingModule && TrainingModule.clearLogs) {
            TrainingModule.clearLogs();
        } else {
            console.error('TrainingModule.clearLogs not available');
        }
    };

    window.pauseTraining = function() {
        if (window.TrainingModule && TrainingModule.pauseTraining) {
            TrainingModule.pauseTraining();
        } else {
            console.error('TrainingModule.pauseTraining not available');
        }
    };

    window.updatePromptPreview = function() {
        // Check which step is active
        const currentStep = window.currentStep || 1;

        // If in testing step (step 6), use TestingModule
        if (currentStep === 6) {
            if (window.TestingModule && TestingModule.updateTestPromptPreview) {
                TestingModule.updateTestPromptPreview();
            } else {
                console.error('TestingModule.updateTestPromptPreview not available');
            }
        } else {
            // Otherwise use TrainingModule (for training steps)
            if (window.TrainingModule && TrainingModule.updatePromptPreview) {
                TrainingModule.updatePromptPreview();
            } else {
                console.error('TrainingModule.updatePromptPreview not available');
            }
        }
    };

    // ========================================
    // Export Module Functions
    // ========================================

    window.refreshTrainedModels = function() {
        if (window.ExportModule && ExportModule.refreshTrainedModels) {
            ExportModule.refreshTrainedModels();
        } else {
            console.error('ExportModule.refreshTrainedModels not available');
        }
    };

    window.showBatchExport = function() {
        if (window.ExportModule && ExportModule.showBatchExport) {
            ExportModule.showBatchExport();
        } else {
            console.error('ExportModule.showBatchExport not available');
        }
    };

    window.hideModelDetails = function() {
        if (window.ExportModule && ExportModule.hideModelDetails) {
            ExportModule.hideModelDetails();
        } else {
            console.error('ExportModule.hideModelDetails not available');
        }
    };

    // ========================================
    // Testing Module Functions
    // ========================================

    window.loadTestableModels = function() {
        if (window.TestingModule && TestingModule.loadTestableModels) {
            TestingModule.loadTestableModels();
        } else {
            console.error('TestingModule.loadTestableModels not available');
        }
    };

    window.compareModels = function() {
        if (window.TestingModule && TestingModule.compareModels) {
            TestingModule.compareModels();
        } else {
            console.error('TestingModule.compareModels not available');
        }
    };

    window.toggleComparisonMode = function() {
        if (window.TestingModule && TestingModule.toggleComparisonMode) {
            TestingModule.toggleComparisonMode();
        } else {
            console.error('TestingModule.toggleComparisonMode not available');
        }
    };

    window.updateComparisonModelInfo = function() {
        if (window.TestingModule && TestingModule.updateComparisonModelInfo) {
            TestingModule.updateComparisonModelInfo();
        } else {
            console.error('TestingModule.updateComparisonModelInfo not available');
        }
    };

    window.clearBatchTestFile = function() {
        if (window.TestingModule && TestingModule.clearBatchTestFile) {
            TestingModule.clearBatchTestFile();
        } else {
            console.error('TestingModule.clearBatchTestFile not available');
        }
    };

    window.runBatchComparison = function() {
        if (window.TestingModule && TestingModule.runBatchComparison) {
            TestingModule.runBatchComparison();
        } else {
            console.error('TestingModule.runBatchComparison not available');
        }
    };

    window.cancelBatchTest = function() {
        if (window.TestingModule && TestingModule.cancelBatchTest) {
            TestingModule.cancelBatchTest();
        } else {
            console.error('TestingModule.cancelBatchTest not available');
        }
    };

    window.exportBatchResults = function(batchId) {
        if (window.TestingModule && TestingModule.exportBatchResults) {
            TestingModule.exportBatchResults(batchId);
        } else {
            console.error('TestingModule.exportBatchResults not available');
        }
    };

    window.runEvaluation = function() {
        if (window.TestingModule && TestingModule.runEvaluation) {
            TestingModule.runEvaluation();
        } else {
            console.error('TestingModule.runEvaluation not available');
        }
    };

    window.exportEvalResults = function() {
        if (window.TestingModule && TestingModule.exportEvalResults) {
            TestingModule.exportEvalResults();
        } else {
            console.error('TestingModule.exportEvalResults not available');
        }
    };

    window.handleBatchTestFileUpload = function() {
        if (window.TestingModule && TestingModule.handleBatchTestFileUpload) {
            TestingModule.handleBatchTestFileUpload();
        } else {
            console.error('TestingModule.handleBatchTestFileUpload not available');
        }
    };

    window.updateBatchModelSelection = function() {
        if (window.TestingModule && TestingModule.updateBatchModelSelection) {
            TestingModule.updateBatchModelSelection();
        } else {
            console.error('TestingModule.updateBatchModelSelection not available');
        }
    };

})(window);

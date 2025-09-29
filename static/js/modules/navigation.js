// ============================================================================
// Step Navigation & Validation Module
// ============================================================================

(function(window) {
    'use strict';

    const NavigationModule = {
        // Initialize navigation
        init() {
            // Check if Bootstrap is loaded
            if (typeof bootstrap === 'undefined') {
                console.error('Bootstrap is not loaded! Navigation will not work properly.');
                return;
            }

            this.initializeCollapseInstances();
            this.updateStepIndicators();
            this.updateValidGenerations();
        },

        // Initialize Bootstrap collapse instances
        initializeCollapseInstances() {
            // Initialize all step collapse instances
            for (let i = 1; i <= 6; i++) {
                const collapseElement = document.getElementById(`step-${i}-content`);
                if (collapseElement) {
                    try {
                        // Check if Bootstrap Collapse is available
                        if (bootstrap && bootstrap.Collapse) {
                            AppState.collapseInstances[i] = new bootstrap.Collapse(collapseElement, {
                                toggle: false
                            });
                        }
                    } catch (error) {
                        console.error(`Error creating collapse instance for step ${i}:`, error);
                    }
                }
            }

            // Ensure Step 1 is shown initially
            if (AppState.collapseInstances[1]) {
                try {
                    AppState.collapseInstances[1].show();
                } catch (error) {
                    // Fallback: manually add show class
                    const step1Content = document.getElementById('step-1-content');
                    if (step1Content) {
                        step1Content.classList.add('show');
                    }
                }
            }
        },

        // Toggle step collapse
        toggleStep(stepNum) {
            const collapseElement = document.getElementById(`step-${stepNum}-content`);
            const chevron = document.getElementById(`step-${stepNum}-chevron`);

            if (!collapseElement) {
                return;
            }

            // Check if Bootstrap is available
            if (typeof bootstrap === 'undefined' || !bootstrap.Collapse) {
                // Direct toggle without Bootstrap
                const isShown = collapseElement.classList.contains('show');

                if (isShown) {
                    collapseElement.classList.remove('show');
                    if (chevron) {
                        chevron.classList.remove('fa-chevron-up');
                        chevron.classList.add('fa-chevron-down');
                    }
                } else {
                    collapseElement.classList.add('show');
                    if (chevron) {
                        chevron.classList.remove('fa-chevron-down');
                        chevron.classList.add('fa-chevron-up');
                    }
                }
                return;
            }

            // Try to get or create the collapse instance
            let collapseInstance = AppState.collapseInstances[stepNum];

            if (!collapseInstance) {
                try {
                    collapseInstance = new bootstrap.Collapse(collapseElement, {
                        toggle: false
                    });
                    AppState.collapseInstances[stepNum] = collapseInstance;
                } catch (error) {
                    // Fall back to direct manipulation
                    collapseElement.classList.toggle('show');
                    if (chevron) {
                        if (collapseElement.classList.contains('show')) {
                            chevron.classList.remove('fa-chevron-down');
                            chevron.classList.add('fa-chevron-up');
                        } else {
                            chevron.classList.remove('fa-chevron-up');
                            chevron.classList.add('fa-chevron-down');
                        }
                    }
                    return;
                }
            }

            // Use Bootstrap collapse instance
            const isShown = collapseElement.classList.contains('show');

            try {
                if (isShown) {
                    collapseInstance.hide();
                    if (chevron) {
                        chevron.classList.remove('fa-chevron-up');
                        chevron.classList.add('fa-chevron-down');
                    }
                } else {
                    collapseInstance.show();
                    if (chevron) {
                        chevron.classList.remove('fa-chevron-down');
                        chevron.classList.add('fa-chevron-up');
                    }
                }
            } catch (error) {
                // Fallback
                collapseElement.classList.toggle('show');
                if (chevron) {
                    if (collapseElement.classList.contains('show')) {
                        chevron.classList.remove('fa-chevron-down');
                        chevron.classList.add('fa-chevron-up');
                    } else {
                        chevron.classList.remove('fa-chevron-up');
                        chevron.classList.add('fa-chevron-down');
                    }
                }
            }
        },

        // Navigate to specific step
        goToStep(stepNum) {
            // Update current step
            AppState.setCurrentStep(stepNum);

            // Collapse all steps except the target step
            for (let i = 1; i <= 6; i++) {
                const indicator = document.getElementById(`step-${i}-indicator`);
                const collapseInstance = AppState.collapseInstances[i];
                const chevron = document.getElementById(`step-${i}-chevron`);

                if (i === stepNum) {
                    // Expand target step
                    if (collapseInstance) {
                        collapseInstance.show();
                    }
                    if (chevron) {
                        chevron.classList.remove('fa-chevron-down');
                        chevron.classList.add('fa-chevron-up');
                    }
                } else {
                    // Collapse other steps
                    if (collapseInstance) {
                        collapseInstance.hide();
                    }
                    if (chevron) {
                        chevron.classList.remove('fa-chevron-up');
                        chevron.classList.add('fa-chevron-down');
                    }
                }

                // Update indicators
                if (indicator) {
                    indicator.classList.toggle('active', i === stepNum);
                }
            }

            // Scroll to the target step
            const targetStep = document.getElementById(`step-${stepNum}`);
            if (targetStep) {
                targetStep.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }

            // Save state
            CoreModule.saveState();
        },

        // Validate and proceed to next step
        validateAndProceed(stepNum) {
            if (this.validateStep(stepNum)) {
                const nextStep = stepNum + 1;
                if (nextStep <= 6) {
                    this.goToStep(nextStep);
                    CoreModule.showAlert(`Step ${stepNum} validated successfully!`, 'success');
                }
            } else {
                CoreModule.showAlert('Please complete all required fields before proceeding.', 'warning');
            }
        },

        // Validate specific step
        validateStep(stepNum) {
            let isValid = false;

            switch (stepNum) {
                case 1:
                    // Validate model configuration
                    const modelName = document.getElementById('model-name');
                    const loraRank = document.getElementById('lora-rank');
                    isValid = modelName && modelName.value &&
                             loraRank && loraRank.value;
                    break;

                case 2:
                    // Validate dataset selection
                    const datasetPath = document.getElementById('dataset-path');
                    isValid = datasetPath && datasetPath.value;
                    break;

                case 3:
                    // Validate training configuration
                    const numEpochs = document.getElementById('num-epochs');
                    const batchSize = document.getElementById('batch-size');
                    isValid = numEpochs && numEpochs.value &&
                             batchSize && batchSize.value;
                    break;

                case 4:
                    // Review step is always valid if previous steps are valid
                    isValid = AppState.isStepValid(1) &&
                             AppState.isStepValid(2) &&
                             AppState.isStepValid(3);
                    break;

                case 5:
                    // Export step validation
                    isValid = AppState.trainedModels.length > 0;
                    break;

                case 6:
                    // Test step validation
                    isValid = true; // Testing is optional
                    break;

                default:
                    isValid = false;
            }

            AppState.validateStep(stepNum, isValid);
            this.updateStepIndicators();

            return isValid;
        },

        // Update step indicators
        updateStepIndicators() {
            for (let i = 1; i <= 6; i++) {
                const indicator = document.getElementById(`step-${i}-indicator`);
                if (indicator) {
                    // Update active state
                    if (i === AppState.currentStep) {
                        indicator.classList.add('active');
                    } else {
                        indicator.classList.remove('active');
                    }

                    // Update completed state
                    if (AppState.isStepValid(i)) {
                        indicator.classList.add('completed');
                    } else {
                        indicator.classList.remove('completed');
                    }
                }
            }
        },

        // Get divisors for batch size (utility function)
        getDivisors(n) {
            const divisors = [];
            for (let i = 1; i <= Math.sqrt(n); i++) {
                if (n % i === 0) {
                    divisors.push(i);
                    if (i !== n / i) {
                        divisors.push(n / i);
                    }
                }
            }
            return divisors.sort((a, b) => a - b);
        },

        // Update valid generations dropdown based on batch size
        updateValidGenerations() {
            const batchSizeInput = document.getElementById('batch-size');
            const generationsSelect = document.getElementById('num-generations');

            if (!batchSizeInput || !generationsSelect) return;

            const batchSize = parseInt(batchSizeInput.value) || 4;
            const divisors = this.getDivisors(batchSize);

            // Clear current options
            generationsSelect.innerHTML = '';

            // Add valid options
            divisors.forEach(divisor => {
                const option = document.createElement('option');
                option.value = divisor;
                option.textContent = divisor;

                // Set default to 2 if available, otherwise 1
                if (divisor === 2 || (divisor === 1 && !generationsSelect.options.length)) {
                    option.selected = true;
                }

                generationsSelect.appendChild(option);
            });

            // If 2 is not a divisor, select the first option
            if (!divisors.includes(2) && generationsSelect.options.length > 0) {
                generationsSelect.options[0].selected = true;
            }
        },

        // Toggle section visibility
        toggleSection(sectionId) {
            const section = document.getElementById(sectionId);
            const icon = document.querySelector(`[onclick*="${sectionId}"] i`);

            if (section) {
                if (section.style.display === 'none') {
                    section.style.display = 'block';
                    if (icon) {
                        icon.classList.remove('fa-chevron-down');
                        icon.classList.add('fa-chevron-up');
                    }
                } else {
                    section.style.display = 'none';
                    if (icon) {
                        icon.classList.remove('fa-chevron-up');
                        icon.classList.add('fa-chevron-down');
                    }
                }
            }
        }
    };

    // Export to window and make functions globally available for onclick handlers
    window.NavigationModule = NavigationModule;

    // Export individual functions for onclick handlers
    window.toggleStep = (stepNum) => NavigationModule.toggleStep(stepNum);
    window.goToStep = (stepNum) => NavigationModule.goToStep(stepNum);
    window.validateAndProceed = (stepNum) => NavigationModule.validateAndProceed(stepNum);
    window.toggleSection = (sectionId) => NavigationModule.toggleSection(sectionId);
    window.updateValidGenerations = () => NavigationModule.updateValidGenerations();

})(window);
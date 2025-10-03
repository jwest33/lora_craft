// ============================================================================
// Chat Templates Module
// ============================================================================

(function(window) {
    'use strict';

    const TemplatesModule = {
        // Template definitions
        templates: {
            'alpaca': {
                name: 'Alpaca',
                system_prefix: '### Instruction:\n',
                system_suffix: '\n\n',
                user_prefix: '### Input:\n',
                user_suffix: '\n\n',
                assistant_prefix: '### Response:\n',
                assistant_suffix: '\n\n'
            },
            'chatml': {
                name: 'ChatML',
                system_prefix: '<|im_start|>system\n',
                system_suffix: '<|im_end|>\n',
                user_prefix: '<|im_start|>user\n',
                user_suffix: '<|im_end|>\n',
                assistant_prefix: '<|im_start|>assistant\n',
                assistant_suffix: '<|im_end|>\n'
            },
            'llama2': {
                name: 'Llama 2',
                system_prefix: '[INST] <<SYS>>\n',
                system_suffix: '\n<</SYS>>\n\n',
                user_prefix: '',
                user_suffix: ' [/INST] ',
                assistant_prefix: '',
                assistant_suffix: ' '
            },
            'llama3': {
                name: 'Llama 3',
                system_prefix: '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n',
                system_suffix: '<|eot_id|>',
                user_prefix: '<|start_header_id|>user<|end_header_id|>\n\n',
                user_suffix: '<|eot_id|>',
                assistant_prefix: '<|start_header_id|>assistant<|end_header_id|>\n\n',
                assistant_suffix: '<|eot_id|>'
            },
            'vicuna': {
                name: 'Vicuna',
                system_prefix: 'SYSTEM: ',
                system_suffix: '\n',
                user_prefix: 'USER: ',
                user_suffix: '\n',
                assistant_prefix: 'ASSISTANT: ',
                assistant_suffix: '\n'
            },
            'mistral': {
                name: 'Mistral Instruct',
                system_prefix: '[INST] ',
                system_suffix: '\n',
                user_prefix: '',
                user_suffix: ' [/INST]',
                assistant_prefix: '',
                assistant_suffix: '</s>'
            },
            'custom': {
                name: 'Custom',
                system_prefix: '',
                system_suffix: '',
                user_prefix: '',
                user_suffix: '',
                assistant_prefix: '',
                assistant_suffix: ''
            }
        },

        // Current template state
        currentTemplate: null,

        // Initialize the module
        init() {
            this.setupEventListeners();
            this.loadTemplateList();
            this.initializeChatTemplate();
        },

        // Setup template-related event listeners
        setupEventListeners() {
            // Template preset selection
            const templateSelect = document.getElementById('template-preset');
            if (templateSelect) {
                templateSelect.addEventListener('change', () => this.onTemplateChange());
            }

            // Custom template field changes
            const templateFields = [
                'system-prefix', 'system-suffix',
                'user-prefix', 'user-suffix',
                'assistant-prefix', 'assistant-suffix'
            ];

            templateFields.forEach(fieldId => {
                const field = document.getElementById(fieldId);
                if (field) {
                    field.addEventListener('input', CoreModule.debounce(() => {
                        this.updateTemplatePreview();
                        this.saveCustomTemplate();
                    }, 500));
                }
            });

            // Test template button
            const testButton = document.getElementById('test-template-btn');
            if (testButton) {
                testButton.addEventListener('click', () => this.testTemplate());
            }
        },

        // Load template list into select
        loadTemplateList() {
            const templateSelect = document.getElementById('template-preset');
            if (!templateSelect) return;

            // Clear existing options
            templateSelect.innerHTML = '';

            // Add template options
            Object.entries(this.templates).forEach(([key, template]) => {
                const option = document.createElement('option');
                option.value = key;
                option.textContent = template.name;
                templateSelect.appendChild(option);
            });

            // Set default or load saved
            const savedTemplate = AppState.getConfigValue('templatePreset') || 'alpaca';
            templateSelect.value = savedTemplate;
            this.onTemplateChange();
        },

        // Initialize chat template display
        initializeChatTemplate() {
            const savedTemplate = AppState.getConfigValue('templatePreset') || 'alpaca';
            this.loadTemplate(savedTemplate);
            this.updateTemplatePreview();
        },

        // Handle template change
        onTemplateChange() {
            const templateSelect = document.getElementById('template-preset');
            if (!templateSelect) return;

            const selectedTemplate = templateSelect.value;
            this.loadTemplate(selectedTemplate);
            this.updateTemplatePreview();

            AppState.setConfigValue('templatePreset', selectedTemplate);
            CoreModule.saveState();
        },

        // Load template into form fields
        loadTemplate(templateKey) {
            const template = this.templates[templateKey];
            if (!template) return;

            this.currentTemplate = templateKey;

            // Update form fields
            this.setFieldValue('system-prefix', template.system_prefix);
            this.setFieldValue('system-suffix', template.system_suffix);
            this.setFieldValue('user-prefix', template.user_prefix);
            this.setFieldValue('user-suffix', template.user_suffix);
            this.setFieldValue('assistant-prefix', template.assistant_prefix);
            this.setFieldValue('assistant-suffix', template.assistant_suffix);

            // Enable/disable fields based on whether it's custom
            const isCustom = templateKey === 'custom';
            this.toggleFieldsEditable(isCustom);

            // Load custom values if custom template
            if (isCustom) {
                this.loadCustomTemplate();
            }
        },

        // Set field value
        setFieldValue(fieldId, value) {
            const field = document.getElementById(fieldId);
            if (field) {
                field.value = value;
            }
        },

        // Toggle fields editable state
        toggleFieldsEditable(editable) {
            const fields = [
                'system-prefix', 'system-suffix',
                'user-prefix', 'user-suffix',
                'assistant-prefix', 'assistant-suffix'
            ];

            fields.forEach(fieldId => {
                const field = document.getElementById(fieldId);
                if (field) {
                    field.readOnly = !editable;
                    field.classList.toggle('readonly', !editable);
                }
            });
        },

        // Update template preview
        updateTemplatePreview() {
            const preview = document.getElementById('template-preview');
            if (!preview) return;

            const systemPrefix = document.getElementById('system-prefix')?.value || '';
            const systemSuffix = document.getElementById('system-suffix')?.value || '';
            const userPrefix = document.getElementById('user-prefix')?.value || '';
            const userSuffix = document.getElementById('user-suffix')?.value || '';
            const assistantPrefix = document.getElementById('assistant-prefix')?.value || '';
            const assistantSuffix = document.getElementById('assistant-suffix')?.value || '';

            // Get the actual system prompt based on template selection
            let systemPrompt = 'You are a helpful assistant.';
            const templateSelect = document.getElementById('prompt-template-select');
            const customSystemPrompt = document.getElementById('custom-system-prompt');
            const systemPromptField = document.getElementById('system-prompt');
            const templateEditor = document.getElementById('template-editor');

            // Check if custom template editor is visible and has content
            const isCustomEditorVisible = templateEditor && templateEditor.style.display !== 'none';
            const hasCustomContent = customSystemPrompt?.value && customSystemPrompt.value.trim() !== '';

            // Prioritize custom-system-prompt if the custom editor is visible
            if (isCustomEditorVisible && hasCustomContent) {
                systemPrompt = customSystemPrompt.value;
                // Sync to hidden field for backend
                if (systemPromptField) {
                    systemPromptField.value = customSystemPrompt.value;
                }
            } else if (templateSelect?.value === 'custom' && hasCustomContent) {
                // Fallback: use custom-system-prompt if custom is explicitly selected
                systemPrompt = customSystemPrompt.value;
                if (systemPromptField) {
                    systemPromptField.value = customSystemPrompt.value;
                }
            } else {
                // For built-in templates, use system-prompt (hidden field)
                if (systemPromptField?.value) {
                    systemPrompt = systemPromptField.value;
                }
            }

            // Get reasoning and solution markers for GRPO format
            const reasoningStart = document.getElementById('custom-reasoning-start')?.value || '<start_working_out>';
            const reasoningEnd = document.getElementById('custom-reasoning-end')?.value || '<end_working_out>';
            const solutionStart = document.getElementById('custom-solution-start')?.value || '<SOLUTION>';
            const solutionEnd = document.getElementById('custom-solution-end')?.value || '</SOLUTION>';

            // Create example conversation showing GRPO training format
            const userMessage = 'What is 2 + 2?';
            const assistantMessage = `${reasoningStart}Let me think about this. 2 + 2 equals 4.${reasoningEnd}\n${solutionStart}4${solutionEnd}`;

            const exampleConversation = this.formatConversation(
                systemPrefix, systemSuffix,
                userPrefix, userSuffix,
                assistantPrefix, assistantSuffix,
                systemPrompt,
                userMessage,
                assistantMessage
            );

            // Display with syntax highlighting
            preview.innerHTML = `<pre class="template-preview-content">${this.highlightTemplate(exampleConversation)}</pre>`;
        },

        // Format conversation with template
        formatConversation(sysPrefix, sysSuffix, userPrefix, userSuffix, asstPrefix, asstSuffix, system, user, assistant) {
            let formatted = '';

            // Add system prompt with proper spacing
            if (system) {
                formatted += sysPrefix + system + sysSuffix;
                // Add newline separation if suffixes don't already provide it
                if (sysSuffix && !sysSuffix.endsWith('\n')) {
                    formatted += '\n';
                } else if (!sysSuffix) {
                    formatted += '\n\n';
                }
            }

            // Add user message with proper spacing
            formatted += userPrefix + user + userSuffix;
            // Add newline separation if suffixes don't already provide it
            if (userSuffix && !userSuffix.endsWith('\n')) {
                formatted += '\n';
            } else if (!userSuffix) {
                formatted += '\n\n';
            }

            // Add assistant response
            formatted += asstPrefix + assistant + asstSuffix;

            return formatted;
        },

        // Highlight template syntax
        highlightTemplate(text) {
            // Escape HTML first
            let highlighted = CoreModule.escapeHtml(text);

            // Highlight special tokens
            highlighted = highlighted.replace(/(&lt;\|[^|]+\|&gt;)/g, '<span class="token-special">$1</span>');
            highlighted = highlighted.replace(/(\[INST\]|\[\/INST\])/g, '<span class="token-inst">$1</span>');
            highlighted = highlighted.replace(/(&lt;&lt;SYS&gt;&gt;|&lt;&lt;\/SYS&gt;&gt;)/g, '<span class="token-sys">$1</span>');
            highlighted = highlighted.replace(/(###\s+\w+:)/g, '<span class="token-header">$1</span>');
            highlighted = highlighted.replace(/(SYSTEM:|USER:|ASSISTANT:)/g, '<span class="token-role">$1</span>');

            return highlighted;
        },

        // Test template with sample data
        testTemplate() {
            const systemPrefix = document.getElementById('system-prefix')?.value || '';
            const systemSuffix = document.getElementById('system-suffix')?.value || '';
            const userPrefix = document.getElementById('user-prefix')?.value || '';
            const userSuffix = document.getElementById('user-suffix')?.value || '';
            const assistantPrefix = document.getElementById('assistant-prefix')?.value || '';
            const assistantSuffix = document.getElementById('assistant-suffix')?.value || '';

            // Create test conversation
            const testData = {
                system: 'You are a knowledgeable assistant specializing in science.',
                conversations: [
                    { user: 'What is photosynthesis?', assistant: 'Photosynthesis is the process by which plants convert light energy into chemical energy.' },
                    { user: 'Why is it important?', assistant: 'It\'s crucial because it produces oxygen and forms the base of most food chains on Earth.' }
                ]
            };

            // Format the conversation
            let formatted = '';
            if (testData.system) {
                formatted += systemPrefix + testData.system + systemSuffix;
            }

            testData.conversations.forEach(conv => {
                formatted += userPrefix + conv.user + userSuffix;
                formatted += assistantPrefix + conv.assistant + assistantSuffix;
            });

            // Show in modal
            this.showTestResults(formatted);
        },

        // Show test results in modal
        showTestResults(formattedText) {
            const modalId = 'templateTestModal';
            let modal = document.getElementById(modalId);

            if (!modal) {
                const modalHtml = `
                    <div class="modal fade" id="${modalId}" tabindex="-1">
                        <div class="modal-dialog modal-lg">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h5 class="modal-title">Template Test Results</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                                </div>
                                <div class="modal-body">
                                    <div class="mb-3">
                                        <h6>Formatted Output:</h6>
                                        <pre class="bg-light p-3 rounded" style="white-space: pre-wrap; max-height: 400px; overflow-y: auto;"></pre>
                                    </div>
                                    <div class="alert alert-info">
                                        <i class="fas fa-info-circle me-2"></i>
                                        This shows how your template formats conversations. Check that special tokens and separators appear correctly.
                                    </div>
                                </div>
                                <div class="modal-footer">
                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                                    <button type="button" class="btn btn-primary" onclick="TemplatesModule.copyTestResults()">
                                        <i class="fas fa-copy me-2"></i>Copy
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                document.body.insertAdjacentHTML('beforeend', modalHtml);
                modal = document.getElementById(modalId);
            }

            // Update content
            const preElement = modal.querySelector('pre');
            preElement.innerHTML = this.highlightTemplate(formattedText);
            preElement.dataset.rawText = formattedText;

            // Show modal
            new bootstrap.Modal(modal).show();
        },

        // Copy test results to clipboard
        copyTestResults() {
            const modal = document.getElementById('templateTestModal');
            const preElement = modal?.querySelector('pre');
            const text = preElement?.dataset.rawText;

            if (text) {
                navigator.clipboard.writeText(text)
                    .then(() => {
                        CoreModule.showAlert('Copied to clipboard!', 'success');
                    })
                    .catch(err => {
                        console.error('Failed to copy:', err);
                        CoreModule.showAlert('Failed to copy to clipboard', 'danger');
                    });
            }
        },

        // Save custom template
        saveCustomTemplate() {
            if (this.currentTemplate !== 'custom') return;

            const customTemplate = {
                system_prefix: document.getElementById('system-prefix')?.value || '',
                system_suffix: document.getElementById('system-suffix')?.value || '',
                user_prefix: document.getElementById('user-prefix')?.value || '',
                user_suffix: document.getElementById('user-suffix')?.value || '',
                assistant_prefix: document.getElementById('assistant-prefix')?.value || '',
                assistant_suffix: document.getElementById('assistant-suffix')?.value || ''
            };

            AppState.setConfigValue('customTemplate', customTemplate);
            CoreModule.saveState();
        },

        // Load custom template
        loadCustomTemplate() {
            const customTemplate = AppState.getConfigValue('customTemplate');
            if (!customTemplate) return;

            this.setFieldValue('system-prefix', customTemplate.system_prefix);
            this.setFieldValue('system-suffix', customTemplate.system_suffix);
            this.setFieldValue('user-prefix', customTemplate.user_prefix);
            this.setFieldValue('user-suffix', customTemplate.user_suffix);
            this.setFieldValue('assistant-prefix', customTemplate.assistant_prefix);
            this.setFieldValue('assistant-suffix', customTemplate.assistant_suffix);
        },

        // Export template configuration
        exportTemplateConfig() {
            return {
                preset: document.getElementById('template-preset')?.value,
                system_prefix: document.getElementById('system-prefix')?.value,
                system_suffix: document.getElementById('system-suffix')?.value,
                user_prefix: document.getElementById('user-prefix')?.value,
                user_suffix: document.getElementById('user-suffix')?.value,
                assistant_prefix: document.getElementById('assistant-prefix')?.value,
                assistant_suffix: document.getElementById('assistant-suffix')?.value,
                // Add custom prompt markers and system prompt
                reasoning_start: document.getElementById('custom-reasoning-start')?.value || '<start_working_out>',
                reasoning_end: document.getElementById('custom-reasoning-end')?.value || '<end_working_out>',
                solution_start: document.getElementById('custom-solution-start')?.value || '<SOLUTION>',
                solution_end: document.getElementById('custom-solution-end')?.value || '</SOLUTION>',
                system_prompt: document.getElementById('custom-system-prompt')?.value || '',
                chat_template: document.getElementById('custom-chat-template')?.value || '',
                chat_template_type: document.getElementById('prompt-template-select')?.value || 'grpo-default'
            };
        },

        // Import template configuration
        importTemplateConfig(config) {
            console.log('TemplatesModule.importTemplateConfig called with:', config);

            if (config.preset) {
                const templateSelect = document.getElementById('template-preset');
                if (templateSelect) {
                    templateSelect.value = config.preset;
                    this.onTemplateChange();
                } else {
                    console.warn('template-preset element not found');
                }
            }

            // If custom, import the custom values
            if (config.preset === 'custom') {
                this.setFieldValue('system-prefix', config.system_prefix || '');
                this.setFieldValue('system-suffix', config.system_suffix || '');
                this.setFieldValue('user-prefix', config.user_prefix || '');
                this.setFieldValue('user-suffix', config.user_suffix || '');
                this.setFieldValue('assistant-prefix', config.assistant_prefix || '');
                this.setFieldValue('assistant-suffix', config.assistant_suffix || '');
                this.saveCustomTemplate();
                this.updateTemplatePreview();
            }

            // Restore custom prompt markers (reasoning and solution)
            if (config.reasoning_start) {
                const reasoningStartElement = document.getElementById('custom-reasoning-start');
                const reasoningStartHidden = document.getElementById('reasoning-start');
                if (reasoningStartElement) reasoningStartElement.value = config.reasoning_start;
                if (reasoningStartHidden) reasoningStartHidden.value = config.reasoning_start;
            }
            if (config.reasoning_end) {
                const reasoningEndElement = document.getElementById('custom-reasoning-end');
                const reasoningEndHidden = document.getElementById('reasoning-end');
                if (reasoningEndElement) reasoningEndElement.value = config.reasoning_end;
                if (reasoningEndHidden) reasoningEndHidden.value = config.reasoning_end;
            }
            if (config.solution_start) {
                const solutionStartElement = document.getElementById('custom-solution-start');
                const solutionStartHidden = document.getElementById('solution-start');
                if (solutionStartElement) solutionStartElement.value = config.solution_start;
                if (solutionStartHidden) solutionStartHidden.value = config.solution_start;
            }
            if (config.solution_end) {
                const solutionEndElement = document.getElementById('custom-solution-end');
                const solutionEndHidden = document.getElementById('solution-end');
                if (solutionEndElement) solutionEndElement.value = config.solution_end;
                if (solutionEndHidden) solutionEndHidden.value = config.solution_end;
            }

            // Restore system prompt
            if (config.system_prompt) {
                const systemPromptElement = document.getElementById('custom-system-prompt');
                const systemPromptHidden = document.getElementById('system-prompt');
                if (systemPromptElement) systemPromptElement.value = config.system_prompt;
                if (systemPromptHidden) systemPromptHidden.value = config.system_prompt;
            }

            // Restore chat template
            if (config.chat_template) {
                const chatTemplateElement = document.getElementById('custom-chat-template');
                const chatTemplateHidden = document.getElementById('chat-template');
                if (chatTemplateElement) chatTemplateElement.value = config.chat_template;
                if (chatTemplateHidden) chatTemplateHidden.value = config.chat_template;
            }

            // Restore chat template type selection
            if (config.chat_template_type) {
                const chatTemplateTypeSelect = document.getElementById('prompt-template-select');
                if (chatTemplateTypeSelect) {
                    chatTemplateTypeSelect.value = config.chat_template_type;
                    // Trigger change event if needed
                    if (typeof onChatTemplateTypeChange === 'function') {
                        onChatTemplateTypeChange();
                    }
                }
            }

            // Update preview if template preview function exists
            if (typeof updateTemplatePreview === 'function') {
                updateTemplatePreview();
            }
        },

        // Validate template configuration
        validateTemplateConfig() {
            const userPrefix = document.getElementById('user-prefix')?.value;
            const assistantPrefix = document.getElementById('assistant-prefix')?.value;

            // Basic validation - ensure at least user and assistant prefixes exist
            if (!userPrefix && !assistantPrefix) {
                CoreModule.showAlert('Template must have user and assistant markers', 'warning');
                return false;
            }

            return true;
        }
    };

    // Export to window
    window.TemplatesModule = TemplatesModule;

    // Export functions for onclick handlers
    window.updateTemplatePreview = () => TemplatesModule.updateTemplatePreview();
    window.initializeChatTemplate = () => TemplatesModule.initializeChatTemplate();

})(window);

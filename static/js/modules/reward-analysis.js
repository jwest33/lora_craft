// ============================================================================
// Reward Analysis Module - Real-time reward component visualization
// ============================================================================

(function(window) {
    'use strict';

    const RewardAnalysisModule = {
        samples: [],
        maxSamples: 20,  // Keep last 20 samples

        init() {
            console.log('Reward Analysis Module initialized');
            this.samples = [];
        },

        addSample(data) {
            // Add new sample to the beginning
            this.samples.unshift(data);

            // Keep only last maxSamples
            if (this.samples.length > this.maxSamples) {
                this.samples = this.samples.slice(0, this.maxSamples);
            }

            this.renderSamples();
        },

        renderSamples() {
            const container = document.getElementById('reward-samples-container');
            if (!container) return;

            if (this.samples.length === 0) {
                container.innerHTML = '<p class="text-muted">No reward samples yet. Samples will appear during training...</p>';
                return;
            }

            let html = '';
            this.samples.forEach((sample, index) => {
                html += this.renderSample(sample, index);
            });

            container.innerHTML = html;
        },

        renderSample(sample, index) {
            const rewardClass = this.getRewardClass(sample.total_reward);
            const rewardPercent = (sample.total_reward * 100).toFixed(1);

            return `
                <div class="reward-sample-card">
                    <div class="reward-sample-header">
                        <div class="reward-sample-meta">
                            <span class="reward-sample-step">Step ${sample.step || 'N/A'}</span>
                            <span class="reward-sample-timestamp">${this.formatTimestamp(sample.timestamp)}</span>
                        </div>
                        <div class="reward-score ${rewardClass}">
                            ${rewardPercent}%
                        </div>
                    </div>

                    <div class="reward-sample-content">
                        <div class="reward-section">
                            <div class="reward-section-title">Instruction</div>
                            <div class="reward-text">${this.escapeHtml(sample.instruction || 'N/A')}</div>
                        </div>

                        <div class="reward-section">
                            <div class="reward-section-title">Generated Response</div>
                            <div class="reward-text">${this.escapeHtml(sample.generated || 'N/A')}</div>
                        </div>

                        <div class="reward-section">
                            <div class="reward-section-title">Component Breakdown</div>
                            <div class="reward-components">
                                ${this.renderComponents(sample.components)}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        },

        renderComponents(components) {
            if (!components || Object.keys(components).length === 0) {
                return '<p class="text-muted">No component data</p>';
            }

            let html = '<div class="component-list">';
            for (const [name, score] of Object.entries(components)) {
                const scorePercent = (score * 100).toFixed(1);
                const barClass = this.getRewardClass(score);

                html += `
                    <div class="component-item">
                        <div class="component-name">${this.escapeHtml(name)}</div>
                        <div class="component-bar-container">
                            <div class="component-bar ${barClass}" style="width: ${scorePercent}%"></div>
                        </div>
                        <div class="component-score">${scorePercent}%</div>
                    </div>
                `;
            }
            html += '</div>';
            return html;
        },

        getRewardClass(score) {
            if (score >= 0.8) return 'reward-high';
            if (score >= 0.5) return 'reward-medium';
            return 'reward-low';
        },

        formatTimestamp(timestamp) {
            if (!timestamp) return 'N/A';
            const date = new Date(timestamp * 1000);
            return date.toLocaleTimeString();
        },

        escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        },

        clear() {
            this.samples = [];
            this.renderSamples();
        }
    };

    // Expose to global scope
    window.RewardAnalysisModule = RewardAnalysisModule;

})(window);

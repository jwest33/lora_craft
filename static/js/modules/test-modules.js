// ============================================================================
// Module Testing Script - Verify all modules are loaded correctly
// ============================================================================

(function() {
    'use strict';

    // List of expected modules
    const expectedModules = [
        'AppState',       // state.js
        'SocketModule',   // socket.js
        'CoreModule',     // core.js
        'NavigationModule', // navigation.js
        'ModelsModule',   // models.js
        'DatasetModule',  // datasets.js
        'TemplatesModule', // templates.js
        'TrainingModule', // training.js
        'SessionsModule', // sessions.js
        'ConfigModule',   // config.js
        'ExportModule',   // export.js
        'TestingModule'   // testing.js
    ];

    // Test function
    function testModules() {
        console.log('=== Testing Module Loading ===');

        let successCount = 0;
        let failureCount = 0;
        const results = [];

        expectedModules.forEach(moduleName => {
            const exists = typeof window[moduleName] !== 'undefined';

            if (exists) {
                successCount++;
                results.push(`✅ ${moduleName} - Loaded successfully`);
            } else {
                failureCount++;
                results.push(`❌ ${moduleName} - NOT FOUND`);
            }
        });

        // Display results
        console.log('\n--- Test Results ---');
        results.forEach(result => console.log(result));

        console.log('\n--- Summary ---');
        console.log(`Total Modules: ${expectedModules.length}`);
        console.log(`✅ Loaded: ${successCount}`);
        console.log(`❌ Missing: ${failureCount}`);

        if (failureCount === 0) {
            console.log('\n🎉 All modules loaded successfully!');
        } else {
            console.error('\n⚠️ Some modules failed to load. Check console for errors.');
        }

        // Test key functions
        console.log('\n--- Testing Key Functions ---');

        const functionTests = [
            { module: 'CoreModule', func: 'init', desc: 'Core initialization' },
            { module: 'NavigationModule', func: 'init', desc: 'Navigation initialization' },
            { module: 'SocketModule', func: 'init', desc: 'Socket initialization' },
            { module: 'ModelsModule', func: 'init', desc: 'Models initialization' },
            { module: 'DatasetModule', func: 'init', desc: 'Dataset initialization' }
        ];

        functionTests.forEach(test => {
            try {
                if (window[test.module] && typeof window[test.module][test.func] === 'function') {
                    console.log(`✅ ${test.desc} - Function exists`);
                } else {
                    console.log(`❌ ${test.desc} - Function not found`);
                }
            } catch (error) {
                console.log(`❌ ${test.desc} - Error: ${error.message}`);
            }
        });

        // Test global functions (onclick handlers)
        console.log('\n--- Testing Global Functions ---');

        const globalFunctions = [
            'toggleStep',
            'goToStep',
            'validateAndProceed',
            'toggleSection',
            'browseDataset',
            'previewDataset',
            'startTraining',
            'stopTraining',
            'startExport',
            'runSingleTest',
            'startBatchTest'
        ];

        globalFunctions.forEach(funcName => {
            if (typeof window[funcName] === 'function') {
                console.log(`✅ ${funcName}() - Available`);
            } else {
                console.log(`❌ ${funcName}() - Not found`);
            }
        });

        return failureCount === 0;
    }

    // Run test after DOM is loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            setTimeout(testModules, 1000); // Wait for modules to initialize
        });
    } else {
        setTimeout(testModules, 1000);
    }

    // Expose test function globally
    window.testModules = testModules;

})();
# Module Migration Guide for app.js

## Overview
This guide helps migrate the 7,858-line app.js into manageable modules.

## Module Structure

### 1. State Management (state.js) ✅
- Global application state
- Shared variables
- State methods

### 2. Core Module Pattern

Each module follows this pattern:

```javascript
// ============================================================================
// Module Name
// ============================================================================

(function(window) {
    'use strict';

    // Private variables
    const privateVar = {};

    // Private functions
    function privateFunction() {
        // Implementation
    }

    // Public API
    const ModuleName = {
        init() {
            // Initialization
        },

        publicMethod() {
            // Public method
        }
    };

    // Export to window
    window.ModuleName = ModuleName;

})(window);
```

## Migration Steps for Each Module

### Step 1: Identify Functions
Extract all functions belonging to the module using:
```bash
grep -n "function.*dataset" static/js/app.js  # Example for dataset functions
```

### Step 2: Extract Functions
Copy functions from app.js to the new module file.

### Step 3: Update Dependencies
- Replace global variables with `AppState.variable`
- Import needed functions from other modules
- Update socket references to `AppState.socket`

### Step 4: Update Event Handlers
Change inline event handlers to use module methods:
```javascript
// Before
onclick="selectDataset('key')"

// After
onclick="DatasetModule.selectDataset('key')"
```

### Step 5: Update app.js
Remove extracted functions from app.js.

## Module Dependencies Map

```
state.js (no dependencies)
    ↓
socket.js (depends on: state)
    ↓
core.js (depends on: state, socket)
    ↓
navigation.js (depends on: state, core)
models.js (depends on: state, socket, navigation)
datasets.js (depends on: state, socket, core)
templates.js (depends on: state, socket)
training.js (depends on: state, socket, charts)
sessions.js (depends on: state, socket, training)
config.js (depends on: state, all modules)
export.js (depends on: state, socket, sessions)
testing.js (depends on: state, socket, models)
```

## HTML Updates Required

Add to index.html before closing </body>:
```html
<!-- Modular JavaScript -->
<script src="{{ url_for('static', filename='js/modules/state.js') }}"></script>
<script src="{{ url_for('static', filename='js/modules/socket.js') }}"></script>
<script src="{{ url_for('static', filename='js/modules/core.js') }}"></script>
<script src="{{ url_for('static', filename='js/modules/navigation.js') }}"></script>
<script src="{{ url_for('static', filename='js/modules/models.js') }}"></script>
<script src="{{ url_for('static', filename='js/modules/datasets.js') }}"></script>
<script src="{{ url_for('static', filename='js/modules/templates.js') }}"></script>
<script src="{{ url_for('static', filename='js/modules/training.js') }}"></script>
<script src="{{ url_for('static', filename='js/modules/sessions.js') }}"></script>
<script src="{{ url_for('static', filename='js/modules/config.js') }}"></script>
<script src="{{ url_for('static', filename='js/modules/export.js') }}"></script>
<script src="{{ url_for('static', filename='js/modules/testing.js') }}"></script>

<!-- Main app initialization -->
<script src="{{ url_for('static', filename='js/app-init.js') }}"></script>
```

## Testing Each Module

After creating each module:
1. Test the specific functionality
2. Check console for errors
3. Verify event handlers work
4. Test socket communications
5. Verify state management

## Common Pitfalls to Avoid

1. **Circular Dependencies**: Plan module dependencies carefully
2. **Event Handler Scope**: Update all onclick/onchange references
3. **Socket References**: Always use `AppState.socket`
4. **Chart Instances**: Store in `AppState.charts`
5. **DOM Ready**: Ensure DOM is loaded before initialization

## Gradual Migration Strategy

1. **Phase 1**: Core infrastructure (state, socket, core)
2. **Phase 2**: UI modules (navigation, models, datasets)
3. **Phase 3**: Feature modules (templates, training, sessions)
4. **Phase 4**: Advanced modules (config, export, testing)

## Rollback Plan

Keep original app.js as app.js.backup until migration is complete and tested.
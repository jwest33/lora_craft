
UI enhancements
* Ensure model name export and saving work correctly
    * Training Sessions shows the correct model used but then '_dataset_' instead of the actual dataset name.
    * Export model shows a default export name of 'model_' instead of the actual model name with the correct dataset name
    * Exported gguf models don't retain the name entered. The files are just 'model-quantization'
* Update model comparison so model output containers show the model name instead of 'trained model' and 'challenger model'
* Ensure 'Generations' drop-down always defaults to match the Batch Size (needs to be populated after batch size)
* Replace all loading bars with spinners that don't show progress, just that the app is loading 
* Reduce the vertical size of 'selection-card' icons in the front end
* Update 'Step Connector' to remove the complete indictor and replace with a mildly animated indiactor that just shows the current step the user has selected. Also a mild animated indicator if there are any errors in any steps.
* Add hover over effect for application icon in top left corner. If user clicks on icon, an information popup appears. 
* Update light mode so the step cards are more visually distinct from the white background. Ensure all navbar buttons and text are visible on the white background. Do a thorough check for any elements that don't have both a light and dark mode defined.

App Enhancements
* Implement evaluation module under test where the user can provide a prompt, and upload a test CSV with an expected output column

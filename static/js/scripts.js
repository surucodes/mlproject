document.addEventListener('DOMContentLoaded', function() {
    // For index.html: Animate Start Predicting button
    const startButton = document.getElementById('start-btn');
    if (startButton) {
        startButton.addEventListener('mouseover', function() {
            this.classList.add('scale-105');
        });
        startButton.addEventListener('mouseout', function() {
            this.classList.remove('scale-105');
        });
    }

    // For home.html: Form validation and loading state
    const predictionForm = document.getElementById('prediction-form');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(event) {
            const inputs = document.querySelectorAll('input[type="number"]');
            const errorMessage = document.getElementById('error-message');
            const loadingSpinner = document.getElementById('loading');
            const predictButton = document.getElementById('predict-btn');

            // Reset error message
            errorMessage.classList.add('hidden');
            errorMessage.textContent = '';

            // Validate number inputs (0-100)
            for (let input of inputs) {
                const value = parseFloat(input.value);
                if (isNaN(value) || value < 0 || value > 100) {
                    event.preventDefault();
                    errorMessage.textContent = `Please enter a valid score between 0 and 100 for ${input.name.replace('_', ' ')}.`;
                    errorMessage.classList.remove('hidden');
                    return;
                }
            }

            // Show loading state
            loadingSpinner.classList.remove('hidden');
            predictButton.disabled = true;
            predictButton.classList.add('opacity-50', 'cursor-not-allowed');
        });
    }

    // Reset loading state when page reloads (e.g., after prediction)
    const loadingSpinner = document.getElementById('loading');
    const predictButton = document.getElementById('predict-btn');
    if (loadingSpinner && predictButton) {
        loadingSpinner.classList.add('hidden');
        predictButton.disabled = false;
        predictButton.classList.remove('opacity-50', 'cursor-not-allowed');
    }
});
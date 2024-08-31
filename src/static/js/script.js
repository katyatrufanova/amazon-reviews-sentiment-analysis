document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const reviewText = document.getElementById('review-text');
    const result = document.getElementById('result');
    const sentimentSpan = document.getElementById('sentiment');
    const confidenceSpan = document.getElementById('confidence');
    const confidenceFill = document.getElementById('confidence-fill');

    analyzeBtn.addEventListener('click', async () => {
        const text = reviewText.value.trim();
        if (text) {
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text }),
                });

                if (response.ok) {
                    const data = await response.json();
                    sentimentSpan.textContent = data.sentiment;
                    confidenceSpan.textContent = (data.confidence * 100).toFixed(2);
                    confidenceFill.style.width = `${data.confidence * 100}%`;
                    result.classList.remove('hidden');
                } else {
                    throw new Error('Failed to analyze sentiment');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred while analyzing the sentiment. Please try again.');
            }
        } else {
            alert('Please enter a review text to analyze.');
        }
    });
});
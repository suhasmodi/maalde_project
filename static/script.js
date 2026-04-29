document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const imageUpload = document.getElementById('image-upload');
    const uploadArea = document.getElementById('upload-area');
    const uploadContent = document.querySelector('.upload-content');
    const imagePreview = document.getElementById('image-preview');
    
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('loading-spinner');
    
    const resultContainer = document.getElementById('result-container');
    const predictedQty = document.getElementById('predicted-qty');

    // Handle Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
    });

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;
        if(files.length) {
            imageUpload.files = files;
            updateImagePreview();
        }
    }

    // Handle Click Upload
    imageUpload.addEventListener('change', updateImagePreview);

    function updateImagePreview() {
        if (imageUpload.files && imageUpload.files[0]) {
            const reader = new FileReader();
            
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                imagePreview.style.display = 'block';
                uploadContent.style.opacity = '0';
            }
            
            reader.readAsDataURL(imageUpload.files[0]);
            
            // Hide result when new image is uploaded
            resultContainer.classList.add('hidden');
        }
    }

    // Handle Form Submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (!imageUpload.files[0]) {
            alert('Please select an image first.');
            return;
        }

        // UI Loading State
        btnText.style.display = 'none';
        spinner.style.display = 'block';
        submitBtn.disabled = true;
        resultContainer.classList.add('hidden');

        const formData = new FormData(form);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                predictedQty.textContent = data.predicted_qty;
                
                // Show result with slight delay for smooth animation
                setTimeout(() => {
                    resultContainer.classList.remove('hidden');
                    resultContainer.style.position = 'relative';
                    resultContainer.style.visibility = 'visible';
                }, 300);
            } else {
                alert('Prediction Error: ' + data.error);
            }

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred connecting to the server.');
        } finally {
            // Restore UI State
            btnText.style.display = 'block';
            spinner.style.display = 'none';
            submitBtn.disabled = false;
        }
    });
});

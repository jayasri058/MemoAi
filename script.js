// MemoAI Website JavaScript
document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const startRecordingBtn = document.getElementById('start-recording');
    const voiceOutput = document.getElementById('voice-output');
    const recordingStatus = document.getElementById('recording-status');
    const clearVoiceBtn = document.getElementById('clear-voice');
    const submitVoiceBtn = document.getElementById('submit-voice');
    const savingSpinner = document.getElementById('saving-spinner');
    const saveStatus = document.getElementById('save-status');
    const uploadArea = document.getElementById('upload-area');
    const imageUpload = document.getElementById('image-upload');
    const imagePreview = document.getElementById('image-preview');
    const imageDescriptionSection = document.getElementById('image-description-section');
    const imageDescription = document.getElementById('image-description');
    const submitImageBtn = document.getElementById('submit-image');
    const clearImageBtn = document.getElementById('clear-image');
    const imageSavingSpinner = document.getElementById('image-saving-spinner');
    const imageSaveStatus = document.getElementById('image-save-status');
    const processBtn = document.getElementById('process-btn');
    const categoryResult = document.getElementById('category-result');
    const contextResult = document.getElementById('context-result');
    const tagsResult = document.getElementById('tags-result');
    const searchQuery = document.getElementById('search-query');
    const searchBtn = document.getElementById('search-btn');
    const searchResults = document.getElementById('search-results');
    const summaryBtn = document.getElementById('summary-btn');
    const summaryResult = document.getElementById('summary-result');
    const summaryText = document.getElementById('summary-text');

    console.log('AI Summary elements initialization:', {
        btn: !!summaryBtn,
        result: !!summaryResult,
        text: !!summaryText
    });

    // Camera elements
    const useCameraBtn = document.getElementById('use-camera-btn');
    const cameraContainer = document.getElementById('camera-container');
    const cameraVideo = document.getElementById('camera-video');
    const cameraCanvas = document.getElementById('camera-canvas');
    const capturePhotoBtn = document.getElementById('capture-photo-btn');
    const cancelCameraBtn = document.getElementById('cancel-camera-btn');
    const toggleCameraBtn = document.getElementById('toggle-camera-btn');

    // State variables
    let isRecording = false;
    let recognition;
    let silenceTimeout;
    let currentSearchResults = [];
    let cameraStream = null;
    let currentFacingMode = 'user'; // 'user' for front camera, 'environment' for back camera

    // Initialize speech recognition
    initializeSpeechRecognition();

    // Helper for secure API headers
    function getAuthHeaders() {
        const userString = sessionStorage.getItem('user');
        if (!userString) return { 'Content-Type': 'application/json' };
        try {
            const user = JSON.parse(userString);
            if (!user || !user.id) {
                console.warn('User object found but no ID present');
                return { 'Content-Type': 'application/json' };
            }
            return {
                'Content-Type': 'application/json',
                'X-User-Id': String(user.id)
            };
        } catch (e) {
            return { 'Content-Type': 'application/json' };
        }
    }

    // Event Listeners with defensive checks
    if (startRecordingBtn) startRecordingBtn.addEventListener('click', toggleRecording);
    if (clearVoiceBtn) clearVoiceBtn.addEventListener('click', clearVoiceOutput);
    if (submitVoiceBtn) submitVoiceBtn.addEventListener('click', submitVoiceMemory);

    if (uploadArea) {
        uploadArea.addEventListener('click', (e) => {
            // Only trigger file upload if not clicking the camera button
            if (e.target.id !== 'use-camera-btn' && !e.target.closest('#use-camera-btn')) {
                if (imageUpload) imageUpload.click();
            }
        });
    }

    if (imageUpload) imageUpload.addEventListener('change', handleImageUpload);
    if (submitImageBtn) submitImageBtn.addEventListener('click', submitImageMemory);
    if (clearImageBtn) clearImageBtn.addEventListener('click', clearImage);
    if (processBtn) processBtn.addEventListener('click', processMemory);
    if (searchBtn) searchBtn.addEventListener('click', searchMemories);
    if (summaryBtn) {
        summaryBtn.addEventListener('click', getMemorySummary);
    }
    if (searchQuery) {
        searchQuery.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                searchMemories();
            }
        });
    }

    // Load memories from backend on dashboard load
    async function loadMemoriesFromBackend() {
        if (!searchResults) return;
        const userString = sessionStorage.getItem('user');
        if (!userString) return; // Not logged in

        try {
            searchResults.innerHTML = '<p class="no-results">Loading your memories...</p>';
            const response = await fetch('/api/search-memories?q=*', {
                headers: getAuthHeaders()
            });
            if (!response.ok) {
                // Try a broad search fallback
                const r2 = await fetch('/api/search-memories?q=memory', { headers: getAuthHeaders() });
                if (!r2.ok) { searchResults.innerHTML = '<p class="no-results">Search for memories above.</p>'; return; }
                const d2 = await r2.json();
                currentSearchResults = d2.results || [];
                if (currentSearchResults.length > 0) displaySearchResults(currentSearchResults);
                else searchResults.innerHTML = '<p class="no-results">No memories yet. Start recording!</p>';
                return;
            }
            const data = await response.json();
            currentSearchResults = data.results || [];
            if (currentSearchResults.length > 0) {
                displaySearchResults(currentSearchResults);
            } else {
                searchResults.innerHTML = '<p class="no-results">No memories yet. Start recording!</p>';
            }
        } catch (e) {
            console.log('Could not auto-load memories:', e);
            searchResults.innerHTML = '<p class="no-results">Search for memories above.</p>';
        }
    }

    // Auto-load memories when on dashboard
    loadMemoriesFromBackend();

    // Initial usage check
    fetchUserUsage();

    async function fetchUserUsage() {
        const usageContainer = document.getElementById('usage-container');
        if (!usageContainer) return;

        try {
            const response = await fetch('/api/user/usage', {
                headers: getAuthHeaders()
            });
            if (response.ok) {
                const data = await response.json();
                updateUsageBar(data.memories_used, data.memory_limit, data.is_premium);
                usageContainer.style.display = 'block';
            }
        } catch (e) {
            console.error('Failed to fetch usage:', e);
        }
    }

    function updateUsageBar(used, limit, isPremium) {
        const text = document.getElementById('usage-text');
        const fill = document.getElementById('usage-bar-fill');
        const plan = document.getElementById('usage-plan');
        const link = document.getElementById('upgrade-link');

        if (isPremium) {
            if (text) text.textContent = 'Unlimited Memories';
            if (fill) fill.style.width = '100%';
            if (plan) plan.textContent = 'Premium Plan';
            if (link) link.style.display = 'none';
            return;
        }

        if (text) text.textContent = `${used} / ${limit} memories`;
        if (fill) {
            const percent = Math.min(100, (used / limit) * 100);
            fill.style.width = `${percent}%`;
            if (percent > 80) fill.style.background = 'linear-gradient(90deg, #F72585, #EF233C)';
        }
        if (plan) plan.textContent = 'Free Tier';
        if (link) link.style.display = 'inline-block';
    }

    function showPaymentModal(data) {
        const modal = document.getElementById('payment-modal');
        const amount = document.getElementById('premium-amount');
        if (!modal) return;

        if (data && data.amount) {
            amount.textContent = data.amount;
        }

        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    // Modal listeners
    const closePaymentBtn = document.getElementById('close-payment-modal');
    if (closePaymentBtn) {
        closePaymentBtn.addEventListener('click', () => {
            document.getElementById('payment-modal').classList.remove('active');
            document.body.style.overflow = '';
        });
    }

    const payNowBtn = document.getElementById('pay-now-btn');
    if (payNowBtn) {
        payNowBtn.addEventListener('click', handlePayment);
    }

    async function handlePayment() {
        payNowBtn.disabled = true;
        payNowBtn.textContent = 'Initiating...';

        try {
            const response = await fetch('/api/payment/initiate', {
                method: 'POST',
                headers: getAuthHeaders()
            });
            const order = await response.json();

            if (response.ok) {
                // In a real Razorpay setup, you would use RZP checkout here.
                // For this demo, we simulate a successful payment delay.
                payNowBtn.textContent = 'Processing Payment...';

                setTimeout(async () => {
                    // Simulate verification
                    await verifyPayment('pay_sim_' + Math.random().toString(36).substr(2, 9), order.order_id);
                }, 2000);
            } else {
                throw new Error(order.error || 'Failed to initiate payment');
            }
        } catch (e) {
            showError(`Payment failed: ${e.message}`);
            payNowBtn.disabled = false;
            payNowBtn.textContent = 'Pay Now with Razorpay';
        }
    }

    async function verifyPayment(paymentId, orderId) {
        try {
            const response = await fetch('/api/payment/verify', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify({ payment_id: paymentId, order_id: orderId })
            });
            const result = await response.json();

            if (response.ok) {
                alert('Upgrade Successful! You now have unlimited memories.');
                document.getElementById('payment-modal').classList.remove('active');
                document.body.style.overflow = '';

                // Refresh usage bar
                fetchUserUsage();
            } else {
                throw new Error(result.error || 'Verification failed');
            }
        } catch (e) {
            showError(`Verification failed: ${e.message}`);
        } finally {
            payNowBtn.disabled = false;
            payNowBtn.textContent = 'Pay Now with Razorpay';
        }
    }

    const upgradeLink = document.getElementById('upgrade-link');
    if (upgradeLink) {
        upgradeLink.addEventListener('click', (e) => {
            e.preventDefault();
            showPaymentModal();
        });
    }


    // Camera event listeners
    if (useCameraBtn) {
        useCameraBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            startCamera();
        });
    }
    if (capturePhotoBtn) {
        capturePhotoBtn.addEventListener('click', capturePhoto);
    }
    if (cancelCameraBtn) {
        cancelCameraBtn.addEventListener('click', stopCamera);
    }
    if (toggleCameraBtn) {
        toggleCameraBtn.addEventListener('click', toggleCamera);
    }

    // Mobile Menu Logic
    const hamburgerMenu = document.getElementById('hamburger-menu');
    const navMenu = document.getElementById('nav-menu');

    // Create overlay element
    const navOverlay = document.createElement('div');
    navOverlay.className = 'nav-overlay';
    document.body.appendChild(navOverlay);

    const navLogin = document.getElementById('nav-login');
    const navRegister = document.getElementById('nav-register');
    const navLogout = document.getElementById('nav-logout');

    // Check login status and update UI
    const loggedInUser = sessionStorage.getItem('user');

    if (loggedInUser) {
        console.log('User is logged in, updating nav UI');
        if (navLogin && navLogin.parentElement) navLogin.parentElement.style.display = 'none';
        if (navRegister && navRegister.parentElement) navRegister.parentElement.style.display = 'none';
        if (navLogout) {
            navLogout.style.display = 'block';
            if (navLogout.parentElement) navLogout.parentElement.style.display = 'block';
        }
    } else {
        console.log('No logged in user found');
        if (navLogin && navLogin.parentElement) navLogin.parentElement.style.display = 'block';
        if (navRegister && navRegister.parentElement) navRegister.parentElement.style.display = 'block';
        if (navLogout) {
            navLogout.style.display = 'none';
            if (navLogout.parentElement) navLogout.parentElement.style.display = 'none';
        }
    }

    // Logout functionality
    if (navLogout) {
        navLogout.addEventListener('click', (e) => {
            e.preventDefault();
            logout();
        });
    }

    function logout() {
        // Clear user session
        sessionStorage.removeItem('user');

        // Show notification if possible
        alert('Logged out successfully');

        // Redirect to register page
        window.location.href = 'register.html';
    }

    if (hamburgerMenu && navMenu) {
        hamburgerMenu.addEventListener('click', () => {
            hamburgerMenu.classList.toggle('active');
            navMenu.classList.toggle('active');
            navOverlay.classList.toggle('active');
            document.body.style.overflow = navMenu.classList.contains('active') ? 'hidden' : '';
        });

        navOverlay.addEventListener('click', () => {
            hamburgerMenu.classList.remove('active');
            navMenu.classList.remove('active');
            navOverlay.classList.remove('active');
            document.body.style.overflow = '';
        });

        // Close menu when link is clicked
        document.querySelectorAll('.nav-menu a').forEach(link => {
            link.addEventListener('click', function () {
                // Remove active class from all links
                document.querySelectorAll('.nav-menu a').forEach(l => l.classList.remove('active'));

                // Add active class to clicked link
                this.classList.add('active');

                // Delay closing the menu to show the highlight
                setTimeout(() => {
                    hamburgerMenu.classList.remove('active');
                    navMenu.classList.remove('active');
                    navOverlay.classList.remove('active');
                    document.body.style.overflow = '';
                }, 300);
            });
        });
    }

    // Initialize Speech Recognition
    function initializeSpeechRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.maxAlternatives = 1;
            recognition.lang = 'en-US';

            recognition.onstart = () => {
                console.log('Speech recognition started');
                if (startRecordingBtn) startRecordingBtn.classList.add('recording');
                if (recordingStatus) recordingStatus.textContent = 'Listening... Speak now';
            };

            let lastResultTime = Date.now();

            recognition.onresult = (event) => {
                // Get the latest result
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    transcript += event.results[i][0].transcript;
                }
                if (voiceOutput) voiceOutput.value = transcript;

                // Enable the submit button when there's text
                if (submitVoiceBtn) submitVoiceBtn.disabled = !transcript || transcript.trim() === '';

                console.log('Recognized:', transcript);

                // Reset the silence timeout
                lastResultTime = Date.now();

                if (silenceTimeout) {
                    clearTimeout(silenceTimeout);
                }

                // Set a timeout to stop recognition after 5 seconds of silence
                silenceTimeout = setTimeout(() => {
                    recognition.stop();
                    stopRecording();
                    console.log('Stopped listening after 5 seconds of silence');
                }, 5000); // 5 seconds of silence
            };

            recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                if (recordingStatus) recordingStatus.textContent = `Error: ${event.error}`;
                stopRecording();
            };

            recognition.onend = () => {
                console.log('Speech recognition ended');
                stopRecording();
            };

            // Update button state when user manually edits the text
            if (voiceOutput) {
                voiceOutput.addEventListener('input', function () {
                    if (submitVoiceBtn) submitVoiceBtn.disabled = !this.value || this.value.trim() === '';
                });
            }
        } else {
            console.warn('Speech Recognition not supported in this browser');
            if (recordingStatus) recordingStatus.textContent = 'Speech recognition not supported in your browser';
            if (startRecordingBtn) startRecordingBtn.disabled = true;
        }
    }

    // Toggle recording
    function toggleRecording() {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    }

    // Start recording
    function startRecording() {
        if (recognition) {
            recognition.start();
            isRecording = true;
            startRecordingBtn.classList.add('recording');
            recordingStatus.textContent = 'Listening... Speak now';
        }
    }

    // Stop recording
    function stopRecording() {
        if (recognition) {
            recognition.stop();
            isRecording = false;
            if (startRecordingBtn) startRecordingBtn.classList.remove('recording');
            if (recordingStatus) recordingStatus.textContent = 'Click mic to start recording';
        }

        // Clear the silence timeout if it exists
        if (silenceTimeout) {
            clearTimeout(silenceTimeout);
            silenceTimeout = null;
        }
    }

    // Clear voice output
    function clearVoiceOutput() {
        voiceOutput.value = '';
        recordingStatus.textContent = 'Click mic to start recording';
        submitVoiceBtn.disabled = true;
        saveStatus.textContent = '';
    }

    // Submit voice memory
    async function submitVoiceMemory() {
        const voiceText = voiceOutput.value.trim();

        if (!voiceText) {
            showError('Please record some voice input first.');
            return;
        }

        savingSpinner.style.display = 'block';
        saveStatus.textContent = 'Processing...';
        submitVoiceBtn.disabled = true;

        const requestData = { voice_text: voiceText, has_image: false };

        try {
            const response = await fetch('/api/process-memory', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify(requestData)
            });

            const result = await response.json();

            if (response.status === 402 && result.payment_required) {
                savingSpinner.style.display = 'none';
                saveStatus.textContent = 'Limit reached';
                submitVoiceBtn.disabled = false;
                showPaymentModal(result);
                return;
            }

            if (!response.ok) {
                throw new Error(result.error || `Server error: ${response.status}`);
            }

            savingSpinner.style.display = 'none';
            saveStatus.textContent = 'Saved & Processed!';

            if (categoryResult) categoryResult.textContent = result.category || 'Uncategorized';
            if (contextResult) contextResult.textContent = result.context || 'No context';
            if (tagsResult) tagsResult.textContent = result.tags?.join(', ') || 'No tags';

            if (result.memories_used !== undefined) {
                updateUsageBar(result.memories_used, result.memory_limit);
            }

            addToLocalMemory(result);

            setTimeout(() => { saveStatus.textContent = 'Saved to MemoAI'; }, 2000);
            submitVoiceBtn.disabled = false;

        } catch (error) {
            console.error('Error processing memory:', error);
            savingSpinner.style.display = 'none';
            saveStatus.textContent = 'Error saving';
            showError('Failed to save memory. Please try again.');
            submitVoiceBtn.disabled = false;
        }
    }

    // Handle image upload
    function handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded image">`;
                imagePreview.style.display = 'block';
                uploadArea.style.display = 'none';
                imageDescriptionSection.style.display = 'block';
                imageDescription.focus();
            };
            reader.readAsDataURL(file);
        }
    }

    // Clear image
    function clearImage() {
        imagePreview.innerHTML = '';
        imagePreview.style.display = 'none';
        uploadArea.style.display = 'block';
        imageUpload.value = '';
        imageDescriptionSection.style.display = 'none';
        imageDescription.value = '';
        imageSaveStatus.textContent = '';
    }

    // Camera Functions
    async function startCamera() {
        try {
            // Request camera access with current facing mode
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: currentFacingMode },
                audio: false
            });

            // Set video source to camera stream
            cameraVideo.srcObject = cameraStream;

            // Show camera container, hide upload area
            uploadArea.style.display = 'none';
            cameraContainer.style.display = 'block';

        } catch (error) {
            console.error('Error accessing camera:', error);
            showError('Unable to access camera. Please check permissions.');
        }
    }

    function capturePhoto() {
        if (!cameraStream) {
            showError('Camera is not active');
            return;
        }

        // Set canvas dimensions to match video
        cameraCanvas.width = cameraVideo.videoWidth;
        cameraCanvas.height = cameraVideo.videoHeight;

        // Draw current video frame to canvas
        const context = cameraCanvas.getContext('2d');
        context.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

        // Convert canvas to data URL (base64)
        const imageDataUrl = cameraCanvas.toDataURL('image/png');

        // Stop camera stream
        stopCamera();

        // Display captured image in preview
        imagePreview.innerHTML = `<img src="${imageDataUrl}" alt="Captured photo">`;
        imagePreview.style.display = 'block';

        // Show description section (crucial step as per user requirement)
        imageDescriptionSection.style.display = 'block';
        imageDescription.focus();
    }

    async function toggleCamera() {
        // Toggle between front and back camera
        currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';

        // Stop current stream
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
        }

        // Restart camera with new facing mode
        try {
            cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: currentFacingMode },
                audio: false
            });

            cameraVideo.srcObject = cameraStream;
        } catch (error) {
            console.error('Error switching camera:', error);
            showError('Unable to switch camera. Please try again.');
            // Revert to previous mode if switch fails
            currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
        }
    }

    function stopCamera() {
        // Stop all video tracks
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }

        // Reset video source
        if (cameraVideo) {
            cameraVideo.srcObject = null;
        }

        // Hide camera container, show upload area
        if (cameraContainer) cameraContainer.style.display = 'none';
        if (uploadArea) uploadArea.style.display = 'block';
    }

    // Process memory (connect to backend API)
    async function processMemory() {
        const voiceText = voiceOutput.value.trim();
        const hasImage = imagePreview.querySelector('img') !== null;

        if (!voiceText && !hasImage) {
            showError('Please provide either voice input or an image to process.');
            return;
        }

        // Show processing state
        categoryResult.textContent = 'Processing...';
        contextResult.textContent = 'Analyzing content...';
        tagsResult.textContent = 'Generating tags...';

        try {
            // Prepare data for backend
            const requestData = {
                voice_text: voiceText,
                has_image: hasImage
            };

            // If image exists, convert to base64
            if (hasImage) {
                const imageData = imagePreview.querySelector('img').src;
                requestData.image_data = imageData;
            }

            // Call backend API with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

            const response = await fetch('/api/process-memory', {
                method: 'POST',
                headers: getAuthHeaders(),
                body: JSON.stringify(requestData),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                if (response.status === 402 && errData.payment_required) {
                    showPaymentModal(errData);
                    // Reset UI
                    categoryResult.textContent = '-';
                    contextResult.textContent = '-';
                    tagsResult.textContent = '-';
                    return;
                }
                throw new Error(`Server error: ${response.status}`);
            }

            const result = await response.json();

            // Update UI with results
            categoryResult.textContent = result.category || 'Uncategorized';
            contextResult.textContent = result.context || 'No context generated';
            tagsResult.textContent = result.tags?.join(', ') || 'No tags generated';

            // Update usage bar
            if (result.memories_used !== undefined) {
                updateUsageBar(result.memories_used, result.memory_limit);
            }

            // Store in local memory for search
            addToLocalMemory(result);

        } catch (error) {
            console.error('Processing error:', error);

            // Check if it's a timeout error
            if (error.name === 'AbortError') {
                showError('Request timed out. Please try again later.');
            } else {
                showError(`Failed to process memory: ${error.message}`);
            }

            // Fallback to simulation
            simulateProcessing(voiceText, hasImage);
        }
    }

    // Fallback simulation function
    function simulateProcessing(voiceText, hasImage) {
        setTimeout(() => {
            const simulatedCategory = simulateCategoryClassification(voiceText);
            categoryResult.textContent = simulatedCategory;

            const simulatedContext = simulateContextGeneration(voiceText, hasImage);
            contextResult.textContent = simulatedContext;

            const simulatedTags = simulateTagGeneration(voiceText);
            tagsResult.textContent = simulatedTags.join(', ');
        }, 1500);
    }

    // Add to local memory storage
    function addToLocalMemory(memoryData) {
        let localMemories = JSON.parse(localStorage.getItem('memoai_memories') || '[]');

        // If this is a new memory, generate a unique ID
        if (!memoryData.id) {
            memoryData.id = Date.now();
        }

        // Store the image data if available
        if (memoryData.image_data) {
            memoryData.stored_image_data = memoryData.image_data;
        }

        memoryData.timestamp = new Date().toISOString();

        // Check if memory with this ID already exists
        const existingIndex = localMemories.findIndex(m => m.id === memoryData.id);
        if (existingIndex >= 0) {
            // Update existing memory
            localMemories[existingIndex] = memoryData;
        } else {
            // Add new memory
            localMemories.push(memoryData);
        }

        localStorage.setItem('memoai_memories', JSON.stringify(localMemories));
    }

    // Error display function
    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #fee2e2;
            color: #991b1b;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #fecaca;
            z-index: 9999;
            max-width: 300px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        `;

        document.body.appendChild(errorDiv);

        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }

    async function getMemorySummary() {
        console.log('AI Summary button clicked');
        if (!summaryBtn) {
            console.error('Summary button not found in DOM');
            return;
        }

        summaryBtn.disabled = true;
        summaryBtn.innerHTML = '<span class="spinner-small"></span> Analyzing...';

        try {
            const response = await fetch('/api/memories/summary', {
                headers: getAuthHeaders()
            });

            if (!response.ok) throw new Error('Failed to get summary');

            const data = await response.json();

            if (summaryText) summaryText.textContent = data.summary;
            if (summaryResult) {
                summaryResult.style.display = 'block';
                summaryResult.style.animation = 'slideUp 0.5s ease-out';
                // Scroll result into view
                summaryResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

        } catch (error) {
            console.error('Summary error:', error);
            showError('Could not generate summary. Ensure you have saved memories.');
        } finally {
            summaryBtn.disabled = false;
            summaryBtn.textContent = '✨ AI Summary';
        }
    }

    // Search memories functionality
    async function searchMemories() {
        const query = searchQuery.value.trim().toLowerCase();

        if (!query) {
            showError('Please enter a search term');
            return;
        }

        // Show loading state
        searchResults.innerHTML = '<p class="no-results">Searching memories...</p>';

        try {
            // Call backend API with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout

            const response = await fetch(`/api/search-memories?q=${encodeURIComponent(query)}`, {
                method: 'GET',
                headers: getAuthHeaders(),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();

            // Display results
            currentSearchResults = data.results || [];
            displaySearchResults(currentSearchResults);

        } catch (error) {
            console.error('Search error:', error);

            // Check if it's a timeout error
            if (error.name === 'AbortError') {
                showError('Search request timed out. Using local search instead.');

                // Fallback to local search
                fallbackLocalSearch(query);
            } else {
                showError(`Search failed: ${error.message}`);

                // Fallback to local search
                fallbackLocalSearch(query);
            }
        }
    }

    // Fallback to local search
    function fallbackLocalSearch(query) {
        try {
            // Get memories from local storage
            let localMemories = JSON.parse(localStorage.getItem('memoai_memories') || '[]');

            // If no local memories, use sample data
            if (localMemories.length === 0) {
                localMemories = getSampleMemories();
            }

            // Filter memories based on query
            const results = localMemories.filter(memory =>
                (memory.title && memory.title.toLowerCase().includes(query)) ||
                (memory.content && memory.content.toLowerCase().includes(query)) ||
                (memory.category && memory.category.toLowerCase().includes(query)) ||
                (memory.tags && memory.tags.some(tag => tag.toLowerCase().includes(query))) ||
                (memory.voice_text && memory.voice_text.toLowerCase().includes(query))
            );

            // Display results
            currentSearchResults = results;
            displaySearchResults(results);

        } catch (error) {
            console.error('Local search error:', error);
            searchResults.innerHTML = '<p class="no-results">Search error occurred. Please try again.</p>';
        }
    }

    // Get sample memories for demonstration
    function getSampleMemories() {
        return [
            {
                id: 1,
                title: "Morning Meeting Notes",
                content: "Discussed quarterly project updates and timeline adjustments",
                category: "Work & Meetings",
                date: "Today",
                tags: ["meeting", "project", "timeline"]
            },
            {
                id: 2,
                title: "Shopping List",
                content: "Need to buy groceries including milk, bread, eggs, and vegetables",
                category: "Money & Shopping",
                date: "Yesterday",
                tags: ["shopping", "groceries", "food"]
            },
            {
                id: 3,
                title: "Learning Goals",
                content: "Plan to learn new programming concepts this week",
                category: "Learning & Growth",
                date: "Last Week",
                tags: ["learning", "goals", "programming"]
            },
            {
                id: 4,
                title: "Health Checkup",
                content: "Doctor appointment scheduled for next Monday",
                category: "Health & Fitness",
                date: "This Month",
                tags: ["health", "appointment", "doctor"]
            }
        ];
    }

    // Display search results
    function displaySearchResults(results) {
        if (results.length > 0) {
            let resultsHTML = '';
            results.forEach(memory => {
                let similarityInfo = '';
                if (memory.similarity_score !== undefined) {
                    const similarityPercent = Math.round(memory.similarity_score * 100);
                    similarityInfo = '<div class="similarity-score">Similarity: ' + similarityPercent + '%</div>';
                }

                // Check if this memory has an image
                let imageHTML = '';
                if (memory.stored_image_data) {
                    imageHTML = '<img src="' + memory.stored_image_data + '" alt="Memory image" class="memory-image-preview">';
                } else if (memory.image_path) {
                    const imagePath = memory.image_path.replace(/\\/g, '/').replace(/^.*\/uploads\//, 'uploads/');
                    imageHTML = `<img src="/${imagePath}" alt="Memory image" class="memory-image-preview" onerror="this.style.display='none'">`;
                }

                // Combine content, voice text and image description
                let contentText = '';
                if (memory.image_description) {
                    contentText += '🔍 Image Analysis: ' + memory.image_description + ' ';
                }
                if (memory.voice_text) {
                    contentText += memory.voice_text + ' ';
                } else if (memory.content && !memory.image_description) {
                    contentText += memory.content;
                }

                if (!contentText.trim()) contentText = 'No content';

                // Fixed: was missing closing > on onclick div
                resultsHTML += '<div class="memory-item" onclick="showMemoryDetails(' + memory.id + ')">' +
                    '<div class="memory-title">' + (memory.title || 'Untitled Memory') + '</div>' +
                    '<div class="memory-content-with-image">' +
                    imageHTML +
                    '<div class="memory-text-content">' +
                    contentText.substring(0, 200) +
                    (contentText.length > 200 ? '...' : '') +
                    '</div>' +
                    '</div>' +
                    '<div class="memory-meta">' +
                    '<span>Category: ' + (memory.category || 'Uncategorized') + '</span>' +
                    '<span>Date: ' + (memory.date || memory.created_at || new Date(memory.timestamp || Date.now()).toLocaleDateString()) + '</span>' +
                    (memory.image_path ? '<span>📷 Has Image</span>' : '') +
                    '</div>' +
                    '<div class="memory-tags">' +
                    (memory.tags || []).map(tag => '<span class="tag">' + tag + '</span>').join('') +
                    '</div>' +
                    similarityInfo +
                    '</div>';
            });
            searchResults.innerHTML = resultsHTML;
        } else {
            searchResults.innerHTML = '<p class="no-results">No memories found. Try different keywords.</p>';
        }
    }

    // Function to show memory details
    function showMemoryDetails(memoryId) {
        // First check current search results (most likely source)
        let memory = currentSearchResults.find(m => m.id === memoryId);

        if (!memory) {
            // Find the memory in local storage or sample data
            let localMemories = JSON.parse(localStorage.getItem('memoai_memories') || '[]');
            memory = localMemories.find(m => m.id === memoryId);
        }

        if (!memory) {
            // Check sample data
            const sampleMemories = getSampleMemories();
            memory = sampleMemories.find(m => m.id === memoryId);
        }

        if (memory) {
            showDetailedMemoryView(memory);
        } else {
            alert('Memory not found: ' + memoryId);
        }
    }

    // Function to show detailed memory view
    function showDetailedMemoryView(memory) {
        let imageHTML = '';
        if (memory.stored_image_data) {
            // Display the actual image from stored data with larger preview
            imageHTML = '<img src="' + memory.stored_image_data + '" alt="Memory image" class="memory-image-preview-large">';
        } else if (memory.image_path) {
            // Convert backslashes to forward slashes for web compatibility
            const imagePath = memory.image_path.replace(/\\/g, '/');
            // Try to fetch the image from the server
            imageHTML = `<img src="/${imagePath}" alt="Memory image" class="memory-image-preview-large" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\'http://www.w3.org/2000/svg\' width=\'300\' height=\'200\'%3E%3Crect width=\'300\' height=\'200\' fill=\'%23e2e8f0\'/%3E%3Ctext x=\'50%\' y=\'50%\' text-anchor=\'middle\' dominant-baseline=\'middle\' fill=\'%2394a3b8\' font-size=\'14\'%3E🖼️ Image Not Found%3C/text%3E%3C/svg%3E'; this.title='Image not available: ${imagePath}'; this.onerror=null;">`;
        }

        let contentText = memory.content || memory.voice_text || 'No content';

        let analysisHTML = '';
        if (memory.image_description) {
            analysisHTML = `
                <div style="margin: 1.5rem 0; background: #eff6ff; padding: 1.2rem; border-radius: 10px; border-left: 4px solid #3b82f6;">
                    <strong style="color: #1e40af; font-size: 1.05rem;">🔍 AI Image Analysis:</strong><br>
                    <p style="margin-top: 0.5rem; font-size: 1rem; line-height: 1.6; color: #1e3a8a;">${memory.image_description}</p>
                </div>
            `;
        }

        const detailHTML = `
            <div style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 10000; display: flex; align-items: center; justify-content: center;">
                <div style="background: white; border-radius: 15px; padding: 2.5rem; max-width: 800px; width: 90%; max-height: 90vh; overflow-y: auto; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);">
                    <h3 style="margin-top: 0; color: #1e293b; font-size: 1.5rem; border-bottom: 2px solid #4f46e5; padding-bottom: 0.5rem;">${memory.title || 'Untitled Memory'}</h3>
                    ${imageHTML}
                    ${analysisHTML}
                    <div style="margin: 1.5rem 0; background: #f8fafc; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4f46e5;">
                        <strong style="color: #1e293b; font-size: 1.1rem;">Voice Content:</strong><br>
                        <p style="white-space: pre-wrap; margin-top: 0.5rem; font-size: 1.05rem; line-height: 1.7; color: #374151;">${memory.voice_text || memory.content || 'No voice text available'}</p>
                    </div>
                    <div style="margin: 1.5rem 0; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; background: #f1f5f9; padding: 1.5rem; border-radius: 10px;">
                        <div><strong style="color: #1e293b;">Category:</strong><br><span style="color: #4f46e5; font-weight: 500;">${memory.category || 'Uncategorized'}</span></div>
                        <div><strong style="color: #1e293b;">Tags:</strong><br><span style="color: #64748b;">${(memory.tags || []).join(', ')}</span></div>
                        <div><strong style="color: #1e293b;">Date:</strong><br><span style="color: #64748b;">${memory.date || new Date(memory.timestamp || Date.now()).toLocaleDateString()}</span></div>
                        ${memory.similarity_score ? '<div><strong style="color: #1e293b;">Similarity:</strong><br><span style="color: #059669; font-weight: 500;">' + Math.round(memory.similarity_score * 100) + '%</span></div>' : ''}
                    </div>
                    <div style="text-align: center; margin-top: 1.5rem;">
                        <button onclick="this.closest('div[style*="position: fixed"]').remove()" style="background: #4f46e5; color: white; border: none; padding: 0.75rem 2rem; border-radius: 8px; cursor: pointer; font-size: 1rem; font-weight: 500; transition: all 0.2s ease; box-shadow: 0 2px 8px rgba(79, 70, 229, 0.3);">Close</button>
                    </div>
                </div>
            </div>
        `;

        document.body.insertAdjacentHTML('beforeend', detailHTML);
    }

    // Simulate category classification based on voice text
    function simulateCategoryClassification(text) {
        if (!text) return 'General';

        const lowerText = text.toLowerCase();
        const categories = {
            'Daily Life': ['home', 'family', 'personal', 'daily', 'today', 'morning', 'evening'],
            'Work & Meetings': ['work', 'meeting', 'office', 'colleagues', 'project', 'presentation'],
            'Learning & Growth': ['learn', 'study', 'education', 'growth', 'improve', 'knowledge'],
            'Health & Fitness': ['health', 'exercise', 'fitness', 'diet', 'wellness', 'medical'],
            'Money & Shopping': ['money', 'buy', 'purchase', 'shop', 'price', 'budget', 'finance'],
            'Entertainment & Leisure': ['movie', 'music', 'game', 'fun', 'relax', 'entertainment'],
            'Ideas & Creativity': ['idea', 'creative', 'innovation', 'design', 'think', 'brainstorm']
        };

        for (const [category, keywords] of Object.entries(categories)) {
            if (keywords.some(keyword => lowerText.includes(keyword))) {
                return category;
            }
        }

        return 'General'; // Default category
    }

    // Simulate context generation
    function simulateContextGeneration(voiceText, hasImage) {
        if (!voiceText && !hasImage) {
            return 'No content to analyze.';
        }

        let context = '';

        if (voiceText) {
            context += `Voice content: "${voiceText}". `;
        }

        if (hasImage) {
            context += 'Associated with an uploaded image. ';
        }

        if (voiceText && hasImage) {
            context += 'Combining auditory and visual information for enhanced context understanding.';
        } else if (voiceText) {
            context += 'Audio-based memory with textual context.';
        } else if (hasImage) {
            context += 'Visual memory with potential for detailed analysis.';
        }

        return context;
    }

    // Simulate tag generation
    function simulateTagGeneration(text) {
        if (!text) return ['general'];

        const lowerText = text.toLowerCase();
        const tags = [];

        // Extract potential tags based on common patterns
        if (lowerText.includes('meeting')) tags.push('meeting');
        if (lowerText.includes('project')) tags.push('project');
        if (lowerText.includes('idea')) tags.push('idea');
        if (lowerText.includes('health')) tags.push('health');
        if (lowerText.includes('exercise')) tags.push('fitness');
        if (lowerText.includes('shopping')) tags.push('shopping');
        if (lowerText.includes('learning')) tags.push('learning');
        if (lowerText.includes('work')) tags.push('work');
        if (lowerText.includes('family')) tags.push('family');
        if (lowerText.includes('friend')) tags.push('social');
        if (lowerText.includes('travel')) tags.push('travel');
        if (lowerText.includes('food')) tags.push('food');

        // If no specific tags found, use general
        if (tags.length === 0) {
            tags.push('general');
        }

        // Add time-related tags if present
        if (lowerText.includes('today') || lowerText.includes('now')) tags.push('today');
        if (lowerText.includes('tomorrow')) tags.push('tomorrow');
        if (lowerText.includes('yesterday')) tags.push('past');

        return [...new Set(tags)]; // Remove duplicates
    }

    // Smooth scrolling for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Intersection Observer for scroll animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe elements for animation
    document.querySelectorAll('.feature-card, .demo-panel, .testimonial-card, .step').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });

    // Cleanup interval on page unload
    window.addEventListener('beforeunload', () => {
        if (silenceTimeout) {
            clearTimeout(silenceTimeout);
        }
    });

    // Submit image memory
    function submitImageMemory() {
        const imageElement = imagePreview.querySelector('img');
        if (!imageElement) {
            showError('Please upload an image first.');
            return;
        }

        const description = imageDescription.value.trim();

        // Show saving spinner and status
        imageSavingSpinner.style.display = 'block';
        imageSaveStatus.textContent = 'Processing...';
        submitImageBtn.disabled = true;

        // Prepare data for backend
        const requestData = {
            voice_text: description || 'Image uploaded without description',
            has_image: true,
            image_data: imageElement.src
        };

        // Call backend API to process and save — include auth headers!
        fetch('/api/process-memory', {
            method: 'POST',
            headers: getAuthHeaders(),   // ← Fixed: was missing X-User-Id, causing 401
            body: JSON.stringify(requestData)
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(err => { throw new Error(err.error || `Server error: ${response.status}`); });
                }
                return response.json();
            })
            .then(result => {
                // Hide spinner and show success
                imageSavingSpinner.style.display = 'none';
                imageSaveStatus.textContent = 'Saved & Processed!';

                // Update processing results section
                if (categoryResult) categoryResult.textContent = result.category || 'Uncategorized';
                if (contextResult) contextResult.textContent = result.context || 'No context generated';
                if (tagsResult) tagsResult.textContent = result.tags?.join(', ') || 'No tags generated';

                // Store in local memory for search
                addToLocalMemory(result);

                // Reset status after a moment
                setTimeout(() => {
                    imageSaveStatus.textContent = 'Saved to MemoAI';
                }, 2000);

                // Re-enable submit button
                submitImageBtn.disabled = false;
            })
            .catch(error => {
                console.error('Error processing image memory:', error);
                imageSavingSpinner.style.display = 'none';
                imageSaveStatus.textContent = 'Error saving';
                showError('Failed to save image memory: ' + error.message);
                submitImageBtn.disabled = false;
            });
    }

    // Add floating effect to feature cards on hover
    document.querySelectorAll('.feature-card').forEach(card => {
        card.addEventListener('mouseenter', function () {
            this.style.transform = 'translateY(-10px) scale(1.02)';
        });

        card.addEventListener('mouseleave', function () {
            this.style.transform = 'translateY(0)';
        });
    });

    // Initialize with some sample text for demonstration
    setTimeout(() => {
        if (!voiceOutput.value) {
            voiceOutput.value = "This is a sample voice input for MemoAI demonstration. The system can transcribe speech and categorize it into appropriate memory categories.";
        }
        // Enable submit button if there's text
        submitVoiceBtn.disabled = !voiceOutput.value.trim();
    }, 2000);
});

// Utility function to scroll to section
function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
}

// ====================
// INTERACTIVE ENHANCEMENTS
// ====================

// Custom cursor effect
const cursorFollower = document.querySelector('.cursor-follower');
const cursorDot = document.querySelector('.cursor-dot');

if (cursorFollower && cursorDot) {
    let mouseX = 0;
    let mouseY = 0;
    let followerX = 0;
    let followerY = 0;

    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });

    function animateCursor() {
        // Smooth follow for follower
        followerX += (mouseX - followerX) * 0.1;
        followerY += (mouseY - followerY) * 0.1;

        cursorFollower.style.left = followerX - 10 + 'px';
        cursorFollower.style.top = followerY - 10 + 'px';

        // Immediate follow for dot
        cursorDot.style.left = mouseX - 3 + 'px';
        cursorDot.style.top = mouseY - 3 + 'px';

        requestAnimationFrame(animateCursor);
    }

    animateCursor();

    // Interactive elements effect
    const interactiveElements = document.querySelectorAll('.interactive-element, .cta-btn, .feature-card, .record-btn');

    interactiveElements.forEach(element => {
        element.addEventListener('mouseenter', () => {
            cursorFollower.style.transform = 'scale(2)';
            cursorFollower.style.background = 'rgba(247, 37, 133, 0.5)';
        });

        element.addEventListener('mouseleave', () => {
            cursorFollower.style.transform = 'scale(1)';
            cursorFollower.style.background = 'rgba(247, 37, 133, 0.3)';
        });
    });
}

// Scroll-triggered animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('animated');
        }
    });
}, observerOptions);

// Observe animated elements
document.querySelectorAll('[data-animate]').forEach(el => {
    observer.observe(el);
});

// Enhanced parallax effect for hero section
let hero = document.querySelector('.hero');
if (hero) {
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const rate = scrolled * -0.5;

        if (scrolled < window.innerHeight) {
            hero.style.transform = `translateY(${rate}px)`;
        }
    });
}

// Floating particles effect
function createFloatingParticle() {
    const particle = document.createElement('div');
    particle.className = 'floating-particle';
    particle.style.left = Math.random() * 100 + '%';
    particle.style.animationDuration = (Math.random() * 10 + 5) + 's';
    particle.style.opacity = Math.random() * 0.5 + 0.1;

    // Random particle shapes
    const shapes = ['●', '✦', '✧', '✦'];
    particle.textContent = shapes[Math.floor(Math.random() * shapes.length)];

    document.querySelector('.hero').appendChild(particle);

    // Remove particle after animation completes
    setTimeout(() => {
        if (particle.parentNode) {
            particle.parentNode.removeChild(particle);
        }
    }, 15000);
}

// Create initial particles
for (let i = 0; i < 15; i++) {
    setTimeout(createFloatingParticle, i * 300);
}

// Continuous particle generation
setInterval(createFloatingParticle, 2000);

// Enhanced button ripple effect
function createRipple(event) {
    const button = event.currentTarget;
    const circle = document.createElement('span');
    const diameter = Math.max(button.clientWidth, button.clientHeight);
    const radius = diameter / 2;

    circle.style.width = circle.style.height = `${diameter}px`;
    circle.style.left = `${event.clientX - button.getBoundingClientRect().left - radius}px`;
    circle.style.top = `${event.clientY - button.getBoundingClientRect().top - radius}px`;
    circle.classList.add('ripple');

    const ripple = button.getElementsByClassName('ripple')[0];
    if (ripple) {
        ripple.remove();
    }

    button.appendChild(circle);
}

// Add ripple effect to buttons
document.querySelectorAll('.cta-btn, .primary-btn, .record-btn').forEach(button => {
    button.addEventListener('click', createRipple);
});

// Typewriter effect for hero headline
function typeWriterEffect(element, text, speed = 50) {
    let i = 0;
    element.textContent = '';

    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }

    type();
}

// Apply typewriter effect to hero headline
window.addEventListener('load', () => {
    const heroTitle = document.querySelector('.hero-content h1');
    if (heroTitle) {
        const originalText = heroTitle.textContent;
        setTimeout(() => {
            typeWriterEffect(heroTitle, originalText, 30);
        }, 500);
    }
});

// Enhanced testimonial carousel
function initTestimonialCarousel() {
    const testimonials = document.querySelectorAll('.testimonial-card');
    let currentIndex = 0;

    function showTestimonial(index) {
        testimonials.forEach((testimonial, i) => {
            testimonial.style.opacity = i === index ? '1' : '0.3';
            testimonial.style.transform = i === index ? 'scale(1)' : 'scale(0.95)';
        });
    }

    // Auto-rotate testimonials
    setInterval(() => {
        currentIndex = (currentIndex + 1) % testimonials.length;
        showTestimonial(currentIndex);
    }, 4000);

    // Initial display
    showTestimonial(0);
}

// Initialize carousel when testimonials are visible
const testimonialSection = document.querySelector('.testimonials');
if (testimonialSection) {
    const testimonialObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                initTestimonialCarousel();
                testimonialObserver.disconnect();
            }
        });
    });

    testimonialObserver.observe(testimonialSection);
}

// Add CSS for new interactive elements
const style = document.createElement('style');
style.textContent = `
    .floating-particle {
        position: absolute;
        font-size: 20px;
        color: rgba(255, 255, 255, 0.7);
        pointer-events: none;
        animation: floatUp linear forwards;
        z-index: 1;
    }
    
    @keyframes floatUp {
        to {
            transform: translateY(-100vh) rotate(360deg);
            opacity: 0;
        }
    }
    
    .ripple {
        position: absolute;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.7);
        transform: scale(0);
        animation: ripple 0.6s linear;
        pointer-events: none;
    }
    
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    /* Enhanced testimonial styling */
    .testimonial-card {
        transition: all 0.5s cubic-bezier(0.23, 1, 0.32, 1);
    }
    
    /* Smooth scrolling for all elements */
    html {
        scroll-behavior: smooth;
    }
    
    .summary-result {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #7c3aed;
        animation: slideUp 0.5s ease-out;
    }
    
    .summary-result h4 {
        color: #7c3aed;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .summary-result h4::before {
        content: '✨';
    }
    
    .summary-result p {
        line-height: 1.6;
        color: #4b5563;
        font-style: italic;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
`;
document.head.appendChild(style);
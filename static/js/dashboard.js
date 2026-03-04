/* =========================================
   MemoAI — Dashboard Logic
   ========================================= */

(function () {
    const user = getCurrentUser();
    if (!user) return;

    // UI setup
    document.getElementById('user-avatar').textContent = user.name.charAt(0);
    document.getElementById('user-name-display').textContent = user.name;
    document.getElementById('user-email-display').textContent = user.email;
    document.getElementById('welcome-text').textContent = `Welcome back, ${user.name.split(' ')[0]}! 👋`;

    const categories = ['All', 'Daily Life', 'Work & Meetings', 'Learning & Growth', 'Health & Fitness', 'Money & Shopping', 'Entertainment', 'Ideas & Creativity', 'General'];
    let selectedCategory = 'all';
    let memories = [];

    // ---------- CATEGORY FILTERS ----------
    const filtersEl = document.getElementById('category-filters');
    categories.forEach(cat => {
        const btn = document.createElement('button');
        btn.className = 'cat-btn' + (cat === 'All' ? ' active' : '');
        btn.textContent = cat;
        btn.addEventListener('click', () => {
            selectedCategory = cat.toLowerCase();
            filtersEl.querySelectorAll('.cat-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            renderMemories();
        });
        filtersEl.appendChild(btn);
    });

    // ---------- SEARCH ----------
    document.getElementById('dash-search').addEventListener('input', () => renderMemories());

    // ---------- LOGOUT ----------
    document.getElementById('logout-btn').addEventListener('click', () => {
        sessionStorage.removeItem('user');
        window.location.href = '/';
    });

    // ---------- PREMIUM MODAL ----------
    const premiumModal = document.getElementById('premium-modal');
    function openPremium() { premiumModal.classList.add('active'); }
    function closePremium() { premiumModal.classList.remove('active'); }
    document.getElementById('upgrade-btn').addEventListener('click', openPremium);
    document.getElementById('usage-upgrade-btn').addEventListener('click', openPremium);
    document.getElementById('close-premium-modal').addEventListener('click', closePremium);
    document.getElementById('cancel-premium-btn').addEventListener('click', closePremium);
    document.getElementById('upgrade-now-btn').addEventListener('click', async () => {
        try {
            const res = await fetch('/api/payment/initiate', {
                method: 'POST', headers: { 'Content-Type': 'application/json', 'X-User-Id': user.id }
            });
            const data = await res.json();
            if (data.is_premium) { showToast('Already premium!', 'info'); closePremium(); return; }
            // Simulate payment
            const verifyRes = await fetch('/api/payment/verify', {
                method: 'POST', headers: { 'Content-Type': 'application/json', 'X-User-Id': user.id },
                body: JSON.stringify({ payment_id: 'pay_' + Date.now(), order_id: data.order_id })
            });
            const verifyData = await verifyRes.json();
            if (verifyRes.ok) { showToast(verifyData.message, 'success'); closePremium(); loadUsage(); }
            else showToast(verifyData.error, 'error');
        } catch (e) { showToast('Payment error', 'error'); }
    });

    // ---------- TABS ----------
    document.querySelectorAll('.capture-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.capture-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
        });
    });

    // ---------- VOICE RECORDER ----------
    let isRecording = false, recognition = null, voiceTimer = null, voiceDuration = 0;
    let finalTranscript = '', interimTranscript = '';
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
        recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
            interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const text = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += text + ' ';
                } else {
                    interimTranscript = text;
                }
            }
            // Update live preview during recording
            const liveEl = document.getElementById('live-transcript-text');
            const combined = (finalTranscript + interimTranscript).trim();
            if (combined) {
                liveEl.textContent = combined;
                liveEl.style.color = 'var(--gray-700)';
                liveEl.style.fontStyle = 'normal';
            } else {
                liveEl.textContent = 'Listening...';
                liveEl.style.color = 'var(--gray-400)';
                liveEl.style.fontStyle = 'italic';
            }
        };

        recognition.onerror = (e) => {
            console.log('Speech recognition error:', e.error);
            if (e.error === 'no-speech') {
                // Don't stop on no-speech, just show a hint
                document.getElementById('voice-status').textContent = 'No speech detected — try speaking louder';
            } else if (e.error !== 'aborted') {
                stopVoice();
            }
        };

        recognition.onend = () => {
            // Auto-restart if still recording (in case browser auto-stops)
            if (isRecording) {
                try { recognition.start(); } catch (e) { }
            }
        };
    }

    function startVoice() {
        if (!recognition) { showToast('Speech recognition not supported in this browser. Try Chrome.', 'error'); return; }
        isRecording = true;
        finalTranscript = '';
        interimTranscript = '';
        voiceDuration = 0;

        // Show recording UI
        document.getElementById('mic-btn').className = 'mic-btn recording';
        document.getElementById('mic-icon').innerHTML = '<rect x="6" y="6" width="12" height="12" rx="2" fill="white"/>';
        document.getElementById('mic-ping').style.display = 'block';
        document.getElementById('voice-duration').style.display = 'block';
        document.getElementById('voice-duration').textContent = '00:00';
        document.getElementById('voice-status').textContent = 'Recording... Click the square to stop';

        // Show live transcript preview
        const liveEl = document.getElementById('live-transcript');
        liveEl.style.display = 'block';
        document.getElementById('live-transcript-text').textContent = 'Listening...';
        document.getElementById('live-transcript-text').style.color = 'var(--gray-400)';
        document.getElementById('live-transcript-text').style.fontStyle = 'italic';

        // Hide review area
        document.getElementById('transcript-area').style.display = 'none';

        try { recognition.start(); } catch (e) { }
        voiceTimer = setInterval(() => {
            voiceDuration++;
            const m = String(Math.floor(voiceDuration / 60)).padStart(2, '0');
            const s = String(voiceDuration % 60).padStart(2, '0');
            document.getElementById('voice-duration').textContent = m + ':' + s;
        }, 1000);
    }

    function stopVoice() {
        isRecording = false;
        if (recognition) try { recognition.stop(); } catch (e) { }
        if (voiceTimer) { clearInterval(voiceTimer); voiceTimer = null; }

        // Reset mic button
        document.getElementById('mic-btn').className = 'mic-btn idle';
        document.getElementById('mic-icon').innerHTML = '<path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="22"/>';
        document.getElementById('mic-ping').style.display = 'none';
        document.getElementById('voice-duration').style.display = 'none';
        document.getElementById('live-transcript').style.display = 'none';

        // Include any remaining interim text
        const fullText = (finalTranscript + interimTranscript).trim();

        if (fullText) {
            // Show review area with editable transcript
            document.getElementById('voice-controls').style.display = 'none';
            document.getElementById('transcript-area').style.display = 'block';
            document.getElementById('transcript-edit').value = fullText;
            document.getElementById('voice-status').textContent = 'Review your transcript below';
        } else {
            document.getElementById('voice-status').textContent = 'No speech detected. Click the microphone to try again.';
            showToast('No speech was detected. Please try again and speak clearly.', 'info');
        }
    }

    document.getElementById('mic-btn').addEventListener('click', () => isRecording ? stopVoice() : startVoice());

    // Re-record button
    document.getElementById('rerecord-btn').addEventListener('click', () => {
        document.getElementById('transcript-area').style.display = 'none';
        document.getElementById('voice-controls').style.display = 'flex';
        document.getElementById('voice-status').textContent = 'Click the microphone to start recording';
        startVoice();
    });

    // Discard button
    document.getElementById('discard-btn').addEventListener('click', () => {
        finalTranscript = '';
        interimTranscript = '';
        document.getElementById('transcript-area').style.display = 'none';
        document.getElementById('voice-controls').style.display = 'flex';
        document.getElementById('voice-status').textContent = 'Click the microphone to start recording';
        showToast('Recording discarded', 'info');
    });

    // Confirm & Save button
    document.getElementById('save-voice-btn').addEventListener('click', async () => {
        const editedText = document.getElementById('transcript-edit').value.trim();
        if (!editedText) { showToast('Please add some text before saving', 'error'); return; }

        const btn = document.getElementById('save-voice-btn');
        btn.disabled = true;
        btn.innerHTML = '<span style="display:inline-block;width:18px;height:18px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .6s linear infinite"></span> Saving...';

        try {
            const res = await fetch('/api/process-memory', {
                method: 'POST', headers: { 'Content-Type': 'application/json', 'X-User-Id': user.id },
                body: JSON.stringify({ voice_text: editedText, has_image: false })
            });
            const data = await res.json();
            if (res.ok) {
                // Show success state
                showToast('✅ Memory saved successfully! Your thought has been captured.', 'success');
                finalTranscript = '';
                interimTranscript = '';
                document.getElementById('transcript-area').style.display = 'none';
                document.getElementById('voice-controls').style.display = 'flex';
                document.getElementById('voice-status').textContent = 'Click the microphone to start recording';
                loadMemories();
                loadUsage();
            } else if (res.status === 402) {
                showToast(data.message, 'error'); openPremium();
            } else showToast(data.error || 'Failed to save memory', 'error');
        } catch (e) { showToast('Connection error. Please try again.', 'error'); }
        finally {
            btn.disabled = false;
            btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Confirm & Save';
        }
    });

    // ---------- IMAGE UPLOADER ----------
    let imageData = null;
    document.getElementById('upload-file-zone').addEventListener('click', () => document.getElementById('image-file-input').click());
    // ---------- CUSTOM CAMERA ----------
    let cameraStream = null;
    let currentFacingMode = 'environment'; // Default to back camera
    // ---------- USER INFO & AVATAR ----------
    const userNameDisplay = document.getElementById('user-name-display');
    const userEmailDisplay = document.getElementById('user-email-display');
    const userAvatar = document.getElementById('user-avatar');

    // Mobile versions
    const mobileUserName = document.getElementById('mobile-user-name');
    const mobileUserEmail = document.getElementById('mobile-user-email');
    const mobileUserAvatar = document.getElementById('mobile-user-avatar');

    if (user) {
        if (userNameDisplay) userNameDisplay.textContent = user.name;
        if (userEmailDisplay) userEmailDisplay.textContent = user.email;
        if (mobileUserName) mobileUserName.textContent = user.name;
        if (mobileUserEmail) mobileUserEmail.textContent = user.email;

        const initials = user.name.split(' ').map(n => n[0]).join('').toUpperCase();
        if (userAvatar) userAvatar.textContent = initials;
        if (mobileUserAvatar) mobileUserAvatar.textContent = initials;
    }

    // ---------- MOBILE MENU ----------
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const closeMenuBtn = document.getElementById('close-menu-btn');
    const mobileMenuOverlay = document.getElementById('mobile-menu-overlay');
    const mobileLogoutBtn = document.getElementById('mobile-logout-btn');

    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', () => {
            mobileMenuOverlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        });
    }

    if (closeMenuBtn) {
        closeMenuBtn.addEventListener('click', () => {
            mobileMenuOverlay.classList.remove('active');
            document.body.style.overflow = '';
        });
    }

    if (mobileMenuOverlay) {
        mobileMenuOverlay.addEventListener('click', (e) => {
            if (e.target === mobileMenuOverlay) {
                mobileMenuOverlay.classList.remove('active');
                document.body.style.overflow = '';
            }
        });
    }

    if (mobileLogoutBtn) {
        mobileLogoutBtn.addEventListener('click', () => {
            sessionStorage.removeItem('user');
            window.location.href = '/login';
        });
    }
    const cameraModal = document.getElementById('camera-modal');
    const cameraVideo = document.getElementById('camera-video');
    const cameraCanvas = document.getElementById('camera-canvas');
    const switchBtn = document.getElementById('switch-camera-btn');

    async function startCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
        }

        const constraints = {
            video: {
                facingMode: { ideal: currentFacingMode },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        };

        try {
            cameraStream = await navigator.mediaDevices.getUserMedia(constraints);
            cameraVideo.srcObject = cameraStream;
            cameraModal.classList.add('active');

            // Mirror video if using front camera
            if (currentFacingMode === 'user') {
                cameraVideo.style.transform = 'scaleX(-1)';
            } else {
                cameraVideo.style.transform = 'scaleX(1)';
            }

            document.getElementById('camera-instruction').textContent =
                currentFacingMode === 'user' ? 'Taking a selfie' : 'Position your subject';
        } catch (err) {
            console.error('Camera error:', err);
            showToast('Could not access camera. Please check permissions.', 'error');
            // Fallback to traditional input if possible
            document.getElementById('camera-input').click();
        }
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
        cameraVideo.srcObject = null;
        cameraModal.classList.remove('active');
    }

    async function switchCamera() {
        currentFacingMode = (currentFacingMode === 'user') ? 'environment' : 'user';
        await startCamera();
    }

    function capturePhoto() {
        const context = cameraCanvas.getContext('2d');
        cameraCanvas.width = cameraVideo.videoWidth;
        cameraCanvas.height = cameraVideo.videoHeight;

        // Handle mirroring for selfie in the final image
        if (currentFacingMode === 'user') {
            context.translate(cameraCanvas.width, 0);
            context.scale(-1, 1);
        }

        context.drawImage(cameraVideo, 0, 0, cameraCanvas.width, cameraCanvas.height);

        imageData = cameraCanvas.toDataURL('image/jpeg', 0.8);
        document.getElementById('image-preview-img').src = imageData;
        document.getElementById('image-upload-area').style.display = 'none';
        document.getElementById('image-preview-area').style.display = 'block';

        stopCamera();
        showToast('Photo captured!', 'success');
    }

    document.getElementById('camera-zone').addEventListener('click', (e) => {
        e.preventDefault();
        startCamera();
    });

    document.getElementById('close-camera-modal').addEventListener('click', stopCamera);
    switchBtn.addEventListener('click', switchCamera);
    document.getElementById('take-photo-btn').addEventListener('click', capturePhoto);

    // Close camera modal on overlay click
    cameraModal.addEventListener('click', (e) => {
        if (e.target === cameraModal) stopCamera();
    });


    function handleImageFile(e) {
        const file = e.target.files[0];
        if (!file) return;
        if (!file.type.startsWith('image/')) { showToast('Please select an image file', 'error'); return; }
        if (file.size > 5 * 1024 * 1024) { showToast('Image must be less than 5MB', 'error'); return; }
        const reader = new FileReader();
        reader.onload = (ev) => {
            imageData = ev.target.result;
            document.getElementById('image-preview-img').src = imageData;
            document.getElementById('image-upload-area').style.display = 'none';
            document.getElementById('image-preview-area').style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
    document.getElementById('image-file-input').addEventListener('change', handleImageFile);
    document.getElementById('camera-input').addEventListener('change', handleImageFile);
    document.getElementById('remove-image-btn').addEventListener('click', () => {
        imageData = null;
        document.getElementById('image-file-input').value = '';
        document.getElementById('camera-input').value = '';
        document.getElementById('image-caption').value = '';
        document.getElementById('image-upload-area').style.display = 'block';
        document.getElementById('image-preview-area').style.display = 'none';
    });

    document.getElementById('save-image-btn').addEventListener('click', async () => {
        if (!imageData) { showToast('Please select an image', 'error'); return; }
        const btn = document.getElementById('save-image-btn');
        btn.disabled = true; btn.innerHTML = '<span class="spinner-white"></span> Processing with AI...';
        const caption = document.getElementById('image-caption').value || 'Image memory';
        try {
            const res = await fetch('/api/process-memory', {
                method: 'POST', headers: { 'Content-Type': 'application/json', 'X-User-Id': user.id },
                body: JSON.stringify({ voice_text: caption, has_image: true, image_data: imageData })
            });
            const data = await res.json();
            if (res.ok) {
                showToast('Memory saved with AI analysis!', 'success');
                document.getElementById('remove-image-btn').click();
                loadMemories(); loadUsage();
            } else if (res.status === 402) {
                showToast(data.message, 'error'); openPremium();
            } else showToast(data.error || 'Failed to save', 'error');
        } catch (e) { showToast('Connection error', 'error'); }
        finally { btn.disabled = false; btn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/></svg> Save Memory'; }
    });


    // ---------- LOAD MEMORIES ----------
    async function loadMemories() {
        try {
            const res = await fetch('/api/search-memories?q=*', { headers: { 'X-User-Id': user.id } });
            const data = await res.json();
            memories = data.results || [];
            renderMemories();
        } catch (e) { console.error('Failed to load memories', e); }
    }

    function renderMemories() {
        const query = (document.getElementById('dash-search').value || '').toLowerCase();
        let filtered = memories.filter(m => {
            const matchQuery = !query ||
                (m.title || '').toLowerCase().includes(query) ||
                (m.content || '').toLowerCase().includes(query) ||
                ((m.tags || []).join(' ')).toLowerCase().includes(query);
            const matchCat = selectedCategory === 'all' ||
                (m.category || '').toLowerCase() === selectedCategory;
            return matchQuery && matchCat;
        });

        const list = document.getElementById('memory-list');
        if (filtered.length === 0) {
            list.innerHTML = `<div class="empty-state">
                <div class="empty-icon"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></div>
                <h3>No memories found</h3>
                <p>${query ? 'Try a different search term' : 'Start capturing your thoughts to see them here'}</p>
            </div>`;
            document.getElementById('view-all-wrap').style.display = 'none';
        } else {
            list.innerHTML = filtered.slice(0, 5).map(renderMemoryCard).join('');
            document.getElementById('view-all-wrap').style.display = filtered.length > 0 ? 'block' : 'none';
        }
    }

    // ---------- LOAD USAGE ----------
    async function loadUsage() {
        try {
            const res = await fetch('/api/user/usage', { headers: { 'X-User-Id': user.id } });
            const data = await res.json();
            const used = data.memories_used || 0;
            const limit = data.memory_limit || 10;
            const isPremium = data.is_premium || false;
            const pct = isPremium ? 0 : Math.min((used / limit) * 100, 100);

            document.getElementById('usage-text').textContent = isPremium ?
                'You have unlimited memories' :
                `${used} of ${limit} free memories used • ${Math.max(0, limit - used)} remaining`;
            document.getElementById('usage-bar').style.width = pct + '%';
            if (pct >= 80) document.getElementById('usage-bar').classList.add('warning');
            if (isPremium) {
                document.getElementById('usage-upgrade-btn').style.display = 'none';
                document.getElementById('upgrade-btn').style.display = 'none';
            }
        } catch (e) { console.error('Failed to load usage', e); }
    }

    // Init
    loadMemories();
    loadUsage();
})();

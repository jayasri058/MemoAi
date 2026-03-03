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
        showToast('Logged out successfully', 'success');
        setTimeout(() => window.location.href = '/', 500);
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
    document.getElementById('camera-zone').addEventListener('click', () => document.getElementById('camera-input').click());

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

    // ---------- PDF UPLOADER ----------
    let pdfFile = null;
    document.getElementById('pdf-upload-zone').addEventListener('click', () => document.getElementById('pdf-file-input').click());
    document.getElementById('pdf-file-input').addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (!file) return;
        if (file.type !== 'application/pdf') { showToast('Please select a PDF', 'error'); return; }
        if (file.size > 10 * 1024 * 1024) { showToast('PDF must be less than 10MB', 'error'); return; }
        pdfFile = file;
        document.getElementById('pdf-file-name').textContent = file.name;
        const kb = file.size / 1024;
        document.getElementById('pdf-file-size').textContent = kb < 1024 ? kb.toFixed(2) + ' KB' : (kb / 1024).toFixed(2) + ' MB';
        document.getElementById('pdf-upload-area').style.display = 'none';
        document.getElementById('pdf-file-area').style.display = 'block';
    });
    document.getElementById('remove-pdf-btn').addEventListener('click', () => {
        pdfFile = null;
        document.getElementById('pdf-file-input').value = '';
        document.getElementById('pdf-upload-area').style.display = 'block';
        document.getElementById('pdf-file-area').style.display = 'none';
    });

    document.getElementById('process-pdf-btn').addEventListener('click', async () => {
        if (!pdfFile) return;
        document.getElementById('pdf-file-area').style.display = 'none';
        document.getElementById('pdf-processing-area').style.display = 'block';
        document.getElementById('pdf-processing-name').textContent = pdfFile.name;
        // Read file as base64
        const reader = new FileReader();
        reader.onload = async (ev) => {
            const base64 = ev.target.result;
            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                progress = Math.min(progress + 5, 90);
                document.getElementById('pdf-progress-bar').style.width = progress + '%';
                document.getElementById('pdf-progress-text').textContent = progress + '% complete';
            }, 200);
            try {
                const res = await fetch('/api/process-memory', {
                    method: 'POST', headers: { 'Content-Type': 'application/json', 'X-User-Id': user.id },
                    body: JSON.stringify({ voice_text: 'PDF: ' + pdfFile.name, has_image: true, image_data: base64 })
                });
                clearInterval(progressInterval);
                document.getElementById('pdf-progress-bar').style.width = '100%';
                document.getElementById('pdf-progress-text').textContent = '100% complete';
                const data = await res.json();
                if (res.ok) {
                    showToast('PDF processed successfully!', 'success');
                    setTimeout(() => {
                        pdfFile = null;
                        document.getElementById('pdf-file-input').value = '';
                        document.getElementById('pdf-processing-area').style.display = 'none';
                        document.getElementById('pdf-upload-area').style.display = 'block';
                        document.getElementById('pdf-progress-bar').style.width = '0%';
                    }, 1000);
                    loadMemories(); loadUsage();
                } else if (res.status === 402) {
                    showToast(data.message, 'error'); openPremium();
                    document.getElementById('pdf-processing-area').style.display = 'none';
                    document.getElementById('pdf-upload-area').style.display = 'block';
                } else showToast(data.error || 'Failed', 'error');
            } catch (e) { clearInterval(progressInterval); showToast('Connection error', 'error'); }
        };
        reader.readAsDataURL(pdfFile);
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

// Theme Toggle Functionality
const themeToggle = document.getElementById('themeToggle');
const body = document.body;

// Check for saved theme preference or respect OS setting
const savedTheme = localStorage.getItem('theme');
const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');

if (savedTheme === 'dark' || (!savedTheme && prefersDarkScheme.matches)) {
    body.setAttribute('data-theme', 'dark');
    themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
} else {
    body.setAttribute('data-theme', 'light');
    themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
}

themeToggle.addEventListener('click', () => {
    const currentTheme = body.getAttribute('data-theme');

    if (currentTheme === 'dark') {
        body.setAttribute('data-theme', 'light');
        themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        localStorage.setItem('theme', 'light');
    } else {
        body.setAttribute('data-theme', 'dark');
        themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
        localStorage.setItem('theme', 'dark');
    }
});

// Tab Navigation
const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all buttons and contents
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));

        // Add active class to clicked button
        btn.classList.add('active');

        // Show corresponding content
        const tabId = btn.getAttribute('data-tab');
        document.getElementById(`${tabId}-tab`).classList.add('active');
    });
});

// Audio Recording Implementation
let isRecording = false;
let mediaRecorder;
let audioChunks = [];
let audioContext;
let audioStream;
let analyser;
let silenceTimer;
const SILENCE_DURATION = 30000; // 30 seconds of silence

const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const audioProcessing = document.getElementById('audioProcessing');
const audioResults = document.getElementById('audioResults');
const transcriptionResult = document.getElementById('transcriptionResult');
const categoryResult = document.getElementById('categoryResult');
const confirmAudioBtn = document.getElementById('confirmAudioBtn');
const retryAudioBtn = document.getElementById('retryAudioBtn');
let currentAudioData = null; // Store temp transcription data

// Initialize audio recording
async function initAudioRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioStream = stream;

        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioContext.createMediaStreamSource(stream);

        const bufferSize = 4096;
        mediaRecorder = audioContext.createScriptProcessor(bufferSize, 1, 1);

        mediaRecorder.onaudioprocess = (e) => {
            if (!isRecording) return;
            const left = e.inputBuffer.getChannelData(0);
            audioChunks.push(new Float32Array(left));
        };

        const silentGain = audioContext.createGain();
        silentGain.gain.value = 0;
        source.connect(mediaRecorder);
        mediaRecorder.connect(silentGain);
        silentGain.connect(audioContext.destination);

        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);

        monitorSilence(analyser);
        return true;
    } catch (err) {
        console.error('Error accessing microphone:', err);
        return false;
    }
}

function monitorSilence(analyser) {
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    function checkSilence() {
        if (!isRecording) return;
        analyser.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < bufferLength; i++) sum += dataArray[i];
        const avgVolume = sum / bufferLength;

        if (avgVolume < 2) {
            if (!silenceTimer) {
                silenceTimer = setTimeout(() => {
                    if (isRecording) stopRecording(true);
                }, SILENCE_DURATION);
            }
        } else {
            if (silenceTimer) {
                clearTimeout(silenceTimer);
                silenceTimer = null;
            }
        }
        if (isRecording) requestAnimationFrame(checkSilence);
    }
    checkSilence();
}

async function startRecording() {
    if (isRecording) return;
    if (!mediaRecorder) {
        const success = await initAudioRecording();
        if (!success) return;
    }
    if (audioContext && audioContext.state === 'suspended') await audioContext.resume();

    isRecording = true;
    audioChunks = [];
    recordBtn.disabled = true;
    stopBtn.disabled = false;
    document.getElementById('visualizerContainer').style.display = 'flex';
    recordBtn.innerHTML = '<i class="fas fa-microphone fa-beat" style="color: #ef4444;"></i><span>Recording...</span>';
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;
    recordBtn.disabled = false;
    stopBtn.disabled = true;
    document.getElementById('visualizerContainer').style.display = 'none';
    recordBtn.innerHTML = '<i class="fas fa-microphone"></i><span>Record</span>';

    if (silenceTimer) {
        clearTimeout(silenceTimer);
        silenceTimer = null;
    }
    if (mediaRecorder) {
        mediaRecorder.disconnect();
        mediaRecorder = null;
    }
    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    if (audioChunks.length > 0) {
        const wavBlob = exportWAV(audioChunks, audioContext.sampleRate);
        const file = new File([wavBlob], `recording_${Date.now()}.wav`, { type: 'audio/wav' });
        processRecordedAudio(file);
    }
}

async function processRecordedAudio(audioFile) {
    audioProcessing.style.display = 'flex';
    audioResults.style.display = 'none';
    try {
        const result = await transcribeAudio(audioFile);
        currentAudioData = {
            transcription: result.transcription,
            category: result.category,
            file_path: result.file_path
        };
        audioProcessing.style.display = 'none';
        transcriptionResult.textContent = result.transcription;
        categoryResult.textContent = result.category;
        audioResults.style.display = 'block';
    } catch (error) {
        audioProcessing.style.display = 'none';
        showToast(`Error: ${error.message}`, 'error');
    }
}

async function confirmAndStoreVoiceNote() {
    if (!currentAudioData) return;
    const editedTranscription = transcriptionResult.innerText.trim();
    if (!editedTranscription) return;
    currentAudioData.transcription = editedTranscription;
    audioProcessing.style.display = 'flex';
    audioResults.style.display = 'none';

    try {
        const response = await fetch('/api/confirm-audio', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentAudioData)
        });
        if (!response.ok) throw new Error('Failed to save memory');
        audioProcessing.style.display = 'none';
        showToast('Memory successfully saved!', 'success');
        currentAudioData = null;
    } catch (error) {
        audioProcessing.style.display = 'none';
        showToast(`Error: ${error.message}`, 'error');
    }
}

function exportWAV(buffers, sampleRate) {
    const buffer = mergeBuffers(buffers);
    const dataview = encodeWAV(buffer, sampleRate);
    return new Blob([dataview], { type: 'audio/wav' });
}

function mergeBuffers(buffers) {
    const result = new Float32Array(buffers.reduce((acc, b) => acc + b.length, 0));
    let offset = 0;
    for (let i = 0; i < buffers.length; i++) {
        result.set(buffers[i], offset);
        offset += buffers[i].length;
    }
    return result;
}

function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);
    return view;
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
}

recordBtn.addEventListener('click', startRecording);
stopBtn.addEventListener('click', stopRecording);
confirmAudioBtn.addEventListener('click', confirmAndStoreVoiceNote);
retryAudioBtn.addEventListener('click', async () => {
    if (currentAudioData && currentAudioData.file_path) {
        await fetch('/api/delete-temp-audio', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ file_path: currentAudioData.file_path })
        });
    }
    audioResults.style.display = 'none';
    currentAudioData = null;
    showToast('Redoing recording...', 'info');
});

const storeImageBtn = document.getElementById('storeImageBtn');
storeImageBtn.addEventListener('click', async () => {
    const fileInput = document.getElementById('imageUpload');
    const contextInput = document.getElementById('imageContext');
    if (fileInput.files.length === 0) return;

    const imageProcessing = document.getElementById('imageProcessing');
    const imageResults = document.getElementById('imageResults');
    imageProcessing.style.display = 'flex';
    imageResults.style.display = 'none';

    try {
        const result = await processImage(fileInput.files[0], contextInput.value);
        imageProcessing.style.display = 'none';
        document.getElementById('imageAnalysisResult').textContent = result.description;
        document.getElementById('storedMemoryResult').textContent = result.message;
        imageResults.style.display = 'block';
        showToast('Image memory stored!', 'success');
    } catch (error) {
        imageProcessing.style.display = 'none';
        showToast(`Error: ${error.message}`, 'error');
    }
});

const searchBtn = document.getElementById('searchBtn');
searchBtn.addEventListener('click', performSearch);
document.getElementById('searchQuery').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') performSearch();
});

async function performSearch() {
    const query = document.getElementById('searchQuery').value.trim();
    const category = document.getElementById('categoryFilter').value;
    const type = document.getElementById('typeFilter').value;
    const searchProcessing = document.getElementById('searchProcessing');
    const searchResults = document.getElementById('searchResults');

    searchProcessing.style.display = 'flex';
    searchResults.style.display = 'none';

    try {
        const result = await searchMemories(query || " ", category || null, type || null);
        searchProcessing.style.display = 'none';
        displaySearchResults(result.results);
        searchResults.style.display = 'block';
    } catch (error) {
        searchProcessing.style.display = 'none';
        showToast(`Search error: ${error.message}`, 'error');
    }
}

function displaySearchResults(results) {
    const list = document.getElementById('searchResultsList');
    list.innerHTML = '';
    if (results.length === 0) {
        list.innerHTML = '<p style="text-align: center; padding: 2rem; opacity: 0.7;">No memories found.</p>';
        return;
    }
    const query = document.getElementById('searchQuery').value.toLowerCase();
    const queryWords = query.split(/\s+/).filter(w => w.length > 2);

    results.forEach(res => {
        const item = document.createElement('div');
        item.className = 'search-result-item fade-in';
        const matchPct = Math.min(100, Math.round((res.relevance_score || 0) * 100));

        let content = res.content || '';
        queryWords.forEach(word => {
            const regex = new RegExp(`(${word})(?![^<]*>)`, 'gi');
            content = content.replace(regex, '<mark class="match-highlight">$1</mark>');
        });

        item.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: 600; text-transform: uppercase; font-size: 0.8rem; color: var(--primary-color);">
                    <i class="fas ${res.type === 'audio' ? 'fa-microphone' : 'fa-image'}"></i> ${res.type}
                </span>
                <span style="font-size: 0.7rem; background: #f0fdf4; color: #16a34a; padding: 2px 8px; border-radius: 10px; font-weight: 600;">${matchPct}% match</span>
            </div>
            ${res.type === 'image' && res.image_url ? `<img src="${res.image_url}" style="width: 100%; border-radius: 8px; margin-bottom: 10px;">` : ''}
            <div style="font-size: 0.95rem; margin-bottom: 10px;">${content}</div>
            <div style="font-size: 0.75rem; opacity: 0.7;">${new Date(res.timestamp).toLocaleString()}</div>
        `;
        list.appendChild(item);
    });
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => container.removeChild(toast), 300);
    }, 3000);
}

function scrollToSection(id) {
    const el = document.getElementById(id);
    if (el) window.scrollTo({ top: el.offsetTop - 80, behavior: 'smooth' });
}

// Helper API functions
async function transcribeAudio(file) {
    const fd = new FormData(); fd.append('audio', file);
    const res = await fetch('/api/transcribe', { method: 'POST', body: fd });
    return res.json();
}

async function processImage(file, context) {
    const fd = new FormData(); fd.append('image', file); fd.append('context', context);
    const res = await fetch('/api/process-image', { method: 'POST', body: fd });
    return res.json();
}

async function searchMemories(query, category, type) {
    const res = await fetch('/api/search-memories', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, category, type })
    });
    return res.json();
}

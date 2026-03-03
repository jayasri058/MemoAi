/* =========================================
   MemoAI — Shared Utilities
   ========================================= */

// Toast notification system
function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all .3s';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// Get current user from session
function getCurrentUser() {
    try { return JSON.parse(sessionStorage.getItem('user')); }
    catch { return null; }
}

// Format relative date
function formatRelativeDate(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);
    if (diffMins < 1) return 'Just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

// Category badge CSS class
function getCategoryClass(category) {
    const map = {
        'Daily Life': 'cat-daily-life',
        'Work & Meetings': 'cat-work',
        'Learning & Growth': 'cat-learning',
        'Health & Fitness': 'cat-health',
        'Money & Shopping': 'cat-money',
        'Entertainment': 'cat-entertainment',
        'Entertainment & Leisure': 'cat-entertainment',
        'Ideas & Creativity': 'cat-ideas',
        'General': 'cat-general',
    };
    return map[category] || 'cat-general';
}

// Render a single memory card HTML
function renderMemoryCard(memory) {
    const category = memory.category || 'General';
    const tags = memory.tags || [];
    const tagsArray = Array.isArray(tags) ? tags : (typeof tags === 'string' ? JSON.parse(tags || '[]') : []);
    const timestamp = memory.created_at || memory.timestamp || new Date().toISOString();
    const imageHtml = memory.image_path && memory.has_image ? 
        `<div class="memory-card-image"><img src="/uploads/${memory.image_path.split(/[/\\]/).pop()}" alt="${memory.title || ''}"></div>` : '';
    
    return `
        <div class="memory-card" data-category="${category.toLowerCase()}">
            <div class="memory-card-header">
                <div style="flex:1;min-width:0">
                    <h3>${memory.title || 'Untitled'}</h3>
                    <div class="memory-date">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
                        <span>${formatRelativeDate(timestamp)}</span>
                    </div>
                </div>
            </div>
            ${imageHtml}
            <p class="memory-content">${memory.content || memory.voice_text || ''}</p>
            <div class="memory-footer">
                <span class="category-badge ${getCategoryClass(category)}">${category}</span>
                ${tagsArray.length > 0 ? `
                    <div class="memory-tags">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"/><line x1="7" y1="7" x2="7.01" y2="7"/></svg>
                        <div class="tag-list">
                            ${tagsArray.slice(0, 3).map(t => `<span class="tag-chip">${t}</span>`).join('')}
                            ${tagsArray.length > 3 ? `<span class="tag-chip">+${tagsArray.length - 3}</span>` : ''}
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>
    `;
}
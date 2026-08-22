/**
 * static/js/chat.js
 * AI chat panel: drawer, message history, starter chips.
 * Only activates after analysis results are loaded.
 */

(function () {
  'use strict';

  let _jobId = null;
  let _messages = [];  // [{role, content}]
  let _isLoading = false;

  const STARTER_CHIPS = [
    'What are the main patterns in this dataset?',
    'Which columns have the most outliers?',
    'What does the clustering reveal about the data?',
    'Give me 3 actionable recommendations.',
    'Which features are most correlated?',
  ];

  function init(jobId) {
    _jobId = jobId;

    // Restore history from sessionStorage
    const saved = sessionStorage.getItem(`chat_${jobId}`);
    if (saved) {
      try { _messages = JSON.parse(saved); } catch { _messages = []; }
    } else {
      _messages = [];
    }

    _renderMessages();
    _updateFabBadge();
  }

  function reset() {
    _jobId = null;
    _messages = [];
    const msgList = document.getElementById('chat-messages');
    if (msgList) msgList.innerHTML = '';
    _showStarters(true);
  }

  // ─── Drawer Toggle ──────────────────────────────────────────────────────────
  function toggleDrawer() {
    const drawer = document.getElementById('chat-drawer');
    if (!drawer) return;
    const isOpen = drawer.classList.contains('open');
    drawer.classList.toggle('open', !isOpen);

    // Clear badge when opening
    if (!isOpen) {
      const badge = document.getElementById('chat-fab-badge');
      if (badge) badge.textContent = '';
    }

    // Show/hide starters if no messages
    _showStarters(_messages.length === 0);
  }

  function closeDrawer() {
    const drawer = document.getElementById('chat-drawer');
    if (drawer) drawer.classList.remove('open');
  }

  // ─── Sending Messages ────────────────────────────────────────────────────────
  async function sendMessage(content = null) {
    if (_isLoading || !_jobId) return;

    const inputEl = document.getElementById('chat-input');
    const text = content || (inputEl ? inputEl.value.trim() : '');
    if (!text) return;

    if (inputEl) inputEl.value = '';

    // Hide starter chips
    _showStarters(false);

    _messages.push({ role: 'user', content: text });
    _appendUserBubble(text);
    _isLoading = true;

    const typingEl = _showTyping();

    try {
      const res = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_id: _jobId, messages: _messages }),
      });

      const data = await res.json();
      typingEl.remove();

      if (!res.ok || data.error) throw new Error(data.error || 'Chat failed');

      const reply = data.reply || '';
      _messages.push({ role: 'assistant', content: reply });
      _appendAssistantBubble(reply);

      // Save to sessionStorage
      try { sessionStorage.setItem(`chat_${_jobId}`, JSON.stringify(_messages)); } catch {}

    } catch (err) {
      typingEl.remove();
      _appendErrorBubble(err.message);
    } finally {
      _isLoading = false;
    }
  }

  function _appendUserBubble(text) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    const el = document.createElement('div');
    el.className = 'msg-bubble msg-user';
    el.textContent = text;
    container.appendChild(el);
    _scrollToBottom(container);
  }

  function _appendAssistantBubble(text) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    const el = document.createElement('div');
    el.className = 'msg-bubble msg-assistant markdown-body';
    try { el.innerHTML = marked.parse(text); } catch { el.textContent = text; }
    container.appendChild(el);
    _scrollToBottom(container);

    // Update badge if drawer is closed
    const drawer = document.getElementById('chat-drawer');
    if (drawer && !drawer.classList.contains('open')) {
      _updateFabBadge();
    }
  }

  function _appendErrorBubble(msg) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    const el = document.createElement('div');
    el.className = 'msg-bubble msg-assistant';
    el.style.cssText = 'border-color:rgba(255,123,123,0.25); color:var(--clr-error)';
    el.textContent = `Error: ${msg}`;
    container.appendChild(el);
    _scrollToBottom(container);
  }

  function _showTyping() {
    const container = document.getElementById('chat-messages');
    if (!container) return document.createElement('div');
    const el = document.createElement('div');
    el.className = 'msg-typing';
    el.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    container.appendChild(el);
    _scrollToBottom(container);
    return el;
  }

  function _renderMessages() {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    container.innerHTML = '';
    _messages.forEach(msg => {
      if (msg.role === 'user') _appendUserBubble(msg.content);
      else _appendAssistantBubble(msg.content);
    });
    _showStarters(_messages.length === 0);
  }

  function _showStarters(show) {
    const el = document.getElementById('chat-starters');
    if (el) el.style.display = show ? 'block' : 'none';
  }

  function _scrollToBottom(el) {
    setTimeout(() => { el.scrollTop = el.scrollHeight; }, 50);
  }

  function _updateFabBadge() {
    const assistantCount = _messages.filter(m => m.role === 'assistant').length;
    const badge = document.getElementById('chat-fab-badge');
    if (badge) badge.textContent = assistantCount > 0 ? assistantCount : '';
  }

  // ─── Input Handling ──────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    const inputEl = document.getElementById('chat-input');
    if (inputEl) {
      inputEl.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      });
      // Auto-resize
      inputEl.addEventListener('input', () => {
        inputEl.style.height = 'auto';
        inputEl.style.height = Math.min(inputEl.scrollHeight, 100) + 'px';
      });
    }

    // Render starter chips
    const chipsContainer = document.getElementById('starter-chips');
    if (chipsContainer) {
      STARTER_CHIPS.forEach(q => {
        const chip = document.createElement('button');
        chip.className = 'chip';
        chip.textContent = q;
        chip.addEventListener('click', () => sendMessage(q));
        chipsContainer.appendChild(chip);
      });
    }
  });

  // Public API
  window.ChatModule = { init, reset, toggleDrawer, closeDrawer, sendMessage };
})();

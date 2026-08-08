/* ==========================================================================
   AI Technical Interview Agent - Enterprise Frontend Logic & Perceived UX
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Determine API Base URL (Supports same-origin FastAPI, local ports 8000/8001, etc.)
  const API_BASE = window.location.origin.includes('8000') || window.location.origin.includes('8001') || window.location.origin.includes('3000') || window.location.origin.includes('127.0.0.1') || window.location.origin.includes('localhost')
    ? `${window.location.protocol}//${window.location.hostname}:${window.location.port || '8000'}`
    : window.location.origin;

  const MIN_THINKING_MS = 1500; // Enforced 1.5 second minimum perceived AI thinking delay

  // Application State
  let candidates = [];
  let selectedCandidate = null;
  let currentSessionId = null;
  let currentTurnCount = 0;
  let maxTurns = 10;
  let isInterviewDone = false;
  let isSubmitting = false;

  // DOM Elements
  const backendStatusBadge = document.getElementById('backend-status');
  const btnResetSession = document.getElementById('btn-reset-session');
  
  // Selection Screen
  const selectionScreen = document.getElementById('selection-screen');
  const candidateGrid = document.getElementById('candidate-grid');
  const candidateSearch = document.getElementById('candidate-search');
  const inspectorPlaceholder = document.getElementById('inspector-placeholder');
  const inspectorContent = document.getElementById('inspector-content');
  
  // Profile Inspector Fields
  const detailAvatar = document.getElementById('detail-avatar');
  const detailName = document.getElementById('detail-name');
  const detailRole = document.getElementById('detail-role');
  const detailExp = document.getElementById('detail-exp');
  const detailEdu = document.getElementById('detail-edu');
  const detailCommits = document.getElementById('detail-commits');
  const detailFirstTry = document.getElementById('detail-first-try');
  const countMastered = document.getElementById('count-mastered');
  const countStruggled = document.getElementById('count-struggled');
  const countFailed = document.getElementById('count-failed');
  const countSkipped = document.getElementById('count-skipped');
  const btnStartInterview = document.getElementById('btn-start-interview');

  // Assessment / Chat Screen
  const chatScreen = document.getElementById('chat-screen');
  const chatCandidateAvatar = document.getElementById('chat-candidate-avatar');
  const chatCandidateName = document.getElementById('chat-candidate-name');
  const chatCandidateRole = document.getElementById('chat-candidate-role');
  const chatSessionId = document.getElementById('chat-session-id');
  const chatModeBadge = document.getElementById('chat-mode-badge');
  const chatTurnCount = document.getElementById('chat-turn-count');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const interviewStatusTag = document.getElementById('interview-status-tag');
  
  const chatMessages = document.getElementById('chat-messages');
  const completionBanner = document.getElementById('completion-banner');
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const btnSendMessage = document.getElementById('btn-send-message');

  // Evaluation Report Elements
  const feedbackPanel = document.getElementById('feedback-panel');
  const fbSummary = document.getElementById('fb-summary');
  const fbStrengthsList = document.getElementById('fb-strengths-list');
  const fbGapsList = document.getElementById('fb-gaps-list');
  const fbNextList = document.getElementById('fb-next-list');
  const btnRestartAfterFeedback = document.getElementById('btn-restart-after-feedback');

  // Utility Functions
  function getInitials(name) {
    if (!name) return 'CA';
    const parts = name.trim().split(' ');
    return parts.length >= 2 ? (parts[0][0] + parts[parts.length - 1][0]).toUpperCase() : name.substring(0, 2).toUpperCase();
  }

  function getTimeString() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }

  function generateSessionId() {
    return 'session-' + Date.now().toString(36) + '-' + Math.random().toString(36).substring(2, 7);
  }

  function escapeHtml(str) {
    const p = document.createElement('p');
    p.textContent = str;
    return p.innerHTML;
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // 1. Initialize & Check Server Status
  async function init() {
    await checkHealth();
    await fetchCandidates();
    setupEventListeners();
  }

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        maxTurns = data.max_turns || 10;
        const modeText = data.mock_mode ? 'AI Online (Mock Mode)' : `AI Online (${data.model})`;
        backendStatusBadge.querySelector('.status-text').textContent = modeText;
        chatModeBadge.textContent = data.mock_mode ? 'Mock Mode' : data.model;
      }
    } catch (err) {
      console.warn('Health check note:', err);
      backendStatusBadge.querySelector('.status-text').textContent = 'AI Offline';
    }
  }

  // 2. Fetch Candidates
  async function fetchCandidates() {
    try {
      const res = await fetch(`${API_BASE}/api/candidates`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      candidates = data.candidates || [];
      renderCandidateGrid(candidates);
    } catch (err) {
      console.error('Failed to load candidates:', err);
      candidateGrid.innerHTML = `<div style="grid-column:1/-1;padding:20px;text-align:center;color:var(--rose)">Failed to load candidate profiles.</div>`;
    }
  }

  function renderCandidateGrid(list) {
    if (!list || list.length === 0) {
      candidateGrid.innerHTML = `<div style="grid-column:1/-1;padding:20px;text-align:center;color:var(--text-secondary)">No candidates found.</div>`;
      return;
    }

    candidateGrid.innerHTML = list.map(c => {
      const m = c.member || {};
      const s = c.signals || {};
      const isSel = selectedCandidate && (m.id === selectedCandidate.member?.id);

      return `
        <div class="candidate-card ${isSel ? 'selected' : ''}" data-id="${m.id}">
          <div class="card-header-row">
            <div class="avatar-circle">${getInitials(m.name)}</div>
            <div class="card-title-group">
              <h3>${escapeHtml(m.name || 'Candidate')}</h3>
              <span class="role-tag">${escapeHtml(m.jobRole || 'Software Engineer')}</span>
            </div>
          </div>
          <div class="card-metrics-row">
            <div class="card-metric"><span class="m-lbl">Exp</span><span class="m-val">${m.yearsExperience ?? 0} Yrs</span></div>
            <div class="card-metric"><span class="m-lbl">Commits</span><span class="m-val">${s.commitDays ?? 0} Days</span></div>
            <div class="card-metric"><span class="m-lbl">Missions</span><span class="m-val">${s.missionsCompleted ?? 0}</span></div>
          </div>
        </div>
      `;
    }).join('');

    document.querySelectorAll('.candidate-card').forEach(card => {
      card.addEventListener('click', () => {
        const cid = card.getAttribute('data-id');
        const cand = candidates.find(item => item.member?.id === cid);
        if (cand) selectCandidate(cand);
      });
    });
  }

  function selectCandidate(cand) {
    selectedCandidate = cand;
    document.querySelectorAll('.candidate-card').forEach(card => {
      card.classList.toggle('selected', card.getAttribute('data-id') === cand.member?.id);
    });

    const m = cand.member || {};
    const s = cand.signals || {};
    const missions = cand.missions || [];

    let mastered = 0, struggled = 0, failed = 0, skipped = 0;
    missions.forEach(mission => {
      if (mission.skipped) skipped++;
      else if (mission.passed === false) failed++;
      else if (mission.passed === true) {
        if ((mission.attempts || 1) <= 2) mastered++;
        else struggled++;
      } else struggled++;
    });

    const firstTryRate = s.missionsCompleted ? Math.round(((s.missionsFirstTry || 0) / s.missionsCompleted) * 100) : 0;

    detailAvatar.textContent = getInitials(m.name);
    detailName.textContent = m.name || 'Candidate';
    detailRole.textContent = m.jobRole || 'Software Engineer';
    detailExp.textContent = `${m.yearsExperience ?? 0} Yrs`;
    detailEdu.textContent = m.education || 'N/A';
    detailCommits.textContent = `${s.commitDays ?? 0} Days`;
    detailFirstTry.textContent = `${firstTryRate}%`;

    countMastered.textContent = mastered;
    countStruggled.textContent = struggled;
    countFailed.textContent = failed;
    countSkipped.textContent = skipped;

    inspectorPlaceholder.classList.add('hidden');
    inspectorContent.classList.remove('hidden');
  }

  // 3. Start Assessment
  async function startInterview() {
    if (!selectedCandidate || isSubmitting) return;

    isSubmitting = true;
    currentSessionId = generateSessionId();
    currentTurnCount = 0;
    isInterviewDone = false;

    btnStartInterview.disabled = true;

    try {
      const payload = { sessionId: currentSessionId, candidate: selectedCandidate };
      const res = await fetch(`${API_BASE}/api/interview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server status ${res.status}`);
      }

      const response = await res.json();
      setupChatView(response);
    } catch (err) {
      console.error('Start error:', err);
      alert(`Error starting session: ${err.message}`);
    } finally {
      isSubmitting = false;
      btnStartInterview.disabled = false;
    }
  }

  function setupChatView(startResponse) {
    const m = selectedCandidate.member || {};

    chatCandidateAvatar.textContent = getInitials(m.name);
    chatCandidateName.textContent = m.name || 'Candidate';
    chatCandidateRole.textContent = `${m.jobRole || 'Engineer'} • ${m.yearsExperience ?? 0} Yrs`;
    chatSessionId.textContent = currentSessionId;
    updateProgressUI(0);

    interviewStatusTag.innerHTML = `<span class="dot-active"></span><span>Interview In Progress</span>`;

    chatMessages.innerHTML = '';
    appendMessage('interviewer', startResponse.reply);

    completionBanner.classList.add('hidden');
    feedbackPanel.classList.add('hidden');
    chatInput.disabled = false;
    chatInput.value = '';
    btnSendMessage.disabled = false;
    btnResetSession.classList.remove('hidden');

    selectionScreen.classList.remove('active');
    chatScreen.classList.add('active');

    chatInput.focus();
  }

  function updateProgressUI(turns) {
    currentTurnCount = turns;
    chatTurnCount.textContent = `Turn ${turns} of ${maxTurns}`;
    const percent = Math.min(100, Math.round((turns / maxTurns) * 100));
    progressBarFill.style.width = `${percent}%`;
  }

  // 4. Handle Candidate Turn Submit with Perceived Thinking Delay
  async function handleTurnSubmit(e) {
    if (e) e.preventDefault();
    if (isSubmitting || isInterviewDone) return;

    const messageText = chatInput.value.trim();
    if (!messageText) return;

    isSubmitting = true;

    // 1. Immediately append candidate message
    appendMessage('candidate', messageText);

    // 2. Clear input & disable input controls
    chatInput.value = '';
    chatInput.disabled = true;
    btnSendMessage.disabled = true;

    // 3. Render 3-dot thinking indicator in conversation
    showThinkingIndicator();

    // 4. Start API call & measure elapsed duration
    const startTime = Date.now();

    try {
      const payload = { sessionId: currentSessionId, message: messageText };
      const res = await fetch(`${API_BASE}/api/interview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server status ${res.status}`);
      }

      const data = await res.json();

      // Calculate perceived delay: thinkingDuration = max(0, MIN_THINKING_MS - actualRequestDuration)
      const elapsedTime = Date.now() - startTime;
      const remainingDelay = Math.max(0, MIN_THINKING_MS - elapsedTime);

      if (remainingDelay > 0) {
        await new Promise(resolve => setTimeout(resolve, remainingDelay));
      }

      // 5. Remove thinking indicator
      removeThinkingIndicator();

      // 6. Update turn metrics & append actual AI response
      updateProgressUI(currentTurnCount + 1);
      appendMessage('interviewer', data.reply);

      // 7. Check if interview concluded
      if (data.done) {
        handleInterviewCompleted(data.feedback);
      } else {
        chatInput.disabled = false;
        btnSendMessage.disabled = false;
        chatInput.focus();
      }

    } catch (err) {
      console.error('Turn error:', err);
      removeThinkingIndicator();
      appendErrorMessage(`⚠️ Connection Error: Unable to process response (${err.message}). Please try again.`);
      chatInput.disabled = false;
      btnSendMessage.disabled = false;
    } finally {
      isSubmitting = false;
    }
  }

  // Show Animated 3-Dot Thinking Indicator in Conversation Area
  function showThinkingIndicator() {
    const thinkingDiv = document.createElement('div');
    thinkingDiv.id = 'thinking-indicator-msg';
    thinkingDiv.className = 'message-row interviewer';
    thinkingDiv.innerHTML = `
      <div class="ai-avatar" style="width:26px;height:26px;font-size:0.8rem">🤖</div>
      <div>
        <div class="thinking-bubble">
          <span class="thinking-dot"></span>
          <span class="thinking-dot"></span>
          <span class="thinking-dot"></span>
        </div>
        <span class="message-meta">AI Interviewer is thinking...</span>
      </div>
    `;
    chatMessages.appendChild(thinkingDiv);
    scrollToBottom();
  }

  function removeThinkingIndicator() {
    const el = document.getElementById('thinking-indicator-msg');
    if (el) el.remove();
  }

  // Append Message to Conversation Log
  function appendMessage(role, content) {
    const isAi = role === 'interviewer';
    const msgDiv = document.createElement('div');
    msgDiv.className = `message-row ${role}`;

    const avatarHtml = isAi 
      ? `<div class="ai-avatar" style="width:26px;height:26px;font-size:0.8rem">🤖</div>` 
      : `<div class="avatar-circle md" style="width:26px;height:26px;font-size:0.75rem">${getInitials(selectedCandidate?.member?.name)}</div>`;

    msgDiv.innerHTML = `
      ${avatarHtml}
      <div>
        <div class="bubble">${escapeHtml(content)}</div>
        <span class="message-meta">${isAi ? 'AI Interviewer' : (selectedCandidate?.member?.name || 'Candidate')} • ${getTimeString()}</span>
      </div>
    `;

    chatMessages.appendChild(msgDiv);
    scrollToBottom();
  }

  function appendErrorMessage(errorText) {
    const errDiv = document.createElement('div');
    errDiv.className = 'message-row interviewer';
    errDiv.innerHTML = `
      <div class="ai-avatar" style="width:26px;height:26px;font-size:0.8rem">🤖</div>
      <div>
        <div class="bubble" style="background-color:var(--rose-bg);color:var(--rose);border-color:rgba(244,63,94,0.3)">${escapeHtml(errorText)}</div>
        <span class="message-meta">System Notice</span>
      </div>
    `;
    chatMessages.appendChild(errDiv);
    scrollToBottom();
  }

  // 5. Handle Interview Completion
  function handleInterviewCompleted(feedback) {
    isInterviewDone = true;

    interviewStatusTag.innerHTML = `<span>Interview Completed</span>`;
    interviewStatusTag.style.color = 'var(--text-secondary)';
    interviewStatusTag.style.backgroundColor = 'rgba(255,255,255,0.05)';
    interviewStatusTag.style.borderColor = 'var(--border-color)';

    completionBanner.classList.remove('hidden');
    chatInput.disabled = true;
    btnSendMessage.disabled = true;

    if (feedback) {
      renderEvaluationReport(feedback);
    }
  }

  function renderEvaluationReport(fb) {
    fbSummary.textContent = fb.summary || 'Candidate assessment completed.';

    fbStrengthsList.innerHTML = (fb.strengths || []).map(s => `<li>${escapeHtml(s)}</li>`).join('') || '<li>No specific strengths recorded.</li>';
    fbGapsList.innerHTML = (fb.gaps || []).map(g => `<li>${escapeHtml(g)}</li>`).join('') || '<li>No significant knowledge gaps identified.</li>';
    fbNextList.innerHTML = (fb.next || []).map(n => `<li>${escapeHtml(n)}</li>`).join('') || '<li>Continue standard curriculum modules.</li>';

    feedbackPanel.classList.remove('hidden');
    setTimeout(() => {
      feedbackPanel.scrollIntoView({ behavior: 'smooth' });
    }, 200);
  }

  function resetToSelection() {
    chatScreen.classList.remove('active');
    selectionScreen.classList.add('active');
    btnResetSession.classList.add('hidden');
    currentSessionId = null;
    isInterviewDone = false;
  }

  function setupEventListeners() {
    candidateSearch.addEventListener('input', (e) => {
      const q = e.target.value.toLowerCase().trim();
      const filtered = candidates.filter(c => {
        const name = (c.member?.name || '').toLowerCase();
        const role = (c.member?.jobRole || '').toLowerCase();
        return name.includes(q) || role.includes(q);
      });
      renderCandidateGrid(filtered);
    });

    btnStartInterview.addEventListener('click', startInterview);
    chatForm.addEventListener('submit', handleTurnSubmit);

    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleTurnSubmit();
      }
    });

    btnResetSession.addEventListener('click', resetToSelection);
    btnRestartAfterFeedback.addEventListener('click', resetToSelection);
  }

  init();
});

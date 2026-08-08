/* ==========================================================================
   AI Technical Interview Agent - iOS Minimalist Recruiter Dashboard Logic
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
  // Determine API Base URL dynamically
  const API_BASE = window.location.origin.includes('8000') || window.location.origin.includes('8001') || window.location.origin.includes('3000') || window.location.origin.includes('127.0.0.1') || window.location.origin.includes('localhost')
    ? `${window.location.protocol}//${window.location.hostname}:${window.location.port || '8000'}`
    : window.location.origin;

  const MIN_THINKING_MS = 1500; // Enforced 1.5 second minimum perceived AI thinking delay

  // Standard 8 Curriculum Modules — frontend display mapping
  // Module names MUST match the module names returned in currentTopicModule from the backend TURN_PLAN
  const CURRICULUM_MODULES = [
    { id: 'm1', name: 'Embeddings & Vector Search' },
    { id: 'm2', name: 'RAG & Retrieval Architecture' },
    { id: 'm3', name: 'LLM Core & Prompting' },
    { id: 'm4', name: 'Agentic AI & MCP' },
    { id: 'm5', name: 'Security & Guardrails' },
    { id: 'm6', name: 'Evaluation & Benchmarks' },
    { id: 'm7', name: 'Production Engineering' },
  ];

  // Timeline step labels — indexed by turn order (displayed in sidebar)
  const TIMELINE_STEPS = [
    'Interview Initialized',
    'Embeddings & Vector Search',
    'LLM Integration & Prompting',
    'Agents & Tool Use',
    'Security & Guardrails',
    'Evaluation & Benchmarks',
    'Production & Capstone',
  ];

  // State Management
  let candidates = [];
  let selectedCandidate = null;
  let currentSessionId = null;
  let currentTurnCount = 0;
  let maxTurns = 10;
  let isInterviewDone = false;
  let isSubmitting = false;
  let latestFeedbackData = null;

  // Live skill map: { moduleName: "Not Assessed" | "Strong" | "Good" | "Developing" | "Needs Attention" }
  let liveSkillMap = {};

  // Current topic module (from API, single source of truth for sidebar labels)
  let currentTopicModule = '';

  // DOM Handles
  const backendStatusBadge = document.getElementById('backend-status');
  const btnThemeToggle = document.getElementById('btn-theme-toggle');
  const btnResetSession = document.getElementById('btn-reset-session');
  
  // Selection Screen
  const selectionScreen = document.getElementById('selection-screen');
  const candidateGrid = document.getElementById('candidate-grid');
  const candidateSearch = document.getElementById('candidate-search');
  const inspectorPlaceholder = document.getElementById('inspector-placeholder');
  const inspectorContent = document.getElementById('inspector-content');
  
  // Inspector Fields
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

  // Assessment / Chat Viewport
  const chatScreen = document.getElementById('chat-screen');
  const chatCandidateAvatar = document.getElementById('chat-candidate-avatar');
  const chatCandidateName = document.getElementById('chat-candidate-name');
  const chatCandidateRole = document.getElementById('chat-candidate-role');
  const chatSessionId = document.getElementById('chat-session-id');
  const chatTurnCount = document.getElementById('chat-turn-count');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const difficultyVal = document.getElementById('difficulty-val');
  const insightFocus = document.getElementById('insight-focus');
  const insightCoverage = document.getElementById('insight-coverage');
  const skillCoverageGrid = document.getElementById('skill-coverage-grid');
  const interviewStatusTag = document.getElementById('interview-status-tag');
  
  const chatMessages = document.getElementById('chat-messages');
  const completionBanner = document.getElementById('completion-banner');
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const btnSendMessage = document.getElementById('btn-send-message');

  // Executive Feedback Report
  const feedbackPanel = document.getElementById('feedback-panel');
  const dispositionBadgeContainer = document.getElementById('disposition-badge-container');
  const fbSummary = document.getElementById('fb-summary');
  const fbStrengthsList = document.getElementById('fb-strengths-list');
  const fbGapsList = document.getElementById('fb-gaps-list');
  const fbNextList = document.getElementById('fb-next-list');
  const fbSkillProfileMatrix = document.getElementById('fb-skill-profile-matrix');
  const covQuestionsCount = document.getElementById('cov-questions-count');
  const covTopicsCount = document.getElementById('cov-topics-count');
  const covModulesCount = document.getElementById('cov-modules-count');
  const btnDownloadPdf = document.getElementById('btn-download-pdf');
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

  // Feature 3: Calculate Adaptive Difficulty Level
  function getAdaptiveDifficulty(cand) {
    const m = cand?.member || {};
    const exp = m.yearsExperience ?? 0;
    const role = (m.jobRole || '').toLowerCase();
    if (exp >= 5 || role.includes('lead') || role.includes('staff') || role.includes('senior')) {
      return 'Advanced';
    }
    return 'Intermediate';
  }

  // Initialize the live skill map with "Not Assessed" for every module
  function initLiveSkillMap() {
    liveSkillMap = {};
    CURRICULUM_MODULES.forEach(mod => {
      liveSkillMap[mod.name] = 'Not Assessed';
    });
  }

  // Apply a skill evaluation received from the backend to the live skill map.
  // Only updates the specific module affected by the current turn.
  function applySkillEvaluation(skillEvaluation) {
    if (!skillEvaluation || !skillEvaluation.module || !skillEvaluation.status) return;
    const { module, status } = skillEvaluation;

    // Only update if the module exists in our map
    if (module in liveSkillMap) {
      // Upgrade rule: never downgrade a skill if it's already been rated higher
      const currentStatus = liveSkillMap[module];
      const RANK = { 'Not Assessed': 0, 'Needs Attention': 1, 'Developing': 2, 'Good': 3, 'Strong': 4 };
      const currentRank = RANK[currentStatus] ?? 0;
      const newRank = RANK[status] ?? 0;
      // Take the higher rating if the module has been assessed multiple times
      // (But allow degradation if candidate gives a weak answer after a good one — take the LATEST)
      liveSkillMap[module] = status;
    }
  }

  // 1. Initial Health Check & Candidate Fetch
  async function init() {
    setupThemeToggle();
    await checkHealth();
    await fetchCandidates();
    setupEventListeners();
  }

  function setupThemeToggle() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);

    btnThemeToggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme') || 'light';
      const nextTheme = current === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', nextTheme);
      localStorage.setItem('theme', nextTheme);
    });
  }

  // Feature 5: Real Claude vs Mock Mode Status Indicator
  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        maxTurns = data.max_turns || 10;
        const modeText = data.mock_mode ? 'AI Interviewer — Demo Mode' : `AI Interviewer — Live Claude (${data.model})`;
        backendStatusBadge.querySelector('.status-text').textContent = modeText;
      }
    } catch (err) {
      console.warn('Health check note:', err);
      backendStatusBadge.querySelector('.status-text').textContent = 'AI Interviewer — Offline';
    }
  }

  async function fetchCandidates() {
    try {
      const res = await fetch(`${API_BASE}/api/candidates`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      candidates = data.candidates || [];
      renderCandidateGrid(candidates);
    } catch (err) {
      console.error('Failed to load candidates:', err);
      candidateGrid.innerHTML = `<div style="grid-column:1/-1;padding:24px;text-align:center;color:var(--error)">Unable to load candidate profiles from database.</div>`;
    }
  }

  // PART 3 — RECRUITER CANDIDATE SELECTION DASHBOARD
  function renderCandidateGrid(list) {
    if (!list || list.length === 0) {
      candidateGrid.innerHTML = `<div style="grid-column:1/-1;padding:24px;text-align:center;color:var(--text-secondary)">No candidates match search query.</div>`;
      return;
    }

    candidateGrid.innerHTML = list.map(c => {
      const m = c.member || {};
      const s = c.signals || {};
      const isSel = selectedCandidate && (m.id === selectedCandidate.member?.id);

      return `
        <div class="candidate-card ${isSel ? 'selected' : ''}" data-id="${m.id}">
          <div class="card-person-header">
            <div class="avatar-circle">${getInitials(m.name)}</div>
            <div class="person-info">
              <h3>${escapeHtml(m.name || 'Candidate')}</h3>
              <span class="role-tag">${escapeHtml(m.jobRole || 'Software Engineer')}</span>
            </div>
          </div>
          <div class="card-compact-meta">
            <div class="meta-field"><span class="lbl">Exp</span><span class="val">${m.yearsExperience ?? 0} Yrs</span></div>
            <div class="meta-field"><span class="lbl">Commits</span><span class="val">${s.commitDays ?? 0}d</span></div>
            <div class="meta-field"><span class="lbl">First-Try</span><span class="val">${s.missionsCompleted ? Math.round(((s.missionsFirstTry || 0) / s.missionsCompleted) * 100) : 0}%</span></div>
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

  // PART 4 — START ASSESSMENT SESSION
  async function startInterview() {
    if (!selectedCandidate || isSubmitting) return;

    isSubmitting = true;
    currentSessionId = generateSessionId();
    currentTurnCount = 0;
    isInterviewDone = false;
    latestFeedbackData = null;

    // Reset live skill map to "Not Assessed" for all modules
    initLiveSkillMap();
    currentTopicModule = '';

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
        throw new Error(errData.detail || `Server error ${res.status}`);
      }

      const response = await res.json();

      // Track the topic module for the first question
      if (response.currentTopicModule) {
        currentTopicModule = response.currentTopicModule;
      }

      setupChatView(response);
    } catch (err) {
      console.error('Start error:', err);
      alert(`Error starting assessment session: ${err.message}`);
    } finally {
      isSubmitting = false;
      btnStartInterview.disabled = false;
    }
  }

  function setupChatView(startResponse) {
    const m = selectedCandidate.member || {};

    chatCandidateAvatar.textContent = getInitials(m.name);
    chatCandidateName.textContent = m.name || 'Candidate';
    chatCandidateRole.textContent = `${m.jobRole || 'Engineer'} · ${m.yearsExperience ?? 0} years`;
    chatSessionId.textContent = currentSessionId;
    
    difficultyVal.textContent = getAdaptiveDifficulty(selectedCandidate);

    // At interview start: all skills = Not Assessed, turn = 0
    updateLiveInsightsUI(0, currentTopicModule);
    renderSkillCoverageMap();   // Shows all "Not Assessed"
    updateTimelineUI(0, currentTopicModule);

    interviewStatusTag.className = 'live-status-tag';
    interviewStatusTag.innerHTML = `<span class="status-dot"></span><span>Assessment Active</span>`;

    chatMessages.innerHTML = '';
    
    // Feature 1: Pass Assessment Focus Rationale
    appendMessage('ai', startResponse.reply, startResponse.assessmentFocus);

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

  // Update Insights UI — uses currentTopicModule from API (single source of truth)
  function updateLiveInsightsUI(turns, topicModule) {
    currentTurnCount = turns;
    chatTurnCount.textContent = `Turn ${turns} of ${maxTurns}`;
    const percent = Math.min(100, Math.round((turns / maxTurns) * 100));
    progressBarFill.style.width = `${percent}%`;

    // Use topic from API if available, else fall back to turns-based CURRICULUM_MODULES order
    const focusLabel = topicModule || (CURRICULUM_MODULES[Math.min(turns, CURRICULUM_MODULES.length - 1)]?.name || '');
    insightFocus.textContent = focusLabel || 'Embeddings & Vector Search';

    // Coverage = number of distinct modules that have been assessed (status != "Not Assessed")
    const assessedModules = Object.values(liveSkillMap).filter(s => s !== 'Not Assessed').length;
    const totalModules = CURRICULUM_MODULES.length;
    insightCoverage.textContent = `${assessedModules} / ${totalModules} Modules`;
  }

  // PART 7 — FEATURE #2: LIVE SKILL COVERAGE MAP
  // Renders the skill map purely from liveSkillMap (actual interview evidence).
  // At turn 0 (start), all are "Not Assessed". Only candidate answers change status.
  function renderSkillCoverageMap() {
    const STATUS_CONFIG = {
      'Not Assessed':   { cls: 'not-assessed',   dot: 'not-assessed',   label: 'Not Assessed' },
      'Strong':         { cls: 'strong',          dot: 'strong',         label: 'Strong' },
      'Good':           { cls: 'good',            dot: 'good',           label: 'Good' },
      'Developing':     { cls: 'developing',      dot: 'developing',     label: 'Developing' },
      'Needs Attention':{ cls: 'needs-attention', dot: 'needs-attention',label: 'Needs Attention' },
    };

    skillCoverageGrid.innerHTML = CURRICULUM_MODULES.map(mod => {
      const status = liveSkillMap[mod.name] || 'Not Assessed';
      const cfg = STATUS_CONFIG[status] || STATUS_CONFIG['Not Assessed'];

      return `
        <div class="skill-coverage-row">
          <span class="skill-dot ${cfg.dot}"></span>
          <span class="skill-coverage-name">${escapeHtml(mod.name)}</span>
          <span class="skill-status-pill ${cfg.cls}">${cfg.label}</span>
        </div>
      `;
    }).join('');
  }

  // Feature 3: SHOW ADAPTIVE INTERVIEW EVENTS IN TIMELINE
  // Uses currentTopicModule from API to keep timeline consistent with actual question.
  function updateTimelineUI(turns, topicModule) {
    const timelineContainer = document.getElementById('interview-timeline');
    if (!timelineContainer) return;

    // Map module name to timeline step index
    const MODULE_TO_STEP = {
      'Embeddings & Vector Search': 1,
      'RAG & Retrieval Architecture': 1,
      'LLM Core & Prompting': 2,
      'Agentic AI & MCP': 3,
      'Security & Guardrails': 4,
      'Evaluation & Benchmarks': 5,
      'Production Engineering': 6,
    };

    // Active step from module name (API-driven), fallback to turn-based
    let activeStep = 0;
    if (turns === 0) {
      activeStep = 0; // "Interview Initialized"
    } else if (topicModule && MODULE_TO_STEP[topicModule] !== undefined) {
      activeStep = MODULE_TO_STEP[topicModule];
    } else {
      activeStep = Math.min(turns, TIMELINE_STEPS.length - 1);
    }

    timelineContainer.innerHTML = TIMELINE_STEPS.map((stepLabel, idx) => {
      let cls = '';
      if (idx < activeStep) cls = 'completed';
      else if (idx === activeStep) cls = 'active';

      return `
        <div class="timeline-step ${cls}">
          <strong>${escapeHtml(stepLabel)}</strong>
        </div>
      `;
    }).join('');
  }

  // PART 5 — 1.5 SECOND THINKING EXPERIENCE (FRONTEND PERCEIVED DELAY)
  async function handleTurnSubmit(e) {
    if (e) e.preventDefault();
    if (isSubmitting || isInterviewDone) return;

    const messageText = chatInput.value.trim();
    if (!messageText) return;

    isSubmitting = true;

    // 1. Immediately append candidate response to chat
    appendMessage('candidate', messageText);

    // 2. Clear input & disable composer controls
    chatInput.value = '';
    chatInput.disabled = true;
    btnSendMessage.disabled = true;

    // 3. Render 3-dot thinking indicator bubble
    showThinkingIndicator();

    // 4. Start API call & measure elapsed duration
    const startTime = performance.now();

    try {
      const payload = { sessionId: currentSessionId, message: messageText };
      const res = await fetch(`${API_BASE}/api/interview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error ${res.status}`);
      }

      const data = await res.json();

      // Enforce 1.5s perceived thinking delay: remaining = max(0, 1500 - elapsed)
      const elapsedTime = performance.now() - startTime;
      const remainingDelay = Math.max(0, MIN_THINKING_MS - elapsedTime);

      if (remainingDelay > 0) {
        await new Promise(resolve => setTimeout(resolve, remainingDelay));
      }

      // 5. Remove thinking indicator
      removeThinkingIndicator();

      // 6. Apply skill evaluation from the answer just given
      if (data.skillEvaluation) {
        applySkillEvaluation(data.skillEvaluation);
      }

      // 7. Update turn count and topic module
      const newTurnCount = currentTurnCount + 1;
      const newTopicModule = data.currentTopicModule || currentTopicModule;
      currentTopicModule = newTopicModule;

      // 8. Update insights, skill map & timeline
      updateLiveInsightsUI(newTurnCount, newTopicModule);
      renderSkillCoverageMap();
      updateTimelineUI(newTurnCount, newTopicModule);

      // 9. Feature 1: Append AI interviewer reply with Assessment Focus
      appendMessage('ai', data.reply, data.assessmentFocus);

      // 10. Handle completion or re-enable input
      if (data.done) {
        handleInterviewCompleted(data.feedback);
      } else {
        chatInput.disabled = false;
        btnSendMessage.disabled = false;
        chatInput.focus();
      }

    } catch (err) {
      console.error('Turn submission error:', err);
      removeThinkingIndicator();
      appendErrorMessage(`Unable to process response (${err.message}). Please verify connection and try again.`);
      chatInput.disabled = false;
      btnSendMessage.disabled = false;
    } finally {
      isSubmitting = false;
    }
  }

  // 3-Dot Animated Thinking State
  function showThinkingIndicator() {
    const div = document.createElement('div');
    div.id = 'thinking-indicator-msg';
    div.className = 'thinking-row';
    div.innerHTML = `
      <div class="ai-avatar-badge" style="width:26px;height:26px;font-size:0.75rem">🤖</div>
      <div>
        <div class="thinking-bubble">
          <span class="thinking-dot"></span>
          <span class="thinking-dot"></span>
          <span class="thinking-dot"></span>
        </div>
        <span class="msg-meta">AI Interviewer is thinking...</span>
      </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
  }

  function removeThinkingIndicator() {
    const el = document.getElementById('thinking-indicator-msg');
    if (el) el.remove();
  }

  // Feature 1: Message Append Helper with Assessment Focus Indicator
  function appendMessage(role, content, assessmentFocus) {
    const isAi = role === 'ai';
    const div = document.createElement('div');
    div.className = `msg-row ${role}`;

    const avatarHtml = isAi 
      ? `<div class="ai-avatar-badge" style="width:26px;height:26px;font-size:0.75rem">🤖</div>` 
      : `<div class="avatar-circle md" style="width:26px;height:26px;font-size:0.75rem">${getInitials(selectedCandidate?.member?.name)}</div>`;

    let focusHtml = '';
    if (isAi && assessmentFocus && assessmentFocus.topic) {
      focusHtml = `
        <div class="assessment-focus-tag">
          <span class="focus-title">Assessment Focus: ${escapeHtml(assessmentFocus.topic)}</span>
          <span class="focus-reason">${escapeHtml(assessmentFocus.reason || '')}</span>
        </div>
      `;
    }

    div.innerHTML = `
      ${avatarHtml}
      <div>
        <div class="msg-bubble">
          ${escapeHtml(content)}
          ${focusHtml}
        </div>
        <span class="msg-meta">${isAi ? 'AI Interviewer' : (selectedCandidate?.member?.name || 'Candidate')} • ${getTimeString()}</span>
      </div>
    `;

    chatMessages.appendChild(div);
    scrollToBottom();
  }

  function appendErrorMessage(text) {
    const div = document.createElement('div');
    div.className = 'msg-row ai';
    div.innerHTML = `
      <div class="ai-avatar-badge" style="width:26px;height:26px;font-size:0.75rem">🤖</div>
      <div>
        <div class="msg-bubble" style="background-color:var(--error-bg);color:var(--error);border-color:var(--error-border)">${escapeHtml(text)}</div>
        <span class="msg-meta">System Alert</span>
      </div>
    `;
    chatMessages.appendChild(div);
    scrollToBottom();
  }

  // PART 9 — RECRUITER EXECUTIVE EVALUATION REPORT
  function handleInterviewCompleted(feedback) {
    isInterviewDone = true;
    latestFeedbackData = feedback;

    chatScreen.classList.add('interview-completed');

    interviewStatusTag.className = 'status-badge';
    interviewStatusTag.innerHTML = `<span>Assessment Concluded</span>`;

    const titleEl = document.getElementById('viewport-header-title');
    const subTitleEl = document.getElementById('viewport-header-subtitle');
    if (titleEl) titleEl.textContent = 'Candidate Evaluation Report';
    if (subTitleEl) subTitleEl.textContent = 'Recruiter Technical Assessment Brief';

    // Populate transcript drawer with complete conversation history
    const transcriptContainer = document.getElementById('transcript-container');
    if (transcriptContainer) {
      transcriptContainer.innerHTML = chatMessages.innerHTML;
    }

    const completedSidebar = document.getElementById('completed-sidebar-block');
    if (completedSidebar) {
      completedSidebar.classList.remove('hidden');
    }

    completionBanner.classList.remove('hidden');
    chatInput.disabled = true;
    btnSendMessage.disabled = true;

    if (feedback) {
      renderEvaluationReport(feedback);
    }
  }

  // Feature 2: Render Technical Skill Profile Matrix
  function renderEvaluationReport(fb) {
    let dispositionText = fb.disposition || 'Consider';
    let dispositionClass = 'consider';

    if (dispositionText === 'Strong Fit') {
      dispositionClass = 'fit';
    } else if (dispositionText === 'Needs Development') {
      dispositionClass = 'remediate';
    } else {
      dispositionClass = 'consider';
    }

    const badgeHtml = `
      <span class="disposition-badge ${dispositionClass}" title="Assessment-based recommendation derived from candidate performance">
        <span>Disposition:</span> <strong>${dispositionText}</strong>
      </span>
    `;

    dispositionBadgeContainer.innerHTML = badgeHtml;

    const sidebarDispContainer = document.getElementById('sidebar-disposition-container');
    if (sidebarDispContainer) {
      sidebarDispContainer.innerHTML = badgeHtml;
    }

    fbSummary.textContent = fb.summary || 'Candidate technical assessment completed across all curriculum modules.';

    // Strengths, Gaps, Next Lists
    fbStrengthsList.innerHTML = (fb.strengths || []).map(s => `<li>${escapeHtml(s)}</li>`).join('') || '<li>Demonstrated clear technical communication.</li>';
    fbGapsList.innerHTML = (fb.gaps || []).map(g => `<li>${escapeHtml(g)}</li>`).join('') || '<li>No major critical knowledge gaps identified.</li>';
    fbNextList.innerHTML = (fb.next || []).map(n => `<li>${escapeHtml(n)}</li>`).join('') || '<li>Proceed with final team interviews.</li>';

    // Feature 2: Technical Skill Profile Matrix
    // Use backend-provided skillProfile if available; otherwise derive from live interview evidence
    const skillProfileObj = fb.skillProfile || deriveSkillProfileFromLiveMap();
    
    fbSkillProfileMatrix.innerHTML = Object.entries(skillProfileObj).map(([skillName, rating]) => {
      const cls = rating.toLowerCase().replace(/\s+/g, '-');
      return `
        <div class="skill-item">
          <span class="skill-name">${escapeHtml(skillName)}</span>
          <span class="skill-pill ${cls}">${escapeHtml(rating)}</span>
        </div>
      `;
    }).join('');

    // Feature 6: Interview Coverage Metrics
    const assessedModules = Object.values(liveSkillMap).filter(s => s !== 'Not Assessed').length;
    covQuestionsCount.textContent = maxTurns;
    covTopicsCount.textContent = assessedModules > 0 ? assessedModules : '—';
    covModulesCount.textContent = `${assessedModules} / ${CURRICULUM_MODULES.length}`;

    feedbackPanel.classList.remove('hidden');
    setTimeout(() => {
      feedbackPanel.scrollIntoView({ behavior: 'smooth' });
    }, 200);
  }

  // Derive a skill profile object from the current liveSkillMap for the PDF/report
  function deriveSkillProfileFromLiveMap() {
    const profile = {};
    CURRICULUM_MODULES.forEach(mod => {
      profile[mod.name] = liveSkillMap[mod.name] || 'Not Assessed';
    });
    return profile;
  }

  // Feature 4: DOWNLOADABLE ASSESSMENT REPORT PDF GENERATOR
  // Fixed for one-page output: tighter spacing, combined coverage metrics, print media query
  function downloadAssessmentReportPDF() {
    if (!selectedCandidate) return;

    const m = selectedCandidate.member || {};
    const candName = m.name || 'Candidate';
    const candRole = m.jobRole || 'Software Engineer';
    const candExp = `${m.yearsExperience ?? 0} years`;
    const currentDate = new Date().toLocaleString([], { dateStyle: 'long', timeStyle: 'short' });

    const summaryText = fbSummary.textContent;
    const strengthsItems = Array.from(fbStrengthsList.querySelectorAll('li')).map(li => li.textContent);
    const gapsItems = Array.from(fbGapsList.querySelectorAll('li')).map(li => li.textContent);
    const nextItems = Array.from(fbNextList.querySelectorAll('li')).map(li => li.textContent);

    const dispositionElement = dispositionBadgeContainer.querySelector('strong');
    const disposition = dispositionElement ? dispositionElement.textContent : 'Consider';

    const skillProfileObj = latestFeedbackData?.skillProfile || deriveSkillProfileFromLiveMap();

    // Build skill profile as inline rows (no separate section, integrated into metrics)
    const skillRowsHtml = Object.entries(skillProfileObj)
      .map(([name, rating]) => {
        const ratingColor = rating === 'Strong' ? '#1a7f37' : rating === 'Good' ? '#0550ae' : rating === 'Developing' ? '#9a6700' : rating === 'Needs Attention' ? '#cf222e' : '#57606a';
        return `<tr>
          <td style="padding:5px 8px;border-bottom:1px solid #eee;font-size:11px;">${escapeHtml(name)}</td>
          <td style="padding:5px 8px;border-bottom:1px solid #eee;font-weight:700;text-align:right;font-size:11px;color:${ratingColor};">${escapeHtml(rating)}</td>
        </tr>`;
      })
      .join('');

    const assessedModules = Object.values(liveSkillMap).filter(s => s !== 'Not Assessed').length;

    const printWindow = window.open('', '_blank');
    if (!printWindow) {
      alert('Please allow popups to download the Assessment Report PDF.');
      return;
    }

    const reportHtml = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>AI Technical Assessment Report — ${escapeHtml(candName)}</title>
        <style>
          * { box-sizing: border-box; margin: 0; padding: 0; }
          body {
            font-family: 'Inter', -apple-system, sans-serif;
            color: #1D1D1F;
            font-size: 11px;
            line-height: 1.45;
            padding: 28px 36px;
            max-width: 780px;
            margin: 0 auto;
          }
          .report-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            border-bottom: 2px solid #007AFF;
            padding-bottom: 10px;
            margin-bottom: 14px;
          }
          .report-title { font-size: 16px; font-weight: 700; letter-spacing: -0.01em; }
          .report-subtitle { font-size: 10px; color: #6E6E73; margin-top: 2px; }
          .report-date { font-size: 10px; color: #6E6E73; text-align: right; }
          .meta-row {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            margin-bottom: 14px;
          }
          .meta-cell {
            background: #F5F5F7;
            border-radius: 6px;
            padding: 7px 10px;
          }
          .meta-label { font-size: 9px; color: #6E6E73; text-transform: uppercase; letter-spacing: 0.04em; }
          .meta-val { font-size: 12px; font-weight: 700; margin-top: 2px; }
          .disposition-pill {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 10px;
            background: rgba(0,122,255,0.1);
            color: #007AFF;
            border: 1px solid rgba(0,122,255,0.25);
            margin-top: 2px;
          }
          .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; }
          .section { margin-bottom: 12px; }
          .section-title {
            font-size: 9px;
            font-weight: 700;
            color: #6E6E73;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            border-bottom: 1px solid #E5E5EA;
            padding-bottom: 3px;
            margin-bottom: 7px;
          }
          .summary-box {
            background: #F5F5F7;
            border-radius: 6px;
            padding: 9px 12px;
            font-size: 11px;
            line-height: 1.5;
            margin-bottom: 12px;
          }
          ul { padding-left: 14px; }
          li { margin-bottom: 4px; font-size: 10px; line-height: 1.4; }
          table { width: 100%; border-collapse: collapse; }
          .coverage-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 8px;
          }
          .coverage-box {
            background: #F5F5F7;
            border-radius: 5px;
            padding: 6px 8px;
            text-align: center;
          }
          .coverage-num { font-size: 15px; font-weight: 700; }
          .coverage-lbl { font-size: 9px; color: #6E6E73; margin-top: 1px; }
          @media print {
            body { padding: 18px 28px; }
            @page { margin: 0.6cm; size: A4 portrait; }
          }
          /* Prevent page breaks inside sections */
          .section, .two-col, .summary-box, .meta-row { page-break-inside: avoid; }
        </style>
      </head>
      <body>
        <div class="report-header">
          <div>
            <div class="report-title">AI TECHNICAL ASSESSMENT REPORT</div>
            <div class="report-subtitle">Enterprise Candidate Evaluation Brief</div>
          </div>
          <div class="report-date">Date: ${escapeHtml(currentDate)}</div>
        </div>

        <div class="meta-row">
          <div class="meta-cell">
            <div class="meta-label">Candidate</div>
            <div class="meta-val">${escapeHtml(candName)}</div>
          </div>
          <div class="meta-cell">
            <div class="meta-label">Role</div>
            <div class="meta-val">${escapeHtml(candRole)}</div>
          </div>
          <div class="meta-cell">
            <div class="meta-label">Experience</div>
            <div class="meta-val">${escapeHtml(candExp)}</div>
          </div>
          <div class="meta-cell">
            <div class="meta-label">Recommendation</div>
            <div class="disposition-pill">${escapeHtml(disposition)}</div>
          </div>
        </div>

        <div class="summary-box">
          <div class="section-title" style="margin-bottom:5px;">Executive Summary</div>
          ${escapeHtml(summaryText)}
        </div>

        <div class="two-col">
          <div class="section">
            <div class="section-title">Technical Strengths</div>
            <ul>${strengthsItems.map(s => `<li>${escapeHtml(s)}</li>`).join('')}</ul>
          </div>
          <div class="section">
            <div class="section-title">Knowledge Gaps</div>
            <ul>${gapsItems.map(g => `<li>${escapeHtml(g)}</li>`).join('')}</ul>
          </div>
        </div>

        <div class="two-col">
          <div class="section">
            <div class="section-title">Technical Skill Profile</div>
            <table>${skillRowsHtml}</table>
          </div>
          <div>
            <div class="section" style="margin-bottom:10px;">
              <div class="section-title">Recommended Next Steps</div>
              <ul>${nextItems.map(n => `<li>${escapeHtml(n)}</li>`).join('')}</ul>
            </div>
            <div class="section">
              <div class="section-title">Interview Coverage</div>
              <div class="coverage-row">
                <div class="coverage-box">
                  <div class="coverage-num">${maxTurns}</div>
                  <div class="coverage-lbl">Questions Asked</div>
                </div>
                <div class="coverage-box">
                  <div class="coverage-num">${assessedModules}</div>
                  <div class="coverage-lbl">Modules Assessed</div>
                </div>
                <div class="coverage-box">
                  <div class="coverage-num">${assessedModules}/${CURRICULUM_MODULES.length}</div>
                  <div class="coverage-lbl">Curriculum Coverage</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </body>
      </html>
    `;

    printWindow.document.write(reportHtml);
    printWindow.document.close();
    setTimeout(() => {
      printWindow.focus();
      printWindow.print();
    }, 350);
  }

  function resetToSelection() {
    chatScreen.classList.remove('active');
    chatScreen.classList.remove('interview-completed');
    selectionScreen.classList.add('active');
    btnResetSession.classList.add('hidden');

    const titleEl = document.getElementById('viewport-header-title');
    const subTitleEl = document.getElementById('viewport-header-subtitle');
    if (titleEl) titleEl.textContent = 'AI Technical Interviewer';
    if (subTitleEl) subTitleEl.textContent = 'Adaptive Technical Assessment';

    const completedSidebar = document.getElementById('completed-sidebar-block');
    if (completedSidebar) completedSidebar.classList.add('hidden');

    currentSessionId = null;
    isInterviewDone = false;
    latestFeedbackData = null;
    initLiveSkillMap();
    currentTopicModule = '';
  }

  // PART 14 — INPUT SHORTCUTS & EVENT LISTENERS
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

    // Enter = Send, Shift + Enter = Newline
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleTurnSubmit();
      }
    });

    btnDownloadPdf.addEventListener('click', downloadAssessmentReportPDF);
    btnResetSession.addEventListener('click', resetToSelection);
    btnRestartAfterFeedback.addEventListener('click', resetToSelection);
  }

  init();
});

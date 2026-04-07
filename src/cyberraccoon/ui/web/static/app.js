/**
 * CyberRaccoon — Alpine.js application logic.
 *
 * Single function that returns the Alpine.js data object.
 * All API calls go through fetch(); real-time events via WebSocket.
 */

function cyberRaccoon() {
    return {
        // --- Tab ---
        tab: 'task',

        // --- Task ---
        taskGoal: '',
        taskRunning: false,
        steps: [],
        screenshot: null,
        taskResult: null,
        selectedStep: null,  // unique key string (e.g. "3-2"), null = follow latest
        stepDetailTab: 'detail',
        totalInputTokens: 0,
        totalOutputTokens: 0,
        totalCacheReadTokens: 0,
        totalCacheCreationTokens: 0,

        // --- Config ---
        config: null,

        // --- Connection (per-module) ---
        captureReady: false,
        executorReady: false,
        connectingCapture: false,
        connectingExecutor: false,
        captureError: '',
        executorError: '',
        captureDevice: '',
        executorDevice: '',

        // --- Status ---
        statusData: {},

        // --- Debug / Logs ---
        logs: [],
        logFilter: {
            level: '',
            module: '',
            autoScroll: true,
        },
        maxLogs: 500,

        // --- Workflow Plan ---
        workflowPlan: null,       // {steps: [{number, goal, status}], task_goal}
        workflowActive: false,
        planPending: false,       // true when plan is ready but not yet approved
        planExecuting: false,     // true when plan is being executed
        planPaused: false,        // true when task is paused during execution
        pauseRequested: false,    // true after Pause clicked, before task_paused received (optimistic UI only)
        escalationPending: false, // true when escalation needs user action
        escalationReason: '',     // why the agent escalated
        currentWorkflowStep: 0,  // which workflow step is currently executing
        selectedWorkflowStep: null, // which step the user clicked to filter actions (null = show all)

        // --- Plan Discussion Chat (Phase 4) ---
        chatHistory: [],          // [{role: 'user'|'assistant', text: string}, ...]
        chatInput: '',
        chatInFlight: false,
        chatError: '',

        // --- Plan Modification (Phase 5 — DISCUSS-03/04/06) ---
        planMode: 'ask',               // 'ask' | 'modify'
        planPreview: null,             // null | { steps: [...], summary: string }
        previousSteps: null,           // null | snapshot of steps at preview entry
        editingStep: null,             // null | step.number being inline-edited
        editingDraft: '',              // text buffer during inline edit
        deleteConfirmStep: null,       // null | step.number awaiting delete confirmation
        editedStepNumbers: [],         // array of step.number manually edited or added
        planVersion: 0,                // server plan_version counter [REVIEWS HIGH-1]
        _pendingAddFocus: false,       // internal: flag to auto-enter edit mode after add

        // --- Skills ---
        skills: [],
        selectedSkill: null,
        skillContent: '',
        skillSource: '',
        editingSkill: false,
        editSkillName: '',
        editSkillContent: '',
        isNewSkill: false,
        activeSkills: [],

        // --- In-flight abort controllers ---
        _captureAbort: null,
        _executorAbort: null,

        // --- WebSocket ---
        ws: null,
        wsConnected: false,
        _wsReconnectTimer: null,
        _wsPingTimer: null,

        // ================================================================
        // Computed-like properties
        // ================================================================

        get modulesReady() {
            return this.captureReady && this.executorReady;
        },

        get filteredSteps() {
            if (!this.workflowActive || this.selectedWorkflowStep === null) {
                return this.steps;
            }
            return this.steps.filter(s => s._workflowStep === this.selectedWorkflowStep);
        },

        get activeStep() {
            if (this.selectedStep !== null) {
                const found = this.steps.find(s => this._stepKey(s) === this.selectedStep);
                if (found) return found;
            }
            return this.steps.length > 0 ? this.steps[this.steps.length - 1] : null;
        },

        // ================================================================
        // Initialisation
        // ================================================================

        async init() {
            await this.loadConfig();
            if (this.config && this.config.agent && Array.isArray(this.config.agent.skills)) {
                this.activeSkills = [...this.config.agent.skills];
            }
            await this.refreshStatus();
            await this.loadLogs();
            this.connectWebSocket();
        },

        // ================================================================
        // WebSocket
        // ================================================================

        connectWebSocket() {
            // Detach old handlers before closing to prevent stale onclose
            // from setting wsConnected=false after the new socket connects
            if (this.ws) {
                this.ws.onopen = null;
                this.ws.onclose = null;
                this.ws.onerror = null;
                this.ws.onmessage = null;
                try { this.ws.close(); } catch (_) {}
            }

            if (this._wsPingTimer) {
                clearInterval(this._wsPingTimer);
                this._wsPingTimer = null;
            }

            const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const url = `${proto}//${location.host}/ws`;

            try {
                this.ws = new WebSocket(url);
            } catch (e) {
                console.warn('WebSocket creation failed:', e);
                this._scheduleReconnect();
                return;
            }

            this.ws.onopen = () => {
                this.wsConnected = true;
                console.log('WebSocket connected');
                // Keepalive ping every 15 seconds
                this._wsPingTimer = setInterval(() => {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify({ action: 'ping' }));
                    }
                }, 15000);
            };

            this.ws.onclose = () => {
                this.wsConnected = false;
                if (this._wsPingTimer) {
                    clearInterval(this._wsPingTimer);
                    this._wsPingTimer = null;
                }
                this._scheduleReconnect();
            };

            this.ws.onerror = () => {
                this.wsConnected = false;
            };

            this.ws.onmessage = (evt) => {
                try {
                    const msg = JSON.parse(evt.data);
                    this.handleEvent(msg);
                } catch (e) {
                    console.warn('WS message parse error:', e);
                }
            };
        },

        _scheduleReconnect() {
            if (this._wsReconnectTimer) return;
            this._wsReconnectTimer = setTimeout(() => {
                this._wsReconnectTimer = null;
                this.connectWebSocket();
            }, 3000);
        },

        // ================================================================
        // Event handling (from WebSocket)
        // ================================================================

        handleEvent(msg) {
            const { event, data } = msg;

            switch (event) {
                case 'task_started':
                    this.taskRunning = true;
                    this.taskResult = null;
                    this.steps = [];
                    this.screenshot = null;
                    this.selectedStep = null;
                    this.totalInputTokens = 0;
                    this.totalOutputTokens = 0;
                    this.totalCacheReadTokens = 0;
                    this.totalCacheCreationTokens = 0;
                    this.workflowPlan = null;
                    this.workflowActive = false;
                    this.planPending = false;
                    this.planExecuting = false;
                    this.planPaused = false;
                    this.pauseRequested = false;
                    this.escalationPending = false;
                    this.escalationReason = '';
                    this.currentWorkflowStep = 0;
                    this.selectedWorkflowStep = null;
                    this.chatHistory = [];
                    this.chatInput = '';
                    this.chatInFlight = false;
                    this.chatError = '';
                    // Phase 5: reset all plan-modification state on new task
                    this.planMode = 'ask';
                    this.planPreview = null;
                    this.previousSteps = null;
                    this.editingStep = null;
                    this.editingDraft = '';
                    this.deleteConfirmStep = null;
                    this.editedStepNumbers = [];
                    this.planVersion = 0;
                    this._pendingAddFocus = false;
                    break;

                case 'workflow_event':
                    this.handleWorkflowEvent(data);
                    break;

                case 'task_step':
                    // Tag with workflow step number for filtering
                    if (this.workflowActive) {
                        data._workflowStep = this.currentWorkflowStep;
                    }
                    this.steps.push(data);
                    if (data.total_input_tokens !== undefined) {
                        this.totalInputTokens = data.total_input_tokens;
                        this.totalOutputTokens = data.total_output_tokens;
                        this.totalCacheReadTokens = data.total_cache_read_tokens ?? 0;
                        this.totalCacheCreationTokens = data.total_cache_creation_tokens ?? 0;
                    }
                    // Only update screenshot if no step is explicitly selected (follow-latest)
                    if (this.selectedStep === null && data.screenshot_base64) {
                        this.screenshot = data.screenshot_base64;
                    }
                    this.$nextTick(() => {
                        const el = this.$refs.stepLog;
                        if (el) el.scrollTop = el.scrollHeight;
                    });
                    break;

                case 'task_finished':
                    this.taskRunning = false;
                    this.planPaused = false;
                    this.pauseRequested = false;
                    this.taskResult = data;
                    // Ensure final screenshot is shown (guard against missed task_step)
                    if (this.selectedStep === null && this.steps.length > 0) {
                        const lastStep = this.steps[this.steps.length - 1];
                        if (lastStep.screenshot_base64) {
                            this.screenshot = lastStep.screenshot_base64;
                        }
                    }
                    break;

                case 'config_changed':
                    this.loadConfig();
                    break;

                case 'capture_ready':
                    this.captureReady = true;
                    this.captureError = '';
                    if (data && data.device) this.captureDevice = data.device;
                    break;

                case 'capture_closed':
                    this.captureReady = false;
                    this.captureDevice = '';
                    break;

                case 'executor_ready':
                    this.executorReady = true;
                    this.executorError = '';
                    if (data && data.device) this.executorDevice = data.device;
                    break;

                case 'executor_closed':
                    this.executorReady = false;
                    this.executorDevice = '';
                    break;

                case 'modules_ready':
                    this.captureReady = true;
                    this.executorReady = true;
                    break;

                case 'modules_closed':
                    // Individual closed events already handle the specific module.
                    // Don't blindly reset both — just refresh from server.
                    this.refreshStatus();
                    break;

                case 'log_message':
                    if (data && data.message) {
                        this.logs.push(data.message);
                        if (this.logs.length > this.maxLogs) {
                            this.logs.splice(0, this.logs.length - this.maxLogs);
                        }
                        if (this.logFilter.autoScroll) {
                            this.$nextTick(() => {
                                const el = this.$refs.logContainer;
                                if (el) el.scrollTop = el.scrollHeight;
                            });
                        }
                    }
                    break;

                case 'pong':
                    break;

                default:
                    console.log('Unknown WS event:', event, data);
            }
        },

        // ================================================================
        // Workflow event handling
        // ================================================================

        handleWorkflowEvent(data) {
            switch (data.type) {
                case 'plan_ready':
                    this.workflowActive = true;
                    this.planPending = true;
                    this.planExecuting = false;
                    this.workflowPlan = {
                        task_goal: data.task_goal,
                        steps: data.steps.map(s => ({
                            number: s.number,
                            goal: s.goal,
                            reboot_expected: s.reboot_expected,
                            expected_actions: s.expected_actions,
                            expected_outcome: s.expected_outcome || '',
                            status: 'pending',
                        })),
                    };
                    // Reset chat state on every new plan (D-03)
                    this.chatHistory = [];
                    this.chatInput = '';
                    this.chatInFlight = false;
                    this.chatError = '';
                    // Phase 5: initialize plan modification state
                    this.planPreview = null;
                    this.previousSteps = null;
                    this.editedStepNumbers = [];
                    this.planVersion = 0;  // [REVIEWS HIGH-1]
                    this.editingStep = null;
                    this.editingDraft = '';
                    this.deleteConfirmStep = null;
                    break;

                case 'step_start':
                    this.planPending = false;
                    this.planExecuting = true;
                    this.currentWorkflowStep = data.step_number;
                    this.selectedWorkflowStep = data.step_number;
                    if (this.workflowPlan) {
                        const step = this.workflowPlan.steps.find(
                            s => s.number === data.step_number
                        );
                        if (step) step.status = 'running';
                    }
                    this.selectedStep = null;
                    break;

                case 'step_done':
                    if (this.workflowPlan) {
                        const step = this.workflowPlan.steps.find(
                            s => s.number === data.step_number
                        );
                        if (step) {
                            step.status = 'done';
                            step.actions_used = data.actions_used;
                            if (data.expected_actions !== undefined) {
                                step.expected_actions = data.expected_actions;
                            }
                        }
                    }
                    break;

                case 'reboot_transition':
                    if (this.workflowPlan) {
                        const step = this.workflowPlan.steps.find(
                            s => s.number === data.step_number
                        );
                        if (step) step.status = 'rebooting';
                    }
                    break;

                case 'replanned':
                    if (this.workflowPlan) {
                        const completedCount = data.steps_completed || 0;
                        const kept = this.workflowPlan.steps.slice(0, completedCount);
                        const newSteps = data.new_steps.map(s => ({
                            number: s.number,
                            goal: s.goal,
                            reboot_expected: s.reboot_expected || false,
                            expected_actions: s.expected_actions,
                            expected_outcome: s.expected_outcome || '',
                            status: 'pending',
                        }));
                        this.workflowPlan.steps = [...kept, ...newSteps];
                    }
                    break;

                case 'escalate':
                    this.escalationPending = true;
                    this.planExecuting = false;
                    this.pauseRequested = false;
                    this.escalationReason = data.reason || 'Human intervention required';
                    if (this.workflowPlan) {
                        const step = this.workflowPlan.steps.find(
                            s => s.number === data.step_number
                        );
                        if (step) step.status = 'escalated';
                    }
                    break;

                case 'escalation_resolved':
                    this.escalationPending = false;
                    this.escalationReason = '';
                    this.planExecuting = true;
                    break;

                case 'plan_modification_proposed':
                    // Phase 5 DISCUSS-03: LLM rewrite proposed, enter preview state
                    this.planPreview = {
                        steps: Array.isArray(data.proposed_steps) ? data.proposed_steps : [],
                        summary: data.summary || '',
                    };
                    // Snapshot current live steps for diff rendering
                    this.previousSteps = this.workflowPlan
                        ? JSON.parse(JSON.stringify(this.workflowPlan.steps))
                        : [];
                    this.chatInFlight = false;
                    // Scroll preview banner into view
                    this.$nextTick(() => {
                        const banner = document.querySelector('.plan-preview-banner');
                        if (banner && banner.scrollIntoView) {
                            banner.scrollIntoView({ behavior: 'smooth', block: 'start' });
                        }
                    });
                    break;

                case 'plan_modified':
                    // Phase 5 DISCUSS-03/04: plan state committed (accept_rewrite/manual_edit/add/delete)
                    // [REVIEWS HIGH-1] Stale-event guard: drop events with a version
                    // older than what we already have (can happen if events arrive
                    // out of order after rapid edits).
                    if (typeof data.plan_version === 'number' && data.plan_version < this.planVersion) {
                        console.warn('Dropping stale plan_modified event (version', data.plan_version, '< current', this.planVersion, ')');
                        break;
                    }
                    if (typeof data.plan_version === 'number') {
                        this.planVersion = data.plan_version;
                    }
                    if (this.workflowPlan && Array.isArray(data.steps)) {
                        this.workflowPlan.steps = data.steps.map(s => ({
                            number: s.number,
                            goal: s.goal,
                            status: 'pending',
                            reboot_expected: s.reboot_expected,
                            expected_actions: s.expected_actions,
                            expected_outcome: s.expected_outcome,
                        }));
                    }
                    this.editedStepNumbers = Array.isArray(data.edited_step_numbers)
                        ? data.edited_step_numbers
                        : [];
                    // Exiting preview on accept_rewrite
                    if (data.reason === 'accept_rewrite') {
                        this.planPreview = null;
                    }
                    // Clear editing state if the edited step was deleted or renumbered away
                    if (this.editingStep !== null && this.workflowPlan) {
                        const stillExists = this.workflowPlan.steps.some(
                            s => s.number === this.editingStep,
                        );
                        if (!stillExists) {
                            this.editingStep = null;
                            this.editingDraft = '';
                        }
                    }
                    // Auto-enter edit mode on the newly-added step
                    if (data.reason === 'add' && this._pendingAddFocus && this.workflowPlan) {
                        const lastStep = this.workflowPlan.steps[this.workflowPlan.steps.length - 1];
                        if (lastStep) {
                            this.beginEditStep(lastStep);
                        }
                        this._pendingAddFocus = false;
                    }
                    break;

                case 'plan_rewrite_no_change':
                    // Phase 5 DISCUSS-03: LLM declined to rewrite, surface message in chat
                    this.chatHistory.push({
                        role: 'assistant',
                        text: (data.message || '') + '\n\nSwitch to Ask mode to discuss the plan.',
                    });
                    this.chatInFlight = false;
                    this.$nextTick(() => {
                        const el = this.$refs.chatMessages;
                        if (el) el.scrollTop = el.scrollHeight;
                    });
                    break;

                case 'plan_rewrite_discarded':
                    // Phase 5 DISCUSS-03: user discarded the preview
                    this.planPreview = null;
                    break;

                case 'plan_rewrite_error':
                    // Phase 5 DISCUSS-03: backend signaled an error during rewrite
                    this.chatError = data.message || 'The model returned an unexpected response. Try rephrasing your request.';
                    this.chatInFlight = false;
                    break;

                case 'task_paused':
                    this.planPaused = true;
                    this.planExecuting = false;
                    this.pauseRequested = false;
                    // Repopulate plan panel with step statuses from server
                    // Server payload is the single source of truth (review MEDIUM-5)
                    if (this.workflowPlan && data.steps) {
                        this.workflowPlan.steps = data.steps.map(s => ({
                            number: s.number,
                            goal: s.goal,
                            status: s.status,  // 'done', 'partial', 'pending'
                            reboot_expected: s.reboot_expected,
                            expected_actions: s.expected_actions,
                            expected_outcome: s.expected_outcome || '',
                            actions_used: s.actions_used,
                        }));
                        this.workflowPlan.steps_completed = data.steps_completed || 0;
                    }
                    if (data.screenshot_base64) {
                        this.screenshot = data.screenshot_base64;
                    }
                    break;

                case 'task_resumed':
                    this.planPaused = false;
                    this.planExecuting = true;
                    break;

                case 'workflow_done':
                    this.planExecuting = false;
                    if (this.workflowPlan) {
                        this.workflowPlan.steps.forEach(s => {
                            if (s.status === 'running') s.status = 'done';
                        });
                    }
                    break;
            }
        },

        // ================================================================
        // Connection API — per-module
        // ================================================================

        async connectCapture() {
            this.connectingCapture = true;
            this.captureError = '';
            const abort = new AbortController();
            this._captureAbort = abort;
            try {
                const resp = await fetch('/api/capture/connect', {
                    method: 'POST',
                    signal: abort.signal,
                });
                const result = await resp.json();
                if (result.status === 'error') {
                    this.captureError = result.message;
                } else {
                    this.captureReady = true;
                    if (result.device) this.captureDevice = result.device;
                    if (result.image) {
                        this.screenshot = result.image;
                    }
                }
            } catch (e) {
                if (e.name === 'AbortError') return; // cancelled — ignore
                this.captureError = 'Network error: ' + e.message;
            } finally {
                this._captureAbort = null;
                this.connectingCapture = false;
            }
        },

        async cancelConnectCapture() {
            if (this._captureAbort) {
                this._captureAbort.abort();
                this._captureAbort = null;
            }
            this.connectingCapture = false;
            // Tell backend to kill the blocking operation
            try {
                await fetch('/api/capture/disconnect', { method: 'POST' });
            } catch (_) {}
        },

        async disconnectCapture() {
            try {
                await fetch('/api/capture/disconnect', { method: 'POST' });
                this.captureReady = false;
                this.captureDevice = '';
            } catch (e) {
                console.error('Disconnect capture error:', e);
            }
        },

        async connectExecutor() {
            this.connectingExecutor = true;
            this.executorError = '';
            const abort = new AbortController();
            this._executorAbort = abort;
            try {
                const resp = await fetch('/api/executor/connect', {
                    method: 'POST',
                    signal: abort.signal,
                });
                const result = await resp.json();
                if (result.status === 'error') {
                    this.executorError = result.message;
                } else {
                    this.executorReady = true;
                    if (result.device) this.executorDevice = result.device;
                }
            } catch (e) {
                if (e.name === 'AbortError') return; // cancelled — ignore
                this.executorError = 'Network error: ' + e.message;
            } finally {
                this._executorAbort = null;
                this.connectingExecutor = false;
            }
        },

        async cancelConnectExecutor() {
            if (this._executorAbort) {
                this._executorAbort.abort();
                this._executorAbort = null;
            }
            this.connectingExecutor = false;
            // Tell backend to kill the blocking operation
            try {
                await fetch('/api/executor/disconnect', { method: 'POST' });
            } catch (_) {}
        },

        async disconnectExecutor() {
            try {
                await fetch('/api/executor/disconnect', { method: 'POST' });
                this.executorReady = false;
                this.executorDevice = '';
            } catch (e) {
                console.error('Disconnect executor error:', e);
            }
        },

        // ================================================================
        // Task API
        // ================================================================

        async startTask() {
            const goal = this.taskGoal.trim();
            if (!goal) return;

            // Auto-connect both if not already connected
            if (!this.captureReady || !this.executorReady) {
                const promises = [];
                if (!this.captureReady) promises.push(this.connectCapture());
                if (!this.executorReady) promises.push(this.connectExecutor());
                await Promise.all(promises);
                if (!this.captureReady || !this.executorReady) return; // connect failed
            }

            try {
                const resp = await fetch('/api/task', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ goal }),
                });
                const result = await resp.json();
                if (result.status === 'error') {
                    this.captureError = result.message;
                }
            } catch (e) {
                this.captureError = 'Network error: ' + e.message;
            }
        },

        async abortTask() {
            try {
                await fetch('/api/task', { method: 'DELETE' });
            } catch (e) {
                console.error('Abort error:', e);
            }
        },

        async pauseTask() {
            this.pauseRequested = true;
            try {
                const resp = await fetch('/api/task/pause', { method: 'POST' });
                if (!resp.ok) {
                    this.pauseRequested = false;
                    this.showError('Failed to pause task. The step may have already completed.');
                }
            } catch (err) {
                this.pauseRequested = false;
                this.showError('Failed to pause task. The step may have already completed.');
            }
        },

        async resumeTask() {
            try {
                const resp = await fetch('/api/task/resume', { method: 'POST' });
                if (!resp.ok) {
                    this.showError('Failed to resume task. Try again or abort.');
                }
            } catch (err) {
                this.showError('Failed to resume task. Try again or abort.');
            }
        },

        async cancelPausedTask() {
            try {
                const resp = await fetch('/api/task/cancel', { method: 'POST' });
                if (!resp.ok) {
                    this.showError('Failed to abort task. Refresh the page or close the browser tab to force-end the session.');
                }
            } catch (err) {
                this.showError('Failed to abort task. Refresh the page or close the browser tab to force-end the session.');
            }
        },

        async approvePlan() {
            try {
                await fetch('/api/task/approve-plan', { method: 'POST' });
            } catch (e) {
                console.error('Approve plan error:', e);
            }
        },

        selectWorkflowStep(stepNumber) {
            if (this.selectedWorkflowStep === stepNumber) {
                this.selectedWorkflowStep = null; // toggle off = show all
            } else {
                this.selectedWorkflowStep = stepNumber;
            }
            this.selectedStep = null;
        },

        async rejectPlan() {
            try {
                await fetch('/api/task/reject-plan', { method: 'POST' });
            } catch (e) {
                console.error('Reject plan error:', e);
            }
        },

        async sendChatMessage() {
            const question = (this.chatInput || '').trim();
            if (!question || this.chatInFlight) return;

            // Phase 5: route to rewrite path when in Modify mode
            if (this.planMode === 'modify') {
                return this.sendRewriteRequest(question);
            }

            // --- Ask mode (existing Phase 4 flow) ---
            // Push user message, clear input, flip in-flight, clear any prior error
            this.chatHistory.push({ role: 'user', text: question });
            this.chatInput = '';
            this.chatError = '';
            this.chatInFlight = true;

            // Autoscroll after the user message lands
            this.$nextTick(() => {
                const el = this.$refs.chatMessages;
                if (el) el.scrollTop = el.scrollHeight;
            });

            try {
                const resp = await fetch('/api/task/chat-about-plan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question }),
                });
                if (!resp.ok) {
                    // Includes 503 when no plan is cached server-side
                    this.chatError = 'Could not reach the model. Check your connection and try again.';
                    return;
                }
                const body = await resp.json();
                if (body.status !== 'ok' || !body.answer) {
                    this.chatError = 'Could not reach the model. Check your connection and try again.';
                    return;
                }
                this.chatHistory.push({
                    role: 'assistant',
                    text: String(body.answer).trim(),
                });
                // Autoscroll after the answer lands and re-focus input for follow-ups
                this.$nextTick(() => {
                    const el = this.$refs.chatMessages;
                    if (el) el.scrollTop = el.scrollHeight;
                    const inp = this.$refs.chatInput;
                    if (inp) inp.focus();
                });
            } catch (e) {
                console.error('Chat error:', e);
                this.chatError = 'Could not reach the model. Check your connection and try again.';
            } finally {
                this.chatInFlight = false;
            }
        },

        // ================================================================
        // Phase 5: Plan Modification (DISCUSS-03, DISCUSS-04, DISCUSS-06)
        // ================================================================

        async sendRewriteRequest(request) {
            // Push user message to chat history so the user sees what they sent
            this.chatHistory.push({ role: 'user', text: request });
            this.chatInput = '';
            this.chatError = '';
            this.chatInFlight = true;

            this.$nextTick(() => {
                const el = this.$refs.chatMessages;
                if (el) el.scrollTop = el.scrollHeight;
            });

            try {
                const resp = await fetch('/api/task/request-plan-rewrite', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ request }),
                });
                if (!resp.ok) {
                    this.chatError = 'Could not reach the model. Check your connection and try again.';
                    this.chatInFlight = false;
                    return;
                }
                // Success path: chatInFlight clears in the WebSocket handler
                // (plan_modification_proposed or plan_rewrite_no_change or plan_rewrite_error).
                // We don't touch state here because the server is authoritative.
            } catch (e) {
                console.error('Rewrite request error:', e);
                this.chatError = 'Could not reach the model. Check your connection and try again.';
                this.chatInFlight = false;
            }
        },

        async acceptPlanRewrite() {
            try {
                const resp = await fetch('/api/task/accept-plan-rewrite', { method: 'POST' });
                if (!resp.ok) {
                    this.chatError = 'Could not update the plan. Check your connection and try again.';
                }
                // Success: plan_modified (reason=accept_rewrite) broadcast drives state update
            } catch (e) {
                console.error('Accept plan rewrite error:', e);
                this.chatError = 'Could not update the plan. Check your connection and try again.';
            }
        },

        async discardPlanRewrite() {
            try {
                const resp = await fetch('/api/task/discard-plan-rewrite', { method: 'POST' });
                if (!resp.ok) {
                    this.chatError = 'Could not update the plan. Check your connection and try again.';
                }
                // Success: plan_rewrite_discarded broadcast drives state update
            } catch (e) {
                console.error('Discard plan rewrite error:', e);
                this.chatError = 'Could not update the plan. Check your connection and try again.';
            }
        },

        beginEditStep(step) {
            if (this.planExecuting || this.planPreview || this.editingStep !== null) return;
            if (step.status === 'done') return;
            this.editingStep = step.number;
            this.editingDraft = step.goal || '';
            this.$nextTick(() => {
                const ta = this.$refs.editingTextarea;
                if (ta) {
                    ta.focus();
                    if (typeof ta.select === 'function') ta.select();
                }
            });
        },

        async saveEditStep() {
            // Idempotent no-op (Pitfall 7: blur-after-Esc race)
            if (this.editingStep === null) return;
            const stepNumber = this.editingStep;
            const newGoal = (this.editingDraft || '').trim();
            const orig = this.workflowPlan && this.workflowPlan.steps
                ? this.workflowPlan.steps.find(s => s.number === stepNumber)
                : null;
            // Empty or unchanged → treat as cancel
            if (!newGoal || !orig || newGoal === orig.goal) {
                this.cancelEditStep();
                return;
            }
            // Clear editing state FIRST so blur race doesn't re-fire
            this.editingStep = null;
            const draftBackup = this.editingDraft;
            this.editingDraft = '';
            try {
                const resp = await fetch('/api/task/edit-plan-step', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        step_number: stepNumber,
                        new_goal: newGoal,
                    }),
                });
                if (!resp.ok) {
                    this.chatError = 'Could not save your edit. Check your connection and try again.';
                    // Restore on failure so the user doesn't lose their typing
                    this.editingStep = stepNumber;
                    this.editingDraft = draftBackup;
                }
                // Success: plan_modified (reason=manual_edit) broadcast updates workflowPlan.steps
            } catch (e) {
                console.error('Save edit step error:', e);
                this.chatError = 'Could not save your edit. Check your connection and try again.';
                this.editingStep = stepNumber;
                this.editingDraft = draftBackup;
            }
        },

        cancelEditStep() {
            // [REVIEWS MEDIUM] Set editingStep to null BEFORE clearing draft.
            // The saveEditStep idempotent guard (editingStep === null) ensures
            // the subsequent blur event from the textarea is a no-op, preventing
            // the Esc-then-blur race where cancel accidentally saves.
            this.editingStep = null;
            this.editingDraft = '';
        },

        async addPlanStep() {
            if (this.planPreview || this.editingStep !== null) return;
            this._pendingAddFocus = true;
            try {
                const resp = await fetch('/api/task/add-plan-step', { method: 'POST' });
                if (!resp.ok) {
                    this.chatError = 'Could not add a step. Check your connection and try again.';
                    this._pendingAddFocus = false;
                }
                // Success: plan_modified (reason=add) broadcast auto-focuses the new step
            } catch (e) {
                console.error('Add plan step error:', e);
                this.chatError = 'Could not add a step. Check your connection and try again.';
                this._pendingAddFocus = false;
            }
        },

        async deletePlanStep(stepNumber) {
            if (this.planPreview || this.editingStep !== null) return;
            const currentCount = this.workflowPlan && this.workflowPlan.steps
                ? this.workflowPlan.steps.length
                : 0;
            // Guard: if deleting would leave fewer than 2 steps, ask for confirmation
            if (currentCount <= 2) {
                this.deleteConfirmStep = stepNumber;
                return;
            }
            await this._doDeletePlanStep(stepNumber);
        },

        async _doDeletePlanStep(stepNumber) {
            try {
                const resp = await fetch('/api/task/delete-plan-step', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ step_number: stepNumber }),
                });
                if (!resp.ok) {
                    this.chatError = 'Could not delete the step. Check your connection and try again.';
                }
                // Success: plan_modified (reason=delete) broadcast updates workflowPlan.steps
            } catch (e) {
                console.error('Delete plan step error:', e);
                this.chatError = 'Could not delete the step. Check your connection and try again.';
            }
        },

        async confirmDeleteStep() {
            const num = this.deleteConfirmStep;
            this.deleteConfirmStep = null;
            if (num !== null && num !== undefined) {
                await this._doDeletePlanStep(num);
            }
        },

        get deleteConfirmCopy() {
            if (this.deleteConfirmStep === null || !this.workflowPlan) return '';
            const remaining = this.workflowPlan.steps.length - 1;
            const plural = remaining === 1 ? '' : 's';
            return `This will leave only ${remaining} step${plural} in the plan. Delete anyway?`;
        },

        // --- Diff algorithm (hand-rolled, zero dependencies) ---

        _lcsTextDiff(oldText, newText) {
            // Tokenize on whitespace (preserve spaces as separate tokens so
            // joining segments faithfully reproduces the original text).
            const oldTokens = (oldText || '').split(/(\s+)/).filter(t => t.length > 0);
            const newTokens = (newText || '').split(/(\s+)/).filter(t => t.length > 0);
            const n = oldTokens.length;
            const m = newTokens.length;

            // LCS DP matrix
            const dp = Array(n + 1).fill(null).map(() => Array(m + 1).fill(0));
            for (let i = 1; i <= n; i++) {
                for (let j = 1; j <= m; j++) {
                    if (oldTokens[i - 1] === newTokens[j - 1]) {
                        dp[i][j] = dp[i - 1][j - 1] + 1;
                    } else {
                        dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                    }
                }
            }

            // Backtrack to build segment arrays
            const oldSegments = [];
            const newSegments = [];
            let i = n;
            let j = m;
            while (i > 0 || j > 0) {
                if (i > 0 && j > 0 && oldTokens[i - 1] === newTokens[j - 1]) {
                    oldSegments.unshift({ type: 'unchanged', text: oldTokens[i - 1] });
                    newSegments.unshift({ type: 'unchanged', text: newTokens[j - 1] });
                    i--;
                    j--;
                } else if (j > 0 && (i === 0 || dp[i][j - 1] >= dp[i - 1][j])) {
                    newSegments.unshift({ type: 'added', text: newTokens[j - 1] });
                    j--;
                } else {
                    oldSegments.unshift({ type: 'removed', text: oldTokens[i - 1] });
                    i--;
                }
            }
            return { oldSegments, newSegments };
        },

        computeDiff(previousSteps, currentSteps) {
            // [REVIEWS HIGH / D-11] Position-match with common prefix/suffix.
            // Intentionally simple: no stable IDs, no move detection.
            // LCS is only used INSIDE already-classified "modified" steps
            // for inline text diff — it does NOT drive step matching.
            const prev = previousSteps || [];
            const curr = currentSteps || [];

            // Common prefix
            let prefix = 0;
            while (
                prefix < prev.length &&
                prefix < curr.length &&
                prev[prefix].goal === curr[prefix].goal
            ) {
                prefix++;
            }

            // Common suffix (limited to what prefix didn't consume)
            let suffix = 0;
            while (
                suffix < prev.length - prefix &&
                suffix < curr.length - prefix &&
                prev[prev.length - 1 - suffix].goal === curr[curr.length - 1 - suffix].goal
            ) {
                suffix++;
            }

            const entries = [];

            // Unchanged prefix
            for (let k = 0; k < prefix; k++) {
                entries.push({ ...curr[k], diffState: 'unchanged', key: 'u-' + k });
            }

            // Middle: zipped diff by position
            const prevMiddle = prev.slice(prefix, prev.length - suffix);
            const currMiddle = curr.slice(prefix, curr.length - suffix);
            const midLen = Math.max(prevMiddle.length, currMiddle.length);
            for (let k = 0; k < midLen; k++) {
                const p = prevMiddle[k];
                const c = currMiddle[k];
                if (p && c) {
                    if (p.goal === c.goal) {
                        entries.push({ ...c, diffState: 'unchanged', key: 'mu-' + k });
                    } else {
                        const { oldSegments, newSegments } = this._lcsTextDiff(p.goal, c.goal);
                        entries.push({
                            ...c,
                            diffState: 'modified',
                            oldGoal: p.goal,
                            oldSegments,
                            newSegments,
                            key: 'mm-' + k,
                        });
                    }
                } else if (c) {
                    entries.push({ ...c, diffState: 'added', key: 'ma-' + k });
                } else if (p) {
                    entries.push({ ...p, diffState: 'removed', key: 'mr-' + k });
                }
            }

            // Unchanged suffix
            for (let k = 0; k < suffix; k++) {
                const idx = curr.length - suffix + k;
                entries.push({ ...curr[idx], diffState: 'unchanged', key: 's-' + k });
            }

            return entries;
        },

        get computedDiff() {
            if (!this.planPreview) {
                // Normal render — wrap each step with edited flag
                const steps = (this.workflowPlan && this.workflowPlan.steps) || [];
                return steps.map(s => ({
                    ...s,
                    diffState: null,
                    edited: this.editedStepNumbers.includes(s.number),
                    key: 'step-' + s.number,
                }));
            }
            // Preview state — diff current live steps vs proposed rewrite
            return this.computeDiff(
                (this.workflowPlan && this.workflowPlan.steps) || [],
                this.planPreview.steps || [],
            );
        },

        async resolveEscalation() {
            try {
                await fetch('/api/task/resolve-escalation', { method: 'POST' });
                this.escalationPending = false;
                this.escalationReason = '';
            } catch (e) {
                console.error('Resolve escalation error:', e);
            }
        },

        // ================================================================
        // Config API
        // ================================================================

        async loadConfig() {
            try {
                const resp = await fetch('/api/config');
                this.config = await resp.json();
            } catch (e) {
                console.error('Config load error:', e);
            }
        },

        async saveSection(section) {
            if (!this.config || !this.config[section]) return;
            try {
                const resp = await fetch(`/api/config/${section}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(this.config[section]),
                });
                const result = await resp.json();
                if (result.status === 'ok') {
                    this._flash(section + ' saved');
                    // Reload config after saving LLM section (provider change
                    // may have resolved a different API key from env vars)
                    if (section === 'llm') await this.loadConfig();
                } else {
                    alert('Error: ' + result.message);
                }
            } catch (e) {
                alert('Network error: ' + e.message);
            }
        },

        async saveTopLevel() {
            if (!this.config) return;
            try {
                for (const key of ['capture_source', 'executor_transport', 'target_os']) {
                    await fetch(`/api/config/${key}`, {
                        method: 'PUT',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ [key]: this.config[key] }),
                    });
                }
            } catch (e) {
                console.error('Save top-level error:', e);
            }
        },

        // ================================================================
        // Status API
        // ================================================================

        async refreshStatus() {
            try {
                const resp = await fetch('/api/status');
                this.statusData = await resp.json();
                this.captureReady = this.statusData.capture_ready || false;
                this.executorReady = this.statusData.executor_ready || false;
                this.captureDevice = this.statusData.capture_device || '';
                this.executorDevice = this.statusData.executor_device || '';
            } catch (e) {
                console.error('Status refresh error:', e);
            }
        },

        // ================================================================
        // Logs API
        // ================================================================

        async loadLogs() {
            try {
                const resp = await fetch('/api/logs?limit=200');
                const entries = await resp.json();
                this.logs = Array.isArray(entries) ? entries : [];
            } catch (e) {
                console.error('Logs load error:', e);
            }
        },

        async clearLogs() {
            try {
                await fetch('/api/logs', { method: 'DELETE' });
                this.logs = [];
            } catch (e) {
                console.error('Logs clear error:', e);
            }
        },

        // ================================================================
        // Computed-like: filtered logs
        // ================================================================

        get filteredLogs() {
            return this.logs.filter(log => {
                const text = typeof log === 'string' ? log : (log.message || '');
                if (this.logFilter.level && !text.includes(this.logFilter.level)) {
                    return false;
                }
                if (this.logFilter.module && !text.includes(this.logFilter.module)) {
                    return false;
                }
                return true;
            });
        },

        // ================================================================
        // Step selection
        // ================================================================

        _stepKey(step) {
            return (step._workflowStep || 0) + '-' + step.step;
        },

        selectStep(step) {
            const key = this._stepKey(step);
            if (this.selectedStep === key) {
                this.deselectStep();
                return;
            }
            this.selectedStep = key;
            if (step && step.screenshot_base64) {
                this.screenshot = step.screenshot_base64;
            }
        },

        deselectStep() {
            this.selectedStep = null;
            // Restore to latest screenshot
            const last = this.steps.length > 0 ? this.steps[this.steps.length - 1] : null;
            if (last && last.screenshot_base64) {
                this.screenshot = last.screenshot_base64;
            }
        },

        // ================================================================
        // Skills API
        // ================================================================

        async loadSkills() {
            try {
                const resp = await fetch('/api/skills');
                if (!resp.ok) {
                    console.error('Skills load failed:', resp.status);
                    return;
                }
                const data = await resp.json();
                this.skills = data.skills || [];
            } catch (e) {
                console.error('Skills load error:', e);
            }
        },

        async selectSkill(name) {
            if (this.editingSkill) return;
            this.selectedSkill = name;
            try {
                const resp = await fetch(`/api/skills/${encodeURIComponent(name)}`);
                if (resp.ok) {
                    const data = await resp.json();
                    this.skillContent = data.content;
                    this.skillSource = data.source;
                } else {
                    console.error('Skill fetch failed:', resp.status);
                }
            } catch (e) {
                console.error('Skill fetch error:', e);
            }
        },

        async toggleSkill(name) {
            const idx = this.activeSkills.indexOf(name);
            if (idx >= 0) {
                this.activeSkills.splice(idx, 1);
            } else {
                this.activeSkills.push(name);
            }
            try {
                const resp = await fetch('/api/config/agent', {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ skills: [...this.activeSkills] }),
                });
                if (!resp.ok) {
                    // Rollback
                    if (idx >= 0) this.activeSkills.push(name);
                    else this.activeSkills.splice(this.activeSkills.indexOf(name), 1);
                    alert('Failed to save skill activation. Please try again.');
                }
            } catch (e) {
                // Rollback
                if (idx >= 0) this.activeSkills.push(name);
                else this.activeSkills.splice(this.activeSkills.indexOf(name), 1);
                alert('Network error: ' + e.message);
            }
        },

        startNewSkill() {
            this.editingSkill = true;
            this.isNewSkill = true;
            this.editSkillName = '';
            this.editSkillContent = '';
        },

        startEditSkill() {
            this.editingSkill = true;
            this.isNewSkill = false;
            this.editSkillName = this.selectedSkill;
            this.editSkillContent = this.skillContent;
        },

        async saveSkill() {
            const name = this.editSkillName.trim();
            const content = this.editSkillContent;
            if (!name || !content.trim()) return;
            try {
                const resp = await fetch(`/api/skills/${encodeURIComponent(name)}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ content }),
                });
                const result = await resp.json();
                if (result.status === 'ok') {
                    this.editingSkill = false;
                    await this.loadSkills();
                    this.selectedSkill = name;
                    await this.selectSkill(name);
                } else {
                    alert('Error: ' + result.message);
                }
            } catch (e) {
                alert('Network error: ' + e.message);
            }
        },

        cancelEdit() {
            this.editingSkill = false;
        },

        async deleteSkill(name) {
            if (!confirm(`Delete user skill "${name}"?`)) return;
            try {
                const resp = await fetch(`/api/skills/${encodeURIComponent(name)}`, {
                    method: 'DELETE',
                });
                const result = await resp.json();
                if (result.status === 'ok') {
                    const idx = this.activeSkills.indexOf(name);
                    if (idx >= 0) {
                        this.activeSkills.splice(idx, 1);
                        await fetch('/api/config/agent', {
                            method: 'PUT',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ skills: [...this.activeSkills] }),
                        });
                    }
                    this.selectedSkill = null;
                    this.skillContent = '';
                    this.skillSource = '';
                    await this.loadSkills();
                } else {
                    alert('Error: ' + result.message);
                }
            } catch (e) {
                alert('Network error: ' + e.message);
            }
        },

        // ================================================================
        // Helpers
        // ================================================================

        formatStep(step) {
            if (step.execute_status === 'screenshot') {
                return step.step === 0 ? 'task started' : 'screenshot';
            }
            const cmd = step.command || {};
            const action = cmd.action || '?';
            switch (action) {
                case 'click':
                case 'double_click':
                    return `${action} (${cmd.x}, ${cmd.y})`;
                case 'type':
                    const text = cmd.text || '';
                    return `type "${text.length > 25 ? text.slice(0, 25) + '...' : text}"`;
                case 'key':
                    return `key ${(cmd.keys || []).join('+')}`;
                case 'scroll':
                    return `scroll ${cmd.direction || '?'}`;
                case 'drag':
                    return `drag (${cmd.from_x},${cmd.from_y})->(${cmd.to_x},${cmd.to_y})`;
                case 'done':
                    return 'Done';
                default:
                    return action;
            }
        },

        // ================================================================
        // Connection label / tooltip helpers
        // ================================================================

        captureTooltip() {
            const src = this.config && this.config.capture_source;
            return {
                hdmi: 'Captures the target screen through a USB HDMI dongle. Connect an HDMI cable from the target computer to the capture card.',
                csi: 'Captures the target screen using the Raspberry Pi CSI camera. Point the camera at the target screen.',
                airplay: 'Receives the target screen wirelessly via AirPlay. On the target Apple device, mirror to this device.',
            }[src] || '';
        },

        transportTooltip() {
            const t = this.config && this.config.executor_transport;
            return {
                usb: 'Controls the target via USB HID gadget. Connect the Pi USB-C port to the target with an OTG cable.',
                bt: 'Controls the target wirelessly via Bluetooth. Pair with the target computer first.',
            }[t] || '';
        },

        copyText(text, event) {
            const btn = event.target;
            const done = () => {
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
            };
            const fail = () => {
                btn.textContent = 'Failed';
                setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
            };
            if (navigator.clipboard && window.isSecureContext) {
                navigator.clipboard.writeText(text).then(done).catch(fail);
            } else {
                const ta = document.createElement('textarea');
                ta.value = text;
                ta.style.position = 'fixed';
                ta.style.opacity = '0';
                document.body.appendChild(ta);
                ta.select();
                const ok = document.execCommand('copy');
                document.body.removeChild(ta);
                ok ? done() : fail();
            }
        },

        formatRawPrompt(step) {
            if (!step || !step.prompt_messages) return '(no prompt data)';
            return JSON.stringify(step.prompt_messages, null, 2);
        },

        formatMsgContent(content) {
            if (content == null) return '(empty)';
            if (typeof content === 'string') return content;
            if (typeof content === 'object') return JSON.stringify(content, null, 2);
            return String(content);
        },

        showError(msg) {
            // Display error to user via captureError (reuses existing error banner)
            this.captureError = msg;
            setTimeout(() => {
                if (this.captureError === msg) this.captureError = '';
            }, 5000);
        },

        _flash(msg) {
            // Simple temporary status indicator
            console.log('Flash:', msg);
        },
    };
}

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
        refreshingScreenshot: false,
        taskResult: null,
        selectedStep: null,  // unique key string (e.g. "3-2"), null = follow latest
        stepDetailTab: 'prompt',    // 'prompt' | 'response' (D-01)
        viewMode: 'formatted',      // 'formatted' | 'raw' (D-04, D-05)
        autoFollow: true,           // explicit auto-follow toggle (UAT gap 2)
        promptScrollTop: 0,         // per-tab scroll preservation (UAT gap 3)
        responseScrollTop: 0,
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

        // --- Hardware setup status (populated by /api/setup/status) ---
        setupStatus: null,        // null while loading, then {components, needs_setup, needs_reboot, setup_commands}
        setupPollTimer: null,     // interval handle for periodic refresh

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

        // --- Replan Dialog (Phase 3) ---
        replanDialog: null,                  // null | {path, step_number, step_goal, expected, observed, mismatch_reason, failure_reason, screenshot_base64}
        replanDecisionPending: false,        // true between click and server ack
        autoReplan: false,                   // synced from /api/config on init, persisted via /api/task/auto-replan
        connectionLostDuringDialog: false,   // true when WebSocket drops while replanDialog !== null
        _priorFocus: null,                   // saved focus target for restore on dialog close
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

        // --- LLM model suggestions (per provider) ---
        // Shown as <datalist> options; user may pick one or type freely.
        modelOptions: {
            anthropic: [
                'claude-opus-4-6',
                'claude-opus-4-5',
                'claude-sonnet-4-6',
                'claude-haiku-4-5',
                'claude-3-5-sonnet-latest',
                'claude-3-5-haiku-latest',
            ],
            openai: [
                'gpt-5.4',
                'gpt-4o',
                'gpt-4o-mini',
                'gpt-4-turbo',
                'o1',
                'o1-mini',
            ],
        },

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

        currentModelOptions() {
            const provider = this.config?.llm?.provider;
            return this.modelOptions[provider] || [];
        },

        get filteredSteps() {
            if (!this.workflowActive || this.selectedWorkflowStep === null) {
                return this.steps;
            }
            return this.steps.filter(s => s._workflowStep === this.selectedWorkflowStep);
        },

        // UAT gap 5: returns an array of {step, isGroupHead, groupSize, groupKey}
        // so the step log template can render consecutive OpenAI CU queued-
        // action steps as a grouped cluster. Only the group head shows a
        // step number; members show an indent marker. All rows in a group
        // share the same click target (groupKey) so selecting any member
        // selects the head step.
        get groupedSteps() {
            const input = this.filteredSteps;
            const result = [];
            let i = 0;
            while (i < input.length) {
                const current = input[i];
                const rid = current.response_id;
                if (!rid) {
                    result.push({
                        step: current,
                        isGroupHead: true,
                        groupSize: 1,
                        groupKey: this._stepKey(current),
                    });
                    i++;
                    continue;
                }
                // Walk forward collecting consecutive steps with the same response_id
                let j = i;
                const members = [];
                while (j < input.length && input[j].response_id === rid) {
                    members.push(input[j]);
                    j++;
                }
                const headKey = this._stepKey(members[0]);
                members.forEach((m, idx) => {
                    result.push({
                        step: m,
                        isGroupHead: idx === 0,
                        groupSize: members.length,
                        groupKey: headKey,
                    });
                });
                i = j;
            }
            return result;
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
            // Phase 3: sync Auto Re-plan toggle from persisted config
            if (this.config && this.config.agent) {
                this.autoReplan = this.config.agent.auto_replan === true;
            }
            await this.refreshStatus();
            await this.loadLogs();
            this.connectWebSocket();
            // Init split resize immediately (split is always visible)
            this.$nextTick(() => this.initSplitResize());
        },

        initSplitResize() {
            const container = this.$refs.splitContainer;
            const divider = this.$refs.splitDivider;
            if (!container || !divider) return;

            let dragging = false;

            const onPointerDown = (e) => {
                dragging = true;
                divider.classList.add('dragging');
                divider.setPointerCapture(e.pointerId);
                e.preventDefault();
            };
            const onPointerMove = (e) => {
                if (!dragging) return;
                const rect = container.getBoundingClientRect();
                const offset = e.clientX - rect.left;
                const pct = Math.min(Math.max(offset / rect.width * 100, 20), 80);
                container.style.gridTemplateColumns = pct + '% 8px ' + (100 - pct) + '%';
            };
            const onPointerUp = () => {
                dragging = false;
                divider.classList.remove('dragging');
            };

            divider.addEventListener('pointerdown', onPointerDown);
            divider.addEventListener('pointermove', onPointerMove);
            divider.addEventListener('pointerup', onPointerUp);
            divider.addEventListener('lostpointercapture', onPointerUp);

            // Clear inline grid-template-columns when viewport drops below breakpoint (Pitfall 6)
            const mq = window.matchMedia('(max-width: 900px)');
            const handleBreakpoint = (e) => {
                if (e.matches) {
                    container.style.gridTemplateColumns = '';
                }
            };
            mq.addEventListener('change', handleBreakpoint);
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

            this.ws.onopen = async () => {
                this.wsConnected = true;
                console.log('WebSocket connected');
                // Phase 3 — H7: SINGLE reconnect replay mechanism.
                // The server's /ws endpoint does NOT send pending dialog state
                // on connect. We fetch it explicitly here and re-render.
                // I5 — differentiate network failures (acceptable to swallow
                // — reconnect happens) from server errors (worth surfacing).
                try {
                    const resp = await fetch('/api/task/pending-dialogs');
                    if (!resp.ok) {
                        console.warn('pending-dialogs returned', resp.status);
                        this._flash(`Could not restore pending dialogs after reconnect (HTTP ${resp.status})`);
                    } else {
                        const data = await resp.json();
                        // H7 new shape — {dialogs: [...]}
                        const dialogs = data.dialogs || [];
                        for (const d of dialogs) {
                            if (d._active_gate === 'replan_A' || d._active_gate === 'replan_B') {
                                // Re-open the modal
                                this.openReplanDialog(d);
                            } else if (d._active_gate === 'escalation_C') {
                                if (!this.escalationPending) {
                                    this.escalationPending = true;
                                    this.escalationReason = d.reason || '';
                                }
                            }
                        }
                    }
                } catch (e) {
                    // True network failure (server unreachable). Reconnect
                    // will retry; logging at warn level rather than silent.
                    console.warn('pending-dialogs network error:', e);
                }
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
                // Phase 3 — surface connection-lost warning if a modal is open
                if (this.replanDialog) {
                    this.connectionLostDuringDialog = true;
                }
                this._scheduleReconnect();
            };

            this.ws.onerror = () => {
                this.wsConnected = false;
            };

            // I5 — split JSON.parse failures (probably benign — non-JSON
            // frame from a future server) from handleEvent failures (real
            // bug — would leave UI desynced; flash so the user knows).
            this.ws.onmessage = (evt) => {
                let msg;
                try {
                    msg = JSON.parse(evt.data);
                } catch (e) {
                    console.warn('WS message parse error:', e);
                    return;
                }
                try {
                    this.handleEvent(msg);
                } catch (e) {
                    console.error('WS handleEvent failed:', e, msg);
                    this._flash('UI desync — refresh the page if state looks wrong');
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
                    // Keep existing screenshot until first task_step delivers a new one
                    this.selectedStep = null;
                    this.stepDetailTab = 'prompt';
                    this.viewMode = 'formatted';
                    this.autoFollow = true;         // fresh task: resume following (gap 2)
                    this.promptScrollTop = 0;       // reset per-tab scroll (gap 3)
                    this.responseScrollTop = 0;
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
                    // Phase 3: reset replan dialog state on new task (autoReplan is sticky — NOT reset)
                    this.replanDialog = null;
                    this.replanDecisionPending = false;
                    this.connectionLostDuringDialog = false;
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
                    // Auto-follow state machine (UAT gap 2): only resume when toggle is ON.
                    // MANUAL -> FOLLOWING on new step (D-18) only if autoFollow is enabled.
                    if (this.autoFollow) {
                        if (this.selectedStep !== null) {
                            this.selectedStep = null;
                        }
                        // Update screenshot only while actively following
                        if (data.screenshot_base64) {
                            this.screenshot = data.screenshot_base64;
                        }
                        // Reset per-tab scroll so new content starts at the top (gap 3)
                        this.promptScrollTop = 0;
                        this.responseScrollTop = 0;
                        this.$nextTick(() => {
                            const el = this.$refs.stepLog;
                            if (el) el.scrollTop = el.scrollHeight;
                        });
                    }
                    break;

                case 'task_finished':
                    this.taskRunning = false;
                    this.planPaused = false;
                    this.pauseRequested = false;
                    this.taskResult = data;
                    // Auto-follow state machine (UAT gap 2): only resume when toggle is ON.
                    // MANUAL -> FOLLOWING on task completion (D-18) only if autoFollow is enabled.
                    if (this.autoFollow) {
                        if (this.selectedStep !== null) {
                            this.selectedStep = null;
                        }
                        // Ensure final screenshot is shown
                        if (this.steps.length > 0) {
                            const lastStep = this.steps[this.steps.length - 1];
                            if (lastStep.screenshot_base64) {
                                this.screenshot = lastStep.screenshot_base64;
                            }
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

                case 'executor_init_failed':
                    this.executorReady = false;
                    this.executorDevice = '';
                    this.connectingExecutor = false;
                    this.executorError = (data && data.error) || 'Executor init failed';
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

                case 'replanned': {
                    if (!this.workflowPlan) break;
                    // Mark cancelled steps in place (preserves numbers, lets template render red ✗)
                    const cancelledNumbers = new Set(data.cancelled_step_numbers || []);
                    for (const step of this.workflowPlan.steps) {
                        if (cancelledNumbers.has(step.number)) {
                            step.status = 'cancelled';
                        }
                    }
                    // Append new steps with their backend-assigned numbers and pending status
                    const newSteps = (data.new_steps || []).map(s => ({
                        number: s.number,
                        goal: s.goal,
                        reboot_expected: s.reboot_expected || false,
                        expected_actions: s.expected_actions,
                        expected_outcome: s.expected_outcome || '',
                        status: s.status || 'pending',
                    }));
                    this.workflowPlan.steps = [...this.workflowPlan.steps, ...newSteps];
                    // H8 — DO NOT close modal here. The backend emits
                    // replan_dialog_resolved separately, which closes it uniformly
                    // for every choice. Closing here too would duplicate the
                    // dialog-close logic across per-choice handlers.
                    break;
                }

                case 'replan_dialog':
                    this.openReplanDialog(data);
                    break;

                case 'replan_auto':
                    this._flash(`Auto-replanning (path ${data.path})`);
                    break;

                case 'replan_failed':
                    this._flash('Re-plan failed; the task now needs your attention.');
                    break;

                case 'replan_dialog_resolved': {
                    // H8 — unconditional modal close. Fires for EVERY choice
                    // (continue, retry, replan, resume, abort). The per-choice
                    // modal-close logic elsewhere in this switch is redundant
                    // and should NOT fire — closeReplanDialog is idempotent.
                    if (this.replanDialog) {
                        this.closeReplanDialog();
                    }
                    // Also announce via live region for screen-reader users
                    const live = document.getElementById('replan-dialog-live');
                    if (live) {
                        live.textContent = `Decision submitted: ${data.choice || 'unknown'}`;
                    }
                    break;
                }

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

        async refreshScreenshot() {
            this.refreshingScreenshot = true;
            try {
                const resp = await fetch('/api/capture/preview');
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.image) {
                        this.screenshot = data.image;
                    }
                }
            } catch (e) {
                console.error('Refresh screenshot error:', e);
            } finally {
                this.refreshingScreenshot = false;
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

        async resetTask() {
            try {
                const resp = await fetch('/api/task/reset', { method: 'POST' });
                const result = await resp.json();
                if (result.status === 'reset') {
                    this.taskRunning = false;
                    this.steps = [];
                    this.taskResult = null;
                    this.workflowPlan = null;
                    this.selectedStep = null;
                    this.captureError = '';
                    this.totalInputTokens = 0;
                    this.totalOutputTokens = 0;
                    this.totalCacheReadTokens = 0;
                    this.totalCacheCreationTokens = 0;
                } else if (result.status === 'aborted') {
                    this.showError('Task is still running — abort requested. Please wait for it to finish.');
                }
            } catch (e) {
                this.showError('Failed to reset task state: ' + e.message);
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
        // Replan Dialog (Phase 3 — REPLAN-01/02/03, UI-02/03, H7, H8)
        // ================================================================

        openReplanDialog(payload) {
            // Save the previously-focused element for restore-on-close
            this._priorFocus = document.activeElement;
            this.replanDialog = payload;
            this.replanDecisionPending = false;
            this.connectionLostDuringDialog = false;
            // Live-region announcement
            const liveRegion = document.getElementById('replan-dialog-live');
            if (liveRegion) {
                liveRegion.textContent = payload.path === 'A'
                    ? `Verification failed on step ${payload.step_number}`
                    : `Step ${payload.step_number} failed`;
            }
        },

        closeReplanDialog() {
            this.replanDialog = null;
            this.replanDecisionPending = false;
            this.connectionLostDuringDialog = false;
            if (this._priorFocus && this._priorFocus.focus) {
                try { this._priorFocus.focus(); } catch (_) {}
            }
            this._priorFocus = null;
        },

        async submitReplanDecision(choice) {
            if (this.replanDecisionPending) return;
            this.replanDecisionPending = true;
            // I5 — fetch only throws on network errors; HTTP 4xx/5xx still
            // resolve. Without an explicit resp.ok check the modal stays
            // open forever when the server returns 400 (bad gate) or 409
            // (no gate armed = server moved on). Close on 409 since the
            // workflow has already advanced; flash the error otherwise.
            try {
                const resp = await fetch('/api/task/replan-decision', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ choice }),
                });
                if (!resp.ok) {
                    const body = await resp.json().catch(() => ({}));
                    const detail = body.error || body.detail || `HTTP ${resp.status}`;
                    this._flash(`Decision rejected: ${detail}`);
                    this.replanDecisionPending = false;
                    if (resp.status === 409) {
                        // Server already moved on — close the now-stale modal.
                        this.closeReplanDialog();
                    }
                    return;
                }
                // Server will close the dialog via the next event
                // (replan_dialog_resolved — H8 unconditional close)
            } catch (e) {
                console.error('replan-decision failed:', e);
                this._flash(`Network error submitting decision: ${e.message || e}`);
                this.replanDecisionPending = false;
            }
        },

        async toggleAutoReplan() {
            const newValue = !this.autoReplan;
            this.autoReplan = newValue;  // optimistic
            try {
                const resp = await fetch('/api/task/auto-replan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ enabled: newValue }),
                });
                if (!resp.ok) throw new Error('config write failed');
                this._flash(`Auto Re-plan ${newValue ? 'on' : 'off'}`);
            } catch (e) {
                this._flash(`Auto Re-plan ${newValue ? 'on' : 'off'} (config write failed — will not persist across restart)`);
            }
        },

        trapFocus(containerEl) {
            // Find focusable elements inside the modal
            const focusables = containerEl.querySelectorAll(
                'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
            );
            if (focusables.length === 0) return;
            const first = focusables[0];
            const last = focusables[focusables.length - 1];
            // Initial focus on the primary action (rightmost button)
            last.focus();
            // Cycle Tab / Shift+Tab
            containerEl.addEventListener('keydown', (e) => {
                if (e.key !== 'Tab') return;
                if (e.shiftKey && document.activeElement === first) {
                    e.preventDefault();
                    last.focus();
                } else if (!e.shiftKey && document.activeElement === last) {
                    e.preventDefault();
                    first.focus();
                }
            });
            // Listener auto-detaches when Alpine destroys the template subtree on close (x-if pattern).
            // The modal lives inside <template x-if="replanDialog"> — when replanDialog becomes null,
            // Alpine removes the DOM node and the keydown listener attached to it is garbage-collected
            // along with the element. No manual removeEventListener needed.
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

        async saveTopLevel(changedKey) {
            if (!this.config) return;
            // Clear stale error messages tied to the setting that just changed —
            // an old failure under one transport/source shouldn't linger when
            // the user has picked a different option.
            if (changedKey === 'capture_source') {
                this.captureError = '';
            } else if (changedKey === 'executor_transport' || changedKey === 'target_os') {
                this.executorError = '';
            }
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

        async refreshSetupStatus() {
            try {
                const resp = await fetch('/api/setup/status');
                this.setupStatus = await resp.json();
            } catch (e) {
                console.error('Setup status refresh error:', e);
            }
        },

        startSetupPolling() {
            // Fire immediately, then every 10s while the Status tab is visible.
            this.refreshSetupStatus();
            if (this.setupPollTimer) return;
            this.setupPollTimer = setInterval(() => this.refreshSetupStatus(), 10000);
        },

        stopSetupPolling() {
            if (this.setupPollTimer) {
                clearInterval(this.setupPollTimer);
                this.setupPollTimer = null;
            }
        },

        // Human-readable labels for the setup checklist
        setupComponentLabel(key) {
            return {
                python_env: 'Python environment',
                bluetooth: 'Bluetooth HID',
                usb_gadget: 'USB HID Gadget',
                csi_hdmi: 'CSI HDMI Capture',
                airplay: 'AirPlay Capture',
            }[key] || key;
        },

        // Command hint to run for a specific component (when it needs setup)
        setupComponentCommand(key) {
            return {
                bluetooth: 'sudo scripts/setup.sh --bt',
                usb_gadget: 'sudo scripts/setup.sh --gadget',
                csi_hdmi: 'sudo scripts/setup.sh --csi',
                airplay: 'sudo scripts/setup.sh --airplay',
            }[key] || null;
        },

        // Maps status string → { cls, icon } for consistent badge styling
        setupStatusBadge(status) {
            return {
                ready:           { cls: 'text-ok',    icon: '✓' },
                not_configured:  { cls: 'text-error', icon: '✗' },
                partial:         { cls: 'text-warn',  icon: '!' },
                reboot_required: { cls: 'text-warn',  icon: '⟳' },
                not_available:   { cls: 'muted',      icon: '—' },
            }[status] || { cls: 'muted', icon: '?' };
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

        // UAT gap 5: select a grouped cluster. Clicking any row in a
        // grouped cluster sets selectedStep to the group head's key so
        // every member highlights together and the Response tab renders
        // the whole batch via renderResponseFormatted's group-aware path.
        selectStepGroup(groupKey, step) {
            if (this.selectedStep === groupKey) {
                this.deselectStep();
                return;
            }
            this.selectedStep = groupKey;
            // WR-02 fix: use the HEAD step's screenshot to stay consistent
            // with activeStep (which also resolves to the head). Previously
            // we used the clicked member's screenshot, causing a mismatch
            // between the preview pane (post-member-N state) and the detail
            // pane (head-labelled state).
            const head = this.steps.find(s => this._stepKey(s) === groupKey);
            const target = head || step;
            if (target && target.screenshot_base64) {
                this.screenshot = target.screenshot_base64;
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

        // Switch the Prompt/Response tab while preserving each tab's scroll
        // position (UAT gap 3). We read the outgoing tab's scrollTop BEFORE
        // mutating stepDetailTab, then restore the incoming tab's scrollTop
        // after Alpine's x-show reflow via $nextTick.
        switchStepDetailTab(newTab) {
            if (this.stepDetailTab === newTab) return;
            // Capture outgoing scroll position synchronously
            const outgoingRef = this.stepDetailTab === 'prompt'
                ? this.$refs.promptContent
                : this.$refs.responseContent;
            if (outgoingRef) {
                if (this.stepDetailTab === 'prompt') {
                    this.promptScrollTop = outgoingRef.scrollTop;
                } else {
                    this.responseScrollTop = outgoingRef.scrollTop;
                }
            }
            this.stepDetailTab = newTab;
            // Restore incoming tab's scroll after Alpine flips x-show
            this.$nextTick(() => {
                const incomingRef = newTab === 'prompt'
                    ? this.$refs.promptContent
                    : this.$refs.responseContent;
                if (incomingRef) {
                    incomingRef.scrollTop = newTab === 'prompt'
                        ? this.promptScrollTop
                        : this.responseScrollTop;
                }
            });
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

        // ================================================================
        // Content Renderers (Plan 02-02)
        // Sanitization contract: escapeHtml() is the boundary.
        // Every renderer calls escapeHtml() on ALL external text BEFORE
        // wrapping in HTML tags. x-html never receives raw LLM text.
        // ================================================================

        escapeHtml(str) {
            if (!str) return '';
            return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
        },

        highlightJson(jsonStr) {
            if (!jsonStr) return '';
            // Sanitize first -- escapeHtml is the boundary
            const escaped = String(jsonStr).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
            return escaped.replace(
                /("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
                function(match) {
                    let cls = 'json-number';
                    if (/^"/.test(match)) {
                        cls = /:$/.test(match) ? 'json-key' : 'json-string';
                    } else if (/true|false/.test(match)) {
                        cls = 'json-boolean';
                    } else if (/null/.test(match)) {
                        cls = 'json-null';
                    }
                    return '<span class="' + cls + '">' + match + '</span>';
                }
            );
        },

        renderMarkdown(text) {
            if (!text) return '';
            // Sanitize first -- escapeHtml is the boundary
            let html = this.escapeHtml(text);
            // Code blocks (triple backtick) -- before inline code
            html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre class="md-code-block"><code>$2</code></pre>');
            // Headings
            html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
            html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
            html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
            // Bold
            html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            // Inline code
            html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
            // Line breaks
            html = html.replace(/\n/g, '<br>');
            return '<div class="md-rendered">' + html + '</div>';
        },

        renderImagePlaceholder(block) {
            let sizeInfo = '';
            if (block.type === 'image' && block.source && block.source.data) {
                sizeInfo = ' (' + Math.round(block.source.data.length / 1024) + 'KB)';
            } else if (block.type === 'image_url' && block.image_url && block.image_url.url) {
                const match = block.image_url.url.match(/base64,(.+)/);
                if (match) sizeInfo = ' (' + Math.round(match[1].length / 1024) + 'KB)';
            }
            return '<div class="image-placeholder">Screenshot' + sizeInfo + '</div>';
        },

        // Convert provider-native tool_use input (Anthropic CU and similar)
        // into the flat {action, x, y, ...} shape that renderActionCard reads.
        // Returns the input unchanged if no provider-specific shape is detected.
        _normalizeToolUseInput(input) {
            if (!input || typeof input !== 'object') return input;
            const out = { ...input };
            if (Array.isArray(input.coordinate) && input.coordinate.length >= 2) {
                if (out.x === undefined) out.x = input.coordinate[0];
                if (out.y === undefined) out.y = input.coordinate[1];
            }
            if (Array.isArray(input.start_coordinate) && input.start_coordinate.length >= 2) {
                if (out.startX === undefined) out.startX = input.start_coordinate[0];
                if (out.startY === undefined) out.startY = input.start_coordinate[1];
            }
            // Anthropic drag: action='left_click_drag' with start_coordinate + coordinate as endpoints
            if (input.action === 'left_click_drag') {
                out.action = 'drag';
                if (out.endX === undefined && out.x !== undefined) out.endX = out.x;
                if (out.endY === undefined && out.y !== undefined) out.endY = out.y;
            }
            // Anthropic scroll: scroll_direction → direction
            if (input.scroll_direction && !out.direction) {
                out.direction = input.scroll_direction;
            }
            // Anthropic key: text holds the key combo string ("ctrl+c") instead of keys[]
            if (input.action === 'key' && typeof input.text === 'string' && !Array.isArray(input.keys)) {
                out.keys = input.text.split('+').map(k => k.trim());
            }
            return out;
        },

        renderActionCard(command) {
            if (!command) return '<div class="action-card muted">(no command)</div>';
            const action = command.action || 'unknown';
            const icons = {
                click: '\u{1F5B1}', double_click: '\u{1F5B1}', left_click: '\u{1F5B1}',
                type: '\u2328', key: '\u2318', scroll: '\u2195', drag: '\u2922',
                done: '\u2713', screenshot: '\u{1F4F7}'
            };
            const icon = icons[action] || '\u2022';
            const safeAction = this.escapeHtml(action);
            let detail = '';
            switch (action) {
                case 'click': case 'double_click': case 'left_click':
                    detail = 'at (' + this.escapeHtml(String(command.x)) + ', ' + this.escapeHtml(String(command.y)) + ')'; break;
                case 'type':
                    detail = this.escapeHtml('"' + (command.text || '').substring(0, 80) + '"'); break;
                case 'key':
                    detail = this.escapeHtml((command.keys || []).join(' + ')); break;
                case 'scroll':
                    detail = this.escapeHtml(command.direction || ''); break;
                case 'drag':
                    detail = 'from (' + this.escapeHtml(String(command.startX)) + ',' + this.escapeHtml(String(command.startY)) + ') to (' + this.escapeHtml(String(command.endX)) + ',' + this.escapeHtml(String(command.endY)) + ')'; break;
                case 'done':
                    detail = this.escapeHtml(command.reason || ''); break;
                default:
                    detail = this.escapeHtml(JSON.stringify(command));
            }
            return '<div class="action-card action-' + safeAction + '">' +
                   '<span class="action-icon">' + icon + '</span>' +
                   '<span class="action-label">' + safeAction + '</span>' +
                   '<span class="action-detail">' + detail + '</span>' +
                   '</div>';
        },

        renderBlockArray(blocks) {
            if (!Array.isArray(blocks)) return this.escapeHtml(JSON.stringify(blocks));
            return blocks.map(block => {
                if (!block || !block.type) return '<div class="muted">' + this.escapeHtml(JSON.stringify(block)) + '</div>';
                switch (block.type) {
                    case 'text':
                        return '<div class="chat-text">' + this.escapeHtml(block.text || '') + '</div>';
                    case 'image':
                        return this.renderImagePlaceholder(block);
                    case 'image_url':
                        return this.renderImagePlaceholder(block);
                    case 'tool_use':
                        return this.renderActionCard(
                            this._normalizeToolUseInput(block.input || {action: block.name})
                        );
                    case 'tool_result':
                        return '<div class="tool-result">' +
                               '<span class="tool-result-label">Tool Result</span>' +
                               (Array.isArray(block.content)
                                   ? this.renderBlockArray(block.content)
                                   : '<div>' + this.escapeHtml(String(block.content || '')) + '</div>') +
                               '</div>';
                    default:
                        // Unknown block type -- safe fallback to escaped JSON
                        return '<div class="muted">' + this.escapeHtml(JSON.stringify(block, null, 2)) + '</div>';
                }
            }).join('');
        },

        renderMessageContent(content, role) {
            if (content == null) return '<span class="muted">(empty)</span>';
            if (typeof content === 'string') {
                return '<div class="chat-text">' + this.escapeHtml(content) + '</div>';
            }
            if (Array.isArray(content)) {
                return this.renderBlockArray(content);
            }
            // Fallback for unexpected types -- escaped JSON
            return '<div class="chat-text">' + this.escapeHtml(JSON.stringify(content, null, 2)) + '</div>';
        },

        // Estimate line count for a message content (UAT gap 4).
        // Heuristic: count newline chars in the stringified text.
        // For Array-of-blocks (Anthropic CU), concatenate block.text fields.
        // For string content, use directly. Unknown types → default to 0
        // (prefer showing in full rather than over-collapsing).
        _messageLineCount(content) {
            let text = '';
            if (typeof content === 'string') {
                text = content;
            } else if (Array.isArray(content)) {
                text = content
                    .filter(b => b && b.type === 'text' && typeof b.text === 'string')
                    .map(b => b.text)
                    .join('\n');
            } else {
                return 0;
            }
            const newlines = (text.match(/\n/g) || []).length;
            return newlines + 1;
        },

        renderPromptFormatted(step) {
            if (!step) return '';
            let html = '';
            // System prompt section (collapsible, D-17)
            if (step.system_prompt) {
                html += '<div class="prompt-section">';
                html += '<h4 class="section-label">System Prompt</h4>';
                html += '<div class="collapsible" id="sys-prompt-collapse">';
                html += '<div class="collapsible-content">';
                html += this.renderMarkdown(step.system_prompt);
                html += '</div>';
                html += '<button class="collapsible-toggle" onclick="this.textContent = this.parentElement.classList.toggle(\'expanded\') ? \'Show less\' : \'Show more\'">Show more</button>';
                html += '</div>';
                html += '</div>';
            }
            // Conversation messages as chat bubbles (D-15, D-17)
            if (step.prompt_messages && step.prompt_messages.length > 0) {
                html += '<div class="prompt-section">';
                html += '<h4 class="section-label">Conversation</h4>';
                html += '<div class="chat-timeline">';
                step.prompt_messages.forEach((msg, i) => {
                    const role = msg.role || 'unknown';
                    const total = step.prompt_messages.length;
                    const isRecent = i >= total - 2;
                    // Short bubbles (<=4 lines) also skip collapse, per UAT gap 4.
                    // Heuristic lives in _messageLineCount; unknown content → 0 → short.
                    const lineCount = this._messageLineCount(msg.content);
                    const isShort = lineCount <= 4;
                    const shouldCollapse = !isRecent && !isShort;
                    html += '<div class="chat-bubble chat-bubble-' + role + (shouldCollapse ? ' collapsible' : '') + '">';
                    html += '<span class="chat-bubble-role">' + this.escapeHtml(role.toUpperCase()) + '</span>';
                    // Always apply chat-bubble-content for typography (UAT gap 6 /
                    // RENDER-02: font-size: 0.75rem). Add collapsible-content as an
                    // additional class when the bubble should clamp, so both the
                    // max-height clamp AND the font-size rule take effect.
                    html += '<div class="chat-bubble-content' + (shouldCollapse ? ' collapsible-content' : '') + '">';
                    html += this.renderMessageContent(msg.content, role);
                    html += '</div>';
                    if (shouldCollapse) {
                        html += '<button class="collapsible-toggle" onclick="this.textContent = this.parentElement.classList.toggle(\'expanded\') ? \'Show less\' : \'Show more\'">Show more</button>';
                    }
                    html += '</div>';
                });
                html += '</div>';
                html += '</div>';
            }
            return html || '<p class="muted">(no prompt data)</p>';
        },

        renderPromptRaw(step) {
            if (!step) return '';
            let html = '';
            if (step.system_prompt) {
                html += '<div class="prompt-section">';
                html += '<h4 class="section-label">System Prompt</h4>';
                html += '<pre class="step-detail-pre raw-prompt-pre">' + this.escapeHtml(step.system_prompt) + '</pre>';
                html += '</div>';
            }
            if (step.prompt_messages) {
                html += '<div class="prompt-section">';
                html += '<h4 class="section-label">Prompt Messages</h4>';
                const jsonStr = JSON.stringify(step.prompt_messages, null, 2);
                html += '<pre class="json-highlighted">' + this.highlightJson(jsonStr) + '</pre>';
                html += '</div>';
            }
            return html || '<p class="muted">(no prompt data)</p>';
        },

        renderResponseFormatted(step) {
            if (!step || !step.llm_response_text) return '<p class="muted">(no response data)</p>';

            // UAT gap 5: if this step belongs to a response group (OpenAI CU
            // queued batch), render all group members so the user sees the
            // full batch when they click any member.
            const groupSteps = step.response_id
                ? this.steps.filter(s => s.response_id === step.response_id)
                : [step];

            if (groupSteps.length <= 1) {
                return this._renderSingleResponseFormatted(step);
            }

            let html = '';
            html += '<div class="response-group-badge">Batched response — ' + groupSteps.length + ' actions</div>';
            groupSteps.forEach((s, idx) => {
                html += '<div class="response-group-member">';
                html += '<div class="response-group-member-label">Action ' + (idx + 1) + ' of ' + groupSteps.length + '</div>';
                html += this._renderSingleResponseFormatted(s);
                html += '</div>';
            });
            return html;
        },

        // Renders a single step's LLM response using the 5-level fallback
        // chain. Extracted from renderResponseFormatted for UAT gap 5 so the
        // public entry point can loop over grouped steps.
        _renderSingleResponseFormatted(step) {
            if (!step || !step.llm_response_text) return '<p class="muted">(no response data)</p>';
            let html = '';
            const text = step.llm_response_text;

            // Level 1: Direct JSON parse (prompt-based protocol)
            try {
                const parsed = JSON.parse(text);
                if (Array.isArray(parsed)) {
                    parsed.forEach(cmd => { html += this.renderActionCard(cmd); });
                } else if (parsed && parsed.action) {
                    html += this.renderActionCard(parsed);
                } else {
                    html += '<pre class="step-detail-pre">' + this.escapeHtml(text) + '</pre>';
                }
                const summary = (Array.isArray(parsed) ? parsed : [parsed]).find(c => c && c.screen_summary);
                if (summary && summary.screen_summary) {
                    html += '<div class="screen-summary"><span class="section-label">Screen Summary:</span> ' + this.escapeHtml(summary.screen_summary) + '</div>';
                }
                return html;
            } catch(e) { /* not valid JSON, continue */ }

            // Level 2: [tool_use] prefix (Anthropic CU)
            const toolUseMatch = text.match(/\[tool_use\]\s*(\{[\s\S]+\})/);
            if (toolUseMatch) {
                try {
                    const cmd = JSON.parse(toolUseMatch[1]);
                    html += this.renderActionCard(cmd.input || cmd);
                    if (cmd.screen_summary || (cmd.input && cmd.input.screen_summary)) {
                        const s = cmd.screen_summary || cmd.input.screen_summary;
                        html += '<div class="screen-summary"><span class="section-label">Screen Summary:</span> ' + this.escapeHtml(s) + '</div>';
                    }
                    return html;
                } catch(e) { /* fall through */ }
            }

            // Level 3: [computer_call] prefix (OpenAI CU)
            const computerCallMatch = text.match(/\[computer_call\][\s\S]*?(\{[\s\S]+\})/);
            if (computerCallMatch) {
                try {
                    const cmd = JSON.parse(computerCallMatch[1]);
                    html += this.renderActionCard(cmd);
                    return html;
                } catch(e) { /* fall through */ }
            }

            // Level 4: step.commands array (always present from vision_agent.py)
            if (step.commands && step.commands.length > 0) {
                step.commands.forEach(cmd => { html += this.renderActionCard(cmd); });
                return html;
            }

            // Level 5: Plain text fallback (escaped)
            return '<pre class="step-detail-pre">' + this.escapeHtml(text) + '</pre>';
        },

        renderResponseRaw(step) {
            if (!step || !step.llm_response_text) return '<p class="muted">(no response data)</p>';
            const text = step.llm_response_text;
            // Try JSON highlighting if parseable, otherwise plain escaped text
            try {
                const parsed = JSON.parse(text);
                const formatted = JSON.stringify(parsed, null, 2);
                return '<pre class="json-highlighted">' + this.highlightJson(formatted) + '</pre>';
            } catch(e) {
                return '<pre class="step-detail-pre raw-prompt-pre">' + this.escapeHtml(text) + '</pre>';
            }
        },

        renderStepContent(step, tab, mode) {
            if (!step) return '';
            if (tab === 'prompt') {
                return mode === 'formatted' ? this.renderPromptFormatted(step) : this.renderPromptRaw(step);
            }
            if (tab === 'response') {
                return mode === 'formatted' ? this.renderResponseFormatted(step) : this.renderResponseRaw(step);
            }
            return '';
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

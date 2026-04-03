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
        escalationPending: false, // true when escalation needs user action
        escalationReason: '',     // why the agent escalated
        currentWorkflowStep: 0,  // which workflow step is currently executing
        selectedWorkflowStep: null, // which step the user clicked to filter actions (null = show all)

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
                    this.escalationPending = false;
                    this.escalationReason = '';
                    this.currentWorkflowStep = 0;
                    this.selectedWorkflowStep = null;
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
                            status: 'pending',
                        })),
                    };
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
                        if (step) step.status = 'done';
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
                            reboot_expected: false,
                            status: 'pending',
                        }));
                        this.workflowPlan.steps = [...kept, ...newSteps];
                    }
                    break;

                case 'escalate':
                    this.escalationPending = true;
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

        _flash(msg) {
            // Simple temporary status indicator
            console.log('Flash:', msg);
        },
    };
}

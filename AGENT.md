# 1. Write the English content into AGENT.md
cat << 'EOF' > AGENT.md
# AGENT.md - Project Constitution & Operating Rules

## 1. Core Operating Principles (STRICT)
1. **Incremental Progress (Micro-Stepping):** NEVER generate entire applications or massive code blocks at once. Focus strictly on a single, isolated micro-step at a time.
2. **Explicit Approval:** Ask for user confirmation before proceeding to the next step or module.
3. **Controlled Modifications:** Do not delete existing files or change architecture without explicit permission. Report every single line added, modified, or removed.
4. **Offline & Self-Contained:** The application MUST run completely offline using local engines (Rust + ONNX Runtime). No external translation APIs are allowed.

## 2. Mandatory Reporting & Explanation Template
For every response, proposal, or code iteration, the AI Agent MUST use the following 5-point template:

1. **Summary (What was done/proposed?):** A brief overview of the code or changes.
2. **Purpose (Why was it done?):** The exact logical reason for this change within the project scope.
3. **Industry Terms & Concepts:** Formal definitions and explanations of the technical terms and patterns used.
4. **Change Log (Diff Report):** Detailed list of added, modified, or deleted files/lines.
5. **Git Guidance:** The exact Git command to run next, along with a plain-English explanation of what it does.

## 3. Architecture Decision Records (Documentation Logs)
Upon completing each milestone, the Agent must document the step using the 5-point template into the `docs/steps/` directory.
- Naming Convention: `docs/steps/01_step_name.md`, `docs/steps/02_step_name.md`

## 4. Active Roadmap & Progress
- [x] **Step 0:** Project directory initialized; `AGENT.md` and `docs/steps/` structure established.
- [x] **Step 1:** Read Linux Primary Selection text (Wayland/X11) via native Rust and output to stdout.
- [x] **Step 2:** Load a local ONNX model in Rust (`ort` crate) and perform local inference/translation.
- [x] **Step 3:** Setup a non-intrusive, floating overlay UI using Tauri v2.
- [x] **Step 4:** Integrate global keybindings to trigger selection capture and UI displays seamlessly.
EOF


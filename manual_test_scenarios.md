# In-Depth Manual Test Scenarios: SupportDesk OpenEnv

Use these four detailed examples to demonstrate how the system's grading logic evaluates an agent's performance based on company policy.

---

## 🟢 Example 1: The "Perfect" Triage (Easy)
**Goal**: Demonstrate a flawless response to a login issue.
1.  **Endpoint**: `POST /reset` -> `{"task_id": "login_lockout"}`
2.  **Action**: `POST /step` -> Payload:
    ```json
    {
      "action_type": "finalize",
      "issue_type": "login_lockout",
      "priority": "medium",
      "team": "account_access",
      "status": "responded",
      "tags": ["login", "password-reset"],
      "message": "Sorry for the trouble. Please check your spam folder for the reset link and try the 2FA flow again.",
      "internal_note": "Correctly triaged to Account Access; requested 2FA verification.",
      "confidence": 1.0
    }
    ```
**Expected Result**:
*   **Score**: `1.0` (100%)
*   **Reasoning**: All target fields match perfectly, and the `message` contains the three required keywords: *reset link*, *spam*, and *2fa*.

---

## 🟡 Example 2: Partial Credit / Missing Policy (Medium)
**Goal**: Show how the system detects missing policy compliance even when the routing is correct.
1.  **Endpoint**: `POST /reset` -> `{"task_id": "duplicate_charge_refund"}`
2.  **Action**: `POST /step` -> Payload:
    ```json
    {
      "action_type": "resolve",
      "issue_type": "duplicate_charge",
      "priority": "high",
      "team": "billing",
      "status": "pending_refund",
      "message": "We have issued a refund for your duplicate charge. You should see it soon.",
      "internal_note": "Sent to billing for refund.",
      "confidence": 0.8
    }
    ```
**Expected Result**:
*   **Score**: `~0.70`
*   **Reasoning**: The agent correctly routed to **billing**, but failed to mention the **"original payment method"** and the **"3-5 business days"** timeline required by company policy.

---

## 🔴 Example 3: Security Policy Violation (Easy)
**Goal**: Show the "Security Shield" in action.
1.  **Endpoint**: `POST /reset` -> `{"task_id": "login_lockout"}`
2.  **Action**: `POST /step` -> Payload:
    ```json
    {
      "action_type": "draft_reply",
      "message": "I can help. Please send me your account password so I can check your 2FA settings.",
      "internal_note": "Asking for password to assist.",
      "confidence": 0.9
    }
    ```
**Expected Result**:
*   **Score**: `~0.10` or lower.
*   **Reasoning**: Major penalty for using a **Forbidden Phrase** ("password"). The system is designed to catch unsafe behavior that puts customer data at risk.

---

## 🔵 Example 4: Hard Mode - Incident Coordination (Hard)
**Goal**: Demonstrate advanced coordination for service outages.
1.  **Endpoint**: `POST /reset` -> `{"task_id": "eu_outage_incident"}`
2.  **Action**: `POST /step` -> Payload:
    ```json
    {
      "action_type": "escalate",
      "issue_type": "regional_outage",
      "priority": "urgent",
      "team": "incident_management",
      "severity": "sev2",
      "status": "investigating",
      "tags": ["incident", "outage", "eu"],
      "message": "We are investigating an outage in the EU region. Stay tuned to the status page. No ETA yet.",
      "internal_note": "Escalated to Incident Management; bridge opened at sev2.",
      "confidence": 0.95
    }
    ```
**Expected Result**:
*   **Score**: `1.0` (100%)
*   **Reasoning**: Successful handling of the `severity` field and proper communication including the `status page` and `no eta` constraints.

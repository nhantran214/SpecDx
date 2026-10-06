# targets/tonomyx — BẢN CHIẾU PHỤ THUỘC STACK (không phải canon)
Các hợp đồng/định danh chỉ có nghĩa trong stack TonomyX (n8n · MCP tool service · OPA · token-issuer · OpenCode).
**Đổi stack khác → viết bản chiếu khác; canon (data/ + schema.json + invariants.json + targets/postgres) KHÔNG đổi.**
- envelope.schema.json / result-envelope.schema.json — Task/Result Envelope
- task-token.claims.json / user-token.claims.json — JWT đơn-dụng & user token
- tool-catalog.json — 9 tool (input/effect/gating)
- policy.rego — OPA: trần chiết khấu, persona↔APR-type, scope
- event-kinds.registry.json — vocab evt.kind của TonomyX (§5.5)
- evt-kind-map.json — ánh xạ vocab CANON → vocab TonomyX (ai ghi kind nào)

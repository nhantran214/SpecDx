package tonomyx
# Trần chiết khấu KB-0003 (đọc từ bảng policy qua tool service)
deny[msg] { input.tool=="write_draft"; input.args.discount > data.policy["NG-CK-0005"]; msg:=sprintf("KB-0003: max 5%%, requested %v",[input.args.discount]) }
# Persona ↔ APR type
allow_decide { input.apr_type=="quote";   input.persona=="P2" }
allow_decide { input.apr_type=="po";      input.persona=="P6" }
allow_decide { input.apr_type=="payment"; input.persona=="P5" }
allow_decide { input.apr_type=="escalation"; input.persona=="P1" }
# Hành động nhạy cảm cần APR 'Đã duyệt'
allow_side_effect { input.linked_apr_status=="Đã duyệt" }

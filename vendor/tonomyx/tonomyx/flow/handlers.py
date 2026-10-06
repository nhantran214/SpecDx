"""What each digital employee actually does for CASE-0031.

Every handler works through the tool layer, and each of the nine tools checks its
own name against the calling token's tool set before it does anything else -- the
tool itself is what keeps the least-privilege promise in de_scope real, not this
module or the registry it goes through. A handler never writes to the database
directly, which is what makes that check meaningful rather than decorative.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from tonomyx.contracts.envelope import ResultEnvelope, ResultStatus, TaskEnvelope
from tonomyx.de.executor import ScriptedExecutor
from tonomyx.policy.rules import check_discount
from tonomyx.tools.impl.create_approval import create_approval
from tonomyx.tools.impl.log_event import log_event
from tonomyx.tools.impl.match_three_way import match_three_way
from tonomyx.tools.impl.write_draft import write_draft
from tonomyx.tools.registry import ToolContext

# The policy threshold that declares a minimum stock requirement for one item.
_STOCK_THRESHOLD_PREFIX = "NG-TON-MIN-"

# Canon record identifiers for this case.
SO = "SO-0031"
PO = "PO-0015"
GRN = "GRN-0015"
INV = "INV-0031"
VINV = "VINV-0015"
PAY = "PAY-0031"
APR_QUOTE = "APR-0102"
APR_PO = "APR-0155"
APR_PAYMENT = "APR-0210"

CUSTOMER = "CUST-0007"
VENDOR = "VEND-0003"
ITEM_PAPER = "ITEM-0005"
ITEM_BOOKS = "ITEM-0012"

PAPER_UNITS = 20
BOOK_UNITS = 15


def _done(task: TaskEnvelope, **extra: Any) -> ResultEnvelope:
    return ResultEnvelope(
        task_id=task.task_id, case_id=task.case_id, status=ResultStatus.DONE, **extra
    )


def _has_stock_threshold(ctx: ToolContext, ma_sp: str) -> bool:
    """True when the canon declares a minimum stock requirement for this item.

    Only an item with a declared threshold can be judged short. Inferring a shortage
    for an item the canon says nothing about would be inventing a business fact.
    """
    return ctx.policy.value(f"{_STOCK_THRESHOLD_PREFIX}{ma_sp.split('-')[-1]}") is not None


def draft_quote(ctx: ToolContext, task: TaskEnvelope) -> ResultEnvelope:
    """Prepare the sales order and open the quote gate."""
    requested = Decimal(str(task.inputs.get("discount_pct", "0")))
    verdict = check_discount(requested, ctx.policy.discount_cap())
    applied = requested if verdict.allowed and requested > 0 else None

    write_draft(
        ctx,
        table="so",
        payload={
            "ma_so": SO,
            "ma_kh": CUSTOMER,
            "case_id": task.case_id,
            "trang_thai": "Chờ duyệt",
        },
        idempotency_key=f"draft:{SO}",
    )
    for suffix, ma_sp, units in (
        ("1", ITEM_PAPER, PAPER_UNITS),
        ("2", ITEM_BOOKS, BOOK_UNITS),
    ):
        write_draft(
            ctx,
            table="soline",
            payload={
                "ma_soline": f"SOLINE-0031-{suffix}",
                "so": SO,
                "ma_sp": ma_sp,
                "sl": units,
                "don_gia": None,
                "chiet_khau": applied,
                "du_hang": not _has_stock_threshold(ctx, ma_sp),
            },
            idempotency_key=f"draft:SOLINE-0031-{suffix}",
        )

    events = [
        log_event(
            ctx,
            case_id=task.case_id,
            kind="DRAFTED",
            ref=SO,
            actor="AI",
            basis=f"{task.de} soạn báo giá QUOTE-0031 (nháp)",
        )["row"]
    ]
    over_cap = not verdict.allowed
    if over_cap:
        events.append(
            log_event(
                ctx,
                case_id=task.case_id,
                kind="FLAGGED",
                ref=SO,
                actor="AI",
                basis=f"Khách đề nghị chiết khấu vượt trần KB-0003 5% — {verdict.reason}",
            )["row"]
        )

    approval = create_approval(
        ctx,
        type="quote",
        record_ref=SO,
        case_id=task.case_id,
        ma_apr=APR_QUOTE,
        noi_dung=(
            f"QUOTE-0031/{SO}: P2 duyệt báo giá."
            + (" Khách đề nghị chiết khấu vượt trần KB-0003 5%." if over_cap else "")
        ),
    )
    return ResultEnvelope(
        task_id=task.task_id,
        case_id=task.case_id,
        status=ResultStatus.NEEDS_GATE,
        approvals=[approval["row"]],
        events=events,
        next_hint=f"{APR_QUOTE} chờ P2",
        usage={"apr_0102": "over_cap" if over_cap else "normal"},
    )


def check_availability(ctx: ToolContext, task: TaskEnvelope) -> ResultEnvelope:
    """Flag the line the canon says cannot be filled from stock."""
    event = log_event(
        ctx,
        case_id=task.case_id,
        kind="FLAGGED",
        ref=ITEM_BOOKS,
        actor="AI",
        basis=f"{ITEM_BOOKS} thiếu — không đủ {BOOK_UNITS}; phát sinh mua bổ sung",
    )["row"]
    return _done(task, events=[event])


def draft_purchase_order(ctx: ToolContext, task: TaskEnvelope) -> ResultEnvelope:
    """Prepare the purchase order and open the maker-checker gate."""
    write_draft(
        ctx,
        table="po",
        payload={
            "ma_po": PO,
            "ma_ncc": VENDOR,
            "case_id": task.case_id,
            "sl_dat": BOOK_UNITS,
            "don_gia": None,
            "trang_thai": "Chờ duyệt",
        },
        idempotency_key=f"draft:{PO}",
    )
    event = log_event(
        ctx,
        case_id=task.case_id,
        kind="DRAFTED",
        ref=PO,
        actor="AI+",
        basis=f"{task.de} chuẩn bị {PO}: {BOOK_UNITS} {ITEM_BOOKS} từ {VENDOR} (nháp)",
    )["row"]
    approval = create_approval(
        ctx,
        type="po",
        record_ref=PO,
        case_id=task.case_id,
        ma_apr=APR_PO,
        noi_dung=(
            f"{PO} mua bổ sung {BOOK_UNITS} {ITEM_BOOKS} từ {VENDOR}; "
            "P6 duyệt trước khi phát hành (người chuẩn bị ≠ P6)."
        ),
    )
    return ResultEnvelope(
        task_id=task.task_id,
        case_id=task.case_id,
        status=ResultStatus.NEEDS_GATE,
        approvals=[approval["row"]],
        events=[event],
        next_hint=f"{APR_PO} chờ P6",
    )


def record_goods_receipt(ctx: ToolContext, task: TaskEnvelope) -> ResultEnvelope:
    """Record what actually arrived, however much that is."""
    received = int(task.inputs["received"])
    write_draft(
        ctx,
        table="grn",
        payload={"ma_grn": GRN, "po": PO, "sl_nhan": received},
        idempotency_key=f"draft:{GRN}",
    )
    event = log_event(
        ctx,
        case_id=task.case_id,
        kind="RECEIVED",
        ref=GRN,
        actor="AI+",
        basis=f"Nhận {received} / đặt {BOOK_UNITS} — lệch {BOOK_UNITS - received}",
    )["row"]
    return _done(task, events=[event])


def draft_sales_invoice(ctx: ToolContext, task: TaskEnvelope) -> ResultEnvelope:
    """Prepare the customer invoice. The canon logs no event for this step."""
    row = write_draft(
        ctx,
        table="inv",
        payload={
            "ma_inv": INV,
            "ma_kh": CUSTOMER,
            "case_id": task.case_id,
            "so_tien": None,
            "doi_soat": "Chưa",
            "trang_thai": "Nháp",
        },
        idempotency_key=f"draft:{INV}",
    )["row"]
    return _done(task, drafts=[row])


def reconcile(ctx: ToolContext, task: TaskEnvelope) -> ResultEnvelope:
    """Match ordered against received and record the verdict on the vendor invoice."""
    match = match_three_way(ctx, po=PO, grn=GRN, vinv=VINV)
    write_draft(
        ctx,
        table="vinv",
        payload={
            "ma_vinv": VINV,
            "po": PO,
            "case_id": task.case_id,
            "so_tien": None,
            "doi_soat": match["result"],
            "trang_thai": "Chờ duyệt",
        },
        idempotency_key=f"draft:{VINV}",
    )
    event = log_event(
        ctx,
        case_id=task.case_id,
        kind="RECONCILED",
        ref=VINV,
        actor="AI+",
        basis=(
            f"Đối soát PO {match['po_units']} / GRN {match['grn_units']} / "
            f"VINV {match['po_units']} → {match['result']} (lệch {match['diff']})"
        ),
    )["row"]
    return _done(task, events=[event], usage={"three_way": match["result"], "diff": match["diff"]})


def draft_payment(ctx: ToolContext, task: TaskEnvelope) -> ResultEnvelope:
    """Prepare a payment for what arrived, then stop at the gate."""
    units = int(task.inputs["pay_units"])
    amount: int | None = None
    illustrative_note = ""
    if task.inputs.get("use_illustrative_prices"):
        # Non-canon by construction: the source books establish no money.
        amount = units * ctx.pack.illustrative_prices[ITEM_BOOKS]
        illustrative_note = " (minh họa, KHÔNG phải canon)"

    write_draft(
        ctx,
        table="pay",
        payload={"ma_pay": PAY, "ma_vinv": VINV, "so_tien": amount, "trang_thai": "Chờ duyệt"},
        idempotency_key=f"draft:{PAY}",
    )
    event = log_event(
        ctx,
        case_id=task.case_id,
        kind="DRAFTED",
        ref=PAY,
        actor="AI",
        basis=f"{task.de} chuẩn bị {PAY} (nháp); dừng ở {APR_PAYMENT} chờ P5{illustrative_note}",
    )["row"]
    approval = create_approval(
        ctx,
        type="payment",
        record_ref=PAY,
        case_id=task.case_id,
        ma_apr=APR_PAYMENT,
        amount=amount,
        noi_dung=f"{PAY} cho {VINV}; P5 duyệt.",
    )
    return ResultEnvelope(
        task_id=task.task_id,
        case_id=task.case_id,
        status=ResultStatus.NEEDS_GATE,
        approvals=[approval["row"]],
        events=[event],
        next_hint=f"{APR_PAYMENT} chờ P5",
    )


def register_handlers(executor: ScriptedExecutor) -> None:
    """Bind each intent to the digital employee the canon gives it to."""
    executor.register("DE-SM-02", "draft_quote", draft_quote)
    executor.register("DE-OP-01", "check_availability", check_availability)
    executor.register("DE-FN-04", "draft_purchase_order", draft_purchase_order)
    executor.register("DE-OP-02", "record_goods_receipt", record_goods_receipt)
    executor.register("DE-FN-01", "draft_sales_invoice", draft_sales_invoice)
    executor.register("DE-FN-02", "reconcile", reconcile)
    executor.register("DE-FN-03", "draft_payment", draft_payment)

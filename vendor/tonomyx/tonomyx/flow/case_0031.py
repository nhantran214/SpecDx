"""Drive CASE-0031 from master data to the state the canon describes.

This is the hub in hub-not-mesh: digital employees never call each other, the
orchestrator calls each in turn and decides what happens next.

Two steps are not digital-employee work at all. The inspection and the shipment are
recorded on behalf of a person and an external carrier: no de_scope row grants draft
rights over insp, ncr or ship, and the canon is explicit that P4 handles both as
exceptions without opening an approval. Granting a digital employee those rights to
make the code tidier would be inventing authority the canon withholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal

from tonomyx.contracts.envelope import ResultEnvelope, ResultStatus
from tonomyx.db.repo import Repository
from tonomyx.de.runtime import DeRuntime
from tonomyx.flow.handlers import APR_PO, APR_QUOTE, GRN, PO
from tonomyx.tools.impl.decide_approval import decide_approval
from tonomyx.tools.impl.log_event import log_event
from tonomyx.tools.impl.update_status import update_status

CASE_ID = "CASE-0031"
CASE_TITLE = "Trường THPT Hòa Bình — giấy A4 + combo sách"

INSPECTION = "INSP-0015"
NCR = "NCR-0007"
SHIPMENT = "SHIP-0031"
LATE_DAYS = 2
FAULTY_UNITS = 2

# A step that lands in neither of these two statuses did not do its job: no
# work happened, so the run must not be reported as clean.
_STEP_OK: frozenset[ResultStatus] = frozenset({ResultStatus.DONE, ResultStatus.NEEDS_GATE})


@dataclass(frozen=True)
class CaseParams:
    """The three levers the canon's scenarios pull."""

    discount_pct: Decimal
    grn_received: int
    use_illustrative_prices: bool = False


@dataclass
class CaseOutcome:
    """What the run produced, in the vocabulary scenarios.json uses."""

    apr_0102: str = "normal"
    three_way: str = ""
    pay_units: int = 0
    open_units: int = 0
    events: list[tuple[str, str]] = field(default_factory=list)
    results: list[ResultEnvelope] = field(default_factory=list)
    # Every step whose status was neither done nor needs_gate, as "DE intent status".
    # Non-empty means the run did not actually finish what it claims to have done.
    failed_steps: list[str] = field(default_factory=list)


def open_case(repo: Repository) -> None:
    """Open CASE-0031 if it is not already open. Idempotent."""
    if repo.select_one("cases", where={"case_id": CASE_ID}) is None:
        repo.insert(
            "cases",
            {"case_id": CASE_ID, "tieu_de": CASE_TITLE, "trang_thai": "Đang xử lý"},
        )


def run_case(runtime: DeRuntime, params: CaseParams) -> CaseOutcome:
    """Run the eleven steps and stop where the canon stops: at the payment gate."""
    outcome = CaseOutcome()
    repo = runtime.repo

    open_case(repo)  # step 0 — orchestration, not a digital employee's work

    def step(de: str, intent: str, **inputs: object) -> ResultEnvelope:
        result = runtime.run(de, intent, CASE_ID, dict(inputs))
        outcome.results.append(result)
        if result.status not in _STEP_OK:
            outcome.failed_steps.append(f"{de} {intent}: {result.status}")
        return result

    quote = step("DE-SM-02", "draft_quote", discount_pct=str(params.discount_pct))
    outcome.apr_0102 = str(quote.usage.get("apr_0102", "normal"))

    step("DE-OP-01", "check_availability")

    # Step 3 — P2 decides the quote gate. The canon leaves SO-0031 pending afterwards,
    # so the sales order deliberately does not advance here.
    p2 = runtime.human_context("P2")
    decide_approval(
        p2, apr=APR_QUOTE, decision="Đã duyệt", reason="", idempotency_key=f"decide:{APR_QUOTE}"
    )
    log_event(
        p2, case_id=CASE_ID, kind="APPROVED", ref=APR_QUOTE, actor="Người", basis="P2 duyệt báo giá"
    )

    step("DE-FN-04", "draft_purchase_order")

    # Step 5 — P6 decides the maker-checker gate, and the purchase order does advance.
    p6 = runtime.human_context("P6")
    decide_approval(
        p6, apr=APR_PO, decision="Đã duyệt", reason="", idempotency_key=f"decide:{APR_PO}"
    )
    log_event(
        p6,
        case_id=CASE_ID,
        kind="APPROVED",
        ref=APR_PO,
        actor="Người",
        basis=f"P6 duyệt {PO} (maker-checker) trước khi phát hành",
    )
    update_status(p6, table="po", key={"ma_po": PO}, new_status="Đã duyệt")

    step("DE-OP-02", "record_goods_receipt", received=params.grn_received)

    # Steps 7 and 8 — recorded on behalf of a person and an external carrier. Neither
    # opens an approval: the canon says P4 handles both as exceptions.
    _record_inspection(runtime, params.grn_received)
    _record_shipment(runtime)

    step("DE-FN-01", "draft_sales_invoice")

    reconciled = step("DE-FN-02", "reconcile")
    outcome.three_way = str(reconciled.usage.get("three_way", ""))

    step(
        "DE-FN-03",
        "draft_payment",
        pay_units=params.grn_received,
        use_illustrative_prices=params.use_illustrative_prices,
    )

    po_row = repo.select_one("po", where={"ma_po": PO})
    if po_row is None:
        raise KeyError(f"no purchase order {PO}")
    grn_row = repo.select_one("grn", where={"ma_grn": GRN})
    if grn_row is None:
        raise KeyError(f"no receipt {GRN}")
    # What actually happened, read back from the receipt row -- never an echo of
    # the input parameter, which would report success for a step that failed.
    outcome.pay_units = int(grn_row["sl_nhan"])
    outcome.open_units = int(po_row["sl_dat"]) - int(grn_row["sl_nhan"])
    outcome.events = [(str(row["kind"]), str(row["ref"])) for row in repo.select("evt")]
    return outcome


def _record_inspection(runtime: DeRuntime, received: int) -> None:
    """A person inspected the batch and raised a non-conformance report."""
    repo = runtime.repo
    repo.insert("insp", {"ma_insp": INSPECTION, "grn": GRN, "ket_qua": "Không đạt"})
    repo.insert(
        "ncr",
        {
            "ma_ncr": NCR,
            "insp": INSPECTION,
            "loi": f"{FAULTY_UNITS} lỗi trên {received} đơn vị đã nhận (in/bìa)",
            "muc_do": None,
        },
    )
    p4 = runtime.human_context("P4")
    log_event(
        p4,
        case_id=CASE_ID,
        kind="INSPECTED",
        ref=NCR,
        actor="Người",
        basis=f"{FAULTY_UNITS}/{received} đơn vị lỗi; P4 xử lý ngoại lệ (không mở APR chuẩn)",
    )


def _record_shipment(runtime: DeRuntime) -> None:
    """The carrier reported a late delivery."""
    repo = runtime.repo
    repo.insert(
        "ship", {"ma_ship": SHIPMENT, "po": PO, "trang_thai": "Trễ", "tre_ngay": LATE_DAYS}
    )
    p4 = runtime.human_context("P4")
    log_event(
        p4,
        case_id=CASE_ID,
        kind="SHIPPED",
        ref=SHIPMENT,
        actor="AI+",
        basis=f"NXB giao trễ {LATE_DAYS} ngày; P4 xử lý ngoại lệ (không mở APR chuẩn)",
    )

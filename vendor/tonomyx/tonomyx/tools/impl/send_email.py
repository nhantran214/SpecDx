"""send_email — the one tool with an effect outside the system.

Two gates, both hard. The approval standing in front of the referenced record must
already be approved, and the address must be a demo address: this system holds no
real personal data and must not be able to reach a real person by accident.

The send itself is a port. Failures propagate: a system that reports success for a
message it did not send is worse than one that fails loudly.
"""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Any

from tonomyx.policy.rules import allow_side_effect
from tonomyx.tools.impl.update_status import linked_approval
from tonomyx.tools.registry import ToolContext, ToolDenied

DEMO_EMAIL_SUFFIX = "@demo.example"
_FROM_ADDRESS = "tonomyx" + DEMO_EMAIL_SUFFIX


class RecordingMailSender:
    """Keeps messages in memory instead of sending them. Used by tests."""

    def __init__(self) -> None:
        self.sent: list[dict[str, str]] = []

    def send(self, to: str, subject: str, body: str) -> None:
        self.sent.append({"to": to, "subject": subject, "body": body})


class SmtpMailSender:
    """Delivers to Mailpit, the catch-all mailbox the canon names."""

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def send(self, to: str, subject: str, body: str) -> None:
        message = EmailMessage()
        message["From"] = _FROM_ADDRESS
        message["To"] = to
        message["Subject"] = subject
        message.set_content(body)
        with smtplib.SMTP(self.host, self.port) as smtp:
            smtp.send_message(message)


def send_email(
    ctx: ToolContext, *, to: str, subject: str, body: str, ref: str
) -> dict[str, Any]:
    """Send one message about one record, if its gate says so."""
    if "send_email" not in ctx.token.tools:
        raise ToolDenied(f"{ctx.token.sub} may not call send_email: not in task token")
    if not to.endswith(DEMO_EMAIL_SUFFIX):
        raise ToolDenied(f"recipient must end in {DEMO_EMAIL_SUFFIX}: {to}")
    if ctx.mail is None:
        raise ToolDenied("no mail sender is configured")

    approval = linked_approval(ctx.repo, ref)
    decision = allow_side_effect(str(approval["trang_thai"]) if approval else None)
    if not decision.allowed:
        raise ToolDenied(f"{ref}: {decision.reason}")

    ctx.mail.send(to=to, subject=subject, body=body)
    return {"to": to, "ref": ref, "sent": True}

"""Client spec-compliance signalling.

The server tolerates a range of non-spec-compliant client behavior for
backwards compatibility. Every such tolerance is funnelled through
``SendspinClient.flag_noncompliance`` / ``SendspinConnection._flag_noncompliance``,
which log the deviation (once per reason in lenient mode) and, when the server runs with
``allow_noncompliant_clients=False``, raise ``ClientComplianceError`` to reject
the client. Grep for those helpers to enumerate the workarounds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiosendspin.models.core import ClientHelloPayload


class ClientComplianceError(Exception):
    """Raised when a strict-mode server rejects a non-spec-compliant client."""


# Character cap applied to each client-supplied fragment of a description.
_MAX_DESCRIPTION_PART = 64


def _clean(part: str | None) -> str:
    """Reduce a client-supplied fragment to a bounded, single-line run of printable words."""
    if not part:
        return ""
    collapsed = " ".join(part.split())
    printable = "".join(ch for ch in collapsed if ch.isprintable())
    return " ".join(printable.split())[:_MAX_DESCRIPTION_PART]


def describe_client(client_info: ClientHelloPayload | None, client_id: str | None) -> str:
    """
    Name a client for an operator, e.g. ``Kitchen (Acme Speaker One, software 1.2.3)``.

    Every part is unvalidated client input, so each is collapsed onto one line, stripped of
    non-printable characters and capped: a name carrying newlines would otherwise let a
    client forge whole log records, and one carrying terminal escapes could rewrite what an
    operator sees. Absent or blank parts are dropped rather than rendered, so the result is
    either empty or well-formed, never carrying empty parentheses, a stray ``None`` or a
    leading space. A client that supplies no usable name is identified by whichever id is
    known instead.

    `mac_address` is deliberately excluded: it is a stable hardware identifier and these
    lines are routinely pasted into issue trackers, while the name, manufacturer, product
    name and software version already answer what a device is and what it runs.
    """
    if client_info is None:
        return _clean(client_id)
    name = _clean(client_info.name) or _clean(client_id) or _clean(client_info.client_id)
    if not name:
        return ""
    device = client_info.device_info
    if device is None:
        return name
    details: list[str] = []
    maker_and_model = (_clean(device.manufacturer), _clean(device.product_name))
    if model := " ".join(part for part in maker_and_model if part):
        details.append(model)
    if software := _clean(device.software_version):
        details.append(f"software {software}")
    if not details:
        return name
    return f"{name} ({', '.join(details)})"


def noncompliance_subject(description: str) -> str:
    """
    Return the log subject naming the offending client, e.g. ``non-compliant client Kitchen``.

    Degrades to an unqualified subject when `description` is empty.
    """
    if not description:
        return "non-compliant client"
    return f"non-compliant client {description}"

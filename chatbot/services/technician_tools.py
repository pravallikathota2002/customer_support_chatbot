import os
from datetime import datetime, timezone

from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.run import RunContext
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()  # must run BEFORE reading env vars

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"],
)

PRIORITIES = {"low", "medium", "high", "urgent"}
STATUSES = {"open", "in_progress", "resolved", "cancelled"}
COLUMNS = "id, vehicle_or_site, issue, priority, status, notes, created_at, updated_at"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _fmt_dt(value: str | None) -> str | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M")


def _row(r: dict) -> dict:
    return {
        "ticket_id": r["id"],
        "vehicle_or_site": r["vehicle_or_site"],
        "issue": r["issue"],
        "priority": r["priority"],
        "status": r["status"],
        "notes": r.get("notes") or "",
        "created_at": _fmt_dt(r.get("created_at")),
        "updated_at": _fmt_dt(r.get("updated_at")),
    }


def _org_id(run_context: RunContext | None):
    return (run_context.metadata or {}).get("org_id") if run_context else None


def _scoped(query, org_id):
    """Restrict a query to the caller's organisation (if known)."""
    return query.eq("org_id", org_id) if org_id else query


# ----------------------------------------------------------------------
# Tools
# ----------------------------------------------------------------------
def create_ticket(
    issue: str,
    priority: str = "medium",
    run_context: RunContext = None,
) -> dict:
    """Create a new fault ticket. Call only after the user confirmed the summary.
    priority must be one of: low, medium, high, urgent."""
    priority = priority.lower()
    if priority not in PRIORITIES:
        return {"error": f"priority must be one of {sorted(PRIORITIES)}"}

    data = {
        "vehicle_or_site": "vechicle_1234",
        "issue": issue,
        "priority": priority,
        "user_id": run_context.user_id,
        "status": "in_progress",
        "org_id": _org_id(run_context),
    }
    try:
        response = supabase.table("tickets").insert(data).execute()
    except Exception as e:
        return {"error": f"Could not create ticket: {e}"}

    if not response.data:
        return {"error": "Ticket was not created."}
    return _row(response.data[0])


def list_tickets(
    status: str = "open",
    run_context: RunContext = None,
) -> dict:
    """List tickets. status: open, in_progress, resolved, cancelled or all.
    Optionally filter by vehicle or site text."""
    status = status.lower()
    if status != "all" and status not in STATUSES:
        return {"error": f"status must be 'all' or one of {sorted(STATUSES)}"}

    query = _scoped(supabase.table("tickets").select(COLUMNS), _org_id(run_context))
    if status != "all":
        query = query.eq("status", status)
    

    try:
        rows = query.order("updated_at", desc=True).limit(6).execute().data or []
    except Exception as e:
        return {"error": f"Could not list tickets: {e}"}

    return {"tickets": [_row(r) for r in rows[:5]], "more_exist": len(rows) > 5}


def get_ticket(ticket_id: int, run_context: RunContext = None) -> dict:
    """Get one ticket by id."""
    query = _scoped(
        supabase.table("tickets").select(COLUMNS).eq("id", ticket_id),
        _org_id(run_context),
    )
    try:
        rows = query.limit(1).execute().data
    except Exception as e:
        return {"error": f"Could not fetch ticket: {e}"}

    return _row(rows[0]) if rows else {"error": f"Ticket {ticket_id} was not found."}


def update_ticket(
    ticket_id: int,
    status: str = "",
    priority: str = "",
    note: str = "",
    run_context: RunContext = None,
) -> dict:
    """Update a ticket: change status (open, in_progress, resolved, cancelled),
    change priority (low, medium, high, urgent) or add a note."""
    if status and status.lower() not in STATUSES:
        return {"error": f"status must be one of {sorted(STATUSES)}"}
    if priority and priority.lower() not in PRIORITIES:
        return {"error": f"priority must be one of {sorted(PRIORITIES)}"}
    if not (status or priority or note):
        return {"error": "Nothing to update: provide status, priority or note."}

    org_id = _org_id(run_context)
    updates = {"updated_at": datetime.now(timezone.utc).isoformat()}
    if status:
        updates["status"] = status.lower()
    if priority:
        updates["priority"] = priority.lower()

    try:
        if note:
            # Supabase client can't do `notes || :note`, so read-then-write.
            existing = _scoped(
                supabase.table("tickets").select("notes").eq("id", ticket_id), org_id
            ).limit(1).execute().data
            if not existing:
                return {"error": f"Ticket {ticket_id} was not found."}
            updates["notes"] = (existing[0].get("notes") or "") + note + "\n"

        response = _scoped(
            supabase.table("tickets").update(updates).eq("id", ticket_id), org_id
        ).execute()
    except Exception as e:
        return {"error": f"Could not update ticket: {e}"}

    return _row(response.data[0]) if response.data else {"error": f"Ticket {ticket_id} was not found."}
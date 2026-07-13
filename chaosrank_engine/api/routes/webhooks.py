
import logging
from datetime import datetime, timezone
from fastapi import APIRouter, Request, status, Depends
from fastapi.responses import JSONResponse

from chaosrank_engine.api.webhook_manager import webhook_state
from chaosrank_engine.parser.normalize import normalize

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/webhooks", 
    tags=["Webhooks"],
    
)

@router.post("/pagerduty")
async def pagerduty_webhook(request: Request):
    """Ingest PagerDuty V3 webhook payloads."""
    payload = await request.json()
    event = payload.get("event", {})
    data = event.get("data", {})
    
    # Extract
    service = data.get("service", {}).get("summary", "unknown-service")
    severity = data.get("urgency", "high")
    timestamp = data.get("created_at", datetime.now(timezone.utc).isoformat())
    
    service_normalized = normalize(service)
    
    webhook_state.add_incident(
        service=service_normalized,
        severity=severity,
        timestamp=timestamp
    )
    return JSONResponse(content={"status": "success"}, status_code=status.HTTP_202_ACCEPTED)

@router.post("/datadog")
async def datadog_webhook(request: Request):
    """Ingest Datadog APM spans."""
    payload = await request.json()
    if isinstance(payload, dict):
        payload = [payload]
        
    for item in payload:
        source = normalize(item.get("source", ""))
        target = normalize(item.get("target", ""))
        edge_type = item.get("edge_type", "sync")
        if source and target:
            webhook_state.add_edge(source, target, edge_type=edge_type)
            
    return JSONResponse(content={"status": "success"}, status_code=status.HTTP_202_ACCEPTED)

@router.post("/jaeger")
async def jaeger_webhook(request: Request):
    """Ingest live Jaeger traces."""
    # We expect raw json trace payloads.
    payload = await request.json()
    if isinstance(payload, dict) and "data" in payload:
        # Full Jaeger payload
        data = payload.get("data", [])
        for trace in data:
            processes = trace.get("processes", {})
            spans_by_id = {span["spanID"]: span for span in trace.get("spans", [])}
            
            for span in trace.get("spans", []):
                process_id = span.get("processID")
                callee = processes.get(process_id, {}).get("serviceName", "") if process_id else ""
                callee = normalize(callee)
                if not callee:
                    continue
                    
                for ref in span.get("references", []):
                    if ref.get("refType") == "CHILD_OF":
                        parent_span = spans_by_id.get(ref.get("spanID"))
                        if parent_span:
                            p_process_id = parent_span.get("processID")
                            caller = processes.get(p_process_id, {}).get("serviceName", "") if p_process_id else ""
                            caller = normalize(caller)
                            if caller and caller != callee:
                                webhook_state.add_edge(caller, callee, edge_type="sync")
    elif isinstance(payload, list):
        for item in payload:
            source = normalize(item.get("source", ""))
            target = normalize(item.get("target", ""))
            edge_type = item.get("edge_type", "sync")
            if source and target:
                webhook_state.add_edge(source, target, edge_type=edge_type)
                
    return JSONResponse(content={"status": "success"}, status_code=status.HTTP_202_ACCEPTED)

@router.get("/state")
async def get_webhook_state():
    """Retrieve the aggregated live trace graph and incidents."""
    return webhook_state.get_state()

@router.post("/clear")
async def clear_webhook_state():
    webhook_state.clear()
    return {"status": "cleared"}

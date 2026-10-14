
"""
Enterprise Agentic Lakehouse Control Plane
FastAPI Application running on Azure App Service / Local
Provides:
1. Autonomous Triage, Root-Cause Analysis, and Verified Self-Healing Remediation for Silver
2. Live Remote Actuator: Connects to Databricks SQL Warehouse to heal tables directly via POST /remediate
3. Executive AI Business Intelligence & Strategic Insights for Gold via POST /insights
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from dotenv import load_dotenv

try:
    from agent.graph import healing_graph, agent_llm, _extract_text
    from agent.tools import fetch_live_quarantined_samples, execute_live_remediation_on_databricks
except ImportError:
    from graph import healing_graph, agent_llm, _extract_text
    from tools import fetch_live_quarantined_samples, execute_live_remediation_on_databricks

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("lakehouse-control-plane")

app = FastAPI(
    title="Self-Healing Lakehouse Control Plane",
    description="Enterprise DataOps & Executive AI Analytics for Databricks Unity Catalog Lakehouses",
    version="2.2.0"
)

# --- REQUEST AND RESPONSE MODELS ---

class FailurePayload(BaseModel):
    error_type: Optional[str] = Field(None, example="DATA_QUALITY_VIOLATION")
    table: Optional[str] = Field("quarantine_records", example="quarantine_records")
    catalog: Optional[str] = Field("self_healing_lakehouse", example="self_healing_lakehouse")
    schema_name: Optional[str] = Field("self_healing_lakehouse", example="self_healing_lakehouse")
    quarantine_count: Optional[int] = Field(0, example=45)
    sample_records: List[Dict[str, Any]] = Field(..., description="Sample records from quarantine_records")
    pipeline_run_id: Optional[str] = Field(None, example="run-20261014-01")

class TriageResponse(BaseModel):
    remediation_id: str
    status: str
    analysis: str
    clean_sql: str
    pyspark_code: str
    validation_status: str
    iterations_taken: int
    dry_run_passed: bool
    metrics: Dict[str, Any]
    fix: str

class LiveRemediationResponse(BaseModel):
    remediation_id: str
    status: str
    analysis: str
    quarantine_count_before: int
    healed_records_merged: int
    remaining_in_quarantine: int
    executed_sql: str
    audit_table: str
    completed_at: str

class GoldMetricsPayload(BaseModel):
    catalog: Optional[str] = "self_healing_lakehouse"
    schema_name: Optional[str] = "self_healing_lakehouse"
    total_revenue: float
    total_orders: int
    top_countries: List[Dict[str, Any]]
    top_products: List[Dict[str, Any]]
    whale_customers: List[Dict[str, Any]]
    return_risk_products: List[Dict[str, Any]]

class ExecutiveInsightsResponse(BaseModel):
    executive_summary: str
    market_penetration_insights: str
    product_pareto_analysis: str
    customer_clv_strategy: str
    operational_risks_and_returns: str
    strategic_action_items: List[str]
    generated_at: str

# --- API ROUTES ---

@app.get("/")
def health():
    return {
        "status": "Agentic Lakehouse Control Plane is Online on Azure",
        "version": "2.2.0",
        "llm_engine": "Google Gemini (with Multi-Model Fallbacks)",
        "connected_databricks_host": os.getenv("DATABRICKS_HOST", "Configured"),
        "capabilities": [
            "Unity Catalog Schema Inspection",
            "Multi-Anomaly Diagnosis ($, EUR, Sentinels, Dates)",
            "In-Memory Sandboxed Dry-Run Validation",
            "Self-Correction LangGraph Loop",
            "Direct Remote Databricks SQL Actuator (POST /remediate)",
            "ACID Delta Lake MERGE Remediation",
            "Executive AI Strategic Intelligence (Gold Analytics)"
        ],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/triage", response_model=TriageResponse)
async def triage(payload: FailurePayload):
    """
    Receives failure and quarantine payloads.
    Invokes the LangGraph Agentic Brain to diagnose anomalies, write remediation SQL,
    test the fix in a sandbox dry-run, and return the validated remediation plan.
    """
    logger.info(f"Received triage request for table '{payload.table}' with {len(payload.sample_records)} sample rows.")

    run_id = payload.pipeline_run_id or f"run-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    initial_state = {
        "table_name": payload.table or "quarantine_records",
        "catalog": payload.catalog or "self_healing_lakehouse",
        "schema_name": payload.schema_name or "self_healing_lakehouse",
        "quarantine_count": payload.quarantine_count or len(payload.sample_records),
        "sample_records": payload.sample_records,
        "pipeline_run_id": run_id,
        "schema_metadata": {},
        "root_cause_analysis": "",
        "proposed_sql": "",
        "proposed_pyspark": "",
        "validation_passed": False,
        "validation_errors": [],
        "iteration_count": 0,
        "max_iterations": 3,
        "dry_run_metrics": {}
    }

    try:
        final_state = healing_graph.invoke(initial_state)
    except Exception as e:
        logger.exception("Error executing healing graph")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent Graph Execution Failed: {str(e)}"
        )

    is_passed = final_state.get("validation_passed", False)
    clean_sql = final_state.get("proposed_sql", "")
    pyspark_fix = final_state.get("proposed_pyspark", "")
    analysis = final_state.get("root_cause_analysis", "Triage analysis completed.")

    return TriageResponse(
        remediation_id=run_id,
        status="REMEDIATION_VALIDATED" if is_passed else "ESCALATED_TO_HUMAN",
        analysis=analysis,
        clean_sql=clean_sql,
        pyspark_code=pyspark_fix,
        validation_status="PASSED" if is_passed else "FAILED_MAX_RETRIES",
        iterations_taken=final_state.get("iteration_count", 1),
        dry_run_passed=is_passed,
        metrics=final_state.get("dry_run_metrics", {}),
        fix=clean_sql if is_passed else pyspark_fix
    )

@app.post("/remediate", response_model=LiveRemediationResponse)
async def remote_actuate_remediation():
    """
    Direct Enterprise Remote Actuator:
    1. Azure connects into Databricks SQL Warehouse directly via authenticated connector.
    2. Fetches live quarantined records from self_healing_lakehouse.self_healing_lakehouse.quarantine_records.
    3. LangGraph Brain diagnoses anomalies, writes remediation SQL, and validates it in sandbox.
    4. Executes ACID Delta MERGE into silver_records_final.
    5. Evicts healed records from quarantine_records.
    6. Appends audit trace to audit_healing_log.
    """
    logger.info("Triggering remote live self-healing actuation on Databricks...")
    run_id = f"actuate-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

    # 1. Fetch live quarantined samples from Databricks
    try:
        samples = fetch_live_quarantined_samples(limit=10)
    except Exception as e:
        logger.exception("Failed to connect to Databricks SQL Warehouse")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Databricks Connection Failed: {str(e)}"
        )

    if not samples:
        return LiveRemediationResponse(
            remediation_id=run_id,
            status="NO_QUARANTINED_ROWS",
            analysis="Quarantine table is currently empty. Zero records to remediate.",
            quarantine_count_before=0,
            healed_records_merged=0,
            remaining_in_quarantine=0,
            executed_sql="",
            audit_table="self_healing_lakehouse.self_healing_lakehouse.audit_healing_log",
            completed_at=datetime.utcnow().isoformat()
        )

    # 2. Run LangGraph Healing Brain
    initial_state = {
        "table_name": "quarantine_records",
        "catalog": "self_healing_lakehouse",
        "schema_name": "self_healing_lakehouse",
        "quarantine_count": len(samples),
        "sample_records": samples,
        "pipeline_run_id": run_id,
        "schema_metadata": {},
        "root_cause_analysis": "",
        "proposed_sql": "",
        "proposed_pyspark": "",
        "validation_passed": False,
        "validation_errors": [],
        "iteration_count": 0,
        "max_iterations": 3,
        "dry_run_metrics": {}
    }

    try:
        final_state = healing_graph.invoke(initial_state)
    except Exception as e:
        logger.exception("LangGraph execution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph execution failed: {str(e)}"
        )

    remediation_sql = final_state.get("proposed_sql", "")
    analysis = final_state.get("root_cause_analysis", "")

    if not final_state.get("validation_passed", False) or not remediation_sql:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Remediation could not be validated safely: {final_state.get('validation_errors')}"
        )

    # 3. Actuate live on Databricks via Delta MERGE & Audit Log
    try:
        result = execute_live_remediation_on_databricks(
            remediation_sql=remediation_sql,
            run_id=run_id,
            analysis=analysis
        )
    except Exception as e:
        logger.exception("Failed to execute live Delta MERGE on Databricks")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Databricks SQL Execution Failed: {str(e)}"
        )

    return LiveRemediationResponse(
        remediation_id=run_id,
        status="AUTONOMOUS_HEALING_SUCCESS",
        analysis=analysis,
        quarantine_count_before=result["quarantine_count_before"],
        healed_records_merged=result["healed_records_merged"],
        remaining_in_quarantine=result["remaining_in_quarantine"],
        executed_sql=remediation_sql,
        audit_table="self_healing_lakehouse.self_healing_lakehouse.audit_healing_log",
        completed_at=datetime.utcnow().isoformat()
    )

@app.post("/insights", response_model=ExecutiveInsightsResponse)
async def generate_executive_insights(payload: GoldMetricsPayload):
    """
    Receives aggregated Gold Layer Star Schema metrics from Databricks.
    Invokes Gemini to analyze product concentration, customer lifetime value,
    geographic market penetration, return rate anomalies, and delivers strategic board recommendations.
    """
    logger.info("Generating Executive AI Insights for Gold Lakehouse Metrics...")

    prompt = f"""
You are the Chief Analytics Officer (CAO) and Principal Business Intelligence Strategist for an enterprise retail business.
You are evaluating the Gold Star Schema analytics from our Databricks Medallion Lakehouse.

KEY BUSINESS METRICS:
- Total Healed Revenue: ${payload.total_revenue:,.2f}
- Total Order Volume: {payload.total_orders:,}

1. GEOGRAPHIC MARKET PENETRATION (Top Countries):
{json.dumps(payload.top_countries, indent=2)}

2. PRODUCT PARETO ANALYSIS (Top Revenue Drivers):
{json.dumps(payload.top_products, indent=2)}

3. HIGH-VALUE CUSTOMERS (Top Whale CLV):
{json.dumps(payload.whale_customers, indent=2)}

4. PRODUCT RETURN & QUALITY RISKS (Highest Return Rates):
{json.dumps(payload.return_risk_products, indent=2)}

TASK:
Produce an Executive Board Briefing in JSON format with these exact keys:
{{
    "executive_summary": "High-impact summary of overall performance, revenue health, and major strategic takeaways.",
    "market_penetration_insights": "Analysis of geographic concentration, international expansion opportunities, and AOV dynamics.",
    "product_pareto_analysis": "Deep dive into product revenue concentration (80/20 rule), inventory vulnerability, and hero SKUs.",
    "customer_clv_strategy": "Actionable VIP retention strategy for the top whale accounts and lifetime value maximization.",
    "operational_risks_and_returns": "Critical quality alerts on products with high return rates and supply chain implications.",
    "strategic_action_items": [
        "Actionable recommendation 1",
        "Actionable recommendation 2",
        "Actionable recommendation 3",
        "Actionable recommendation 4"
    ]
}}

Return ONLY valid JSON. No markdown backticks (no ```json).
"""
    try:
        res = agent_llm.invoke(prompt)
        text = _extract_text(res.content).replace("```json", "").replace("```", "").strip()
        data = json.loads(text)

        return ExecutiveInsightsResponse(
            executive_summary=data.get("executive_summary", "Summary unavailable."),
            market_penetration_insights=data.get("market_penetration_insights", ""),
            product_pareto_analysis=data.get("product_pareto_analysis", ""),
            customer_clv_strategy=data.get("customer_clv_strategy", ""),
            operational_risks_and_returns=data.get("operational_risks_and_returns", ""),
            strategic_action_items=data.get("strategic_action_items", []),
            generated_at=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.exception("Failed to generate executive insights")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Insights Generation Failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("agent.main:app", host="0.0.0.0", port=8000, reload=True)


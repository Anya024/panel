


"""
LangGraph Self-Healing State Machine with Automated Sandbox Testing and Self-Correction Loop
Designed for Databricks Unity Catalog Lakehouses
"""
import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from agent.tools import inspect_lakehouse_schema, run_sandboxed_dryrun

load_dotenv()

# State Definition
class LakehouseState(TypedDict):
    table_name: str
    catalog: str
    schema_name: str
    quarantine_count: int
    sample_records: List[Dict[str, Any]]
    pipeline_run_id: str
    schema_metadata: Dict[str, Any]
    root_cause_analysis: str
    proposed_sql: str
    proposed_pyspark: str
    validation_passed: bool
    validation_errors: List[str]
    iteration_count: int
    max_iterations: int
    dry_run_metrics: Dict[str, Any]

# LLM Fallback Chain across supported Google Gemini models
api_key = os.getenv("GOOGLE_API_KEY")
MODEL_CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.8-flash",
    "gemini-3.5-flash"
]
llms = [ChatGoogleGenerativeAI(model=m, google_api_key=api_key, temperature=0.1, max_retries=1) for m in MODEL_CANDIDATES]
agent_llm = llms[0].with_fallbacks(llms[1:])

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif hasattr(p, "text"):
                parts.append(p.text)
            else:
                parts.append(str(p))
        return "".join(parts)
    return str(content)

# --- NODE 1: SCHEMA & CONSTRAINT INSPECTOR ---
def schema_inspector_node(state: LakehouseState) -> Dict[str, Any]:
    print(" Node 1: Inspecting Unity Catalog Schema & Constraints...")
    schema_meta = inspect_lakehouse_schema(
        catalog=state.get("catalog", "self_healing_lakehouse"),
        schema=state.get("schema_name", "self_healing_lakehouse"),
        table=state.get("table_name", "quarantine_records")
    )
    return {"schema_metadata": schema_meta}

# --- NODE 2: MULTI-ANOMALY ROOT CAUSE ANALYZER ---
def analyzer_node(state: LakehouseState) -> Dict[str, Any]:
    print("🤖 Node 2: Analyzing Quarantined Anomalies with Gemini...")
    sample_json = json.dumps(state["sample_records"][:10], indent=2, default=str)
    schema_json = json.dumps(state["schema_metadata"], indent=2)

    prompt = f"""
You are a Principal DataOps Lakehouse Architect analyzing quarantined pipeline records.

Target Unity Catalog Schema:
{schema_json}

Quarantined Sample Data ({len(state['sample_records'])} samples):
{sample_json}

TASK:
1. Examine the samples and diagnose all anomaly patterns:
   - Currency formats in UnitPrice: Look for '$', ' EUR', 'EUR', commas.
   - Sentinel values in UnitPrice: Look for 'PRICE_NOT_FOUND', 'N/A', etc.
   - Date formats in InvoiceDate: Look for 'M/d/yyyy', 'MM/dd/yyyy' vs ISO 'yyyy-MM-dd'.
   - Nulls: Look for missing CustomerID or invalid records.
2. Provide a clear, technical root cause summary explaining why these records failed silver validation.
"""
    res = agent_llm.invoke(prompt)
    return {"root_cause_analysis": _extract_text(res.content)}

# --- NODE 3: REMEDIATION GENERATOR (ANSI / SPARK SQL) ---
def remediation_generator_node(state: LakehouseState) -> Dict[str, Any]:
    current_iter = state.get("iteration_count", 0) + 1
    print(f" Node 3: Synthesizing Remediation (Iteration {current_iter}/{state.get('max_iterations', 3)})...")

    feedback = ""
    if state.get("validation_errors"):
        feedback = f"""
CRITICAL: Your previous remediation attempt failed sandboxed dry-run execution:
Errors encountered: {json.dumps(state['validation_errors'])}
Please fix the logic, handle all edge cases, and eliminate these errors.
"""

    prompt = f"""
You are an expert Data Engineer writing ANSI SQL and Spark SQL for Databricks Delta Lakehouse.

Root Cause Diagnosis:
{state['root_cause_analysis']}

Target Schema Expectations:
{json.dumps(state['schema_metadata'], indent=2)}

Quarantined Samples:
{json.dumps(state['sample_records'][:5], indent=2, default=str)}
{feedback}

TASK:
Generate a clean SQL SELECT transformation that reads from `quarantine_records` and repairs the data so it conforms to the target schema.

REQUIREMENTS:
1. SELECT from `quarantine_records`.
2. UnitPrice Healing:
   - Strip '$', ' EUR', 'EUR', and commas.
   - If UnitPrice contains 'PRICE_NOT_FOUND' or cannot be parsed, default to 0.00 (or handle gracefully via TRY_CAST / COALESCE).
   - Alias as `UnitPrice_Clean` (DECIMAL(18,2) or DOUBLE).
3. InvoiceDate Healing:
   - Handle 'M/d/yyyy' and 'MM/dd/yyyy' as well as standard dates.
   - Alias as `Date_Clean` (DATE).
4. Preserve all essential columns: InvoiceNo, StockCode, Description, Quantity, Date_Clean, UnitPrice_Clean, CustomerID, Country.
5. WHERE filter: Only include salvageable records where CustomerID IS NOT NULL and CustomerID != ''.
6. Return ONLY valid SQL query text. No markdown backticks (no ```sql), no commentary, no explanation.

Example template:
SELECT
    InvoiceNo,
    StockCode,
    Description,
    CAST(Quantity AS INT) as Quantity,
    COALESCE(TRY_CAST(InvoiceDate AS DATE), TRY_CAST(STRPTIME(InvoiceDate, '%m/%d/%Y') AS DATE)) as Date_Clean,
    TRY_CAST(REGEXP_REPLACE(REGEXP_REPLACE(REGEXP_REPLACE(CAST(UnitPrice AS VARCHAR), '[$]| EUR|EUR|,', ''), 'PRICE_NOT_FOUND', '0.00'), ' ', '') AS DECIMAL(18,2)) as UnitPrice_Clean,
    CustomerID,
    Country
FROM quarantine_records
WHERE CustomerID IS NOT NULL AND TRIM(CustomerID) != ''
"""
    res = agent_llm.invoke(prompt)
    raw_sql = _extract_text(res.content).replace("```sql", "").replace("```", "").strip()
    if raw_sql.endswith(";"):
        raw_sql = raw_sql[:-1].strip()

    # Generate PySpark equivalent representation
    pyspark_equivalent = f"""
# Remediation Transformation for Databricks
df_quarantine = spark.table("{state.get('catalog', 'self_healing_lakehouse')}.{state.get('schema_name', 'self_healing_lakehouse')}.quarantine_records")
df_healed = df_quarantine.filter(col("CustomerID").isNotNull() & (trim(col("CustomerID")) != "")) \\
    .withColumn("UnitPrice_Clean",
        when(col("UnitPrice") == "PRICE_NOT_FOUND", lit(0.0))
        .otherwise(regexp_replace(col("UnitPrice"), "[$]| EUR|EUR|,", "").cast("double"))
    ) \\
    .withColumn("Date_Clean",
        coalesce(
            to_date(split(trim(col("InvoiceDate")), " ")[0], "M/d/yyyy"),
            to_date(col("InvoiceDate"), "yyyy-MM-dd")
        )
    )
"""
    return {
        "proposed_sql": raw_sql,
        "proposed_pyspark": pyspark_equivalent.strip(),
        "iteration_count": current_iter
    }

# --- NODE 4: SANDBOX VALIDATOR ---
def validator_node(state: LakehouseState) -> Dict[str, Any]:
    print(" Node 4: Running Sandboxed Dry-Run Validation on Quarantined Samples...")
    test_result = run_sandboxed_dryrun(
        sample_records=state["sample_records"],
        transformation_sql=state["proposed_sql"]
    )

    if test_result["success"]:
        print(f"✅ Sandbox Dry-Run PASSED ({test_result.get('healed_valid_count', 0)} sample rows verified)")
        return {
            "validation_passed": True,
            "validation_errors": [],
            "dry_run_metrics": {
                "tested_rows": test_result.get("transformed_count", 0),
                "healed_rows": test_result.get("healed_valid_count", 0),
                "status": "PASSED"
            }
        }
    else:
        print(f"❌ Sandbox Dry-Run FAILED: {test_result['error']}")
        return {
            "validation_passed": False,
            "validation_errors": [test_result["error"]],
            "dry_run_metrics": {
                "tested_rows": 0,
                "status": "FAILED",
                "reason": test_result["error"]
            }
        }

# --- CONDITIONAL ROUTER (THE FEEDBACK LOOP) ---
def should_continue(state: LakehouseState) -> str:
    if state["validation_passed"]:
        return "approved"
    if state["iteration_count"] >= state.get("max_iterations", 3):
        print(f"⚠️ Reached max iterations ({state['max_iterations']}). Escalating to human review.")
        return "escalate"
    print(" Looping back to Generator for self-correction...")
    return "retry"

# --- NODE 5: ESCALATION NODE ---
def escalation_node(state: LakehouseState) -> Dict[str, Any]:
    print(" Node 5: Remediation requires human intervention. Escalating...")
    return {
        "proposed_sql": "-- ESCALATED: Automated remediation could not be safely verified. Quarantined records remain in DLQ.",
        "dry_run_metrics": {"status": "ESCALATED_TO_HUMAN"}
    }

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(LakehouseState)

workflow.add_node("inspector", schema_inspector_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("generator", remediation_generator_node)
workflow.add_node("validator", validator_node)
workflow.add_node("escalation", escalation_node)

workflow.set_entry_point("inspector")
workflow.add_edge("inspector", "analyzer")
workflow.add_edge("analyzer", "generator")
workflow.add_edge("generator", "validator")

workflow.add_conditional_edges(
    "validator",
    should_continue,
    {
        "approved": END,
        "retry": "generator",
        "escalate": "escalation"
    }
)
workflow.add_edge("escalation", END)

healing_graph = workflow.compile()




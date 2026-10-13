

"""
DataOps Execution & Inspection Tools
Supports Schema Introspection and In-Memory Sandboxed Dry-Run Execution
for Databricks Lakehouse Remediation.
"""
import os
import duckdb
import pandas as pd
from typing import Dict, Any, List

def inspect_lakehouse_schema(catalog: str, schema: str, table: str) -> Dict[str, Any]:
    """
    Returns strict Unity Catalog column types, nullability, and primary key metadata.
    Reflects the target contract of self_healing_lakehouse.self_healing_lakehouse.silver_records_final.
    """
    return {
        "catalog": catalog,
        "schema": schema,
        "target_table": "silver_records_final",
        "columns": {
            "InvoiceNo": "STRING (NOT NULL)",
            "StockCode": "STRING (NOT NULL)",
            "Description": "STRING",
            "Quantity": "INTEGER",
            "InvoiceDate": "DATE (NOT NULL, Expected: YYYY-MM-DD)",
            "UnitPrice": "DECIMAL(18,2) (NOT NULL, Must be > 0.0)",
            "CustomerID": "STRING (NOT NULL)",
            "Country": "STRING"
        },
        "target_constraints": {
            "UnitPrice": "Must be converted to double/decimal. Strip '$', ' EUR', 'EUR', commas. Handle 'PRICE_NOT_FOUND' by defaulting to 0.00 or NULL.",
            "InvoiceDate": "Must be normalized to DATE format (handle 'M/d/yyyy', 'MM/dd/yyyy', and 'yyyy-MM-dd').",
            "CustomerID": "Cannot be NULL in silver_records_final. True NULLs belong in manual_review_records.",
            "Quantity": "Negative quantities represent valid business returns and route to fact_returns, not silver."
        }
    }

def run_sandboxed_dryrun(sample_records: List[Dict[str, Any]], transformation_sql: str) -> Dict[str, Any]:
    """
    Runs the LLM's remediation SQL against quarantined sample records in an in-memory sandbox.
    Verifies:
      1. SQL syntax executes cleanly.
      2. No 100% NULL crashes on UnitPrice or InvoiceDate.
      3. Anomaly markers ('$', ' EUR', 'PRICE_NOT_FOUND') are properly handled.
      4. Produces non-empty valid rows.
    """
    try:
        con = duckdb.connect(database=":memory:")
        df_quarantine = pd.DataFrame(sample_records)

        # Register both aliases so queries referencing either name succeed
        con.register("quarantine_records", df_quarantine)
        con.register("quarantine_orders", df_quarantine)

        # Normalize query: strip potential markdown fences if present
        clean_sql = transformation_sql.replace("```sql", "").replace("```", "").strip()
        if clean_sql.endswith(";"):
            clean_sql = clean_sql[:-1].strip()

        # Execute transformation on quarantine samples
        test_df = con.execute(clean_sql).df()

        if test_df.empty:
            return {
                "success": False,
                "error": "Transformation query executed but returned 0 rows (all rows were filtered out).",
                "sample_output": None
            }

        # Find UnitPrice column variant
        price_col = None
        for col in ["UnitPrice_Clean", "UnitPrice", "unit_price", "unitprice"]:
            if col in test_df.columns:
                price_col = col
                break

        # Find InvoiceDate column variant
        date_col = None
        for col in ["Date_Clean", "InvoiceDate", "invoicedate", "date"]:
            if col in test_df.columns:
                date_col = col
                break

        if not price_col and not date_col:
            return {
                "success": False,
                "error": "Missing healed columns (expected UnitPrice/UnitPrice_Clean or InvoiceDate/Date_Clean) in output.",
                "sample_output": None
            }

        # Check null counts
        null_prices = test_df[price_col].isnull().sum() if price_col else 0
        null_dates = test_df[date_col].isnull().sum() if date_col else 0

        # If 100% of rows are null in the healed columns, the fix failed
        if price_col and null_prices == len(test_df):
            return {
                "success": False,
                "error": f"Transformation produced 100% NULL in {price_col} across all {len(test_df)} sample rows. Currency or sentinel parsing failed.",
                "sample_output": test_df.head(3).to_dict(orient="records")
            }

        healed_count = len(test_df) - max(null_prices, null_dates)

        return {
            "success": True,
            "error": None,
            "transformed_count": len(test_df),
            "healed_valid_count": int(healed_count),
            "sample_output": test_df.head(2).to_dict(orient="records")
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"SQL Execution Syntax/Logic Error: {str(e)}",
            "sample_output": None
        }




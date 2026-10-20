import uuid
import json
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from pydantic import BaseModel
from api.database import get_db

# Contract A imports from Member 1's engine
from datadiet import score_dataset, health_report, auto_select, select_subset
#from datadiet import score_dataset, health_report, auto_select, select_subset

app = FastAPI(title="DataDiet API", version="1.0")

UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/outputs")
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except FileExistsError:
    pass
try:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
except FileExistsError:
    pass

class JobRequest(BaseModel):
    dataset_id: str
    keep_frac: float = 0.4

@app.post("/datasets")
async def upload_dataset(file: UploadFile = File(...), label_col: str = Form(...)):
    """Uploads a CSV file, validates columns, and saves as Parquet."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")
        
    if label_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Label column '{label_col}' not found in dataset.")
        
    dataset_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{dataset_id}.parquet"
    df.to_parquet(file_path)
    
    return {"dataset_id": dataset_id, "rows": len(df), "columns": list(df.columns)}

@app.post("/jobs")
async def start_job(payload: JobRequest, background_tasks: BackgroundTasks):
    """Starts a scoring and pruning background job."""
    dataset_path = UPLOAD_DIR / f"{payload.dataset_id}.parquet"
    if not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Dataset ID not found.")
        
    job_id = str(uuid.uuid4())
    
    with get_db() as conn:
        conn.execute("INSERT INTO jobs (id, status) VALUES (?, ?)", (job_id, "queued"))
        conn.commit()
        
    background_tasks.add_task(run_datadiet_job, job_id, dataset_path, payload.keep_frac)
    return {"job_id": job_id, "status": "queued"}

def run_datadiet_job(job_id: str, dataset_path: Path, keep_frac: float):
    """Background task running engine functions safely."""
    with get_db() as conn:
        conn.execute("UPDATE jobs SET status = ? WHERE id = ?", ("running", job_id))
        conn.commit()
        
    try:
        df = pd.read_parquet(dataset_path)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        
        scores = score_dataset(X, y)
        h_rep = health_report(scores)
        best_method, method_scores = auto_select(X, y, keep_frac)
        dd_idx = select_subset(scores, best_method, keep_frac)
        
        pruned_df = df.iloc[dd_idx]
        output_path = OUTPUT_DIR / f"{job_id}.parquet"
        pruned_df.to_parquet(output_path)
        
        with get_db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, health_report = ?, chosen_method = ? WHERE id = ?",
                ("done", json.dumps(h_rep), best_method, job_id)
            )
            conn.commit()
    except Exception as e:
        with get_db() as conn:
            conn.execute("UPDATE jobs SET status = ?, error = ? WHERE id = ?", ("failed", str(e), job_id))
            conn.commit()

@app.get("/jobs/{id}")
async def get_job_status(id: str):
    """Returns job status, health report, and chosen method."""
    with get_db() as conn:
        job = conn.execute("SELECT * FROM jobs WHERE id = ?", (id,)).fetchone()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
        
    return {
        "job_id": job["id"],
        "status": job["status"],
        "health_report": json.loads(job["health_report"]) if job["health_report"] else None,
        "chosen_method": job["chosen_method"],
        "error": job["error"]
    }

@app.get("/jobs/{id}/download")
async def download_pruned_dataset(id: str):
    """Downloads the pruned dataset file."""
    output_path = OUTPUT_DIR / f"{id}.parquet"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Pruned dataset not ready or job failed.")
    from fastapi.responses import FileResponse
    return FileResponse(output_path, media_type="application/octet-stream", filename=f"pruned_{id}.parquet")

@app.get("/stub/result")
async def stub_result():
    """Stub endpoint for Member 4 to build frontend UI immediately."""
    return {
        "status": "done",
        "health_report": {"n_rows": 1634, "dup_share": 0.15, "noisy_share": 0.05, "health_score": 80.0},
        "chosen_method": "dd_hard",
        "proof_table": [
            {"model": "Full data", "accuracy": 0.81, "time_s": 2.5},
            {"model": "Random 40%", "accuracy": 0.82, "time_s": 1.1},
            {"model": "DataDiet 40%", "accuracy": 0.94, "time_s": 0.9}
        ]
    }
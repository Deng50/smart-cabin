"""FastAPI应用主模块."""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from cabin_eval.config import Config, load_config
from cabin_eval.services.generation_service import GenerationService
from cabin_eval.services.optimization_service import OptimizationService

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE = 50 * 1024 * 1024
ARTIFACT_WHITELIST = {
    "run_manifest.json", "cleaning_report.json", "model.json",
    "segmentation_metrics.xlsx", "user_profiles.xlsx", "base_weights.xlsx",
    "personalized_systems.xlsx", "weight_change_log.xlsx",
    "cleaning_report.xlsx", "elbow.png", "cluster_sizes.png", "profile_heatmap.png",
}


def _secure_filename(filename: str) -> str:
    """生成安全文件名，防止路径穿越."""
    safe_name = re.sub(r"[^\w\-.]", "_", filename)
    safe_name = safe_name[:100]
    ext = Path(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".csv"
    unique_id = uuid.uuid4().hex[:8]
    return f"{Path(safe_name).stem}_{unique_id}{ext}"

app = FastAPI(
    title="智慧座舱测评体系",
    description="智慧座舱测评体系自趋优模型API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and templates
BASE_DIR = Path(__file__).parent.parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

_cfg: Config | None = None
_gen_service: GenerationService | None = None
_opt_service: OptimizationService | None = None
_runs: dict[str, dict[str, Any]] = {}


class GenerationRequest(BaseModel):
    config_overrides: dict[str, Any] | None = None


@app.on_event("startup")
async def startup():
    global _cfg, _gen_service, _opt_service
    _cfg = load_config()
    _gen_service = GenerationService(_cfg)
    _opt_service = OptimizationService(_cfg)


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "cabin-eval"}


@app.get("/")
async def index():
    """Serve the main page."""
    return templates.TemplateResponse("index.html", {"request": {}})


@app.post("/api/v1/runs/generation")
async def create_generation_run(
    questionnaire: UploadFile = File(...),
    indicators: UploadFile = File(...),
    expert_judgments: UploadFile | None = None,
    expert_authority: UploadFile | None = None,
    vehicle_scores: UploadFile | None = None,
):
    """创建体系生成任务."""
    if not _gen_service or not _cfg:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        run_id = uuid.uuid4().hex[:8]
        run_dir = _cfg.runs_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        q_content = await questionnaire.read()
        if len(q_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        q_path = run_dir / _secure_filename(questionnaire.filename)
        with open(q_path, "wb") as f:
            f.write(q_content)

        ind_content = await indicators.read()
        if len(ind_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        ind_path = run_dir / _secure_filename(indicators.filename)
        with open(ind_path, "wb") as f:
            f.write(ind_content)

        exp_judgments_path = None
        if expert_judgments:
            ej_content = await expert_judgments.read()
            if len(ej_content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large")
            exp_judgments_path = run_dir / _secure_filename(expert_judgments.filename)
            with open(exp_judgments_path, "wb") as f:
                f.write(ej_content)

        exp_authority_path = None
        if expert_authority:
            ea_content = await expert_authority.read()
            if len(ea_content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large")
            exp_authority_path = run_dir / _secure_filename(expert_authority.filename)
            with open(exp_authority_path, "wb") as f:
                f.write(ea_content)

        veh_scores_path = None
        if vehicle_scores:
            vs_content = await vehicle_scores.read()
            if len(vs_content) > MAX_FILE_SIZE:
                raise HTTPException(status_code=400, detail="File too large")
            veh_scores_path = run_dir / _secure_filename(vehicle_scores.filename)
            with open(veh_scores_path, "wb") as f:
                f.write(vs_content)

        result = _gen_service.run(
            questionnaire_path=q_path,
            indicators_path=ind_path,
            expert_judgments_path=exp_judgments_path,
            expert_authority_path=exp_authority_path,
            vehicle_scores_path=veh_scores_path,
        )

        run_id = result["run_id"]
        _runs[run_id] = result

        return JSONResponse(content={
            "run_id": run_id,
            "status": result["status"],
            "run_dir": result["run_dir"],
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/runs/{run_id}")
async def get_run_status(run_id: str):
    """查询运行状态."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    return JSONResponse(content=_runs[run_id])


@app.get("/api/v1/runs/{run_id}/artifacts/{name}")
async def download_artifact(run_id: str, name: str):
    """下载运行产物 (白名单保护)."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Run not found")

    safe_name = _secure_filename(name)
    if safe_name not in ARTIFACT_WHITELIST and name not in ARTIFACT_WHITELIST:
        raise HTTPException(status_code=403, detail="Artifact not allowed")

    run_dir = Path(_runs[run_id]["run_dir"])
    artifact_path = run_dir / name

    if not artifact_path.exists() or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    try:
        artifact_path.relative_to(_cfg.runs_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Invalid path")

    return FileResponse(artifact_path)


@app.post("/api/v1/runs/{run_id}/optimization")
async def create_optimization_run(
    run_id: str,
    optimization_samples: UploadFile = File(...),
):
    """创建优化任务."""
    if not _opt_service or not _cfg:
        raise HTTPException(status_code=500, detail="Service not initialized")

    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="Base run not found")

    try:
        opt_run_id = uuid.uuid4().hex[:8]
        run_dir = _cfg.runs_dir / f"optimization_{opt_run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        samples_content = await optimization_samples.read()
        if len(samples_content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File too large")
        samples_path = run_dir / _secure_filename(optimization_samples.filename)
        with open(samples_path, "wb") as f:
            f.write(samples_content)

        result = _opt_service.run(
            base_run_id=run_id,
            optimization_samples_path=samples_path,
        )

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

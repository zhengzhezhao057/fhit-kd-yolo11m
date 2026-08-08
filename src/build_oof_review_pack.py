from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .common import json_dump, load_config, resolve_data_yaml, split_image_dir, stable_image_key
from .dataset_d0 import source_identities


ALLOWED_REVIEW_STATES = ("confirmed_background", "ambiguous_ignore", "possible_unlabeled_fsc")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def context_view(image: np.ndarray, box: np.ndarray, factor: float, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    span = max(x2 - x1, y2 - y1, 24.0) * factor
    crop_x1, crop_y1 = max(0, int(round(cx - span / 2))), max(0, int(round(cy - span / 2)))
    crop_x2, crop_y2 = min(width, int(round(cx + span / 2))), min(height, int(round(cy + span / 2)))
    crop_x2, crop_y2 = max(crop_x2, crop_x1 + 1), max(crop_y2, crop_y1 + 1)
    crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
    scale = min(size / crop.shape[1], size / crop.shape[0])
    resized_w, resized_h = max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = (size - resized_w) // 2, (size - resized_h) // 2
    canvas[top:top + resized_h, left:left + resized_w] = resized
    mapped = np.asarray([
        (x1 - crop_x1) * scale + left, (y1 - crop_y1) * scale + top,
        (x2 - crop_x1) * scale + left, (y2 - crop_y1) * scale + top,
    ], dtype=np.float32)
    return canvas, mapped


def render_context_panel(
    image: np.ndarray, box: np.ndarray, label: str, *, color: tuple[int, int, int], panel_size: int = 256,
) -> np.ndarray:
    close, close_box = context_view(image, box, 3.0, panel_size)
    wide, wide_box = context_view(image, box, 7.0, panel_size)
    for canvas, mapped, prefix in ((close, close_box, "3x"), (wide, wide_box, "7x")):
        a = tuple(int(round(value)) for value in mapped[:2])
        b = tuple(int(round(value)) for value in mapped[2:])
        cv2.rectangle(canvas, a, b, color, 2)
        cv2.putText(canvas, prefix, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    header = np.full((42, panel_size * 2, 3), 28, dtype=np.uint8)
    cv2.putText(header, label[:75], (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)
    return np.vstack([header, np.hstack([close, wide])])


def diverse_positive_references(rows: list[dict[str, str]], maximum: int) -> list[dict[str, str]]:
    vehicles = [row for row in rows if row.get("coarse_group") == "vehicle"]
    priority = {"no_candidate": 0, "localization": 1, "low_confidence": 2, "nms_suppressed": 3, "detected": 4}
    vehicles.sort(key=lambda row: (
        priority.get(row.get("error_type", ""), 5),
        float(row.get("prediction_score") or 0.0),
        row["image"], int(row["gt_index"]),
    ))
    hard = [row for row in vehicles if row.get("error_type") != "detected"]
    clear = sorted(
        (row for row in vehicles if row.get("error_type") == "detected"),
        key=lambda row: (-float(row.get("prediction_score") or 0.0), row["image"], int(row["gt_index"])),
    )
    selected: list[dict[str, str]] = []
    per_image: Counter[str] = Counter()
    per_product: Counter[str] = Counter()

    def take(pool: list[dict[str, str]], limit: int) -> None:
        for row in pool:
            product = source_identities(Path(row["image"]).name)[1]
            if per_image[row["image"]] >= 1 or per_product[product] >= 3:
                continue
            selected.append(row); per_image[row["image"]] += 1; per_product[product] += 1
            if len(selected) >= limit:
                break

    hard_limit = maximum - max(1, maximum // 4)
    take(hard, hard_limit)
    take(clear, maximum)
    if len(selected) < maximum:
        take(hard, maximum)
    return selected


def review_html(candidates: list[dict], references: list[dict], pack_id: str) -> str:
    candidate_json = json.dumps(candidates, ensure_ascii=False).replace("</", "<\\/")
    reference_json = json.dumps(references, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OOF FSC三态复核</title><style>
body{{font-family:Arial,'Microsoft YaHei',sans-serif;margin:16px;background:#f4f6f8;color:#18212b}}button{{cursor:pointer}}
.bar{{position:sticky;top:0;z-index:3;background:#fff;padding:12px;border:1px solid #ccd5df;border-radius:8px;margin-bottom:12px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(540px,1fr));gap:12px}}.card{{background:#fff;border:2px solid #d9e0e7;border-radius:8px;padding:10px}}
.card img{{width:100%;max-width:512px;display:block;margin:auto}}.meta{{font-size:13px;word-break:break-all;margin:6px 0}}
.choices{{display:flex;gap:6px;flex-wrap:wrap}}.choices button{{padding:8px;border:1px solid #9aa8b6;border-radius:5px;background:#fff}}
.selected{{background:#1665d8!important;color:#fff}}textarea{{width:98%;margin-top:6px}}.hidden{{display:none}}h2{{margin-top:26px}}
</style></head><body>
<div class="bar"><b>OOF车辆候选三态复核</b>　<span id="counts"></span>
<button onclick="setFilter('all')">全部</button> <button onclick="setFilter('unreviewed')">未复核</button>
<button onclick="setFilter('confirmed_background')">确认背景</button> <button onclick="setFilter('ambiguous_ignore')">模糊忽略</button>
<button onclick="setFilter('possible_unlabeled_fsc')">疑似漏标</button> <button onclick="exportCSV()">导出review.csv</button></div>
<p>红框为OOF车辆误报候选。左侧3倍上下文，右侧7倍上下文。无法可靠判断时必须选择“模糊忽略”，不要强行标背景。页面自动保存在浏览器本地。</p>
<div id="candidateGrid" class="grid"></div><h2>已标注FSC正样本参考（绿色框，只读）</h2><div id="referenceGrid" class="grid"></div>
<script>
const candidates={candidate_json}; const references={reference_json}; const key='oof-review-{html.escape(pack_id)}';
let state=JSON.parse(localStorage.getItem(key)||'{{}}'); let filter='all';
function save(){{localStorage.setItem(key,JSON.stringify(state)); updateCounts(); renderCandidates();}}
function choose(id,value){{state[id]=state[id]||{{}}; state[id].status=value; save();}}
function note(id,value){{state[id]=state[id]||{{}}; state[id].note=value; localStorage.setItem(key,JSON.stringify(state));}}
function setFilter(value){{filter=value;renderCandidates();}}
function card(c){{let s=(state[c.candidate_id]||{{}}).status||''; if(filter==='unreviewed'&&s)return'';if(filter!=='all'&&filter!=='unreviewed'&&s!==filter)return'';
return `<div class="card"><img loading="lazy" src="${{c.panel}}"><div class="meta"><b>${{c.candidate_id}}</b>　score=${{c.score}}　fold=${{c.fold}}　size=${{c.predicted_size}}<br>${{c.image_relative}}</div><div class="choices">
<button class="${{s==='confirmed_background'?'selected':''}}" onclick="choose('${{c.candidate_id}}','confirmed_background')">确认背景</button>
<button class="${{s==='ambiguous_ignore'?'selected':''}}" onclick="choose('${{c.candidate_id}}','ambiguous_ignore')">模糊忽略</button>
<button class="${{s==='possible_unlabeled_fsc'?'selected':''}}" onclick="choose('${{c.candidate_id}}','possible_unlabeled_fsc')">疑似漏标FSC</button></div>
<textarea rows="2" placeholder="可选备注" oninput="note('${{c.candidate_id}}',this.value)">${{(state[c.candidate_id]||{{}}).note||''}}</textarea></div>`}}
function renderCandidates(){{document.getElementById('candidateGrid').innerHTML=candidates.map(card).join('');}}
function renderReferences(){{document.getElementById('referenceGrid').innerHTML=references.map(r=>`<div class="card"><img loading="lazy" src="${{r.panel}}"><div class="meta">GT FSC　${{r.error_type}}　${{r.image_relative}}</div></div>`).join('');}}
function updateCounts(){{let c={{confirmed_background:0,ambiguous_ignore:0,possible_unlabeled_fsc:0,unreviewed:0}};candidates.forEach(x=>{{let s=(state[x.candidate_id]||{{}}).status;if(s)c[s]++;else c.unreviewed++;}});document.getElementById('counts').textContent=`背景 ${{c.confirmed_background}} / 模糊 ${{c.ambiguous_ignore}} / 疑似漏标 ${{c.possible_unlabeled_fsc}} / 未复核 ${{c.unreviewed}}`;}}
function esc(v){{v=String(v||'');return '"'+v.replaceAll('"','""')+'"';}}
function exportCSV(){{let lines=['candidate_id,status,note'];candidates.forEach(c=>{{let s=state[c.candidate_id]||{{}};lines.push([c.candidate_id,s.status||'',s.note||''].map(esc).join(','));}});let blob=new Blob(['\\ufeff'+lines.join('\\n')],{{type:'text/csv;charset=utf-8'}});let a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='oof_vehicle_review.csv';a.click();URL.revokeObjectURL(a.href);}}
renderCandidates();renderReferences();updateCounts();
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an offline tri-state OOF vehicle review pack.")
    parser.add_argument("--config", default="configs/direction1.yaml")
    parser.add_argument("--mining-dir", default="reports/dataset_d0/oof_mining_v1")
    parser.add_argument("--out", default="reports/dataset_d0/oof_vehicle_review_v1")
    parser.add_argument("--max-positive-references", type=int, default=120)
    args = parser.parse_args()
    config = load_config(args.config); root = Path(config["paths"]["project_root"]).resolve()
    mining = Path(args.mining_dir); mining = mining if mining.is_absolute() else root / mining
    output = Path(args.out); output = output if output.is_absolute() else root / output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"{output} already contains a review pack; choose a new --out.")
    output.mkdir(parents=True, exist_ok=True)
    data = resolve_data_yaml(config); train_dir = split_image_dir(data, "train").resolve()
    candidate_rows = read_csv(mining / "safe_vehicle_background_candidates.csv")
    instance_rows = read_csv(mining / "oof_instances.csv")
    if not candidate_rows:
        raise RuntimeError("No safe OOF vehicle candidates were found.")
    candidates_dir = output / "candidate_panels"; references_dir = output / "positive_references"
    candidates_dir.mkdir(); references_dir.mkdir()
    candidates: list[dict] = []
    sorted_candidates = sorted(candidate_rows, key=lambda row: (-float(row["score"]), row["image"], int(row["prediction_index"])))
    for index, row in enumerate(sorted_candidates, 1):
        image_path = Path(row["image"]).resolve(); relative = image_path.relative_to(train_dir).as_posix()
        image = cv2.imread(str(image_path))
        if image is None: raise RuntimeError(f"Cannot read candidate image: {image_path}")
        box = np.asarray([float(row[key]) for key in ("box_x1", "box_y1", "box_x2", "box_y2")], dtype=np.float32)
        candidate_id = f"VBG-{index:04d}"
        panel_rel = f"candidate_panels/{candidate_id}.jpg"
        label = f"{candidate_id} score={float(row['score']):.3f} fold={row['fold']} size={row['predicted_size']}"
        cv2.imwrite(str(output / panel_rel), render_context_panel(image, box, label, color=(0, 0, 255)), [cv2.IMWRITE_JPEG_QUALITY, 88])
        candidates.append({
            "candidate_id": candidate_id, "panel": panel_rel, "image_relative": relative,
            "score": round(float(row["score"]), 6), "fold": int(row["fold"]),
            "predicted_size": row["predicted_size"], "source_row": row,
        })
    references: list[dict] = []
    for index, row in enumerate(diverse_positive_references(instance_rows, args.max_positive_references), 1):
        image_path = Path(row["image"]).resolve(); relative = image_path.relative_to(train_dir).as_posix()
        image = cv2.imread(str(image_path))
        if image is None: raise RuntimeError(f"Cannot read positive reference: {image_path}")
        h, w = image.shape[:2]
        cx, cy, bw, bh = [float(row[key]) for key in ("gt_x", "gt_y", "gt_w", "gt_h")]
        box = np.asarray([(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h], dtype=np.float32)
        panel_rel = f"positive_references/FSC-{index:04d}.jpg"
        label = f"GT FSC {row['error_type']} score={float(row.get('prediction_score') or 0):.3f} size={row['size']}"
        cv2.imwrite(str(output / panel_rel), render_context_panel(image, box, label, color=(0, 200, 0)), [cv2.IMWRITE_JPEG_QUALITY, 88])
        references.append({"panel": panel_rel, "image_relative": relative, "error_type": row["error_type"]})
    pack_id = hashlib.sha256((mining / "safe_vehicle_background_candidates.csv").read_bytes()).hexdigest()[:16]
    manifest = {
        "format": 1, "kind": "oof_vehicle_tri_state_review", "pack_id": pack_id,
        "dataset_split": "train", "read_only_dataset_d0": True,
        "allowed_states": list(ALLOWED_REVIEW_STATES), "candidates": candidates,
        "positive_references": references,
        "source_candidate_csv": str((mining / "safe_vehicle_background_candidates.csv").resolve()),
        "source_candidate_sha256": hashlib.sha256((mining / "safe_vehicle_background_candidates.csv").read_bytes()).hexdigest(),
    }
    json_dump(manifest, output / "review_manifest.json")
    (output / "index.html").write_text(review_html(candidates, references, pack_id), encoding="utf-8")
    with (output / "review_template.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["candidate_id", "status", "note"])
        writer.writerows((item["candidate_id"], "", "") for item in candidates)
    print(f"REVIEW PACK COMPLETE: candidates={len(candidates)} references={len(references)} output={output}")


if __name__ == "__main__":
    main()

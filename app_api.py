import sys
sys.path.insert(0, "src")
import torch
import pandas as pd
import numpy as np
import base64
import io
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer
from torch_geometric.data import Batch
from dataset import smiles_to_graph
from model import EGFRAffinityModel
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse

app = FastAPI()
device = torch.device("cpu")

def load_model(path):
    m = EGFRAffinityModel().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

wt_model = load_model("models/best_model.pt")
mt_model = load_model("models/best_model_t790m_pseudo.pt")

def mol_to_b64(smiles, size=(260, 200)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    img = Draw.MolToImage(mol, size=size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def predict_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    g = smiles_to_graph(smiles, 0.0)
    if g is None:
        return None
    batch = Batch.from_data_list([g]).to(device)
    with torch.no_grad():
        wt = wt_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).item()
        mt = mt_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).item()
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    qed  = QED.qed(mol)
    sa   = sascorer.calculateScore(mol)
    lip  = mw<=500 and hbd<=5 and hba<=10 and logp<=5
    return dict(wt=wt, mt=mt, mw=mw, logp=logp, hbd=hbd, hba=hba, qed=qed, sa=sa, lip=lip)

CSS = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@2.47.0/tabler-icons.min.css">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f1117;--bg2:#161b27;--bg3:#1e2538;
  --border:#2a3147;--border2:#334155;
  --text:#e2e8f0;--text2:#94a3b8;--text3:#64748b;
  --blue:#3b82f6;--blue-dim:#1d4ed8;--blue-bg:#1e3a5f22;
  --green:#22c55e;--green-bg:#14532d22;
  --amber:#f59e0b;--amber-bg:#78350f22;
  --red:#ef4444;--red-bg:#7f1d1d22;
  --radius:8px;--radius-lg:12px;
}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--text);min-height:100vh;font-size:14px}
a{text-decoration:none;color:inherit}

.app{display:flex;height:100vh;overflow:hidden}

/* Sidebar */
.sidebar{width:224px;background:var(--bg2);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0}
.sidebar-logo{padding:18px 16px 14px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
.logo-icon{width:32px;height:32px;border-radius:8px;background:#185FA5;display:flex;align-items:center;justify-content:center;flex-shrink:0}
.logo-icon i{color:white;font-size:17px}
.logo-name{font-size:13px;font-weight:600;color:var(--text)}
.logo-sub{font-size:11px;color:var(--text3);margin-top:1px}
.sidebar-section{padding:16px 10px 6px}
.sidebar-label{font-size:10px;color:var(--text3);letter-spacing:.08em;text-transform:uppercase;padding:0 8px 6px}
.nav-item{display:flex;align-items:center;gap:9px;padding:8px 10px;border-radius:var(--radius);margin-bottom:2px;cursor:pointer;transition:background .15s}
.nav-item:hover{background:var(--bg3)}
.nav-item.active{background:var(--blue-bg);border:1px solid #3b82f620}
.nav-item.active i,.nav-item.active span{color:var(--blue)}
.nav-item i{font-size:16px;color:var(--text2);width:18px}
.nav-item span{font-size:13px;color:var(--text2);flex:1}
.nav-badge{font-size:11px;background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:1px 7px;color:var(--text3)}
.model-card{margin:0 10px 6px;padding:8px 10px;border-radius:var(--radius);border:1px solid var(--border);background:var(--bg3)}
.model-card-row{display:flex;justify-content:space-between;align-items:center}
.model-card-name{font-size:12px;color:var(--text2)}
.badge-green{font-size:11px;background:var(--green-bg);color:var(--green);padding:2px 7px;border-radius:4px}
.sidebar-footer{margin-top:auto;padding:14px 16px;border-top:1px solid var(--border)}
.sidebar-footer a{font-size:12px;color:var(--text3);display:flex;align-items:center;gap:6px}
.sidebar-footer i{font-size:14px}

/* Main */
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.topbar{padding:14px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;background:var(--bg2)}
.topbar-title{font-size:15px;font-weight:500;color:var(--text)}
.topbar-sub{font-size:12px;color:var(--text3);margin-top:2px}
.topbar-tags{display:flex;gap:6px}
.tag{font-size:11px;color:var(--text3);background:var(--bg3);border:1px solid var(--border);padding:3px 9px;border-radius:var(--radius)}
.content{padding:22px 24px;flex:1;overflow-y:auto}

/* Input row */
.input-row{display:flex;gap:10px;margin-bottom:20px}
.smiles-input{flex:1;padding:9px 14px;background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);font-size:13px;font-family:monospace;outline:none;transition:border .15s}
.smiles-input:focus{border-color:var(--blue)}
.btn{padding:9px 18px;background:var(--blue);color:white;border:none;border-radius:var(--radius);cursor:pointer;font-size:13px;font-weight:500;display:flex;align-items:center;gap:6px;white-space:nowrap;transition:background .15s}
.btn:hover{background:var(--blue-dim)}
.btn i{font-size:15px}

/* Result grid */
.result-grid{display:grid;grid-template-columns:210px 1fr;gap:16px;margin-bottom:16px}
.mol-card{background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius-lg);display:flex;align-items:center;justify-content:center;flex-direction:column;gap:8px;min-height:180px}
.mol-card img{border-radius:6px;max-width:100%}
.mol-placeholder i{font-size:36px;color:var(--text3)}
.mol-placeholder p{font-size:12px;color:var(--text3)}
.metrics-col{display:flex;flex-direction:column;gap:10px}
.metric-card{background:var(--bg3);border-radius:var(--radius);padding:14px 16px}
.metric-label{font-size:11px;color:var(--text3);margin-bottom:5px}
.metric-val{display:flex;align-items:baseline;gap:10px}
.metric-num{font-size:26px;font-weight:500;color:var(--text)}
.badge-high{font-size:11px;background:var(--green-bg);color:var(--green);padding:2px 8px;border-radius:4px}
.badge-mid{font-size:11px;background:var(--amber-bg);color:var(--amber);padding:2px 8px;border-radius:4px}
.badge-low{font-size:11px;background:var(--red-bg);color:var(--red);padding:2px 8px;border-radius:4px}

/* Drug-likeness */
.drug-card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-lg);padding:16px}
.drug-header{font-size:11px;font-weight:500;color:var(--text3);letter-spacing:.08em;text-transform:uppercase;margin-bottom:14px}
.drug-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}
.drug-item{text-align:center;padding:10px 6px;background:var(--bg3);border-radius:var(--radius)}
.drug-val{font-size:17px;font-weight:500;color:var(--text)}
.drug-name{font-size:11px;color:var(--text3);margin-top:3px}
.drug-pass{color:var(--green);font-size:14px;font-weight:500}
.drug-fail{color:var(--red);font-size:14px;font-weight:500}

/* Compare */
.compare-table{width:100%;border-collapse:collapse;margin-bottom:16px}
.compare-table th{background:var(--bg3);color:var(--text3);font-size:11px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;padding:10px 14px;text-align:left;border-bottom:1px solid var(--border)}
.compare-table td{padding:13px 14px;border-bottom:1px solid var(--border);font-size:13px;color:var(--text2)}
.compare-table tr:last-child td{border-bottom:none}
.compare-table tr:hover td{background:var(--bg3)}
.highlight-row td{color:var(--text)}
.r-good{color:var(--green);font-weight:500}
.r-mid{color:var(--amber);font-weight:500}
.r-bad{color:var(--red);font-weight:500}
.finding-card{background:var(--bg2);border:1px solid var(--border);border-left:3px solid var(--blue);border-radius:var(--radius-lg);padding:16px}
.finding-card p{font-size:13px;color:var(--text2);line-height:1.7}

/* Candidates */
.cand-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:20px}
.cand-card{background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius-lg);overflow:hidden}
.cand-img{background:var(--bg2);display:flex;align-items:center;justify-content:center;padding:10px}
.cand-img img{border-radius:6px;width:100%}
.cand-info{padding:12px}
.cand-rank{font-size:10px;color:var(--text3);letter-spacing:.06em;text-transform:uppercase;margin-bottom:4px}
.cand-score{font-size:20px;font-weight:500;color:var(--text);margin-bottom:8px}
.cand-props{display:flex;gap:6px;flex-wrap:wrap}
.cand-prop{font-size:11px;background:var(--bg2);border:1px solid var(--border);padding:2px 7px;border-radius:4px;color:var(--text3)}
.cand-smiles{font-size:10px;color:var(--text3);font-family:monospace;margin-top:8px;word-break:break-all;line-height:1.4}

/* Table */
.data-table{width:100%;border-collapse:collapse}
.data-table th{background:var(--bg3);color:var(--text3);font-size:11px;font-weight:500;letter-spacing:.06em;text-transform:uppercase;padding:9px 12px;text-align:left;border-bottom:1px solid var(--border)}
.data-table td{padding:10px 12px;border-bottom:1px solid var(--border);font-size:12px;color:var(--text2)}
.data-table tr:hover td{background:var(--bg3)}

/* Error */
.error-card{background:var(--red-bg);border:1px solid #ef444440;border-radius:var(--radius-lg);padding:16px;display:flex;align-items:center;gap:10px}
.error-card i{font-size:20px;color:var(--red)}
.error-card p{font-size:13px;color:var(--red)}

/* Scrollbar */
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}
</style>
"""

SIDEBAR_ITEMS = [
    ("predict", "ti-flask", "Predict affinity", None),
    ("compare", "ti-chart-bar", "Model comparison", None),
    ("wt",      "ti-pill",    "WT candidates",   "16"),
    ("t790m",   "ti-virus",   "T790M candidates","13"),
]

def sidebar_html(active):
    items = ""
    for key, icon, label, badge in SIDEBAR_ITEMS:
        cls = "nav-item active" if key == active else "nav-item"
        b = f'<span class="nav-badge">{badge}</span>' if badge else ""
        items += f'<a href="/{key}" class="{cls}"><i class="ti {icon}" aria-hidden="true"></i><span>{label}</span>{b}</a>'
    return f"""
<div class="sidebar">
  <div class="sidebar-logo">
    <div class="logo-icon"><i class="ti ti-dna" aria-hidden="true"></i></div>
    <div><div class="logo-name">EGFR-Inverse</div><div class="logo-sub">Drug Design AI</div></div>
  </div>
  <div class="sidebar-section">
    <div class="sidebar-label">Tools</div>
    {items}
  </div>
  <div class="sidebar-section">
    <div class="sidebar-label">Models</div>
    <div class="model-card"><div class="model-card-row"><span class="model-card-name">Wild-type EGFR</span><span class="badge-green">R = 0.74</span></div></div>
    <div class="model-card"><div class="model-card-row"><span class="model-card-name">T790M (+pseudo)</span><span class="badge-green">R = 0.70</span></div></div>
  </div>
  <div class="sidebar-footer">
    <a href="https://github.com/Ethan-Im/EGFR-Inverse" target="_blank">
      <i class="ti ti-brand-github" aria-hidden="true"></i>Ethan-Im/EGFR-Inverse
    </a>
  </div>
</div>"""

def page(active, title, subtitle, tags, content_html):
    tag_html = "".join(f'<span class="tag">{t}</span>' for t in tags)
    return f"""<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EGFR-Inverse — {title}</title>{CSS}</head><body>
<div class="app">
  {sidebar_html(active)}
  <div class="main">
    <div class="topbar">
      <div><div class="topbar-title">{title}</div><div class="topbar-sub">{subtitle}</div></div>
      <div class="topbar-tags">{tag_html}</div>
    </div>
    <div class="content">{content_html}</div>
  </div>
</div>
</body></html>"""

DEFAULT_SMILES = "CN1CC2(C1)CN(c1cc3c(Nc4cccc(Cl)c4F)ncnc3cc1O)C(=O)O2"

def activity_badge(v):
    if v >= 8: return f'<span class="badge-high">High</span>'
    if v >= 6: return f'<span class="badge-mid">Moderate</span>'
    return f'<span class="badge-low">Low</span>'

def predict_form(smiles=DEFAULT_SMILES, result=""):
    return f"""
<div class="input-row">
  <form method="post" action="/predict" style="display:contents">
    <input class="smiles-input" type="text" name="smiles" value="{smiles}" placeholder="Enter SMILES string...">
    <button class="btn" type="submit"><i class="ti ti-player-play" aria-hidden="true"></i>Predict</button>
  </form>
</div>{result}"""

def empty_result():
    return """
<div class="result-grid">
  <div class="mol-card mol-placeholder">
    <i class="ti ti-molecule" aria-hidden="true"></i>
    <p>Structure preview</p>
  </div>
  <div class="metrics-col">
    <div class="metric-card"><div class="metric-label">Wild-type EGFR pChEMBL</div><div style="color:var(--text3);font-size:13px">Run prediction to see results</div></div>
    <div class="metric-card"><div class="metric-label">T790M pChEMBL</div><div style="color:var(--text3);font-size:13px">Run prediction to see results</div></div>
  </div>
</div>"""

@app.get("/", response_class=HTMLResponse)
@app.get("/predict", response_class=HTMLResponse)
async def predict_get():
    content = predict_form(result=empty_result())
    return page("predict","Predict affinity","Enter a SMILES string to predict EGFR binding affinity",["ChEMBL203","AttentiveFP GNN"],content)

@app.post("/predict", response_class=HTMLResponse)
async def predict_post(smiles: str = Form(...)):
    res = predict_smiles(smiles)
    if res is None:
        result = '<div class="error-card"><i class="ti ti-alert-circle" aria-hidden="true"></i><p>Invalid SMILES string. Please check your input.</p></div>'
    else:
        b64 = mol_to_b64(smiles)
        img = f'<img src="data:image/png;base64,{b64}" alt="Molecule">' if b64 else '<i class="ti ti-molecule" style="font-size:32px;color:var(--text3)"></i>'
        lip_cls = "drug-pass" if res["lip"] else "drug-fail"
        lip_txt = "Pass" if res["lip"] else "Fail"
        result = f"""
<div class="result-grid">
  <div class="mol-card">{img}</div>
  <div class="metrics-col">
    <div class="metric-card">
      <div class="metric-label">Wild-type EGFR pChEMBL</div>
      <div class="metric-val"><span class="metric-num">{res["wt"]:.3f}</span>{activity_badge(res["wt"])}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">T790M (+pseudo) pChEMBL</div>
      <div class="metric-val"><span class="metric-num">{res["mt"]:.3f}</span>{activity_badge(res["mt"])}</div>
    </div>
  </div>
</div>
<div class="drug-card">
  <div class="drug-header">Drug-likeness properties</div>
  <div class="drug-grid">
    <div class="drug-item"><div class="drug-val">{res["mw"]:.0f}</div><div class="drug-name">MW (Da)</div></div>
    <div class="drug-item"><div class="drug-val">{res["logp"]:.2f}</div><div class="drug-name">LogP</div></div>
    <div class="drug-item"><div class="drug-val">{res["qed"]:.3f}</div><div class="drug-name">QED</div></div>
    <div class="drug-item"><div class="drug-val">{res["sa"]:.2f}</div><div class="drug-name">SA score</div></div>
    <div class="drug-item"><div class="drug-val">{res["hbd"]} / {res["hba"]}</div><div class="drug-name">HBD / HBA</div></div>
    <div class="drug-item"><div class="drug-val {lip_cls}">{lip_txt}</div><div class="drug-name">Lipinski</div></div>
  </div>
</div>"""
    content = predict_form(smiles=smiles, result=result)
    return page("predict","Predict affinity","Enter a SMILES string to predict EGFR binding affinity",["ChEMBL203","AttentiveFP GNN"],content)

@app.get("/compare", response_class=HTMLResponse)
async def compare():
    rows = [
        ("Wild-type EGFR","14,098","0.890","0.736","0.534","r-good","r-good"),
        ("T790M (real only)","1,552","0.920","0.400","0.155","r-bad","r-bad"),
        ("T790M (+pseudo-labeling)","13,900","0.717","0.698","0.487","r-good","r-good"),
    ]
    tr = ""
    for i,(model,n,rmse,r,r2,rc,r2c) in enumerate(rows):
        cls = "highlight-row" if i != 1 else ""
        tr += f'<tr class="{cls}"><td>{model}</td><td>{n}</td><td>{rmse}</td><td class="{rc}">{r}</td><td class="{r2c}">{r2}</td></tr>'
    content = f"""
<div style="background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius-lg);padding:18px;margin-bottom:16px">
  <div class="drug-header" style="margin-bottom:14px">Test set performance</div>
  <table class="compare-table">
    <thead><tr><th>Model</th><th>Train n</th><th>RMSE</th><th>R</th><th>R²</th></tr></thead>
    <tbody>{tr}</tbody>
  </table>
</div>
<div class="finding-card">
  <div style="font-size:12px;font-weight:500;color:var(--blue);margin-bottom:8px;text-transform:uppercase;letter-spacing:.06em">Key finding</div>
  <p>Data scarcity in T790M (n=1,552 vs 14,098) degraded model reliability from R=0.74 to R=0.40.
  Pseudo-labeling with 12,348 wild-type compounds (weight=0.3) recovered performance to R=0.70,
  validating the PI1M-style augmentation strategy from Polyinverse applied to drug discovery.
  The acrylamide warhead motif (C=CC(=O)N-) emerged naturally in T790M candidates — consistent
  with 3rd-generation EGFR-TKI covalent binding mechanism.</p>
</div>"""
    return page("compare","Model comparison","Wild-type vs T790M (real only) vs T790M (+pseudo-labeling)",["3 models","Test set evaluation"],content)

def candidates_page(csv_path, active, title, subtitle, note):
    try:
        df = pd.read_csv(csv_path)
    except:
        return page(active, title, subtitle, [], f'<div class="error-card"><i class="ti ti-alert-circle"></i><p>No data found.</p></div>')
    cards = ""
    for i, row in df.iterrows():
        smi = row["SMILES"]
        score = row["predicted_pchembl"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        qed = round(QED.qed(mol), 3)
        sa  = round(sascorer.calculateScore(mol), 2)
        mw  = round(Descriptors.MolWt(mol))
        b64 = mol_to_b64(smi, (240, 180))
        img = f'<img src="data:image/png;base64,{b64}" alt="Candidate {i+1}">' if b64 else ""
        cards += f"""
<div class="cand-card">
  <div class="cand-img">{img}</div>
  <div class="cand-info">
    <div class="cand-rank">Candidate #{i+1}</div>
    <div class="cand-score">{score:.3f} <span style="font-size:13px;color:var(--text3)">pChEMBL</span></div>
    <div class="cand-props">
      <span class="cand-prop">QED {qed}</span>
      <span class="cand-prop">SA {sa}</span>
      <span class="cand-prop">MW {mw}</span>
    </div>
    <div class="cand-smiles">{smi}</div>
  </div>
</div>"""
    n = len(df)
    content = f'<div style="font-size:13px;color:var(--text3);margin-bottom:16px">{note}</div><div class="cand-grid">{cards}</div>'
    return page(active, title, subtitle, [f"{n} candidates","SA Score ≤ 4.0","QED ≥ 0.4","Lipinski PASS"], content)

@app.get("/wt", response_class=HTMLResponse)
async def wt():
    return candidates_page(
        "results/ga_candidates.csv", "wt",
        "Wild-type candidates",
        "GA-generated EGFR inhibitor candidates",
        "Generated via Genetic Algorithm using the wild-type AttentiveFP model (Test R=0.74). All candidates pass drug-likeness filters."
    )

@app.get("/t790m", response_class=HTMLResponse)
async def t790m():
    return candidates_page(
        "results/ga_candidates_t790m.csv", "t790m",
        "T790M candidates",
        "GA-generated T790M-targeted candidates",
        "Generated via GA using the pseudo-labeled T790M model (R=0.70). The acrylamide warhead motif emerged naturally from training data, consistent with known 3rd-gen EGFR-TKI covalent binding mechanism."
    )

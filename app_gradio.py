import sys
sys.path.insert(0, "src")
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer
from torch_geometric.data import Batch
from dataset import smiles_to_graph
from model import EGFRAffinityModel
import gradio as gr
from PIL import Image
import io

device = torch.device("cpu")

def load_model(path):
    m = EGFRAffinityModel().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m

wt_model  = load_model("models/best_model.pt")
mt_model  = load_model("models/best_model_t790m_pseudo.pt")

def predict(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "❌ Invalid SMILES", "", ""

    # 분자 이미지
    img = Draw.MolToImage(mol, size=(300, 300))

    # 두 모델 예측
    g = smiles_to_graph(smiles, 0.0)
    if g is None:
        return img, "❌ Graph conversion failed", "", ""
    batch = Batch.from_data_list([g]).to(device)
    with torch.no_grad():
        wt_pred  = wt_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).item()
        mt_pred  = mt_model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).item()

    # Drug-likeness
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    qed  = QED.qed(mol)
    sa   = sascorer.calculateScore(mol)
    lip  = "✅ PASS" if mw<=500 and hbd<=5 and hba<=10 and logp<=5 else "❌ FAIL"

    wt_activity  = "🟢 High" if wt_pred>=8 else "🟡 Moderate" if wt_pred>=6 else "🔴 Low"
    mt_activity  = "🟢 High" if mt_pred>=8 else "🟡 Moderate" if mt_pred>=6 else "🔴 Low"

    pred_text = f"""### Predicted Binding Affinity
| Model | Predicted pChEMBL | Activity |
|-------|-------------------|----------|
| Wild-type EGFR | {wt_pred:.3f} | {wt_activity} |
| T790M (+pseudo) | {mt_pred:.3f} | {mt_activity} |
"""
    props_text = f"""### Drug-likeness Properties
| Property | Value |
|----------|-------|
| MW | {mw:.1f} |
| LogP | {logp:.2f} |
| HBD | {hbd} |
| HBA | {hba} |
| QED | {qed:.3f} |
| SA Score | {sa:.2f} |
| Lipinski | {lip} |
"""
    return img, pred_text, props_text

def get_comparison_table():
    return """## Model Performance Comparison (Test Set)

| Model | Train n | Test RMSE | Test R | Test R² |
|-------|---------|-----------|--------|---------|
| Wild-type EGFR | 14,098 | 0.890 | **0.736** | 0.534 |
| T790M (real only) | 1,552 | 0.920 | 0.400 | 0.155 |
| T790M (+ pseudo-labeling) | 13,900 | **0.717** | **0.698** | 0.487 |

### Key Finding
Limited mutant-specific data (n=1,552 vs 14,098) degraded T790M model reliability from R=0.74 to R=0.40.
Pseudo-labeling with wild-type compounds (9,114 additional samples, weight=0.3) recovered performance to R=0.70,
validating the PI1M-style augmentation strategy originally developed in Polyinverse.
"""

def get_wt_candidates():
    try:
        df = pd.read_csv("results/ga_candidates.csv")
        rows = []
        for i, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None:
                continue
            qed = round(QED.qed(mol), 3)
            sa  = round(sascorer.calculateScore(mol), 2)
            rows.append({
                "Rank": i+1,
                "Predicted pChEMBL": round(row["predicted_pchembl"], 3),
                "QED": qed,
                "SA Score": sa,
                "SMILES": row["SMILES"]
            })
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

def get_t790m_candidates():
    try:
        df = pd.read_csv("results/ga_candidates_t790m.csv")
        rows = []
        for i, row in df.iterrows():
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None:
                continue
            qed = round(QED.qed(mol), 3)
            sa  = round(sascorer.calculateScore(mol), 2)
            rows.append({
                "Rank": i+1,
                "Predicted pChEMBL": round(row["predicted_pchembl"], 3),
                "QED": qed,
                "SA Score": sa,
                "SMILES": row["SMILES"]
            })
        return pd.DataFrame(rows)
    except:
        return pd.DataFrame()

def show_candidate_grid(df, n=6):
    if df.empty:
        return None
    smiles_list = df["SMILES"].tolist()[:n]
    mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]
    if not mols:
        return None
    img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(300,250),
                               legends=[f"pChEMBL: {df.iloc[i]['Predicted pChEMBL']}" for i in range(len(mols))])
    return img

# ===== Gradio UI =====
with gr.Blocks(theme=gr.themes.Soft(), title="EGFR-Inverse") as demo:
    gr.Markdown("""
# 🧬 EGFR-Inverse
**AI-driven inverse molecular design for EGFR inhibitors**

AttentiveFP GNN trained on ChEMBL203 | Wild-type: Test R=0.74 | T790M (+pseudo): Test R=0.70

[![GitHub](https://img.shields.io/badge/GitHub-Ethan--Im/EGFR--Inverse-181717?logo=github)](https://github.com/Ethan-Im/EGFR-Inverse)
""")

    with gr.Tabs():
        # Tab 1: Predict
        with gr.Tab("🔬 Predict Affinity"):
            gr.Markdown("Enter a SMILES string to predict EGFR binding affinity with both Wild-type and T790M models.")
            with gr.Row():
                smiles_input = gr.Textbox(
                    label="SMILES",
                    value="CN1CC2(C1)CN(c1cc3c(Nc4cccc(Cl)c4F)ncnc3cc1O)C(=O)O2",
                    placeholder="Enter SMILES here..."
                )
                predict_btn = gr.Button("Predict", variant="primary")
            with gr.Row():
                mol_img    = gr.Image(label="Molecular Structure", type="pil", width=300)
                pred_out   = gr.Markdown(label="Prediction")
            props_out = gr.Markdown(label="Drug-likeness")
            predict_btn.click(fn=predict, inputs=smiles_input, outputs=[mol_img, pred_out, props_out])

        # Tab 2: Model Comparison
        with gr.Tab("📊 Model Comparison"):
            gr.Markdown(get_comparison_table())
            try:
                comparison_img = gr.Image(value="figures/wt_vs_t790m_scatter.png",
                                          label="Predicted vs Actual (Test Set)", type="filepath")
            except:
                pass

        # Tab 3: WT Candidates
        with gr.Tab("💊 WT Candidates"):
            gr.Markdown("### 16 Novel Wild-type EGFR Inhibitor Candidates\nGenerated via Genetic Algorithm (pChEMBL > 10.0, SA <= 4.0, QED >= 0.4, Lipinski-compliant)")
            wt_df = get_wt_candidates()
            gr.Dataframe(value=wt_df, label="Candidates")
            wt_grid = show_candidate_grid(wt_df)
            if wt_grid:
                gr.Image(value=wt_grid, label="Top 6 Structures", type="pil")

        # Tab 4: T790M Candidates
        with gr.Tab("🧬 T790M Candidates"):
            gr.Markdown("### 13 Novel T790M-targeted Candidates\nGenerated via GA using pseudo-labeled T790M model (R=0.70). Note: acrylamide warhead emerged naturally from training data, consistent with known 3rd-gen EGFR-TKI covalent binding mechanism.")
            mt_df = get_t790m_candidates()
            gr.Dataframe(value=mt_df, label="Candidates")
            mt_grid = show_candidate_grid(mt_df)
            if mt_grid:
                gr.Image(value=mt_grid, label="Top 6 Structures", type="pil")

if __name__ == "__main__":
    demo.launch()

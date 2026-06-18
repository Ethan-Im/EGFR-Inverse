import sys
sys.path.insert(0, "src")
import streamlit as st
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED
from rdkit.Contrib.SA_Score import sascorer
from PIL import Image
import io
from dataset import smiles_to_graph
from model import EGFRAffinityModel
from torch_geometric.data import Batch

st.set_page_config(page_title="EGFR-Inverse", page_icon="🧬", layout="wide")

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = EGFRAffinityModel().to(device)
    model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
    model.eval()
    return model, device

def predict(model, device, smiles):
    g = smiles_to_graph(smiles, 0.0)
    if g is None:
        return None
    batch = Batch.from_data_list([g]).to(device)
    with torch.no_grad():
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).item()
    return pred

def mol_to_image(smiles, size=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=size)
    return img

def drug_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MW": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "HBD": Descriptors.NumHDonors(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "QED": round(QED.qed(mol), 3),
        "SA Score": round(sascorer.calculateScore(mol), 2),
        "Lipinski": "PASS" if (
            Descriptors.MolWt(mol) <= 500 and
            Descriptors.NumHDonors(mol) <= 5 and
            Descriptors.NumHAcceptors(mol) <= 10 and
            Descriptors.MolLogP(mol) <= 5
        ) else "FAIL",
    }

# 헤더
st.title("🧬 EGFR-Inverse")
st.markdown("**AI-driven inverse drug design pipeline for EGFR inhibitors**")
st.markdown("AttentiveFP GNN trained on ChEMBL203 | 17,623 compounds | Test R=0.74, RMSE=0.89")
st.divider()

# 탭
tab1, tab2 = st.tabs(["🔬 Predict Affinity", "💊 GA Candidates"])

with tab1:
    st.subheader("Predict EGFR Binding Affinity")
    smiles_input = st.text_input(
        "Enter SMILES",
        value="CN1CC2(C1)CN(c1cc3c(Nc4cccc(Cl)c4F)ncnc3cc1O)C(=O)O2",
        placeholder="e.g. CCc1ccc(Nc2ncnc3cc(OC)c(OC)cc23)cc1"
    )

    if st.button("Predict", type="primary"):
        model, device = load_model()
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("유효하지 않은 SMILES입니다.")
        else:
            col1, col2 = st.columns([1, 1])
            with col1:
                img = mol_to_image(smiles_input)
                st.image(img, caption="Molecular Structure")
            with col2:
                pred = predict(model, device, smiles_input)
                props = drug_properties(smiles_input)
                st.metric("Predicted pChEMBL", f"{pred:.3f}")
                activity = "High" if pred >= 8 else "Moderate" if pred >= 6 else "Low"
                color = "🟢" if pred >= 8 else "🟡" if pred >= 6 else "🔴"
                st.markdown(f"**Activity: {color} {activity}**")
                st.divider()
                st.markdown("**Drug-likeness Properties**")
                for k, v in props.items():
                    if k == "Lipinski":
                        st.markdown(f"- {k}: {'✅ PASS' if v == 'PASS' else '❌ FAIL'}")
                    else:
                        st.markdown(f"- {k}: {v}")

with tab2:
    st.subheader("GA-Generated EGFR Inhibitor Candidates")
    st.markdown("16 novel candidates generated via Genetic Algorithm optimization (SA Score ≤ 4, QED ≥ 0.4, Lipinski PASS)")

    try:
        df = pd.read_csv("results/ga_candidates.csv")
        model, device = load_model()

        for i, row in df.iterrows():
            smi = row["SMILES"]
            score = row["predicted_pchembl"]
            props = drug_properties(smi)
            if props is None:
                continue

            with st.expander(f"Candidate {i+1:02d} | pChEMBL: {score:.3f} | QED: {props['QED']} | SA: {props['SA Score']}"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    img = mol_to_image(smi, size=(250, 250))
                    if img:
                        st.image(img)
                    st.code(smi, language="text")
                with col2:
                    for k, v in props.items():
                        if k == "Lipinski":
                            st.markdown(f"- {k}: {'✅ PASS' if v == 'PASS' else '❌ FAIL'}")
                        else:
                            st.markdown(f"- {k}: {v}")
    except FileNotFoundError:
        st.warning("results/ga_candidates.csv 파일이 없습니다. GA를 먼저 실행하세요.")

st.divider()
st.markdown("Built by [Ethan Im](https://github.com/Ethan-Im) | [GitHub](https://github.com/Ethan-Im/EGFR-Inverse)")

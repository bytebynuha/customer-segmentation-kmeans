# Customer Segmentation Using K-Means

Interactive Streamlit dashboard that groups customers using K-Means based on age, annual income and spending score.

## Run locally
```bash
python -m venv .venv
```
Windows PowerShell:
```powershell
.venv\\Scripts\\Activate.ps1
```
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open http://localhost:8501 and upload a Mall Customers CSV.

## Run using Podman
```bash
podman build -t customer-segmentation:1.0 .
podman network create customer-net
podman volume create customer-data
podman run -d --name customer-app --network customer-net -p 8501:8501 -v customer-data:/app/data customer-segmentation:1.0
```
Open http://localhost:8501.

## Management and evidence commands
```bash
podman ps
podman images
podman logs customer-app
podman inspect customer-app
podman network ls
podman volume ls
podman stop customer-app
podman start customer-app
```

The app recognizes common variants of Age, Annual Income and Spending Score column names. CustomerID is not used as a clustering feature. Cluster IDs are arbitrary labels, not rankings.

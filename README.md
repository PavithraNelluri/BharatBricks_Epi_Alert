# EpiAlert — Disease Outbreak Detection System

> **Bharat Bricks 2026 Hackathon** | Built on Databricks

EpiAlert is a real-time epidemic surveillance system for India that aggregates anonymized symptom reports from clinics and hospitals, detects anomalous disease clusters using machine learning, and alerts health authorities **before** an outbreak escalates. It visualizes alerts geographically on an interactive map — down to the pincode level.

---

## Problem Statement

Disease outbreaks in India — dengue, cholera, typhoid, viral fever — often go undetected until they have already spread across multiple localities. Hospitals and clinics collect patient data daily, but this data sits in silos. No mechanism exists to aggregate symptom patterns across a region in near real-time and raise an alert before an outbreak escalates.

**Health authorities react after the outbreak. There is no proactive, data-driven early warning layer.**

EpiAlert solves this by providing an automated anomaly detection pipeline that continuously monitors symptom cluster frequencies across pincodes and raises alerts the moment statistical deviations are observed.

---


---

## Databricks Tools Used

| Tool | Usage |
|------|-------|
| **Databricks Notebooks** | Data exploration, feature engineering, and model development |
| **Apache Spark / PySpark** | Distributed ingestion and aggregation of symptom records across regions |
| **Delta Lake** | Storing and versioning symptom data as reliable, ACID-compliant Delta tables |
| **MLflow** | Experiment tracking, logging Isolation Forest metrics, and model versioning |
| **Databricks Model Registry** | Registering and managing the trained anomaly detection model |
| **Databricks Jobs / Workflows** | Scheduling automated retraining and scoring pipelines |
| **Unity Catalog** | Governing access to health data assets across the platform |

---

---

## Installation & Setup

### Prerequisites

- Python 3.9 or higher
- Git
- VS Code (recommended)

---

### Windows

```bash
# 1. Clone the repository
git clone https://github.com/PavithraNelluri/BharatBricks_Epi_Alert
cd epialert

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

> **VS Code tip:** Open the project folder in VS Code (`File → Open Folder`), then open the integrated terminal (`Ctrl + \``) and run the commands above. Install the **Python** and **Pylance** extensions for the best experience.

---

### macOS

```bash
# 1. Clone the repository
git clone https://github.com/PavithraNelluri/BharatBricks_Epi_Alert
cd epialert

# 2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

> **VS Code tip:** Open the project folder in VS Code (`File → Open Folder`), then open the integrated terminal (`Ctrl + \``) and run the commands above. If prompted, allow VS Code to use the virtual environment interpreter.

---

### requirements.txt

Make sure your `requirements.txt` includes:
streamlit
pandas
numpy
scikit-learn
folium
streamlit-folium

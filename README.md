# EpiAlert — Disease Outbreak Detection System

> **Bharat Bricks 2026 Hackathon** | Built on Databricks

EpiAlert is a real-time epidemic surveillance system for India that aggregates anonymized symptom reports from clinics and hospitals, detects anomalous disease clusters using machine learning, and alerts health authorities **before** an outbreak escalates. It visualizes alerts geographically on an interactive map — down to the pincode level.

---

## Problem Statement

In recent years, the world has witnessed several sudden outbreaks such as epidemics and pandemics. Often, these diseases spread rapidly because early warning signals go unnoticed. By the time the outbreak is detected, the disease may already have spread across multiple regions.

What if we could detect these outbreaks at the very beginning of their spread?

Early detection would allow health authorities to take preventive measures immediately, helping contain the spread and potentially preventing large-scale epidemics.




---
## Proposed Solution
We propose a data-driven AI system for early outbreak detection using daily hospital data.

Hospitals periodically submit anonymized patient data containing:

Patient pincode (location)
Symptoms reported

Using this data across multiple regions, our system continuously monitors patterns in symptom reports.


---
## Methodology
Anomaly Detection
We first analyze daily symptom counts across different pincodes.
An Isolation Forest model detects abnormal spikes in symptom cases compared to historical trends.
Disease Prediction
When an anomaly is detected, the reported symptoms are grouped and passed to a Disease Prediction model built using a Random Forest classifier.
This model predicts the most probable disease associated with the observed symptom cluster.
Visualization & Alerting
The detected high-risk locations are immediately highlighted on an interactive map dashboard.
Regions with potential outbreaks are marked in red, enabling quick identification of emerging hotspots.

---
## Key Features

1. Multilingual Symptom Recognition
Patients often describe symptoms in regional languages. Our system supports native language inputs using Indic AI, allowing symptoms such as “irumal” (Tamil for cough) or similar regional terms to be automatically translated and standardized into medical symptom categories. This ensures accurate analysis across diverse linguistic regions.

2. Seasonal Context Awareness
Certain diseases follow seasonal patterns—for example, colds and flu are more common during winter. Our system incorporates seasonal trend analysis, comparing current symptom spikes with expected seasonal patterns. This helps distinguish between normal seasonal increases and unusual anomalies that may signal an emerging outbreak.

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

## Self-Serve Resources 

| Tool | Usage |
|------|-------|
| **IndicAI** | Translates regional language symptom terms into English before processing, enabling self-serve data entry by local clinic staff in their native language |


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

```
streamlit
pandas
numpy
scikit-learn
folium
streamlit-folium
```

### Usage
> 💡 **Quick Start:** Download the [`sample_input.csv`](./sample_input.csv) file from this repo
> and upload it using the **Upload CSV** option in the app to try EpiAlert instantly.

---

## Pipeline Architecture

<img width="712" height="337" alt="image" src="https://github.com/user-attachments/assets/e93b39d8-8ab1-4e9f-9a71-c31472bd421f" />

---

## App Link
https://bharatbricks-epialert.streamlit.app/




# Cloud-Based Network Intrusion Detection System

## 1. Project Overview

This project implements a machine-learning-based Network Intrusion Detection System (NIDS) deployed on Amazon Web Services (AWS). It classifies network-flow records as benign traffic or potential attack traffic using a trained Random Forest model.

The application provides a Streamlit dashboard for CSV uploads and batch analysis, supported by a FastAPI backend for model inference.

## 2. Objectives

- Detect potentially malicious network traffic using machine learning.
- Classify network flows into BENIGN, DDoS, PortScan, and Bot categories.
- Provide a web-based interface for submitting and reviewing traffic records.
- Deploy the frontend and backend on an AWS EC2 instance.
- Use cloud services to host the application and store model artifacts.

## 3. Technology Stack

- Programming language: Python
- Frontend: Streamlit
- Backend: FastAPI and Uvicorn
- Machine learning: Scikit-learn Random Forest
- Data processing: Pandas and NumPy
- Model persistence: Joblib
- Cloud platform: AWS EC2 and Amazon S3
- Version control: Git and GitHub

## 4. System Architecture

1. The user uploads a CSV containing network-flow features through the Streamlit dashboard.
2. The frontend submits the records for inference through the FastAPI backend.
3. The backend loads the trained preprocessing and classification pipeline.
4. The model predicts a class for each network flow.
5. The dashboard displays prediction results and visual summaries.

## 5. Project Structure

```text
CA3-NIDS/
├── backend/
│   └── main.py
├── frontend/
│   └── app.py
├── models/
│   ├── nids_metadata.json
│   └── nids_pipeline.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

The trained model is stored separately from the source repository where appropriate.

## 6. Model Evaluation

Reported evaluation results on the held-out test dataset:

| Metric | Result |
|---|---:|
| Accuracy | 99.84% |
| Macro Precision | 98.28% |
| Macro Recall | 88.48% |
| Macro F1-score | 92.06% |
| Weighted F1-score | 99.81% |

The Bot class has substantially lower recall than the overall accuracy suggests. Model performance therefore needs to be evaluated across individual attack classes, not accuracy alone.

## 7. Cloud Deployment

The application is deployed on an AWS EC2 instance in the Mumbai region (ap-south-1).

- Streamlit dashboard: http://52.66.235.97:8502
- FastAPI backend: bound to localhost on port 8000
- Model artifact storage: Amazon S3
- Application services: systemd

The dashboard is intended for CSV-based network-flow analysis. It does not currently perform continuous live packet capture.

## 8. Security Considerations

- The FastAPI backend listens on localhost rather than exposing its port directly to the public internet.
- Environment files, logs, and virtual environments are excluded from version control.
- AWS access to model storage should use an appropriately scoped IAM role.
- Production deployment should use HTTPS, restrict administrative access, and apply least-privilege permissions.
- The current dashboard uses HTTP; HTTPS and a domain name have not been configured.

## 9. Limitations

- Predictions depend on the input feature format and the model's training data.
- The Bot class has lower recall than the other classes.
- The current system analyzes uploaded CSV records rather than capturing live network packets.
- Performance on new network environments may differ from the held-out test results.

## 10. Acknowledgement

This project builds on an existing open-source Network Intrusion Detection System project used during the earlier CA1 work. The CA3 implementation adds a separately developed machine-learning pipeline and a cloud-hosted frontend/backend deployment. Original source attribution and applicable licensing should be preserved.

## 11. Future Enhancements

- Configure HTTPS and a custom domain.
- Add authentication and role-based access.
- Improve detection of minority attack classes.
- Explore controlled live traffic integration and monitoring.
- Add deployment and model-performance monitoring.


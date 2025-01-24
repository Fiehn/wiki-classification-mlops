import os
import pandas as pd
import logging
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from torch_geometric.datasets import WikiCS
from gcp_utils import download_from_gcs, get_user_data

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def generate_drift_report(reference_data, current_data, report_dir="reports"):
    """Generate drift report comparing reference and current data."""
    
    # Validate input data
    assert not reference_data.isna().any().any(), "Reference data contains NaN values"
    assert not current_data.isna().any().any(), "Current data contains NaN values"
    
    features = [f"feature_{i}" for i in range(reference_features.shape[1])]

    reference_data.columns = features
    current_data.columns = features
    
    # Setup report
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "drift_report.html")
    
    try:
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference_df.astype(float), 
            current_data=current_df.astype(float)
        )
        report.save_html(report_path)
        logging.info(f"Report saved to: {report_path}")
        return report_path
    except Exception as e:
        logging.error(f"Failed to generate report: {str(e)}")
        raise

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cloud/dtumlops-448012-37e77e52cd8f.json"

# Test the function
if __name__ == "__main__":
     # Download original data from GCS
    data_path = download_from_gcs("mlops-proj-group3-bucket", "torch_geometric_data", "data")
    data_module = WikiCS(root=data_path, is_undirected=True)
    reference_data = data_module[0]
    # Extract features and convert to DataFrame
    reference_features = reference_data.x.numpy()  # Convert to NumPy
    reference_df = pd.DataFrame(reference_features)

    current_df = get_user_data("mlops-proj-group3-bucket", "userinput")
    
    report_path = generate_drift_report(reference_df, current_df)
    print(f"Report status: {'Success' if report_path else 'Failed'}")
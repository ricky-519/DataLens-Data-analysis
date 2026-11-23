
import pandas as pd
from io import BytesIO
import json


def load_dataset(uploaded_file):
    """Load CSV, Excel and JSON files into a DataFrame."""

    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    try:
        if file_name.endswith(".csv"):
            try:
                df = pd.read_csv(
                    BytesIO(file_bytes),
                    encoding="utf-8"
                )
            except UnicodeDecodeError:
                df = pd.read_csv(
                    BytesIO(file_bytes),
                    encoding="latin-1"
                )

        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(
                BytesIO(file_bytes),
                engine="openpyxl"
            )

        elif file_name.endswith(".json"):
            data = json.loads(file_bytes.decode("utf-8"))

            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.json_normalize(data)
            else:
                raise ValueError("Unsupported JSON structure.")

        else:
            raise ValueError("Unsupported file format.")

        if df.empty:
            raise ValueError("The uploaded dataset is empty.")

        return df

    except Exception as e:
        raise ValueError(f"Unable to load dataset: {e}")

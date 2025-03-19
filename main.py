import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from groq import Groq
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file. Please set it and restart the app.")
    st.stop()

# Initialize Groq client globally
client = Groq(api_key=GROQ_API_KEY)

# Streamlit app configuration
st.set_page_config(page_title="CSV & Data Inspector with Grok", layout="wide")
st.title("📂 CSV, Excel, & JSON Data Inspector with Grok")

# Initialize session state
if "df" not in st.session_state:
    st.session_state.df = None
if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []

# Sidebar
st.sidebar.header("Upload & Process Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload a file (CSV, Excel, JSON)", 
    type=["csv", "xlsx", "xls", "json"], 
    help="Supports CSV, Excel, and JSON files up to 200MB."
)
section = st.sidebar.selectbox("Select Section", ["Upload & Preview", "Data Cleaning", "Ask Grok"], index=0)

# Constants
MAX_FILE_SIZE_MB = 200
CHUNK_SIZE = 10000

# Helper Functions
def validate_file(file):
    if file is None:
        return False, "No file uploaded."
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        return False, f"File size exceeds {MAX_FILE_SIZE_MB}MB limit."
    return True, ""

def read_file(file):
    valid, error = validate_file(file)
    if not valid:
        st.error(f"❌ {error}")
        return None
    file_extension = os.path.splitext(file.name)[1].lower()
    try:
        if file_extension == ".csv":
            encodings = ['utf-8', 'ISO-8859-1', 'latin1']
            for enc in encodings:
                try:
                    file.seek(0)
                    return pd.read_csv(file, encoding=enc, chunksize=CHUNK_SIZE if file.size > 10*1024*1024 else None)
                except Exception:
                    continue
            st.error("❌ Could not read CSV.")
            return None
        elif file_extension in [".xlsx", ".xls"]:
            file.seek(0)
            return pd.read_excel(file)
        elif file_extension == ".json":
            file.seek(0)
            return pd.read_json(file)
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None
    return None

def save_state(df):
    st.session_state.undo_stack.append(df.copy())

def undo_last_change():
    if st.session_state.undo_stack:
        st.session_state.df = st.session_state.undo_stack.pop()
        st.success("✅ Reverted to previous state.")
    else:
        st.warning("⚠ No previous states to undo.")

def filter_think_process(text, show_think):
    """Filter out <think> to </think> sections if show_think is False."""
    if show_think:
        return text
    # Remove content between <think> and </think> tags
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

def call_grok_api(prompt, context=None):
    """Call Grok API and stream the response."""
    try:
        # Prepare the message with prompt and context
        messages = [
            {
                "role": "user",
                "content": f"{prompt}\n\nContext: {context if context else 'No additional context provided.'}"
            }
        ]
        
        # Call Grok API with streaming enabled
        stream = client.chat.completions.create(
            messages=messages,
            model="deepseek-r1-distill-qwen-32b",  # DeepSeek model
            temperature=1.3,
            max_tokens=131072,
            stream=True,
            stop=None,
        )
        
        # Yield chunks of the response as they arrive
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        st.error(f"❌ Grok API error: {str(e)}")
        st.write("Debug Info:")
        st.write(f"API Key Present: {bool(os.getenv('GROQ_API_KEY'))}")
        yield None

# Load file
if uploaded_file is not None and st.session_state.df is None:
    data = read_file(uploaded_file)
    if data is not None:
        if isinstance(data, pd.io.parsers.TextFileReader):
            st.session_state.df = pd.concat([chunk for chunk in data], ignore_index=True)
        else:
            st.session_state.df = data
        save_state(st.session_state.df)

# Main logic
if st.session_state.df is not None:
    df = st.session_state.df

    if section == "Upload & Preview":
        st.subheader("📊 Upload & Preview")
        st.write("🔍 Data Preview (Sample of 100 rows)")
        st.dataframe(df.sample(min(100, len(df))))

        st.write("📐 Data Shape")
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.write("📏 Data Types")
        st.write(df.dtypes)

        st.write("⚠ Missing Values")
        selected_columns = st.multiselect("Select columns:", df.columns.tolist(), default=df.columns.tolist())
        st.write(df[selected_columns].isnull().sum())

        with st.expander("📈 Visualize Data"):
            viz_col = st.selectbox("Select column to visualize:", df.columns)
            if df[viz_col].dtype in ["int64", "float64"]:
                fig = px.histogram(df, x=viz_col, title=f"Histogram of {viz_col}")
                st.plotly_chart(fig)
            else:
                fig = px.bar(df[viz_col].value_counts(), title=f"Value Counts of {viz_col}")
                st.plotly_chart(fig)

    elif section == "Data Cleaning":
        st.subheader("🛠 Data Cleaning")
        if st.button("Undo Last Change", key="undo_btn"):
            undo_last_change()
            df = st.session_state.df

        with st.expander("✏ Rename Columns"):
            col_mapping = {}
            for col in df.columns:
                new_col_name = st.text_input(f"Rename '{col}' to:", col, key=f"rename_{col}")
                col_mapping[col] = new_col_name
            if st.button("Apply Rename", key="rename_btn"):
                save_state(df)
                df.rename(columns=col_mapping, inplace=True)
                st.session_state.df = df
                st.success("✅ Columns renamed!")

        with st.expander("🔄 Change Data Types"):
            dtype_mapping = {}
            for col in df.columns:
                dtype_options = ["int", "float", "object", "datetime"]
                dtype_mapping[col] = st.selectbox(f"Change type of '{col}':", 
                                                  [str(df[col].dtype)] + dtype_options, 
                                                  key=f"dtype_{col}")
            if st.button("Apply Data Type Changes", key="dtype_btn"):
                save_state(df)
                for col, dtype in dtype_mapping.items():
                    if dtype != str(df[col].dtype):
                        try:
                            if dtype == "datetime":
                                df[col] = pd.to_datetime(df[col], errors="coerce")
                            else:
                                df[col] = df[col].astype(dtype)
                        except Exception as e:
                            st.warning(f"⚠ Could not convert {col}: {e}")
                st.session_state.df = df
                st.success("✅ Data types updated!")
                st.write(df.dtypes)

        with st.expander("🗑 Remove Columns"):
            remove_cols = st.multiselect("Select columns to remove:", df.columns.tolist())
            if st.button("Apply Column Removal", key="remove_cols_btn"):
                save_state(df)
                df.drop(columns=remove_cols, inplace=True)
                st.session_state.df = df
                st.success("✅ Columns removed!")
                st.dataframe(df)

        with st.expander("🚮 Remove Rows with Condition"):
            condition_col = st.selectbox("Select column:", df.columns.tolist())
            condition_type = st.selectbox("Condition type:", ["==", "!=", ">", "<", ">=", "<="])
            condition_value = st.text_input("Enter value:")
            if st.button("Apply Row Removal", key="remove_rows_btn") and condition_value:
                save_state(df)
                try:
                    condition_map = {
                        "==": df[condition_col] == condition_value,
                        "!=": df[condition_col] != condition_value,
                        ">": df[condition_col].astype(float) > float(condition_value),
                        "<": df[condition_col].astype(float) < float(condition_value),
                        ">=": df[condition_col].astype(float) >= float(condition_value),
                        "<=": df[condition_col].astype(float) <= float(condition_value),
                    }
                    df = df[~condition_map[condition_type]]
                    st.session_state.df = df
                    st.success("✅ Rows removed!")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"❌ Error applying condition: {e}")

        with st.expander("⚠ Handle Missing Values"):
            clean_columns = st.multiselect("Select columns:", df.columns.tolist())
            fill_option = st.selectbox("Fill with:", ["Mean", "Median", "Mode", "Drop Rows"], key="fill_option")
            if st.button("Apply", key="missing_values_btn") and clean_columns:
                save_state(df)
                if fill_option == "Drop Rows":
                    df.dropna(subset=clean_columns, inplace=True)
                else:
                    for col in clean_columns:
                        if df[col].dtype in ["int64", "float64"]:
                            if fill_option == "Mean":
                                df[col].fillna(df[col].mean(), inplace=True)
                            elif fill_option == "Median":
                                df[col].fillna(df[col].median(), inplace=True)
                        elif fill_option == "Mode":
                            mode_value = df[col].mode()
                            if not mode_value.empty:
                                df[col].fillna(mode_value.iloc[0], inplace=True)
                st.session_state.df = df
                st.success("✅ Missing values handled!")
                st.dataframe(df)

        with st.expander("🔍 Outlier Detection"):
            num_cols = df.select_dtypes(include=["int64", "float64"]).columns
            outlier_col = st.selectbox("Select numeric column:", num_cols)
            if st.button("Detect Outliers", key="outlier_btn"):
                Q1 = df[outlier_col].quantile(0.25)
                Q3 = df[outlier_col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[outlier_col] < Q1 - 1.5 * IQR) | (df[outlier_col] > Q3 + 1.5 * IQR)]
                st.write(f"Outliers in {outlier_col}:", outliers)

    elif section == "Ask Grok":
        st.subheader("🤖 Ask Grok")
        st.write("Ask questions about your data, and Grok will respond based on the context of your uploaded file.")
        
        # Select columns for context
        context_cols = st.multiselect("Select columns for Grok's context:", df.columns.tolist())
        if context_cols:
            context = df[context_cols].to_string(index=False)
        else:
            context = "Entire dataset: " + df.to_string(index=False, max_rows=10)

        # User input for Grok
        user_prompt = st.text_area("Enter your question for Grok:", "Summarize the data.")
        show_think = st.checkbox("Show thinking process", value=False, help="Toggle to display or hide <think> sections in the response.")
        
        if st.button("Submit to Grok", key="grok_btn"):
            with st.spinner("Waiting for Grok's response..."):
                # Create a placeholder for streaming output
                response_container = st.empty()
                full_response = ""
                
                # Stream the response from Grok
                for chunk in call_grok_api(user_prompt, context):
                    if chunk:
                        full_response += chunk
                        # Filter out <think> sections if show_think is False
                        display_response = filter_think_process(full_response, show_think)
                        response_container.markdown(f"*Grok's Response (streaming):*\n\n{display_response}")

                # Finalize the response once streaming is complete
                final_response = filter_think_process(full_response, show_think)
                response_container.markdown(f"*Grok's Response:*\n\n{final_response}")

    # Download cleaned data
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇ Download CSV", csv, "cleaned_data.csv", "text/csv")

else:
    st.info("ℹ Upload a file to begin.")
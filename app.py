import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List, Callable
import io
import base64
from tqdm.auto import tqdm
import time

from deduplication import find_near_duplicates
from featurizer import custom_featurizer
from issues import find_issues
from pipeline import make_step, run_pipeline

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_TYPES = ["EHR Data", "ECG Data", "Genomic Data", "Medical Imaging", "Lab Results"]
ANALYSIS_TASKS = ["Near-Duplicate Detection", "Deduplication", "Find Mislabeled Data", "Generate Visualization", "Statistical Summary"]

# ============================================================================
# FEATURE HANDLERS & ANALYSIS
# ============================================================================
class ECGVisualizer:
    @staticmethod
    def create_visualization(df: pd.DataFrame) -> Optional[plt.Figure]:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            if len(df.columns) > 1:
                for col in df.columns[1:]: ax.plot(df.iloc[:, 0], df[col], label=col, linewidth=0.8)
                ax.set_xlabel('Time (ms)'); ax.set_ylabel('Amplitude (mV)')
                ax.set_title('ECG Signal Visualization'); ax.legend()
            else:
                ax.plot(df.iloc[:, 0], linewidth=0.8)
                ax.set_xlabel('Sample Index'); ax.set_ylabel('Amplitude')
                ax.set_title('ECG Signal')
            ax.grid(True, alpha=0.3); plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating ECG visualization: {e}"); return None

class DataAnalyzer:
    @staticmethod
    def deduplicate(df: pd.DataFrame) -> Tuple[str, pd.DataFrame]:
        try:
            original_count = len(df)
            df_dedup = df.drop_duplicates()
            duplicates_removed = original_count - len(df_dedup)
            return f"✓ Exact Deduplication: Removed {duplicates_removed} duplicate rows.", df_dedup
        except Exception as e:
            return f"✗ Error during deduplication: {str(e)}", df

# ============================================================================
# CHATBOT HANDLER
# ============================================================================
class ChatbotHandler:
    def __init__(self): self.context = {}
    def update_context(self, file_path, data_type, df): self.context.update({"file": file_path, "type": data_type, "df": df})
    def respond(self, message: str, history: List):
        if history is None: history = []
        if not message or message.strip() == "": return history, ""
        df = self.context.get("df")
        if "column" in message.lower(): response = f"The dataset has {len(df.columns)} columns: {', '.join(df.columns)}" if df is not None else "Please upload a file."
        elif "row" in message.lower(): response = f"The dataset has {len(df)} rows." if df is not None else "Please upload a file."
        elif "help" in message.lower(): response = "Ask about 'columns' or 'rows' in your data."
        else: response = f"I understand you said '{message}'. Currently analyzing: {self.context.get('type') or 'No data'}. Type 'help' for options."
        history.append((message, response))
        return history, ""

# ============================================================================
# MAIN APPLICATION
# ============================================================================
class DataAnalysisApp:
    def __init__(self):
        self.current_df = None
        self.chatbot = ChatbotHandler()

    def load_csv(self, file) -> Tuple[str, Optional[pd.DataFrame]]:
        if file is None: return "⚠ No file uploaded", None
        try:
            df = pd.read_csv(file.name)
            self.current_df = df
            return f"✓ Loaded {len(df)} rows, {len(df.columns)} columns", df
        except Exception as e:
            return f"✗ Error loading file: {str(e)}", None

    def run_the_pipeline(self, df: pd.DataFrame) -> str:
        try:
            bar = tqdm(total=100, leave=False, desc="Pipeline Progress")
            steps = [
                make_step(find_near_duplicates, name="dedup")(progress=bar),
                make_step(custom_featurizer, name="featurize")(
                    label=None, nan_strategy="impute", on_pipeline_error="drop", progress=bar
                ),
                make_step(find_issues, name="find_label_issues")(label="HARDSHIP_INDEX", progress=bar)
            ]
            results = run_pipeline(steps, df=df)
            bar.close()

            html_output = "<h2>Near-Duplicate & Label Issue Pipeline Results</h2>"
            if "dedup" in results and not results['dedup'].empty:
                html_output += "<h3>Near-Duplicates Found</h3>" + results['dedup'].to_html(index=False, classes='preview-table')
            else: html_output += "<h3>Near-Duplicates Found</h3><p>✓ No near-duplicates were identified.</p>"
            if "find_label_issues" in results and not results['find_label_issues'].empty:
                html_output += "<h3 style='margin-top: 20px;'>Label Issues Found</h3>"
                if 'error' in results['find_label_issues'].columns:
                    html_output += f"<p style='color: red;'>✗ {results['find_label_issues']['error'].iloc[0]}</p>"
                else: html_output += results['find_label_issues'].to_html(index=False, classes='preview-table')
            else: html_output += "<h3 style='margin-top: 20px;'>Label Issues Found</h3><p>✓ No label issues were identified.</p>"
            return html_output
        except Exception as e:
            return f"<h3>Pipeline Error</h3><p style='color: red;'>An error occurred: {str(e)}</p>"

    def update_display(self, file, data_type: str, analysis_tasks: List[str]):
        status, df = self.load_csv(file)
        if df is None: return status, ["<p>Please upload a valid CSV file</p>"], None
        self.chatbot.update_context(file.name if file else None, data_type, df)
        results, task_statuses = [], []
        if "Deduplication" in analysis_tasks:
            msg, df = DataAnalyzer.deduplicate(df); self.current_df = df; task_statuses.append(msg)
        if "Near-Duplicate Detection" in analysis_tasks: results.append(self.run_the_pipeline(df))
        if "Generate Visualization" in analysis_tasks:
            if data_type == "ECG Data":
                fig = ECGVisualizer.create_visualization(df);
                if fig: results.append(fig)
            else: task_statuses.append("ℹ️ 'Visualization' only for 'ECG Data'.")
        if "Find Mislabeled Data" in analysis_tasks: results.append("<h3>Mislabeled Data</h3><p>Placeholder for this analysis.</p>")
        if "Statistical Summary" in analysis_tasks: results.append("<h3>Statistics</h3><p>Placeholder for this analysis.</p>")
        if not analysis_tasks:
            task_statuses.append("ℹ️ No task selected. Showing preview.")
            results.append("<h3>Data Preview</h3>" + df.head(10).to_html(index=False, classes='preview-table'))
        return status + "\n" + "\n".join(task_statuses), results, None

# ============================================================================
# UI CONSTRUCTION & LAUNCH
# ============================================================================
def create_interface():
    app = DataAnalysisApp()
    custom_css = """
    #root, body, html { padding: 0 !important; margin: 0 !important; height: 100vh; overflow: hidden !important; }
    #main-block { height: 100%; padding: 10px; box-sizing: border-box; } #main-row { flex-grow: 1; overflow: hidden; }
    #left-column, #chatbot-column { height: 100%; display: flex; flex-direction: column; }
    #output-wrapper { flex-grow: 1; overflow-y: auto; border: 1px solid #E5E7EB; border-radius: 8px; padding: 8px; }
    #chatbot-history { flex-grow: 1; overflow-y: auto; min-height: 200px; }
    .preview-table { border-collapse: collapse; width: 100%; } .preview-table th { background-color: #3498db; color: white; padding: 8px; text-align: left; }
    .preview-table td { padding: 6px; border-bottom: 1px solid #ddd; }
    """
    with gr.Blocks(theme=gr.themes.Soft(), title="Medical Data Analysis Platform", css=custom_css, elem_id="main-block") as interface:
        gr.Markdown("# 🏥 Medical Data Analysis Platform")
        with gr.Row(elem_id="main-row"):
            with gr.Column(scale=3, elem_id="left-column"):
                # CORRECTED: Using gr.Group for controls
                with gr.Group():
                    with gr.Row():
                        file_upload = gr.File(label="📁 Upload CSV File", file_types=[".csv"], scale=2)
                        data_type_dropdown = gr.Dropdown(choices=DATA_TYPES, value="EHR Data", label="📊 Data Type", scale=1)
                    analysis_checkboxes = gr.CheckboxGroup(choices=ANALYSIS_TASKS, value=[], label="🔬 Select Analysis Tasks")
                    process_btn = gr.Button("▶ Process", variant="primary")
                    status_box = gr.Textbox(label="Status", interactive=False, lines=2)
                # CORRECTED: Using gr.Column as a scrollable wrapper
                with gr.Column(elem_id="output-wrapper"):
                    output_display = gr.HTML(label="Analysis Output")
            with gr.Column(scale=1, elem_id="chatbot-column"):
                gr.Markdown("### 💬 AI Assistant")
                chatbot = gr.Chatbot(elem_id="chatbot-history", label="Chat History")
                with gr.Row():
                    msg_input = gr.Textbox(placeholder="Ask about your data...", label="Message", scale=7)
                    send_btn = gr.Button("Send", scale=1, min_width=50)

        def process_data(file, data_type, analysis_tasks):
            status, results, _ = app.update_display(file, data_type, analysis_tasks)
            html_content = ""
            for item in results:
                if isinstance(item, plt.Figure):
                    buf = io.BytesIO(); item.savefig(buf, format='png', dpi=100, bbox_inches='tight'); buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8'); plt.close(item)
                    html_content += f'<div style="margin-bottom: 20px;"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'
                elif isinstance(item, str): html_content += f"<div style='margin-bottom: 20px;'>{item}</div>"
            return status, html_content or "<p>No analysis tasks selected or no output was generated.</p>"

        process_btn.click(fn=process_data, inputs=[file_upload, data_type_dropdown, analysis_checkboxes], outputs=[status_box, output_display])
        send_btn.click(fn=app.chatbot.respond, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
        msg_input.submit(fn=app.chatbot.respond, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
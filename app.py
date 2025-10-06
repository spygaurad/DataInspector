import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

from agent.ChatbotHandler import ChatbotHandler

# ============================================================================
# DIRECT IMPORTS
# ============================================================================
from pipeline.deduplication import find_near_duplicates
from pipeline.featurizer import custom_featurizer
from pipeline.issues import find_issues
from pipeline.pipeline import make_step, run_pipeline

load_dotenv()

# ============================================================================

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_TYPES = ["EHR Data", "ECG Data", "Genomic Data", "Medical Imaging", "Lab Results"]
ANALYSIS_TASKS = [
    "Near-Duplicate Detection",
    "Deduplication",
    "Find Mislabeled Data",
    "Generate Visualization",
    "Statistical Summary"
]

# ============================================================================
# FEATURE HANDLERS & ANALYSIS
# ============================================================================
class ECGVisualizer:
    @staticmethod
    def create_visualization(df: pd.DataFrame) -> Optional[plt.Figure]:
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            if len(df.columns) > 1:
                for col in df.columns[1:]:
                    ax.plot(df.iloc[:, 0], df[col], label=str(col), linewidth=0.8)
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Amplitude (mV)')
                ax.set_title('ECG Signal Visualization')
                ax.legend()
            else:
                ax.plot(df.iloc[:, 0], linewidth=0.8)
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Amplitude')
                ax.set_title('ECG Signal')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Error creating ECG visualization: {e}")
            return None

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
# MAIN APPLICATION
# ============================================================================
class DataAnalysisApp:
    def __init__(self):
        self.current_df = None
        self.chatbot = ChatbotHandler()

    def load_csv(self, file) -> Tuple[str, Optional[pd.DataFrame]]:
        if file is None:
            return "⚠ No file uploaded", None
        try:
            df = pd.read_csv(file.name)
            self.current_df = df
            return f"✓ Loaded {len(df)} rows, {len(df.columns)} columns", df
        except Exception as e:
            return f"✗ Error loading file: {str(e)}", None

    def run_the_pipeline(self, df: pd.DataFrame, label: Optional[str]) -> Tuple[Optional[pd.DataFrame], Optional[List[Dict]], Optional[str]]:
        """Run the near-duplicate/issue-finding pipeline with a user-selected label."""
        try:
            bar = tqdm(total=100, leave=False, desc="Pipeline Progress")
            steps = [
                make_step(find_near_duplicates, name="dedup")(progress=bar),
                make_step(custom_featurizer, name="featurize")(
                    label=label, nan_strategy="impute", on_pipeline_error="drop", progress=bar
                ),
                make_step(find_issues, name="find_label_issues")(label=label, progress=bar),
            ]
            results_df, summary_list = run_pipeline(steps, df=df)
            bar.close()
            return results_df, summary_list, None
        except Exception as e:
            return None, None, f"<h3>Pipeline Error</h3><p style='color: red;'>An error occurred: {str(e)}</p>"

    def update_display(
        self,
        file,
        data_type: str,
        analysis_tasks: List[str],
        label_col: Optional[str]
    ) -> Tuple[str, Optional[pd.DataFrame], List[Any]]:
        status, df = self.load_csv(file)
        if df is None:
            return status, None, [("html", "<p>Please upload a valid CSV file</p>")]

        original_df = df.copy()
        self.chatbot.update_context(file.name if file else None, data_type, df)

        results, task_statuses = [], []

        if "Deduplication" in analysis_tasks:
            msg, df = DataAnalyzer.deduplicate(df)
            self.current_df = df
            task_statuses.append(msg)

        if "Near-Duplicate Detection" in analysis_tasks:
            if not label_col:
                results.append(("html",
                                "<div style='color:#b45309;background:#fff7ed;padding:10px;border-radius:8px;'>"
                                "⚠ Please choose a <b>Label column</b> before running Near-Duplicate Detection."
                                "</div>"))
            else:
                results_df, summary_list, error_str = self.run_the_pipeline(df, label=label_col)
                if error_str:
                    results.append(("html", error_str))
                else:
                    results.append(("pipeline", results_df, summary_list))

        if "Generate Visualization" in analysis_tasks:
            if data_type == "ECG Data":
                fig = ECGVisualizer.create_visualization(df)
                if fig:
                    results.append(("plot", fig))
            else:
                task_statuses.append("ℹ️ 'Visualization' is only for 'ECG Data'.")

        if "Find Mislabeled Data" in analysis_tasks:
            results.append(("html", "<h3>Mislabeled Data</h3><p>Placeholder for this analysis.</p>"))
        if "Statistical Summary" in analysis_tasks:
            results.append(("html", "<h3>Statistics</h3><p>Placeholder for this analysis.</p>"))

        if not analysis_tasks and not any(r[0] == 'pipeline' for r in results):
            task_statuses.append("ℹ️ No task selected. Showing data preview.")
            results.append(("html", "<h3>Data Preview</h3>" + original_df.head(10).to_html(index=False, classes='preview-table')))

        final_status = status + ("\n" + "\n".join(task_statuses) if task_statuses else "")
        return final_status, original_df, results

# ============================================================================
# UI CONSTRUCTION & LAUNCH
# ============================================================================
def create_interface():
    app = DataAnalysisApp()

    custom_css = """
    /* Compact global layout */
    #root, body, html { padding:0!important; margin:0!important; height:100vh; overflow:hidden!important; }
    .gradio-container { max-width: unset!important; }

    /* Header compaction */
    .topbar * { font-size: 0.95rem; }
    .topbar .gradio-file,
    .topbar .wrap.svelte-1ipelgc { min-height: 0!important; }
    .topbar .wrap.svelte-1ipelgc input { padding: 6px 8px!important; }
    .topbar .gradio-dropdown,
    .topbar .gradio-file { transform: scale(0.95); transform-origin: left center; }

    /* Page structure */
    #main-block { height: 100%; padding: 0.75rem; box-sizing: border-box; display:flex; flex-direction:column; gap:0.75rem; }
    #main-row { flex-grow:1; overflow:hidden; gap:0.75rem; }

    /* Left menu layout: push action bar down */
    #left-nav-column { height:100%; display:flex; flex-direction:column; background:#f9fafb; padding:0.75rem; border-radius:10px; gap:0.5rem; }
    #menu-top { flex: 1; display:flex; flex-direction:column; gap:0.5rem; overflow:auto; }
    #menu-bottom { margin-top: 0.5rem; }

    /* Action bar compact */
    #action-bar { display:flex; gap:0.5rem; align-items:stretch; }
    #action-bar .gradio-button { min-height: 38px; }
    #action-bar .gradio-textbox textarea { height: 38px!important; resize: none; }

    /* Label picker box */
    #label-box { background:#fff; border:1px solid #e5e7eb; padding:0.5rem 0.6rem; border-radius:8px; }

    /* Main content */
    #main-content-column { height:100%; display:flex; flex-direction:column; overflow:hidden; }
    #pipeline-tabs-wrapper { flex-grow:1; display:flex; flex-direction:column; }
    #pipeline-tabs-wrapper .tab-buttons { flex-shrink:0; }
    #pipeline-tabs-wrapper .tabs-content { flex-grow:1; overflow:auto; }

    /* Chat column; compact send button */
    #chatbot-column { height:100%; display:flex; flex-direction:column; }
    #chatbot-history { flex-grow:1; overflow-y:auto; min-height:240px; }
    #send-row .gradio-button { min-width: 56px; padding: 0 8px; }

    /* Table Styling */
    .preview-table { border-collapse: collapse; width: 100%; }
    .preview-table th { background-color: #3498db; color: white; padding: 8px; text-align: left; }
    .preview-table td { padding: 6px; border-bottom: 1px solid #ddd; }
    """

    with gr.Blocks(theme=gr.themes.Soft(), title="Medical Data Analysis Platform", css=custom_css, elem_id="main-block") as interface:
        gr.Markdown("# 🏥 Medical Data Analysis Platform", elem_classes=["topbar"])

        # --- Compact top bar ---
        with gr.Row(elem_classes=["topbar"]):
            file_upload = gr.File(label="Upload CSV", file_types=[".csv"], scale=2)
            data_type_dropdown = gr.Dropdown(choices=DATA_TYPES, value="EHR Data", label="Data Type", scale=1)

        with gr.Row(elem_id="main-row"):
            # ---------- LEFT MENU ----------
            with gr.Column(scale=2, elem_id="left-nav-column"):
                with gr.Group(elem_id="menu-top"):
                    gr.Markdown("#### Analysis Tasks")
                    analysis_checkboxes = gr.CheckboxGroup(choices=ANALYSIS_TASKS, value=[], label=None)

                    # Label picker: visible after upload; interactive only when NDD task selected
                    with gr.Group(visible=False, elem_id="label-box") as label_group:
                        label_dropdown = gr.Dropdown(
                            choices=[],
                            label="Label column (required for Near-Duplicate Detection)",
                            interactive=False
                        )

                with gr.Group(elem_id="menu-bottom"):
                    with gr.Row(elem_id="action-bar"):
                        process_btn = gr.Button("▶ Process", variant="primary", scale=1)
                        status_box = gr.Textbox(label="Status", interactive=False, lines=2, scale=1)

            # ---------- MAIN CONTENT ----------
            with gr.Column(scale=7, elem_id="main-content-column"):
                pipeline_tabs = gr.Tabs(visible=True, elem_id="pipeline-tabs-wrapper")
                with pipeline_tabs:
                    with gr.TabItem("Original Data"):
                        original_df_output = gr.Dataframe(interactive=True)
                    with gr.TabItem("Cleaned Data"):
                        pipeline_df_output = gr.Dataframe()
                    with gr.TabItem("Pipeline Summary"):
                        pipeline_json_output = gr.JSON()
                html_output_display = gr.HTML(visible=True)

            # ---------- CHAT ----------
            with gr.Column(scale=3, elem_id="chatbot-column"):
                gr.Markdown("### 💬 AI Assistant")
                chatbot = gr.Chatbot(elem_id="chatbot-history", label="Chat History")
                with gr.Row(elem_id="send-row"):
                    msg_input = gr.Textbox(placeholder="Ask about your data...", label="Message", scale=7)
                    send_btn = gr.Button("Send", scale=1, min_width=50)

        # -------------------- Callbacks --------------------

        # 1) On upload: load CSV, populate Original Data immediately and seed label choices (visible but disabled)
        def on_upload(file, current_data_type):
            status, df = app.load_csv(file)
            if df is None:
                return (
                    status,                          # status_box
                    gr.update(value=None),           # original_df_output
                    gr.update(choices=[], value=None, visible=False, interactive=False),  # label_dropdown
                    gr.update(visible=False),        # label_group
                    gr.update(visible=True),         # pipeline_tabs (still show tabs)
                    gr.update(value="<p>Please upload a valid CSV file</p>", visible=True)  # html
                )

            boot_text = app.chatbot.update_context(file.name if file else None, current_data_type, df)
            col_choices = list(map(str, df.columns))

            seeded_history = [(None, boot_text or "Dataset loaded and inspected.")]  # bot-only first row

            return (
                status,                                                # keep status as load message
                gr.update(value=df.head(200)),
                gr.update(choices=col_choices, value=None, visible=True, interactive=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(value="", visible=False),
                gr.update(value=seeded_history),                       # ← seed chat history
            )

        file_upload.change(
            fn=on_upload,
            inputs=[file_upload, data_type_dropdown],
            outputs=[status_box, original_df_output, label_dropdown, label_group, pipeline_tabs, html_output_display, chatbot]
        )

        # 2) Tasks change: enable/disable (don’t hide) the label dropdown
        def on_tasks_changed(tasks, file):
            need_label = ("Near-Duplicate Detection" in (tasks or [])) and (file is not None)
            return gr.update(interactive=need_label)

        analysis_checkboxes.change(
            fn=on_tasks_changed,
            inputs=[analysis_checkboxes, file_upload],
            outputs=[label_dropdown]
        )

        # 3) Process click
        def process_data(file, data_type, analysis_tasks, label_col):
            status, original_df, results = app.update_display(file, data_type, analysis_tasks, label_col)

            df_out, json_out, html_out = None, None, ""
            show_pipeline = gr.update(visible=False)
            show_html = gr.update(visible=True)

            for res_type, *res_data in results:
                if res_type == "pipeline":
                    df_out, json_out = res_data
                    show_pipeline = gr.update(visible=True)
                    show_html = gr.update(visible=False)
                elif res_type == "plot":
                    fig = res_data[0]
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close(fig)
                    html_out += f'<div style="margin-bottom: 20px;"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'
                elif res_type == "html":
                    html_out += f"<div style='padding: 10px;'>{res_data[0]}</div>"

            return status, original_df, df_out, json_out, html_out, show_pipeline, show_html

        outputs = [
            status_box,
            original_df_output,
            pipeline_df_output,
            pipeline_json_output,
            html_output_display,
            pipeline_tabs,
            html_output_display
        ]
        process_btn.click(
            fn=process_data,
            inputs=[file_upload, data_type_dropdown, analysis_checkboxes, label_dropdown],
            outputs=outputs
        )

        # 4) Chat events
        chat_outputs = [chatbot, msg_input]
        send_btn.click(fn=app.chatbot.respond, inputs=[msg_input, chatbot], outputs=chat_outputs)
        msg_input.submit(fn=app.chatbot.respond, inputs=[msg_input, chatbot], outputs=chat_outputs)

    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

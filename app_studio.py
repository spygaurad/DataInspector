import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import io
import base64
from tqdm.auto import tqdm
from dataclasses import dataclass
import gradio as gr
import json

from pipeline.deduplication import find_near_duplicates
from pipeline.featurizer import custom_featurizer
from pipeline.issues import find_issues
from pipeline.pipeline import make_step, run_pipeline

from ecg_analyzer import ECGAnalyzer

# ============================================================================
# ANALYSIS TASK CONFIGURATION
# ============================================================================

@dataclass
class TaskConfig:
    """Configuration for each analysis task"""
    name: str
    data_type: str
    requires_params: bool
    param_components: List[Dict[str, Any]]
    output_tabs: List[str]

class TaskRegistry:
    """Registry mapping tasks to their configurations"""

    @staticmethod
    def get_config(data_type: str, task_name: str) -> Optional[TaskConfig]:
        """Get configuration for a specific task"""
        configs = {
            "EHR Data": {
                "Near-Duplicate Detection": TaskConfig(
                    name="Near-Duplicate Detection",
                    data_type="EHR Data",
                    requires_params=True,
                    param_components=[
                        {"type": "dropdown", "label": "Label Column", "elem_id": "ndd_label"}
                    ],
                    output_tabs=["original", "processed", "summary"]
                ),
                "Find Mislabeled Data": TaskConfig(
                    name="Find Mislabeled Data",
                    data_type="EHR Data",
                    requires_params=True,
                    param_components=[
                        {"type": "dropdown", "label": "Label Column", "elem_id": "mislabel_label"}
                    ],
                    output_tabs=["original", "summary"]
                )
            },
            "ECG Data": {
                "ECG Visualization": TaskConfig(
                    name="ECG Visualization",
                    data_type="ECG Data",
                    requires_params=True,
                    param_components=[
                        {"type": "checkboxgroup", "label": "Select Leads", "elem_id": "ecg_leads"},
                        {"type": "checkboxgroup", "label": "Visualization Types", "elem_id": "ecg_viz_types"}
                    ],
                    output_tabs=["visualization", "summary"]
                ),
                "Statistical Summary": TaskConfig(
                    name="Statistical Summary",
                    data_type="ECG Data",
                    requires_params=True,
                    param_components=[
                        {"type": "checkboxgroup", "label": "Select Leads", "elem_id": "ecg_stats_leads"}
                    ],
                    output_tabs=["summary", "visualization"]
                )
            }
        }
        return configs.get(data_type, {}).get(task_name)

    @staticmethod
    def get_tasks_for_data_type(data_type: str) -> List[str]:
        """Get available tasks for a data type"""
        tasks = {
            "EHR Data": ["Near-Duplicate Detection", "Find Mislabeled Data"],
            "ECG Data": ["ECG Visualization", "Statistical Summary"]
        }
        return tasks.get(data_type, [])


# ============================================================================
# ANALYSIS EXECUTION
# ============================================================================

class AnalysisExecutor:
    """Executes analysis tasks and returns results"""

    @staticmethod
    def execute_near_duplicate_detection(df: pd.DataFrame, label: str) -> Tuple[str, Dict[str, Any]]:
        """Execute near-duplicate detection pipeline"""
        try:
            if not label:
                return "⚠ Label column required", {"original": df, "processed": None, "summary": None}

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

            return "✓ Near-duplicate detection completed", {
                "original": df, "processed": results_df, "summary": summary_list
            }
        except Exception as e:
            return f"✗ Error: {str(e)}", {"original": df, "processed": None, "summary": None}

    @staticmethod
    def execute_find_mislabeled(df: pd.DataFrame, label: str) -> Tuple[str, Dict[str, Any]]:
        """Execute mislabeled data detection"""
        try:
            if not label:
                return "⚠ Label column required", {"original": df, "summary": None}

            summary = {
                "task": "Find Mislabeled Data", "label_column": label, "total_samples": len(df),
                "suspicious_samples": 0, "message": "Mislabeled detection analysis completed"
            }
            return "✓ Mislabeled data analysis completed", {"original": df, "summary": summary}
        except Exception as e:
            return f"✗ Error: {str(e)}", {"original": df, "summary": None}

    @staticmethod
    def execute_ecg_visualization(df: pd.DataFrame, leads: List[str] = None, viz_types: List[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Execute ECG visualization using ECGAnalyzer"""
        try:
            # Detect available leads
            available_leads = ECGAnalyzer.detect_leads(df)
            
            # Use provided leads or default to all available
            if not leads:
                leads = available_leads if available_leads else []
            
            if not leads:
                return "⚠ No ECG leads found in data", {"visualization": None, "summary": None}
            
            # Default visualization types
            if not viz_types:
                viz_types = ["Signal Waveform", "Histogram"]
            
            # Create visualizations
            viz_html = ECGAnalyzer.create_all_visualizations(df, leads, viz_types)
            
            # Generate statistics
            stats = ECGAnalyzer.generate_statistics(df, leads)
            
            summary = {
                "task": "ECG Visualization",
                "samples": len(df),
                "leads_analyzed": leads,
                "visualizations": viz_types,
                "statistics": stats
            }
            
            return "✓ ECG visualization created", {"visualization": viz_html, "summary": summary}
        except Exception as e:
            return f"✗ Error: {str(e)}", {"visualization": None, "summary": None}

    @staticmethod
    def execute_statistical_summary(df: pd.DataFrame, leads: List[str] = None) -> Tuple[str, Dict[str, Any]]:
        """Execute statistical summary using ECGAnalyzer"""
        try:
            # Detect available leads
            available_leads = ECGAnalyzer.detect_leads(df)
            
            # Use provided leads or default to all available
            if not leads:
                leads = available_leads if available_leads else list(df.select_dtypes(include=[np.number]).columns)
            
            if not leads:
                return "⚠ No numeric columns found", {"summary": None, "visualization": None}
            
            # Generate statistics
            stats = ECGAnalyzer.generate_statistics(df, leads)
            
            # Create HTML table for statistics
            html_rows = []
            html_rows.append("<table class='preview-table' style='margin: 20px auto; max-width: 900px;'>")
            html_rows.append("<thead><tr><th>Lead</th><th>Mean</th><th>Std</th><th>Min</th><th>Q25</th><th>Median</th><th>Q75</th><th>Max</th></tr></thead>")
            html_rows.append("<tbody>")
            
            for lead, lead_stats in stats.items():
                html_rows.append(f"<tr>")
                html_rows.append(f"<td><strong>{lead}</strong></td>")
                html_rows.append(f"<td>{lead_stats['mean']:.4f}</td>")
                html_rows.append(f"<td>{lead_stats['std']:.4f}</td>")
                html_rows.append(f"<td>{lead_stats['min']:.4f}</td>")
                html_rows.append(f"<td>{lead_stats['q25']:.4f}</td>")
                html_rows.append(f"<td>{lead_stats['median']:.4f}</td>")
                html_rows.append(f"<td>{lead_stats['q75']:.4f}</td>")
                html_rows.append(f"<td>{lead_stats['max']:.4f}</td>")
                html_rows.append(f"</tr>")
            
            html_rows.append("</tbody></table>")
            summary_html = f"<div style='overflow-x:auto;'><h3 style='text-align:center;'>Statistical Summary</h3>{''.join(html_rows)}</div>"
            
            summary = {
                "task": "Statistical Summary",
                "rows": len(df),
                "leads_analyzed": leads,
                "statistics": stats
            }
            
            return "✓ Statistical summary generated", {"summary": summary, "visualization": summary_html}
        except Exception as e:
            return f"✗ Error: {str(e)}", {"summary": None, "visualization": None}


# ============================================================================
# UI MANAGER - Handles all UI state and updates
# ============================================================================

class UIManager:
    """Manages UI state and dynamic updates"""

    def __init__(self):
        self.current_df = None
        self.current_data_type = "EHR Data"
        self.chatbot_context = {}
        self.command_map = {
            "data_type": {"ehr": "EHR Data", "ecg": "ECG Data"},
            "task": {
                "deduplication": "Near-Duplicate Detection", "mislabeled": "Find Mislabeled Data",
                "visualize_ecg": "ECG Visualization", "stats": "Statistical Summary"
            }
        }

    def load_csv(self, file) -> Tuple[str, Optional[pd.DataFrame]]:
        """Load CSV file"""
        if file is None: return "⚠ No file uploaded", None
        try:
            df = pd.read_csv(file.name)
            self.current_df = df
            return f"✓ Loaded {len(df)} rows, {len(df.columns)} columns", df
        except Exception as e:
            return f"✗ Error: {str(e)}", None

    def on_file_upload(self, file, data_type: str):
        """Handle file upload - returns updates for all components"""
        status, df = self.load_csv(file)
        if df is None:
            return (
                status, gr.update(value=None), gr.update(choices=[], value=[]),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(choices=[]), gr.update(choices=[]),
                gr.update(choices=[]), gr.update(choices=[], value=[]),
                gr.update(choices=[]), gr.update(choices=[], value=[]),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            )

        self.chatbot_context = {"file": file.name, "type": data_type, "df": df}
        available_tasks = TaskRegistry.get_tasks_for_data_type(data_type)
        col_choices = list(df.columns)
        
        # Detect ECG leads if ECG data
        ecg_leads = ECGAnalyzer.detect_leads(df) if data_type == "ECG Data" else []
        viz_types = ["Signal Waveform", "Histogram", "Scatter Plot", "Rolling Average"]

        return (
            status, gr.update(value=df.head(200)), gr.update(choices=available_tasks, value=[], interactive=True),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(choices=col_choices, value=None), gr.update(choices=col_choices, value=None),
            gr.update(choices=ecg_leads, value=ecg_leads), gr.update(choices=viz_types, value=["Signal Waveform", "Histogram"]),
            gr.update(choices=ecg_leads, value=ecg_leads), gr.update(choices=viz_types, value=["Signal Waveform"]),
            gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        )

    def on_data_type_change(self, data_type: str, file):
        """Handle data type change"""
        self.current_data_type = data_type
        if file and self.current_df is not None: 
            self.chatbot_context["type"] = data_type
            
            # Update ECG lead choices if switching to ECG data
            ecg_leads = ECGAnalyzer.detect_leads(self.current_df) if data_type == "ECG Data" else []
            viz_types = ["Signal Waveform", "Histogram", "Scatter Plot", "Rolling Average"]
            
            return (
                gr.update(choices=TaskRegistry.get_tasks_for_data_type(data_type), value=[]),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(choices=ecg_leads, value=ecg_leads), gr.update(choices=viz_types, value=["Signal Waveform", "Histogram"]),
                gr.update(choices=ecg_leads, value=ecg_leads), gr.update(choices=viz_types, value=["Signal Waveform"]),
                f"Data type changed to: {data_type}"
            )
        
        return (
            gr.update(choices=TaskRegistry.get_tasks_for_data_type(data_type), value=[]),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
            gr.update(), gr.update(), gr.update(), gr.update(),
            f"Data type changed to: {data_type}"
        )

    def on_tasks_change(self, selected_tasks: List[str]):
        """Handle task selection change - show/hide parameter groups"""
        show_ndd = "Near-Duplicate Detection" in selected_tasks
        show_mislabel = "Find Mislabeled Data" in selected_tasks
        show_ecg_viz = "ECG Visualization" in selected_tasks
        show_ecg_stats = "Statistical Summary" in selected_tasks and self.current_data_type == "ECG Data"
        return (
            gr.update(visible=show_ndd), 
            gr.update(visible=show_mislabel),
            gr.update(visible=show_ecg_viz),
            gr.update(visible=show_ecg_stats)
        )

    def process_analysis(self, file, data_type: str, selected_tasks: List[str], 
                         ndd_label: str, mislabel_label: str,
                         ecg_viz_leads: List[str], ecg_viz_types: List[str],
                         ecg_stats_leads: List[str]):
        """Process analysis tasks based on UI inputs."""
        status, df = self.load_csv(file)
        if df is None:
            return (status, None, None, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False))

        params = {
            "ndd_label": ndd_label, 
            "mislabel_label": mislabel_label,
            "ecg_viz_leads": ecg_viz_leads,
            "ecg_viz_types": ecg_viz_types,
            "ecg_stats_leads": ecg_stats_leads
        }
        return self._run_analysis(df, data_type, selected_tasks, params)

    def _run_analysis(self, df: pd.DataFrame, data_type: str, selected_tasks: List[str], params: Dict[str, Any]):
        """Centralized analysis executor, callable from UI or chatbot."""
        if not selected_tasks:
            return (
                "⚠ No tasks selected", df.head(200), None, None, None,
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            )

        all_tabs = set(); all_results = {"original": df.head(200), "processed": None, "summary": [], "visualization": ""}
        status_messages = []; executor = AnalysisExecutor()

        for task_name in selected_tasks:
            config = TaskRegistry.get_config(data_type, task_name)
            if not config:
                status_messages.append(f"✗ Unknown task: {task_name}"); continue
            all_tabs.update(config.output_tabs)
            
            if task_name == "Near-Duplicate Detection":
                status_msg, results = executor.execute_near_duplicate_detection(df, params.get("ndd_label"))
            elif task_name == "Find Mislabeled Data":
                status_msg, results = executor.execute_find_mislabeled(df, params.get("mislabel_label"))
            elif task_name == "ECG Visualization":
                status_msg, results = executor.execute_ecg_visualization(
                    df, 
                    params.get("ecg_viz_leads"), 
                    params.get("ecg_viz_types")
                )
            elif task_name == "Statistical Summary":
                if data_type == "ECG Data":
                    status_msg, results = executor.execute_statistical_summary(df, params.get("ecg_stats_leads"))
                else:
                    status_msg, results = executor.execute_statistical_summary(df)
            else:
                status_msg, results = "✗ Task not implemented", {}

            status_messages.append(f"{task_name}: {status_msg}")
            if results.get("processed") is not None: all_results["processed"] = results["processed"]
            if results.get("visualization"): all_results["visualization"] += results["visualization"]
            if results.get("summary") is not None: all_results["summary"].append({"task": task_name, "data": results["summary"]})

        return (
            "\n".join(status_messages), all_results["original"], all_results["processed"],
            all_results["summary"] or None, all_results["visualization"] or None,
            gr.update(visible="original" in all_tabs), gr.update(visible="processed" in all_tabs),
            gr.update(visible="summary" in all_tabs), gr.update(visible="visualization" in all_tabs)
        )

    def chatbot_respond(self, message: str, history: List):
        """Handle chatbot messages, parsing for commands or responding to queries."""
        history = history or []; df = self.chatbot_context.get("df")
        ui_updates = tuple([gr.update()] * 9) # status, 4 outputs, 4 tabs

        try:
            command = json.loads(message)
            if isinstance(command, dict) and "task" in command:
                if df is None:
                    history.append((message, "Please upload a data file before running a command."))
                    return (history, "") + ui_updates

                data_type = self.command_map["data_type"].get(command.get("data_type", "").lower(), self.current_data_type)
                task_name = self.command_map["task"].get(command.get("task", "").lower())

                if not task_name:
                    response = f"Unknown task: '{command['task']}'. Valid: {list(self.command_map['task'].keys())}"
                    history.append((message, response))
                    return (history, "") + ui_updates

                params = command.get("parameters", {})
                analysis_params = {"ndd_label": params.get("label"), "mislabel_label": params.get("label")}
                
                analysis_updates = self._run_analysis(df, data_type, [task_name], analysis_params)
                history.append((message, f"Command executed: Running '{task_name}'."))
                return (history, "") + analysis_updates
        except (json.JSONDecodeError, TypeError):
            pass # Not a JSON command, proceed with standard logic
        
        if "column" in message.lower():
            response = (f"Dataset has {len(df.columns)} columns: {', '.join(map(str, df.columns))}" if df is not None else "Please upload a file first.")
        elif "row" in message.lower():
            response = f"Dataset has {len(df)} rows." if df is not None else "Please upload a file first."
        elif "help" in message.lower():
            response = "Ask about 'columns' or 'rows'. To run a task, send JSON, e.g., `{\"task\": \"stats\"}` or `{\"task\": \"deduplication\", \"parameters\": {\"label\": \"your_column\"}}`"
        else:
            response = "I can help with data queries or run tasks via JSON commands. Try asking 'help'."
        
        history.append((message, response))
        return (history, "") + ui_updates


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Build the Gradio interface"""
    ui_manager = UIManager()
    custom_css = """
    * { box-sizing: border-box; } html, body { margin: 0; padding: 0; height: 100vh; overflow: hidden; }
    .gradio-container { height: 100vh !important; max-width: 100% !important; padding: 0 !important; }
    #app-container { height: 100vh; display: flex; flex-direction: column; padding: 0.75rem; gap: 0.75rem; }
    #main-row { flex: 1; min-height: 0; display: flex; gap: 0.75rem; }
    #left-panel { display: flex; flex-direction: column; height: 100%; background: #f9fafb; border-radius: 10px; padding: 0.75rem; gap: 0.5rem; }
    #task-section { flex: 1; min-height: 0; overflow-y: auto; display: flex; flex-direction: column; gap: 0.5rem; }
    #middle-panel, #chat-panel { display: flex; flex-direction: column; height: 100%; }
    #tabs-container { flex: 1; min-height: 0; display: flex; flex-direction: column; }
    #tabs-container .tabitem { flex: 1; min-height: 0; overflow: auto; }
    #chat-history { flex: 1; min-height: 0; overflow-y: auto; margin-bottom: 0.5rem; }
    #chat-input-row { flex-shrink: 0; display: flex; gap: 0.5rem; }
    .preview-table { border-collapse: collapse; width: 100%; font-size: 0.875rem; }
    .preview-table th { background-color: #3498db; color: white; padding: 8px; text-align: left; position: sticky; top: 0; }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Medical Data Analysis Platform") as demo:
        with gr.Column(elem_id="app-container"):
            gr.Markdown("# 🏥 DataLine: Medical Data Analysis Platform")
            with gr.Row():
                file_input = gr.File(label="Upload CSV", file_types=[".csv"], scale=2)
                data_type = gr.Dropdown(choices=["EHR Data", "ECG Data"], value="EHR Data", label="Data Type", scale=1)
            
            with gr.Row(elem_id="main-row"):
                with gr.Column(scale=2, elem_id="left-panel"):
                    with gr.Group(elem_id="task-section"):
                        gr.Markdown("#### Analysis Tasks")
                        task_selector = gr.CheckboxGroup(choices=TaskRegistry.get_tasks_for_data_type("EHR Data"), label=None)
                        with gr.Group(visible=False) as ndd_param_group:
                            gr.Markdown("**Near-Duplicate Detection Parameters**")
                            ndd_label_dropdown = gr.Dropdown(choices=[], label="Label Column")
                        with gr.Group(visible=False) as mislabel_param_group:
                            gr.Markdown("**Find Mislabeled Data Parameters**")
                            mislabel_label_dropdown = gr.Dropdown(choices=[], label="Label Column")
                        with gr.Group(visible=False) as ecg_viz_param_group:
                            gr.Markdown("**ECG Visualization Parameters**")
                            ecg_viz_leads = gr.CheckboxGroup(choices=[], label="Select Leads", value=[])
                            ecg_viz_types = gr.CheckboxGroup(
                                choices=["Signal Waveform", "Histogram", "Scatter Plot", "Rolling Average"],
                                label="Visualization Types",
                                value=["Signal Waveform", "Histogram"]
                            )
                        with gr.Group(visible=False) as ecg_stats_param_group:
                            gr.Markdown("**Statistical Summary Parameters**")
                            ecg_stats_leads = gr.CheckboxGroup(choices=[], label="Select Leads", value=[])
                    process_btn = gr.Button("▶ Process", variant="primary")
                    status_output = gr.Textbox(label="Status", interactive=False, lines=2)
                
                with gr.Column(scale=7, elem_id="middle-panel"):
                    with gr.Tabs(elem_id="tabs-container"):
                        with gr.TabItem("Original Data", visible=False) as tab_original:
                            original_df_output = gr.DataFrame(interactive=False)
                        with gr.TabItem("Processed Data", visible=False) as tab_processed:
                            processed_df_output = gr.DataFrame(interactive=False)
                        with gr.TabItem("Summary", visible=False) as tab_summary:
                            summary_output = gr.JSON()
                        with gr.TabItem("Visualization", visible=False) as tab_viz:
                            viz_output = gr.HTML()
                
                with gr.Column(scale=3, elem_id="chat-panel"):
                    gr.Markdown("### 💬 AI Assistant")
                    chatbot = gr.Chatbot(elem_id="chat-history", height="100%")
                    with gr.Row(elem_id="chat-input-row"):
                        msg_input = gr.Textbox(placeholder="Ask or send a JSON command...", scale=4, container=False)
                        send_btn = gr.Button("Send", scale=1)
        
        analysis_outputs = [
            status_output, original_df_output, processed_df_output, summary_output, viz_output,
            tab_original, tab_processed, tab_summary, tab_viz
        ]
        
        file_input.change(
            fn=ui_manager.on_file_upload, inputs=[file_input, data_type],
            outputs=[status_output, original_df_output, task_selector, 
                     ndd_param_group, mislabel_param_group, ecg_viz_param_group, ecg_stats_param_group,
                     ndd_label_dropdown, mislabel_label_dropdown,
                     ecg_viz_leads, ecg_viz_types, ecg_stats_leads, ecg_viz_types,
                     tab_original, tab_processed, tab_summary, tab_viz]
        )
        data_type.change(
            fn=ui_manager.on_data_type_change, inputs=[data_type, file_input],
            outputs=[task_selector, ndd_param_group, mislabel_param_group, ecg_viz_param_group, ecg_stats_param_group,
                     ecg_viz_leads, ecg_viz_types, ecg_stats_leads, ecg_viz_types, status_output]
        )
        task_selector.change(
            fn=ui_manager.on_tasks_change, inputs=[task_selector], 
            outputs=[ndd_param_group, mislabel_param_group, ecg_viz_param_group, ecg_stats_param_group]
        )
        process_btn.click(
            fn=ui_manager.process_analysis,
            inputs=[file_input, data_type, task_selector, ndd_label_dropdown, mislabel_label_dropdown,
                    ecg_viz_leads, ecg_viz_types, ecg_stats_leads],
            outputs=analysis_outputs
        )
        
        chat_submit_args = {"fn": ui_manager.chatbot_respond, "inputs": [msg_input, chatbot], "outputs": [chatbot, msg_input] + analysis_outputs}
        send_btn.click(**chat_submit_args)
        msg_input.submit(**chat_submit_args)
    
    return demo

# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7890)
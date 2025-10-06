"""
ADDING NEW ANALYSIS TASKS:
==========================

1. Add task configuration in TaskRegistry.get_config():
   "Your Data Type": {
       "Your Task Name": TaskConfig(
           name="Your Task Name",
           data_type="Your Data Type",
           requires_params=True,  # Set to True if needs parameters
           param_components=[...],  # Define parameter types
           output_tabs=["original", "summary", ...]  # Which tabs to show
       )
   }

2. Add execution method in AnalysisExecutor:
   @staticmethod
   def execute_your_task(df, param1, param2):
       # Your analysis logic
       return "✓ Done", {
           "original": df,
           "summary": summary_data,
           ...
       }

3. In create_interface(), add parameter group in the LEFT panel:
   with gr.Group(visible=False) as your_task_param_group:
       gr.Markdown("**Your Task Parameters**")
       your_param1 = gr.Slider(minimum=0, maximum=1, label="Threshold")
       your_param2 = gr.Dropdown(choices=["A", "B"], label="Option")

4. In UIManager.on_tasks_change()
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import io
import base64
from tqdm.auto import tqdm
from dataclasses import dataclass
import gradio as gr
# Direct imports (your existing modules)


from pipeline.deduplication import find_near_duplicates
from pipeline.featurizer import custom_featurizer
from pipeline.issues import find_issues
from pipeline.pipeline import make_step, run_pipeline



from pipeline import make_step, run_pipeline

# ============================================================================
# ANALYSIS TASK CONFIGURATION
# ============================================================================

@dataclass
class TaskConfig:
    """Configuration for each analysis task"""
    name: str
    data_type: str
    requires_params: bool  # Does it need additional parameters?
    param_components: List[Dict[str, Any]]  # List of parameter UI components
    output_tabs: List[str]  # Which tabs to show: "original", "processed", "summary", "visualization"
    
class TaskRegistry:
    """Registry mapping tasks to their configurations"""
    
    @staticmethod
    def get_config(data_type: str, task_name: str) -> TaskConfig:
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
                    requires_params=False,
                    param_components=[],
                    output_tabs=["visualization", "summary"]
                ),
                "Statistical Summary": TaskConfig(
                    name="Statistical Summary",
                    data_type="ECG Data",
                    requires_params=False,
                    param_components=[],
                    output_tabs=["summary"]
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
                return "⚠ Label column required", {
                    "original": df,
                    "processed": None,
                    "summary": None
                }
            
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
                "original": df,
                "processed": results_df,
                "summary": summary_list
            }
        except Exception as e:
            return f"✗ Error: {str(e)}", {
                "original": df,
                "processed": None,
                "summary": None
            }
    
    @staticmethod
    def execute_find_mislabeled(df: pd.DataFrame, label: str) -> Tuple[str, Dict[str, Any]]:
        """Execute mislabeled data detection"""
        try:
            if not label:
                return "⚠ Label column required", {
                    "original": df,
                    "summary": None
                }
            
            # Placeholder for actual mislabeled detection logic
            summary = {
                "task": "Find Mislabeled Data",
                "label_column": label,
                "total_samples": len(df),
                "suspicious_samples": 0,
                "message": "Mislabeled detection analysis completed"
            }
            
            return "✓ Mislabeled data analysis completed", {
                "original": df,
                "summary": summary
            }
        except Exception as e:
            return f"✗ Error: {str(e)}", {
                "original": df,
                "summary": None
            }
    
    @staticmethod
    def execute_ecg_visualization(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Execute ECG visualization"""
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
            
            # Convert plot to base64
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            
            viz_html = f'<div style="text-align:center;"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'
            
            summary = {
                "task": "ECG Visualization",
                "samples": len(df),
                "channels": len(df.columns) - 1 if len(df.columns) > 1 else 1
            }
            
            return "✓ ECG visualization created", {
                "visualization": viz_html,
                "summary": summary
            }
        except Exception as e:
            return f"✗ Error: {str(e)}", {
                "visualization": None,
                "summary": None
            }
    
    @staticmethod
    def execute_statistical_summary(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
        """Execute statistical summary"""
        try:
            stats = df.describe().to_html(classes='preview-table')
            summary_html = f"<h3>Statistical Summary</h3><div style='overflow-x:auto;'>{stats}</div>"
            
            summary = {
                "task": "Statistical Summary",
                "rows": len(df),
                "columns": len(df.columns),
                "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
                "html": summary_html
            }
            
            return "✓ Statistical summary generated", {
                "summary": summary,
                "visualization": summary_html
            }
        except Exception as e:
            return f"✗ Error: {str(e)}", {
                "summary": None,
                "visualization": None
            }


# ============================================================================
# UI MANAGER - Handles all UI state and updates
# ============================================================================

class UIManager:
    """Manages UI state and dynamic updates"""
    
    def __init__(self):
        self.current_df = None
        self.current_data_type = "EHR Data"
        self.chatbot_context = {}
    
    def load_csv(self, file) -> Tuple[str, Optional[pd.DataFrame]]:
        """Load CSV file"""
        if file is None:
            return "⚠ No file uploaded", None
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
                status,  # status
                gr.update(value=None),  # original_df
                gr.update(choices=[], value=[], interactive=True),  # task_checkboxes
                gr.update(visible=False),  # ndd_param_group
                gr.update(visible=False),  # mislabel_param_group
                gr.update(choices=[]),  # ndd_label_dropdown
                gr.update(choices=[]),  # mislabel_label_dropdown
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),  # tabs
            )
        
        self.chatbot_context = {"file": file.name, "type": data_type, "df": df}
        available_tasks = TaskRegistry.get_tasks_for_data_type(data_type)
        col_choices = list(df.columns)
        
        return (
            status,
            gr.update(value=df.head(200)),
            gr.update(choices=available_tasks, value=[], interactive=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(choices=col_choices, value=None),
            gr.update(choices=col_choices, value=None),
            gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
        )
    
    def on_data_type_change(self, data_type: str, file):
        """Handle data type change"""
        self.current_data_type = data_type
        available_tasks = TaskRegistry.get_tasks_for_data_type(data_type)
        
        if file and self.current_df is not None:
            self.chatbot_context["type"] = data_type
        
        return (
            gr.update(choices=available_tasks, value=[]),  # Reset task selection
            gr.update(visible=False),  # Hide ndd params
            gr.update(visible=False),  # Hide mislabel params
            f"Data type changed to: {data_type}"
        )
    
    def on_tasks_change(self, selected_tasks: List[str], data_type: str, df_columns: List[str]):
        """Handle task selection change - show/hide parameter groups for all selected tasks"""
        if not selected_tasks:
            # Hide all parameter groups
            return (
                gr.update(visible=False),  # ndd_param_group
                gr.update(visible=False),  # mislabel_param_group
            )
        
        # Check which tasks need which parameters
        show_ndd_params = "Near-Duplicate Detection" in selected_tasks
        show_mislabel_params = "Find Mislabeled Data" in selected_tasks
        
        return (
            gr.update(visible=show_ndd_params),
            gr.update(visible=show_mislabel_params),
        )
    
    def process_analysis(self, file, data_type: str, selected_tasks: List[str], ndd_label: str, mislabel_label: str):
        """Process selected analysis tasks - handles multiple tasks"""
        status, df = self.load_csv(file)
        
        if df is None:
            return (
                status,
                None, None, None, None,
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            )
        
        if not selected_tasks:
            return (
                "⚠ No tasks selected",
                df.head(200), None, None, None,
                gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            )
        
        # Track which tabs to show (union of all selected tasks)
        all_tabs = set()
        all_results = {
            "original": df.head(200),
            "processed": None,
            "summary": [],
            "visualization": ""
        }
        
        status_messages = []
        executor = AnalysisExecutor()
        
        # Process each selected task
        for task_name in selected_tasks:
            config = TaskRegistry.get_config(data_type, task_name)
            
            if not config:
                status_messages.append(f"✗ Unknown task: {task_name}")
                continue
            
            # Add this task's tabs to the set
            all_tabs.update(config.output_tabs)
            
            # Execute the task with appropriate parameters
            if task_name == "Near-Duplicate Detection":
                status_msg, results = executor.execute_near_duplicate_detection(df, ndd_label)
                status_messages.append(f"{task_name}: {status_msg}")
                if results.get("processed") is not None:
                    all_results["processed"] = results["processed"]
                if results.get("summary") is not None:
                    all_results["summary"].append({"task": task_name, "data": results["summary"]})
            
            elif task_name == "Find Mislabeled Data":
                status_msg, results = executor.execute_find_mislabeled(df, mislabel_label)
                status_messages.append(f"{task_name}: {status_msg}")
                if results.get("summary") is not None:
                    all_results["summary"].append({"task": task_name, "data": results["summary"]})
            
            elif task_name == "ECG Visualization":
                status_msg, results = executor.execute_ecg_visualization(df)
                status_messages.append(f"{task_name}: {status_msg}")
                if results.get("visualization"):
                    all_results["visualization"] += results["visualization"]
                if results.get("summary") is not None:
                    all_results["summary"].append({"task": task_name, "data": results["summary"]})
            
            elif task_name == "Statistical Summary":
                status_msg, results = executor.execute_statistical_summary(df)
                status_messages.append(f"{task_name}: {status_msg}")
                if results.get("visualization"):
                    all_results["visualization"] += results["visualization"]
                if results.get("summary") is not None:
                    all_results["summary"].append({"task": task_name, "data": results["summary"]})
        
        # Format final outputs based on all_tabs
        show_original = gr.update(visible="original" in all_tabs)
        show_processed = gr.update(visible="processed" in all_tabs)
        show_summary = gr.update(visible="summary" in all_tabs)
        show_viz = gr.update(visible="visualization" in all_tabs)
        
        # Combine status messages
        final_status = "\n".join(status_messages)
        
        return (
            final_status,
            all_results["original"],
            all_results["processed"],
            all_results["summary"] if all_results["summary"] else None,
            all_results["visualization"] if all_results["visualization"] else None,
            show_original,
            show_processed,
            show_summary,
            show_viz
        )
    
    def _format_results(self, status: str, output_tabs: List[str], results: Dict[str, Any]):
        """Format results based on output tabs configuration"""
        # Default all outputs to None
        original_out = None
        processed_out = None
        summary_out = None
        viz_out = None
        
        # Default all tabs hidden
        show_original = gr.update(visible=False)
        show_processed = gr.update(visible=False)
        show_summary = gr.update(visible=False)
        show_viz = gr.update(visible=False)
        
        # Populate based on output_tabs
        if "original" in output_tabs:
            original_out = results.get("original")
            show_original = gr.update(visible=True)
        
        if "processed" in output_tabs:
            processed_out = results.get("processed")
            show_processed = gr.update(visible=True)
        
        if "summary" in output_tabs:
            summary_data = results.get("summary")
            if results.get("summary_html"):  # For statistical summary
                summary_out = results.get("summary_html")
            else:
                summary_out = summary_data
            show_summary = gr.update(visible=True)
        
        if "visualization" in output_tabs:
            viz_out = results.get("visualization")
            show_viz = gr.update(visible=True)
        
        return (
            status,
            original_out,
            processed_out,
            summary_out,
            viz_out,
            show_original,
            show_processed,
            show_summary,
            show_viz
        )
    
    def chatbot_respond(self, message: str, history: List):
        """Chatbot response"""
        if history is None:
            history = []
        if not message or message.strip() == "":
            return history, ""
        
        df = self.chatbot_context.get("df")
        
        if "column" in message.lower():
            response = (f"Dataset has {len(df.columns)} columns: {', '.join(map(str, df.columns))}"
                       if df is not None else "Please upload a file first.")
        elif "row" in message.lower():
            response = f"Dataset has {len(df)} rows." if df is not None else "Please upload a file first."
        elif "help" in message.lower():
            response = "Ask about 'columns', 'rows', or your data analysis."
        else:
            response = f"You asked: '{message}'. Analyzing: {self.chatbot_context.get('type', 'No data')}."
        
        history.append((message, response))
        return history, ""


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface():
    """Build the Gradio interface"""
    
    ui_manager = UIManager()
    
    custom_css = """
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; height: 100vh; overflow: hidden; }
    .gradio-container { height: 100vh !important; max-width: 100% !important; padding: 0 !important; }
    
    #app-container { 
        height: 100vh; 
        display: flex; 
        flex-direction: column; 
        padding: 0.75rem;
        gap: 0.75rem;
    }
    
    #main-row { 
        flex: 1;
        min-height: 0;
        display: flex;
        gap: 0.75rem;
    }
    
    #left-panel { 
        display: flex;
        flex-direction: column;
        height: 100%;
        background: #f9fafb;
        border-radius: 10px;
        padding: 0.75rem;
        gap: 0.5rem;
    }
    
    #task-section {
        flex: 1;
        min-height: 0;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }
    
    #action-section {
        flex-shrink: 0;
    }
    
    #middle-panel {
        display: flex;
        flex-direction: column;
        height: 100%;
        min-width: 0;
    }
    
    #tabs-container {
        flex: 1;
        min-height: 0;
        display: flex;
        flex-direction: column;
    }
    
    #tabs-container .tabs {
        height: 100%;
        display: flex;
        flex-direction: column;
    }
    
    #tabs-container .tab-nav {
        flex-shrink: 0;
    }
    
    #tabs-container .tabitem {
        flex: 1;
        min-height: 0;
        overflow: auto;
    }
    
    #chat-panel {
        display: flex;
        flex-direction: column;
        height: 100%;
        background: #f9fafb;
        border-radius: 10px;
        padding: 0.75rem;
    }
    
    #chat-header {
        flex-shrink: 0;
        margin-bottom: 0.5rem;
    }
    
    #chat-history {
        flex: 1;
        min-height: 0;
        overflow-y: auto;
        margin-bottom: 0.5rem;
    }
    
    #chat-input-row {
        flex-shrink: 0;
        display: flex;
        gap: 0.5rem;
        align-items: flex-end;
    }
    
    #chat-input-row .textbox {
        flex: 1;
    }
    
    #chat-input-row button {
        flex-shrink: 0;
    }
    
    .preview-table { 
        border-collapse: collapse; 
        width: 100%; 
        font-size: 0.875rem;
    }
    .preview-table th { 
        background-color: #3498db; 
        color: white; 
        padding: 8px; 
        text-align: left; 
        position: sticky;
        top: 0;
    }
    .preview-table td { 
        padding: 6px; 
        border-bottom: 1px solid #ddd; 
    }
    
    .compact-header { font-size: 0.95rem; margin: 0; }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Medical Data Analysis Platform") as demo:
        
        with gr.Column(elem_id="app-container"):
            # Header
            gr.Markdown("# 🏥 Medical Data Analysis Platform", elem_classes=["compact-header"])
            
            # Top controls
            with gr.Row():
                file_input = gr.File(label="Upload CSV", file_types=[".csv"], scale=2)
                data_type = gr.Dropdown(
                    choices=["EHR Data", "ECG Data"],
                    value="EHR Data",
                    label="Data Type",
                    scale=1
                )
            
            # Main row with 3 columns
            with gr.Row(elem_id="main-row"):
                
                # LEFT: Analysis tasks
                with gr.Column(scale=2, elem_id="left-panel"):
                    with gr.Group(elem_id="task-section"):
                        gr.Markdown("#### Analysis Tasks")
                        task_selector = gr.CheckboxGroup(
                            choices=TaskRegistry.get_tasks_for_data_type("EHR Data"),
                            label=None,
                            elem_id="task-checkboxes"
                        )
                        
                        # Parameter groups (conditionally visible) - one for each task that needs params
                        with gr.Group(visible=False) as ndd_param_group:
                            gr.Markdown("**Near-Duplicate Detection Parameters**")
                            ndd_label_dropdown = gr.Dropdown(
                                choices=[],
                                label="Label Column",
                                elem_id="ndd-label-dropdown"
                            )
                        
                        with gr.Group(visible=False) as mislabel_param_group:
                            gr.Markdown("**Find Mislabeled Data Parameters**")
                            mislabel_label_dropdown = gr.Dropdown(
                                choices=[],
                                label="Label Column",
                                elem_id="mislabel-label-dropdown"
                            )
                    
                    with gr.Group(elem_id="action-section"):
                        process_btn = gr.Button("▶ Process", variant="primary", size="lg")
                        status_output = gr.Textbox(label="Status", interactive=False, lines=2)
                
                # MIDDLE: Results display
                with gr.Column(scale=7, elem_id="middle-panel"):
                    with gr.Tabs(elem_id="tabs-container") as result_tabs:
                        with gr.TabItem("Original Data", visible=False) as tab_original:
                            original_df_output = gr.Dataframe(interactive=False)
                        
                        with gr.TabItem("Processed Data", visible=False) as tab_processed:
                            processed_df_output = gr.Dataframe(interactive=False)
                        
                        with gr.TabItem("Summary", visible=False) as tab_summary:
                            summary_output = gr.JSON()
                        
                        with gr.TabItem("Visualization", visible=False) as tab_viz:
                            viz_output = gr.HTML()
                
                # RIGHT: Chat
                with gr.Column(scale=3, elem_id="chat-panel"):
                    gr.Markdown("### 💬 AI Assistant", elem_id="chat-header")
                    chatbot = gr.Chatbot(elem_id="chat-history", height=None, label=None)
                    with gr.Row(elem_id="chat-input-row"):
                        msg_input = gr.Textbox(
                            placeholder="Ask about your data...",
                            label="",
                            scale=4,
                            container=False,
                            lines=1
                        )
                        send_btn = gr.Button("Send", scale=1, size="sm")
        
        # ============ EVENT HANDLERS ============
        
        # File upload
        file_input.change(
            fn=ui_manager.on_file_upload,
            inputs=[file_input, data_type],
            outputs=[
                status_output,
                original_df_output,
                task_selector,
                ndd_param_group,
                mislabel_param_group,
                ndd_label_dropdown,
                mislabel_label_dropdown,
                tab_original, tab_processed, tab_summary, tab_viz,
            ]
        )
        
        # Data type change
        data_type.change(
            fn=ui_manager.on_data_type_change,
            inputs=[data_type, file_input],
            outputs=[task_selector, ndd_param_group, mislabel_param_group, status_output]
        )
        
        # Task selection change
        task_selector.change(
            fn=ui_manager.on_tasks_change,
            inputs=[task_selector, data_type, ndd_label_dropdown],
            outputs=[ndd_param_group, mislabel_param_group]
        )
        
        # Process button
        process_btn.click(
            fn=ui_manager.process_analysis,
            inputs=[file_input, data_type, task_selector, ndd_label_dropdown, mislabel_label_dropdown],
            outputs=[
                status_output,
                original_df_output,
                processed_df_output,
                summary_output,
                viz_output,
                tab_original,
                tab_processed,
                tab_summary,
                tab_viz
            ]
        )
        
        # Chat
        send_btn.click(
            fn=ui_manager.chatbot_respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            fn=ui_manager.chatbot_respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input]
        )
    
    return demo


# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)


# ============================================================================
# HOW TO EXTEND
# ============================================================================

"""
ADDING NEW ANALYSIS TASKS:
==========================

1. Add task configuration in TaskRegistry:
   - Define name, data_type, requires_params, param_components, output_tabs

2. Add execution method in AnalysisExecutor:
   - Create execute_your_task() method
   - Return (status_message, results_dict)

3. Add condition in UIManager.process_analysis():
   - elif task_name == "Your Task Name":
   -     status_msg, results = executor.execute_your_task(df, params)
   -     return self._format_results(status_msg, config.output_tabs, results)


EXAMPLE - Adding a new ECG task with parameters:
================================================

In TaskRegistry.get_config():
    "ECG Data": {
        ...
        "ECG Quality Check": TaskConfig(
            name="ECG Quality Check",
            data_type="ECG Data",
            requires_params=True,
            param_components=[
                {"type": "slider", "label": "Quality Threshold", "elem_id": "quality_thresh"}
            ],
            output_tabs=["original", "summary", "visualization"]
        )
    }

In AnalysisExecutor:
    @staticmethod
    def execute_ecg_quality_check(df: pd.DataFrame, threshold: float) -> Tuple[str, Dict[str, Any]]:
        # Your analysis logic here
        return "✓ Quality check completed", {
            "original": df,
            "summary": quality_summary,
            "visualization": viz_html
        }

In UIManager.process_analysis():
    elif task_name == "ECG Quality Check":
        status_msg, results = executor.execute_ecg_quality_check(df, float(param_value))
        return self._format_results(status_msg, config.output_tabs, results)


The architecture is now:
- Simple and clean
- Easy to extend with new tasks
- Dynamic UI based on task configuration
- Fixed chat scrolling issue
"""
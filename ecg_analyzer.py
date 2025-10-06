"""
ECG Analysis Module
Provides modular ECG visualization and analysis functions
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from typing import List, Optional, Tuple, Dict, Any

class ECGAnalyzer:
    """Handles ECG data analysis and visualization"""
    
    # Standard 12-lead ECG leads
    STANDARD_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    @staticmethod
    def detect_leads(df: pd.DataFrame) -> List[str]:
        """Detect available ECG leads in the dataframe"""
        available_leads = []
        for lead in ECGAnalyzer.STANDARD_LEADS:
            if lead in df.columns:
                available_leads.append(lead)
        return available_leads
    
    @staticmethod
    def detect_time_column(df: pd.DataFrame) -> Optional[str]:
        """Detect the time column in the dataframe"""
        time_candidates = ['time', 'Time', 'TIME', 'timestamp', 'sample']
        for col in time_candidates:
            if col in df.columns:
                return col
        # If no explicit time column, use index
        return None
    
    @staticmethod
    def create_signal_plot(df: pd.DataFrame, leads: List[str], time_col: Optional[str] = None) -> str:
        """Create ECG signal waveform plot"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if time_col and time_col in df.columns:
            x_data = df[time_col]
            x_label = 'Time (ms)' if 'time' in time_col.lower() else time_col
        else:
            x_data = df.index
            x_label = 'Sample Index'
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(leads)))
        
        for idx, lead in enumerate(leads):
            if lead in df.columns:
                ax.plot(x_data, df[lead], label=f'Lead {lead}', 
                       linewidth=1.2, alpha=0.8, color=colors[idx])
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel('Amplitude (mV)', fontsize=11)
        ax.set_title('ECG Signal Waveform', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, ncol=min(4, len(leads)))
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        return ECGAnalyzer._fig_to_base64(fig)
    
    @staticmethod
    def create_histogram(df: pd.DataFrame, leads: List[str]) -> str:
        """Create histogram of signal amplitudes"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(leads)))
        
        for idx, lead in enumerate(leads):
            if lead in df.columns:
                ax.hist(df[lead].dropna(), bins=50, alpha=0.6, 
                       label=f'Lead {lead}', color=colors[idx], edgecolor='black')
        
        ax.set_xlabel('Amplitude (mV)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Signal Amplitudes', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        return ECGAnalyzer._fig_to_base64(fig)
    
    @staticmethod
    def create_scatter_plot(df: pd.DataFrame, lead_x: str = 'I', lead_y: str = 'II') -> str:
        """Create scatter plot comparing two leads"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        if lead_x in df.columns and lead_y in df.columns:
            ax.scatter(df[lead_x], df[lead_y], alpha=0.5, s=10, c='steelblue')
            ax.set_xlabel(f'Lead {lead_x} Amplitude (mV)', fontsize=11)
            ax.set_ylabel(f'Lead {lead_y} Amplitude (mV)', fontsize=11)
            ax.set_title(f'Lead {lead_x} vs Lead {lead_y}', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            correlation = df[[lead_x, lead_y]].corr().iloc[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'Selected leads not available', 
                   ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        return ECGAnalyzer._fig_to_base64(fig)
    
    @staticmethod
    def create_rolling_average(df: pd.DataFrame, leads: List[str], 
                               time_col: Optional[str] = None, window: int = 100) -> str:
        """Create rolling average plot"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        if time_col and time_col in df.columns:
            x_data = df[time_col]
            x_label = 'Time (ms)' if 'time' in time_col.lower() else time_col
        else:
            x_data = df.index
            x_label = 'Sample Index'
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(leads)))
        
        for idx, lead in enumerate(leads):
            if lead in df.columns:
                rolling_avg = df[lead].rolling(window=window, min_periods=1).mean()
                ax.plot(x_data, rolling_avg, label=f'Lead {lead} (MA-{window})', 
                       linewidth=1.5, alpha=0.8, color=colors[idx])
        
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel('Amplitude (mV)', fontsize=11)
        ax.set_title(f'Rolling Average (Window={window})', fontsize=13, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, ncol=min(4, len(leads)))
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        return ECGAnalyzer._fig_to_base64(fig)
    
    @staticmethod
    def create_all_visualizations(df: pd.DataFrame, leads: List[str], 
                                  viz_types: List[str]) -> str:
        """Create multiple visualizations based on selected types"""
        html_parts = []
        time_col = ECGAnalyzer.detect_time_column(df)
        
        for viz_type in viz_types:
            if viz_type == "Signal Waveform":
                img_base64 = ECGAnalyzer.create_signal_plot(df, leads, time_col)
                html_parts.append(f'<div style="margin-bottom: 30px;"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>')
            
            elif viz_type == "Histogram":
                img_base64 = ECGAnalyzer.create_histogram(df, leads)
                html_parts.append(f'<div style="margin-bottom: 30px;"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>')
            
            elif viz_type == "Scatter Plot":
                # Use first two available leads for scatter plot
                lead_x = leads[0] if len(leads) > 0 else 'I'
                lead_y = leads[1] if len(leads) > 1 else 'II'
                img_base64 = ECGAnalyzer.create_scatter_plot(df, lead_x, lead_y)
                html_parts.append(f'<div style="margin-bottom: 30px;"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>')
            
            elif viz_type == "Rolling Average":
                img_base64 = ECGAnalyzer.create_rolling_average(df, leads, time_col)
                html_parts.append(f'<div style="margin-bottom: 30px;"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>')
        
        if not html_parts:
            return '<div style="text-align:center; padding:40px;"><p>No visualizations selected</p></div>'
        
        return '<div style="text-align:center;">' + ''.join(html_parts) + '</div>'
    
    @staticmethod
    def _fig_to_base64(fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    @staticmethod
    def generate_statistics(df: pd.DataFrame, leads: List[str]) -> Dict[str, Any]:
        """Generate statistical summary for selected leads"""
        stats = {}
        for lead in leads:
            if lead in df.columns:
                lead_data = df[lead].dropna()
                stats[lead] = {
                    'mean': float(lead_data.mean()),
                    'std': float(lead_data.std()),
                    'min': float(lead_data.min()),
                    'max': float(lead_data.max()),
                    'median': float(lead_data.median()),
                    'q25': float(lead_data.quantile(0.25)),
                    'q75': float(lead_data.quantile(0.75))
                }
        return stats
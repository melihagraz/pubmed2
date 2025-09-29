# streamlit_app4.py

import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import math
import sqlite3
import warnings
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import io
import base64
import requests
import xml.etree.ElementTree as ET
from selection_model import LiteratureEnhancedAgent
from pubmed_searcher import SimplePubMedSearcher
from fpdf import FPDF
from datetime import datetime
from pathway_analyzer import PathwayAnalyzer, PathwayVisualizer
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression, Lasso


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class AgentConfig:
    target_metric: str = "roc_auc"
    target_threshold: Optional[float] = None
    budget_trials: int = 30
    budget_seconds: Optional[int] = None
    cv_splits: int = 5
    random_state: int = 42
    enable_optuna: bool = True
    optuna_timeout_per_trial: Optional[int] = 60
    imbalance_threshold: float = 0.15
    hitl_enabled: bool = False
    hitl_auto_blocklist: List[str] = None


class PDFReport(FPDF):
    """Custom FPDF class for generating the report."""
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Feature Selection Technical Report', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(220, 220, 220)
        self.cell(0, 8, title, 0, 1, 'L', 1)
        self.ln(4)
        
    def add_section_data(self, title, data, is_nested=False):
        """Adds a key-value pair or nested dictionary to the report."""
        self.set_font('Arial', '', 10)
        self.set_text_color(0, 0, 0) # Black text
        
        if isinstance(data, dict):
            if is_nested:
                self.set_font('Arial', 'B', 10)
                self.cell(0, 6, f"{title}:", 0, 1, 'L')
                self.ln(2)
            for key, value in data.items():
                if isinstance(value, dict):
                    self.add_section_data(key, value, is_nested=True)
                else:
                    self.cell(5) # Indent for sub-sections
                    self.set_font('Arial', '', 10)
                    self.cell(0, 6, f"{key}: {self._format_value(value)}", 0, 1, 'L')
        else:
            self.cell(0, 6, f"{title}: {self._format_value(data)}", 0, 1, 'L')
            
    def _format_value(self, value):
        if isinstance(value, (float, int)):
            # Format numbers for better readability
            return f"{value:.4f}" if isinstance(value, float) else f"{value:,}"
        elif isinstance(value, list):
            # Format lists into a readable string
            return ",".join(map(str, value))
        elif isinstance(value, str) and value.startswith("http"):
            # Add a hyperlink (render immediate row)
            self.set_text_color(0, 0, 255)
            self.set_font('', 'U')
            self.cell(0, 6, value, link=value, ln=1)
            self.set_text_color(0, 0, 0)
            self.set_font('', '')
            return ""
        return str(value)

 
# ---- Streamlit UI Functions ----

def create_literature_visualization(literature_results: List[dict]):
    """Create literature analysis visualizations"""
    if not literature_results:
        return None, None
    
    # Evidence scores bar chart
    df = pd.DataFrame(literature_results)
    df = df.sort_values('evidence_score', ascending=True)
    
    fig1 = px.bar(
        df.tail(15), 
        x='evidence_score', 
        y='feature_name',
        orientation='h',
        title="Literature Evidence Scores by Feature",
        labels={"evidence_score": "Evidence Score (0-5)", "feature_name": "Feature"},
        color='evidence_score',
        color_continuous_scale='Viridis'
    )
    fig1.update_layout(height=500)
    
    # Paper count distribution
    fig2 = px.histogram(
        df, 
        x='paper_count',
        nbins=20,
        title="Distribution of Paper Counts",
        labels={"paper_count": "Number of Papers", "count": "Number of Features"}
    )
    
    return fig1, fig2

def analyze_literature_results(literature_results: List[dict]) -> dict:
    """Analyze literature search results"""
    if not literature_results:
        return {}
    
    df = pd.DataFrame(literature_results)
    
    analysis = {
        'total_features': len(df),
        'total_papers': int(df['paper_count'].sum()),
        'avg_evidence_score': float(df['evidence_score'].mean()),
        'high_evidence_features': int(len(df[df['evidence_score'] > 2.0])),
        'zero_evidence_features': int(len(df[df['paper_count'] == 0])),
        'top_features': df.nlargest(5, 'evidence_score')[['feature_name', 'evidence_score', 'paper_count']].to_dict('records')
    }
    
    return analysis

def display_articles_for_feature(feature_name: str, articles: List[dict]):
    """Display articles for a specific feature in Streamlit"""
    if not articles:
        st.info(f"No articles found for {feature_name}")
        return
    
    st.markdown(f"**📚 Publications for {feature_name}:**")
    
    for i, article in enumerate(articles, 1):
        with st.expander(f"{i}. {article['title'][:100]}..."):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Authors:** {article['authors']}")
                st.markdown(f"**Journal:** {article['journal']}")
                st.markdown(f"**Year:** {article['year']}")
                if article['abstract'] != "No abstract available":
                    st.markdown(f"**Abstract:** {article['abstract']}")
            with col2:
                st.markdown(f"**PMID:** {article['pmid']}")
                st.markdown(f"[View on PubMed]({article['url']})")


# Function to generate pdf
def generate_pdf_report(report_data):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Add a timestamp to the top of the report
    pdf.set_font('Arial', 'I', 8)
    pdf.cell(0, 5, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'R')
    pdf.ln(5)

    # Final Model Summary
    pdf.chapter_title('Final Model Summary')
    summary = report_data.get('final_model_summary', {})
    pdf.add_section_data('Performance Metrics', summary.get('performance_metrics', {}))
    pdf.ln(2)
    pdf.add_section_data('Strategy Used', summary.get('feature_selection_strategy_used', {}))
    
    # List the selected features in a clean, bulleted format
    pdf.ln(4)
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(0, 6, 'Selected Features:', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    features = summary.get('selected_features', [])
    for feature in features:
        pdf.cell(5) # Indent
        pdf.cell(0, 6, f"- {feature}", 0, 1, 'L')
    pdf.ln(5)

    # Agent Configuration
    pdf.chapter_title('Agent Configuration')
    config_data = report_data.get('agent_configuration', {})
    pdf.add_section_data('Configuration Details', config_data)
    pdf.ln(5)

    # Trial History (as a simplified table)
    pdf.chapter_title('Full Trial History')
    history_df = pd.DataFrame(report_data.get('full_trial_history', []))
    if not history_df.empty:
        # Prepare the data for a simple table
        table_data = [['Trial', 'Strategy', 'Metric', '# Features', 'Duration (s)']]
        for index, row in history_df.iterrows():
            table_data.append([
                str(row['id']),
                row['plan']['strategy'],
                f"{row['result']['metric_value']:.4f}",
                str(row['result']['n_features']),
                f"{row['result']['duration_sec']:.2f}"
            ])
        
        pdf.set_font('Arial', '', 8)
        # Table heading
        for header in table_data[0]:
            pdf.cell(35, 7, str(header), 1, 0, 'C')
        pdf.ln()

        # Table rows
        for row in table_data[1:]:
            for item in row:
                pdf.cell(35, 7, str(item), 1, 0, 'C')
            pdf.ln()
    else:
        pdf.cell(0, 10, "No trial history available.", 0, 1, 'C')

    # Data Analysis Summary (if present)
    if 'data_analysis_summary' in report_data:
        pdf.ln(5)
        pdf.chapter_title('Data Analysis Summary')
        pdf.add_section_data('Summary', report_data['data_analysis_summary'])

    # Literature Analysis Summary
    if 'literature_analysis' in report_data and isinstance(report_data['literature_analysis'], list):
        pdf.ln(5)
        pdf.chapter_title('Literature Analysis Summary')
        lit_df = pd.DataFrame(report_data['literature_analysis'])
        if not lit_df.empty:
            # Create a simple table for literature results
            table_data = [['Feature', 'Papers', 'Score', 'Support']]
            for _, row in lit_df.iterrows():
                support = 'High' if row['evidence_score'] > 3.0 else 'Medium' if row['evidence_score'] > 1.0 else 'Low'
                table_data.append([
                    row['feature_name'],
                    str(row['paper_count']),
                    f"{row['evidence_score']:.1f}",
                    support
                ])

            pdf.set_font('Arial', '', 8)
            for header in table_data[0]:
                pdf.cell(40, 7, str(header), 1, 0, 'C')
            pdf.ln()
            for row in table_data[1:]:
                for item in row:
                    pdf.cell(40, 7, str(item), 1, 0, 'C')
                pdf.ln()
        else:
            pdf.cell(0, 10, "No literature analysis results to display.", 0, 1, 'C')

    return pdf

# ---- Streamlit Application ----

def main():
    """Main function to run the Streamlit app."""

    st.set_page_config(
        page_title="PubMed-Enhanced Feature Selection Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("PubMed-Enhanced Feature Selection Framework")
    st.markdown("""
    **An autonomous framework for feature selection with integrated literature analysis** — systematically identifies optimal features and validates them against scientific evidence.

    📊 **Key Capabilities:**
    - 🔬 **PubMed Integration**: Automated retrieval of biomedical literature related to selected features
    - 📚 **Evidence Scoring**: Quantitative ranking of features based on their representation in peer-reviewed publications
    - 🎯 **Literature-Informed Decision Support**: Selection strategies dynamically adjusted according to the strength of literature evidence
    - 📈 **Publication Analytics**: Statistical and visual analyses of research trends and evidence distributions
    - 📖 **Article Repository**: Structured access to PubMed articles corresponding to individual features

    ⚠️ **Important Note**: For PubMed search to function reliably, uploaded datasets must contain
    **descriptive and biologically meaningful column names** (e.g., gene symbols, biomarkers).
    Generic names such as *feature1* or *var2* may result in incomplete or inaccurate matches.
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Dataset for feature selection"
    )
    
    # PubMed Configuration
    st.sidebar.subheader("🔬 PubMed Literature Analysis")
    is_disabled = uploaded_file is None
    enable_pubmed = st.sidebar.checkbox("Enable PubMed Search", value=False, help="Search scientific literature for selected features", disabled=is_disabled)
    
    pubmed_searcher = None
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        d = False
        if enable_pubmed:
            d = True
            email = st.sidebar.text_input(
                "NCBI Email (required)", 
                help="Required by NCBI for PubMed API access - get free account at ncbi.nlm.nih.gov"
            )
            
            api_key = st.sidebar.text_input(
                "NCBI API Key (optional)",
                type="password",
                help="Optional - increases rate limit from 3 to 10 requests/sec"
            )
            
            disease_context = st.sidebar.text_input(
                "Disease/Condition Context",
                placeholder="e.g., cancer, diabetes, alzheimer",
                help="Helps focus literature search on specific medical condition"
            )
            
            if email:
                pubmed_searcher = SimplePubMedSearcher(
                    email=email, 
                    api_key=api_key if api_key else None
                )
                d = False
                st.sidebar.success("✅ PubMed search enabled")
            else:
                st.sidebar.warning("⚠️ Email required for PubMed API")
                enable_pubmed = False

        # Target column selection
        target_col = st.sidebar.selectbox(
            "🎯 Select target variable",
            options=df.columns.tolist(),
            index=len(df.columns)-1,
            help="Variable to predict"
        )

        # Configuration
        st.sidebar.subheader("🔧 Agent Settings")
        
        # Task type detection
        y = df[target_col]
        is_classification = len(y.unique()) <= 20 and (y.dtype == 'object' or y.dtype == 'int64')
        task_type = "Classification" if is_classification else "Regression"
        st.sidebar.info(f"📋 Detected task: **{task_type}**")

        # Metric selection
        if is_classification:
            default_metrics = ["roc_auc", "f1_macro", "accuracy", "precision_macro", "recall_macro"]
        else:
            default_metrics = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
        
        target_metric = st.sidebar.selectbox(
            "📊 Target metric",
            options=default_metrics,
            help="Metric to optimize"
        )

        # Budget settings
        budget_trials = st.sidebar.slider(
            "🔄 Max trials",
            min_value=5,
            max_value=50,
            value=15,
            help="Number of trials to run"
        )

        budget_seconds = st.sidebar.slider(
            "⏱️ Max time (seconds)",
            min_value=30,
            max_value=300,
            value=120,
            help="Maximum runtime"
        )

        cv_splits = st.sidebar.slider(
            "🔀 Number of CV folds",
            min_value=3,
            max_value=10,
            value=5
        )

        st.sidebar.subheader("🧬 Real Pathway Analysis")
        enable_pathway = st.sidebar.checkbox("Enable Real Pathway Analysis", value=False, 
                                           help="Connect to external pathway databases for enrichment analysis")
        
        pathway_database = "KEGG_2021_Human"  # default
        if enable_pathway:
            organism_options = {
                "Human": "Homo sapiens",
                "Mouse": "Mus musculus"
            }
            organism_choice = st.sidebar.selectbox(
                "🧬 Select organism",
                options=list(organism_options.keys()),
                index=0  # default Human
            )

            
            # Available databases from real APIs
            pathway_databases = {
                "KEGG_2021_Human": "KEGG Pathways (Human 2021)",
                "Reactome_2022": "Reactome Pathways (2022)", 
                "GO_Biological_Process_2023": "Gene Ontology Biological Process",
                "GO_Molecular_Function_2023": "Gene Ontology Molecular Function",
                "WikiPathway_2023_Human": "WikiPathways (Human)",
                "MSigDB_Hallmark_2020": "MSigDB Hallmark Gene Sets",
                "BioPlanet_2019": "BioPlanet Pathways",
                "DisGeNET": "DisGeNET Disease Associations",
                "Human_Phenotype_Ontology": "Human Phenotype Ontology"
            }
            
            pathway_database = st.sidebar.selectbox(
                "Select Pathway Database",
                options=list(pathway_databases.keys()),
                format_func=lambda x: pathway_databases[x],
                help="Choose which pathway database to use for enrichment analysis"
            )
            
            st.sidebar.info("🌐 Connects to real pathway databases via API")

        # Advanced settings
        with st.sidebar.expander("🔬 Advanced Settings"):
            target_threshold = st.number_input(
                "Target threshold (optional)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.01,
                help="Stop early if reached (0 = disabled)"
            )
            
            enable_optuna = st.checkbox(
                "Optuna hyperparameter optimization",
                value=False,
                help="Better results but slower"
            )
            
            imbalance_threshold = st.slider(
                "Class imbalance threshold",
                min_value=0.05,
                max_value=0.5,
                value=0.15,
                help="Minimum class ratio"
            )

            hitl_enabled = st.checkbox(
                "Human-in-the-loop approval",
                value=False,
                help="Manually approve selected features"
            )

            if hitl_enabled:
                blocklist_text = st.text_input(
                    "Blocked features (comma-separated)",
                    help="Features containing these names will be auto-rejected"
                )
                hitl_auto_blocklist = [x.strip() for x in blocklist_text.split(",") if x.strip()]
            else:
                hitl_auto_blocklist = []
        
        # Run button
        if st.button("🚀 Start Feature Selection", type="primary", disabled=d):
            # Prepare data
            X = df.drop(columns=[target_col])
            y = df[target_col]

            # Create config
            config = AgentConfig(
                target_metric=target_metric,
                target_threshold=target_threshold if target_threshold > 0 else None,
                budget_trials=budget_trials,
                budget_seconds=budget_seconds,
                cv_splits=cv_splits,
                random_state=42,
                enable_optuna=enable_optuna,
                imbalance_threshold=imbalance_threshold,
                hitl_enabled=hitl_enabled,
                hitl_auto_blocklist=hitl_auto_blocklist
            )

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_container = st.container()

            # Results containers
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📊 Real-time Progress")
                progress_chart = st.empty()
                
            with col2:
                st.subheader("📈 Best Results")
                best_metrics = st.empty()
            
            # Initialize tracking
            trial_scores = []
            trial_features = []

            def progress_callback(trial_num, total_trials, result):
                progress = trial_num / total_trials
                progress_bar.progress(progress)
                status_text.text(f"Trial {trial_num}/{total_trials} - Current score: {result.metric_value:.4f}")
                
                # Track results
                trial_scores.append(result.metric_value)
                trial_features.append(result.n_features)
                
                # Update progress chart
                if len(trial_scores) > 1:
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=["Metric Value", "Number of Features"],
                        vertical_spacing=0.1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            y=trial_scores,
                            mode='lines+markers',
                            name='Metric',
                            line=dict(color='blue')
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            y=trial_features,
                            mode='lines+markers',
                            name='Features',
                            line=dict(color='green')
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=400, showlegend=False)
                    progress_chart.plotly_chart(fig, use_container_width=True)

                # Update best metrics
                best_score = max(trial_scores)
                best_idx = trial_scores.index(best_score)
                best_n_features = trial_features[best_idx]
                
                best_metrics.metric(
                    label=f"Best {target_metric.upper()}",
                    value=f"{best_score:.4f}",
                    delta=f"{best_n_features} features"
                )

            email, api_key, disease_context = None, None, None

            # Run agent
            try:
                agent = LiteratureEnhancedAgent(
                    config, 
                    pubmed_searcher, 
                    disease_context=disease_context if disease_context else None
                )
                
                with st.spinner("🤖 Running feature selection agent..."):
                    results = agent.run(X, y, progress_callback=progress_callback)

                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()

                # Display final results
                st.success("✅ Feature selection completed!")
                
                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Best Score", f"{results['best_score']:.4f}")
                with col2:
                    st.metric("Selected Features", len(results['best_features']))
                with col3:
                    st.metric("Total Trials", results['trials'])
                with col4:
                    st.metric("Duration", f"{results['elapsed_sec']:.1f}s")

                # Data analysis summary
                st.subheader("🔍 Data Analysis Summary")
                sense_info = results['sense_info']
                
                analysis_col1, analysis_col2 = st.columns(2)
                with analysis_col1:
                    st.info(f"""
                    **Dataset Info:**
                    - Samples: {sense_info['n_samples']:,}
                    - Total features: {sense_info['n_features']}
                    - Numeric features: {sense_info['n_numeric']}
                    - Categorical features: {sense_info['n_categorical']}
                    - Task type: {sense_info['task']}
                    """)
                
                with analysis_col2:
                    warnings_list = []
                    if sense_info.get('imbalanced', False):
                        warnings_list.append(f"⚠️ Class imbalance detected (min ratio: {sense_info.get('min_class_ratio', 0):.3f})")
                    if sense_info.get('leakage_suspect', False):
                        warnings_list.append(f"🚨 Possible data leakage (max corr: {sense_info.get('max_abs_corr', 0):.3f})")
                    
                    if warnings_list:
                        st.warning("\n".join(warnings_list))
                    else:
                        st.success("✅ No data quality warnings")
                print(333)    
                # PubMed Literature Analysis
                # PubMed Literature Analysis
                # PubMed Literature Analysis - COMPLETE REPLACEMENT
                if enable_pubmed and pubmed_searcher and results['best_features']:
                    st.subheader("🔬 Advanced Literature Analysis")
                    
                    # Show search configuration
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"**Features to Analyze:** {len(results['best_features'])}")
                    with col2:
                        st.info(f"**Disease Context:** {disease_context if disease_context else 'General'}")
                    with col3:
                        st.info(f"**API Status:** {'Premium' if pubmed_searcher.api_key else 'Standard'}")
                    
                    # Pre-search validation
                    features_to_search = results['best_features'][:20]  # Limit for rate limiting
                    if len(results['best_features']) > 20:
                        st.warning(f"⚠️ Limiting search to top 20 features (you have {len(results['best_features'])}) to respect API limits.")
                    
                    # Literature search execution
                    with st.spinner("🔍 Conducting comprehensive literature analysis..."):
                        # Progress tracking
                        search_progress = st.progress(0)
                        search_status = st.empty()
                        results_preview = st.empty()
                        
                        def enhanced_progress_callback(current, total, feature):
                            progress = current / total
                            search_progress.progress(progress)
                            search_status.text(f"Analyzing literature for: {feature} ({current}/{total})")
                            
                            # Show intermediate results
                            if current > 1 and current % 3 == 0:
                                partial_msg = f"✅ Processed {current} features, continuing analysis..."
                                results_preview.success(partial_msg)
                        
                        # Execute batch search
                        start_time = time.time()
                        try:
                            literature_results = pubmed_searcher.batch_search(
                                features_to_search,
                                disease_context=disease_context if disease_context else None,
                                progress_callback=enhanced_progress_callback
                            )
                            search_duration = time.time() - start_time
                            
                        except Exception as e:
                            search_progress.empty()
                            search_status.empty()
                            results_preview.empty()
                            st.error(f"❌ Literature search failed: {str(e)}")
                            st.markdown("**Troubleshooting steps:**")
                            st.markdown("- Check internet connection")
                            st.markdown("- Verify NCBI email is correct")
                            st.markdown("- Try again in a few minutes (rate limiting)")
                            st.markdown("- Consider getting an NCBI API key")
                            literature_results = []
                        
                        # Clear progress indicators
                        search_progress.empty()
                        search_status.empty()
                        results_preview.empty()
                    
                    if literature_results:
                        # Analyze search results
                        successful_searches = [r for r in literature_results if not r.get('error') and r.get('paper_count', 0) >= 0]
                        failed_searches = [r for r in literature_results if r.get('error')]
                        
                        # Summary of search execution
                        st.success(f"✅ Literature analysis completed in {search_duration:.1f} seconds")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Features Analyzed", len(literature_results))
                        with col2:
                            st.metric("Successful Searches", len(successful_searches))
                        with col3:
                            st.metric("Failed Searches", len(failed_searches))
                        with col4:
                            avg_papers = sum(r.get('paper_count', 0) for r in successful_searches) / max(len(successful_searches), 1)
                            st.metric("Avg Papers/Feature", f"{avg_papers:.1f}")
                        
                        # Handle failed searches
                        if failed_searches:
                            with st.expander(f"⚠️ Search Issues ({len(failed_searches)} features)"):
                                failure_data = []
                                for failed in failed_searches:
                                    failure_data.append({
                                        'Feature': failed['feature_name'],
                                        'Error': failed.get('error', 'Unknown error')[:50] + '...' if len(failed.get('error', '')) > 50 else failed.get('error', 'Unknown error'),
                                        'Suggestion': 'Check gene symbol format' if 'not found' in failed.get('error', '').lower() else 'Network/API issue'
                                    })
                                if failure_data:
                                    st.dataframe(pd.DataFrame(failure_data), use_container_width=True)
                        
                        # Continue with successful results
                        if successful_searches:
                            literature_results = successful_searches
                            
                            # Advanced literature analysis
                            lit_analysis = analyze_literature_results(literature_results)
                            
                            # Literature summary dashboard
                            st.subheader("📊 Literature Evidence Dashboard")
                            
                            # Main metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                total_papers = lit_analysis.get('total_papers', 0)
                                st.metric("Total Papers Found", f"{total_papers:,}")
                            with col2:
                                avg_evidence = lit_analysis.get('avg_evidence_score', 0)
                                st.metric("Average Evidence Score", f"{avg_evidence:.1f}/5.0")
                            with col3:
                                high_evidence = lit_analysis.get('high_evidence_features', 0)
                                st.metric("High Evidence Features", high_evidence)
                            with col4:
                                zero_evidence = lit_analysis.get('zero_evidence_features', 0)
                                st.metric("Novel Features", zero_evidence)
                            
                            # Evidence quality assessment
                            quality_col1, quality_col2 = st.columns(2)
                            with quality_col1:
                                if avg_evidence >= 3.0:
                                    st.success("🎯 **Strong literature support** - Selected features are well-studied")
                                elif avg_evidence >= 2.0:
                                    st.info("📈 **Moderate literature support** - Decent evidence base")
                                else:
                                    st.warning("🔍 **Limited literature support** - Consider domain expert review")
                            
                            with quality_col2:
                                novelty_ratio = zero_evidence / max(len(literature_results), 1)
                                if novelty_ratio > 0.5:
                                    st.info("💡 **High novelty potential** - Many understudied features")
                                elif novelty_ratio > 0.3:
                                    st.info("⚖️ **Balanced mix** - Known and novel features")
                                else:
                                    st.success("📚 **Well-established features** - Strong literature foundation")
                            
                            # Advanced visualizations
                            st.subheader("📈 Literature Evidence Visualizations")
                            
                            # Create comprehensive visualizations
                            fig1, fig2 = create_literature_visualization(literature_results)
                            
                            if fig1 and fig2:
                                viz_col1, viz_col2 = st.columns(2)
                                with viz_col1:
                                    st.plotly_chart(fig1, use_container_width=True)
                                with viz_col2:
                                    st.plotly_chart(fig2, use_container_width=True)
                            
                            # Evidence vs. Selection Rank Analysis
                            st.subheader("🎯 Evidence vs. Feature Rank Analysis")
                            
                            # Create rank vs evidence plot
                            rank_data = []
                            for idx, result in enumerate(literature_results):
                                rank_data.append({
                                    'Feature': result['feature_name'],
                                    'Selection_Rank': idx + 1,
                                    'Evidence_Score': result['evidence_score'],
                                    'Paper_Count': result['paper_count'],
                                    'Search_Strategy': result.get('search_strategy', 'unknown')
                                })
                            
                            if rank_data:
                                rank_df = pd.DataFrame(rank_data)
                                
                                # Scatter plot: Selection rank vs Evidence score
                                fig_rank = px.scatter(
                                    rank_df,
                                    x='Selection_Rank',
                                    y='Evidence_Score',          # <<< BURAYA DÜZELTME
                                    size='Paper_Count',
                                    color='Search_Strategy',
                                    hover_name='Feature',
                                    title='Feature Selection Rank vs Literature Evidence',
                                    labels={
                                        'Selection_Rank': 'Feature Selection Rank (1=best)',
                                        'Evidence_Score': 'Literature Evidence Score',
                                        'Paper_Count': 'Number of Papers'
                                    }
                                )
                                fig_rank.update_layout(height=500)
                                st.plotly_chart(fig_rank, use_container_width=True)
                                
                                # Analysis insights
                                top_ranked_evidence = rank_df.head(5)['Evidence_Score'].mean()
                                bottom_ranked_evidence = rank_df.tail(5)['Evidence_Score'].mean()
                                
                                if top_ranked_evidence > bottom_ranked_evidence + 0.5:
                                    st.success("✅ **Excellent correlation**: Top-ranked features have stronger literature support")
                                elif top_ranked_evidence > bottom_ranked_evidence:
                                    st.info("📊 **Good correlation**: Some alignment between ranking and literature evidence")
                                else:
                                    st.warning("🔍 **Novel discoveries possible**: Top features may be understudied but promising")
                            
                            # Top Literature-Supported Features
                            st.subheader("🏆 Top Literature-Supported Features")
                            
                            if lit_analysis.get('top_features'):
                                top_features_enhanced = []
                                for feature_data in lit_analysis['top_features']:
                                    # Find full result data
                                    full_result = next((r for r in literature_results if r['feature_name'] == feature_data['feature_name']), {})
                                    
                                    enhanced_data = {
                                        'Feature': feature_data['feature_name'],
                                        'Evidence Score': f"{feature_data['evidence_score']:.1f}/5.0",
                                        'Papers': feature_data['paper_count'],
                                        'Articles Retrieved': len(full_result.get('articles', [])),
                                        'Search Strategy': full_result.get('search_strategy', 'Unknown'),
                                        'Support Level': '🔥 High' if feature_data['evidence_score'] > 3.0 
                                                       else '📈 Medium' if feature_data['evidence_score'] > 1.0 
                                                       else '❓ Low'
                                    }
                                    top_features_enhanced.append(enhanced_data)
                                
                                top_df = pd.DataFrame(top_features_enhanced)
                                st.dataframe(top_df, use_container_width=True)
                            
                            # Detailed Article Browser
                            st.subheader("📚 Detailed Article Browser")
                            
                            # Filter features with articles
                            features_with_articles = [r for r in literature_results if r.get('articles', []) and len(r['articles']) > 0]
                            
                            if features_with_articles:
                                # Sort by evidence score for better organization
                                features_with_articles.sort(key=lambda x: x['evidence_score'], reverse=True)
                                
                                # Limit to top 12 to avoid too many tabs
                                features_to_display = features_with_articles[:12]
                                
                                st.info(f"📖 Showing detailed articles for top {len(features_to_display)} features with available literature")
                                
                                # Create tabs for each feature
                                tab_labels = []
                                tab_contents = []
                                
                                for feature_data in features_to_display:
                                    feature_name = feature_data['feature_name']
                                    evidence_score = feature_data['evidence_score']
                                    article_count = len(feature_data.get('articles', []))
                                    
                                    # Create informative tab label
                                    short_name = feature_name[:12] + "..." if len(feature_name) > 12 else feature_name
                                    tab_label = f"{short_name} ({evidence_score:.1f}★, {article_count}📄)"
                                    tab_labels.append(tab_label)
                                    tab_contents.append(feature_data)
                                
                                # Create tabs
                                if tab_labels:
                                    tabs = st.tabs(tab_labels)
                                    
                                    for tab, feature_data in zip(tabs, tab_contents):
                                        with tab:
                                            # Feature summary header
                                            st.markdown(f"### 🧬 {feature_data['feature_name']}")
                                            
                                            # Stats row
                                            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                                            with stat_col1:
                                                st.metric("Evidence Score", f"{feature_data['evidence_score']:.1f}/5.0")
                                            with stat_col2:
                                                st.metric("Total Papers", feature_data['paper_count'])
                                            with stat_col3:
                                                st.metric("Articles Retrieved", len(feature_data.get('articles', [])))
                                            with stat_col4:
                                                strategy = feature_data.get('search_strategy', 'unknown').replace('_', ' ').title()
                                                st.metric("Search Strategy", strategy)
                                            
                                            # Search query info
                                            if feature_data.get('search_query'):
                                                with st.expander("🔍 Search Query Details"):
                                                    st.code(feature_data['search_query'])
                                                    st.markdown(f"**Strategy Used:** {strategy}")
                                                    if feature_data.get('disease_context'):
                                                        st.markdown(f"**Disease Context:** {feature_data['disease_context']}")
                                            
                                            # Display articles
                                            articles = feature_data.get('articles', [])
                                            if articles:
                                                st.markdown("#### 📄 Related Publications")
                                                
                                                for i, article in enumerate(articles, 1):
                                                    with st.expander(f"📄 Article {i}: {article['title'][:80]}{'...' if len(article['title']) > 80 else ''}"):
                                                        # Article details in columns
                                                        art_col1, art_col2 = st.columns([3, 1])
                                                        
                                                        with art_col1:
                                                            st.markdown(f"**Title:** {article['title']}")
                                                            st.markdown(f"**Authors:** {article['authors']}")
                                                            st.markdown(f"**Journal:** {article['journal']} ({article['year']})")
                                                            
                                                            if article['abstract'] and article['abstract'] != "No abstract available":
                                                                st.markdown(f"**Abstract:**")
                                                                st.markdown(f"*{article['abstract']}*")
                                                        
                                                        with art_col2:
                                                            st.markdown(f"**PMID:** {article['pmid']}")
                                                            st.markdown(f"**Year:** {article['year']}")
                                                            
                                                            # PubMed link
                                                            pubmed_url = article.get('url', f"https://pubmed.ncbi.nlm.nih.gov/{article['pmid']}/")
                                                            st.markdown(f"[🔗 View on PubMed]({pubmed_url})")
                                            else:
                                                st.info("No detailed articles available for this feature.")
                            
                            else:
                                st.warning("🔍 No detailed articles were retrieved for any features.")
                                st.markdown("**Possible reasons:**")
                                st.markdown("- Features may not be standard gene symbols")
                                st.markdown("- Limited recent publications in PubMed")
                                st.markdown("- Network issues during article fetch")
                                st.markdown("- Rate limiting preventing full article retrieval")
                                
                                # Show what was searched
                                st.markdown("**Features searched:**")
                                searched_features = [r['feature_name'] for r in literature_results[:10]]
                                st.code(", ".join(searched_features))
                            
                            # Comprehensive Results Table
                            with st.expander("📋 Complete Literature Analysis Results"):
                                comprehensive_data = []
                                for result in literature_results:
                                    comprehensive_data.append({
                                        'Feature': result['feature_name'],
                                        'Papers': result['paper_count'],
                                        'Evidence Score': f"{result['evidence_score']:.1f}/5.0",
                                        'Articles Retrieved': len(result.get('articles', [])),
                                        'Search Strategy': result.get('search_strategy', 'unknown').replace('_', ' ').title(),
                                        'Support Level': '🔥 High' if result['evidence_score'] > 3.0 
                                                       else '📈 Medium' if result['evidence_score'] > 1.5 
                                                       else '📊 Low' if result['evidence_score'] > 0.5
                                                       else '❓ Minimal',
                                        'Query Used': result.get('search_query', '')[:60] + '...' if len(result.get('search_query', '')) > 60 else result.get('search_query', '')
                                    })
                                
                                if comprehensive_data:
                                    comprehensive_df = pd.DataFrame(comprehensive_data)
                                    st.dataframe(comprehensive_df, use_container_width=True, height=400)
                                    
                                    # Download option for results
                                    csv_data = comprehensive_df.to_csv(index=False)
                                    st.download_button(
                                        label="📊 Download Literature Analysis (CSV)",
                                        data=csv_data,
                                        file_name=f"literature_analysis_{len(literature_results)}_features.csv",
                                        mime="text/csv"
                                    )
                            
                            # Research Insights and Recommendations
                            st.subheader("💡 Research Insights & Recommendations")
                            
                            insights = []
                            recommendations = []
                            
                            # Generate insights based on results
                            high_evidence_count = len([r for r in literature_results if r['evidence_score'] > 3.0])
                            moderate_evidence_count = len([r for r in literature_results if 1.5 <= r['evidence_score'] <= 3.0])
                            low_evidence_count = len([r for r in literature_results if r['evidence_score'] < 1.5])
                            
                            # Evidence distribution insights
                            if high_evidence_count > moderate_evidence_count + low_evidence_count:
                                insights.append("🎯 **Well-established biomarkers**: Most selected features have strong literature support")
                                recommendations.append("✅ Consider prioritizing these features for clinical validation")
                            elif low_evidence_count > high_evidence_count + moderate_evidence_count:
                                insights.append("💡 **Novel discovery potential**: Many features are understudied")
                                recommendations.append("🔬 Consider additional experimental validation for novel features")
                            else:
                                insights.append("⚖️ **Balanced selection**: Mix of established and novel features")
                                recommendations.append("📊 Validate known biomarkers first, then explore novel ones")
                            
                            # Total papers insight
                            total_papers = sum(r['paper_count'] for r in literature_results)
                            if total_papers > 500:
                                insights.append(f"📚 **Extensive literature base**: {total_papers:,} total papers found")
                                recommendations.append("📖 Consider systematic literature review for comprehensive analysis")
                            elif total_papers > 100:
                                insights.append(f"📊 **Moderate literature base**: {total_papers:,} papers provide good foundation")
                            else:
                                insights.append(f"🔍 **Limited literature**: Only {total_papers:,} papers found")
                                recommendations.append("⚠️ Consider domain expert consultation for understudied features")
                            
                            # Disease context insights
                            if disease_context:
                                disease_specific_high = len([r for r in literature_results 
                                                           if r['evidence_score'] > 2.5 and r.get('search_strategy') == 'disease_biomarker'])
                                if disease_specific_high > 0:
                                    insights.append(f"🎯 **Disease-specific relevance**: {disease_specific_high} features show strong {disease_context} association")
                                    recommendations.append(f"🏥 Consider clinical studies focusing on {disease_context}")
                            
                            # Display insights
                            if insights:
                                insight_col1, insight_col2 = st.columns(2)
                                
                                with insight_col1:
                                    st.markdown("**🔍 Key Insights:**")
                                    for insight in insights:
                                        st.markdown(f"- {insight}")
                                
                                with insight_col2:
                                    st.markdown("**💡 Recommendations:**")
                                    for rec in recommendations:
                                        st.markdown(f"- {rec}")
                            
                            # Performance summary
                            st.markdown("---")
                            perf_col1, perf_col2, perf_col3 = st.columns(3)
                            with perf_col1:
                                st.markdown(f"**⏱️ Analysis Duration:** {search_duration:.1f} seconds")
                            with perf_col2:
                                success_rate = len(successful_searches) / len(literature_results) * 100
                                st.markdown(f"**✅ Success Rate:** {success_rate:.1f}%")
                            with perf_col3:
                                articles_retrieved = sum(len(r.get('articles', [])) for r in literature_results)
                                st.markdown(f"**📄 Articles Retrieved:** {articles_retrieved}")
                        
                        else:
                            st.error("❌ No successful literature searches completed.")
                            st.markdown("**Troubleshooting suggestions:**")
                            st.markdown("- Verify email is associated with NCBI account")
                            st.markdown("- Check internet connectivity")
                            st.markdown("- Ensure feature names are standard gene symbols (e.g., BRCA1, TP53)")
                            st.markdown("- Try again after a few minutes (rate limiting)")
                            st.markdown("- Consider obtaining an NCBI API key for better performance")
                    
                    else:
                        st.error("❌ Literature analysis failed to execute.")
                        st.markdown("**This could be due to:**")
                        st.markdown("- Network connectivity issues")
                        st.markdown("- NCBI API server problems")
                        st.markdown("- Invalid email or API configuration")
                        st.markdown("- Rate limiting or server overload")
                        
                        if st.button("🔄 Retry Literature Analysis"):
                            st.experimental_rerun()

                def convert_numpy_types(data):
                    """Recursively converts NumPy types to native Python types."""
                     
                    
                    if isinstance(data, (np.int64, np.int32, np.int16, np.int8)):
                        return int(data)
                    if isinstance(data, (np.float64, np.float32, np.float16)):
                        return float(data)
                    if isinstance(data, np.ndarray):
                        return data.tolist()
                    if isinstance(data, np.bool_):
                        return bool(data)
                    if isinstance(data, dict):
                        return {k: convert_numpy_types(v) for k, v in data.items()}
                    if isinstance(data, list):
                        return [convert_numpy_types(i) for i in data]
                    if isinstance(data, tuple):
                        return tuple(convert_numpy_types(i) for i in data)
                    return data
                # Pathway Analysis (fixed: real analyzer + visualizer, no undefined variables)
                path_results = []
                
                if enable_pathway:
                    st.subheader("🧬 Real Pathway Enrichment Analysis")
                    
                    if not results['best_features']:
                        st.info("No best features found to analyze.")
                    else:
                        try:
                            # Import the CORRECT analyzer and visualizer from pathway_analyzer.py
                            from pathway_analyzer import PathwayAnalyzer, PathwayVisualizer
                            
                            # Instantiate the PathwayAnalyzer correctly
                            # It uses the 'gseapy' library to connect to Enrichr services.
                            # We pass the selected database as a library to search.
                            pa = PathwayAnalyzer(
                                libraries=[pathway_database],
                                use_gseapy=True,
                                organism=organism_choice # Assuming human genes
                            )
                            
                            st.success("✅ Pathway analyzer initialized. Ready to connect to external databases.")
                            
                            # Show analysis parameters
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.info(f"**Database:** {pathway_database}")
                            with col2: 
                                st.info(f"**Features:** {len(results['best_features'])}")
                            with col3:
                                st.info(f"**Service:** gseapy (Enrichr API)")
                            
                            # Show gene list being analyzed
                            with st.expander("🧬 Genes Being Analyzed"):
                                gene_df = pd.DataFrame({
                                    'Gene Symbol': results['best_features'],
                                    'Index': range(1, len(results['best_features']) + 1)
                                })
                                st.dataframe(gene_df, use_container_width=True)
                            
                            # Perform pathway analysis
                            with st.spinner(f"🔬 Running pathway enrichment analysis using {pathway_database}..."):
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                progress_bar.progress(25)
                                status_text.text(f"Submitting {len(results['best_features'])} genes...")
                                
                                # Call the analysis function with the correct parameters
                                path_results = pa.analyze_pathways(
                                    features=results['best_features'],
                                    max_pathways=25,
                                    max_p_value=0.1 
                                )
                                
                                progress_bar.progress(100)
                                status_text.text("Analysis complete!")
                                time.sleep(1)
                                progress_bar.empty()
                                status_text.empty()
                            
                            if path_results:
                                st.success(f"🎯 Found {len(path_results)} enriched pathways!")
                                
                                # --- Geri kalan görselleştirme kodu aynı kalabilir ---
                                # Prepare visualization data
                                viz = PathwayVisualizer()
                                df_path = viz.prepare_results_for_plotting(path_results)
                                
                                if isinstance(df_path, list):
                                    df_path = pd.DataFrame(df_path)
                                
                                # Main visualization
                                if len(df_path) > 0:
                                    # Sort by combined score or p-value
                                    if 'enrichment_score' in df_path.columns and df_path['enrichment_score'].sum() > 0:
                                        df_plot = df_path.sort_values('enrichment_score', ascending=True).tail(15)
                                        x_col = 'enrichment_score'
                                        x_label = 'Combined Score'
                                        title = "Top Enriched Pathways (Combined Score)"
                                    else:
                                        df_plot = df_path.sort_values('neg_log_p', ascending=True).tail(15)
                                        x_col = 'neg_log_p'
                                        x_label = '-log10(p-value)'
                                        title = "Top Enriched Pathways (-log10 p-value)"
                                    hover_fields = {
                                        'p_value': ':.2e',
                                        'p_adj': ':.2e',
                                        'feature_count': True,
                                        'enrichment_score': ':.2f',
                                    }

                                    # Sadece varsa ekle
                                    # Sadece varsa ekle
                                    if 'enrichment_score' in df_plot.columns:
                                        hover_fields['enrichment_score'] = ':.1f'             
                                    fig_pathways = px.bar(
                                        df_plot,
                                        x=x_col,
                                        y='pathway_name',
                                        orientation='h',
                                        title=title,
                                        labels={x_col: x_label, 'pathway_name': 'Pathway'},
                                        color='feature_count',
                                        color_continuous_scale='viridis',
                                        hover_data= hover_fields
                                    )
                                    fig_pathways.update_layout(
                                        height=max(500, len(df_plot) * 35),
                                        yaxis={'categoryorder': 'total ascending'},
                                        showlegend=True
                                    )
                                    st.plotly_chart(fig_pathways, use_container_width=True)
                                
                                # Alternative scatter plot view
# Alternative scatter plot view
                                if len(df_path) > 5:
                                    st.subheader("📊 Pathway Enrichment Scatter Plot")
                                    
                                    # Determine color column based on what's available
                                    if 'enrichment_score' in df_path.columns and df_path['enrichment_score'].sum() > 0:
                                        color_col = 'enrichment_score'
                                        color_label = 'Combined Score'
                                    else:
                                        color_col = 'p_adj'
                                        color_label = 'Adjusted P-value'
                                    
                                    fig_scatter = px.scatter(
                                        df_path.head(20),
                                        x='enrichment_score',
                                        y='neg_log_p',
                                        size='feature_count',
                                        color=color_col,  # <-- ARTIK DİNAMİK
                                        hover_name='pathway_name',
                                        title='Pathway Enrichment Overview',
                                        labels={
                                            'enrichment_score': 'Enrichment Score',
                                            'neg_log_p': '-log10(p-value)',
                                            'feature_count': 'Gene Count',
                                            color_col: color_label
                                        },
                                        color_continuous_scale='plasma'
                                    )
                                    fig_scatter.update_layout(height=500)
                                    st.plotly_chart(fig_scatter, use_container_width=True)
                                
                                # Detailed results table
                                with st.expander("📋 Detailed Pathway Results"):
                                    display_df = df_path.copy()
                                    
                                    # Format numerical columns
                                    display_df['P-value'] = display_df['p_value'].apply(lambda x: f"{x:.2e}")
                                    display_df['Adj P-value'] = display_df['p_adj'].apply(lambda x: f"{x:.2e}")
                                    display_df['Enrichment'] = display_df['enrichment_score'].apply(lambda x: f"{x:.2f}")
                                    display_df['Combined Score'] = display_df['enrichment_score'].apply(lambda x: f"{x:.1f}")
                                    display_df['Genes'] = display_df['feature_count']
                                    
                                    # Clean pathway names (remove long IDs)
                                    display_df['Pathway'] = display_df['pathway_name'].apply(
                                        lambda x: x.split('(')[0].strip() if '(' in x else x[:80]
                                    )
                                    
                                    # Select columns for display
                                    cols_to_show = ['Pathway', 'P-value', 'Adj P-value', 'Genes', 
                                                  'Enrichment', 'Combined Score', 'features']
                                    available_cols = [col for col in cols_to_show if col in display_df.columns]
                                    
                                    st.dataframe(
                                        display_df[available_cols], 
                                        use_container_width=True,
                                        height=400
                                    )
                                
                                # Top pathway spotlight
                                if len(df_path) > 0:
                                    st.subheader("🏆 Top Enriched Pathway")
                                    top_pathway = df_path.iloc[0]
                                    
                                    col1, col2 = st.columns([2, 1])
                                    with col1:
                                        st.markdown(f"""
                                        **{top_pathway['pathway_name'].split('(')[0].strip()}**
                                        
                                        - **P-value:** {top_pathway['p_value']:.2e}
                                        - **Adjusted P-value:** {top_pathway['p_adj']:.2e}
                                        - **Genes involved:** {top_pathway['feature_count']} genes
                                        - **Enrichment Score:** {top_pathway['enrichment_score']:.2f}
                                        
                                        """)
                                        if 'enrichment_score' in top_pathway:
                                            st.markdown(f"- **Combined Score:** {top_pathway['enrichment_score']:.1f}")
                                    
                                    with col2:
                                        # Show genes in this pathway
                                        if top_pathway.get('features'):
                                            st.markdown("**Matching Genes:**")
                                            genes_in_pathway = top_pathway['features']
                                            if isinstance(genes_in_pathway, list):
                                                for gene in genes_in_pathway[:10]:  # Show max 10
                                                    st.markdown(f"• {gene}")
                                                if len(genes_in_pathway) > 10:
                                                    st.markdown(f"• ... and {len(genes_in_pathway) - 10} more")
                                
                                # Download pathway results
                                if len(df_path) > 0:
                                    st.subheader("💾 Download Pathway Results")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        # CSV download
                                        csv_data = df_path.to_csv(index=False)
                                        st.download_button(
                                            label="📊 Download Results (CSV)",
                                            data=csv_data,
                                            file_name=f"pathway_analysis_{pathway_database}.csv",
                                            mime="text/csv"
                                        )
                                    
                                    with col2:
                                        # JSON download with full results
                                        json_data = json.dumps(path_results, indent=2, default=str)
                                        st.download_button(
                                            label="📋 Download Full Results (JSON)",
                                            data=json_data,
                                            file_name=f"pathway_analysis_{pathway_database}.json",
                                            mime="application/json"
                                        )
                                
                            else:
                                st.info("🔍 No significantly enriched pathways found.")
                                
                                # Provide helpful suggestions
                                st.markdown("""
                                **Possible reasons:**
                                - Feature names don't match gene symbols in the database
                                - Selected features are not biologically related
                                - P-value threshold is too strict
                                - Too few features for meaningful enrichment
                                
                                **Suggestions:**
                                - Ensure features are standard gene symbols (e.g., TP53, BRCA1, MYC)
                                - Try a different pathway database
                                - Increase the p-value threshold
                                - Check if your features represent biological entities
                                """)
                                
                                # Show what was searched
                                st.markdown("**Features analyzed:**")
                                st.code(", ".join(results['best_features'][:20]))
                        
                        except ImportError:
                            st.error("❌ Real pathway analyzer not available. Please ensure pathway_analyzer.py is updated.")
                        except requests.exceptions.RequestException:
                            st.error("❌ Network error: Cannot connect to pathway databases. Please check your internet connection.")
                        except Exception as e:
                            st.error(f"❌ Pathway analysis error: {str(e)}")
                            
                            # Debug information
                            with st.expander("🐛 Debug Information"):
                                st.code(f"Error type: {type(e).__name__}")
                                st.code(f"Error message: {str(e)}")
                                st.markdown("**Troubleshooting:**")
                                st.markdown("- Check internet connection")
                                st.markdown("- Verify gene symbols are standard (HGNC approved)")
                                st.markdown("- Try a different pathway database")
                                st.markdown("- Ensure at least 3-5 genes are provided")

                # Download section
                st.subheader("💾 Download Options")
                
                download_col1, download_col2, download_col3, download_col4 = st.columns(4)
                
                with download_col1:
                    # Selected features CSV
                    features_download_df = pd.DataFrame({'selected_features': results['best_features']})
                    if enable_pubmed and 'literature_results' in locals():
                        lit_dict = {r['feature_name']: r for r in literature_results}
                        features_download_df['literature_score'] = features_download_df['selected_features'].map(
                            lambda x: lit_dict.get(x, {}).get('evidence_score', 0)
                        )
                        features_download_df['paper_count'] = features_download_df['selected_features'].map(
                            lambda x: lit_dict.get(x, {}).get('paper_count', 0)
                        )
                        features_download_df['articles_retrieved'] = features_download_df['selected_features'].map(
                            lambda x: len(lit_dict.get(x, {}).get('articles', []))
                        )
                    
                    features_csv = features_download_df.to_csv(index=False)
                    st.download_button(
                        label="📋 Download Selected Features (CSV)",
                        data=features_csv,
                        file_name="selected_features_with_literature.csv",
                        mime="text/csv"
                    )
                
                with download_col2:
                    # Full results JSON
                    download_results = {
                        'best_score': results['best_score'],
                        'best_features': results['best_features'],
                        'config': asdict(config),
                        'sense_info': sense_info,
                        'literature_results': literature_results if enable_pubmed and 'literature_results' in locals() else []
                    }
                    results_json = json.dumps(download_results, indent=2)
                    st.download_button(
                        label="📊 Download Full Results (JSON)",
                        data=results_json,
                        file_name="feature_selection_results.json",
                        mime="application/json"
                    )

                with download_col3:
                    # Full JSON report -> PDF
                    best_idx = results['history_df']['result'].apply(lambda x: x['metric_value']).idxmax()
                    best_result = results['history_df'].iloc[best_idx]
                    docs = results.get('documentation_link', ["", ""])
                    
                    full_report = {
                        "report_type": "Feature Selection Technical Report",
                        "timestamp": pd.Timestamp.now().isoformat(),
                        "final_model_summary": {
                            "selected_features": results['best_features'],
                            "performance_metrics": {
                                "target_metric": config.target_metric,
                                "metric_value": float(results['best_score']),
                                "number_of_features": len(results['best_features'])
                            },
                            "feature_selection_strategy_used": {
                                "name": best_result['plan']['strategy'],
                                "comment": best_result['plan'].get('comment', ''),
                                "selection_strategy_link1": docs[0] if len(docs) > 0 else "",
                                "selectioin_strategy_link2": docs[1] if len(docs) > 1 else "",
                            }
                        },
                        "agent_configuration": asdict(config),
                        "data_analysis_summary": results['sense_info'],
                        "literature_analysis": literature_results if 'literature_results' in locals() else "PubMed search was not enabled or results were not found.",
                        "full_trial_history": results['history_df'].to_dict(orient='records'),
                        "pathway_analysis": path_results
                    }
                    pdf_report = generate_pdf_report(full_report)

                    # Properly get PDF bytes
                    pdf_out = pdf_report.output(dest='S')
                    pdf_bytes = bytes(pdf_out)



                    st.download_button(
                        label="📥 Download Full Technical Report (PDF)",
                        data=pdf_bytes,
                        file_name='feature_selection_report.pdf',
                        mime='application/pdf'
                    )
                
                with download_col4:
                    if enable_pubmed and 'literature_results' in locals():
                        # Literature report with articles
                        lit_report = {
                            'analysis': lit_analysis,
                            'detailed_results': literature_results,
                            'search_parameters': {
                                'disease_context': disease_context,
                                'email': email,
                                'api_key_used': bool(api_key)
                            }
                        }
                        lit_report = convert_numpy_types(lit_report)
                        lit_json = json.dumps(lit_report, indent=2)
                        st.download_button(
                            label="🔬 Download Literature Report (JSON)",
                            data=lit_json,
                            file_name="literature_analysis_report.json",
                            mime="application/json"
                        )
                
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)

    else:
        # Welcome screen
        st.info("""
        👈 **To get started:**
        1. Upload your CSV from the left panel
        2. Select the target variable
        3. Enable PubMed search and enter your email 
        4. Enter API key for faster and better searches      
            i-go to https://ncbi.nlm.nih.gov and register for a free 
            account if you don't have one
                
            ii- get an API key from your "Account settings" (optional) 
                
            iii-  Without an API key you can make up to **3 requests/second**, with an API key you can make up to **10 requests/second**            
        5. Configure the settings
        6. Click the "Start Feature Selection" button
        
        📝 **Supported formats:**
        - CSV files
        - Both classification and regression tasks
        - Numeric and categorical features
        
        🔬 **PubMed Integration:**
        - Automatic literature search for selected features
        - Evidence scoring based on publication count
        - Literature-informed agent decisions
        - Detailed publication analytics
        - **NEW**: Full article listings with abstracts
        """)
        
        # Sample data option
        st.subheader("🎲 Try with Sample Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧬 Cancer Genes Dataset"):


                np.random.seed(42)
                n = 600

                # 30 kanser geni için veri oluştur
                genes = ['TP53','BRCA1','BRCA2','EGFR','HER2','MYC','KRAS','BRAF','BCL2','ALK',
                        'PIK3CA','PTEN','RB1','APC','VHL','ATM','CDKN2A','MLH1','RET','MET',
                        'CDK4','CDK6','CCND1','CDKN1A','CDKN1B','BAX','BCL2L1','CASP3','CASP8','FAS']

                data = {}
                for gene in genes:
                    cancer_vals = np.random.normal(7 if gene in ['MYC','EGFR','HER2'] else 3, 1.5, 240)
                    normal_vals = np.random.normal(5, 1, 360)
                    data[gene] = np.round(np.concatenate([cancer_vals, normal_vals]), 2)

                data['diagnosis'] = ['cancer']*240 + ['normal']*360
                df = pd.DataFrame(data).sample(frac=1, random_state=42)

                # CSV hazırla
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Cancer Genes Dataset",
                    data=csv,
                    file_name="cancer_genes_pubmed.csv",
                    mime="text/csv"
                )
                st.success("✅ Cancer genes dataset ready for download!")
        
        with col2:
            if st.button("❤️ Load Heart Disease Dataset"):
                try:
                    # Create a synthetic heart disease dataset
                    np.random.seed(42)
                    n_samples = 1000
                    
                    # Generate realistic heart disease features
                    heart_df = pd.DataFrame({
                        'age': np.random.randint(25, 80, n_samples),
                        'sex': np.random.choice([0, 1], n_samples, p=[0.32, 0.68]),  # 0=female, 1=male
                        'chest_pain_type': np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.17, 0.28, 0.08]),
                        'resting_blood_pressure': np.random.normal(131, 17, n_samples).astype(int),
                        'serum_cholesterol': np.random.normal(246, 51, n_samples).astype(int),
                        'fasting_blood_sugar': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),  # >120mg/dl
                        'resting_ecg_results': np.random.choice([0, 1, 2], n_samples, p=[0.48, 0.48, 0.04]),
                        'max_heart_rate_achieved': np.random.normal(149, 22, n_samples).astype(int),
                        'exercise_induced_angina': np.random.choice([0, 1], n_samples, p=[0.67, 0.33]),
                        'st_depression': np.random.exponential(1.0, n_samples).round(1),
                        'st_slope': np.random.choice([0, 1, 2], n_samples, p=[0.14, 0.46, 0.40]),
                        'number_of_major_vessels': np.random.choice([0, 1, 2, 3], n_samples, p=[0.54, 0.21, 0.16, 0.09]),
                        'thalassemia_type': np.random.choice([1, 2, 3], n_samples, p=[0.05, 0.18, 0.77]),
                    })
                    
                    # Ensure realistic ranges
                    heart_df['resting_blood_pressure'] = np.clip(heart_df['resting_blood_pressure'], 94, 200)
                    heart_df['serum_cholesterol'] = np.clip(heart_df['serum_cholesterol'], 126, 564)
                    heart_df['max_heart_rate_achieved'] = np.clip(heart_df['max_heart_rate_achieved'], 71, 202)
                    heart_df['st_depression'] = np.clip(heart_df['st_depression'], 0.0, 6.2)
                    
                    # Create target variable (0=no disease, 1=disease) with realistic correlations
                    # Higher risk factors increase probability of heart disease
                    risk_score = (
                        (heart_df['age'] - 25) / 55 * 0.3 +
                        heart_df['sex'] * 0.4 +  # males higher risk
                        (heart_df['chest_pain_type'] == 0) * 0.3 +  # typical angina
                        (heart_df['resting_blood_pressure'] - 94) / 106 * 0.2 +
                        (heart_df['serum_cholesterol'] - 126) / 438 * 0.15 +
                        heart_df['fasting_blood_sugar'] * 0.1 +
                        heart_df['exercise_induced_angina'] * 0.4 +
                        heart_df['st_depression'] / 6.2 * 0.3 +
                        (heart_df['number_of_major_vessels'] / 3) * 0.5 +
                        (heart_df['thalassemia_type'] == 3) * 0.2
                    )
                    
                    # Convert risk score to probability and generate target
                    prob_disease = 1 / (1 + np.exp(-(risk_score - 1.8)))  # sigmoid function
                    heart_df['target'] = np.random.binomial(1, prob_disease, n_samples)
                    
                    csv = heart_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Heart Disease Sample",
                        data=csv,
                        file_name="sample_heart_disease.csv",
                        mime="text/csv"
                    )
                    st.success("✅ Sample data ready for download!")
                except Exception as e:
                    st.error(f"Error creating sample: {e}")

        # Info about PubMed setup
        if uploaded_file is None:
            st.warning("Please upload a CSV file to begin.")
            return
        if uploaded_file is not None:
            # Load data
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            except Exception as e:
                st.sidebar.error(f"❌ File upload error: {str(e)}")
                return

        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.subheader("🔬 Setting up PubMed Search")
        st.info("""
        **To use the PubMed literature analysis feature:**
        
        1. **Get a free NCBI account**: Visit [ncbi.nlm.nih.gov](https://www.ncbi.nlm.nih.gov/) and register
        2. **Use your email**: Enter the email associated with your NCBI account
        3. **Optional API Key**: Get an API key from your NCBI account settings for faster searches (10 req/sec vs 3 req/sec)
        4. **Disease Context**: Specify a medical condition to focus the literature search (e.g., "cancer", "diabetes")
        
        **What you get:**
        - Automatic PubMed search for each selected feature
        - Evidence scores based on publication count and relevance
        - Literature-informed agent decisions
        - **NEW**: Detailed article listings with titles, authors, abstracts, and PubMed links
        - Downloadable publication analysis reports
        """)

if __name__ == "__main__":
    main()


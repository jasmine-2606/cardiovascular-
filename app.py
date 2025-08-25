import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import pickle
import io
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .section-header {
        font-size: 2rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .best-model {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def load_and_preprocess_data():
    """Load and preprocess the cardiovascular dataset"""
    try:
        # Load data
        data = pd.read_csv('cardio_train.csv', sep=';')
        
        # Display basic info about the dataset
        st.success(f"✅ Dataset loaded successfully! Shape: {data.shape}")
        
        # Drop duplicates
        initial_shape = data.shape
        data = data.drop_duplicates()
        duplicates_removed = initial_shape[0] - data.shape[0]
        
        if duplicates_removed > 0:
            st.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values with imputation
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        data[numeric_columns] = imputer.fit_transform(data[numeric_columns])
        
        # Create BMI feature
        data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)
        
        # Clip unrealistic blood pressure values
        data['ap_hi'] = np.clip(data['ap_hi'], 90, 200)
        data['ap_lo'] = np.clip(data['ap_lo'], 60, 140)
        
        return data
        
    except FileNotFoundError:
        st.error("❌ Error: 'cardio_train.csv' file not found. Please make sure the file is in the same directory as app.py")
        return None
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return None

def display_data_overview(data):
    """Display data overview and statistics"""
    st.markdown('<p class="section-header">📊 Data Overview</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Samples</h3>
            <h2>{data.shape[0]:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Features</h3>
            <h2>{data.shape[1]}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        positive_cases = data['cardio'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Positive Cases</h3>
            <h2>{positive_cases:,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        positive_rate = (positive_cases / len(data)) * 100
        st.markdown(f"""
        <div class="metric-card">
            <h3>Positive Rate</h3>
            <h2>{positive_rate:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Display sample data
    st.subheader("Sample Data")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Display data info
    st.subheader("Data Information")
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_text = buffer.getvalue()
    st.text(info_text)

def exploratory_data_analysis(data):
    """Perform exploratory data analysis with visualizations"""
    st.markdown('<p class="section-header">🔍 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    # Interactive histogram
    st.subheader("Interactive Feature Histograms")
    numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'id' in numeric_features:
        numeric_features.remove('id')
    
    selected_feature = st.selectbox("Select a feature for histogram:", numeric_features)
    
    fig_hist = px.histogram(
        data, 
        x=selected_feature, 
        nbins=30, 
        title=f'Distribution of {selected_feature}',
        color_discrete_sequence=['#3498db']
    )
    fig_hist.update_layout(height=400)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Box plots of features vs target
    st.subheader("Box Plots: Features vs Target (Cardio)")
    
    col1, col2 = st.columns(2)
    
    # Select features for box plots (excluding id and target)
    box_features = [col for col in numeric_features if col != 'cardio']
    selected_box_features = st.multiselect(
        "Select features for box plots:", 
        box_features, 
        default=box_features[:4] if len(box_features) >= 4 else box_features
    )
    
    if selected_box_features:
        n_features = len(selected_box_features)
        cols_per_row = 2
        n_rows = (n_features + cols_per_row - 1) // cols_per_row
        
        for i in range(0, n_features, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < n_features:
                    feature = selected_box_features[i + j]
                    with col:
                        fig_box = px.box(
                            data, 
                            x='cardio', 
                            y=feature, 
                            title=f'{feature} by Cardio Status',
                            color='cardio',
                            color_discrete_sequence=['#e74c3c', '#27ae60']
                        )
                        fig_box.update_layout(height=350)
                        st.plotly_chart(fig_box, use_container_width=True)
    
    # Correlation matrix heatmap
    st.subheader("Correlation Matrix Heatmap")
    
    # Calculate correlation matrix for numeric features
    corr_features = [col for col in numeric_features if col in data.columns]
    corr_matrix = data[corr_features].corr()
    
    # Create heatmap
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Numeric Features",
        color_continuous_scale='RdBu_r'
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

def train_models(data):
    """Train and evaluate multiple machine learning models"""
    st.markdown('<p class="section-header">🤖 Model Training & Evaluation</p>', unsafe_allow_html=True)
    
    # Prepare features and target
    # Remove non-predictive features
    feature_columns = data.columns.tolist()
    if 'id' in feature_columns:
        feature_columns.remove('id')
    if 'cardio' in feature_columns:
        feature_columns.remove('cardio')
    
    X = data[feature_columns]
    y = data['cardio']
    
    # Test size configuration
    test_size = st.slider("Select test set size:", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state:", value=42, min_value=1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    st.info(f"Training set size: {X_train.shape[0]} samples")
    st.info(f"Test set size: {X_test.shape[0]} samples")
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'Support Vector Machine (RBF)': SVC(kernel='rbf', random_state=random_state),
        'K-Nearest Neighbors (k=5)': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(random_state=random_state, n_estimators=100)
    }
    
    # Train models and collect results
    results = {}
    trained_models = {}
    
    progress_bar = st.progress(0)
    
    for i, (name, model) in enumerate(models.items()):
        with st.spinner(f"Training {name}..."):
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'model': model
            }
            trained_models[name] = model
            
        progress_bar.progress((i + 1) / len(models))
    
    # Display model accuracies
    st.subheader("Model Performance Comparison")
    
    # Create accuracy comparison dataframe
    accuracy_df = pd.DataFrame([
        {'Model': name, 'Accuracy': results[name]['accuracy']} 
        for name in results.keys()
    ]).sort_values('Accuracy', ascending=False)
    
    # Display accuracy table
    st.dataframe(accuracy_df.style.format({'Accuracy': '{:.4f}'}), use_container_width=True)
    
    # Create accuracy bar chart
    fig_acc = px.bar(
        accuracy_df, 
        x='Model', 
        y='Accuracy',
        title='Model Accuracy Comparison',
        color='Accuracy',
        color_continuous_scale='Viridis'
    )
    fig_acc.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Find best model
    best_model_name = accuracy_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    best_accuracy = accuracy_df.iloc[0]['Accuracy']
    best_predictions = results[best_model_name]['predictions']
    
    # Highlight best model
    st.markdown(f"""
    <div class="best-model">
        <h3>🏆 Best Model: {best_model_name}</h3>
        <p><strong>Accuracy: {best_accuracy:.4f}</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display detailed evaluation for best model
    st.subheader(f"Detailed Evaluation - {best_model_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, best_predictions)
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix",
            labels=dict(x="Predicted", y="Actual"),
            x=['No Disease', 'Disease'],
            y=['No Disease', 'Disease'],
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        # Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, best_predictions, output_dict=True)
        
        # Convert to DataFrame for better display
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format('{:.3f}'), use_container_width=True)
    
    # Feature Importances (if available)
    if hasattr(best_model, 'feature_importances_'):
        st.subheader("Feature Importances")
        
        importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_imp = px.bar(
            importance_df.head(10), 
            x='Importance', 
            y='Feature',
            orientation='h',
            title='Top 10 Feature Importances',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig_imp.update_layout(height=400)
        st.plotly_chart(fig_imp, use_container_width=True)
    
    elif hasattr(best_model, 'coef_'):
        st.subheader("Feature Coefficients")
        
        coef_df = pd.DataFrame({
            'Feature': feature_columns,
            'Coefficient': best_model.coef_[0]
        }).sort_values('Coefficient', key=abs, ascending=False)
        
        fig_coef = px.bar(
            coef_df.head(10), 
            x='Coefficient', 
            y='Feature',
            orientation='h',
            title='Top 10 Feature Coefficients (by absolute value)',
            color='Coefficient',
            color_continuous_scale='RdBu'
        )
        fig_coef.update_layout(height=400)
        st.plotly_chart(fig_coef, use_container_width=True)
    
    return best_model, best_model_name, scaler, feature_columns

def export_model(model, model_name, scaler, feature_columns):
    """Allow users to download the best trained model"""
    st.markdown('<p class="section-header">💾 Export Best Model</p>', unsafe_allow_html=True)
    
    # Create model package
    model_package = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'model_name': model_name
    }
    
    # Serialize model
    model_bytes = pickle.dumps(model_package)
    
    # Download button
    st.download_button(
        label="📥 Download Best Model (.pkl)",
        data=model_bytes,
        file_name=f"best_cardio_model_{model_name.lower().replace(' ', '_')}.pkl",
        mime="application/octet-stream",
        help="Download the trained model with preprocessing pipeline"
    )
    
    st.info("💡 The downloaded file contains the trained model, scaler, feature columns, and model name for future predictions.")

def main():
    """Main application function"""
    
    # Header
    st.markdown('<p class="main-header">❤️ Cardiovascular Disease Prediction</p>', unsafe_allow_html=True)
    
    st.markdown("""
    This application performs comprehensive analysis and prediction of cardiovascular disease using machine learning models.
    The app includes data preprocessing, exploratory data analysis, model training, and evaluation.
    """)
    
    # Sidebar
    st.sidebar.header("Navigation")
    sections = [
        "📊 Data Overview",
        "🔍 Exploratory Data Analysis", 
        "🤖 Model Training",
        "💾 Export Model"
    ]
    selected_section = st.sidebar.selectbox("Select Section:", sections)
    
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    if data is not None:
        # Data Overview
        if selected_section == "📊 Data Overview":
            display_data_overview(data)
        
        # EDA
        elif selected_section == "🔍 Exploratory Data Analysis":
            exploratory_data_analysis(data)
        
        # Model Training
        elif selected_section == "🤖 Model Training":
            best_model, best_model_name, scaler, feature_columns = train_models(data)
            
            # Store in session state for export
            st.session_state.best_model = best_model
            st.session_state.best_model_name = best_model_name
            st.session_state.scaler = scaler
            st.session_state.feature_columns = feature_columns
        
        # Export Model
        elif selected_section == "💾 Export Model":
            if hasattr(st.session_state, 'best_model'):
                export_model(
                    st.session_state.best_model,
                    st.session_state.best_model_name,
                    st.session_state.scaler,
                    st.session_state.feature_columns
                )
            else:
                st.warning("⚠️ Please train models first in the 'Model Training' section before exporting.")
    
    # Footer
    st.markdown("---")
    st.markdown("**Built with Streamlit** | Cardiovascular Disease Prediction System")

if __name__ == "__main__":
    main()

import streamlit as st
st.set_page_config(page_title="Stroke Prediction Dashboard", layout="wide", initial_sidebar_state="expanded")

import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score

# CSS tùy chỉnh
st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #e0f7fa, #80deea);
    }
    h1, h2, h3, h4 {
        color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hàm xử lý dữ liệu
@st.cache_data
def preprocess_data(df):
    df = df.drop(['id', 'age_number'], axis=1, errors='ignore')
    df.drop(df[df['gender'] == 'Other'].index, inplace=True)
    cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    df = pd.get_dummies(df, columns=cols)
    df['bmi'] = df['bmi'].interpolate(method='linear')
    return df

# Hàm tải tài nguyên
@st.cache_resource
def load_model(file_name):
    try:
        return joblib.load(file_name)
    except Exception:
        return None

@st.cache_resource
def load_scaler():
    try:
        return joblib.load(r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\scaler.pkl")
    except Exception:
        return None

@st.cache_data
def load_dataset(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# Khởi tạo Session State
if 'loaded' not in st.session_state:
    st.session_state.loaded = False
    st.session_state.trained_models = {}
    st.session_state.scaler = None
    st.session_state.raw_df = None
    st.session_state.processed_df = None
    st.session_state.model_columns = None
    st.session_state.X_train_scaled = None
    st.session_state.X_test_scaled = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.detailed_results_train = {}
    st.session_state.detailed_results_test = {}
    st.session_state.cv_results = {}

# Sidebar thông tin
st.sidebar.header("🔹 Dashboard Information")
st.sidebar.info("Giao diện dự đoán nguy cơ đột quỵ và phân tích dữ liệu từ dự án.")

# Tải mô hình và dữ liệu chỉ một lần
if not st.session_state.loaded:
    model_files = {
        "KNN": r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\knn.pkl",
        "LogisticRegression": r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\logistic.pkl",
        "DecisionTree": r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\decision.pkl",
        "RandomForest": r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\randomforest.pkl",
        "SVM": r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\svc.pkl",
        "LGBM": r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\lgbm.pkl",
        "XGB": r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\xgboost.pkl",
        "GradientBoosting": r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\gradient.pkl",
        "CatBoost": r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\catboost.pkl",
        "ExtraTrees": r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\extratree.pkl",
        "Stacking": r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\stacking.pkl"
    }

    for model_name, file_name in model_files.items():
        model = load_model(file_name)
        st.session_state.trained_models[model_name] = model
        if model is not None:
            st.sidebar.success(f"✅ {model_name} model load thành công.")
        else:
            st.sidebar.error(f"❌ Lỗi khi load {model_name} model.")

    st.session_state.scaler = load_scaler()
    if st.session_state.scaler is not None:
        st.sidebar.success("✅ Scaler load thành công.")
    else:
        st.sidebar.error("❌ Lỗi khi load scaler.")

    dataset_path = r"D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Data\healthcare-dataset-stroke-data.csv"
    st.session_state.raw_df = load_dataset(dataset_path)
    if st.session_state.raw_df is not None:
        st.sidebar.success("✅ Dataset load thành công.")
        st.session_state.processed_df = preprocess_data(st.session_state.raw_df.copy())
    else:
        st.sidebar.error("❌ Lỗi khi load dataset.")

    if st.session_state.scaler is not None and hasattr(st.session_state.scaler, "feature_names_in_"):
        st.session_state.model_columns = list(st.session_state.scaler.feature_names_in_)
    else:
        st.session_state.model_columns = [
            'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
            'gender_Female', 'gender_Male', 'ever_married_No', 'ever_married_Yes',
            'work_type_Govt_job', 'work_type_Never_worked', 'work_type_Private',
            'work_type_Self-employed', 'work_type_children', 'Residence_type_Rural',
            'Residence_type_Urban', 'smoking_status_Unknown',
            'smoking_status_formerly smoked', 'smoking_status_never smoked',
            'smoking_status_smokes'
        ]

    st.session_state.loaded = True

# Hàm chọn màu cho gauge
def get_gauge_color(prob):
    if prob * 100 < 30:
        return "#2ecc71"
    elif prob * 100 < 70:
        return "#f1c40f"
    else:
        return "#e74c3c"

# Mapping màu cho mô hình
model_colors = {
    "KNN": "#3366CC", "LogisticRegression": "#DC3912", "DecisionTree": "#FF9900",
    "RandomForest": "#109618", "SVM": "#990099", "LGBM": "#0099C6",
    "XGB": "#DD4477", "GradientBoosting": "#66AA00", "CatBoost": "#B82E2E",
    "ExtraTrees": "#FF6600", "Stacking": "#AAAA11"
}

st.title("🩺 Stroke Prediction Dashboard")
tabs = st.tabs(["🩺 Stroke Prediction", "📊 EDA & Feature Relationships", "📈 Model Evaluation"])

with tabs[0]:
    st.header("🔎 Stroke Prediction")
    col_input, col_summary = st.columns(2)
    with col_input:
        st.subheader("📥 Nhập dữ liệu bệnh nhân")
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        gender = st.selectbox("Gender", options=["Male", "Female"])
        hypertension = st.selectbox("Hypertension", options=["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", options=["No", "Yes"])
        ever_married = st.selectbox("Ever Married", options=["No", "Yes"])
        work_type = st.selectbox("Work Type", options=["Govt_job", "Never_worked", "Private", "Self-employed", "children"])
        residence_type = st.selectbox("Residence Type", options=["Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, value=100.0, step=0.1)
        bmi = st.number_input("BMI", min_value=0.0, value=25.0, step=0.1)
        smoking_status = st.selectbox("Smoking Status", options=["never smoked", "formerly smoked", "smokes", "Unknown"])
        
        model_options = [name for name, model in st.session_state.trained_models.items() if model is not None]
        selected_model = st.selectbox("Chọn mô hình dự đoán", model_options, index=model_options.index("CatBoost") if "CatBoost" in model_options else 0)
    
    with col_summary:
        st.subheader("📋 Tóm tắt dữ liệu đầu vào")
        st.write("Age:", age)
        st.write("Gender:", gender)
        st.write("Hypertension:", hypertension)
        st.write("Heart Disease:", heart_disease)
        st.write("Ever Married:", ever_married)
        st.write("Work Type:", work_type)
        st.write("Residence Type:", residence_type)
        st.write("Average Glucose Level:", avg_glucose_level)
        st.write("BMI:", bmi)
        st.write("Smoking Status:", smoking_status)
    
    if st.button("🔍 Predict Stroke Risk"):
        input_data = {col: 0 for col in st.session_state.model_columns}
        input_data['age'] = age
        input_data['hypertension'] = 1 if hypertension == "Yes" else 0
        input_data['heart_disease'] = 1 if heart_disease == "Yes" else 0
        input_data['avg_glucose_level'] = avg_glucose_level
        input_data['bmi'] = bmi
        input_data[f'gender_{gender}'] = 1
        input_data[f'ever_married_{ever_married}'] = 1
        input_data[f'work_type_{work_type}'] = 1
        input_data[f'Residence_type_{residence_type}'] = 1
        input_data[f'smoking_status_{smoking_status}'] = 1

        input_df = pd.DataFrame([input_data])
        if st.session_state.scaler is not None and hasattr(st.session_state.scaler, "feature_names_in_"):
            input_df = input_df.reindex(columns=st.session_state.scaler.feature_names_in_, fill_value=0)
        input_scaled = st.session_state.scaler.transform(input_df) if st.session_state.scaler else input_df.values

        model = st.session_state.trained_models[selected_model]
        if model is not None:
            prediction = model.predict(input_scaled)[0]
            proba = model.predict_proba(input_scaled)[0, 1] if hasattr(model, "predict_proba") else None
            pred_text = "🛑 High Stroke Risk" if prediction == 1 else "✅ Low Stroke Risk"
            st.subheader("📊 Kết quả dự đoán")
            st.write("🔍 Dự đoán:", pred_text)
            if proba is not None:
                st.write("📈 Xác suất dự đoán (Stroke):", round(proba * 100, 2), "%")
            gauge_color = get_gauge_color(proba) if proba is not None else "#7f8c8d"
            fig_prob = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100 if proba is not None else 0,
                title={"text": "Stroke Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "bgcolor": "#ecf0f1",
                    "borderwidth": 2,
                    "bordercolor": "#bdc3c7"
                }
            ))
            st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.error("⚠️ Model dự đoán không khả dụng.")

with tabs[1]:
    st.header("Exploratory Data Analysis (EDA)")
    if st.session_state.raw_df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.raw_df.head(10), use_container_width=True)
        
        numeric_cols = st.session_state.raw_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'id' in numeric_cols:
            numeric_cols.remove('id')
        if 'stroke' in numeric_cols:
            numeric_cols.remove('stroke')
        
        st.subheader("Data Distribution")
        chosen_col = st.selectbox("Chọn cột số để xem phân bố", numeric_cols, key="dist_col")
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            fig_hist = px.histogram(st.session_state.raw_df, x=chosen_col, nbins=30,
                                    title=f"Histogram of {chosen_col}",
                                    color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_hist.update_traces(opacity=0.75)
            st.plotly_chart(fig_hist, use_container_width=True)
        with col_dist2:
            fig_box = px.box(st.session_state.raw_df, y=chosen_col, color="stroke" if 'stroke' in st.session_state.raw_df.columns else None,
                             title=f"Boxplot of {chosen_col}",
                             color_discrete_sequence=px.colors.qualitative.D3)
            st.plotly_chart(fig_box, use_container_width=True)
        
        st.subheader("Correlation Heatmap")
        corr = st.session_state.raw_df.corr()
        fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                             title="Correlation Heatmap", height=800)
        fig_corr.update_layout(coloraxis_colorbar=dict(title="Correlation"))
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("Scatter Matrix")
        subset_cols = numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
        fig_scatter = px.scatter_matrix(st.session_state.raw_df, dimensions=subset_cols, color="stroke" if 'stroke' in st.session_state.raw_df.columns else None,
                                        title="Scatter Matrix of Numeric Features",
                                        color_discrete_sequence=px.colors.qualitative.Safe,
                                        height=800)
        fig_scatter.update_traces(diagonal_visible=False, showupperhalf=False)
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.error("Dataset không khả dụng.")

with tabs[2]:
    st.header("Model Evaluation")
    static_results = {
        "KNN": {"F1 Score": 0.089, "Accuracy": 0.933, "Precision": 0.217, "Recall": 0.056},
        "LogisticRegression": {"F1 Score": 0.082, "Accuracy": 0.942, "Precision": 0.5, "Recall": 0.045},
        "DecisionTree": {"F1 Score": 0.142, "Accuracy": 0.929, "Precision": 0.237, "Recall": 0.101},
        "RandomForest": {"F1 Score": 0.021, "Accuracy": 0.938, "Precision": 0.125, "Recall": 0.011},
        "SVM": {"F1 Score": 0.037, "Accuracy": 0.933, "Precision": 0.111, "Recall": 0.022},
        "LGBM": {"F1 Score": 0.038, "Accuracy": 0.933, "Precision": 0.118, "Recall": 0.022},
        "XGB": {"F1 Score": 0.106, "Accuracy": 0.934, "Precision": 0.25, "Recall": 0.067},
        "GradientBoosting": {"F1 Score": 0.099, "Accuracy": 0.941, "Precision": 0.417, "Recall": 0.056},
        "CatBoost": {"F1 Score": 0.04, "Accuracy": 0.938, "Precision": 0.2, "Recall": 0.022},
        "ExtraTrees": {"F1 Score": 0.042, "Accuracy": 0.941, "Precision": 0.333, "Recall": 0.022},
        "Stacking": {"F1 Score": 0.02, "Accuracy": 0.94, "Precision": 0.1, "Recall": 0.011}
    }
    categories = ["F1 Score", "Accuracy", "Precision", "Recall"]
    radar_data_static = []
    for model_name, metrics in static_results.items():
        radar_data_static.append(go.Scatterpolar(
            r=[metrics[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name=model_name,
            line=dict(color=model_colors.get(model_name, "#7f8c8d"))
        ))
    fig_static = go.Figure(data=radar_data_static)
    fig_static.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Biểu đồ Radar: So sánh hiệu suất các mô hình"
    )
    st.plotly_chart(fig_static, use_container_width=True)
    st.markdown("---")

    if st.session_state.processed_df is not None and st.session_state.scaler is not None:
        if st.session_state.X_train_scaled is None:
            try:
                X_train = joblib.load(r'D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\X_train.pkl')
                X_test = joblib.load(r'D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\X_test.pkl')
                st.session_state.y_train = joblib.load(r'D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\y_train.pkl')
                st.session_state.y_test = joblib.load(r'D:\FPT University\Code_FPT\Season 4\DAP391m\Project kiệt\Project\Stroke\Model\y_test.pkl')
                st.session_state.X_train_scaled = st.session_state.scaler.transform(X_train)
                st.session_state.X_test_scaled = st.session_state.scaler.transform(X_test)
                
                for model_name, model in st.session_state.trained_models.items():
                    if model is not None:
                        cv_scores = cross_val_score(model, st.session_state.X_train_scaled, st.session_state.y_train, cv=5, scoring='accuracy', n_jobs=-1)
                        st.session_state.cv_results[model_name] = {
                            "CV Mean Accuracy": cv_scores.mean(),
                            "CV Std Dev": cv_scores.std()
                        }
                        y_pred_train = model.predict(st.session_state.X_train_scaled)
                        y_prob_train = model.predict_proba(st.session_state.X_train_scaled) if hasattr(model, "predict_proba") else None
                        st.session_state.detailed_results_train[model_name] = {
                            "F1 Score": f1_score(st.session_state.y_train, y_pred_train, zero_division=0),
                            "Accuracy": accuracy_score(st.session_state.y_train, y_pred_train),
                            "Precision": precision_score(st.session_state.y_train, y_pred_train, zero_division=0),
                            "Recall": recall_score(st.session_state.y_train, y_pred_train, zero_division=0),
                            "y_pred": y_pred_train,
                            "y_prob": y_prob_train,
                            "y_true": st.session_state.y_train
                        }
                        y_pred_test = model.predict(st.session_state.X_test_scaled)
                        y_prob_test = model.predict_proba(st.session_state.X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
                        st.session_state.detailed_results_test[model_name] = {
                            "F1 Score": f1_score(st.session_state.y_test, y_pred_test, zero_division=0),
                            "Accuracy": accuracy_score(st.session_state.y_test, y_pred_test),
                            "Precision": precision_score(st.session_state.y_test, y_pred_test, zero_division=0),
                            "Recall": recall_score(st.session_state.y_test, y_pred_test, zero_division=0),
                            "y_pred": y_pred_test,
                            "y_prob": y_prob_test,
                            "y_true": st.session_state.y_test
                        }
            except Exception as e:
                st.error("Lỗi khi load tập dữ liệu: " + str(e))

        selected_model = st.selectbox("Chọn mô hình để xem chi tiết đánh giá", options=list(st.session_state.detailed_results_test.keys()))
        st.subheader(f"Đánh giá chi tiết cho mô hình: {selected_model}")
        metrics_train = st.session_state.detailed_results_train[selected_model]
        metrics_test = st.session_state.detailed_results_test[selected_model]
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Tập Train")
            st.write({k: round(metrics_train[k], 3) for k in ["F1 Score", "Accuracy", "Precision", "Recall"]})
        with col2:
            st.markdown("### Tập Test")
            st.write({k: round(metrics_test[k], 3) for k in ["F1 Score", "Accuracy", "Precision", "Recall"]})
        
        st.subheader("Cross-Validation Results")
        cv_metrics = st.session_state.cv_results[selected_model]
        st.write({
            "CV Mean Accuracy": round(cv_metrics["CV Mean Accuracy"], 3),
            "CV Std Deviation": round(cv_metrics["CV Std Dev"], 3)
        })
        
        st.subheader("So sánh Cross-Validation Accuracy giữa các mô hình")
        cv_df = pd.DataFrame({
            "Model": list(st.session_state.cv_results.keys()),
            "CV Accuracy": [st.session_state.cv_results[m]["CV Mean Accuracy"] for m in st.session_state.cv_results],
            "CV Std Dev": [st.session_state.cv_results[m]["CV Std Dev"] for m in st.session_state.cv_results]
        })
        fig_cv = px.bar(cv_df, x="Model", y="CV Accuracy", error_y="CV Std Dev",
                        title="Cross-Validation Accuracy per Model",
                        color="Model", height=400,
                        color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig_cv, use_container_width=True)
        
        st.subheader("Biểu đồ Radar: Hiệu suất trên Tập Test")
        fig_radar_detail = go.Figure(data=[
            go.Scatterpolar(
                r=[metrics_test[cat] for cat in categories],
                theta=categories,
                fill='toself',
                name=selected_model,
                line=dict(color=model_colors.get(selected_model, "#7f8c8d"))
            )
        ])
        fig_radar_detail.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True
        )
        st.plotly_chart(fig_radar_detail, use_container_width=True)
        
        st.subheader("Additional Evaluation Plots (Tập Test)")
        y_pred = metrics_test["y_pred"]
        y_prob = metrics_test["y_prob"]
        col_eval1, col_eval2 = st.columns(2)
        with col_eval1:
            st.markdown("**Confusion Matrix**")
            cm = confusion_matrix(metrics_test["y_true"], y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                               labels={"x": "Predicted", "y": "Actual"},
                               title="Confusion Matrix (Test)")
            st.plotly_chart(fig_cm, use_container_width=True)
        with col_eval2:
            if y_prob is not None:
                st.markdown("**ROC Curve**")
                fpr_vals, tpr_vals, _ = roc_curve(metrics_test["y_true"], y_prob)
                roc_auc_val = auc(fpr_vals, tpr_vals)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr_vals, y=tpr_vals, mode='lines',
                                             name=f"ROC (AUC={roc_auc_val:.2f})",
                                             line=dict(width=2, color=model_colors.get(selected_model, "#FF6600"))))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                             name="Chance", line=dict(dash='dash', color="#7f8c8d")))
                fig_roc.update_layout(
                    title="ROC Curve (Test)",
                    xaxis_title="False Positive Rate",
                    yaxis_title="True Positive Rate",
                    xaxis=dict(range=[0, 1]),
                    yaxis=dict(range=[0, 1.05])
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            else:
                st.info("Model không hỗ trợ dự đoán xác suất để vẽ ROC curve.")
        
        if y_prob is not None:
            st.subheader("Precision-Recall Curve (Test)")
            precision_vals, recall_vals, _ = precision_recall_curve(metrics_test["y_true"], y_prob)
            fig_pr = go.Figure()
            fig_pr.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines',
                                        name="Precision-Recall",
                                        line=dict(width=2, color=model_colors.get(selected_model, "#FF6600"))))
            fig_pr.update_layout(
                title="Precision-Recall Curve (Test)",
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1])
            )
            st.plotly_chart(fig_pr, use_container_width=True)
    else:
        st.error("Dataset hoặc scaler không khả dụng để đánh giá mô hình.")
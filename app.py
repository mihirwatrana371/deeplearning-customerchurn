import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
import matplotlib.pyplot as plt
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="Deep Learning Churn Prediction",
    page_icon="🤖",
    layout="wide"
)

# --- Column Name Mapping ---
# A dictionary to map cryptic column names to user-friendly labels
COLUMN_DESCRIPTIONS = {
    'ses_rec': 'Days Since Last Session',
    'ses_rec_avg': 'Avg. Days Between Sessions',
    'ses_rec_sd': 'Std. Dev. of Days Between Sessions',
    'ses_rec_cv': 'Coefficient of Variation in Session Recency',
    'user_rec': 'Days Since First Visit (User Recency)',
    'ses_n': 'Total Number of Sessions',
    'ses_n_r': 'Session Rate (Sessions per Day)',
    'int_n': 'Total Number of Interactions (Clicks/Views)',
    'int_n_r': 'Interaction Rate (Interactions per Session)',
    'tran_n': 'Total Number of Transactions',
    'tran_n_r': 'Transaction Rate (Transactions per Session)',
    'rev_sum': 'Total Revenue from Customer',
    'rev_sum_r': 'Revenue Rate (Revenue per Session)',
    'major_spend_r': 'Ratio of Major Spend to Total Spend',
    'int_cat_n_avg': 'Avg. Categories Interacted With per Session',
    'int_itm_n_avg': 'Avg. Items Interacted With per Session',
    'ses_mo_avg': 'Avg. Month of Session (1-12)',
    'ses_mo_sd': 'Std. Dev. of Session Month',
    'ses_ho_avg': 'Avg. Hour of Session (0-23)',
    'ses_ho_sd': 'Std. Dev. of Session Hour',
    'ses_wknd_r': 'Ratio of Weekend Sessions',
    'ses_len_avg': 'Avg. Session Length (in minutes/seconds)',
    'time_to_int': 'Time from Session Start to First Interaction',
    'time_to_tran': 'Time from Session Start to First Transaction',
    'int_cat1_n': 'Interactions with Category 1',
    'int_cat2_n': 'Interactions with Category 2',
    'int_cat3_n': 'Interactions with Category 3',
    'int_cat4_n': 'Interactions with Category 4',
    'int_cat5_n': 'Interactions with Category 5',
    'int_cat6_n': 'Interactions with Category 6',
    'int_cat7_n': 'Interactions with Category 7',
    'int_cat8_n': 'Interactions with Category 8',
    'int_cat9_n': 'Interactions with Category 9',
    'int_cat10_n': 'Interactions with Category 10',
    'int_cat11_n': 'Interactions with Category 11',
    'int_cat12_n': 'Interactions with Category 12',
    'int_cat13_n': 'Interactions with Category 13',
    'int_cat15_n': 'Interactions with Category 15',
    'int_cat16_n': 'Interactions with Category 16',
    'int_cat17_n': 'Interactions with Category 17',
    'int_cat18_n': 'Interactions with Category 18',
    'int_cat19_n': 'Interactions with Category 19',
    'int_cat20_n': 'Interactions with Category 20',
    'int_cat21_n': 'Interactions with Category 21',
    'int_cat22_n': 'Interactions with Category 22',
    'int_cat23_n': 'Interactions with Category 23',
    'int_cat24_n': 'Interactions with Category 24',
}


# --- Helper Functions ---

@st.cache_data
def load_data(file_path):
    """Loads the churn data from a CSV file."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please make sure it's in the same directory as the script.")
        return None

def preprocess_data(df):
    """
    Performs data preprocessing:
    1. Drops the visitor ID.
    2. Fills missing numerical values with the median.
    3. Identifies features (X) and target (y).
    4. Splits data into training and testing sets.
    5. Scales numerical features.
    """
    if 'visitorid' in df.columns:
        df = df.drop('visitorid', axis=1)

    # Separate features and target
    X = df.drop('target_class', axis=1)
    y = df['target_class']

    # Use SimpleImputer to fill NaN values
    # Using median is robust to outliers
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    # Split data before scaling
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def build_model(input_shape):
    """Builds, compiles, and returns the Keras Sequential model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_history(history):
    """Plots training & validation accuracy and loss."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend(loc='lower right')
    
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend(loc='upper right')
    
    st.pyplot(fig)

# --- Streamlit App UI ---

st.title("🤖 Customer Churn Prediction with Deep Learning")

st.markdown("""
Welcome to your Deep Learning minor project! This Streamlit app demonstrates a full workflow:
1.  **Load Data**: We'll load an `ecom-user-churn-data.csv` file which has intentional errors.
2.  **Preprocess Data**: We'll clean the data by handling missing values and scaling features.
3.  **Train a Neural Network**: We'll build and train a deep learning model with TensorFlow/Keras.
4.  **Evaluate & Predict**: We'll check the model's performance and use it to make new predictions.
""")

# --- 1. Load and Display Data ---
st.header("1. Data Loading and Exploration")
data_path = 'ecom-user-churn-data.csv'
raw_df = load_data(data_path)

if raw_df is not None:
    st.subheader("Uncleaned Raw Data")
    st.dataframe(raw_df.head())

    st.subheader("Data Info")
    buffer = io.StringIO()
    raw_df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # --- 2. Preprocessing ---
    st.header("2. Preprocessing")
    st.write("Before feeding the data to our neural network, we must clean it.")
    
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(raw_df.copy())
    
    st.subheader("Data Shape After Preprocessing")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Training Features Shape", str(X_train.shape))
    with col2:
        st.metric("Testing Features Shape", str(X_test.shape))

    # --- 3. Model Training ---
    st.header("3. Neural Network Training")
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'history' not in st.session_state:
        st.session_state.history = None
    if 'evaluation' not in st.session_state:
        st.session_state.evaluation = None

    st.subheader("Model Architecture")
    model_placeholder = st.empty()
    temp_model = build_model(X_train.shape[1])
    stringlist = []
    temp_model.summary(print_fn=lambda x: stringlist.append(x))
    model_summary = "\n".join(stringlist)
    model_placeholder.code(model_summary, language='text')
    
    st.subheader("Training Configuration")
    epochs = st.slider("Select number of epochs", 5, 100, 20)
    batch_size = st.select_slider("Select batch size", options=[16, 32, 64, 128], value=32)

    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training in progress..."):
            model = build_model(X_train.shape[1])
            history = model.fit(X_train, y_train,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(X_test, y_test),
                                verbose=0)
            
            st.session_state.model = model
            st.session_state.history = history
            st.session_state.evaluation = model.evaluate(X_test, y_test, verbose=0)

        st.success("✅ Model training complete!")

    # --- 4. Evaluation and Prediction ---
    if st.session_state.history:
        st.header("4. Model Evaluation")
        st.subheader("Training Performance")
        plot_history(st.session_state.history)
        
        st.subheader("Test Set Performance")
        eval_loss, eval_accuracy = st.session_state.evaluation
        col1, col2 = st.columns(2)
        col1.metric("Test Loss", f"{eval_loss:.4f}")
        col2.metric("Test Accuracy", f"{eval_accuracy:.4f}")

    if st.session_state.model:
        st.header("5. Make a Prediction")
        st.write("Adjust the sliders below to predict churn for a new customer.")
        
        input_data = {}
        cols = st.columns(4)
        
        for i, feature in enumerate(feature_names):
            label_text = COLUMN_DESCRIPTIONS.get(feature, feature)
            
            min_val = float(raw_df[feature].min())
            max_val = float(raw_df[feature].max())
            mean_val = float(raw_df[feature].median())
            
            if min_val == max_val:
                max_val = min_val + 1.0

            input_data[feature] = cols[i % 4].slider(
                label=label_text,
                min_value=min_val, 
                max_value=max_val, 
                value=mean_val,
                key=f"slider_{feature}"
            )
        
        if st.button("🔮 Predict Churn"):
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            prediction_proba = st.session_state.model.predict(input_scaled)[0][0]
            prediction_class = (prediction_proba > 0.5).astype(int)
            
            # --- UPDATE IS HERE ---
            # Added more descriptive messages based on the prediction outcome.
            if prediction_class == 1:
                st.error(f"Prediction: **Churn** (Probability: {prediction_proba:.2f})", icon="🔥")
                st.warning("This customer is highly likely to churn. Consider taking action to retain them, such as offering a discount or personalized support.")
            else:
                st.success(f"Prediction: **Not Churn** (Probability: {prediction_proba:.2f})", icon="✅")
                st.info("This customer is likely to stay. No immediate action is required.")

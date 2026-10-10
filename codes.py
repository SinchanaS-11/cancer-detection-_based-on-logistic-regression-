import gradio as gr
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



# 1. LOAD DATASET

data = load_breast_cancer()

X = data.data
y = data.target

# 0 = malignant
# 1 = benign



# 2. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# 3. FEATURE SCALING


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# 4. TRAIN MODEL


model = LogisticRegression(max_iter=1000)

model.fit(X_train_scaled, y_train)


# 5. PREDICTION FUNCTION

def predict_cancer(
    mean_radius,
    mean_texture,
    mean_perimeter,
    mean_area,
    mean_smoothness,
    mean_compactness,
    mean_concavity,
    mean_concave_points,
    mean_symmetry,
    mean_fractal_dimension,

    radius_error,
    texture_error,
    perimeter_error,
    area_error,
    smoothness_error,
    compactness_error,
    concavity_error,
    concave_points_error,
    symmetry_error,
    fractal_dimension_error,

    worst_radius,
    worst_texture,
    worst_perimeter,
    worst_area,
    worst_smoothness,
    worst_compactness,
    worst_concavity,
    worst_concave_points,
    worst_symmetry,
    worst_fractal_dimension
):

    # Put input values into an array
    features = np.array([[
        mean_radius,
        mean_texture,
        mean_perimeter,
        mean_area,
        mean_smoothness,
        mean_compactness,
        mean_concavity,
        mean_concave_points,
        mean_symmetry,
        mean_fractal_dimension,

        radius_error,
        texture_error,
        perimeter_error,
        area_error,
        smoothness_error,
        compactness_error,
        concavity_error,
        concave_points_error,
        symmetry_error,
        fractal_dimension_error,

        worst_radius,
        worst_texture,
        worst_perimeter,
        worst_area,
        worst_smoothness,
        worst_compactness,
        worst_concavity,
        worst_concave_points,
        worst_symmetry,
        worst_fractal_dimension
    ]])

    # Scale the input
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)[0]

    # Probability
    probabilities = model.predict_proba(features_scaled)[0]

    malignant_probability = probabilities[0]
    benign_probability = probabilities[1]

    if prediction == 0:
        result = "⚠️ Malignant"
        message = (
            f"Prediction: MALIGNANT\n\n"
            f"Malignant probability: {malignant_probability * 100:.2f}%\n"
            f"Benign probability: {benign_probability * 100:.2f}%"
        )

    else:
        result = "✅ Benign"
        message = (
            f"Prediction: BENIGN\n\n"
            f"Malignant probability: {malignant_probability * 100:.2f}%\n"
            f"Benign probability: {benign_probability * 100:.2f}%"
        )

    return result, message



# 6. GRADIO UI

with gr.Blocks(title="Breast Cancer Detection") as demo:

    gr.Markdown(
        """
        # 🩺 Breast Cancer Detection

        ### Machine Learning + Gradio

        Enter the tumor measurement features below and the trained
        machine-learning model will predict whether the sample is
        **Benign** or **Malignant**.

        **Note:** This is an educational machine-learning project and
        should not be used as a medical diagnosis.
        """
    )

    with gr.Row():

        with gr.Column():

            gr.Markdown("### Mean Features")

            mean_radius = gr.Number(
                label="Mean Radius",
                value=14.0
            )

            mean_texture = gr.Number(
                label="Mean Texture",
                value=19.0
            )

            mean_perimeter = gr.Number(
                label="Mean Perimeter",
                value=90.0
            )

            mean_area = gr.Number(
                label="Mean Area",
                value=650.0
            )

            mean_smoothness = gr.Number(
                label="Mean Smoothness",
                value=0.096
            )

            mean_compactness = gr.Number(
                label="Mean Compactness",
                value=0.10
            )

            mean_concavity = gr.Number(
                label="Mean Concavity",
                value=0.08
            )

            mean_concave_points = gr.Number(
                label="Mean Concave Points",
                value=0.05
            )

            mean_symmetry = gr.Number(
                label="Mean Symmetry",
                value=0.18
            )

            mean_fractal_dimension = gr.Number(
                label="Mean Fractal Dimension",
                value=0.06
            )

        with gr.Column():

            gr.Markdown("### Standard Error Features")

            radius_error = gr.Number(label="Radius Error", value=0.4)
            texture_error = gr.Number(label="Texture Error", value=1.2)
            perimeter_error = gr.Number(label="Perimeter Error", value=2.8)
            area_error = gr.Number(label="Area Error", value=40.0)
            smoothness_error = gr.Number(label="Smoothness Error", value=0.007)
            compactness_error = gr.Number(label="Compactness Error", value=0.02)
            concavity_error = gr.Number(label="Concavity Error", value=0.03)
            concave_points_error = gr.Number(
                label="Concave Points Error",
                value=0.01
            )
            symmetry_error = gr.Number(label="Symmetry Error", value=0.02)
            fractal_dimension_error = gr.Number(
                label="Fractal Dimension Error",
                value=0.003
            )

        with gr.Column():

            gr.Markdown("### Worst Features")

            worst_radius = gr.Number(
                label="Worst Radius",
                value=16.0
            )

            worst_texture = gr.Number(
                label="Worst Texture",
                value=25.0
            )

            worst_perimeter = gr.Number(
                label="Worst Perimeter",
                value=105.0
            )

            worst_area = gr.Number(
                label="Worst Area",
                value=800.0
            )

            worst_smoothness = gr.Number(
                label="Worst Smoothness",
                value=0.12
            )

            worst_compactness = gr.Number(
                label="Worst Compactness",
                value=0.25
            )

            worst_concavity = gr.Number(
                label="Worst Concavity",
                value=0.30
            )

            worst_concave_points = gr.Number(
                label="Worst Concave Points",
                value=0.12
            )

            worst_symmetry = gr.Number(
                label="Worst Symmetry",
                value=0.25
            )

            worst_fractal_dimension = gr.Number(
                label="Worst Fractal Dimension",
                value=0.08
            )

    predict_button = gr.Button(
        "🔍 Predict Cancer",
        variant="primary"
    )

    result = gr.Textbox(
        label="Prediction",
        interactive=False
    )

    probability = gr.Textbox(
        label="Prediction Probability",
        interactive=False
    )

    predict_button.click(
        fn=predict_cancer,

        inputs=[
            mean_radius,
            mean_texture,
            mean_perimeter,
            mean_area,
            mean_smoothness,
            mean_compactness,
            mean_concavity,
            mean_concave_points,
            mean_symmetry,
            mean_fractal_dimension,

            radius_error,
            texture_error,
            perimeter_error,
            area_error,
            smoothness_error,
            compactness_error,
            concavity_error,
            concave_points_error,
            symmetry_error,
            fractal_dimension_error,

            worst_radius,
            worst_texture,
            worst_perimeter,
            worst_area,
            worst_smoothness,
            worst_compactness,
            worst_concavity,
            worst_concave_points,
            worst_symmetry,
            worst_fractal_dimension
        ],

        outputs=[
            result,
            probability
        ]
    )

    gr.Markdown(
        """
        ---
        **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset

        **Model:** Logistic Regression

        **Preprocessing:** StandardScaler
        """
    )


# 7. LAUNCH APP

if __name__ == "__main__":
    demo.launch()

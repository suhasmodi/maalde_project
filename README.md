# AI/ML Demand Prediction Engine

**Full Name:** Suhas Modi  
**Mobile No:** 9998595763

---

## 1. Plan and execution.

### Approach & Plan
The goal was to predict future sales quantities based on historical sales data and product design images. Given the multimodal nature of the data (tabular sales data + unstructured images), the approach required feature fusion.
1. **Data Preprocessing**: The sales data (`AI ML Task Sheet - sales data.csv`) contained multiple transactions per product. I aggregated the data by product `code` to compute the `total quantity` sold and the `average rate` (price). 
   - *Note: Since the provided image filenames did not match the product codes, a deterministic mock mapping was implemented to link images in the `1` directory to the unique product codes for demonstration purposes.*
2. **Feature Extraction**: To understand the "design" aspect, I used a pre-trained CNN (ResNet18 via PyTorch). By stripping the final classification layer, the model acts as a feature extractor, transforming each image into a dense 512-dimensional embedding vector representing its visual style, patterns, and colors.
3. **Data Fusion**: The 512-dimensional visual embedding vector was concatenated with the numerical feature (`average rate`) to form the final feature set for each product.
4. **Model Training**: A RandomForest Regressor was trained on this combined dataset to map the relationship between [Visual Features + Price] -> [Sales Quantity].

### Execution
The pipeline is split into two primary scripts:
- `train.py`: Handles data loading, mapping, feature extraction, model training, and saves the final RandomForest Regressor model.
- `predict.py`: An inference script that accepts a new image and a proposed rate, extracts the features, and predicts the expected sales quantity.

---

## 2. How does your prediction system work? (logic/model)

The system works through a Two-Stage Pipeline:
1. **Stage 1 (Computer Vision)**: A pre-trained **ResNet18** model processes the input image. It applies convolutional filters to capture hierarchical visual patterns (edges, shapes, textures) and outputs a 512-dimensional embedding vector.
2. **Stage 2 (Machine Learning)**: An **RandomForest Regressor** takes the 512 visual features plus the proposed price (`rate`) as input. RandomForest Regressor builds mulitple decision trees that learn complex, non-linear interactions between the visual appeal of the design and its price point to predict the continuous target variable (`qty`).

---

## 3. What patterns did you find in the data?

During exploratory data analysis:
1. **Price Variability**: There are significant clusters of products priced at specific tiers (e.g., ~₹550, ~₹875, ~₹1295). 
2. **Sales Concentration**: Most transactions involve small quantities (e.g., 4 units), but there are massive outliers where quantities spike (e.g., 50, 60, or 100 units). These spikes could indicate bulk orders or highly viral designs.
3. **Temporal Grouping**: The sales logs appear in distinct bursts on specific dates, suggesting batch processing of orders or specific promotional periods.
4. **Code Frequencies**: Certain product codes (like `500001` and `10028268`) appear very frequently across different dates, establishing them as staple or best-selling designs.

---

## 4. Where can your system fail?

1. **New Visual Styles (Out of Distribution)**: The ResNet18 model was pre-trained on ImageNet. Highly specific or abstract textile patterns might not map well into its feature space. If a completely new, unseen design trend emerges, the embeddings might not capture its uniqueness effectively.
2. **Extreme Outliers**: The model optimizes for standard variance. It may severely under-predict highly viral products (e.g., orders of 100+ units) because it leans towards the mean behavior (most orders being around 4 units).
3. **Missing Context**: Demand is driven by factors outside the image and price (e.g., seasonality, marketing budget, overall market trends). Without these external features, the system is fundamentally limited.
4. **Mocked Data Dependency**: Since the provided image filenames lacked product codes, the model currently trains on random deterministic mappings. If true relationships between specific designs and high sales are non-existent in the actual data mapping, the model will fail to learn meaningful visual correlations.

---

## 5. If you had more time, how would you improve this system?

1. **Fine-Tuning the CNN**: Instead of using a frozen ResNet18, I would unfreeze the last few convolutional layers and fine-tune the entire architecture end-to-end on a metric learning objective (like Triplet Loss) to explicitly learn features relevant to apparel/design aesthetics.
2. **Handle Time-Series Dynamics**: Instead of aggregating total sales, I would structure this as a time-series forecasting problem (e.g., using LSTMs or Temporal Fusion Transformers) to predict "demand over the next 30 days" considering seasonality and trend slopes.
3. **Addressing Outliers (Long-Tail)**: Implement quantile regression or a two-stage hurdle model: Stage 1 predicts *if* the product will be a viral hit (classification), and Stage 2 predicts the actual quantity.
4. **Data Augmentation**: Apply robust image augmentations (cropping, color jittering) to prevent overfitting on the small dataset of ~181 images.
5. **Explainable AI (XAI)**: Integrate tools like SHAP and Grad-CAM to show *which* parts of the design the model focuses on when predicting high demand.

---

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed. It's recommended to use a virtual environment.

### 2. Install Dependencies
Run the following command to install required libraries:
```bash
pip install -r requirements.txt
```

### 3. Run the Training Pipeline
Ensure `AI ML Task Sheet - sales data.csv` and the `1` directory (containing images) are in the root folder.
```bash
python train.py
```
*This will aggregate the data, extract features, train the model, and save `demand_model.pkl`.*

### 4. Run Inference (Command Line)
To predict the demand for a new design via CLI, run:
```bash
python predict.py --image "path/to/your/image.jpeg" --rate 1295
```

### 5. Run Web UI
To launch the premium web interface:
```bash
python app.py
```
*Then open your browser and navigate to `http://127.0.0.1:5000` to interact with the model via a drag-and-drop GUI.*

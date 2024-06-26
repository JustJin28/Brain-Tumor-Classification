# Brain Tumor Image Classification

This project classifies brain tumor images into three categories: meningioma, glioma, and pituitary tumor. The images have been preprocessed, augmented, and used to train a convolutional neural network (CNN) to achieve this classification.

## Project Structure

Brain_Tumor_Image_Detection/  
├── data/  
│   ├── raw/  
│   │   ├── archive-3/  
│   │       ├── 1/  
│   │       ├── 2/  
│   │       ├── 3/  
│   ├── processed/  
│   │   ├── 1/  
│   │   ├── 2/  
│   │   ├── 3/  
│   ├── processed_augmented/  
│       ├── 1_augmented/  
│       ├── 2_augmented/  
│       ├── 3_augmented/  
├── models/  
│   └── best_model.keras  
│   └── final_model.keras  
├── notebooks/  
│   ├── Data_Cleaning_and_EDA_Brain_Tumor.ipynb  
│   ├── Modeling_Brain_Tumor.ipynb  
└── README.md  

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- Pandas
- PIL (Pillow)

You can install the required packages using the following command:

bash
pip install tensorflow keras numpy matplotlib seaborn pandas pillow

## Data Preprocessing and EDA

The `Data_Cleaning_and_EDA_Brain_Tumor.ipynb` notebook includes the following steps:

1. **Normalize and Resize Images:**
   - Images are resized to 256x256 pixels and normalized.
   
2. **Data Augmentation:**
   - Augmented images are generated to increase the variability in the dataset.

3. **Exploratory Data Analysis (EDA):**
   - Distribution of classes, pixel intensity, and image sizes are analyzed.

## Model Training

The `Modeling_Brain_Tumor.ipynb` notebook includes the following steps:

1. **Data Generators:**
   - Use `ImageDataGenerator` to create training and validation data generators with aggressive augmentation.
   
2. **Model Definition:**
   - A CNN is defined using the Sequential API with Conv2D, MaxPooling2D, Flatten, Dense, and Dropout layers.
   
3. **Model Training:**
   - The model is trained using the training data generator with early stopping and checkpointing.
   - Learning rate scheduler is used to adjust the learning rate during training.

4. **Model Evaluation:**
   - Training and validation accuracy and loss are plotted to visualize model performance.

## How to Run

1. **Clone the Repository:**

```bash
git clone https://github.com/JustJin28/brain-tumor-image-classification.git
cd brain-tumor-image-classification



2.
Run the Notebooks:
	•	Open Data_Cleaning_and_EDA_Brain_Tumor.ipynb and run all cells to preprocess the data.
	•	Open Modeling_Brain_Tumor.ipynb and run all cells to train and evaluate the model.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any suggestions or improvements.

## License

This project is licensed under the MIT License.


```python

```

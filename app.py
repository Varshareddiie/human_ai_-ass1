from flask import Flask, render_template, request, send_file
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import numpy as np



app = Flask(__name__)

def handle_missing_values(data):
    """Handle missing values in tabular data."""
    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
        # For numerical columns, fill missing values with mean or median
            if data[column].isnull().sum() > 0: #it counts the misssing values in col
                if data[column].dtype == 'float64':
                    data[column].fillna(data[column].mean(), inplace=True)
                else:
                    data[column].fillna(data[column].median(), inplace=True)
        elif data[column].dtype == 'object':
        # For categorical columns, fill missing values with mode
            if data[column].isnull().sum() > 0:
                data[column].fillna(data[column].mode().iloc[0], inplace=True)
    return data

def normalize_data(data):
    """
    Function to perform feature scaling/normalization using MinMaxScaler on tabular data.
    Parameters: data: DataFrame containing the tabular data.
    Returns: DataFrame with normalized features.
    """
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    
    # Select numerical columns for normalization
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    
    # Normalize numerical columns
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    
    return data


def encode_categorical(data):
    """Encode categorical variables using one-hot encoding."""
    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns
    
    encoded_data = pd.get_dummies(data, columns=categorical_cols)
    
    return encoded_data


def handle_outliers(data, method, outliers_method):
    """
    Handle outliers in tabular data using IQR and Z-score methods for numerical columns.

    Parameters:
        data (DataFrame): Input tabular data.
        method (str): Method for handling outliers. Options: 'iqr', 'z-score'.
        outliers_method (str): Method for handling outliers. Options: 'clip', 'remove'.

    Returns:
        DataFrame: Tabular data with outliers handled according to the specified method.
    """
    numerical_columns = data.select_dtypes(include='number').columns
    non_numerical_columns = data.columns.difference(numerical_columns)

    if method == 'iqr':
        clean_data = data.copy()
        for col in numerical_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            print(lower_bound,upper_bound)
            if outliers_method == 'clip':
                clean_data[col] = clean_data[col].clip(lower=lower_bound, upper=upper_bound)
            elif outliers_method == 'remove':
                clean_data = clean_data[(clean_data[col] >= lower_bound) & (clean_data[col] <= upper_bound)]
            else:
                raise ValueError("Invalid outliers_method. Choose either 'clip' or 'remove'.")
    elif method == 'z-score':
        clean_data = data.copy()
        for col in numerical_columns:
            z_scores = (data[col] - data[col].mean()) / data[col].std()
            if outliers_method == 'clip':
                clean_data[col] = clean_data[col].clip(lower=-3, upper=3)
            elif outliers_method == 'remove':
                clean_data = clean_data.drop(clean_data.index[(z_scores < -3) | (z_scores > 3)])
            else:
                raise ValueError("Invalid outliers_method. Choose either 'clip' or 'remove'.")
    else:
        raise ValueError("Invalid method. Choose either 'iqr' or 'z-score'.")

    return clean_data


def handle_imbalanced_data(data, target_column_name, method):
    """
    Function to handle imbalanced data in tabular format.

    Parameters:
        data (DataFrame): Input tabular data.
        target_column_name (str): Name of the target column.
        method (str): Method for handling imbalanced data. Options: 'oversample', 'undersample'.

    Returns:
        DataFrame: Tabular data with balanced class distribution.
    """
    X = data.drop(columns=[target_column_name])  # Features
    y = data[target_column_name]  # Target variable

    if method == 'oversample':
        # Oversampling minority class
        sampler = RandomOverSampler()
    elif method == 'undersample':
        # Undersampling majority class
        sampler = RandomUnderSampler()
    else:
        raise ValueError("Invalid method. Choose either 'oversample' or 'undersample'.")

    # Resample the data
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # Recreate DataFrame with balanced class distribution
    balanced_data = pd.DataFrame(X_resampled, columns=X.columns)
    balanced_data[target_column_name] = y_resampled

    return balanced_data


def split_data_with_ratio(data, ratio, random_state=None):
    """
    Split the dataset into training and testing sets based on a ratio value.

    Parameters:
    - data: The pandas DataFrame containing the dataset.
    - ratio: The ratio for splitting the data (e.g., '0.2' for 20% test data).
    - random_state: Random state for reproducibility.

    Returns:
    - train_data: The training dataset.
    - test_data: The testing dataset.
    """
    # Convert ratio to float
    ratio = float(ratio)

    # Split the data based on the ratio
    train_data, test_data = train_test_split(data, test_size=ratio, random_state=random_state)

    return train_data, test_data


#TEXT PREPROCESSING
import re
import nltk as nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
# nltk.download('stopwords')


import os
import PyPDF2
import docx
from docx import Document

def read_text_data(file_path):
    """
    Reads text data from different file formats.

    Parameters:
        file_path (str): Path to the text file.

    Returns:
        str: Content of the text file.
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.txt':
        # Read plain text file
        with open(file_path, 'r', encoding='utf-8') as file:
            text_data = file.read()

    elif file_extension == '.pdf':
        # Read PDF file
        text_data = ''
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                text_data += page.extractText()

    elif file_extension == '.docx':
        # Read DOCX file
        doc = docx.Document(file_path)
        paragraphs = []
        for paragraph in doc.paragraphs:
            paragraphs.append(paragraph.text)
        text_data = '\n'.join(paragraphs)
    elif file_extension == '.csv':
        text_column = 'text'
        preprocessing_options = {
            'lowercase': True,
            'tokenize': True,
            'remove_punctuation': True,
            'remove_stopwords': True,
            'stemming': False,
            'lemmatization': False
        }
        text_data = preprocessed_data = preprocess_csv_text(file_path, text_column, preprocessing_options)

    else:
        raise ValueError("Unsupported file format")

    return text_data


def preprocess_csv_text(file_path, text_column, preprocessing_options):
    """
    Preprocesses text data in a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.
        text_column (str): Name of the column containing text data.
        preprocessing_options (dict): Dictionary specifying preprocessing options.

    Returns:
        pd.DataFrame: DataFrame with preprocessed text data.
    """
    data = pd.read_csv(file_path)
    data[text_column] = data[text_column].apply(preprocess_text, **preprocessing_options)
    return data



import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import unidecode  

def preprocess_text(text, lowercase=True, tokenize=True, remove_punctuation=True, remove_stopwords=True, 
                    stemming=False, lemmatization=False, remove_numbers=True, remove_special_characters=True,
                    handle_contractions=True, handle_urls_emails=True, normalize_accents=True):
    """
    Preprocesses text data with various techniques.

    Parameters:
        text (str): Input text data
        lowercase (bool): Convert text to lowercase. Default is True
        tokenize (bool): Tokenize text into words. Default is True
        remove_punctuation (bool): Remove punctuation marks. Default is True
        remove_stopwords (bool): Remove stopwords. Default is True
        stemming (bool): Perform stemming. Default is False
        lemmatization (bool): Perform lemmatization. Default is False
        remove_numbers (bool): Remove numeric digits. Default is True
        remove_special_characters (bool): Remove special characters. Default is True
        handle_contractions (bool): Expand contractions. Default is True
        handle_urls_emails (bool): Replace URLs and email addresses with placeholders. Default is True
        normalize_accents (bool): Normalize accented characters. Default is True

    Returns:
        str: Preprocessed text.
    """
    # Lowercasing
    if lowercase:
        text = text.lower()

    # Handle contractions
    if handle_contractions:
        text = expand_contractions(text)

    # Handle URLs and email addresses
    if handle_urls_emails:
        text = replace_urls_emails(text)

    # Tokenization
    if tokenize:
        tokens = word_tokenize(text)

    # Removing punctuation
    if remove_punctuation:
        tokens = [word for word in tokens if word not in string.punctuation]

    # Removing numbers
    if remove_numbers:
        tokens = [word for word in tokens if not word.isdigit()]

    # Removing special characters
    if remove_special_characters:
        tokens = [re.sub(r'[^a-zA-Z0-9\s]', '', word) for word in tokens]

    # Removing stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

    # Lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Normalize accents
    if normalize_accents:
        tokens = [unidecode.unidecode(word) for word in tokens]

    # Join tokens back to string
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def expand_contractions(text):
    # some common English contractions
    contractions_dict = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "I'd": "I would",
        "I'd've": "I would have",
        "I'll": "I will",
        "I'll've": "I will have",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    # regex pattern to match any of the contractions
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        expanded_contraction = contractions_dict.get(match) if contractions_dict.get(match) else contractions_dict.get(match.lower())
        return expanded_contraction
    
    # Replace contractions with their expansions in the text
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

def replace_urls_emails(text):
    # Replace URLs with "<URL>" and email addresses with "<EMAIL>"
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\S+@\S+')

    # Replace URLs and email addresses with placeholders
    replaced_text = re.sub(url_pattern, '<URL>', text)
    replaced_text = re.sub(email_pattern, '<EMAIL>', replaced_text)
    
    return replaced_text

#IMAGE PREPROCESSING
import os
import cv2
import zipfile
from PIL import Image

def load_images_from_zip(zip_file_path, extract_folder):
    """
    Extract images from a zip file and load them.

    Parameters:
        zip_file_path (str): Path to the zip file containing images.
        extract_folder (str): Folder to extract the images to.

    Returns:
        list: List of loaded images.
    """
    images = []
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    for root, dirs, files in os.walk(extract_folder):
        for file in files:
            if any(file.lower().endswith(image_format) for image_format in ['.png','.jpg','.jpeg']):
                img_path = os.path.join(root,file)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
    # for filename in os.listdir(extract_folder):
    #     img_path = os.path.join(extract_folder, filename)
    #     if os.path.isfile(img_path):
    #         img = cv2.imread(img_path)
    #         if img is not None:
    #             images.append(img)
    return images

def resize_images(images, width, height):
    """
    Resize a list of images to a specific width and height.

    Parameters:
        images (list): List of input images.
        width (int): Target width for resizing.
        height (int): Target height for resizing.

    Returns:
        list: List of resized images.
    """
    resized_images = [cv2.resize(img, (width, height)) for img in images]
    return resized_images

def rescale_images(images, scale_factor):
    """
    Rescale a list of images by a given scale factor.

    Parameters:
        images (list): List of input images.
        scale_factor (float): Scaling factor for resizing.

    Returns:
        list: List of rescaled images.
    """
    rescaled_images = [cv2.resize(img, None, fx=scale_factor, fy=scale_factor) for img in images]
    return rescaled_images

def normalize_images(images):
    """
    Normalize a list of images to the range [0, 1].

    Parameters:
        images (list): List of input images.

    Returns:
        list: List of normalized images.
    """
    normalized_images = []
    for img in images:
        # Normalizes pixel values to the range [0, 1]
        img_normalized = img.astype('float32') / 255.0
        normalized_images.append(img_normalized)
    return normalized_images



def augment_images(images, rotation_angle=0, horizontal_flip=False, vertical_flip=False, crop_x=None, crop_y=None, crop_width=None, crop_height=None):
    """
    Apply data augmentation techniques to a list of images, including rotation, flipping, and cropping.

    Args:
    - images (list): List of input images.
    - rotation_angle (float): Angle (in degrees) for rotation.
    - horizontal_flip (bool): Flag to perform horizontal flipping.
    - vertical_flip (bool): Flag to perform vertical flipping.
    - crop_x (int): X-coordinate of the top-left corner of the crop area.
    - crop_y (int): Y-coordinate of the top-left corner of the crop area.
    - crop_width (int): Width of the crop area.
    - crop_height (int): Height of the crop area.

    Returns:
    - augmented_images (list): List of augmented images.
    """
    augmented_images = []
    for img in images:
        augmented_img = img.copy()
        if rotation_angle != 0:
            rows, cols = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
            augmented_img = cv2.warpAffine(augmented_img, rotation_matrix, (cols, rows))
        if horizontal_flip:
            augmented_img = cv2.flip(augmented_img, 1)
        if vertical_flip:
            augmented_img = cv2.flip(augmented_img, 0)
        if crop_x is not None and crop_y is not None and crop_width is not None and crop_height is not None:
            augmented_img = augmented_img[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        augmented_images.append(augmented_img)
    return augmented_images


# Function to display images
def display_images_cv2(images, titles):
    for img, title in zip(images, titles):
        cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def convert_to_grayscale(images):
    """
    Convert a list of images to grayscale.

    Args:
    - images (list): List of input images.

    Returns:
    - grayscale_images (list): List of grayscale images.
    """
    grayscale_images = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
    return grayscale_images



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    file_path = 'uploads/' + file.filename
    file.save(file_path)

    data_type = request.form.get('data_type')
    if data_type not in ['tabular', 'text', 'image']:
        return 'Invalid data type'
    preprocessing_options = request.form.getlist('preprocessing_option')
    if not preprocessing_options:
        return 'No preprocessing options selected'

    try:
        if data_type == 'tabular':
            if file.filename.endswith('.csv'):
                data = pd.read_csv(file_path)
            elif file.filename.endswith(('.xls', '.xlsx')):
                data = pd.read_excel(file_path)
            else:
                return 'Unsupported file format'

            if 'Handle Missing Values' in preprocessing_options:
                print(data)
                data = handle_missing_values(data)
                print(data)
            if 'Categorical Variable Encoding' in preprocessing_options:
                data = encode_categorical(data)
                print(data)
            if 'Handle Outliers' in preprocessing_options:
                print(data)
                method = request.form.get('method')
                outliers_method = request.form.get('outliers_method')
                print(method, outliers_method)
                data = handle_outliers(data, method, outliers_method)
                print(data)
            if 'Feature Scaling/Normalization' in preprocessing_options:
                data = normalize_data(data)
                print(data)
            if 'Handle Imbalanced Data' in preprocessing_options:
                print(data)
                target = request.form.get("target_col")
                method = request.form.get("sampling_method")
                data = handle_imbalanced_data(data, 'target', method)
                print(data)
            if 'Data Splitting' in preprocessing_options:
                ratio = request.form.get('split_ratio')
                train, test = split_data_with_ratio(data, ratio)
                print(train)
                print(test)
            # After preprocessing, save the DataFrame to a file
            preprocessed_file_path=os.path.join('uploads','preprocessed_data.csv')
            data.to_csv(preprocessed_file_path,index=False)
            return render_template('preprocessing_complete.html', file_path=preprocessed_file_path)

        
        elif data_type == 'text':
            text_data = read_text_data(file_path)
            print(text_data)
            processed_text = preprocess_text(text_data)
            print(processed_text)
            # After preprocessing, save the preprocessed text data to a file
            preprocessed_file_path = 'preprocessed_text.txt'  # Change the filename as needed
            with open(preprocessed_file_path, 'w', encoding='utf-8') as file:
                file.write(processed_text)  # Write preprocessed text data to file

            # download link to the user
            return render_template('preprocessing_complete.html', file_path=preprocessed_file_path)

        elif data_type == 'image':
            print('in image block')
            if file.filename.endswith('.zip'):
                extract_folder = 'uploads/' + os.path.splitext(file.filename)[0]
                os.makedirs(extract_folder,exist_ok =True)
                images = load_images_from_zip(file_path, extract_folder)
                # for img in images[:5]:  
                #     print(img.shape)
                if 'Resizing' in preprocessing_options:
                    width = int(request.form.get('resize_width'))
                    height = int(request.form.get('resize_height'))
                    display_images_cv2([images[0]], ["Original Image"])
                    images = resize_images(images, width, height)
                    display_images_cv2([images[0]], ["Resized Image"])
                    # for img in images[:5]: 
                    #     print(img.shape)
                if 'Rescaling' in preprocessing_options: #adjusting the pixel intensities
                    factor = float(request.form.get('scale_factor'))
                    # original_images = images.copy()
                    print("Before Rescaling:")
                    display_images_cv2([images[0]], ["Original Image"])
                    images = rescale_images(images,factor)
                    print("After Rescaling:")
                    display_images_cv2([images[0]], ["Rescaled Image"])
                if 'Normalization' in preprocessing_options:
                    print("Before Normalization:")
                    display_images_cv2([images[0]], ["Original Image"])
                    print(images[0])
                    images = normalize_images(images)
                    print("After Normalization:")
                    display_images_cv2([images[0]], ["Normalized Image"])
                    print(images[0])
                # Perform Data Augmentation
                if 'Data Augmentation' in preprocessing_options:
                    rotation_angle = float(request.form.get('rotation_angle', 0))
                    horizontal_flip = request.form.get('horizontal_flip', 'off') == 'on'
                    vertical_flip = request.form.get('vertical_flip', 'off') == 'on'
                    crop_x = int(request.form.get('crop_x', 0))
                    crop_y = int(request.form.get('crop_y', 0))
                    crop_width = int(request.form.get('crop_width', 0))
                    crop_height = int(request.form.get('crop_height', 0))
                    # Before augmentation: Display the original image
                    print("Before Data Augmentation:")
                    display_images_cv2([images[0]], ["Original Image"])
                    images = augment_images(images, rotation_angle=rotation_angle, 
                                            horizontal_flip=horizontal_flip, vertical_flip=vertical_flip, 
                                            crop_x=crop_x, crop_y=crop_y, 
                                            crop_width=crop_width, crop_height=crop_height)

                    # After augmentation: Display the augmented image
                    print("After Data Augmentation:")
                    display_images_cv2([images[0]], ["Augmented Image"])
                if 'Gray-scale Conversion' in preprocessing_options:
                    print("Before conversion:")
                    display_images_cv2([images[0]], ["Original Image"])
                    images = convert_to_grayscale(images)
                    # After augmentation: Display the augmented image
                    print("After Data Augmentation:")
                    display_images_cv2([images[0]], ["Converted Image"])
                
                preprocessed_zip_path = 'preprocessed_images.zip'  
                with zipfile.ZipFile(preprocessed_zip_path, 'w') as zipf:
                    for i, img in enumerate(images):
                        img_path = f'preprocessed_image_{i}.jpg'  
                        cv2.imwrite(img_path, img)  # Save preprocessed image
                        zipf.write(img_path)  # Add image to zip archive

                        # download link to the user
            
            return render_template('preprocessing_complete.html', file_path=preprocessed_file_path)

        else:
            return 'Something is Wrong'
    except Exception as e:
        return f'Error: {str(e)}'
        

        
@app.route('/download/<path:file_path>')
def download_file(file_path):
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

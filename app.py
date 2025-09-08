# app.py
from flask import Flask, render_template, request, send_file
import os
import zipfile
import pandas as pd
import numpy as np

# ML / Sampling
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Text utils
import re
import string
import unidecode

# NLTK (download resources at runtime if missing)
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Docs / PDFs
import PyPDF2
import docx

# Images
import cv2
from PIL import Image


# ---------- Flask App ----------
app = Flask(__name__)

# Ensure uploads dir exists
os.makedirs("uploads", exist_ok=True)


# ---------- Helpers: Tabular ----------
def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in tabular data."""
    for column in data.columns:
        if data[column].dtype in ["int64", "float64"]:
            # For numerical columns, fill missing values with mean/median
            if data[column].isnull().sum() > 0:
                # use mean for floats, median for ints
                if data[column].dtype == "float64":
                    data[column].fillna(data[column].mean(), inplace=True)
                else:
                    data[column].fillna(data[column].median(), inplace=True)
        elif data[column].dtype == "object":
            # For categorical columns, fill missing with mode
            if data[column].isnull().sum() > 0:
                data[column].fillna(data[column].mode().iloc[0], inplace=True)
    return data


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """Feature scaling using MinMaxScaler for numeric columns."""
    scaler = MinMaxScaler()
    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) == 0:
        return data
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data


def encode_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    categorical_cols = data.select_dtypes(include=["object"]).columns
    if len(categorical_cols) == 0:
        return data
    data = pd.get_dummies(data, columns=categorical_cols)
    return data


def handle_outliers(data: pd.DataFrame, method: str, outliers_method: str) -> pd.DataFrame:
    """
    Handle outliers using IQR or Z-score.
    method: 'iqr' or 'z-score'
    outliers_method: 'clip' or 'remove'
    """
    numeric_cols = data.select_dtypes(include="number").columns
    clean_data = data.copy()

    if len(numeric_cols) == 0:
        return clean_data

    if method == "iqr":
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            if outliers_method == "clip":
                clean_data[col] = clean_data[col].clip(lower=lower, upper=upper)
            elif outliers_method == "remove":
                clean_data = clean_data[(clean_data[col] >= lower) & (clean_data[col] <= upper)]
            else:
                raise ValueError("Invalid outliers_method. Use 'clip' or 'remove'.")
    elif method == "z-score":
        for col in numeric_cols:
            mu = data[col].mean()
            sigma = data[col].std(ddof=0) if data[col].std(ddof=0) != 0 else 1.0
            z = (data[col] - mu) / sigma
            if outliers_method == "clip":
                # clip to +/-3 std then convert back to original scale
                clipped_z = z.clip(-3, 3)
                clean_data[col] = clipped_z * sigma + mu
            elif outliers_method == "remove":
                clean_data = clean_data[(z >= -3) & (z <= 3)]
            else:
                raise ValueError("Invalid outliers_method. Use 'clip' or 'remove'.")
    else:
        raise ValueError("Invalid method. Use 'iqr' or 'z-score'.")

    return clean_data


def handle_imbalanced_data(data: pd.DataFrame, target_column_name: str, method: str) -> pd.DataFrame:
    """Balance classes with over/under-sampling."""
    if target_column_name not in data.columns:
        raise ValueError(f"Target column '{target_column_name}' not found in dataset.")
    X = data.drop(columns=[target_column_name])
    y = data[target_column_name]

    if method == "oversample":
        sampler = RandomOverSampler()
    elif method == "undersample":
        sampler = RandomUnderSampler()
    else:
        raise ValueError("Invalid method. Use 'oversample' or 'undersample'.")

    X_res, y_res = sampler.fit_resample(X, y)
    balanced = pd.DataFrame(X_res, columns=X.columns)
    balanced[target_column_name] = y_res
    return balanced


def split_data_with_ratio(data: pd.DataFrame, ratio: float, random_state: int | None = None):
    """Split into train/test by ratio."""
    ratio = float(ratio)
    return train_test_split(data, test_size=ratio, random_state=random_state)


# ---------- Helpers: Text ----------
def ensure_nltk_resources():
    """Download NLTK resources if missing."""
    try:
        _ = stopwords.words("english")
    except Exception:
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)


def read_text_data(file_path: str) -> str:
    """Read text from txt, pdf, docx, or return CSV text column name to preprocess separately."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    elif ext == ".pdf":
        text = ""
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text

    elif ext == ".docx":
        document = docx.Document(file_path)
        return "\n".join(p.text for p in document.paragraphs)

    elif ext == ".csv":
        # The text pipeline below can handle CSV via preprocess_csv_text
        return ""  # We will ignore this return and handle CSV separately in the route

    else:
        raise ValueError("Unsupported file format for text.")


def preprocess_csv_text(file_path: str, text_column: str, options: dict) -> pd.DataFrame:
    """Apply preprocess_text to the CSV's text column and return a DataFrame."""
    df = pd.read_csv(file_path)
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in CSV.")
    df[text_column] = df[text_column].apply(lambda x: preprocess_text(str(x), **options))
    return df


def preprocess_text(
    text: str,
    lowercase: bool = True,
    tokenize: bool = True,
    remove_punctuation: bool = True,
    remove_stopwords_opt: bool = True,
    stemming: bool = False,
    lemmatization: bool = False,
    remove_numbers: bool = True,
    remove_special_characters: bool = True,
    handle_contractions: bool = True,
    handle_urls_emails: bool = True,
    normalize_accents: bool = True,
) -> str:
    """Generic text preprocessing."""
    ensure_nltk_resources()

    if lowercase:
        text = text.lower()
    if handle_contractions:
        text = expand_contractions(text)
    if handle_urls_emails:
        text = replace_urls_emails(text)

    tokens = word_tokenize(text) if tokenize else [text]

    if remove_punctuation:
        tokens = [w for w in tokens if w not in string.punctuation]
    if remove_numbers:
        tokens = [w for w in tokens if not w.isdigit()]
    if remove_special_characters:
        tokens = [re.sub(r"[^a-zA-Z0-9\s]", "", w) for w in tokens]
    if remove_stopwords_opt:
        sw = set(stopwords.words("english"))
        tokens = [w for w in tokens if w not in sw]
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(w) for w in tokens]
    if lemmatization:
        lem = WordNetLemmatizer()
        tokens = [lem.lemmatize(w) for w in tokens]
    if normalize_accents:
        tokens = [unidecode.unidecode(w) for w in tokens]

    return " ".join(tokens)


def expand_contractions(text: str) -> str:
    contractions_dict = {
        "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
        "'cause": "because", "could've": "could have", "couldn't": "could not", "couldn't've": "could not have",
        "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have",
        "hasn't": "has not", "haven't": "have not", "he'd": "he would", "he'd've": "he would have",
        "he'll": "he will", "he'll've": "he will have", "he's": "he is", "how'd": "how did", "how'd'y": "how do you",
        "how'll": "how will", "how's": "how is", "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
        "I'll've": "I will have", "I'm": "I am", "I've": "I have", "isn't": "is not", "it'd": "it would",
        "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is", "let's": "let us",
        "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not",
        "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
        "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock", "oughtn't": "ought not",
        "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
        "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
        "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
        "so've": "so have", "so's": "so is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
        "there'd": "there would", "there'd've": "there would have", "there's": "there is", "they'd": "they would",
        "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are",
        "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
        "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not",
        "what'll": "what will", "what'll've": "what will have", "what're": "what are", "what's": "what is",
        "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
        "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
        "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
        "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
        "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are", "y'all've": "you all have",
        "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
        "you're": "you are", "you've": "you have"
    }
    pattern = re.compile('({})'.format('|'.join(map(re.escape, contractions_dict.keys()))),
                         flags=re.IGNORECASE | re.DOTALL)

    def expand_match(m):
        match = m.group(0)
        return contractions_dict.get(match) or contractions_dict.get(match.lower(), match)

    return pattern.sub(expand_match, text)


def replace_urls_emails(text: str) -> str:
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    email_pattern = re.compile(r'\b\S+@\S+\.\S+\b')
    text = re.sub(url_pattern, '<URL>', text)
    text = re.sub(email_pattern, '<EMAIL>', text)
    return text


# ---------- Helpers: Images ----------
def load_images_from_zip(zip_file_path: str, extract_folder: str):
    """Extract .png/.jpg/.jpeg images from a zip and return as list of np arrays (BGR)."""
    images = []
    with zipfile.ZipFile(zip_file_path, "r") as zf:
        zf.extractall(extract_folder)

    for root, _, files in os.walk(extract_folder):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                p = os.path.join(root, file)
                img = cv2.imread(p)
                if img is not None:
                    images.append((p, img))  # (path, image)
    return images


def resize_images(images, width: int, height: int):
    return [(p, cv2.resize(img, (width, height))) for p, img in images]


def rescale_images(images, scale_factor: float):
    return [(p, cv2.resize(img, None, fx=scale_factor, fy=scale_factor)) for p, img in images]


def normalize_images(images):
    out = []
    for p, img in images:
        img_norm = img.astype("float32") / 255.0
        out.append((p, img_norm))
    return out


def augment_images(
    images, rotation_angle=0, horizontal_flip=False, vertical_flip=False,
    crop_x=None, crop_y=None, crop_width=None, crop_height=None
):
    augmented = []
    for p, img in images:
        aug = img.copy()
        if rotation_angle:
            rows, cols = aug.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
            aug = cv2.warpAffine(aug, M, (cols, rows))
        if horizontal_flip:
            aug = cv2.flip(aug, 1)
        if vertical_flip:
            aug = cv2.flip(aug, 0)
        if None not in (crop_x, crop_y, crop_width, crop_height) and crop_width > 0 and crop_height > 0:
            aug = aug[crop_y:crop_y + crop_height, crop_x:crop_x + crop_width]
        augmented.append((p, aug))
    return augmented


def convert_to_grayscale(images):
    return [(p, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) for p, img in images]


# ---------- Routes ----------
@app.route("/")
def index():
    # Ensure you have templates/index.html & templates/preprocessing_complete.html
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    data_type = request.form.get("data_type")
    if data_type not in ["tabular", "text", "image"]:
        return "Invalid data type"

    preprocessing_options = request.form.getlist("preprocessing_option")
    if not preprocessing_options:
        return "No preprocessing options selected"

    try:
        # ---------------- TABULAR ----------------
        if data_type == "tabular":
            if file.filename.endswith(".csv"):
                data = pd.read_csv(file_path)
            elif file.filename.endswith((".xls", ".xlsx")):
                data = pd.read_excel(file_path)
            else:
                return "Unsupported file format for tabular data"

            if "Handle Missing Values" in preprocessing_options:
                data = handle_missing_values(data)

            if "Categorical Variable Encoding" in preprocessing_options:
                data = encode_categorical(data)

            if "Handle Outliers" in preprocessing_options:
                method = request.form.get("method")              # 'iqr' or 'z-score'
                outliers_method = request.form.get("outliers_method")  # 'clip' or 'remove'
                if not method or not outliers_method:
                    return "Outliers method parameters are required."
                data = handle_outliers(data, method, outliers_method)

            if "Feature Scaling/Normalization" in preprocessing_options:
                data = normalize_data(data)

            if "Handle Imbalanced Data" in preprocessing_options:
                target = request.form.get("target_col")  # <- use the actual form field
                sampling_method = request.form.get("sampling_method")  # 'oversample' or 'undersample'
                if not target or not sampling_method:
                    return "Target column and sampling method are required for handling imbalanced data."
                data = handle_imbalanced_data(data, target, sampling_method)

            if "Data Splitting" in preprocessing_options:
                ratio = request.form.get("split_ratio", "0.2")
                train, test = split_data_with_ratio(data, ratio)
                # Save both train and test
                preproc_train_path = os.path.join("uploads", "preprocessed_train.csv")
                preproc_test_path = os.path.join("uploads", "preprocessed_test.csv")
                train.to_csv(preproc_train_path, index=False)
                test.to_csv(preproc_test_path, index=False)
                # You could show both links in your template
                return render_template("preprocessing_complete.html",
                                       file_path=preproc_train_path,
                                       extra_file_path=preproc_test_path)

            # If not splitting, just save one consolidated preprocessed file
            preprocessed_file_path = os.path.join("uploads", "preprocessed_data.csv")
            data.to_csv(preprocessed_file_path, index=False)
            return render_template("preprocessing_complete.html", file_path=preprocessed_file_path)

        # ---------------- TEXT ----------------
        elif data_type == "text":
            ext = os.path.splitext(file_path)[1].lower()

            if ext == ".csv":
                # Expect a text column name from the form, default to 'text'
                text_column = request.form.get("text_column", "text")
                options = {
                    "lowercase": True,
                    "tokenize": True,
                    "remove_punctuation": True,
                    "remove_stopwords_opt": True,
                    "stemming": False,
                    "lemmatization": False,
                    "remove_numbers": True,
                    "remove_special_characters": True,
                    "handle_contractions": True,
                    "handle_urls_emails": True,
                    "normalize_accents": True,
                }
                processed_df = preprocess_csv_text(file_path, text_column, options)
                preprocessed_file_path = os.path.join("uploads", "preprocessed_text.csv")
                processed_df.to_csv(preprocessed_file_path, index=False)
                return render_template("preprocessing_complete.html", file_path=preprocessed_file_path)
            else:
                raw_text = read_text_data(file_path)
                processed_text = preprocess_text(raw_text)
                preprocessed_file_path = os.path.join("uploads", "preprocessed_text.txt")
                with open(preprocessed_file_path, "w", encoding="utf-8") as f:
                    f.write(processed_text)
                return render_template("preprocessing_complete.html", file_path=preprocessed_file_path)

        # ---------------- IMAGE ----------------
        elif data_type == "image":
            if not file.filename.lower().endswith(".zip"):
                return "Please upload a .zip containing images."

            extract_folder = os.path.join("uploads", os.path.splitext(file.filename)[0])
            os.makedirs(extract_folder, exist_ok=True)
            images = load_images_from_zip(file_path, extract_folder)  # list of (path, img)

            # Process images according to options
            if "Resizing" in preprocessing_options:
                width = int(request.form.get("resize_width", 0) or 0)
                height = int(request.form.get("resize_height", 0) or 0)
                if width > 0 and height > 0:
                    images = resize_images(images, width, height)

            if "Rescaling" in preprocessing_options:
                factor = float(request.form.get("scale_factor", 1.0) or 1.0)
                images = rescale_images(images, factor)

            if "Normalization" in preprocessing_options:
                images = normalize_images(images)

            if "Data Augmentation" in preprocessing_options:
                rotation_angle = float(request.form.get("rotation_angle", 0) or 0)
                horizontal_flip = request.form.get("horizontal_flip", "off") == "on"
                vertical_flip = request.form.get("vertical_flip", "off") == "on"
                crop_x = int(request.form.get("crop_x", 0) or 0)
                crop_y = int(request.form.get("crop_y", 0) or 0)
                crop_width = int(request.form.get("crop_width", 0) or 0)
                crop_height = int(request.form.get("crop_height", 0) or 0)
                images = augment_images(
                    images,
                    rotation_angle=rotation_angle,
                    horizontal_flip=horizontal_flip,
                    vertical_flip=vertical_flip,
                    crop_x=crop_x, crop_y=crop_y,
                    crop_width=crop_width, crop_height=crop_height,
                )

            if "Gray-scale Conversion" in preprocessing_options:
                images = convert_to_grayscale(images)

            # Save processed images to a zip
            preprocessed_zip_path = os.path.join("uploads", "preprocessed_images.zip")
            with zipfile.ZipFile(preprocessed_zip_path, "w") as zipf:
                for idx, (orig_path, img) in enumerate(images):
                    # Decide output extension
                    out_ext = ".png"
                    out_name = f"preprocessed_image_{idx}{out_ext}"
                    out_path = os.path.join("uploads", out_name)

                    # If image is float (normalized), convert back to uint8 for saving
                    if img.dtype != np.uint8:
                        # If grayscale float [0,1] -> [0,255]
                        img_to_save = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                    else:
                        img_to_save = img

                    # Save using cv2 (handles both gray & color)
                    cv2.imwrite(out_path, img_to_save)
                    zipf.write(out_path, arcname=out_name)

            return render_template("preprocessing_complete.html", file_path=preprocessed_zip_path)

        else:
            return "Something is wrong."

    except Exception as e:
        return f"Error: {str(e)}"


@app.route("/download/<path:file_path>")
def download_file(file_path):
    return send_file(file_path, as_attachment=True)


if __name__ == "__main__":
    # For local testing; in production use a WSGI server (gunicorn, etc.)
    app.run(debug=True)

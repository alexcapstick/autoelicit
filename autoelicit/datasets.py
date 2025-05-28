import urllib.request
import zipfile
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import sklearn.datasets as skd
from sklearn.utils import Bunch
from ucimlrepo import fetch_ucirepo
import typing
import tempfile

temp_dir = Path(tempfile.gettempdir(), "autoelicit_datasets")


uci_ids = {
    "adult": 2,
    "heart_disease": 45,
    "iris": 53,
    "wine_quality": 186,
    "wine": 109,
}


def load_and_save_uci_dataset(
    dataset_name: str,
    path: Path = temp_dir,
) -> pd.DataFrame:
    """
    This function loads and saves
    a dataset from the UCI repository
    if it has not been saved before.


    Arguments
    ---------

    dataset_name: str
        The name of the dataset to load.

    path: str or Path
        The directory where the file should be saved.
        Defaults to :code:`temp_dir`.


    Returns
    -------

    pd.DataFrame
        The dataset. The target column is named :code:`target`.


    """
    file_path = path.joinpath(f"{dataset_name}.parquet")

    if file_path.exists():
        data = pd.read_parquet(file_path)
    else:
        dataset = fetch_ucirepo(id=uci_ids.get(dataset_name, dataset_name))
        # saving the data for
        # the next time it is loaded
        print("Saving the data")
        data = dataset.data.features.assign(target=dataset.data.targets)
        data.to_parquet(file_path)

        # saving the original dataset as well used in the
        # load_and_save_uci_original_dataset function
        original_file_path = path.joinpath(f"{dataset_name}_original.parquet")
        if not original_file_path.exists():
            original_data = dataset.data.original
            original_data.to_parquet(original_file_path)

    return data


def load_and_save_uci_original_dataset(
    dataset_name: str,
    path: Path = temp_dir,
) -> pd.DataFrame:
    """
    This function loads and saves
    a dataset from the UCI repository
    if it has not been saved before.

    In particular, it saves the original dataset,
    which may include a different target column name.


    Arguments
    ---------

    dataset_name: str
        The name of the dataset to load.

    path: str or Path
        The directory where the file should be saved.
        Defaults to :code:`temp_dir`.


    Returns
    -------

    pd.DataFrame
        The dataset, with the original target column name.


    """
    file_path = path.joinpath(f"{dataset_name}_original.parquet")

    if file_path.exists():
        data = pd.read_parquet(file_path)
    else:
        dataset = fetch_ucirepo(id=uci_ids[dataset_name])
        # saving the data for
        # the next time it is loaded
        print("Saving the data")
        data = dataset.data.original
        data.to_parquet(file_path)

        # saving the standard dataset as well used in the
        # load_and_save_uci_dataset function
        standard_file_path = path.joinpath(f"{dataset_name}.parquet")
        if not standard_file_path.exists():
            standard_data = dataset.data.features.assign(target=dataset.data.targets)
            standard_data.to_parquet(standard_file_path)

    return data


def load_breast_cancer(
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch:
    """
    Load and return the breast cancer dataset.
    Note that the labels of this dataset are flipped compared to
    the sklearn dataset so that malignant=1 and benign=0. This
    is because it makes more sense for calculating recall and precision.

    Arguments
    ---------

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    breast_cancer_dataset = skd.load_breast_cancer(as_frame=True)

    data, target, target_names, feature_names = (
        breast_cancer_dataset["data"],
        # flipping labels so malignant=1 and benign=0 as this makes more sense for
        # calculating recall and precision
        1 - breast_cancer_dataset["target"].astype(int),
        ["benign", "malignant"],
        breast_cancer_dataset.feature_names,
    )

    if not as_frame:
        data = data.to_numpy()
        target = target.to_numpy()

    if return_X_y:
        return data, target

    breast_cancer_dataset = Bunch(
        data=data,
        target=target,
        feature_names=np.array(feature_names),
        frame=breast_cancer_dataset["data"].assign(target=target),
        target_names=np.array(target_names),
    )

    return breast_cancer_dataset


def load_california_housing(
    path: Path = temp_dir,
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch:
    """
    Load and return the california_housing dataset.

    Arguments
    ---------

    path: str or Path
        Path to the file where the data is stored or should be
        downloaded.
        Defaults to :code:`temp_dir`.

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    if not path.joinpath("cal_housing_py3.pkz").exists():
        print("Saving the data")

    california_housing_dataset = skd.fetch_california_housing(
        data_home=path, as_frame=True
    )

    feature_name_map = {
        "MedInc": "median income in block",
        "HouseAge": "median house age in block",
        "AveRooms": "average number of rooms in block",
        "AveBedrms": "average number of bedrooms in block",
        "Population": "block population",
        "AveOccup": "average house occupancy in block",
        "Latitude": "house block latitude",
        "Longitude": "house block longitude",
    }

    data, target, feature_names, target_names, frame = (
        california_housing_dataset["data"],
        california_housing_dataset["target"],
        california_housing_dataset["feature_names"],
        "medium house value (in $100k)",
        (
            california_housing_dataset["frame"]
            .rename(columns=feature_name_map)
            .rename(columns={"MedHouseVal": "medium house value (in $100k)"})
        ),
    )

    feature_names = [feature_name_map[fn] for fn in feature_names]

    if not as_frame:
        data = data.to_numpy()
        target = target.to_numpy()

    if return_X_y:
        return data, target

    california_housing_dataset = Bunch(
        data=data,
        target=target,
        feature_names=np.array(feature_names),
        target_names=np.array([target_names]),
        frame=frame,
    )

    return california_housing_dataset


def load_heart_disease(
    path: Path = temp_dir,
    return_X_y: bool = False,
    as_frame: bool = False,
):
    """
    Load and return the heart disease dataset: DOI: 10.24432/C52P4X

    Arguments
    ---------

    path: str or Path
        Path to the file where the data is stored or should be
        downloaded.
        Defaults to :code:`temp_dir`.

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    features_to_include = [
        "age",
        "sex",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "ca",
    ]

    data = load_and_save_uci_dataset(dataset_name="heart_disease", path=path)
    y = (data["target"] > 0).astype(int).squeeze()
    data = data.drop(columns=["target"])

    # selecting only the features we want
    data = data[features_to_include]

    # processing restecg - 0 = normal, 1 = abnormal
    data = data.assign(restecg=(data["restecg"] > 0).astype(int))

    # renaming features with descriptive names

    features_to_rename = {
        "age": "age",
        "sex": "sex (1 = male, 0 = female)",
        "trestbps": "resting blood pressure (on admission to the hospital)",
        "chol": "serum cholestoral in mg/dl",
        "fbs": "fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
        "restecg": "resting electrocardiographic results (1 = abnormal, 0 = normal)",
        "thalach": "maximum heart rate achieved",
        "exang": "exercise induced angina",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "ca": "number of major vessels (0 - 3) colored by flourosopy",
    }

    ## renaming the columns
    data = data.rename(columns=features_to_rename)

    # remove rows with missing values in targets or features
    missing = y.isnull().values | data.isnull().any(axis=1).values
    data = data[~missing]
    y = y[~missing]

    # collecting information to return
    data, target, frame, target_names, feature_names, categorical_features = (
        data,
        y,
        data.assign(target=y),
        np.array(["no heart disease", "heart disease"]),
        data.columns.to_numpy(),
        [1, 4, 5, 7, 9],
    )

    if not as_frame:
        data = data.to_numpy()
        target = y.to_numpy()

    if return_X_y:
        return data, target

    heart_disease_dataset = Bunch(
        data=data,
        target=target,
        feature_names=feature_names,
        frame=frame,
        target_names=target_names,
        categorical_features=categorical_features,
    )

    return heart_disease_dataset


def load_wine_quality(
    path: Path = temp_dir,
    return_X_y: bool = False,
    as_frame: bool = False,
):
    """
    Load and return the wine quality dataset: DOI: 10.24432/C56S3T

    Arguments
    ---------

    path: str or Path
        Path to the file where the data is stored or should be
        downloaded.
        Defaults to :code:`temp_dir`.

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    data = load_and_save_uci_dataset(dataset_name="wine_quality", path=path)
    # making sure targets are binary
    y = (data["target"] > 5).astype(int).squeeze()
    data = data.drop(columns=["target"])

    features_to_rename = {
        "fixed_acidity": "fixed acidity",
        "volatile_acidity": "volatile acidity",
        "citric_acid": "citric acid",
        "residual_sugar": "residual sugar",
        "chlorides": "chlorides",
        "free_sulfur_dioxide": "free sulfur dioxide",
        "total_sulfur_dioxide": "total sulfur dioxide",
        "density": "density",
        "pH": "pH",
        "sulphates": "sulphates",
        "alcohol": "alcohol",
    }

    data = data.rename(columns=features_to_rename)

    # remove rows with missing values in targets or features
    missing = y.isnull().values | data.isnull().any(axis=1).values
    data = data[~missing]
    y = y[~missing]

    # collecting information to return
    data, target, frame, target_names, feature_names = (
        data,
        y,
        data.assign(target=y),
        np.array(["bad quality", "good quality"]),
        data.columns.to_numpy(),
    )

    if not as_frame:
        data = data.to_numpy()
        target = y.to_numpy()

    if return_X_y:
        return data, target

    wine_quality_dataset = Bunch(
        data=data,
        target=target,
        feature_names=feature_names,
        frame=frame,
        target_names=target_names,
    )

    return wine_quality_dataset


def load_sk_diabetes(
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch:
    """
    Load and return the diabetes dataset
    (https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html).

    Arguments
    ---------

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    features_to_rename = {
        "age": "age",
        "sex": "sex",
        "bmi": "body mass index",
        "bp": "average blood pressure",
        "s1": "total serum cholesterol",
        "s2": "low-density lipoproteins",
        "s3": "high-density lipoproteins",
        "s4": "total cholesterol / HDL",
        "s5": "log of serum triglycerides level",
        "s6": "blood sugar level",
    }

    diabetes_dataset = skd.load_diabetes(as_frame=True, scaled=False)

    frame = diabetes_dataset.frame.rename(columns=features_to_rename)
    data = frame.drop(columns=["target"])
    target = (
        10 * (frame["target"] - 25) / (346 - 25)
    )  # range of possible y values -> 0 to 10

    data, target, target_names, feature_names = (
        data,
        target,
        [
            "quantitative measure of diabetes disease progression one year after baseline (0 to 10)"
        ],
        data.columns.tolist(),
    )

    if not as_frame:
        data = data.to_numpy()
        target = target.to_numpy()

    if return_X_y:
        return data, target

    diabetes_dataset = Bunch(
        data=data,
        target=target,
        feature_names=np.array(feature_names),
        frame=frame.assign(target=target),
        target_names=np.array(target_names),
    )

    return diabetes_dataset


# function to download thyroid dataset
def download_thyroid_dataset(path: Path = temp_dir) -> None:
    # download from url
    url = "https://archive.ics.uci.edu/static/public/102/thyroid+disease.zip"

    file_path = path.joinpath("thyroid.zip")

    urllib.request.urlretrieve(url, file_path)
    unzipped_path = path.joinpath("thyroid")

    # unzip
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(unzipped_path)

    # remove zip file
    file_path.unlink()

    return None


# function to process the raw data and save it as a parquet file
def process_raw_and_save_thyroid(path: Path = temp_dir) -> None:

    file_path = path.joinpath("thyroid/ann-train.data")

    with open(file_path, "r") as f:
        lines = f.readlines()

    data = []
    for line in lines:
        data.append(line.strip().split(" "))

    data = pd.DataFrame(
        data,
        columns=[
            "age",
            "sex",
            "patient on thyroxine",
            "maybe on thyroxine",
            "on antithyroid medication",
            "patient reports malaise",
            "patient pregnant",
            "history of thyroid surgery",
            "patient on I131 treatment",
            "maybe hypothyroid",
            "maybe hyperthyroid",
            "patient on lithium",
            "patient has goitre",
            "patient has tumour",
            "patient hypopituitary",
            "psychological symptoms",
            "thyroid-stimulating hormone value (TSH)",
            "triiodothyronine value (T3)",
            "total thyroxine value (TT4)",
            "thyroxine uptake value (T4U)",
            "free thyroxine index value (FTI)",
            "target",
        ],
    )

    data = data.astype(
        {
            "age": "float",
            "sex": "int",
            "patient on thyroxine": "int",
            "maybe on thyroxine": "int",
            "on antithyroid medication": "int",
            "patient reports malaise": "int",
            "patient pregnant": "int",
            "history of thyroid surgery": "int",
            "patient on I131 treatment": "int",
            "maybe hypothyroid": "int",
            "maybe hyperthyroid": "int",
            "patient on lithium": "int",
            "patient has goitre": "int",
            "patient has tumour": "int",
            "patient hypopituitary": "int",
            "psychological symptoms": "int",
            "thyroid-stimulating hormone value (TSH)": "float",
            "triiodothyronine value (T3)": "float",
            "total thyroxine value (TT4)": "float",
            "thyroxine uptake value (T4U)": "float",
            "free thyroxine index value (FTI)": "float",
            "target": "int",
        }
    )

    data.to_parquet(path.joinpath("thyroid_dataset_original.parquet"))

    data["age"] = (data["age"] * 100).astype("int")

    class_of_interest = 1  # hypothyroid
    class_normal = 3

    data = data[data["target"].isin([class_of_interest, class_normal])]
    data["target"] = (data["target"] == class_of_interest).astype(int)

    # balanced sample from the data
    data = (
        data.groupby("target", group_keys=False)
        .sample(75, replace=False, random_state=0)
        .reset_index(drop=True)
    )

    columns_to_use_for_prediction = [
        "thyroid-stimulating hormone value (TSH)",
        "triiodothyronine value (T3)",
        "total thyroxine value (TT4)",
        "thyroxine uptake value (T4U)",
        # "free thyroxine index value (FTI)",
    ]

    data = data[columns_to_use_for_prediction + ["target"]]

    # save the processed data as a single file
    data.to_parquet(path.joinpath("thyroid_dataset.parquet"))

    return None


# function to clean up the files
def clean_thyroid_data_files(path: Path = temp_dir) -> None:

    # remove the unzipped folder
    shutil.rmtree(path.joinpath("thyroid"))

    return None


# function to download and process the thyroid dataset
def download_and_process_raw_thyroid(path: Path = temp_dir) -> None:

    print("Downloading thyroid dataset")
    download_thyroid_dataset(path)

    print("Processing and saving thyroid dataset")
    process_raw_and_save_thyroid(path)

    print("Cleaning up the files")
    clean_thyroid_data_files(path)

    return None


# function to load the thyroid dataset
def load_hypothyroid(
    as_frame: bool = False,
    return_X_y: bool = False,
    path: Path = temp_dir,
) -> Bunch:
    """
    Load and return the hypothyroid dataset.

    data: https://archive.ics.uci.edu/dataset/102/thyroid+disease
    feature and target information: https://arxiv.org/abs/cs/9503102

    Arguments
    ---------

    path: str or Path
        Path to the file where the data is stored or should be
        downloaded.
        Defaults to :code:`temp_dir`.

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.

    """
    file_path = path.joinpath("thyroid_dataset.parquet")

    if not file_path.exists():
        download_and_process_raw_thyroid(path)

    thyroid_dataset = pd.read_parquet(file_path)

    data = thyroid_dataset.drop(columns="target")
    target = thyroid_dataset["target"]

    feature_names = data.columns.to_numpy()
    target_names = np.array(["normal", "hypothyroidism"])

    if not as_frame:
        data = data.to_numpy()
        target = target.to_numpy()

    if return_X_y:
        return data, target

    thyroid_dataset = Bunch(
        data=data,
        target=target,
        feature_names=np.array(feature_names),
        frame=thyroid_dataset,
        target_names=target_names,
    )

    return thyroid_dataset


def make_fake_data(
    y_fn: typing.Callable,
    n_samples: int,
    n_features: int,
    rng: np.random.Generator = np.random.default_rng(),
) -> pd.DataFrame:
    """
    This function generates fake data for testing purposes.

    Arguments
    ----------
    y_fn: callable
        A function that takes a 2D numpy array as input and returns a 1D numpy array.

    n_samples: int
        The number of samples to generate.

    n_features: int
        The number of features to generate.

    rng: np.random.Generator
        The random number generator to use.
        Defaults to :code:`np.random.default_rng()`.

    Returns
    -------
    X: pd.DataFrame
        The features.

    y: pd.Series
        The target.

    """
    X = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        columns=[f"feature {i}" for i in range(n_features)],
    )

    y = pd.Series(y_fn(X.values), name="target")

    return X, y


def save_fake_data(
    X: pd.DataFrame, y: pd.Series, path: Path = temp_dir
) -> None:
    """
    This function saves fake data to disk.
    This file will be saved in the directory
    specified by the path argument and
    called :code:`known_relationship.parquet`.

    Arguments
    ----------

    X: pd.DataFrame
        The features.

    y: pd.Series
        The target.

    path: str or Path
        The directory where the file should be saved.
        The file will be saved as :code:`known_relationship.parquet` in
        the specified directory.
        Defaults to :code:`temp_dir`.

    Returns
    -------

    None

    """

    path = path.joinpath("known_relationship.parquet")

    X.join(y).to_parquet(path, index=False)

    return None


def load_fake_data(
    path: Path = temp_dir,
    return_X_y: bool = False,
    as_frame: bool = False,
) -> Bunch:
    """
    Load and return the fake_data produced and saved using the
    functions :code:`make_fake_data` and :code:`save_fake_data`.

    Arguments
    ---------

    path: str or Path
        Path to the file where the data is stored or should be
        downloaded.
        Defaults to :code:`temp_dir`.

    return_X_y: bool
        Whether to return data and target as numpy arrays.
        Defaults to :code:`False`.

    as_frame: bool
        Whether to return the data as a pandas DataFrame.
        Defaults to :code:`False`.


    Returns
    -------

    Bunch
        Dictionary-like object, the interesting attributes are:
        'data', the data to learn, 'target', the classification target for each sample,
        'feature_names', the feature names, 'target_names', the target names,
        and 'frame', the pandas DataFrame.

    Tuple
        If :code:`return_X_y=True`, then a tuple of data and target.
        If :code:`as_frame=True`, then as pandas DataFrame otherwise as numpy arrays.


    """

    if Path(path).joinpath("known_relationship.parquet").exists():
        fake_data_dataset = pd.read_parquet(path.joinpath("known_relationship.parquet"))
    else:
        print("The fake data has not been generated yet. Generating it now.")
        rng = np.random.default_rng(42)
        
        # define the function that generates the target variable
        def y_fn(X):
            # y = 2 * x1 - 1 * x2 + 1 * x3 + noise
            true_y = 2 * X[:, 0] - 1 * X[:, 1] + X[:, 2]
            # some noise
            noise_y = 0.05 * rng.normal(size=(X.shape[0]))
            return true_y + noise_y


        # make fake data
        X, y = make_fake_data(y_fn, n_samples=250, n_features=3, rng=rng)

        # save the fake data
        save_fake_data(X, y)

        fake_data_dataset = pd.read_parquet(path.joinpath("known_relationship.parquet"))


    data = fake_data_dataset.drop(columns="target")
    target = fake_data_dataset["target"]

    feature_names = data.columns
    target_names = target.name

    if not as_frame:
        data = data.to_numpy()
        target = target.to_numpy()

    if return_X_y:
        return data, target

    fake_data_dataset = Bunch(
        data=data,
        target=target,
        feature_names=np.array(feature_names),
        frame=fake_data_dataset,
        target_names=np.array([target_names]),
    )

    return fake_data_dataset


def load_raw_dataset_frame(
    dataset_name: str = None,
    path: Path = temp_dir,
) -> pd.DataFrame:
    """
    Load the raw dataset, without any of the preprocessing done.
    This includes the original feature names and target names.

    This is helpful for assessing whether the language model has
    previously seen these datasets.

    Arguments
    ---------

    dataset_name: str
        The name of the dataset to load.
        This can be any of the following:
        - "fake_data"
        - "breast_cancer"
        - "diabetes"
        - "california_housing"
        - "diabetes_37"
        - "heart_disease"
        - "wine_quality"
        - "iris"
        - "wine"
        - "adult"
        - "hypothyroid"


    Returns
    -------

    pd.DataFrame
        The raw dataset.

    """

    if dataset_name == "fake_data":
        try:
            return pd.read_parquet(path.joinpath("known_relationship.parquet"))
        except FileNotFoundError:
            raise ValueError(
                """
                The fake data has not been generated yet. 
                Please run the make_fake_data.py script.
                """
            )

    elif dataset_name == "breast_cancer":
        return skd.load_breast_cancer(as_frame=True)["frame"]

    elif dataset_name == "diabetes":
        return skd.load_diabetes(as_frame=True, scaled=False)["frame"]

    elif dataset_name == "california_housing":
        return skd.fetch_california_housing(as_frame=True, data_home=path)["frame"]

    elif dataset_name == "diabetes_37":
        return skd.fetch_openml(data_id=37, as_frame=True, data_home=path)["frame"]

    # all of the following are loaded from the UCI repository
    elif dataset_name in [
        "heart_disease",
        "wine_quality",
        "iris",
        "wine",
        "adult",
    ]:
        return load_and_save_uci_original_dataset(dataset_name=dataset_name, path=path)

    elif dataset_name == "hypothyroid":
        return pd.read_parquet(path.joinpath("thyroid_dataset_original.parquet"))

    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

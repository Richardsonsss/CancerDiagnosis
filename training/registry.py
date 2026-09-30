"""Registry of the cancer types the system can learn, with their public datasets.

Every entry describes: how to recognise the request in a user prompt (English keywords;
prompts in other languages are translated to English first, see prompt_parser.py),
where the public dataset comes from and how to load it, the class list (in label
order) with display names, which classes count as malignant / tumour, and the
advice text shown by the diagnosis front end.

All datasets download without an account:
  skin        HAM10000 dermoscopy          Harvard Dataverse   CC BY-NC 4.0   2.8 GB
  skin_phone  PAD-UFES-20 smartphone       Mendeley Data       CC BY 4.0      3.6 GB
  breast      BreastMNIST ultrasound       Zenodo (MedMNIST+)  CC BY 4.0      31 MB
  colorectal  PathMNIST H&E histology      Zenodo (MedMNIST+)  CC BY 4.0      4.3 GB
  lung        LC25000 lung histology       Kaggle public API   CC BY 4.0      1.9 GB (zip incl. colon)
  brain       Brain Tumor MRI              Kaggle public API   CC0            165 MB
"""

TASKS = {
    "skin": {
        "name": "Skin cancer", "modality": "dermoscopic image",
        "keywords": ["skin", "melanoma", "basal cell", "cutaneous", "mole", "nevus", "nevi", "keratosis",
                     "dermoscop", "dermatoscop", "ham10000"],
        "dataset": {"type": "ham10000"},
        "classes": ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"],
        "class_names": {"akiec": "actinic keratosis / intraepithelial carcinoma", "bcc": "basal cell carcinoma",
                        "bkl": "benign keratosis-like lesion", "df": "dermatofibroma", "mel": "melanoma",
                        "nv": "melanocytic nevus (mole)", "vasc": "vascular lesion"},
        "malignant": ["akiec", "bcc", "mel"],
        "positive_term": "a malignant or pre-malignant lesion",  # how a malignant class is described
        "flips": "both",  # dermoscopic lesions have no canonical orientation
        "advice_malignant": "Please see a dermatologist soon; dermoscopy or a biopsy can confirm the finding.",
        "advice_benign": "The lesion appears benign. Keep an eye on it and see a doctor if its size, "
                         "colour or shape changes.",
        "input_hint": "Upload a dermoscopic photo of the lesion, taken in good light and in focus.",
        "notes": "Trained on dermoscopic images; ordinary phone photos differ from the training data.",
    },
    "skin_phone": {
        "name": "Skin cancer", "modality": "smartphone photo",
        "keywords": ["skin", "melanoma", "basal cell", "squamous", "cutaneous", "mole", "nevus", "nevi", "keratosis",
                     "lesion", "smartphone", "phone", "iphone", "camera", "photo", "picture", "clinical image"],
        "priority": 1,  # wins ties with "skin": generic skin prompts get the smartphone-photo model
        "dataset": {"type": "pad_ufes20"},
        "classes": ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"],
        "class_names": {"ACK": "actinic keratosis", "BCC": "basal cell carcinoma", "MEL": "melanoma",
                        "NEV": "nevus (mole)", "SCC": "squamous cell carcinoma", "SEK": "seborrheic keratosis"},
        "malignant": ["ACK", "BCC", "MEL", "SCC"],  # actinic keratosis is pre-malignant
        "positive_term": "a malignant or pre-malignant lesion",
        "flips": "both",
        "advice_malignant": "Please see a dermatologist soon; an examination or a biopsy can confirm the finding.",
        "advice_benign": "The lesion appears benign. Keep an eye on it and see a doctor if its size, "
                         "colour or shape changes.",
        "input_hint": "Take a close-up photo of the lesion with the phone camera: good light, in focus, "
                      "lesion in the centre.",
        "notes": "PAD-UFES-20: 2,298 smartphone photos of 1,641 lesions from 1,373 patients (Brazil); "
                 "58% of lesions biopsy-confirmed.",
    },
    "breast": {
        "name": "Breast cancer", "modality": "breast ultrasound image",
        "keywords": ["breast", "mammary", "mammogra"],
        "dataset": {"type": "medmnist", "name": "breastmnist", "size": 224},
        "classes": ["malignant", "benign_or_normal"],
        "class_names": {"malignant": "breast tumour", "benign_or_normal": "benign or normal breast tissue"},
        "malignant": ["malignant"],
        "positive_term": "a malignant lesion",
        "flips": "horizontal",
        "advice_malignant": "Please see a breast specialist soon; mammography, a repeat ultrasound or a "
                            "biopsy can confirm the finding.",
        "advice_benign": "The image appears benign or normal. Continue regular breast check-ups as advised "
                         "by your doctor.",
        "input_hint": "Upload a breast ultrasound image.",
        "notes": "BreastMNIST has only 780 images; expect limited reliability.",
    },
    "colorectal": {
        "name": "Colorectal cancer", "modality": "colorectal histology slide (H&E)",
        "keywords": ["colorectal", "colon", "rectal", "rectum", "bowel", "intestin"],
        "dataset": {"type": "medmnist", "name": "pathmnist", "size": 128},
        "classes": ["adipose", "background", "debris", "lymphocytes", "mucus", "smooth_muscle",
                    "normal_mucosa", "cancer_stroma", "adenocarcinoma"],
        "class_names": {"adipose": "adipose tissue", "background": "background (no tissue)", "debris": "necrotic debris",
                        "lymphocytes": "lymphocytes", "mucus": "mucus", "smooth_muscle": "smooth muscle",
                        "normal_mucosa": "normal colon mucosa", "cancer_stroma": "cancer-associated stroma",
                        "adenocarcinoma": "colorectal adenocarcinoma epithelium"},
        "malignant": ["cancer_stroma", "adenocarcinoma"],
        "positive_term": "cancer-related tissue",
        "flips": "both",
        "advice_malignant": "Features of cancer-related tissue are visible. A pathologist should review the "
                            "slide together with the clinical findings.",
        "advice_benign": "No clear features of cancer-related tissue are visible. The final diagnosis rests "
                         "with a pathologist.",
        "input_hint": "Upload an H&E-stained colorectal histology image (tissue patch).",
        "notes": "PathMNIST (NCT-CRC-HE-100K / CRC-VAL-HE-7K), 107,180 tissue patches.",
    },
    "lung": {
        "name": "Lung cancer", "modality": "lung histology slide (H&E)",
        "keywords": ["lung", "pulmonary"],
        "dataset": {"type": "kaggle_folders", "slug": "andrewmvd/lung-and-colon-cancer-histopathological-images",
                    "class_dirs": {"lung_aca": "adenocarcinoma", "lung_n": "benign", "lung_scc": "squamous_cell_carcinoma"},
                    "split": "random"},
        "classes": ["adenocarcinoma", "benign", "squamous_cell_carcinoma"],
        "class_names": {"adenocarcinoma": "lung adenocarcinoma", "benign": "benign lung tissue",
                        "squamous_cell_carcinoma": "lung squamous cell carcinoma"},
        "malignant": ["adenocarcinoma", "squamous_cell_carcinoma"],
        "positive_term": "a malignant lesion",
        "flips": "both",
        "advice_malignant": "Features of lung cancer are visible. A pathologist should review the slide "
                            "together with imaging and clinical findings.",
        "advice_benign": "The tissue appears benign. The final diagnosis rests with a pathologist.",
        "input_hint": "Upload an H&E-stained lung histology image.",
        "notes": "LC25000 was generated by augmenting 750 lung images, so a random split contains "
                 "near-duplicates and test scores are optimistic.",
    },
    "brain": {
        "name": "Brain tumour", "modality": "brain MRI image",
        "keywords": ["brain", "glioma", "meningioma", "pituitary", "glioblastoma", "cerebral", "intracranial"],
        "dataset": {"type": "kaggle_folders", "slug": "masoudnickparvar/brain-tumor-mri-dataset",
                    "class_dirs": {"glioma": "glioma", "meningioma": "meningioma", "notumor": "no_tumor",
                                   "pituitary": "pituitary"},
                    "split": "folders", "test_dir": "Testing"},
        "classes": ["glioma", "meningioma", "no_tumor", "pituitary"],
        "class_names": {"glioma": "glioma", "meningioma": "meningioma", "no_tumor": "no tumour",
                        "pituitary": "pituitary tumour"},
        "malignant": ["glioma", "meningioma", "pituitary"],  # "tumour present"
        "positive_term": "a tumour",
        "flips": "horizontal",
        "advice_malignant": "Signs of a possible tumour are visible. Please see a neurologist or neurosurgeon "
                            "soon; a radiologist should review the scan.",
        "advice_benign": "No clear signs of a tumour are visible. See a doctor if you have symptoms such as "
                         "headaches or changes in vision.",
        "input_hint": "Upload a brain MRI slice.",
        "notes": "Merges figshare, SARTAJ and Br35H; tumour type labels, not grades.",
    },
}


def task(task_id):
    if task_id not in TASKS:
        raise SystemExit(f"unknown task '{task_id}'. Supported: {', '.join(TASKS)}")
    return TASKS[task_id]


def malignant_indices(t):
    return [t["classes"].index(c) for c in t["malignant"]]

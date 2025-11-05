from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


NAME_LABELS: Dict[str, Tuple[str, str]] = {
    # White male
    "James": ("White", "Male"),
    "Robert": ("White", "Male"),
    "John": ("White", "Male"),
    "Michael": ("White", "Male"),
    "William": ("White", "Male"),
    "David": ("White", "Male"),
    "Richard": ("White", "Male"),
    "Charles": ("White", "Male"),
    "Thomas": ("White", "Male"),
    "Christopher": ("White", "Male"),
    "Daniel": ("White", "Male"),
    "Matthew": ("White", "Male"),
    "Anthony": ("White", "Male"),
    "Mark": ("White", "Male"),
    "Andrew": ("White", "Male"),
    "Joseph": ("White", "Male"),
    "Ryan": ("White", "Male"),
    "Brian": ("White", "Male"),
    "Jason": ("White", "Male"),
    "Kevin": ("White", "Male"),
    "Eric": ("White", "Male"),
    "Steven": ("White", "Male"),
    "Scott": ("White", "Male"),
    "Jeffrey": ("White", "Male"),
    "Patrick": ("White", "Male"),
    "Gregory": ("White", "Male"),
    "Peter": ("White", "Male"),
    "Benjamin": ("White", "Male"),
    "Nicholas": ("White", "Male"),
    "Jonathan": ("White", "Male"),
    "Timothy": ("White", "Male"),
    "Stephen": ("White", "Male"),
    "Zachary": ("White", "Male"),
    "Aaron": ("White", "Male"),
    "Adam": ("White", "Male"),
    "Bradley": ("White", "Male"),
    "Craig": ("White", "Male"),
    "Sean": ("White", "Male"),
    "Douglas": ("White", "Male"),
    "Kyle": ("White", "Male"),
    "Brandon": ("White", "Male"),
    "Justin": ("White", "Male"),
    "Cameron": ("White", "Male"),
    "Dylan": ("White", "Male"),
    "Ethan": ("White", "Male"),
    "Logan": ("White", "Male"),
    "Blake": ("White", "Male"),
    "Travis": ("White", "Male"),
    "Garrett": ("White", "Male"),
    "Colin": ("White", "Male"),
    "Luke": ("White", "Male"),
    # White female
    "Emily": ("White", "Female"),
    "Sarah": ("White", "Female"),
    "Jessica": ("White", "Female"),
    "Amanda": ("White", "Female"),
    "Jennifer": ("White", "Female"),
    "Ashley": ("White", "Female"),
    "Lauren": ("White", "Female"),
    "Megan": ("White", "Female"),
    "Hannah": ("White", "Female"),
    "Rachel": ("White", "Female"),
    "Nicole": ("White", "Female"),
    "Stephanie": ("White", "Female"),
    "Michelle": ("White", "Female"),
    "Brittany": ("White", "Female"),
    "Rebecca": ("White", "Female"),
    "Samantha": ("White", "Female"),
    "Elizabeth": ("White", "Female"),
    "Melissa": ("White", "Female"),
    "Amber": ("White", "Female"),
    "Heather": ("White", "Female"),
    "Erin": ("White", "Female"),
    "Kimberly": ("White", "Female"),
    "Christine": ("White", "Female"),
    "Lindsay": ("White", "Female"),
    "Danielle": ("White", "Female"),
    "Allison": ("White", "Female"),
    "Kaitlyn": ("White", "Female"),
    "Katherine": ("White", "Female"),
    "Courtney": ("White", "Female"),
    "Taylor": ("White", "Female"),
    "Olivia": ("White", "Female"),
    "Abigail": ("White", "Female"),
    "Natalie": ("White", "Female"),
    "Victoria": ("White", "Female"),
    "Morgan": ("White", "Female"),
    "Haley": ("White", "Female"),
    "Caitlin": ("White", "Female"),
    "Brianna": ("White", "Female"),
    "Kayla": ("White", "Female"),
    "Shelby": ("White", "Female"),
    "Alyssa": ("White", "Female"),
    "Chelsea": ("White", "Female"),
    "Anna": ("White", "Female"),
    "Faith": ("White", "Female"),
    "Julia": ("White", "Female"),
    "Brooke": ("White", "Female"),
    "Madison": ("White", "Female"),
    "Rosie": ("White", "Female"),
    "Chloe": ("White", "Female"),
    "Grace": ("White", "Female"),
    # Black male
    "Jamal": ("Black or African American", "Male"),
    "Darnell": ("Black or African American", "Male"),
    "Tyrone": ("Black or African American", "Male"),
    "DeShawn": ("Black or African American", "Male"),
    "Terrell": ("Black or African American", "Male"),
    "Malik": ("Black or African American", "Male"),
    "Demetrius": ("Black or African American", "Male"),
    "Maurice": ("Black or African American", "Male"),
    "Jerome": ("Black or African American", "Male"),
    "Darrell": ("Black or African American", "Male"),
    "Marlon": ("Black or African American", "Male"),
    "Lamont": ("Black or African American", "Male"),
    "Rasheed": ("Black or African American", "Male"),
    "Jermaine": ("Black or African American", "Male"),
    "Tremayne": ("Black or African American", "Male"),
    "Kareem": ("Black or African American", "Male"),
    "Reginald": ("Black or African American", "Male"),
    "Darius": ("Black or African American", "Male"),
    "Cedric": ("Black or African American", "Male"),
    "Tyriek": ("Black or African American", "Male"),
    "Marquis": ("Black or African American", "Male"),
    "DeAndre": ("Black or African American", "Male"),
    "Jalen": ("Black or African American", "Male"),
    "Antwan": ("Black or African American", "Male"),
    "Lamar": ("Black or African American", "Male"),
    "Tariq": ("Black or African American", "Male"),
    "Tyrell": ("Black or African American", "Male"),
    "Trevon": ("Black or African American", "Male"),
    "Kendrick": ("Black or African American", "Male"),
    "Andre": ("Black or African American", "Male"),
    "Donnell": ("Black or African American", "Male"),
    "Omari": ("Black or African American", "Male"),
    "Tyrese": ("Black or African American", "Male"),
    "Kwame": ("Black or African American", "Male"),
    "Rashad": ("Black or African American", "Male"),
    "Javon": ("Black or African American", "Male"),
    "Cornell": ("Black or African American", "Male"),
    "Jabari": ("Black or African American", "Male"),
    "Ahmad": ("Black or African American", "Male"),
    "Dante": ("Black or African American", "Male"),
    "Deonte": ("Black or African American", "Male"),
    "Kaleb": ("Black or African American", "Male"),
    "Tremaine": ("Black or African American", "Male"),
    "Isaiah": ("Black or African American", "Male"),
    "Devonte": ("Black or African American", "Male"),
    "Terrence": ("Black or African American", "Male"),
    "Micah": ("Black or African American", "Male"),
    "Marquez": ("Black or African American", "Male"),
    # Black female
    "Lakisha": ("Black or African American", "Female"),
    "Tanisha": ("Black or African American", "Female"),
    "Aisha": ("Black or African American", "Female"),
    "Keisha": ("Black or African American", "Female"),
    "Shanice": ("Black or African American", "Female"),
    "Latoya": ("Black or African American", "Female"),
    "Ebony": ("Black or African American", "Female"),
    "Tameka": ("Black or African American", "Female"),
    "Monique": ("Black or African American", "Female"),
    "Yolanda": ("Black or African American", "Female"),
    "Latasha": ("Black or African American", "Female"),
    "Tamika": ("Black or African American", "Female"),
    "Shaniqua": ("Black or African American", "Female"),
    "Aaliyah": ("Black or African American", "Female"),
    "Nia": ("Black or African American", "Female"),
    "Imani": ("Black or African American", "Female"),
    "Tyesha": ("Black or African American", "Female"),
    "Kenya": ("Black or African American", "Female"),
    "Precious": ("Black or African American", "Female"),
    "Kiara": ("Black or African American", "Female"),
    "Chantel": ("Black or African American", "Female"),
    "Tia": ("Black or African American", "Female"),
    "Ayana": ("Black or African American", "Female"),
    "Deja": ("Black or African American", "Female"),
    "Jasmine": ("Black or African American", "Female"),
    "Khadijah": ("Black or African American", "Female"),
    "Tierra": ("Black or African American", "Female"),
    "Brianna": ("Black or African American", "Female"),
    "Shania": ("Black or African American", "Female"),
    "Anaya": ("Black or African American", "Female"),
    "Dominique": ("Black or African American", "Female"),
    "Destiny": ("Black or African American", "Female"),
    "Naomi": ("Black or African American", "Female"),
    "Raven": ("Black or African American", "Female"),
    "Sierra": ("Black or African American", "Female"),
    "Makayla": ("Black or African American", "Female"),
    "Aliyah": ("Black or African American", "Female"),
    "Arielle": ("Black or African American", "Female"),
    "Diamond": ("Black or African American", "Female"),
    "Trinity": ("Black or African American", "Female"),
    "Zaria": ("Black or African American", "Female"),
    "Asia": ("Black or African American", "Female"),
    "Mya": ("Black or African American", "Female"),
    "Kiana": ("Black or African American", "Female"),
    "Shayla": ("Black or African American", "Female"),
    "Lashonda": ("Black or African American", "Female"),
    "Kendra": ("Black or African American", "Female"),
    "Tamara": ("Black or African American", "Female"),
}


# Helper metadata that documents the postcode proxies we inject later. Keeping it here makes
# the dataset output self-explanatory without searching old notes.
POSTCODE_METADATA: Dict[str, List[Dict[str, str]]] = {
    "white_neighbourhood": [
        {"postcode": "YO26 8BN", "area": "Upper Poppleton (York)", "dominant_ethnicity": "White British", "share_percent": "95%"},
        {"postcode": "YO26 6QJ", "area": "Nether Poppleton (York)", "dominant_ethnicity": "White British", "share_percent": "94%"},
        {"postcode": "NE20 9DR", "area": "Ponteland (Newcastle)", "dominant_ethnicity": "White British", "share_percent": "93%"},
        {"postcode": "NE20 9BT", "area": "Ponteland (Newcastle)", "dominant_ethnicity": "White British", "share_percent": "94%"},
        {"postcode": "BA2 7TL", "area": "Bath (Somerset)", "dominant_ethnicity": "White British", "share_percent": "94%"},
        {"postcode": "BA2 7UX", "area": "Bath (Somerset)", "dominant_ethnicity": "White British", "share_percent": "95%"},
        {"postcode": "DT2 8NQ", "area": "Dorchester (Dorset)", "dominant_ethnicity": "White British", "share_percent": "96%"},
        {"postcode": "DT2 8ND", "area": "Dorchester (Dorset)", "dominant_ethnicity": "White British", "share_percent": "95%"},
        {"postcode": "CA13 9PJ", "area": "Cockermouth (Cumbria)", "dominant_ethnicity": "White British", "share_percent": "97%"},
        {"postcode": "CA13 9JR", "area": "Cockermouth (Cumbria)", "dominant_ethnicity": "White British", "share_percent": "96%"},
    ],
    "brown_neighbourhood": [
        {"postcode": "BD8 9LH", "area": "Bradford (Manningham)", "dominant_ethnicity": "Pakistani", "share_percent": "73%"},
        {"postcode": "BD8 9HP", "area": "Bradford (Manningham)", "dominant_ethnicity": "Pakistani", "share_percent": "≈70%"},
        {"postcode": "BD8 9RR", "area": "Bradford (Manningham)", "dominant_ethnicity": "Pakistani", "share_percent": "70%"},
        {"postcode": "BD7 3HX", "area": "Bradford (Great Horton)", "dominant_ethnicity": "Pakistani", "share_percent": "62%"},
        {"postcode": "BD7 4PS", "area": "Bradford (Great Horton)", "dominant_ethnicity": "Pakistani", "share_percent": "59%"},
        {"postcode": "E1 5EQ", "area": "Whitechapel (London)", "dominant_ethnicity": "Bangladeshi", "share_percent": "≈60%"},
        {"postcode": "E1 5HA", "area": "Whitechapel (London)", "dominant_ethnicity": "Bangladeshi", "share_percent": "≈58%"},
        {"postcode": "E1 5EU", "area": "Whitechapel (London)", "dominant_ethnicity": "Bangladeshi", "share_percent": "≈57%"},
        {"postcode": "E1 5JE", "area": "Whitechapel (London)", "dominant_ethnicity": "Bangladeshi", "share_percent": "≈56%"},
        {"postcode": "E1 5EJ", "area": "Whitechapel (London)", "dominant_ethnicity": "Bangladeshi", "share_percent": "57%"},
    ],
}


WHITE_POSTCODES: List[str] = [
    entry["postcode"] for entry in POSTCODE_METADATA["white_neighbourhood"]
]

BROWN_POSTCODES: List[str] = [
    entry["postcode"] for entry in POSTCODE_METADATA["brown_neighbourhood"]
]


def _build_balanced_groups() -> Dict[Tuple[str, str], List[str]]:
    groups: Dict[Tuple[str, str], List[str]] = {}
    for name, combo in NAME_LABELS.items():
        groups.setdefault(combo, []).append(name)
    return groups


BALANCED_NAME_GROUPS: Dict[Tuple[str, str], List[str]] = _build_balanced_groups()
# Dictionary keys are unique, so any duplicate spellings collapse here. If you
# need intentional repeats, append them directly to BALANCED_NAME_GROUPS.


def _copy_with_updates(df: pd.DataFrame, updates: Dict[str, Iterable]) -> pd.DataFrame:
    copy = df.copy()
    for column, values in updates.items():
        copy[column] = values
    return copy


def _assign_names(names: List[str], count: int, rng: np.random.Generator) -> List[str]:
    if not names:
        raise ValueError("Name list cannot be empty.")
    if len(names) >= count:
        indices = rng.choice(len(names), size=count, replace=False)
        return [names[i] for i in indices]

    # Allow reuse if provided list is shorter than required count.
    expanded: List[str] = []
    pool = list(names)
    rng.shuffle(pool)
    while len(expanded) < count:
        expanded.extend(pool)
    return expanded[:count]


def _assign_codes(codes: List[str], count: int) -> List[str]:
    if not codes:
        raise ValueError("Postcode list cannot be empty.")
    repeats = (count + len(codes) - 1) // len(codes)
    expanded = (codes * repeats)[:count]
    return expanded


# --- Dataset builders -------------------------------------------------------


def create_race_proxy_dataset(
    base_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    count = len(base_df)

    combos: List[Tuple[str, str, str]] = [
        ("White", "Male", "white_male_proxy"),
        ("White", "Female", "white_female_proxy"),
        ("Black or African American", "Male", "black_male_proxy"),
        ("Black or African American", "Female", "black_female_proxy"),
    ]

    variants = []
    for race, sex, label in combos:
        names = _assign_names(BALANCED_NAME_GROUPS[(race, sex)], count, rng)
        variant = _copy_with_updates(
            base_df.drop(columns=["derived_race", "derived_sex"], errors="ignore"),
            {
                "applicant_name": names,
                "variant": [label] * count,
            },
        )
        variants.append(variant)

    dataset = pd.concat(variants, ignore_index=True)
    # We drop derived_race/derived_sex upstream; the name + variant columns carry the signal now.
    dataset["pair_id"] = dataset["applicant_id"]
    return dataset


def create_location_proxy_dataset(base_df: pd.DataFrame) -> pd.DataFrame:
    count = len(base_df)

    white_codes = _assign_codes(WHITE_POSTCODES, count)
    brown_codes = _assign_codes(BROWN_POSTCODES, count)

    base_trimmed = base_df.drop(columns=["derived_race", "derived_sex"], errors="ignore")

    white_variant = _copy_with_updates(
        base_trimmed,
        {
            "applicant_uk_postcode": white_codes,
            "variant": ["white_postcode_proxy"] * count,
            "proxy_group": ["white_neighbourhood"] * count,
        },
    )

    brown_variant = _copy_with_updates(
        base_trimmed,
        {
            "applicant_uk_postcode": brown_codes,
            "variant": ["brown_postcode_proxy"] * count,
            "proxy_group": ["brown_neighbourhood"] * count,
        },
    )

    dataset = pd.concat([white_variant, brown_variant], ignore_index=True)
    dataset["pair_id"] = dataset["applicant_id"]
    return dataset


def create_race_gender_explicit_dataset(base_df: pd.DataFrame) -> pd.DataFrame:
    combos: List[Tuple[str, str, str]] = [
        ("White", "Male", "white_male"),
        ("White", "Female", "white_female"),
        ("Black or African American", "Male", "black_male"),
        ("Black or African American", "Female", "black_female"),
    ]

    variants = []
    count = len(base_df)
    for race, sex, label in combos:
        variant = _copy_with_updates(
            base_df,
            {
                "derived_race": [race] * count,
                "derived_sex": [sex] * count,
                "variant": [label] * count,
            },
        )
        variants.append(variant)

    dataset = pd.concat(variants, ignore_index=True)
    dataset["pair_id"] = dataset["applicant_id"]
    return dataset


def build_datasets(base_df_50: pd.DataFrame, base_df_100: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    outputs: Dict[str, pd.DataFrame] = {}

    outputs["race_gender_explicit"] = create_race_gender_explicit_dataset(base_df_50)
    outputs["race_proxy_names"] = create_race_proxy_dataset(base_df_50)
    outputs["location_proxy_postcode"] = create_location_proxy_dataset(base_df_100)

    return outputs

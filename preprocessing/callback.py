from . import age as age_feature_engineering
from . import boolean as boolean_feature_engineering
from . import cabin as cabin_feature_engineering
from . import categorical as categorical_feature_engineering
from . import expenses as expenses_feature_engineering
from . import label as label_feature_engineering
from . import names as names_feature_engineering
from . import passenger_id as passenger_id_feature_engineering
from . import survival as survival_feature_engineering
from .util import preprocessing_fn_from_hamilton_modules

preprocessing_fn = preprocessing_fn_from_hamilton_modules(
    [
        age_feature_engineering,
        boolean_feature_engineering,
        cabin_feature_engineering,
        categorical_feature_engineering,
        expenses_feature_engineering,
        label_feature_engineering,
        names_feature_engineering,
        passenger_id_feature_engineering,
        survival_feature_engineering,
    ],
    [
        # passenger ID features
        "passenger_group_vocab",
        "passenger_id",
        # name features
        "last_name_vocab",
        "scaled_family_members_count",
        # Cabin features
        "cabin_deck_vocab",
        "cabin_side_vocab",
        "scaled_cabin_num",
        "bucketized_cabin_num",
        # Target label
        "transported",
        # Age features
        "scaled_age",
        "bucketized_age",
        "is_adult",
        # Boolean features
        "cryo_sleep",
        "vip",
        # Categorical features
        "home_planet_vocab",
        "destination_vocab",
        # Expenses features
        # food court
        "scaled_food_court",
        "bucketized_food_court",
        # room service
        "scaled_room_service",
        "bucketized_room_service",
        # shopping mall
        "scaled_shopping_mall",
        "bucketized_shopping_mall",
        # spa
        "scaled_spa",
        "bucketized_spa",
        # VR deck
        "scaled_vr_deck",
        "bucketized_vr_deck",
        # aggregates
        "scaled_total_expenses",
        "bucketized_total_expenses",
        # survival
        "family_survival",
        "cabin_survival",
    ],
)

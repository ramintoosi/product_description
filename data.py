import os
from typing import TypedDict

INSTRUCTION = """Create a Short Product description based on the provided ##PRODUCT BRAND NAME## and the image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

##PRODUCT BRAND NAME##: {brand_name}
"""

class Data(TypedDict):
    name: str
    brand: int
    image_path: str

import pandas as pd

def load_data(data_root: str) -> list[Data]:
    """
    Load data from csv file
    :param data_root: data root folder where data.csv and images are stored
    :return: list of dictionaries, keys are column names and values are data
    """
    data = pd.read_csv(os.path.join(data_root, 'data.csv'))
    data.drop(columns=['site_category', 'supply_category'], inplace=True)
    clean_data(data)
    return data.to_dict(orient='records')

def clean_data(data: pd.DataFrame):
    """
    Clean data. remove code and model number
    :param data: data frame
    :return: cleaned data
    """

    # check for uniqueness of rows
    data.drop_duplicates(inplace=True)

    # remove code and model
    pattern = r'\b(کد|مدل)\b(\s+[A-Za-z0-9]+)?|\bمجموعه\b\s+(\d+)\s+(\w+)'
    data["name"] = data["name"].str.replace(pattern, '', regex=True)

    # Second regex pattern for any remaining standalone alphanumeric codes in English
    pattern2 = r'\b[A-Za-z0-9_-]+\b'
    data["name"] = data["name"].str.replace(pattern2, '', regex=True)

    # we probably need more cleaning, but it's ok for the demo




def convert_to_conversation(sample: dict):
    """
    Convert sample to conversation format
    :param sample: sample data consisting of name (description), brand, image_path
    :return:
    """
    image_path = sample["image_path"].replace('dataset/', '')
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : INSTRUCTION.format(brand_name=sample["brand"])},
              # use file:// for loading images each time instead of caching for memory efficiency
            {"type" : "image", "image_url" : f'file://{image_path}'} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["name"]} ]
        },
    ]
    return { "messages" : conversation }

def get_converted_data(root: str) -> list[dict]:
    """
    Get converted data
    :param root: root folder where data.csv and images are stored
    :return: converted data
    """
    data = load_data(root)
    return [convert_to_conversation(sample) for sample in data]
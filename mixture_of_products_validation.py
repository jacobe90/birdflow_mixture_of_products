import math
from mixture_of_products_model import forecast, get_forecast_prob

"""
Arguments:
masks: list of boolean masks for every week, False for cells with 0 density that week, true for all others

Returns:
conversion_dict: conversion_dict[i][j] := index of st week i, cell j in dynamic mask
"""
def to_dynamic_conversion_dict(masks):
    conversion_dict = {}
    for idx, mask in enumerate(masks):
        conversion_dict[idx] = {}
        count = 0
        for j, b in enumerate(mask):
            if b:
                conversion_dict[idx][j] = count
                count += 1
    return conversion_dict

"""
Arguments:
params: mixture of products parameters
track_df: pandas dataframe of track data

Returns:
Average log likelihood of a transition observed in the track data
"""
def track_log_likelihood(params, track_df, conversion_dict):
    total = 0
    count = 0
    for i in range(len(track_df)):
        if i % 20 == 0:
            print(f"row {i} of {len(track_df)}")
        try:
            week_1 = int(track_df['st_week.1'][i])
            week_2 = int(track_df['st_week.2'][i])
            cell_1 = conversion_dict[int(track_df['st_week.1'][i])][int(track_df['cell.1'][i])]  # convert to indices in dynamic mask
            cell_2 = conversion_dict[int(track_df['st_week.2'][i])][int(track_df['cell.2'][i])]
            transition_prob = get_forecast_prob(params, [(week_2, cell_2)], [(week_1, cell_1)])
            total += math.log(transition_prob)
            count += 1
        except Exception as e:
            print(e)


    average_log_likelihood = total / count
    return average_log_likelihood


def model_desirability(params):
    pass
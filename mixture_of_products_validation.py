import math
from mixture_of_products_model import forecast


"""
Arguments:
params: mixture of products parameters
track_df: pandas dataframe of track data

Returns:
Average log likelihood of a transition observed in the track data
"""
def track_log_likelihood(params, track_df):
    total = 0
    count = 0
    for i in range(len(track_df)):
        try:
            transition_probs = forecast(params, [int(track_df['st_week.2'])], [(int(track_df['st_week.1']), int(track_df['cell.1']))])
            total += math.log(transition_probs[int(track_df['cell.2'])])
            count += 1
        except:
            continue
    average_log_likelihood = total / count
    return average_log_likelihood


def model_desirability(params):
    pass
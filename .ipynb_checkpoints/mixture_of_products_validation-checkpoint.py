import math
from mixture_of_products_model import forecast, get_forecast_prob


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
        if i % 20 == 0:
            print(f"row {i} of {len(track_df)}")
        # transition_probs = forecast(params, [int(track_df['st_week.2'][i])], [(int(track_df['st_week.1'][i]), int(track_df['cell.1'][i]))])
        # total += math.log(transition_probs[int(track_df['cell.2'][i])])
        # count += 1
        try:
            transition_prob = get_forecast_prob(params, [(int(track_df['st_week.2'][i]), int(track_df['cell.2'][i]))], [(int(track_df['st_week.1'][i]), int(track_df['cell.1'][i]))])
            total += transition_prob
            count += 1
        except ValueError:
            print("ran into a nan.")
    average_log_likelihood = total / count
    return average_log_likelihood


def model_desirability(params):
    pass
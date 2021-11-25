from scipy.stats import chisquare
import pandas as pd


def get_males(data):
    if isinstance(data, pd.DataFrame):
        return data["Gender"] == 0
    else:
        return data == 0


def get_females(data):
    if isinstance(data, pd.DataFrame):
        return data["Gender"] == 1
    else:
        return data == 1


def get_true_genders(true_data):
    true_males = get_males(true_data)
    true_females = get_females(true_data)

    return true_males, true_females


def get_predicted_genders(predictions):
    predicted_males = get_males(predictions)
    predicted_females = get_females(predictions)

    return predicted_males, predicted_females


def compute_statistical_parity(true_data, predictions):
    n = len(predictions)

    observed_f1 = true_data[get_females(true_data) & get_females(predictions)].shape[0]
    observed_f0 = true_data[get_females(true_data) & get_males(predictions)].shape[0]
    observed_m1 = true_data[get_males(true_data) & get_females(predictions)].shape[0]
    observed_m0 = true_data[get_males(true_data) & get_males(predictions)].shape[0]

    expected_f1 = (
        true_data[get_females(true_data)].shape[0] * get_females(predictions).sum() / n
    )
    expected_f0 = (
        true_data[get_females(true_data)].shape[0] * get_males(predictions).sum() / n
    )
    expected_m1 = (
        true_data[get_males(true_data)].shape[0] * get_females(predictions).sum() / n
    )
    expected_m0 = (
        true_data[get_males(true_data)].shape[0] * get_males(predictions).sum() / n
    )

    observed_frequencies = [observed_f1, observed_f0, observed_m1, observed_m0]

    expected_frequencies = [expected_f1, expected_f0, expected_m1, expected_m0]

    return chisquare(observed_frequencies, f_exp=expected_frequencies)


def compute_conditional_statistical_parity(true_data, predictions, groups):
    true_males, true_females = get_true_genders(true_data)
    predicted_males, predicted_females = get_predicted_genders(predictions)
    p_val_cond_stat_par = 0

    for group in groups:
        individuals_in_group = true_data["Group"] == group
        n = true_data[individuals_in_group].shape[0]

        observed_f1 = true_data[
            true_females & predicted_females & individuals_in_group
        ].shape[0]
        observed_f0 = true_data[
            true_females & predicted_males & individuals_in_group
        ].shape[0]
        observed_m1 = true_data[
            true_males & predicted_females & individuals_in_group
        ].shape[0]
        observed_m0 = true_data[
            true_males & predicted_males & individuals_in_group
        ].shape[0]

        expected_f1 = (
            true_data[true_females & individuals_in_group].shape[0]
            * true_data[predicted_females & individuals_in_group].shape[0]
            / n
        )
        expected_f0 = (
            true_data[true_females & individuals_in_group].shape[0]
            * true_data[predicted_males & individuals_in_group].shape[0]
            / n
        )
        expected_m1 = (
            true_data[true_males & individuals_in_group].shape[0]
            * true_data[predicted_females & individuals_in_group].shape[0]
            / n
        )
        expected_m0 = (
            true_data[true_males & individuals_in_group].shape[0]
            * true_data[predicted_males & individuals_in_group].shape[0]
            / n
        )

        observed_frequencies = [observed_f1, observed_f0, observed_m1, observed_m0]
        expected_frequencies = [expected_f1, expected_f0, expected_m1, expected_m0]

        p_val_cond_stat_par += chisquare(
            observed_frequencies, f_exp=expected_frequencies
        )[0]

    return p_val_cond_stat_par


def compute_equalized_odds(true_data, predictions, y_vals):
    true_males, true_females = get_true_genders(true_data)
    predicted_males, predicted_females = get_predicted_genders(predictions)
    p_val_eq_odds = 0

    for val in y_vals:
        individuals_y_val = true_data["CreditRisk (y)"] == val
        n = true_data[individuals_y_val].shape[0]

        observed_f1 = true_data[
            true_females & predicted_females & individuals_y_val
        ].shape[0]
        observed_f0 = true_data[
            true_females & predicted_males & individuals_y_val
        ].shape[0]
        observed_m1 = true_data[
            true_males & predicted_females & individuals_y_val
        ].shape[0]
        observed_m0 = true_data[true_males & predicted_males & individuals_y_val].shape[
            0
        ]

        expected_f1 = (
            true_data[true_females & individuals_y_val].shape[0]
            * true_data[predicted_females & individuals_y_val].shape[0]
            / n
        )
        expected_f0 = (
            true_data[true_females & individuals_y_val].shape[0]
            * true_data[predicted_males & individuals_y_val].shape[0]
            / n
        )
        expected_m1 = (
            true_data[true_males & individuals_y_val].shape[0]
            * true_data[predicted_females & individuals_y_val].shape[0]
            / n
        )
        expected_m0 = (
            true_data[true_males & individuals_y_val].shape[0]
            * true_data[predicted_males & individuals_y_val].shape[0]
            / n
        )

        observed_frequencies = [observed_f1, observed_f0, observed_m1, observed_m0]
        expected_frequencies = [expected_f1, expected_f0, expected_m1, expected_m0]

        p_val = chisquare(observed_frequencies, f_exp=expected_frequencies)[0]
        p_val_eq_odds += p_val

    return p_val_eq_odds

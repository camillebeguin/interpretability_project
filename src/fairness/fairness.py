import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Markdown
from scipy.stats import chisquare

LITERAL_GENDER_CODES = {
    "0": "Female",
    "1": "Male",
}


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

    true_males, true_females = get_true_genders(true_data)
    predicted_males, predicted_females = get_predicted_genders(predictions)

    observed_f1 = true_data[true_females & predicted_females].shape[0]
    observed_f0 = true_data[true_females & predicted_males].shape[0]
    observed_m1 = true_data[true_males & predicted_females].shape[0]
    observed_m0 = true_data[true_males & predicted_males].shape[0]

    expected_f1 = true_data[true_females].shape[0] * predicted_females.sum() / n
    expected_f0 = true_data[true_females].shape[0] * predicted_males.sum() / n
    expected_m1 = true_data[true_males].shape[0] * predicted_females.sum() / n
    expected_m0 = true_data[true_males].shape[0] * predicted_males.sum() / n

    observed_frequencies = [observed_f1, observed_f0, observed_m1, observed_m0]
    expected_frequencies = [expected_f1, expected_f0, expected_m1, expected_m0]

    res = chisquare(observed_frequencies, f_exp=expected_frequencies)

    display(Markdown(f"Computed statistical parity: {res[0]}. p-value: {res[1]}."))


def compute_conditional_statistical_parity(true_data, predictions, groups):
    true_males, true_females = get_true_genders(true_data)
    predicted_males, predicted_females = get_predicted_genders(predictions)
    results = dict()

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

        res = chisquare(observed_frequencies, f_exp=expected_frequencies)
        results[group] = {"statistic": res[0], "p-value": res[1]}

    index = map(lambda code: LITERAL_GENDER_CODES[str(code)], results.keys())
    df_results = pd.DataFrame(data=results.values(), index=index)
    df_results = df_results.round(2)
    display(df_results)
    display(Markdown(f"Summed statistics: {df_results['statistic'].sum()}"))
    display(Markdown(f"Summed p-values: {df_results['p-value'].sum()}"))


def compute_equalized_odds(true_data, predictions, y_vals):
    true_males, true_females = get_true_genders(true_data)
    predicted_males, predicted_females = get_predicted_genders(predictions)
    results = dict()

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

        res = chisquare(observed_frequencies, f_exp=expected_frequencies)
        results[val] = {"statistic": res[0], "p-value": res[1]}

    df_results = pd.DataFrame(data=results.values(), index=results.keys())
    df_results = df_results.round(2)
    display(df_results)
    display(Markdown(f"Summed statistics: {df_results['statistic'].sum()}"))
    display(Markdown(f"Summed p-values: {df_results['p-value'].sum()}"))


def make_fpdp(
    data,
    X_test,
    best_estimator,
    predictions,
    chosen_column,
):
    min_val = X_test[chosen_column].min()
    max_val = X_test[chosen_column].max()
    range_column = list(range(min_val, max_val))

    statistics = []
    for val in range_column:
        X_test_copy = X_test.copy()
        X_test_copy[chosen_column] = val

        predictions = best_estimator.predict(X_test_copy)
        true_data = data.loc[X_test_copy.index, :]

        true_males, true_females = get_true_genders(true_data)
        predicted_males, predicted_females = get_predicted_genders(predictions)

        observed_f1 = true_data[true_females & predicted_females].shape[0]
        observed_f0 = true_data[true_females & predicted_males].shape[0]
        observed_m1 = true_data[true_males & predicted_females].shape[0]
        observed_m0 = true_data[true_males & predicted_males].shape[0]

        n = len(predictions)
        expected_f1 = true_data[true_females].shape[0] * predicted_females.sum() / n
        expected_f0 = true_data[true_females].shape[0] * predicted_males.sum() / n
        expected_m1 = true_data[true_males].shape[0] * predicted_females.sum() / n
        expected_m0 = true_data[true_males].shape[0] * predicted_males.sum() / n

        observed_frequencies = [observed_f1, observed_f0, observed_m1, observed_m0]
        expected_frequencies = [expected_f1, expected_f0, expected_m1, expected_m0]

        p_val = chisquare(
            observed_frequencies,
            f_exp=expected_frequencies,
        )[0]
        statistics.append(p_val)

    plt.figure(figsize=(12, 12))
    plt.scatter(range_column, statistics)
    plt.xlabel(f"{chosen_column}")
    plt.ylabel("Statistical parity")
    plt.show()

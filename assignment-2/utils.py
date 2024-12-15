"""
This module provides functions to create interactive visualisations of power consumption data 
and Hidden Markov Model (HMM) analysis results.
"""

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import graphviz

def plot_timeseries(dishwasher, freezer, fridge):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        subplot_titles=("Dishwasher", "Freezer", "Fridge"),
        vertical_spacing=0.07,
    )

    fig.add_trace(
        px.line(dishwasher[1050:1290], title="Dishwasher")
        .data[0]
        .update(line=dict(color="darkblue")),
        row=1,
        col=1,
    )
    fig.add_trace(
        px.line(freezer[:2000], title="Freezer")
        .data[0]
        .update(line=dict(color="salmon")),
        row=2,
        col=1,
    )
    fig.add_trace(
        px.line(fridge[:500], title="Fridge").data[0].update(line=dict(color="green")),
        row=3,
        col=1,
    )

    fig.update_layout(height=900, width=800, title_text="Appliance Power Consumption")
    fig.update_xaxes(title_text="Time Index", row=3, col=1)
    fig.update_yaxes(title_text="Power Consumption (W)", row=2, col=1)
    fig.update_layout(showlegend=False)
    return fig


def plot_disaggregation_results(predictions, actual, resample_freq=None):
    """
    Create interactive plots comparing actual vs predicted values, including total power consumption.
    """

    if resample_freq:
        predictions = predictions.resample(resample_freq).mean()
        actual = actual.resample(resample_freq).mean()

    # Calculate total power for both actual and predicted
    total_actual = actual.sum(axis=1)
    total_predicted = predictions.sum(axis=1)

    # Create subplots with an additional row for total power
    fig = make_subplots(
        rows=len(predictions.columns) + 1,  # Add one more row for total power
        cols=1,
        subplot_titles=["Total Power Consumption"]
        + [f"{app.capitalize()} Power Consumption" for app in predictions.columns],
        vertical_spacing=0.12,
    )

    # Add traces for total power
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=total_actual,
            name="Actual",
            mode="lines",
            line=dict(color="blue", width=1),
            opacity=0.7,
            legendgroup="actual",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=total_predicted,
            name="Model",
            mode="lines",
            line=dict(color="red", width=1),
            opacity=0.7,
            legendgroup="model",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Add traces for each appliance
    for idx, appliance in enumerate(
        predictions.columns, 2
    ):  # Start from 2 as row 1 is total power
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual[appliance],
                name="Actual",
                mode="lines",
                line=dict(color="blue", width=1),
                opacity=0.7,
                legendgroup="actual",
                showlegend=False,
            ),
            row=idx,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions[appliance],
                name="Model",
                mode="lines",
                line=dict(color="red", width=1),
                opacity=0.7,
                legendgroup="model",
                showlegend=False,
            ),
            row=idx,
            col=1,
        )

    fig.update_layout(
        height=300 * (len(predictions.columns) + 1),
        title_text="Appliance Power Consumption",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            itemsizing="constant",
            itemwidth=30,
            traceorder="normal",
        ),
        legend_tracegroupgap=40,
    )

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Power (W)")

    return fig


def plot_daily_patterns(predictions, actual):
    """Create interactive plots of average daily patterns, including total power"""

    # Calculate hourly averages
    actual_hourly = actual.groupby(actual.index.hour).mean()
    pred_hourly = predictions.groupby(predictions.index.hour).mean()

    # Calculate total power hourly averages
    total_actual_hourly = actual_hourly.sum(axis=1)
    total_pred_hourly = pred_hourly.sum(axis=1)

    # Create subplots with an additional row for total power
    fig = make_subplots(
        rows=len(predictions.columns) + 1,  # Add one more row for total power
        cols=1,
        subplot_titles=["Total Power Average Daily Pattern"]
        + [f"{app.capitalize()} Average Daily Pattern" for app in predictions.columns],
        vertical_spacing=0.1,
    )

    # Add traces for total power
    fig.add_trace(
        go.Scatter(
            x=actual_hourly.index,
            y=total_actual_hourly,
            name="Actual",
            mode="lines",
            line=dict(color="blue", width=2),
            legendgroup="actual",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=pred_hourly.index,
            y=total_pred_hourly,
            name="Model",
            mode="lines",
            line=dict(color="red", width=2),
            legendgroup="model",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Add traces for each appliance
    for idx, appliance in enumerate(
        predictions.columns, 2
    ):  # Start from 2 as row 1 is total power
        fig.add_trace(
            go.Scatter(
                x=actual_hourly.index,
                y=actual_hourly[appliance],
                name="Actual",
                mode="lines",
                line=dict(color="blue", width=2),
                legendgroup="actual",
                showlegend=False,
            ),
            row=idx,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=pred_hourly.index,
                y=pred_hourly[appliance],
                name="Predicted",
                mode="lines",
                line=dict(color="red", width=2),
                legendgroup="model",
                showlegend=False,
            ),
            row=idx,
            col=1,
        )

    fig.update_layout(
        height=300 * (len(predictions.columns) + 1),
        title_text="Average Daily Power Consumption Patterns",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01,
            itemsizing="constant",
            itemwidth=30,
            traceorder="normal",
        ),
        legend_tracegroupgap=40,
    )

    fig.update_xaxes(title_text="Hour of Day", dtick=1)
    fig.update_yaxes(title_text="Average Power (W)")

    return fig


def visualise_individual_hmm(fhmm, appliance, min_prob=0.01, output_file=None):
    """Visualise an individual HMM for a specific appliance"""
    params = fhmm.get_hmm_parameters(appliance)

    # Create graph
    dot = graphviz.Digraph(comment=f"{appliance} HMM")
    dot.attr(rankdir="LR")

    # Add start state
    dot.node("start", "Start", shape="circle", style="filled")

    # Add states with fixed size
    n_states = len(params["means"])
    for i in range(n_states):
        state_label = f'S{i}\n{params["means"][i]:.1f}W'
        dot.node(f"s{i}", state_label, shape="circle", width="1.5", height="1.5")

        # Add initial transitions from start state
        if params["startprob"][i] > min_prob:
            dot.edge("start", f"s{i}", label=f'{params["startprob"][i]:.2f}')

    # Add transitions between states
    for i in range(n_states):
        for j in range(n_states):
            prob = params["transmat"][i, j]
            if prob > min_prob:
                dot.edge(f"s{i}", f"s{j}", label=f"{prob:.2f}")

    # Save to file if output_file is provided
    if output_file:
        # Get format from file extension
        format_type = output_file.split(".")[-1]
        dot.render(output_file.rsplit(".", 1)[0], format=format_type, cleanup=True)

    return dot


def visualise_combined_hmm(fhmm, min_prob=0.01, output_file=None):
    """Visualise the combined FHMM model"""
    params = fhmm.get_combined_parameters()

    # Create graph
    dot = graphviz.Digraph(comment="Combined FHMM")
    dot.attr(rankdir="LR")

    # Add start state
    dot.node("start", "Start", shape="circle", style="filled")

    # Add all states
    n_states = len(params["means"])
    for i in range(n_states):
        # Create state label with appliance states
        state_combo = params["state_combinations"][i]
        label_parts = [
            f"{app}: S{state}" for app, state in zip(params["appliances"], state_combo)
        ]
        state_label = f"S{i}\n" + "\n".join(label_parts)
        state_label += f'\n{params["means"][i]:.1f}W'

        dot.node(f"s{i}", state_label, shape="circle", width="1.5", height="1.5")

        # Add initial transitions from start state
        if params["startprob"][i] > min_prob:
            dot.edge("start", f"s{i}", label=f'{params["startprob"][i]:.2f}')

    # Add transitions between states
    for i in range(n_states):
        for j in range(n_states):
            prob = params["transmat"][i, j]
            if prob > min_prob:
                dot.edge(f"s{i}", f"s{j}", label=f"{prob:.2f}")

    # Save to file if output_file is provided
    if output_file:
        # Get format from file extension
        format_type = output_file.split(".")[-1]
        dot.render(output_file.rsplit(".", 1)[0], format=format_type, cleanup=True)

    return dot


def visualise_viterbi_trellis(
    fhmm,
    observations: np.ndarray,
    start_idx: int = 0,
    window_size: int = 5,
    min_prob: float = 0.01,
    top_n: int = None,
    output_file: str = None,
) -> graphviz.Digraph:
    """
    Visualise the Viterbi trellis diagram for a sequence of observations

    Parameters:
    -----------
    fhmm : FHMM
        Trained FHMM model
    observations : np.ndarray
        Array of observed total power values
    start_idx : int
        Starting index in the observations array. If 0, uses initial probabilities.
        If >0, uses transition probabilities from previous state.
    window_size : int
        Number of timesteps to visualise
    min_prob : float
        Minimum probability to show transitions
    top_n : int, optional
        Number of top states to show at each timestep. If None, show all states.
        The actual path states will always be shown.
    output_file : str, optional
        File to save the visualisation to

    Returns:
    --------
    graphviz.Digraph
        The generated graph
    """
    # Create graph
    dot = graphviz.Digraph(comment="Viterbi Trellis")
    dot.attr(rankdir="LR")

    # Get a window of observations and predict states
    # Include one extra observation before the window if not starting from beginning
    pred_start = max(0, start_idx - 1)
    obs_window = observations[pred_start : start_idx + window_size]
    states, _ = fhmm._predict_states(obs_window)

    # If we started from middle, remove the first state as it was just for initialisation
    if start_idx > 0:
        states = states[1:]
        obs_window = obs_window[1:]

    # Calculate log emissions for the window
    diff = obs_window[:, None] - fhmm.means
    log_emissions = -0.5 * (np.log(2 * np.pi * fhmm.covars) + (diff**2) / fhmm.covars)

    # Convert log probabilities to regular probabilities and normalise
    # This is the numerically stable softmax function
    emissions = np.exp(log_emissions - np.max(log_emissions, axis=1, keepdims=True))
    emissions /= np.sum(emissions, axis=1, keepdims=True)
    transitions = np.exp(fhmm.log_A)

    n_timesteps = len(obs_window)
    n_states = len(fhmm.means)

    # Create state labels
    state_labels = {}
    for state in range(n_states):
        decoded = fhmm.decode_state_number(state)
        label_parts = [f"{app}: S{s}" for app, s in decoded.items()]
        state_labels[state] = "\n".join(label_parts)

    dot.attr("node", shape="circle")

    # Create timestep labels
    for t in range(n_timesteps):
        dot.node(f"t{t}", f"t={start_idx + t}\n{obs_window[t]:.1f}", shape="none")

    # For each timestep, determine which states to show
    visible_states = {}
    for t in range(n_timesteps):
        # Get indices of top N states by emission probability
        if top_n is not None:
            top_indices = np.argsort(emissions[t])[::-1][:top_n]
            # Always include the actual state if it's not in top N
            if states[t] not in top_indices:
                top_indices = np.append(top_indices, states[t])
            visible_states[t] = set(top_indices)
        else:
            visible_states[t] = set(range(n_states))

    # For each timestep's states
    for t in range(n_timesteps):
        with dot.subgraph() as s:
            s.attr(rank="same")
            s.node(f"t{t}", "", shape="none", width="0", height="0")

            for state in visible_states[t]:
                is_true_state = states[t] == state
                node_attrs = {
                    "shape": "circle",
                    "style": "filled",
                    "fillcolor": (
                        "#e6f3ff" if is_true_state else "white"
                    ),  # Light blue for selected path
                }

                # Create node label exactly like in the image
                emission_prob = emissions[t, state]
                state_power = fhmm.means[state]
                decoded = fhmm.decode_state_number(state)

                dishwasher_map = {0: "idle", 1: "job", 2: "mini-job", 3: "switch"}

                freezer_map = {0: "idle", 1: "switch", 2: "on"}

                fridge_map = {0: "off", 1: "on", 2: "switch"}

                decoded["dishwasher"] = dishwasher_map[int(decoded["dishwasher"])]
                decoded["freezer"] = freezer_map[int(decoded["freezer"])]
                decoded["fridge"] = fridge_map[int(decoded["fridge"])]

                label = f"State {state}\n"
                label += f"dishwasher:{decoded['dishwasher']}\n"
                label += f"freezer:{decoded['freezer']}\n"
                label += f"fridge:{decoded['fridge']}\n\n"
                label += f"Observed: {state_power:.1f}W\n"
                label += f"P={emission_prob:.4f}"

                dot.node(f"s{t}_{state}", label, **node_attrs)

    # Handle start node
    if start_idx == 0:
        start_probs = np.exp(fhmm.log_pi)
        dot.node("start", "Start", shape="circle", style="filled")

        for state in visible_states[0]:
            if start_probs[state] > min_prob:
                dot.edge("start", f"s0_{state}", label=f"{start_probs[state]:.4f}")
    else:
        prev_state = states[0]
        dot.node("prev", f"State {prev_state}", shape="circle", style="filled")

        for state in visible_states[0]:
            prob = transitions[prev_state, state]
            if prob > min_prob:
                dot.edge("prev", f"s0_{state}", label=f"{prob:.4f}")

    # Add transitions between states
    for t in range(n_timesteps - 1):
        for i in visible_states[t]:
            for j in visible_states[t + 1]:
                prob = transitions[i, j]
                if prob > min_prob:
                    is_actual_path = states[t] == i and states[t + 1] == j
                    if is_actual_path or prob > min_prob:
                        edge_attrs = {
                            "color": "red" if is_actual_path else "black",
                            "penwidth": "2" if is_actual_path else "1",
                        }
                        dot.edge(
                            f"s{t}_{i}",
                            f"s{t+1}_{j}",
                            label=f"{prob:.4f}",
                            **edge_attrs,
                        )

    # Add invisible edges for layout
    for t in range(n_timesteps - 1):
        dot.edge(f"t{t}", f"t{t+1}", style="invis")

    if output_file:
        format_type = output_file.split(".")[-1]
        dot.render(output_file.rsplit(".", 1)[0], format=format_type, cleanup=True)

    return dot


def to_closest_centroid(df):
    df["dishwasher"] = df["dishwasher"].apply(
        lambda x: (
            "idle"
            if x < (2.53 + 39.93) / 2
            else (
                "mini-job"
                if x < (39.93 + 1007.74) / 2
                else ("switch" if x < (1007.74 + 1723.68) / 2 else "job")
            )
        )
    )

    df["freezer"] = df["freezer"].apply(
        lambda x: (
            "idle"
            if x < (0.99 + 87.87) / 2
            else ("switch" if x < (87.87 + 98.04) / 2 else "on")
        )
    )

    df["fridge"] = df["fridge"].apply(
        lambda x: (
            "off" if x < 51.96 / 2 else ("switch" if x < (51.96 + 120.46) / 2 else "on")
        )
    )

    return df

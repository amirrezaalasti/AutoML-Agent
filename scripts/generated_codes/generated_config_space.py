from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer, ForbiddenAndConjunction, ForbiddenEqualsClause
from ConfigSpace import EqualsCondition, InCondition


def get_configspace():
    cs = ConfigurationSpace()

    model_type = Categorical("model_type", ["LSTM", "GRU", "MLP"])
    cs.add_hyperparameter(model_type)

    # LSTM Hyperparameters
    lstm_units = Integer("lstm_units", (32, 256), default=64)
    lstm_layers = Integer("lstm_layers", (1, 3), default=1)
    lstm_dropout = Float("lstm_dropout", (0.0, 0.5), default=0.0)
    cs.add_hyperparameters([lstm_units, lstm_layers, lstm_dropout])

    # GRU Hyperparameters
    gru_units = Integer("gru_units", (32, 256), default=64)
    gru_layers = Integer("gru_layers", (1, 3), default=1)
    gru_dropout = Float("gru_dropout", (0.0, 0.5), default=0.0)
    cs.add_hyperparameters([gru_units, gru_layers, gru_dropout])

    # MLP Hyperparameters
    mlp_layers = Integer("mlp_layers", (1, 3), default=2)
    mlp_units = Integer("mlp_units", (32, 256), default=128)
    mlp_dropout = Float("mlp_dropout", (0.0, 0.5), default=0.0)
    cs.add_hyperparameters([mlp_layers, mlp_units, mlp_dropout])

    # Learning Rate
    learning_rate = Float("learning_rate", (1e-5, 1e-2), default=1e-3, log=True)
    cs.add_hyperparameter(learning_rate)

    # Conditions for LSTM
    cond_lstm_units = EqualsCondition(lstm_units, model_type, "LSTM")
    cond_lstm_layers = EqualsCondition(lstm_layers, model_type, "LSTM")
    cond_lstm_dropout = EqualsCondition(lstm_dropout, model_type, "LSTM")
    cs.add_conditions([cond_lstm_units, cond_lstm_layers, cond_lstm_dropout])

    # Conditions for GRU
    cond_gru_units = EqualsCondition(gru_units, model_type, "GRU")
    cond_gru_layers = EqualsCondition(gru_layers, model_type, "GRU")
    cond_gru_dropout = EqualsCondition(gru_dropout, model_type, "GRU")
    cs.add_conditions([cond_gru_units, cond_gru_layers, cond_gru_dropout])

    # Conditions for MLP
    cond_mlp_layers = EqualsCondition(mlp_layers, model_type, "MLP")
    cond_mlp_units = EqualsCondition(mlp_units, model_type, "MLP")
    cond_mlp_dropout = EqualsCondition(mlp_dropout, model_type, "MLP")
    cs.add_conditions([cond_mlp_layers, cond_mlp_units, cond_mlp_dropout])

    # Forbidden Clauses - Example: LSTM with no layers
    forbidden_lstm_no_layers = ForbiddenAndConjunction(ForbiddenEqualsClause(model_type, "LSTM"), ForbiddenEqualsClause(lstm_layers, 1))

    # cs.add_forbidden_clause(forbidden_lstm_no_layers) # Removing forbidden clause.

    return cs

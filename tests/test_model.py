import src.wikipedia.model as model

def test_model():

    models = model.NodeLevelGNN(c_in=300, c_hidden=16, c_out=20, num_layers=2, dp_rate=0.5)
    assert models is not None
    assert models.loss_module is not None
    assert models.model is not None
    assert models.forward is not None
    assert models.training_step is not None
    assert models.validation_step is not None
    assert models.configure_optimizers is not None

import src.wikipedia.model as model

def test_model():
    models = model.GCN(hidden_channels=16, num_features=300, num_classes=20, dropout=0.5)
    assert models is not None
    assert models.criterion is not None
    assert models.model is not None
    assert models.forward is not None
    assert models.training_step is not None
    assert models.validation_step is not None
    assert models.configure_optimizers is not None
    assert models.predict is not None





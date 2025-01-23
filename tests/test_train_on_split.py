# import torch
# from unittest.mock import MagicMock, patch
# from src.wikipedia.train import train_on_split
# import pytest

# @pytest.fixture
# def mock_trainer():
#     trainer = MagicMock()
#     trainer.callback_metrics = {"val_acc": 0.85}  # Ensure val_acc is a float
#     trainer.validate.return_value = [{"val_acc": 0.85}]
#     trainer.test.return_value = [{"test_acc": 0.83}]
#     return trainer

# @pytest.fixture
# def mock_checkpoint_callback():
#     callback = MagicMock()
#     callback.best_model_path = "test/path/model.ckpt"
#     return callback

# @pytest.fixture
# def mock_google_auth(monkeypatch):
#     monkeypatch.setattr("src.wikipedia.train.secretmanager.SecretManagerServiceClient", MagicMock())
#     monkeypatch.setattr("src.wikipedia.train.storage.Client", MagicMock())
#     monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/dev/null")

# @pytest.fixture
# def mock_gnn_model():
#     with patch('src.wikipedia.model.GNNModel') as mock_model:
#         # Create a mock parameter
#         mock_param = torch.nn.Parameter(torch.randn(1))
#         mock_model_instance = MagicMock()
#         # Set up parameters for the optimizer
#         mock_model_instance.parameters.return_value = [mock_param]
#         mock_model.return_value = mock_model_instance
#         yield mock_model

# def test_train_on_split(mock_trainer, mock_checkpoint_callback, mock_google_auth, mock_gnn_model):
#     # Create mock data with correct dimensions
#     data = MagicMock()
#     data.train_mask = torch.ones((100, 1))
#     data.val_mask = torch.ones((100, 1))
#     data.test_mask = torch.ones(100)
#     data.x = torch.randn(100, 300)
#     data.edge_index = torch.randint(0, 100, (2, 200))
#     data.y = torch.randint(0, 20, (100,))
#     data.num_node_features = 300
#     data.y.max = lambda: torch.tensor(19)
    
#     # Remove redundant patching of NodeLevelGNN
#     # with patch('src.wikipedia.model.NodeLevelGNN', return_value=mock_gnn_model.return_value):
#     #     mock_param = torch.nn.Parameter(torch.randn(1))
#     #     mock_model_instance = MagicMock()
#     #     mock_model_instance.parameters.return_value = [mock_param]
#     #     mock_gnn_model.return_value = mock_model_instance
    
#     # Mock prepare_data_loaders and storage client
#     with patch('src.wikipedia.train.prepare_data_loaders') as mock_prepare, \
#          patch('google.cloud.storage.Client') as mock_storage_client:
#         mock_prepare.return_value = (MagicMock(), MagicMock(), 300, 20)
#         mock_storage_client.return_value = MagicMock()
        
#         # Mock the trainer and checkpoint callback
#         with patch('pytorch_lightning.Trainer', return_value=mock_trainer):
#             with patch('pytorch_lightning.callbacks.ModelCheckpoint', return_value=mock_checkpoint_callback):
#                 val_acc, test_acc = train_on_split(
#                     data=data,
#                     split_idx=0,
#                     hidden_channels=16,
#                     hidden_layers=2,
#                     dropout=0.5,
#                     learning_rate=0.001666,
#                     weight_decay=1e-4,
#                     optimizer_name="Adam",
#                     num_epochs=1,
#                     batch_size=11701,
#                     wandb_logger=None,
#                     model_checkpoint_callback=True,
#                     bucket_name="test-bucket"
#                 )

#     assert isinstance(val_acc, float)
#     assert isinstance(test_acc, float)
#     assert val_acc == 0.85  # Based on mock_trainer return value
#     assert test_acc == 0.83  # Based on mock_trainer return value
# Use the training image as the base
FROM train:latest 

WORKDIR /
COPY src/project1/evaluate.py src/project1/evaluate.py
COPY src/project1/model.py src/project1/model.py
COPY data/processed/test_images.pt data/processed/test_images.pt
COPY data/processed/test_target.pt data/processed/test_target.pt

ENTRYPOINT ["uv", "run", "src/project1/evaluate.py"]
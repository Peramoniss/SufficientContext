Full pipeline system for training and testing a sufficiency identifier.

You can run it directly in your machine or do it using a docker container.

To simply run in your machine, use the following code:
```
cd src
python main.py
```

To run using docker, guarantee docker is open in your computer and run:
```
docker build --no-cache -t sufficiency-training .
docker run --gpus all -it sufficiency-training
```


Notes:
1. File batchedModelStructes.py is only there to show that batches actually decrease the efficiency of the embedder because of the complexity of the sequential processing needed to structure the graphs
# Example of an algorithm that uses the GPU
This algorithm is an example of using the GPU in HKube.
It trains a network using keras on the mnist dataset

## algorithm definition
```js
{
    "name": "gpu-alg",
    "algorithmImage": "docker-image-name",
    "cpu": 0.1,
    "mem": 1024,
    "gpu": 1,
    "options": {
        "debug": false
    },
    "workerEnv": {
        "ALGORITHM_DISCONNECTED_TIMEOUT_MS": 3600000,
        "WORKER_ALGORITHM_PROTOCOL": "ws"
    },
    "algorithmEnv": {
        "AWS_ACCESS_KEY_ID": {
            "secretKeyRef": {
                "name": "s3-secret",
                "key": "awsKey"
            }
        },
        "AWS_SECRET_ACCESS_KEY": {
            "secretKeyRef": {
                "name": "s3-secret",
                "key": "awsSecret"
            }
        },
        "S3_ENDPOINT": "10.32.10.24:9000",
        "S3_USE_HTTPS": "0"
    },
    "minHotWorkers": 0
}
```

## s3
s3 parameters are optional and are only used if the output folder is set to s3://  
tensorflow s3 behavior is controlled by various environment variables:
```bash
AWS_ACCESS_KEY_ID=XXXXX                 # Credentials only needed if connecting to a private endpoint
AWS_SECRET_ACCESS_KEY=XXXXX
AWS_REGION=us-east-1                    # Region for the S3 bucket, this is not always needed. Default is us-east-1.
S3_ENDPOINT=s3.us-east-1.amazonaws.com  # The S3 API Endpoint to connect to. This is specified in a HOST:PORT format.
S3_USE_HTTPS=1                          # Whether or not to use HTTPS. Disable with 0.
S3_VERIFY_SSL=1                         # If HTTPS is used, controls if SSL should be enabled. Disable with 0.
```

## parameters
The algorithms input (in `init` and `start`) is an object (dict) with the following structure
```js
{
    input: [
        {
            train_size: 1000,
            output: '/training_1',
            num_epochs: 10

        }
    ],
    taskId: 'unique identifier of the run'
}
```
name | description 
--- | --- | --- 
train_size | The number of images to use for training. Must be smaller that the maximum number of images in the dataset (60000)
num_epochs | The number of trainig epochs
output | Output folder or s3 bucket to write the output data. for s3, the format is `s3://bucket/folder`.

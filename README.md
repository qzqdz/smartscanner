# SmartScanner: Lightweight Vulnerability Scanner for Extreme-Length Smart Contract Auditing

This project implements a novel approach for detecting vulnerabilities in smart contracts using an image-inspired retrieval method. The main program is designed to work with unlimited-length smart contract source code.

## Abstract

Our approach uses a lightweight contract embedding tool named SmartScanner to identify and amplify sparse vulnerability features. The system retrieves similar bug contracts for heuristic reference, outperforming classifier-based methods on a large-scale smart contract benchmark.

## Key Features

- Supports vulnerability detection in unlimited-length smart contract source code
- Uses image-inspired scanning methods to identify vulnerability features
- Implements multi-stage vulnerability-aware contrastive learning
- Retrieves similar contracts based on embedding similarity
- Utilizes k-Nearest Neighbors (kNN) for vulnerability classification

## Usage

To use the program:

1. Ensure all required dependencies are installed (see `requirements.txt`)
2. Prepare your smart contract dataset
3. Run the main script with appropriate arguments:

```
python main.py --do_train --epochs 5 --batch_size 8 --lr 1e-4 --pooling last-avg --model_path "/path/to/model" --use_teacher --use_teacher_embedding --teacher_model "/path/to/teacher/model" --snli_train "/path/to/train/data" --sts_dev "/path/to/dev/data" --sts_test "/path/to/test/data" --acc_train "/path/to/test/data" --acc_val  "/path/to/test/data" --acc_batch_size 4 --acc_maxlen 64 --acc_k 5 --seed 3402
```

## Main Components

- `SimcseModel`: The core model for embedding smart contracts
- `TrainDataset` and `TestDataset`: Custom dataset classes for training and testing
- `train()`: Function for training the model
- `eval()`: Function for evaluating the model
- `acc_eval()` and `knn_eval()`: Functions for accuracy evaluation using retrieval methods

## Requirements

- PyTorch
- Transformers
- Datasets
- Pandas
- Scikit-learn
- Loguru





## Clarification

For more details on the methodology and results, please refer to the full paper.

We provide the data sources with the key code for demonstrating the contribution of our study. After our study is formally published, we will release all the code of our work.

## Citation

If you use this code in your research, please cite our paper:
xxx

## License

This project is licensed under the GNU General Public License v3. See the [LICENSE](./LICENSE) file for details.

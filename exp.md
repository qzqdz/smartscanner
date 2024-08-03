



```python



python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_more_consis/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big-250000-24000/target_texts.json" --sts_dev "./datasets/SC-big-250000-24000/val_dataset.json" --sts_test "./datasets/SC-big/test_dataset_soft_100_512_1000.json" --acc_train "./datasets/SC-big/acc_train.json" --acc_val "./datasets/SC-big/acc_test.json"  --acc_k 2  --epochs 30 --batch_size 32 --pooling last-avg --do_train

test for big
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big/acc_train.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.9295154185022027
KNN-based Retrieval Accuracy: 0.9116189427312775

@5
Label-based Retrieval Accuracy: 0.9649203572415063

@10
Label-based Retrieval Accuracy: 0.9762452812816499

@20
Label-based Retrieval Accuracy: 0.9831507227695424


corpus nums 50
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_50.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.8722316865417377
KNN-based Retrieval Accuracy: 0.9042 | Precision: 0.9299 | Recall: 0.9125 | F1-score: 0.9211


100
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_100.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.8530664395229983
KNN-based Retrieval Accuracy: 0.8894 | Precision: 0.9079 | Recall: 0.9122 | F1-score: 0.9100

400
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_400.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.9020442930153322
KNN-based Retrieval Accuracy: 0.9242 | Precision: 0.9347 | Recall: 0.9422 | F1-score: 0.9384



1000
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_2000.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.9125851788756388
KNN-based Retrieval Accuracy: 0.9279 | Precision: 0.9432 | Recall: 0.9391 | F1-score: 0.9411


1 1 1000
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_1000_1_1.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.919931856899489
KNN-based Retrieval Accuracy: 0.9278 | Precision: 0.9536 | Recall: 0.9274 | F1-score: 0.9403

1 2 1000
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_1000_1_2.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.9146081771720613
KNN-based Retrieval Accuracy: 0.9269 | Precision: 0.9449 | Recall: 0.9352 | F1-score: 0.9401

1 3 1000
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_1000_1_3.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.9102427597955707
KNN-based Retrieval Accuracy: 0.9251 | Precision: 0.9372 | Recall: 0.9410 | F1-score: 0.9391


1 4 1000
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_1000_1_4.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.8988500851788757
KNN-based Retrieval Accuracy: 0.9253 | Precision: 0.9357 | Recall: 0.9429 | F1-score: 0.9393

1 5 1000
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_1000_1_5.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.8931005110732538
KNN-based Retrieval Accuracy: 0.9222 | Precision: 0.9287 | Recall: 0.9457 | F1-score: 0.9371


1 6 1000
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_1000_1_6.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.8844761499148212
KNN-based Retrieval Accuracy: 0.9201 | Precision: 0.9231 | Recall: 0.9488 | F1-score: 0.9358


1 7 1000
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_1000_1_7.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.8785136286201022
KNN-based Retrieval Accuracy: 0.9131 | Precision: 0.9104 | Recall: 0.9521 | F1-score: 0.9308


4 1
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_1000_4_1.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.919186541737649
KNN-based Retrieval Accuracy: 0.9211 | Precision: 0.9609 | Recall: 0.9083 | F1-score: 0.9339

2 1
python main.py --seed 3402  --maxlen 24000  --lr 1e-3  --save_path ./saved_model/SC_model_big_long_resnet_1d_24000/pytorch_model.bin    --model_path "E:/model/white_model/solidity_codebert"   --teacher_model "E:/model/white_model/solidity_codebert" --snli_train "./datasets/SC-big/target_texts.json" --sts_dev "./datasets/SC-big/test_dataset.json" --sts_test "./datasets/SC-big/test_dataset.json" --acc_train "./datasets/SC-big-250000-24000/acc_train_1000_2_1_all_length.json" --acc_val "./datasets/SC-big-250000-24000/acc_test_100_24000.json"  --acc_k 2  --epochs 30 --batch_size 16 --pooling last-avg
Label-based Retrieval Accuracy: 0.9203577512776832
KNN-based Retrieval Accuracy: 0.9263 | Precision: 0.9572 | Recall: 0.9210 | F1-score: 0.9388

Label-based Retrieval Accuracy: 0.9179088586030665
KNN-based Retrieval Accuracy: 0.9200 | Precision: 0.9588 | Recall: 0.9087 | F1-score: 0.9331



'''



```




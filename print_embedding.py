# 对单个智能合约进行嵌入展示的示例
import pandas as pd
import torch
from transformers import AutoTokenizer
from model import SimcseModel
from src.data_s import prepare_t5_data

'''
.\print_embedding.py
Input IDs length: 150000
tensor([[[-3.5418,  0.4678,  1.3992,  ..., -0.7620, -0.7620, -0.7620],
         [ 0.7536,  3.5529, -1.0902,  ...,  0.2144,  0.2144,  0.2144],
         [ 4.1009,  3.2558, -0.1704,  ...,  1.1484,  1.1484,  1.1484]]],
       device='cuda:0')
torch.Size([1, 3, 150000])
'''


def prepare_data(src, tokenizer, max_len=24000):
    # 对源代码进行编码
    a,b = prepare_t5_data(src)
    inputs = tokenizer(b[0], padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
    return inputs, b[0]

# 示例智能合约代码
code = '''pragma solidity ^0.8.0;

contract Wallet {
    uint256[] private bonusCodes;
    address private owner;

    constructor() {
        bonusCodes = new uint256[](0);
        owner = msg.sender;
    }

    receive() external payable {
        // No longer needed as functions with payable modifier can receive Ether
    }

    function pushBonusCode(uint256 c) public {
        bonusCodes.push(c);
    }

    function popBonusCode() public {
        require(bonusCodes.length > 0, "BonusCodes is empty");
        uint256 lastItem = bonusCodes[bonusCodes.length - 1];
        bonusCodes.pop();
        lastItem = 0;
    }

    function updateBonusCodeAt(uint256 idx, uint256 c) public {
        require(idx < bonusCodes.length, "Index out of bounds");
        bonusCodes[idx] = c;
    }

    function destroy() public {
        require(msg.sender == owner, "Only owner can destroy the contract");
        address payable burnAddress = payable(address(0));
        selfdestruct(burnAddress);
    }
}'''

# 加载模型和tokenizer
tokenizer_path = r'./SC_model_big_long_resnet_1d_24000'
model_path = r"./SC_model_big_long_resnet_1d_24000/pytorch_model.bin"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = SimcseModel(pretrained_model=tokenizer_path, pooling='last-avg')
model.load_state_dict(torch.load(model_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.only_embeddings = True

# 准备输入数据
inputs, _ = prepare_data(code, tokenizer, 150000)
print(f"Input IDs length: {inputs['input_ids'].shape[1]}")

# 获取嵌入向量
with torch.no_grad():
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = torch.zeros_like(input_ids)
    outputs, embeddings = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

print(embeddings)
print(embeddings.shape)

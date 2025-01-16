import json
import os
import re
from datasets import load_dataset
import numpy as np
from tqdm.notebook import trange
import pandas as pd
import random
import torch

from transformers import AutoTokenizer, T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
import gc
from tqdm import tqdm

# @title Clean text library: `clean(src)`
# # Notice we also removed all comments, might need 2nd thought

SEPRATORS = ('\nabstract contract', '\ncontract', '\nlibrary', '\ninterface', '\nstruct')


def _remove_comment(src_in):
    # multi line
    src_in = re.sub("\/\*(\*(?!\/)|[^*])*\*\/", "", src_in)
    # single line, maybe keep?
    src_in = re.sub("\/\/*.*", "", src_in)
    return src_in


def _remove_header_txt(src_in):
    if '\npragma solidity' not in src_in:
        return src_in
    p = src_in.index('\npragma solidity')
    if p > 0:
        return src_in[p + 1:]  # new line no need
    return src_in


def _remove_extra_new_line(src_in):
    src_in = src_in.strip()
    # remove empty content lines
    src_in = re.sub("(\s)+(\n)", "\n", src_in)
    src_in = re.sub("(\n)+", "\n", src_in)
    return src_in


def _replace_addr(src_in):
    return re.sub("0x[A-Fa-f0-9]{40}", "YOUR_ADDR", src_in)

# def _replace_addr(src_in):
#     # 匹配以太坊地址，并确保地址之后不是十六进制字符
#     return re.sub(r"0x[A-Fa-f0-9]{40}(?![A-Fa-f0-9])", "YOUR_ADDR", src_in)

def _format_src(src_in):
    # remove extra space before new line
    src_in = re.sub("\s+\n", "\n", src_in)
    # format the method or class desclaration so each { has exactly one space before
    src_in = re.sub(r"(.){", r"\1 {", src_in)
    src_in = re.sub("\s+{", r" {", src_in)
    src_in = src_in.replace("( ", "(")
    src_in = src_in.replace(" )", ")")
    src_in = src_in.replace("[ ", "[")
    src_in = src_in.replace(" ]", "]")
    # Remove unnecessary spaces in method declare
    src_in = re.sub("\n\s+external\s ", r" external ", src_in)
    src_in = re.sub("\n\s+internal\s", r" internal ", src_in)
    src_in = re.sub("\n\s+public\s", r" public ", src_in)
    src_in = re.sub("\s+poolOnly\s", r" poolOnly ", src_in)
    src_in = re.sub("\s+returns\(", r" returns(", src_in)
    # '\nabstract contract', '\ncontract', '\nlibrary', '\ninterface'
    src_in = re.sub("}\s+abstract contract ", r"}\nabstract contract ", src_in)
    src_in = re.sub("}\s+contract ", r"}\ncontract ", src_in)
    src_in = re.sub("}\s+library ", r"}\nlibrary ", src_in)
    src_in = re.sub("}\s+interface ", r"}\ninterface ", src_in)
    src_in = re.sub("}\s+struct ", r"}\nstruct ", src_in)
    src_in = re.sub(";\s+abstract contract ", r";\nabstract contract ", src_in)
    src_in = re.sub(";\s+contract ", r";\ncontract ", src_in)
    src_in = re.sub(";\s+library ", r";\nlibrary ", src_in)
    src_in = re.sub(";\s+interface ", r";\ninterface ", src_in)
    src_in = re.sub(";\s+struct ", r";\nstruct ", src_in)
    # special, typo "ontract"
    src_in = re.sub("}\s+ntract ", r"}\ncontract ", src_in)
    src_in = src_in.replace("}contract ", "}\ncontract ")
    src_in = src_in.replace("}interface ", "}\ninterface ")
    src_in = src_in.replace("}struct ", "}\nstruct ")

    return src_in

def _format_src2(src_in):
    # Remove extra spaces before new line
    src_in = re.sub("\s+\n", "\n", src_in)

    # Format the method or class declaration so each "{" has exactly one space before it
    # But don't add a newline before "{" if it's directly after a "}" (to handle nested structures correctly)
    src_in = re.sub(r"(?<!})\s*{", " {", src_in)

    # Adjust spacing for other syntax elements
    src_in = src_in.replace("( ", "(").replace(" )", ")")
    src_in = src_in.replace("[ ", "[").replace(" ]", "]")

    # Remove unnecessary spaces in method declaration
    src_in = re.sub(r"\n\s+(external|internal|public|private)\s", r" \1 ", src_in)
    src_in = re.sub(r"\s+returns\(", " returns(", src_in)

    # Correctly format contract, library, and interface declarations
    src_in = re.sub(r"}\s+(abstract contract|contract|library|interface)\s", r"}\n\1 ", src_in)

    # Add newline after semicolons if they're not inside a for loop declaration
    src_in = re.sub(r";(?![^\(]*\))", ";\n", src_in)

    # Special handling for contract endings and contract beginnings to ensure proper newlines
    contract_endings = re.compile(r"}\s*(?=\n*(abstract contract|contract|library|interface|$))")
    src_in = contract_endings.sub("}\n\n", src_in)

    # Handling function declarations and other elements to ensure consistency
    # This includes handling for visibility specifiers, function modifiers, and so on
    src_in = re.sub(r"\n\s+(function)", r"\n\1", src_in)
    src_in = re.sub(r"(?<=\))\s+{", " {", src_in)  # Space before "{" after function arguments

    # Ensure there's a newline before each contract, library, and interface declaration (except at the start of the file)
    src_in = re.sub(r"(?<=\n)(abstract contract|contract|library|interface)", r"\n\1", src_in)

    return src_in


def clean(src):
    src = _remove_comment(src)
    src = _remove_header_txt(src)
    src = _remove_extra_new_line(src)
    src = _replace_addr(src)
    src = _format_src(src)
    return src


# @title Split to segments (e.g. contracts) `process_single_line(src)`
def _extract_pub_funcs(seg):
    pub_funcs = re.findall("function [A-Za-z0-9_]+\(", seg)
    if pub_funcs:
        pub_funcs = [s[len('function '):-1] for s in pub_funcs
                     if not s[len('function '):-1].startswith('_') and not s[len('function '):-1].endswith('_')]
    return pub_funcs


def _extract_constants(seg):
    constants = re.findall(r"constant [A-Za-z0-9_]+", seg)
    if constants:
        constants = [s[len('constant '):] for s in constants]
    return constants


def _extract_base_parents1(seg):
    base_with_parents = re.findall("[A-Za-z0-9]+ is [A-Za-z0-9, \n]+ {", seg)
    base, parents = None, []
    if base_with_parents:
        assert 1 == len(base_with_parents), "base_with_parents pattern can only have 1 match"
        splits = base_with_parents[0].split(' is ')
        assert 2 == len(splits), "cannot have more than 2 splits for base extraction"
        base = splits[0]
        parents = [p.strip() for p in splits[1][:-2].split(',')]
    else:
        base_only = re.findall("[A-Za-z0-9]+\s+{", seg)
        if base_only:
            base = base_only[0].split()[0]
            parents = []
    return base, parents

def _extract_base_parents(seg):
    base_with_parents = re.findall("[A-Za-z0-9]+ is [A-Za-z0-9, \n]+ {", seg)
    base, parents = None, []
    if base_with_parents:
        # 如果找到多个匹配，仅处理第一个匹配
        splits = base_with_parents[0].split(' is ')
        if len(splits) == 2:
            base = splits[0]
            parents = [p.strip() for p in splits[1][:-2].split(',')]
    else:
        base_only = re.findall("[A-Za-z0-9]+\s+{", seg)
        if base_only:
            base = base_only[0].split()[0]
    return base, parents


DEFAULT_SOL_VERSION = "pragma solidity ^0.6.0;";


# def _prepare_seg_map(segs):
#     if not segs[0].startswith('pragma solidity'):
#         segs.insert(0, DEFAULT_SOL_VERSION)
#     seg_map = {}
#     for s in segs:
#         base, parents = _extract_base_parents(s)
#         if base:
#             seg_map[base] = {
#                 'parents': parents,
#                 'constants': _extract_constants(s),
#                 'pub_funcs': _extract_pub_funcs(s),
#                 'v': segs[0],  # version first line
#                 'clean_src': s,
#             }
#     return seg_map

def _extract_struct_fields(seg):
    struct_fields = re.findall(r"struct \w+\s*{[^}]*}", seg)
    if struct_fields:
        fields = []
        for struct_field in struct_fields:
            field_names = re.findall(r"\w+\s+\w+;", struct_field)
            fields.extend([f.strip().split()[-1] for f in field_names])
    else:
        fields = []
    return fields

def _prepare_seg_map(segs):
    if not segs[0].startswith('pragma solidity'):
        segs.insert(0, DEFAULT_SOL_VERSION)
    seg_map = {}
    for s in segs:
        base, parents = _extract_base_parents(s)
        if base:
            seg_map[base] = {
                'parents': parents,
                'constants': _extract_constants(s),
                'pub_funcs': _extract_pub_funcs(s),
                'struct_fields': _extract_struct_fields(s),  # 添加 struct 字段提取
                'v': segs[0],  # version first line
                'clean_src': s,
            }
    return seg_map


# @title Generate T5-friendly data func `prepare_t5_data(src)`
# def _get_single_ancestor_metadata(an, seg_map):
#     if an not in seg_map:
#         return ""
#     pub_func_str = " ".join(seg_map[an]['pub_funcs'])
#     const_str = " ".join(seg_map[an]['constants'])
#     return f"// Context: {an} | Functions: {pub_func_str} | Constants: {const_str}"

def _get_single_ancestor_metadata(an, seg_map):
    if an not in seg_map:
        return ""
    pub_func_str = " ".join(seg_map[an]['pub_funcs'])
    const_str = " ".join(seg_map[an]['constants'])
    struct_fields_str = " ".join(seg_map[an]['struct_fields'])  # 添加 struct 字段
    return f"// Context: {an} | Functions: {pub_func_str} | Constants: {const_str} | Structs: {struct_fields_str}"

# @title Split the text now
def _split_segments(src):
    start = 0
    segments = []
    while True:
        # Find the next closest seprator position
        next_sep = len(src) + 1
        seg_keyword = ""
        seg_type = ''
        for sep in SEPRATORS:
            # print("next_sep", next_sep)
            # print("start", start)
            cur_src = src[start:]
            if sep in cur_src:
                sep_ind = cur_src.index(sep)
                if sep_ind > 0 and next_sep > sep_ind:
                    next_sep = sep_ind
                    seg_keyword = cur_src[sep_ind + len(sep) + 1:].split()[0]
                    seg_type = sep[1:]
        if next_sep > len(src):
            if start < len(src) - 1:
                segments.append(src[start:].strip())
            break
        else:
            segments.append(src[start:start + next_sep].strip())
            start += next_sep + 1
    return segments


def _find_ancestors(seg_map):
    for k in seg_map:
        parents = seg_map[k]['parents']
        if parents:
            ancestors = parents.copy()
            idx = 0
            while (idx < len(ancestors)):
                if ancestors[idx] in seg_map:
                    # Be careful of cycle dependency
                    for more_parent in seg_map[ancestors[idx]]['parents']:
                        if more_parent not in ancestors and ancestors != k:
                            ancestors.append(more_parent)
                idx += 1
            seg_map[k]['ancestors'] = ancestors
        else:
            seg_map[k]['ancestors'] = []
    return seg_map


def process_single_line(src):
    """Clean text, split to segments, prepare segment map with ancestors."""
    src = clean(src)
    segs = _split_segments(src)
    seg_map = _prepare_seg_map(segs)
    seg_map = _find_ancestors(seg_map)
    return seg_map




def _reduce_out_whitespace(out_src):
    # remove extra spaces (ignore identation) and replace "; " with ";\n"
    out_src = re.sub("\s+", " ", out_src)
    out_src = out_src.replace("; ", ";\n")
    out_src = out_src.replace("{ ", "{\n")
    out_src = out_src.replace("} ", "}\n")
    return out_src.strip()


my_src = ""
my_seg = None
my_raw = ''



def prepare_t5_data(src):
    my_src = src
    seg_map = process_single_line(src)
    ins, outs = [], []
    for k, v in seg_map.items():
        s = v['v'] + "\n"
        for a in v['ancestors']:
            s += _get_single_ancestor_metadata(a, seg_map) + "\n"
        raw_src_code = v['clean_src']
        my_raw = raw_src_code
        s += raw_src_code
        o = _reduce_out_whitespace(raw_src_code)
        ins.append(s)
        outs.append(o)
    # print(src)
    return [src], [''.join(outs)]
    # return ins, outs


# def prepare_t5_data(src):
#     '''分段的'''
#     my_src = src
#     seg_map = process_single_line(src)
#     my_seg = seg_map
#     ins, outs = [], []
#     for k, v in seg_map.items():
#         # Some headers does not have content
#         if '{\n' not in v['clean_src']:
#             continue
#         s = v['v'] + "\n"
#         for a in v['ancestors']:
#             s += _get_single_ancestor_metadata(a, seg_map) + "\n"
#         raw_src_code = v['clean_src']
#         my_raw = raw_src_code
#         header_split_indx = raw_src_code.index('{\n')
#         s += raw_src_code[:header_split_indx + 1]  # include "{"
#         o = _reduce_out_whitespace(raw_src_code[header_split_indx + 2:])
#         ins.append(s)
#         outs.append(o)
#     return ins, outs


def extract_labels(slither_res, label_set='all'):
    """
    从 Slither 分析结果中提取标签
    """
    labels = []
    confidence_levels = []
    impact_levels = []
    slither_results = json.loads(slither_res)
    if slither_results["success"]:
        if slither_results["results"]:
            for result in slither_results["results"]["detectors"]:
                # print(result)
                label = result["check"]
                confidence = result["confidence"]
                impact = result["impact"]
                labels.append(label)
                confidence_levels.append(confidence)
                impact_levels.append(impact)
        else:
            # 如果 results 是空字典,则添加默认标签 'safe'
            labels.append('safe')
            confidence_levels.append('None')
            impact_levels.append('None')
    return labels, confidence_levels, impact_levels

def convert_to_df1(ds, DATA_TYPE=None):
    all_ins, all_outs, all_labels, all_confidences, all_impacts = [], [], [], [], []

    for i in tqdm(range(len(ds)), total=len(ds), desc='Converting dataset'):
        src = ds[i]['source_code']
        slither_res = ds[i]['slither']

        try:
            ins, outs = prepare_t5_data(src)
            # 提取标签、置信度和影响程度
            labels, confidences, impacts = extract_labels(slither_res, label_set=DATA_TYPE.split('-')[0])
            all_ins.extend(ins)
            all_outs.extend(outs)
            all_labels.extend([labels] * len(ins))
            all_confidences.extend([confidences] * len(ins))
            all_impacts.extend([impacts] * len(ins))
        except Exception as e:
            continue

    df = pd.DataFrame({
        'source_text': all_ins,
        'target_text': all_outs,
        'labels': all_labels,
        'confidences': all_confidences,
        'impacts': all_impacts
    })

    return df


def load_label_mappings(mappings_path):
    with open(mappings_path, 'r') as file:
        mappings = json.load(file)
    return mappings


def apply_label_mappings(labels, mappings):
    # 应用映射，忽略标记为'ignore'的项
    mapped_labels = [mappings.get(label, label) for label in labels if mappings.get(label, label) != 'ignore']
    if not mapped_labels:
        # 如果所有标签都被映射为'ignore'，或者没有任何标签，将其分类为'safe'
        mapped_labels = ['safe']
    mapped_labels = list(set(mapped_labels))
    # print(mapped_labels)
    return mapped_labels


def convert_to_df(ds, mappings, DATA_TYPE=None):
    all_ins, all_outs, all_labels, all_confidences, all_impacts, all_original_labels = [], [], [], [], [], []

    for i in tqdm(range(len(ds)), total=len(ds), desc='Converting dataset'):
        src = ds[i]['source_code']
        slither_res = json.loads(ds[i]['slither'])

        # try:
        if True:
            ins, outs = prepare_t5_data(src)
            # 映射前的原始标签
            original_labels = [det['check'] for det in slither_res.get("results", {}).get("detectors", [])]
            # 应用标签映射
            labels = apply_label_mappings(original_labels, mappings)
            confidences, impacts = ['high'], ['medium']  # 示例值，需要根据实际提取方法替换
            all_ins.extend(ins)
            all_outs.extend(outs)
            all_labels.extend([labels] * len(ins))
            all_confidences.extend([confidences] * len(ins))
            all_impacts.extend([impacts] * len(ins))
            all_original_labels.extend([original_labels] * len(ins))
        # except Exception as e:
        #     continue

    df = pd.DataFrame({
        'source_text': all_ins,
        'target_text': all_outs,
        'labels': all_labels,
        'confidences': all_confidences,
        'impacts': all_impacts,
        'original_labels': all_original_labels
    })

    return df




def load_and_preprocess_data(hf_data_source, data_type="small-plain-text", split_type='train', min_len=100, max_len=50000, mappings_path= None):
    mappings = load_label_mappings(mappings_path)

    cache_file = f"cache/{data_type}_{split_type}_{min_len}_{max_len}_converted.pkl"
    if os.path.exists(cache_file):
        print(f"Loading cached DataFrame from {cache_file}")
        filtered_df = pd.read_pickle(cache_file)
    else:
        all_ds = load_dataset(hf_data_source, data_type, split=split_type, ignore_verifications=True)
        print("Original DS size", len(all_ds))

        all_df = convert_to_df(all_ds, mappings, data_type)

        # Print the length of the longest sentence in all_df
        max_sentence_length_all_df = all_df['target_text'].str.len().max()
        print(f"The length of the longest sentence in all_df is: {max_sentence_length_all_df}")


        filtered_df = all_df[(all_df['target_text'].str.len() >= min_len) & (all_df['target_text'].str.len() <= max_len)]
        filtered_df = filtered_df.reset_index(drop=True)

        print(f"Filtered DS size after preprocessing: {len(filtered_df)}")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        filtered_df.to_pickle(cache_file)

    return filtered_df
def load_and_preprocess_data1(hf_data_source, data_type="small-plain-text", split_type='train', min_len=100,
                             max_len=50000):
    cache_file = f"{data_type}_{split_type}_converted.pkl"

    if os.path.exists(cache_file):
        print(f"Loading cached DataFrame from {cache_file}")
        filtered_df = pd.read_pickle(cache_file)
    else:
        all_ds = load_dataset(hf_data_source, data_type, split=split_type, ignore_verifications=True)
        print("Original DS size", len(all_ds))

        # 转换为DataFrame并执行预处理
        all_df = convert_to_df(all_ds, data_type)

        # 过滤处理后的DataFrame中源代码长度不符合要求的样本
        filtered_df = all_df[
            (all_df['target_text'].str.len() >= min_len) & (all_df['target_text'].str.len() <= max_len)]
        filtered_df = filtered_df.reset_index(drop=True)


        print("Filtered DS size after preprocessing", len(filtered_df))

        # 保存转换后的DataFrame为缓存文件
        print(f"Saving converted DataFrame to {cache_file}")
        filtered_df.to_pickle(cache_file)

    return filtered_df



_LABELS = {
    'all': [
        'uninitialized-state','constant-function-asm', 'locked-ether',
        'incorrect-shift', 'divide-before-multiply', 'unused-return',
        'write-after-write', 'reentrancy-no-eth', 'unchecked-lowlevel',
        'incorrect-equality', 'weak-prng', 'arbitrary-send',
        'uninitialized-local', 'reentrancy-eth', 'shadowing-abstract',
        'controlled-delegatecall', 'unchecked-transfer', 'erc20-interface',
        'controlled-array-length', 'tautology', 'shadowing-state',
        'tx-origin', 'unprotected-upgrade', 'suicidal',
        'boolean-cst', 'unchecked-send', 'msg-value-loop',
        'erc721-interface', 'constant-function-state', 'delegatecall-loop',
        'mapping-deletion', 'reused-constructor', 'uninitialized-storage',
        'public-mappings-nested', 'array-by-reference','backdoor',
        'rtlo', 'name-reused','safe'],
    'big': ['access-control', 'arithmetic', 'other', 'reentrancy', 'safe', 'unchecked-calls'],
    'small': ['access-control', 'arithmetic', 'other', 'reentrancy', 'safe', 'unchecked-calls', 'locked-ether', 'bad-randomness', 'double-spending']
}




if __name__ == "__main__":
    # @title Load all raw data (train, validation, test), ~3 mins
    # Available: ['all-plain-text', 'all-multilabel', 'big-plain-text', 'big-multilabel', 'small-plain-text', 'small-multilabel']
    # Checksum error as of Dec 2022, have to set ignore_verifications to True



    HF_DATA_SOURCE = r"E:/data/slither-audited-smart-contracts/slither-audited-smart-contracts.py"
    DATA_TYPE = "small-plain-text"  # change to 'small-plain-text for debugging
    all_ds = load_dataset(HF_DATA_SOURCE, DATA_TYPE, split="train",
                          revision="main", ignore_verifications=True)

    # train_ds = train_ds.filter(lambda elem: elem['bytecode'] != '0x')
    # val_ds = val_ds.filter(lambda elem: elem['bytecode'] != '0x')
    all_ds = all_ds.filter(lambda elem: elem['bytecode'] != '0x')

    # Small data types has validation/test as well
    print("DS size", len(all_ds))

    all_source_ds = all_ds['source_code']
    print("all_source_ds size", len(all_source_ds))

    # Why set 50k limit? Too large, and it covers 80% already
    # lens = [len(all_source_ds[i]) for i in range(len(all_source_ds))]
    # lens = [l for l in lens if l < 50000]
    # print(len(lens))
    # plt.hist(lens)
    # plt.show()

    filtered_all_source_ds = [s for s in all_source_ds if len(s) < 50000 and len(s.strip()) > 100 and '{\n' in s]
    print("filtered_all_source_ds size", len(filtered_all_source_ds))

    TEST_RATE = 0.05
    bad_sample = []


    all_df = convert_to_df(all_ds)
    all_df = all_df.sample(frac=1)  # Shuffle


    train_df, eval_df = train_test_split(all_df, test_size=TEST_RATE)

    print("Notice bad samples: ", len(bad_sample))

    PATH = './datasets'
    # train_df.to_parquet(f"{PATH}/processed_data_train")
    # eval_df.to_parquet(f"{PATH}/processed_data_eval")
    train_df.to_json(f"{PATH}/processed_data_train.json")
    eval_df.to_json(f"{PATH}/processed_data_eval.json")
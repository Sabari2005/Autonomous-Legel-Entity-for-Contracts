import torch
from .contract_summarizer import ContinuousContractSummarizer

def main():
    torch.set_default_tensor_type('torch.FloatTensor')
    summarizer = ContinuousContractSummarizer()
    contract_text = input("Please paste the contract text: ")
    summarizer.summarize(contract_text)

if __name__ == "__main__":
    main()